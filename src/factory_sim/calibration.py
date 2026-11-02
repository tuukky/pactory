"""기준 데이터 보정 실행 (PLAN_v2 §8, D8·D9) + 워밍업 MSER-5 (D24).

보정 전용 시드 1~9만 쓴다. 현상 파라미터(고장·준비·품질)만 override로 바꿔 보고, 설계 변수는 건드리지
않는다. 이 CLI는 '한 번의 보정 반복'을 실행하고 범위 판정표를 남긴다. 어떤 배율을 채택했는지는
docs/v2/calibration.md에 사람이 기록하고 rev_A.yaml에 동결한다.

Usage:
    python -m factory_sim.calibration --config configs/v2/rev_A.yaml --seeds 1-9 --label iter0
    python -m factory_sim.calibration --config configs/v2/rev_A.yaml --seeds 1-9 --label setup_x0.2 \
        --override stations.R1.setup_group_change_s=60 --override stations.S3.setup_group_change_s=48
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from line_sim.experiments import mean_ci, parse_seeds

from .config import ConfigError, horizon_s, resolve_config
from .experiments import default_workers, execute
from .metrics import CSV_ENCODING
from .run import environment_info
from .warmup import decide_warmup_h, hourly_good_from_dir, mser5

CALIBRATION_SEEDS = range(1, 10)
# 현실성 범위 (PLAN_v2 §8). 분모는 부하 기준(D8).
RANGES = {
    "down_ratio_load": (0.05, 0.12),
    "setup_ratio.R1": (0.05, 0.15),
    "setup_ratio.S3": (0.05, 0.15),
    "eol_first_pass_yield": (0.93, 0.97),
}
# 보정 대상이 될 수 있는 현상 파라미터 접두어 (설계 변수 보정 금지)
PHENOMENON_KEYS = ("failures.", "products.groups.", "stations.R1.setup_group_change_s",
                   "stations.S3.setup_group_change_s")


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ConfigError([f"override 형식: key=value ({text!r})"])
    k, v = text.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def check_phenomenon_only(overrides: Mapping[str, Any]) -> list[str]:
    bad = [k for k in overrides
           if not k.startswith(PHENOMENON_KEYS) or k.endswith(".share")]
    return [f"보정은 현상 파라미터만 허용 (설계 변수·품종 비율 금지): {bad}"] if bad else []


def summarize_calibration(rows: list[dict]) -> pd.DataFrame:
    out = []
    for key, (lo, hi) in RANGES.items():
        vals = [r[key] for r in rows if r.get(key) is not None]
        ci = mean_ci(vals)
        mean = ci["mean"]
        out.append({"metric": key, "range_low": lo, "range_high": hi, "n": ci["n"], "mean": mean,
                    "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                    "in_range": None if mean is None else lo <= mean <= hi})
    return pd.DataFrame(out)


def run_calibration(config: str | Path, seeds: list[int], *, overrides: Mapping[str, Any] | None = None,
                    label: str = "iter", workers: int | None = None, results_root: str | Path = "results/v2",
                    verbose: bool = True) -> tuple[Path, pd.DataFrame]:
    bad = [s for s in seeds if s not in CALIBRATION_SEEDS]
    if bad:
        raise ConfigError([f"보정은 시드 1~9만 사용 (§8): {bad}"])
    overrides = dict(overrides or {})
    errs = check_phenomenon_only(overrides)
    if errs:
        raise ConfigError(errs)
    cfg = resolve_config(config, overrides)
    h = hashlib.sha256(json.dumps({"cfg": cfg, "label": label}, sort_keys=True, default=str).encode()).hexdigest()
    exp_id = f"calibration_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{h[:8]}"
    exp_dir = Path(results_root) / exp_id
    exp_dir.mkdir(parents=True)
    env_info = environment_info()
    command = f"python -m factory_sim.calibration --config {config} --label {label} " + \
        " ".join(f"--override {k}={v}" for k, v in overrides.items())
    tasks = [{"scenario": label, "rev": str(cfg.get("revision")), "condition": "base", "seed": s,
              "resource_class": None, "cfg": cfg, "out_dir": str(exp_dir / f"seed_{s}"), "rel_dir": f"seed_{s}",
              "detail": "summary", "experiment_id": exp_id, "command": command, "env_info": env_info}
             for s in seeds]
    (exp_dir / "plan.json").write_text(json.dumps({"experiment_id": exp_id, "label": label, "config": str(config),
                                                   "overrides": overrides, "seeds": seeds}, ensure_ascii=False,
                                                  indent=2, default=str), encoding="utf-8")
    index = pd.DataFrame(execute(tasks, workers or default_workers(), verbose))
    index.to_csv(exp_dir / "runs_index.csv", index=False, encoding=CSV_ENCODING)

    per_seed, mser_rows = [], []
    for r in index.itertuples(index=False):
        if r.status == "failed":
            continue
        m = json.loads((exp_dir / r.run_dir / "metrics.json").read_text(encoding="utf-8"))
        c = m["calibration"]
        row = {"seed": int(r.seed), "down_ratio_load": c["down_ratio_load"],
               "eol_first_pass_yield": c["eol_first_pass_yield"]}
        for sid, v in (c.get("setup_ratio") or {}).items():
            row[f"setup_ratio.{sid}"] = v
        for sid, v in (c.get("down_ratio_by_station") or {}).items():
            row[f"down_ratio.{sid}"] = v
        per_seed.append(row)
        ms = mser5(hourly_good_from_dir(exp_dir / r.run_dir, horizon_s(cfg)))
        mser_rows.append({"seed": int(r.seed), "d_batches": ms["d_batches"], "d_hours": ms["d_hours"],
                          "n_batches": ms["n_batches"]})
    pd.DataFrame(per_seed).to_csv(exp_dir / "calibration_per_seed.csv", index=False, encoding=CSV_ENCODING)
    summary = summarize_calibration(per_seed)
    summary.to_csv(exp_dir / "calibration_summary.csv", index=False, encoding=CSV_ENCODING)
    mser = pd.DataFrame(mser_rows)
    mser.to_csv(exp_dir / "mser.csv", index=False, encoding=CSV_ENCODING)
    decision = decide_warmup_h(mser["d_hours"].tolist() if len(mser) else [],
                               planned_h=float(cfg["run"]["warmup_s"]) / 3600.0)
    (exp_dir / "warmup_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if verbose:
        print(summary.to_string(index=False))
        print(f"MSER-5 d*(h) 시드별: {mser['d_hours'].tolist() if len(mser) else []} → {decision}")
        print(f"실패 {int((index['status'] == 'failed').sum())} | 저장: {exp_dir}")
    return exp_dir, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="v2 기준 데이터 보정 (시드 1~9)")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default="1-9")
    p.add_argument("--label", default="iter")
    p.add_argument("--override", action="append", default=[], help="key=value (현상 파라미터만)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--results-root", default="results/v2")
    a = p.parse_args(argv)
    try:
        ov = dict(_parse_override(t) for t in a.override)
        run_calibration(a.config, parse_seeds(a.seeds), overrides=ov, label=a.label, workers=a.workers,
                        results_root=a.results_root)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
