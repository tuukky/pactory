"""Replicated scenario experiments and statistics (§11).

Usage:
    python -m line_sim.experiments --matrix configs/experiments.yaml --phase discovery
    python -m line_sim.experiments --matrix configs/experiments.yaml --phase confirmation --candidate S1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from .config import ConfigError, config_hash, load_yaml, resolve_config
from .engine import run_simulation
from .metrics import CSV_ENCODING
from .run import collect_metadata, environment_info, write_failed_run, write_run_dir

CANDIDATE_TOKEN = "CANDIDATE"
MAX_COMBINATIONS = 2
NO_CLEAR_DIFF = "이번 실험으로 명확한 차이를 확인하지 못함"

# (metric name, path in metrics.json)
SUMMARY_METRICS: list[tuple[str, tuple[str, ...]]] = [
    ("shift.good_shipped", ("shift", "good_shipped")),
    ("shift.overall_otd_rate", ("shift", "overall_otd_rate")),
    ("shift.avg_wip", ("shift", "avg_wip")),
    ("shift.unshipped.total", ("shift", "unshipped", "total")),
    ("shift.unshipped.unreleased", ("shift", "unshipped", "unreleased")),
    ("shift.unshipped.wip", ("shift", "unshipped", "wip")),
    ("shift.unshipped.scrapped", ("shift", "unshipped", "scrapped")),
    ("full_run.completion_time_s", ("full_run", "completion_time_s")),
    ("full_run.line_time_mean_s", ("full_run", "line_time", "all", "mean_s")),
    ("full_run.order_age_mean_s", ("full_run", "order_age", "all", "mean_s")),
    ("full_run.scrapped_total", ("full_run", "scrapped_total")),
]


SUMMARY_COLUMNS = ["scenario", "condition", "metric", "n_ok", "n_failed", "n_null", "mean", "sd", "ci_low",
                   "ci_high"]
BY_PRODUCT_COLUMNS = ["scenario", "condition", "product_type", "metric", "n_ok", "n_failed", "n_null", "mean", "sd",
                      "ci_low", "ci_high"]
PAIRED_COLUMNS = ["scenario", "baseline", "condition", "metric", "n_pairs", "dropped_seeds", "base_mean", "cand_mean",
                  "mean_d", "ci_low", "ci_high", "improvement_pct", "ci_includes_zero", "note"]


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding=CSV_ENCODING)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


# ------------------------------------------------------------------------- statistics
def mean_ci(values: Iterable[float], level: float = 0.95) -> dict[str, Any]:
    """Mean and two-sided Student-t confidence interval. CI is None when n < 2."""
    v = np.asarray([float(x) for x in values], dtype=float)
    n = int(len(v))
    out: dict[str, Any] = {"n": n, "mean": None, "sd": None, "half_width": None, "ci_low": None, "ci_high": None}
    if n == 0:
        return out
    out["mean"] = float(v.mean())
    if n < 2:
        return out
    sd = float(v.std(ddof=1))
    hw = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n))
    out.update(sd=sd, half_width=hw, ci_low=out["mean"] - hw, ci_high=out["mean"] + hw)
    return out


def paired_diff(cand: Mapping[int, float | None], base: Mapping[int, float | None]) -> dict[str, Any]:
    """d_i = cand_i - base_i paired by seed. Seeds missing or null on either side are dropped."""
    seeds = sorted(set(cand) | set(base))
    paired = [s for s in seeds if cand.get(s) is not None and base.get(s) is not None]
    dropped = [s for s in seeds if s not in paired]
    d = [float(cand[s]) - float(base[s]) for s in paired]
    ci = mean_ci(d)
    return {
        "n_pairs": len(paired), "dropped_seeds": dropped, "mean_d": ci["mean"], "sd_d": ci["sd"],
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
        "cand_mean": float(np.mean([cand[s] for s in paired])) if paired else None,
        "base_mean": float(np.mean([base[s] for s in paired])) if paired else None,
    }


def improvement_pct(base_mean: float | None, cand_mean: float | None) -> float | None:
    if base_mean is None or cand_mean is None or base_mean <= 0:
        return None
    return (cand_mean - base_mean) / base_mean * 100.0


# ------------------------------------------------------------------------- matrix handling
REQUIRED_MATRIX_KEYS = ("base_config", "scenarios", "conditions", "phases")


def load_matrix(path: str | Path) -> dict:
    path = Path(path)
    m = load_yaml(path)
    missing = [k for k in REQUIRED_MATRIX_KEYS if not m.get(k)]
    if missing:
        raise ConfigError([f"{path}: 필수 키 누락 {missing}"])
    m["_path"] = str(path)
    base_path = (path.parent / str(m["base_config"])).resolve()
    if not base_path.is_file():
        raise ConfigError([f"base_config 파일 없음: {base_path}"])
    m["_base_config_path"] = str(base_path)
    baseline = m.get("baseline_scenario", "B0")
    if baseline not in m["scenarios"]:
        raise ConfigError([f"baseline_scenario {baseline}가 scenarios에 없음"])
    combos = m.get("combinations") or {}
    if len(combos) > MAX_COMBINATIONS:
        raise ConfigError([f"조합안은 최대 {MAX_COMBINATIONS}개 ({len(combos)}개 정의됨)"])
    return m


def _merge(parts: list[tuple[str, Mapping]]) -> dict:
    merged: dict[str, Any] = {}
    owner: dict[str, str] = {}
    errors = []
    for name, ov in parts:
        for k, v in (ov or {}).items():
            clash = [o for o in merged if o == k or o.startswith(k + ".") or k.startswith(o + ".")]
            if clash:
                errors.append(f"override 키 충돌: {k} ({name}) vs {clash[0]} ({owner[clash[0]]})")
                continue
            merged[k] = v
            owner[k] = name
    if errors:
        raise ConfigError(errors)
    return merged


def scenario_overrides(matrix: Mapping, name: str) -> dict:
    scenarios = matrix.get("scenarios") or {}
    combos = matrix.get("combinations") or {}
    if len(combos) > MAX_COMBINATIONS:
        raise ConfigError([f"조합안은 최대 {MAX_COMBINATIONS}개 ({len(combos)}개 정의됨)"])
    if name in scenarios:
        return dict(scenarios[name].get("overrides") or {})
    if name in combos:
        combo = combos[name]
        comps = list(combo.get("components") or [])
        if len(comps) < 2:
            raise ConfigError([f"조합 {name}: 구성 시나리오 2개 이상 필요"])
        missing = [c for c in comps if c not in scenarios]
        if missing:
            raise ConfigError([f"조합 {name}: 없는 구성 시나리오 {missing}"])
        if not str(combo.get("rationale") or "").strip():
            raise ConfigError([f"조합 {name}: 선택 근거(rationale)를 기록해야 함 (PLAN §11.3)"])
        return _merge([(c, scenarios[c].get("overrides") or {}) for c in comps])
    raise ConfigError([f"정의되지 않은 시나리오: {name}"])


def scenario_info(matrix: Mapping, name: str) -> dict:
    scenarios = matrix.get("scenarios") or {}
    combos = matrix.get("combinations") or {}
    if name in scenarios:
        s = scenarios[name]
        return {"description": s.get("description"), "physical_delta": s.get("physical_delta")}
    c = combos.get(name) or {}
    comps = c.get("components") or []
    return {"description": " + ".join(comps), "components": comps, "rationale": c.get("rationale"),
            "physical_delta": " / ".join(str(scenarios[x].get("physical_delta")) for x in comps if x in scenarios)}


def resolve_run_config(matrix: Mapping, scenario: str, condition: str) -> dict:
    cond = (matrix.get("conditions") or {}).get(condition)
    if cond is None:
        raise ConfigError([f"정의되지 않은 조건: {condition}"])
    overrides = _merge([(f"condition:{condition}", cond.get("overrides") or {}),
                        (scenario, scenario_overrides(matrix, scenario))])
    return resolve_config(matrix["_base_config_path"], overrides)


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    try:
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                if int(b) < int(a):
                    raise ValueError(f"범위 역순: {part}")
                seeds.extend(range(int(a), int(b) + 1))
            elif part:
                seeds.append(int(part))
    except ValueError as exc:
        raise ConfigError([f"--seeds 형식 오류 ({text!r}): {exc}"]) from exc
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        raise ConfigError([f"--seeds에 시드가 없음 ({text!r})"])
    return seeds


def expand_runs(matrix: Mapping, phase: str, candidate: str | None = None, scenarios: list[str] | None = None,
                seeds: list[int] | None = None) -> list[dict[str, Any]]:
    phases = matrix.get("phases") or {}
    if phase not in phases:
        raise ConfigError([f"정의되지 않은 phase: {phase}"])
    ph = phases[phase]
    baseline = matrix.get("baseline_scenario", "B0")
    names = list(scenarios) if scenarios else list(ph.get("scenarios") or [])
    if CANDIDATE_TOKEN in names:
        cand = candidate or matrix.get("confirmation_candidate")
        if not cand:
            raise ConfigError(["최종 후보가 지정되지 않음 (--candidate 또는 confirmation_candidate)"])
        if cand == baseline:
            raise ConfigError([f"최종 후보가 기준 시나리오({baseline})와 같음"])
        names = [cand if n == CANDIDATE_TOKEN else n for n in names]
    names = list(dict.fromkeys(names))
    if not names:
        raise ConfigError(["실행할 시나리오가 없음"])
    if seeds is None:
        try:
            s = ph["seeds"]
            seeds = list(range(int(s["start"]), int(s["start"]) + int(s["count"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError([f"phases.{phase}.seeds 형식 오류: {exc!r}"]) from exc
    seeds = list(dict.fromkeys(int(x) for x in seeds))
    if not seeds:
        raise ConfigError(["시드가 0개"])
    conditions = list(dict.fromkeys(ph.get("conditions") or []))
    if not conditions:
        raise ConfigError([f"phases.{phase}.conditions 비어 있음"])
    for c in conditions:
        if c not in (matrix.get("conditions") or {}):
            raise ConfigError([f"정의되지 않은 조건: {c}"])
    for n in names:
        scenario_overrides(matrix, n)  # validates names early
    return [{"scenario": sc, "condition": cond, "seed": seed}
            for cond in conditions for sc in names for seed in seeds]


def matrix_hash(matrix: Mapping) -> str:
    clean = {k: v for k, v in matrix.items() if not k.startswith("_")}
    return config_hash({"matrix": clean, "base_config": load_yaml(matrix["_base_config_path"])})


def make_experiment_id(phase: str, mhash: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{phase}_{now.strftime('%Y%m%dT%H%M%S')}_{mhash[:8]}"


def run_dir_name(scenario: str, condition: str) -> str:
    return f"{scenario}__{condition}"


# ------------------------------------------------------------------------- execution
def run_experiment(matrix_path: str | Path, phase: str, *, candidate: str | None = None,
                   scenarios: list[str] | None = None, seeds: list[int] | None = None,
                   experiment_id: str | None = None, results_root: str | Path = "results",
                   command: str | None = None, verbose: bool = True) -> tuple[Path, int]:
    """Run all (scenario, condition, seed) cells sequentially. Returns (experiment dir, n_failed)."""
    matrix = load_matrix(matrix_path)
    if phase == "confirmation" and not (candidate or matrix.get("confirmation_candidate")) and not scenarios:
        raise ConfigError(["confirmation phase는 최종 후보(--candidate)가 필요함"])
    runs = expand_runs(matrix, phase, candidate, scenarios, seeds)
    mhash = matrix_hash(matrix)
    experiment_id = experiment_id or make_experiment_id(phase, mhash)
    exp_dir = Path(results_root) / experiment_id
    if exp_dir.exists():
        raise FileExistsError(f"실험 디렉토리가 이미 존재함 (덮어쓰기 금지): {exp_dir}")
    exp_dir.mkdir(parents=True)

    scen_names = list(dict.fromkeys(r["scenario"] for r in runs))
    cond_names = list(dict.fromkeys(r["condition"] for r in runs))
    env_info = environment_info()  # git / package metadata queried once per experiment
    meta = collect_metadata(run_id=experiment_id, seed=0, cfg=None, status="running", command=command,
                            env_info=env_info)
    experiment = {
        "experiment_id": experiment_id,
        "experiment_name": matrix.get("experiment_name"),
        "phase": phase,
        "matrix_path": matrix["_path"],
        "matrix_hash": mhash,
        "base_config_path": matrix["_base_config_path"],
        "baseline_scenario": matrix.get("baseline_scenario", "B0"),
        "candidate": candidate or matrix.get("confirmation_candidate"),
        "scenarios": {s: scenario_info(matrix, s) for s in scen_names},
        "conditions": {c: matrix["conditions"][c] for c in cond_names},
        "seeds": sorted({r["seed"] for r in runs}),
        "n_runs": len(runs),
        "created_at": meta["created_at"],
        "command": meta["command"],
        "engine_version": meta["engine_version"],
        "python": meta["python"],
        "packages": meta["packages"],
        "git_commit": meta["git_commit"],
        "git_dirty": meta["git_dirty"],
    }
    (exp_dir / "experiment.json").write_text(json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8")

    index_rows = []
    n_failed = 0
    for i, r in enumerate(runs, 1):
        sc, cond, seed = r["scenario"], r["condition"], r["seed"]
        rel = Path(run_dir_name(sc, cond)) / f"seed_{seed}"
        out = exp_dir / rel
        run_id = f"{sc}__{cond}__seed_{seed}"
        kw = dict(experiment_id=experiment_id, scenario=sc, condition=cond, command=command, env_info=env_info)
        t0 = time.perf_counter()
        cfg = None
        try:
            cfg = resolve_run_config(matrix, sc, cond)
            result = run_simulation(cfg, seed, run_id=run_id)
            wall = time.perf_counter() - t0
            write_run_dir(result, out, wall_time_s=wall, **kw)
            status, err, chash = result.status, None, result.config_hash
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            wall = time.perf_counter() - t0
            write_failed_run(out, run_id=run_id, seed=seed, cfg=cfg, exc=exc, wall_time_s=wall, **kw)
            status, err, chash = "failed", repr(exc), None if cfg is None else config_hash(cfg)
            n_failed += 1
        index_rows.append({"experiment_id": experiment_id, "scenario": sc, "condition": cond, "seed": seed,
                           "run_id": run_id, "run_dir": rel.as_posix(), "status": status, "config_hash": chash,
                           "error_message": err, "wall_time_s": wall})
        if verbose:
            print(f"[{i}/{len(runs)}] {run_id}: {status} ({wall:.2f}s)", flush=True)
    pd.DataFrame(index_rows).to_csv(exp_dir / "runs_index.csv", index=False, encoding=CSV_ENCODING)
    summarize(exp_dir)
    return exp_dir, n_failed


def _get(d: Mapping, path: tuple[str, ...]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur or cur[k] is None:
            return None
        cur = cur[k]
    return cur


def _load_values(exp_dir: Path, index: pd.DataFrame) -> pd.DataFrame:
    """Long table: scenario, condition, seed, status, metric, value (value None for null/failed)."""
    rows = []
    for r in index.itertuples(index=False):
        metrics = None
        if r.status != "failed":
            metrics = json.loads((exp_dir / r.run_dir / "metrics.json").read_text(encoding="utf-8"))
        paths = list(SUMMARY_METRICS)
        if metrics is not None:
            for p in metrics["shift"]["by_product"]:
                paths.append((f"shift.by_product.{p}.otd_rate", ("shift", "by_product", p, "otd_rate")))
        for name, path in paths:
            v = None if metrics is None else _get(metrics, path)
            rows.append({"scenario": r.scenario, "condition": r.condition, "seed": int(r.seed),
                         "status": r.status, "metric": name, "value": None if v is None else float(v)})
    return pd.DataFrame(rows, columns=["scenario", "condition", "seed", "status", "metric", "value"])


def summarize(experiment_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Write summary.csv, summary_by_product.csv and paired.csv from saved runs."""
    exp_dir = Path(experiment_dir)
    experiment = json.loads((exp_dir / "experiment.json").read_text(encoding="utf-8"))
    index = pd.read_csv(exp_dir / "runs_index.csv", encoding=CSV_ENCODING)
    long = _load_values(exp_dir, index)
    base_name = experiment.get("baseline_scenario", "B0")
    metric_order = list(dict.fromkeys(long["metric"])) if len(long) else []

    summary_rows, product_rows = [], []
    for (sc, cond), cell_index in index.groupby(["scenario", "condition"], sort=False):
        n_failed = int((cell_index["status"] == "failed").sum())
        cell = long[(long.scenario == sc) & (long.condition == cond) & (long.status != "failed")]
        for metric in metric_order:
            vals = cell.loc[cell.metric == metric, "value"]
            if metric.startswith("shift.by_product.") and vals.empty and n_failed == 0:
                continue
            ok = vals.dropna()
            ci = mean_ci(ok.tolist())
            row = {"scenario": sc, "condition": cond, "metric": metric, "n_ok": int(len(ok)), "n_failed": n_failed,
                   "n_null": int(vals.isna().sum()), "mean": ci["mean"], "sd": ci["sd"],
                   "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
            if metric.startswith("shift.by_product."):
                product_rows.append({**row, "product_type": metric.split(".")[2], "metric": "otd_rate"})
            else:
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    by_product = pd.DataFrame(product_rows, columns=BY_PRODUCT_COLUMNS)

    paired_rows = []
    ok_long = long[long.status != "failed"]
    for cond in dict.fromkeys(index["condition"]):
        cands = [s for s in dict.fromkeys(index.loc[index.condition == cond, "scenario"]) if s != base_name]
        if base_name not in set(index.loc[index.condition == cond, "scenario"]):
            continue
        for sc in cands:
            for metric in metric_order:
                def as_map(name):
                    sel = ok_long[(ok_long.scenario == name) & (ok_long.condition == cond) & (ok_long.metric == metric)]
                    return {int(s): (None if pd.isna(v) else float(v)) for s, v in zip(sel.seed, sel.value)}
                # Seeds of failed runs are included as missing so they show up in dropped_seeds.
                cand_map = {int(s): None for s in index.loc[(index.scenario == sc) & (index.condition == cond), "seed"]}
                base_map = {int(s): None for s in index.loc[(index.scenario == base_name) & (index.condition == cond),
                                                             "seed"]}
                cand_map.update(as_map(sc))
                base_map.update(as_map(base_name))
                pdiff = paired_diff(cand_map, base_map)
                if pdiff["n_pairs"] == 0 and metric.startswith("shift.by_product."):
                    continue
                includes_zero = None
                note = ""
                if pdiff["ci_low"] is None:
                    note = "짝 수 부족 — CI 계산 불가"
                else:
                    includes_zero = bool(pdiff["ci_low"] <= 0 <= pdiff["ci_high"])
                    if includes_zero:
                        note = NO_CLEAR_DIFF
                paired_rows.append({
                    "scenario": sc, "baseline": base_name, "condition": cond, "metric": metric,
                    "n_pairs": pdiff["n_pairs"], "dropped_seeds": ";".join(map(str, pdiff["dropped_seeds"])),
                    "base_mean": pdiff["base_mean"], "cand_mean": pdiff["cand_mean"], "mean_d": pdiff["mean_d"],
                    "ci_low": pdiff["ci_low"], "ci_high": pdiff["ci_high"],
                    "improvement_pct": improvement_pct(pdiff["base_mean"], pdiff["cand_mean"]),
                    "ci_includes_zero": includes_zero, "note": note,
                })
    paired = pd.DataFrame(paired_rows, columns=PAIRED_COLUMNS)
    summary.to_csv(exp_dir / "summary.csv", index=False, encoding=CSV_ENCODING)
    by_product.to_csv(exp_dir / "summary_by_product.csv", index=False, encoding=CSV_ENCODING)
    paired.to_csv(exp_dir / "paired.csv", index=False, encoding=CSV_ENCODING)
    return {"summary": summary, "summary_by_product": by_product, "paired": paired}


def compare_runs(run_dirs: Iterable[str | Path]) -> pd.DataFrame:
    """One row per saved run directory with its key metrics (no statistics)."""
    rows = []
    for d in run_dirs:
        d = Path(d)
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        row: dict[str, Any] = {"run_dir": str(d), "run_id": meta.get("run_id"), "scenario": meta.get("scenario"),
                               "condition": meta.get("condition"), "seed": meta.get("seed"),
                               "status": meta.get("status"), "config_hash": meta.get("config_hash")}
        if meta.get("status") != "failed":
            m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            for name, path in SUMMARY_METRICS:
                row[name] = _get(m, path)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="시나리오 반복 실험 실행")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--phase", required=True, choices=["discovery", "confirmation"])
    parser.add_argument("--candidate", help="confirmation 최종 후보 (예: S1, C1)")
    parser.add_argument("--scenarios", help="쉼표 구분 시나리오 목록 (phase 기본값 대체)")
    parser.add_argument("--seeds", help="예: 1000-1029 또는 100,101")
    parser.add_argument("--experiment-id")
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args(argv)
    command = "python -m line_sim.experiments " + " ".join(argv if argv is not None else sys.argv[1:])
    try:
        exp_dir, n_failed = run_experiment(
            args.matrix, args.phase, candidate=args.candidate,
            scenarios=[s.strip() for s in args.scenarios.split(",")] if args.scenarios else None,
            seeds=parse_seeds(args.seeds) if args.seeds else None,
            experiment_id=args.experiment_id, results_root=args.results_root, command=command)
    except (ConfigError, FileExistsError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    paired = read_csv_or_empty(exp_dir / "paired.csv")
    print(f"\n실험 저장: {exp_dir}")
    if paired.empty:
        print("  짝 비교 없음 (같은 조건에 기준 시나리오와 다른 시나리오가 함께 실행되지 않음)")
    sel = paired[paired.metric == "shift.good_shipped"] if not paired.empty else paired
    for r in sel.itertuples(index=False):
        ci = "n/a" if pd.isna(r.ci_low) else f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]"
        print(f"  {r.condition:>14} {r.scenario:>4} - {r.baseline}: good_shipped d={r.mean_d:+.2f} 95%CI {ci}"
              f" {'' if pd.isna(r.note) else r.note}")
    if n_failed:
        print(f"실패 run {n_failed}개 — runs_index.csv 확인", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
