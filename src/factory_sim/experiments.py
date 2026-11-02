"""v2 공정설계 실험 러너 (PLAN_v2 §9): 병렬 실행, 요약, 짝 비교, 후보 선택, 개정 동결.

실험 행렬(configs/v2/experiments.yaml)의 단위는 study다. study 하나 = 기준 Rev(BASE) + 단일 변경 후보들
(+ 근거 있는 조합 ≤ 2). 후보는 기준 Rev에 대한 override 또는 다른 Rev(``rev:``) 자체다.

Usage:
    python -m factory_sim.experiments run --matrix configs/v2/experiments.yaml --study A_diag --phase explore --workers 12
    python -m factory_sim.experiments summarize results/v2/<experiment_id>
    python -m factory_sim.experiments select results/v2/<experiment_id>
    python -m factory_sim.experiments freeze --matrix configs/v2/experiments.yaml --study rev1 --candidate H1 --as B

run 디렉토리: results/v2/<experiment_id>/<rev>__<candidate>__<condition>/seed_<n>/ (summary 모드)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from line_sim.experiments import NO_CLEAR_DIFF, make_experiment_id, mean_ci, paired_diff, parse_seeds

from .config import ConfigError, apply_overrides, load_config, load_yaml, resolve_config
from .entities import AgvState, MachineState, OperatorState
from .metrics import CSV_ENCODING
from .run import environment_info, write_failed_run, write_run_dir

BASE = "BASE"
MAX_COMBINATIONS = 2
STATUS_OK = "complete_at_horizon"

# 보고 지표: (이름, metrics.json 경로, 좋은 방향 +1 높을수록 / -1 낮을수록)
METRICS: dict[str, tuple[tuple[str, ...], int]] = {
    "factory.demand_fulfil_ratio": (("factory", "demand_fulfil_ratio"), +1),
    "factory.otd_rate": (("factory", "otd_rate"), +1),
    "factory.lead_time_cohort_mean_s": (("factory", "lead_time_arrival_cohort", "mean_lower_bound_s"), -1),
    "factory.lead_time_cohort_p90_s": (("factory", "lead_time_arrival_cohort", "p90_lower_bound_s"), -1),
    "factory.lead_time_shipped_mean_s": (("factory", "lead_time", "mean_s"), -1),
    "factory.weekly_good": (("factory", "weekly_good"), +1),
    "factory.good_rate_per_h": (("factory", "good_rate_per_h"), +1),
    "factory.avg_wip": (("factory", "avg_wip"), -1),
    "factory.avg_backlog": (("factory", "avg_backlog"), -1),
    "factory.flow_time_mean_s": (("factory", "flow_time_release_to_ship", "mean_s"), -1),
    "factory.release_blocked_k_s": (("factory", "release_blocked_k_s"), -1),
    "factory.release_blocked_stockout_s": (("factory", "release_blocked_stockout_s"), -1),
    "factory.first_pass_yield": (("factory", "first_pass_yield"), +1),
    "factory.oven_mean_batch": (("factory", "oven_mean_batch"), +1),
}
# 목표 지표 → (metric 이름, targets 키, 비교)
TARGET_METRICS = {
    "throughput": ("factory.demand_fulfil_ratio", "demand_fulfil_ratio_min", ">="),
    "otd": ("factory.otd_rate", "otd_min", ">="),
    "leadtime": ("factory.lead_time_cohort_mean_s", "lead_time_mean_max_s", "<="),
}
DERIVED = ("operators.busy_mean", "operators.busy_max", "agvs.busy_mean", "agvs.avg_load")
FAILED_ROW = {"status": "failed"}


# ---------------------------------------------------------------------------- matrix
def load_matrix(path: str | Path) -> dict:
    path = Path(path)
    m = load_yaml(path)
    missing = [k for k in ("revisions", "studies", "conditions", "phases", "resource_classes") if k not in m]
    if missing:
        raise ConfigError([f"실험 행렬 필수 키 없음: {missing}"])
    m["_dir"] = str(path.parent.resolve())
    return m


def rev_path(matrix: Mapping, rev: str) -> Path:
    if rev not in matrix["revisions"]:
        raise ConfigError([f"알 수 없는 Rev: {rev}"])
    return Path(matrix["_dir"]) / matrix["revisions"][rev]


def rev_config(matrix: Mapping, rev: str) -> dict:
    """Rev 설정(검증 전 원본, include 해석 완료)."""
    return load_config(rev_path(matrix, rev))


def study(matrix: Mapping, name: str) -> dict:
    if name not in matrix["studies"]:
        raise ConfigError([f"알 수 없는 study: {name}"])
    s = matrix["studies"][name] or {}
    if "base_rev" not in s:
        raise ConfigError([f"study {name}: base_rev 필요"])
    return s


def _flat(d: Mapping, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, Mapping) and v:
            out.update(_flat(v, key))
        else:
            out[key] = v
    return out


def _get_path(d: Any, dotted: str) -> Any:
    node = d
    for p in dotted.split("."):
        if not isinstance(node, Mapping) or p not in node:
            return None
        node = node[p]
    return node


def validate_candidate(base_cfg: Mapping, name: str, cand: Mapping) -> list[str]:
    """단일 변경 원칙 검사 (§9.3, 분석 §8-13 5번).

    - 기본: override 키 1개
    - buffer_realloc: 여러 키 허용, 대상 버퍼 용량 합 불변
    - skill_add: operators.skills.* 키만
    - machine_add: stations.X.machines 1개 + layout.nodes.* (+ operators.skills.*)
    """
    errs: list[str] = []
    if "rev" in cand:
        return errs
    ov = cand.get("overrides") or {}
    rc = cand.get("resource_class")
    if not ov:
        errs.append(f"후보 {name}: overrides 비어 있음")
        return errs
    keys = list(ov)
    if rc == "buffer_realloc":
        if not all(k.endswith(".capacity") and k.startswith("stations.") for k in keys):
            errs.append(f"후보 {name}: buffer_realloc은 stations.*.capacity 키만")
        else:
            before = [_get_path(base_cfg, k) for k in keys]
            if any(not isinstance(v, int) for v in before) or any(not isinstance(v, int) for v in ov.values()):
                errs.append(f"후보 {name}: 재배분 대상은 유한 정수 용량이어야 함")
            elif sum(before) != sum(ov.values()):
                errs.append(f"후보 {name}: 총 버퍼 슬롯 불변 위반 ({sum(before)} → {sum(ov.values())})")
    elif rc == "skill_add":
        if not all(k.startswith("operators.skills.") for k in keys):
            errs.append(f"후보 {name}: skill_add는 operators.skills.* 키만")
    elif rc == "machine_add":
        mk = [k for k in keys if k.startswith("stations.") and k.endswith(".machines")]
        rest = [k for k in keys if k not in mk]
        if len(mk) != 1 or not all(k.startswith(("layout.nodes.", "operators.skills.")) for k in rest):
            errs.append(f"후보 {name}: machine_add는 stations.X.machines 1개 + layout.nodes.*/operators.skills.*")
    elif len(keys) != 1:
        errs.append(f"후보 {name}: 단일 변경 원칙 위반 (키 {len(keys)}개: {keys})")
    return errs


def scenarios(matrix: Mapping, study_name: str) -> dict[str, dict]:
    """study의 시나리오: BASE + 후보 + 조합. 값 = {rev, overrides, resource_class, description}."""
    s = study(matrix, study_name)
    base_rev = s["base_rev"]
    base_cfg = rev_config(matrix, base_rev)
    rcs = matrix["resource_classes"]
    out = {BASE: {"rev": base_rev, "overrides": {}, "resource_class": None, "rank": -1,
                  "description": f"기준 Rev.{base_rev}"}}
    errs: list[str] = []
    cands = s.get("candidates") or {}
    for name, c in cands.items():
        if name == BASE:
            errs.append("후보 이름 BASE는 예약어")
            continue
        rc = c.get("resource_class")
        if rc not in rcs:
            errs.append(f"후보 {name}: resource_class {rc!r}가 resource_classes에 없음")
        errs += validate_candidate(base_cfg, name, c)
        out[name] = {"rev": c.get("rev", base_rev), "overrides": dict(c.get("overrides") or {}),
                     "resource_class": rc, "rank": rcs.get(rc), "description": c.get("description", "")}
    combos = s.get("combinations") or {}
    if len(combos) > MAX_COMBINATIONS:
        errs.append(f"조합은 최대 {MAX_COMBINATIONS}개 (현재 {len(combos)})")
    for name, c in combos.items():
        comps = c.get("components") or []
        if not str(c.get("rationale") or "").strip():
            errs.append(f"조합 {name}: 선택 근거(rationale)를 기록해야 함 (PLAN §9.3)")
        if len(comps) < 2 or any(x not in cands for x in comps):
            errs.append(f"조합 {name}: 단일 후보 2개 이상을 참조해야 함 ({comps})")
            continue
        merged: dict[str, Any] = {}
        for x in comps:
            for k, v in (cands[x].get("overrides") or {}).items():
                if k in merged and merged[k] != v:
                    errs.append(f"조합 {name}: 키 충돌 {k}")
                merged[k] = v
        ranks = [rcs.get(cands[x].get("resource_class"), 0) for x in comps]
        out[name] = {"rev": base_rev, "overrides": merged, "resource_class": "combination",
                     "rank": sum(ranks), "description": c.get("description", " + ".join(comps)),
                     "components": comps}
    if errs:
        raise ConfigError(errs)
    return out


def resolve_scenario_config(matrix: Mapping, sc: Mapping, condition: str) -> dict:
    if condition not in matrix["conditions"]:
        raise ConfigError([f"알 수 없는 조건: {condition}"])
    cfg = apply_overrides(rev_config(matrix, sc["rev"]), sc["overrides"])
    cond_ov = (matrix["conditions"][condition] or {}).get("overrides") or {}
    return resolve_config(cfg, cond_ov)


def run_dir_name(rev: str, scenario: str, condition: str) -> str:
    return f"{rev}__{scenario}__{condition}"


def expand_runs(matrix: Mapping, study_name: str, phase: str, seeds: list[int] | None = None,
                only: list[str] | None = None) -> list[dict]:
    if phase not in matrix["phases"]:
        raise ConfigError([f"알 수 없는 phase: {phase}"])
    ph = matrix["phases"][phase]
    if seeds is None:
        seeds = list(range(int(ph["seeds"]["start"]), int(ph["seeds"]["start"]) + int(ph["seeds"]["count"])))
    seeds = sorted(set(int(s) for s in seeds))
    if not seeds:
        raise ConfigError(["시드가 비어 있음"])
    s = study(matrix, study_name)
    scs = scenarios(matrix, study_name)
    if only:
        unknown = [x for x in only if x not in scs]
        if unknown:
            raise ConfigError([f"알 수 없는 시나리오: {unknown}"])
        scs = {k: v for k, v in scs.items() if k in only or k == BASE}
    conds = list(ph.get("conditions") or ["base"])
    base_conds = list(s.get("baseline_conditions") or conds)
    runs = []
    for name, sc in scs.items():
        for cond in conds:
            if name == BASE and cond not in base_conds:
                continue
            for seed in seeds:
                runs.append({"scenario": name, "rev": sc["rev"], "condition": cond, "seed": seed,
                             "resource_class": sc["resource_class"], "rank": sc["rank"]})
    return runs


def matrix_hash(matrix: Mapping, study_name: str) -> str:
    payload = {"study": study_name, "s": matrix["studies"][study_name], "conditions": matrix["conditions"],
               "revisions": {r: _file_hash(rev_path(matrix, r)) for r in matrix["revisions"]}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "missing"


# ---------------------------------------------------------------------------- execution
def _run_one(task: Mapping) -> dict:
    """워커: run 1회 실행 후 디렉토리 기록. 예외는 failed run으로 남긴다 (조용히 빼지 않음)."""
    from .engine import run_simulation  # 워커 프로세스에서 import

    out = Path(task["out_dir"])
    row = {k: task[k] for k in ("scenario", "rev", "condition", "seed", "resource_class")}
    row["run_dir"] = task["rel_dir"]
    t0 = time.perf_counter()
    try:
        res = run_simulation(task["cfg"], task["seed"], run_id=f"seed_{task['seed']}", detail=task["detail"])
        wall = time.perf_counter() - t0
        metrics = write_run_dir(res, out, experiment_id=task["experiment_id"], scenario=task["scenario"],
                                condition=task["condition"], wall_time_s=wall, command=task["command"],
                                env_info=task["env_info"])
        row.update(status=res.status, wall_time_s=wall, checks_ok=bool(metrics["checks"]["all_ok"]), error="")
    except Exception as exc:  # noqa: BLE001 - failed run 기록
        wall = time.perf_counter() - t0
        write_failed_run(out, run_id=f"seed_{task['seed']}", seed=task["seed"], cfg=task["cfg"], exc=exc,
                         experiment_id=task["experiment_id"], scenario=task["scenario"],
                         condition=task["condition"], wall_time_s=wall, command=task["command"],
                         env_info=task["env_info"])
        row.update(status="failed", wall_time_s=wall, checks_ok=False, error=repr(exc))
    return row


def default_workers() -> int:
    return max(1, min(12, (os.cpu_count() or 2) - 2))


def execute(tasks: list[dict], workers: int, verbose: bool = True) -> list[dict]:
    rows: list[dict] = []
    if workers <= 1:
        for i, t in enumerate(tasks, 1):
            rows.append(_run_one(t))
            if verbose:
                print(f"  [{i}/{len(tasks)}] {t['rel_dir']} {rows[-1]['status']}", flush=True)
        return rows
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, row in enumerate(ex.map(_run_one, tasks, chunksize=1), 1):
            rows.append(row)
            if verbose and (i % max(1, len(tasks) // 20) == 0 or i == len(tasks)):
                print(f"  [{i}/{len(tasks)}] 완료", flush=True)
    return rows


def run_experiment(matrix_path: str | Path, study_name: str, phase: str, *, workers: int | None = None,
                   seeds: list[int] | None = None, only: list[str] | None = None,
                   experiment_id: str | None = None, results_root: str | Path | None = None,
                   detail: str = "summary", verbose: bool = True, command: str | None = None) -> tuple[Path, int]:
    matrix = load_matrix(matrix_path)
    runs = expand_runs(matrix, study_name, phase, seeds=seeds, only=only)
    scs = scenarios(matrix, study_name)
    # 설정은 실행 전 모두 검증 (잘못된 후보가 섞인 채 일부만 실행되지 않게)
    cfgs: dict[tuple[str, str], dict] = {}
    for r in runs:
        key = (r["scenario"], r["condition"])
        if key not in cfgs:
            cfgs[key] = resolve_scenario_config(matrix, scs[r["scenario"]], r["condition"])
    mhash = matrix_hash(matrix, study_name)
    experiment_id = experiment_id or make_experiment_id(f"{study_name}_{phase}", mhash)
    root = Path(results_root) if results_root else Path(matrix["_dir"]) / matrix.get("results_root", "results/v2")
    exp_dir = root / experiment_id
    if exp_dir.exists():
        raise FileExistsError(f"실험 디렉토리가 이미 존재함: {exp_dir}")
    exp_dir.mkdir(parents=True)
    env_info = environment_info()
    command = command or f"python -m factory_sim.experiments run --study {study_name} --phase {phase}"
    tasks = []
    for r in runs:
        rel = f"{run_dir_name(r['rev'], r['scenario'], r['condition'])}/seed_{r['seed']}"
        tasks.append({**r, "cfg": cfgs[(r["scenario"], r["condition"])], "out_dir": str(exp_dir / rel),
                      "rel_dir": rel, "detail": detail, "experiment_id": experiment_id, "command": command,
                      "env_info": env_info})
    plan = {"experiment_id": experiment_id, "study": study_name, "phase": phase, "matrix_hash": mhash,
            "study_def": matrix["studies"][study_name], "scenarios": {k: {kk: vv for kk, vv in v.items()}
                                                                    for k, v in scs.items()},
            "conditions": sorted({r["condition"] for r in runs}), "seeds": sorted({r["seed"] for r in runs}),
            "targets": cfgs[(BASE, runs[0]["condition"])].get("targets") if (BASE, runs[0]["condition"]) in cfgs
            else None, "n_runs": len(runs), "env_info": env_info}
    (exp_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    if verbose:
        print(f"{experiment_id}: run {len(runs)}개, workers {workers or default_workers()}", flush=True)
    t0 = time.perf_counter()
    rows = execute(tasks, workers or default_workers(), verbose)
    index = pd.DataFrame(rows).sort_values(["scenario", "condition", "seed"]).reset_index(drop=True)
    index.to_csv(exp_dir / "runs_index.csv", index=False, encoding=CSV_ENCODING)
    n_failed = int((index["status"] == "failed").sum())
    summarize(exp_dir)
    if verbose:
        print(f"완료 {time.perf_counter() - t0:.0f}s, 실패 {n_failed}", flush=True)
    return exp_dir, n_failed


# ---------------------------------------------------------------------------- summary
def _derived(m: Mapping) -> dict[str, float | None]:
    ops = m.get("operators") or {}
    busy = [1.0 - d.get(OperatorState.IDLE, 0.0) for d in ops.values()]
    agvs = m.get("agvs") or {}
    abusy = [sum(d.get(s, 0.0) for s in AgvState.BUSY) for d in agvs.values()]
    trips = m.get("agv_trips") or {}
    return {"operators.busy_mean": sum(busy) / len(busy) if busy else None,
            "operators.busy_max": max(busy) if busy else None,
            "agvs.busy_mean": sum(abusy) / len(abusy) if abusy else None,
            "agvs.avg_load": trips.get("mean_load_in_window")}


def _load_metrics(exp_dir: Path, index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(지표 long 표, 스테이션 상태 long 표). 완료 run만."""
    vals, sts = [], []
    for r in index.itertuples(index=False):
        if r.status == "failed":
            continue
        m = json.loads((exp_dir / r.run_dir / "metrics.json").read_text(encoding="utf-8"))
        key = {"scenario": r.scenario, "condition": r.condition, "seed": int(r.seed)}
        for name, (path, _) in METRICS.items():
            vals.append({**key, "metric": name, "value": _get_path(m, ".".join(path))})
        for name, v in _derived(m).items():
            vals.append({**key, "metric": name, "value": v})
        for sid, d in (m.get("stations") or {}).items():
            for s in MachineState.ALL:
                sts.append({**key, "station": sid, "state": s, "value": d.get(s, 0.0)})
        for oid, d in (m.get("operators") or {}).items():
            sts.append({**key, "station": f"op:{oid}", "state": "BUSY", "value": 1.0 - d.get(OperatorState.IDLE, 0)})
    return pd.DataFrame(vals), pd.DataFrame(sts)


def summarize(exp_dir: str | Path) -> dict[str, pd.DataFrame]:
    exp_dir = Path(exp_dir)
    index = pd.read_csv(exp_dir / "runs_index.csv", encoding=CSV_ENCODING)
    plan = json.loads((exp_dir / "plan.json").read_text(encoding="utf-8"))
    vals, sts = _load_metrics(exp_dir, index)
    fails = index[index["status"] == "failed"].groupby(["scenario", "condition"]).size().to_dict()

    rows = []
    for (sc, cond), g in index.groupby(["scenario", "condition"]):
        sub = vals[(vals.scenario == sc) & (vals.condition == cond)] if len(vals) else vals
        for metric in list(METRICS) + list(DERIVED):
            v = sub[sub.metric == metric]["value"] if len(sub) else pd.Series(dtype=float)
            ok = v.dropna()
            ci = mean_ci(ok.tolist())
            rows.append({"scenario": sc, "condition": cond, "metric": metric, "n_ok": int(len(ok)),
                         "n_failed": int(fails.get((sc, cond), 0)), "n_null": int(v.isna().sum()),
                         "mean": ci["mean"], "sd": ci["sd"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(exp_dir / "summary.csv", index=False, encoding=CSV_ENCODING)

    st_rows = []
    if len(sts):
        for (sc, cond, sid, s), g in sts.groupby(["scenario", "condition", "station", "state"]):
            ci = mean_ci(g["value"].tolist())
            st_rows.append({"scenario": sc, "condition": cond, "station": sid, "state": s, "mean": ci["mean"],
                            "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})
    stations = pd.DataFrame(st_rows)
    stations.to_csv(exp_dir / "stations_summary.csv", index=False, encoding=CSV_ENCODING)

    paired = paired_table(vals, plan)
    paired.to_csv(exp_dir / "paired.csv", index=False, encoding=CSV_ENCODING)
    targets = evaluate_targets(summary, plan.get("targets") or {})
    targets.to_csv(exp_dir / "targets.csv", index=False, encoding=CSV_ENCODING)
    return {"summary": summary, "stations": stations, "paired": paired, "targets": targets}


def paired_table(vals: pd.DataFrame, plan: Mapping) -> pd.DataFrame:
    """후보 − BASE, 같은 조건·같은 시드 짝 비교. comparable_to_base=false 조건은 BASE가 없으면 생략."""
    cols = ["scenario", "baseline", "condition", "metric", "direction", "resource_class", "n_pairs",
            "dropped_seeds", "base_mean", "cand_mean", "mean_d", "ci_low", "ci_high", "verdict"]
    if not len(vals):
        return pd.DataFrame(columns=cols)
    rows = []
    scs = plan.get("scenarios") or {}
    for (sc, cond), g in vals.groupby(["scenario", "condition"]):
        if sc == BASE:
            continue
        b = vals[(vals.scenario == BASE) & (vals.condition == cond)]
        if not len(b):
            continue
        for metric in list(METRICS) + list(DERIVED):
            cm = {int(r.seed): r.value for r in g[g.metric == metric].itertuples() if pd.notna(r.value)}
            bm = {int(r.seed): r.value for r in b[b.metric == metric].itertuples() if pd.notna(r.value)}
            pdf = paired_diff(cm, bm)
            direction = METRICS[metric][1] if metric in METRICS else 0
            rows.append({"scenario": sc, "baseline": BASE, "condition": cond, "metric": metric,
                         "direction": direction, "resource_class": (scs.get(sc) or {}).get("resource_class"),
                         "n_pairs": pdf["n_pairs"], "dropped_seeds": " ".join(map(str, pdf["dropped_seeds"])),
                         "base_mean": pdf["base_mean"], "cand_mean": pdf["cand_mean"], "mean_d": pdf["mean_d"],
                         "ci_low": pdf["ci_low"], "ci_high": pdf["ci_high"],
                         "verdict": _verdict(pdf, direction)})
    return pd.DataFrame(rows, columns=cols)


def _verdict(pdf: Mapping, direction: int) -> str:
    lo, hi = pdf.get("ci_low"), pdf.get("ci_high")
    if lo is None or hi is None or (lo <= 0 <= hi):
        return NO_CLEAR_DIFF
    if direction == 0:
        return "증가" if lo > 0 else "감소"
    better = (lo > 0) if direction > 0 else (hi < 0)
    return "개선" if better else "악화"


def evaluate_targets(summary: pd.DataFrame, targets: Mapping) -> pd.DataFrame:
    """시나리오×조건별 목표 판정: 평균 기준(탐색 판정) + CI 경계 기준(확인 보고용, D28)."""
    rows = []
    if not len(summary):
        return pd.DataFrame(rows)
    for (sc, cond), g in summary.groupby(["scenario", "condition"]):
        row: dict[str, Any] = {"scenario": sc, "condition": cond}
        flags, ci_flags = [], []
        for tname, (metric, key, op) in TARGET_METRICS.items():
            r = g[g.metric == metric]
            if not len(r) or key not in targets:
                continue
            r = r.iloc[0]
            thr = float(targets[key])
            mean = r["mean"]
            bound = r["ci_low"] if op == ">=" else r["ci_high"]
            ok = None if pd.isna(mean) else (mean >= thr if op == ">=" else mean <= thr)
            ci_ok = None if bound is None or pd.isna(bound) else (bound >= thr if op == ">=" else bound <= thr)
            row.update({f"{tname}_mean": mean, f"{tname}_ci_low": r["ci_low"], f"{tname}_ci_high": r["ci_high"],
                        f"{tname}_target": thr, f"{tname}_ok": ok, f"{tname}_ci_ok": ci_ok})
            flags.append(ok)
            ci_flags.append(ci_ok)
        row["all_ok"] = None if any(f is None for f in flags) else all(flags)
        row["all_ci_ok"] = None if any(f is None for f in ci_flags) else all(ci_flags)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- selection (§9.2)
def select_candidate(paired: pd.DataFrame, primary_metric: str, condition: str = "base",
                     ranks: Mapping[str, int] | None = None) -> dict[str, Any]:
    """§9.2 채택 기준.

    1) primary 짝 차이 95% CI가 개선 방향으로 0을 포함하지 않음
    2) 다른 목표 지표 중 CI 전체가 악화 측인 것이 없음
    3) 적격 후보 중 자원 순위(rank) 최소, 동률이면 primary 개선 평균(방향 보정) 최대
    """
    if primary_metric not in METRICS:
        raise ConfigError([f"primary_metric {primary_metric!r}는 METRICS에 없음"])
    p = paired[paired.condition == condition]
    target_metrics = [m for m, _, _ in TARGET_METRICS.values()]
    table = []
    for sc, g in p.groupby("scenario"):
        prim = g[g.metric == primary_metric]
        if not len(prim):
            continue
        prim = prim.iloc[0]
        improves = prim["verdict"] == "개선"
        worsened = [m for m in target_metrics if m != primary_metric
                    and len(g[g.metric == m]) and g[g.metric == m].iloc[0]["verdict"] == "악화"]
        rank = (ranks or {}).get(sc)
        if rank is None:
            rank = 99
        gain = None if pd.isna(prim["mean_d"]) else float(prim["mean_d"]) * METRICS[primary_metric][1]
        table.append({"scenario": sc, "rank": rank, "primary_mean_d": prim["mean_d"], "primary_ci_low":
                      prim["ci_low"], "primary_ci_high": prim["ci_high"], "gain": gain,
                      "improves": bool(improves), "worsened_targets": ",".join(worsened),
                      "eligible": bool(improves and not worsened)})
    tdf = pd.DataFrame(table)
    chosen = None
    if len(tdf) and tdf["eligible"].any():
        el = tdf[tdf["eligible"]].sort_values(["rank", "gain"], ascending=[True, False])
        chosen = str(el.iloc[0]["scenario"])
    return {"chosen": chosen, "table": tdf}


def select_from_dir(exp_dir: str | Path, condition: str = "base") -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    plan = json.loads((exp_dir / "plan.json").read_text(encoding="utf-8"))
    primary = (plan.get("study_def") or {}).get("primary_metric")
    if not primary:
        raise ConfigError(["study에 primary_metric이 실행 전에 기록되지 않음"])
    paired = pd.read_csv(exp_dir / "paired.csv", encoding=CSV_ENCODING)
    ranks = {k: v.get("rank") for k, v in (plan.get("scenarios") or {}).items()}
    res = select_candidate(paired, primary, condition, ranks)
    res["table"].to_csv(exp_dir / "selection.csv", index=False, encoding=CSV_ENCODING)
    (exp_dir / "selection.json").write_text(json.dumps({"primary_metric": primary, "chosen": res["chosen"]},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------- freeze
def freeze_revision(matrix_path: str | Path, study_name: str, candidate: str, new_rev: str,
                    experiment_id: str | None = None) -> tuple[Path, dict[str, tuple[Any, Any]]]:
    """후보를 새 Rev로 동결: 기준 Rev + override를 자기완결 YAML(rev_<X>.yaml)로 쓰고 평탄화 diff를 반환."""
    matrix = load_matrix(matrix_path)
    scs = scenarios(matrix, study_name)
    if candidate not in scs or candidate == BASE:
        raise ConfigError([f"동결할 후보가 아님: {candidate}"])
    sc = scs[candidate]
    base = rev_config(matrix, sc["rev"])
    new = resolve_config(apply_overrides(base, sc["overrides"]))
    new["revision"] = new_rev
    new["frozen_from"] = {"base_rev": sc["rev"], "study": study_name, "candidate": candidate,
                          "description": sc["description"], "experiment_id": experiment_id}
    out = Path(matrix["_dir"]) / f"rev_{new_rev}.yaml"
    if out.exists():
        raise FileExistsError(f"이미 존재: {out}")
    header = (f"# 모터 공장 v2 — Rev.{new_rev} (Rev.{sc['rev']} + {candidate}: {sc['description']}).\n"
              f"# experiments.py freeze로 생성한 자기완결 설정. 손으로 고치지 말 것. 근거: docs/v2/design_log.md\n")
    out.write_text(header + yaml.safe_dump(new, sort_keys=False, allow_unicode=True), encoding="utf-8")
    fb, fn = _flat(resolve_config(base)), _flat(new)
    diff = {k: (fb.get(k), fn.get(k)) for k in sorted(set(fb) | set(fn))
            if fb.get(k) != fn.get(k) and k not in ("revision",) and not k.startswith("frozen_from")}
    return out, diff


# ---------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="모터 공장 v2 공정설계 실험")
    sub = parser.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--matrix", default="configs/v2/experiments.yaml")
    pr.add_argument("--study", required=True)
    pr.add_argument("--phase", required=True)
    pr.add_argument("--seeds", help="예: 100-119")
    pr.add_argument("--scenarios", help="쉼표 구분 (BASE는 자동 포함)")
    pr.add_argument("--workers", type=int, default=None)
    pr.add_argument("--experiment-id")
    pr.add_argument("--results-root")
    ps = sub.add_parser("summarize")
    ps.add_argument("exp_dir")
    pc = sub.add_parser("select")
    pc.add_argument("exp_dir")
    pc.add_argument("--condition", default="base")
    pf = sub.add_parser("freeze")
    pf.add_argument("--matrix", default="configs/v2/experiments.yaml")
    pf.add_argument("--study", required=True)
    pf.add_argument("--candidate", required=True)
    pf.add_argument("--as", dest="new_rev", required=True)
    pf.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    try:
        if args.cmd == "run":
            seeds = parse_seeds(args.seeds) if args.seeds else None
            only = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None
            cmd = "python -m factory_sim.experiments " + " ".join(argv if argv is not None else sys.argv[1:])
            exp_dir, n_failed = run_experiment(args.matrix, args.study, args.phase, workers=args.workers,
                                               seeds=seeds, only=only, experiment_id=args.experiment_id,
                                               results_root=args.results_root, command=cmd)
            print(f"저장: {exp_dir}")
            return 1 if n_failed else 0
        if args.cmd == "summarize":
            summarize(args.exp_dir)
            return 0
        if args.cmd == "select":
            res = select_from_dir(args.exp_dir, args.condition)
            print(res["table"].to_string(index=False))
            print(f"채택: {res['chosen']}")
            return 0
        if args.cmd == "freeze":
            out, diff = freeze_revision(args.matrix, args.study, args.candidate, args.new_rev, args.experiment_id)
            print(f"동결: {out}")
            for k, (a, b) in diff.items():
                print(f"  {k}: {a} → {b}")
            return 0
    except (ConfigError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
