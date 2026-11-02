"""v2 지표: 공장 KPI, 목표 판정, 자원 상태 비율(손실 분해), 보정 지표, 보존 검사.

모든 값은 orders.csv + resource_states.csv + stats.json(+ 설정)에서 계산한다. 그래서 저장된
run 디렉토리에서 다시 계산한 값이 메모리 값과 같다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from .config import WEEK_S, window_s
from .entities import AgvState, MachineState, OperatorState

TOL_S = 1e-6
CSV_ENCODING = "utf-8-sig"

# §6 손실 범주 (설비 상태 → 표시 이름)
LOSS_CATEGORIES = {
    MachineState.PROCESSING: "가공",
    MachineState.SETUP: "준비",
    MachineState.STARVED: "굶주림",
    MachineState.BLOCKED_DOWNSTREAM: "막힘(하류)",
    MachineState.BLOCKED_TRANSPORT: "막힘(운반)",
    MachineState.WAIT_OPERATOR: "작업자 대기",
    MachineState.DOWN: "고장",
}
STATES_BY_TYPE = {"machine": MachineState.ALL, "operator": OperatorState.ALL, "agv": AgvState.ALL}


@dataclass
class RunData:
    run_id: str
    status: str
    config: dict
    orders: pd.DataFrame
    resource_states: pd.DataFrame
    stats: dict
    sim_end_s: float


def load_run_dir(path: str | Path) -> RunData:
    path = Path(path)
    with open(path / "config_resolved.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    stats = json.loads((path / "stats.json").read_text(encoding="utf-8"))
    orders = pd.read_csv(path / "orders.csv", encoding=CSV_ENCODING, float_precision="round_trip")
    states = pd.read_csv(path / "resource_states.csv", encoding=CSV_ENCODING, float_precision="round_trip",
                         keep_default_na=False, na_values=[""])
    return RunData(str(meta.get("run_id")), meta.get("status"), cfg, orders, states, stats,
                   float(stats["sim_end_s"]))


def _as_run_data(x: Any) -> RunData:
    if isinstance(x, RunData):
        return x
    if isinstance(x, (str, Path)):
        return load_run_dir(x)
    return RunData(x.run_id, x.status, x.config, x.orders_frame(), x.resource_states, x.stats, x.sim_end_s)


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num) / float(den)


def _dist(values: pd.Series) -> dict[str, Any]:
    v = values.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean_s": None, "p50_s": None, "p90_s": None}
    return {"n": int(len(v)), "mean_s": float(v.mean()), "p50_s": float(np.percentile(v, 50)),
            "p90_s": float(np.percentile(v, 90))}


def _state_table(states: pd.DataFrame, rtype: str, col: str) -> dict[str, dict[str, float]]:
    sub = states[states["resource_type"] == rtype]
    out: dict[str, dict[str, float]] = {}
    for rid, st, v in zip(sub["resource_id"], sub["state"], sub[col]):
        out.setdefault(str(rid), {})[str(st)] = float(v)
    return out


def order_window_residence_sum(orders: pd.DataFrame, sim_end: float, w0: float, w1: float) -> float:
    rel = orders["released_s"].astype(float)
    end = orders["terminal_s"].astype(float).fillna(sim_end)
    res = (np.minimum(end, w1) - np.maximum(rel, w0)).clip(lower=0)
    return float(res[rel.notna()].sum())


# ---------------------------------------------------------------------------- main API
def compute_metrics(result_or_dir: Any) -> dict[str, Any]:
    rd = _as_run_data(result_or_dir)
    cfg, o, states, stats = rd.config, rd.orders, rd.resource_states, rd.stats
    w0, w1 = window_s(cfg)
    span = w1 - w0
    groups = list(cfg["products"]["groups"])

    arr = o["arrival_s"].astype(float)
    due = o["due_s"].astype(float)
    rel = o["released_s"].astype(float)
    term = o["terminal_s"].astype(float)
    shipped = o["outcome"] == "shipped"
    scrapped = o["outcome"] == "scrapped"
    # 관측창은 출하·도착·납기 모두 반열린 구간 [w0, w1)
    in_win = term.notna() & (term >= w0) & (term < w1)
    good_win = shipped & in_win
    arr_win = (arr >= w0) & (arr < w1)
    n_good = int(good_win.sum())
    n_arr = int(arr_win.sum())

    due_win = (due >= w0) & (due < w1)
    on_time = shipped & term.notna() & (term <= due)
    otd_by_group = {}
    for g in groups:
        gm = due_win & (o["group"] == g)
        otd_by_group[g] = {"n_due": int(gm.sum()), "on_time": int((gm & on_time).sum()),
                           "otd_rate": _ratio(int((gm & on_time).sum()), int(gm.sum()))}
    otd = _ratio(int((due_win & on_time).sum()), int(due_win.sum()))

    lead = (term - arr)[good_win]
    flow = (term - rel)[good_win]
    lead_by_group = {g: _dist(lead[o.loc[good_win.index[good_win], "group"] == g]) for g in groups}
    # 도착 코호트 리드타임(W2): 관측창 도착분 전체. 출하·폐기는 실제값, 미종결은 sim_end − 도착(하한).
    censored = arr_win & term.isna()
    cohort = (term.fillna(rd.sim_end_s) - arr)[arr_win].to_numpy(dtype=float)
    cohort_lt = {"n": int(len(cohort)), "n_censored": int(censored.sum()),
                 "mean_lower_bound_s": float(cohort.mean()) if len(cohort) else None,
                 "p90_lower_bound_s": float(np.percentile(cohort, 90)) if len(cohort) else None}
    incomplete = rel.notna() & term.isna()
    inc_age = rd.sim_end_s - arr[incomplete]

    inspected = in_win & o["first_inspection"].notna()
    n_insp = int(inspected.sum())
    fpy = _ratio(int((inspected & (o["first_inspection"] == "pass")).sum()), n_insp)
    rework_rate = _ratio(int((inspected & o["reworked"].astype(bool)).sum()), n_insp)

    factory = {
        "window_s": [w0, w1],
        "arrivals_in_window": n_arr,
        "good_in_window": n_good,
        "weekly_good": n_good * WEEK_S / span,
        "weekly_demand": n_arr * WEEK_S / span,
        "good_rate_per_h": n_good / (span / 3600.0),
        "demand_fulfil_ratio": _ratio(n_good, n_arr),
        "scrapped_in_window": int((scrapped & in_win).sum()),
        "otd_rate": otd,
        "otd_by_group": otd_by_group,
        "lead_time": _dist(lead),
        "lead_time_arrival_cohort": cohort_lt,
        "lead_time_by_group": lead_by_group,
        "flow_time_release_to_ship": _dist(flow),
        "incomplete_at_end": {"n": int(incomplete.sum()),
                              "age_mean_s": float(inc_age.mean()) if len(inc_age) else None},
        "backlog_at_end": int(rel.isna().sum()),
        "avg_wip": stats["wip"]["avg"],
        "max_wip": stats["wip"]["max"],
        "avg_backlog": stats["backlog"]["avg"],
        "kit_avg_occupied": stats["kit_occupied"]["avg"],
        "release_blocked_k_s": stats["release_blocked_k"]["area"],
        "release_blocked_stockout_s": stats["release_blocked_stockout"]["area"],
        "housing_avg_on_hand": stats["housing"]["level"]["avg"] if stats["housing"]["enabled"] else None,
        "first_pass_yield": fpy,
        "rework_rate": rework_rate,
        "oven_mean_batch": stats["batches"]["mean_size_in_window"],
    }

    tg = cfg.get("targets") or {}
    tput_ok = None if "demand_fulfil_ratio_min" not in tg or factory["demand_fulfil_ratio"] is None else \
        factory["demand_fulfil_ratio"] >= tg["demand_fulfil_ratio_min"]
    otd_ok = None if "otd_min" not in tg or otd is None else otd >= tg["otd_min"]
    # 목표 판정은 보수적으로 도착 코호트 평균 하한을 쓴다(출하분만의 평균은 절단 편향으로 낙관적).
    lt = cohort_lt["mean_lower_bound_s"]
    lt_ok = None if "lead_time_mean_max_s" not in tg or lt is None else lt <= tg["lead_time_mean_max_s"]
    flags = [tput_ok, otd_ok, lt_ok]
    targets = {"throughput_ok": tput_ok, "otd_ok": otd_ok, "leadtime_ok": lt_ok,
               "all_ok": None if any(f is None for f in flags) else all(flags)}

    win = {t: _state_table(states, t, "window_s") for t in STATES_BY_TYPE}
    full = {t: _state_table(states, t, "full_s") for t in STATES_BY_TYPE}
    machines = {mid: {s: d.get(s, 0.0) / span for s in MachineState.ALL} for mid, d in win["machine"].items()}
    station_of = {str(r): str(s) for r, s in zip(states["resource_id"], states["station_id"])
                  if str(s) not in ("", "nan", "None")}
    stations: dict[str, dict[str, float]] = {}
    for mid, d in machines.items():
        sid = station_of.get(mid)
        agg = stations.setdefault(sid, {s: 0.0 for s in MachineState.ALL} | {"machines": 0})
        agg["machines"] += 1
        for s in MachineState.ALL:
            agg[s] += d[s]
    for agg in stations.values():
        n = agg["machines"]
        for s in MachineState.ALL:
            agg[s] /= n
    forming = stats.get("batch_forming_s") or {}
    for mid, sec in forming.items():
        if mid in machines:
            machines[mid]["STARVED_batch_forming"] = sec / span
    operators = {oid: {s: d.get(s, 0.0) / span for s in OperatorState.ALL} for oid, d in win["operator"].items()}
    agvs = {aid: {s: d.get(s, 0.0) / span for s in AgvState.ALL} for aid, d in win["agv"].items()}

    # 보정 지표 (D8: 부하 기준)
    tot = lambda st, s: sum(d.get(s, 0.0) for mid, d in win["machine"].items() if station_of.get(mid) == st)  # noqa: E731
    proc_all = sum(d.get(MachineState.PROCESSING, 0.0) for d in win["machine"].values())
    down_all = sum(d.get(MachineState.DOWN, 0.0) for d in win["machine"].values())
    setup_st = [sid for sid, st in cfg["stations"].items() if float(st.get("setup_group_change_s", 0) or 0) > 0]
    calibration = {
        "down_ratio_load": _ratio(down_all, proc_all + down_all),
        "down_ratio_window": _ratio(down_all, span * len(win["machine"])),
        "down_ratio_by_station": {sid: _ratio(tot(sid, MachineState.DOWN),
                                              tot(sid, MachineState.DOWN) + tot(sid, MachineState.PROCESSING))
                                  for sid in cfg["stations"]},
        "setup_ratio": {sid: _ratio(tot(sid, MachineState.SETUP),
                                    tot(sid, MachineState.SETUP) + tot(sid, MachineState.PROCESSING))
                        for sid in setup_st},
        "eol_first_pass_yield": fpy,
    }

    # 보존 검사
    errs = []
    for rtype in STATES_BY_TYPE:
        for d in win[rtype].values():
            errs.append(abs(sum(d.values()) - span))
        for d in full[rtype].values():
            errs.append(abs(sum(d.values()) - rd.sim_end_s))
    max_err = float(max(errs)) if errs else 0.0
    job_sum = order_window_residence_sum(o, rd.sim_end_s, w0, w1)
    wip_area = stats["wip"]["area"]
    n_wip_end = int(incomplete.sum())
    mass_ok = (len(o) == int(rel.isna().sum()) + n_wip_end + int(shipped.sum()) + int(scrapped.sum())
               and stats["shipped"] == int(shipped.sum()) and stats["scrapped"] == int(scrapped.sum()))
    h = stats["housing"]
    inv_ok = (not h["enabled"]) or (h["initial"] + h["received"] - h["consumed"] == h["on_hand_end"]
                                    and h["min_on_hand"] >= 0)
    agv_blocked = sum(d.get(AgvState.BLOCKED, 0.0) for d in full["agv"].values())
    checks = {
        "time_conservation": {"ok": max_err <= TOL_S, "max_abs_err_s": max_err},
        "wip_area": {"ok": abs(wip_area - job_sum) <= 1e-6 * max(1.0, abs(job_sum)), "area": wip_area,
                     "job_sum": job_sum},
        "mass_balance": {"ok": bool(mass_ok)},
        "inventory_balance": {"ok": bool(inv_ok)},
        "kit_bound": {"ok": stats["kit_max_occupied"] <= stats["kit_slots"], "max": stats["kit_max_occupied"]},
        "agv_never_blocked": {"ok": agv_blocked == 0.0, "blocked_s": agv_blocked},
    }
    checks["all_ok"] = all(c["ok"] for c in checks.values())
    return _clean({"run_id": rd.run_id, "status": rd.status, "factory": factory, "targets": targets,
                   "machines": machines, "stations": stations, "operators": operators, "agvs": agvs,
                   "agv_trips": stats["agv_trips"], "calibration": calibration, "checks": checks})


def loss_decomposition(metrics: Mapping) -> pd.DataFrame:
    """설비 × §6 손실 범주 비율 (관측창 기준)."""
    rows = []
    for mid, d in metrics["machines"].items():
        row = {"machine_id": mid}
        for s, name in LOSS_CATEGORIES.items():
            row[name] = d.get(s, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def top_losses(metrics: Mapping, n: int = 3) -> dict[str, list[tuple[str, float]]]:
    """스테이션별 가공을 뺀 상위 손실 n개."""
    out = {}
    for sid, d in metrics["stations"].items():
        losses = [(LOSS_CATEGORIES[s], d[s]) for s in MachineState.ALL if s != MachineState.PROCESSING]
        out[sid] = sorted(losses, key=lambda x: -x[1])[:n]
    return out


def _clean(obj: Any) -> Any:
    """numpy 스칼라·NaN을 JSON 호환 값으로."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if math.isnan(v) else v
    return obj
