"""KPIs computed from the event log, job table and machine intervals (§9).

All values come from the raw event-based records; no time sampling is used.
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

TOL_S = 1e-6
STATES = ("PROCESSING", "SETUP", "BLOCKED", "STARVED", "DOWN")
CSV_ENCODING = "utf-8-sig"


@dataclass
class RunData:
    run_id: str
    status: str
    config: dict
    events: pd.DataFrame
    jobs: pd.DataFrame
    intervals: pd.DataFrame
    sim_end_s: float
    completion_time_s: float | None


def load_run_dir(path: str | Path) -> RunData:
    path = Path(path)
    with open(path / "config_resolved.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    events = pd.read_csv(path / "events.csv", encoding=CSV_ENCODING, float_precision="round_trip",
                         dtype={"job_id": "Int64", "buffer_count": "Int64"})
    jobs = pd.read_csv(path / "jobs.csv", encoding=CSV_ENCODING, float_precision="round_trip")
    intervals = pd.read_csv(path / "machine_intervals.csv", encoding=CSV_ENCODING, float_precision="round_trip",
                            dtype={"job_id": "Int64"})
    sim_end = float(events.loc[events.event_type == "SIM_END", "time_s"].iloc[-1])
    status = meta.get("status")
    completion = float(jobs["terminal_at_s"].max()) if status == "complete" else None
    return RunData(str(meta.get("run_id")), status, cfg, events, jobs, intervals, sim_end, completion)


def _as_run_data(result_or_dir: Any) -> RunData:
    if isinstance(result_or_dir, RunData):
        return result_or_dir
    if isinstance(result_or_dir, (str, Path)):
        return load_run_dir(result_or_dir)
    r = result_or_dir  # engine.RunResult
    return RunData(r.run_id, r.status, r.config, r.events, r.jobs_frame(), r.machine_intervals, r.sim_end_s,
                   r.completion_time_s)


# ---------------------------------------------------------------------------- helpers
def _f(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else float(num) / float(den)


def _last_per_time(times: pd.Series, values: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"time_s": times.to_numpy(dtype=float), "value": values.to_numpy(dtype=float)})
    return df.groupby("time_s", sort=True).last().reset_index()


def wip_step_series(events: pd.DataFrame) -> pd.DataFrame:
    """WIP after each event time (step function). Columns: time_s, value."""
    ev = events.sort_values("event_seq")
    delta = ev["event_type"].map({"JOB_RELEASED": 1, "JOB_SHIPPED": -1, "JOB_SCRAPPED": -1}).fillna(0)
    mask = delta != 0
    return _last_per_time(ev.loc[mask, "time_s"], delta[mask].cumsum())


def cumulative_series(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    ev = events.sort_values("event_seq")
    sel = ev[ev["event_type"] == event_type]
    return _last_per_time(sel["time_s"], pd.Series(np.arange(1, len(sel) + 1), index=sel.index))


def buffer_series(events: pd.DataFrame, buffer_id: str) -> pd.DataFrame:
    ev = events.sort_values("event_seq")
    sel = ev[(ev["buffer_id"] == buffer_id) & ev["buffer_count"].notna()]
    return _last_per_time(sel["time_s"], sel["buffer_count"].astype(float))


def step_area(series: pd.DataFrame, w0: float, w1: float, initial: float = 0.0) -> float:
    """Integral over [w0, w1] of a right-continuous step function (value holds after each time)."""
    area, cur, prev = 0.0, float(initial), float(w0)
    for t, v in zip(series["time_s"].to_numpy(), series["value"].to_numpy()):
        if t <= w0:
            cur = v
            continue
        if t >= w1:
            break
        area += cur * (t - prev)
        prev, cur = t, v
    area += cur * (w1 - prev)
    return float(area)


def step_max(series: pd.DataFrame, w0: float, w1: float, initial: float = 0.0) -> float:
    cur = float(initial)
    vals = []
    for t, v in zip(series["time_s"].to_numpy(), series["value"].to_numpy()):
        if t <= w0:
            cur = v
        elif t <= w1:
            vals.append(v)
    return float(max([cur] + vals))


def state_time_in_window(intervals: pd.DataFrame, w0: float, w1: float) -> dict[str, dict[str, float]]:
    """{machine_id: {state: seconds}} with intervals clipped to [w0, w1]."""
    out: dict[str, dict[str, float]] = {}
    for mid in pd.unique(intervals["machine_id"]):
        out[str(mid)] = {s: 0.0 for s in STATES}
    s = np.clip(intervals["start_s"].to_numpy(dtype=float), w0, w1)
    e = np.clip(intervals["end_s"].to_numpy(dtype=float), w0, w1)
    for mid, state, d in zip(intervals["machine_id"], intervals["state"], e - s):
        out[str(mid)][str(state)] += float(d)
    return out


def job_window_residence_sum(jobs: pd.DataFrame, sim_end: float, w0: float, w1: float) -> float:
    adm = jobs["admitted_at_s"].astype(float)
    end = jobs["terminal_at_s"].astype(float).fillna(sim_end)
    res = (np.minimum(end, w1) - np.maximum(adm, w0)).clip(lower=0)
    return float(res[adm.notna()].sum())


def _dist(values: pd.Series) -> dict[str, Any]:
    v = values.dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return {"n": 0, "mean_s": None, "p50_s": None, "p90_s": None}
    return {"n": int(len(v)), "mean_s": float(v.mean()),
            "p50_s": float(np.percentile(v, 50, method="linear")),
            "p90_s": float(np.percentile(v, 90, method="linear"))}


def _station_roles(cfg: Mapping) -> tuple[str | None, str | None]:
    """(inspect station id, rework input buffer id)."""
    inspect = rework_buf = None
    for sid, st in cfg["line"]["stations"].items():
        rtype = (st.get("route") or {}).get("type")
        if rtype == "inspect" and inspect is None:
            inspect = sid
        if rtype == "rework" and rework_buf is None:
            rework_buf = st["input_buffer"]["id"]
    return inspect, rework_buf


def snapshot_at(events: pd.DataFrame, t: float) -> dict[str, Any]:
    """Line state after all events with time_s <= t (for dashboard replay)."""
    ev = events[events["time_s"] <= t].sort_values("event_seq")
    buffers = {}
    for bid, grp in ev[ev["buffer_id"].notna() & ev["buffer_count"].notna()].groupby("buffer_id"):
        buffers[str(bid)] = int(grp["buffer_count"].iloc[-1])
    machines: dict[str, dict[str, Any]] = {}
    for mid, grp in ev[ev["event_type"] == "STATE_CHANGE"].groupby("machine_id"):
        row = grp.iloc[-1]
        machines[str(mid)] = {"state": row["to_state"],
                              "job_id": None if pd.isna(row["job_id"]) else int(row["job_id"]),
                              "product_type": None if pd.isna(row["product_type"]) else row["product_type"]}
    counts = ev["event_type"].value_counts()
    released = int(counts.get("JOB_RELEASED", 0))
    shipped = int(counts.get("JOB_SHIPPED", 0))
    scrapped = int(counts.get("JOB_SCRAPPED", 0))
    orders = int(counts.get("ORDER_CREATED", 0))
    return {"time_s": float(t), "buffers": buffers, "machines": machines, "orders": orders,
            "unreleased": orders - released, "wip": released - shipped - scrapped,
            "shipped": shipped, "scrapped": scrapped}


# ---------------------------------------------------------------------------- main API
def compute_metrics(result_or_dir: Any, window: tuple[float, float] | None = None) -> dict[str, Any]:
    rd = _as_run_data(result_or_dir)
    cfg = rd.config
    if window is None:
        window = tuple(cfg["run"]["observation_window_s"])
    w0, w1 = float(window[0]), float(window[1])
    span = w1 - w0
    ev, jobs, iv = rd.events, rd.jobs, rd.intervals
    products = list(cfg["orders"]["counts"])
    inspect_station, rework_buffer = _station_roles(cfg)

    adm = jobs["admitted_at_s"].astype(float)
    term = jobs["terminal_at_s"].astype(float)
    outcome = jobs["outcome"]
    shipped_mask = outcome == "shipped"
    scrapped_mask = outcome == "scrapped"
    term_by_end = term.notna() & (term <= w1)
    term_in = term_by_end & (term >= w0)
    # Due-date KPIs count cumulatively up to w1 (consistent with the unshipped breakdown);
    # shipments inside [w0, w1] are used only for the shipping rate.
    good_mask = shipped_mask & term_by_end
    n_orders = int(len(jobs))
    good = int(good_mask.sum())
    good_in_window = int((shipped_mask & term_in).sum())

    by_product = {}
    for p in products:
        pm = jobs["product_type"] == p
        n_p = int(pm.sum())
        s_p = int((good_mask & pm).sum())
        by_product[p] = {"orders": n_p, "shipped": s_p, "otd_rate": _ratio(s_p, n_p)}

    unreleased = int((adm.isna() | (adm > w1)).sum())
    scrapped_shift = int((scrapped_mask & term_by_end).sum())
    wip_mask = adm.notna() & (adm <= w1) & ~term_by_end
    wip_at_end = int(wip_mask.sum())

    wip_series = wip_step_series(ev)
    wip_area = step_area(wip_series, w0, w1)
    buffer_ids = [st["input_buffer"]["id"] for st in cfg["line"]["stations"].values()]
    buffer_avg, buffer_max = {}, {}
    for bid in buffer_ids:
        bs = buffer_series(ev, bid)
        buffer_avg[bid] = step_area(bs, w0, w1) / span
        buffer_max[bid] = int(step_max(bs, w0, w1))

    st_win = state_time_in_window(iv, w0, w1)
    machines = {}
    for mid, d in st_win.items():
        machines[mid] = {
            "processing": d["PROCESSING"] / span, "setup": d["SETUP"] / span, "blocked": d["BLOCKED"] / span,
            "starved": d["STARVED"] / span, "down": d["DOWN"] / span,
            "occupied": (d["PROCESSING"] + d["SETUP"] + d["BLOCKED"] + d["DOWN"]) / span,
        }

    line_time = (term - adm)
    order_age = (term - jobs["order_ready_at_s"].astype(float))
    completed_flow = _dist(line_time[term_in])
    completed_flow = {"basis": "completed_only", **completed_flow}
    inc_age = (w1 - adm[wip_mask])
    incomplete = {"n": wip_at_end,
                  "age_mean_s": float(inc_age.mean()) if len(inc_age) else None,
                  "age_max_s": float(inc_age.max()) if len(inc_age) else None}

    def _quality(t_hi: float | None):
        sel = ev if t_hi is None else ev[(ev["time_s"] >= w0) & (ev["time_s"] <= t_hi)]
        n_pass = int((sel["event_type"] == "INSPECT_PASS").sum())
        n_fail = int((sel["event_type"] == "INSPECT_FAIL").sum())
        n_rw = 0 if rework_buffer is None else int(((sel["event_type"] == "BUFFER_ENTER")
                                                    & (sel["buffer_id"] == rework_buffer)).sum())
        n_insp = n_pass + n_fail
        return _ratio(n_pass, n_insp), _ratio(n_rw, n_insp)

    fpy_shift, rw_shift = _quality(w1)
    shift = {
        "window_s": [w0, w1],
        "orders_total": n_orders,
        "good_shipped": good,
        "good_shipped_in_window": good_in_window,
        "good_rate_per_h": good_in_window / (span / 3600.0),
        "by_product": by_product,
        "overall_otd_rate": _ratio(good, n_orders),
        "unshipped": {"total": n_orders - good, "unreleased": unreleased, "wip": wip_at_end,
                      "scrapped": scrapped_shift},
        "avg_wip": wip_area / span,
        "buffer_avg": buffer_avg,
        "buffer_max": buffer_max,
        "machines": machines,
        "completed_flow_time": completed_flow,
        "incomplete": incomplete,
        "first_pass_yield": fpy_shift,
        "rework_rate": rw_shift,
    }

    complete = rd.status == "complete"
    fpy_all, rw_all = _quality(None)
    good_total = int(shipped_mask.sum())
    scrapped_total = int(scrapped_mask.sum())
    full_by_product = {}
    for p in products:
        pm = jobs["product_type"] == p
        full_by_product[p] = {
            "orders": int(pm.sum()), "shipped": int((shipped_mask & pm).sum()),
            "scrapped": int((scrapped_mask & pm).sum()),
            "line_time_mean_s": _f(line_time[pm].mean()) if complete else None,
            "order_age_mean_s": _f(order_age[pm].mean()) if complete else None,
        }
    full_run = {
        "status": rd.status,
        "completion_time_s": rd.completion_time_s if complete else None,
        "sim_end_s": rd.sim_end_s,
        "good_total": good_total,
        "scrapped_total": scrapped_total,
        "first_pass_yield": fpy_all,
        "rework_rate": rw_all,
        "final_yield": _ratio(good_total, good_total + scrapped_total) if complete else None,
        "line_time": {"all": _dist(line_time), "shipped": _dist(line_time[shipped_mask]),
                      "scrapped": _dist(line_time[scrapped_mask])} if complete else None,
        "order_age": {"all": _dist(order_age), "shipped": _dist(order_age[shipped_mask]),
                      "scrapped": _dist(order_age[scrapped_mask])} if complete else None,
        "by_product": full_by_product,
    }

    # Checks: time conservation, WIP area identity, mass balance.
    errs = []
    for a, b in ((w0, w1), (0.0, rd.sim_end_s)):
        for d in state_time_in_window(iv, a, b).values():
            errs.append(abs(sum(d.values()) - (b - a)))
    max_err = float(max(errs)) if errs else 0.0
    job_sum = job_window_residence_sum(jobs, rd.sim_end_s, w0, w1)
    unreleased_end = int(adm.isna().sum())
    wip_end = int((adm.notna() & term.isna()).sum())
    mass_ok = (n_orders == good + unreleased + wip_at_end + scrapped_shift
               and n_orders == unreleased_end + wip_end + good_total + scrapped_total)
    checks = {
        "time_conservation": {"ok": max_err <= TOL_S, "max_abs_err_s": max_err},
        "wip_area": {"ok": abs(wip_area - job_sum) <= TOL_S, "area": wip_area, "job_sum": job_sum},
        "mass_balance": {"ok": bool(mass_ok)},
    }
    return _clean({"run_id": rd.run_id, "status": rd.status, "shift": shift, "full_run": full_run,
                   "checks": checks})


def _clean(obj: Any) -> Any:
    """Convert numpy scalars / NaN to JSON-safe Python values."""
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
