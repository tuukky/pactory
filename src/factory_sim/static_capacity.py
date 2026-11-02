"""정적 부하 계산 (시뮬레이션 전 손계산, PLAN_v2 §5). 결과는 가설이지 결론이 아니다.

- 공정 능력(대/h): 순수 → +준비(FIFO 군 전환 확률 1−Σp²) → +고장(가공 × (1 + MTTR/MTBF))
  → 실효(오븐만 만재 대신 입력 버퍼 제약 배치·형성 대기 반영, 나머지는 +고장과 같음). 순위는 실효 기준.
- 이용률 @ 수요, 예상 병목 순위
- 작업자 부하, AGV 부하(적재 1개·4개), 하우징 (s, Q) 공급 상한, 키트 K 상한(Little 근사)

Usage:
    python -m factory_sim.static_capacity --config configs/v2/rev_A.yaml
"""

from __future__ import annotations

import argparse
import sys
from typing import Mapping

import pandas as pd

from .config import ConfigError, resolve_config
from .layout import Layout
from .routing import ProcessGraph


def _shares(cfg: Mapping) -> dict[str, float]:
    return {g: float(gc["share"]) for g, gc in cfg["products"]["groups"].items()}


def mean_station_time(cfg: Mapping, sid: str) -> float:
    """품종 가중 평균 처리시간(초). SKU 계수 평균과 time_scale 반영."""
    st = cfg["stations"][sid]
    f = cfg["products"]["sku_factors"]
    fmean = sum(f) / len(f)
    scale = float((cfg.get("time_scale") or {}).get(sid, 1.0))
    return sum(p * float(st["times_s"][g]) for g, p in _shares(cfg).items()) * fmean * scale


def group_change_prob(cfg: Mapping) -> float:
    return 1.0 - sum(p * p for p in _shares(cfg).values())


def oven_effective(cfg: Mapping, sid: str, demand: float) -> dict[str, float]:
    """입력 버퍼 제약을 반영한 오븐 실효 배치 크기와 능력(형성 시작 기준 최대 대기, C1).

    가동 중에는 입력 버퍼 + 상류 설비(막힌 채 1개씩)까지만 쌓이고, 하역 직후 일괄 적재된다.
    그 뒤 최대 대기 동안 수요율로 더 들어온다. 배치 = min(만재, 이월 + 형성 중 도착).
    주기 = 가공 + 형성 시간(만재가 먼저 되면 그보다 짧다).
    """
    st = cfg["stations"][sid]
    b = st["batch"]
    full = int(b["max_size"])
    proc = float(b["process_s"]) * float((cfg.get("time_scale") or {}).get(sid, 1.0))
    wait = float(b["max_wait_s"])
    n = int(st["machines"])
    rate = demand / 3600.0 / n  # 오븐 여러 대는 번갈아 형성 → 대당 도착률
    if rate <= 0:
        return {"batch": float(full), "cycle_s": proc, "cap_per_h": n * full * 3600.0 / proc, "carried": 0.0}
    in_cap = (st.get("input_buffer") or {}).get("capacity")
    upstream = 0
    for pc in cfg["routing"]["parts"].values():
        if sid in pc["path"]:
            i = pc["path"].index(sid)
            upstream = int(cfg["stations"][pc["path"][i - 1]]["machines"]) if i > 0 else 0
    arrived = rate * proc
    carried = arrived if in_cap is None else min(float(in_cap + upstream), arrived)
    if carried >= full:
        batch, forming = float(full), 0.0
    else:
        forming = min(wait, (full - carried) / rate)
        batch = min(float(full), carried + rate * forming)
    cycle = proc + forming
    return {"batch": batch, "cycle_s": cycle, "cap_per_h": n * batch * 3600.0 / cycle, "carried": carried}


def compute(cfg: Mapping) -> dict[str, pd.DataFrame | dict]:
    demand = float(cfg["demand"].get("rate_per_h") or 0.0)
    graph = ProcessGraph.from_cfg(cfg)
    shares = _shares(cfg)
    groups = cfg["products"]["groups"]
    fail = sum(shares[g] * float(gc["fail_rate"]) for g, gc in groups.items())
    rework_on = bool(graph.assembly) and graph.max_rework >= 1
    # 재검사 합격 비율(주문당) = Σ 비율 × 불합격률 × 회복률
    rec = sum(shares[g] * float(gc["fail_rate"]) * float(gc["recover_prob"]) for g, gc in groups.items())
    rw_visits = fail if rework_on else 0.0
    p_change = group_change_prob(cfg)
    fcfg = cfg["failures"]
    classes = fcfg.get("classes") or {}

    rows = []
    for sid, st in cfg["stations"].items():
        n = int(st["machines"])
        visits = 1.0
        if sid == graph.inspect:
            visits = 1.0 + rw_visits
        elif sid == graph.rework:
            visits = rw_visits
        oven = st.get("kind") == "batch_oven"
        if oven:
            b = st["batch"]
            t_unit = float(b["process_s"]) * float((cfg.get("time_scale") or {}).get(sid, 1.0)) / int(b["max_size"])
            setup_unit = 0.0
        else:
            t_unit = mean_station_time(cfg, sid)
            setup_unit = p_change * float(st.get("setup_group_change_s", 0) or 0)
        fc = classes.get(st.get("failure_class")) if fcfg["enabled"] else None
        fail_factor = 1.0 if fc is None else 1.0 + float(fc["mttr_s"]) / (float(fc["mtbf_proc_s"]) *
                                                                           float(fcfg["mtbf_scale"]))

        def cap(per_unit: float) -> float:
            load = per_unit * visits
            return n * 3600.0 / load if load > 0 else float("inf")

        cap_pure = cap(t_unit)
        cap_setup = cap(t_unit + setup_unit)
        cap_fail = cap(t_unit * fail_factor + setup_unit)
        cap_eff = cap_fail
        note = ""
        if oven:
            # 만재 가정 능력과 입력 버퍼 제약 실효 능력을 함께 낸다(C1).
            eff = oven_effective(cfg, sid, demand)
            cap_eff = eff["cap_per_h"] / fail_factor
            full = int(st["batch"]["max_size"])
            wait = float(st["batch"]["max_wait_s"])
            note = (f"cap_pure·cap_fail은 만재 {full} 가정. 실효는 입력 버퍼 제약 배치 {eff['batch']:.1f}, "
                    f"주기 {eff['cycle_s']:.0f}s(형성 대기 상한 {wait:.0f}s)")
        rows.append({"station": sid, "machines": n, "mean_time_s": t_unit, "visits_per_order": visits,
                     "setup_per_unit_s": setup_unit, "cap_pure_per_h": cap_pure, "cap_setup_per_h": cap_setup,
                     "cap_fail_per_h": cap_fail, "cap_effective_per_h": cap_eff,
                     "util_at_demand": demand / cap_eff if cap_eff > 0 else float("inf"), "note": note})
    stations = pd.DataFrame(rows).sort_values("util_at_demand", ascending=False).reset_index(drop=True)
    stations["rank"] = range(1, len(stations) + 1)

    # 작업자 부하
    op_s = 0.0
    for sid, st in cfg["stations"].items():
        mode = st.get("operator", "none")
        if mode == "none":
            continue
        visits = rw_visits if sid == graph.rework else 1.0
        if visits <= 0:
            continue
        setup_unit = p_change * float(st.get("setup_group_change_s", 0) or 0)
        op_s += visits * (setup_unit + (mean_station_time(cfg, sid) if mode == "full" else 0.0))
    n_ops = len((cfg["operators"].get("skills") or {}))
    operators = {"operator_s_per_order": op_s, "busy_operators_at_demand": op_s * demand / 3600.0,
                 "operators": n_ops, "util_at_demand": (op_s * demand / 3600.0) / n_ops if n_ops else None}

    # AGV 부하: 왕복 근사(공차 이동 ≈ 적재 이동)
    agv = cfg["agv"]
    lanes = []
    if agv["mode"] == "fleet":
        lay = Layout.from_cfg(cfg)
        for p, pc in graph.parts.items():
            src = pc["from"]
            dst = f"{pc['path'][0]}_IN" if pc["path"] else "KIT"
            lanes.append((f"{src}→{dst}", lay.distance(src, dst), 1.0))
            if pc["path"] and pc["to"] == "KIT":
                lanes.append((f"{pc['path'][-1]}_OUT→KIT", lay.distance(f"{pc['path'][-1]}_OUT", "KIT"), 1.0))
        if graph.assembly:
            a, e, r, k = graph.assembly, graph.inspect, graph.rework, graph.pass_to
            to_pack = (1.0 - fail) + (rec if rework_on else 0.0)  # 최초 합격 + 재검사 합격
            lanes += [(f"{a}→{e}", lay.distance(f"{a}_OUT", f"{e}_IN"), 1.0),
                      (f"{e}→{k}", lay.distance(f"{e}_OUT", f"{k}_IN"), to_pack),
                      (f"{e}→{r}", lay.distance(f"{e}_OUT", f"{r}_IN"), rw_visits),
                      (f"{r}→{e}", lay.distance(f"{r}_OUT", f"{e}_IN"), rw_visits)]
            if graph.ship_to == "DOCK":
                lanes.append((f"{k}→DOCK", lay.distance(f"{k}_OUT", "DOCK"), to_pack))
    speed, ls, us = float(agv.get("speed_mps", 1)), float(agv.get("load_s", 0)), float(agv.get("unload_s", 0))
    ch = agv.get("charge") or {}
    avail = 1.0
    if ch.get("duration_s"):
        avail = float(ch["after_busy_s"]) / (float(ch["after_busy_s"]) + float(ch["duration_s"]))
    agv_rows = []
    for name, d, freq in lanes:
        trip = 2.0 * d / speed + ls + us
        agv_rows.append({"lane": name, "distance_m": d, "trips_per_order": freq, "trip_s": trip})
    lanes_df = pd.DataFrame(agv_rows, columns=["lane", "distance_m", "trips_per_order", "trip_s"])
    agv_out = {"lanes": lanes_df, "count": int(agv.get("count", 0) or 0), "availability": avail}
    for batch in (1, int(agv.get("capacity", 1) or 1)):
        s_per_order = float((lanes_df["trip_s"] * lanes_df["trips_per_order"]).sum()) / batch if len(lanes_df) else 0
        agv_out[f"agv_needed_batch{batch}"] = s_per_order * demand / 3600.0 / avail

    # 하우징 (s, Q) 공급 상한 (재고위치 기준, 주문 1건씩 미착)
    h = cfg.get("housing") or {}
    housing = {"enabled": bool(h.get("enabled"))}
    if h.get("enabled") and demand > 0:
        s, q, lead_h = float(h["reorder_point"]), float(h["order_qty"]), float(h["lead_time_s"]) / 3600.0
        if demand * lead_h <= s:
            supply = demand
        else:
            supply = min(demand, q / (lead_h + max(q - s, 0.0) / demand))
        housing |= {"demand_during_lead": demand * lead_h, "reorder_point": s, "supply_cap_per_h": supply}

    # 키트 K 상한 (Little): 투입 → FA 체류 하한 ≈ 긴 서브라인 순수 가공합 + 오븐 + 운반 2회
    kit = {}
    if graph.assembly:
        path_t = []
        for p, pc in graph.parts.items():
            t = 0.0
            for sid in pc["path"]:
                st = cfg["stations"][sid]
                t += float(st["batch"]["process_s"]) if st.get("kind") == "batch_oven" else mean_station_time(cfg, sid)
            path_t.append(t)
        trip = float(lanes_df["trip_s"].mean()) if len(lanes_df) else 0.0
        resid = max(path_t) + 2 * trip
        k = int(cfg["release"]["kit_slots"])
        kit = {"kit_slots": k, "min_kit_residence_s": resid, "conwip_cap_per_h": k * 3600.0 / resid}

    return {"stations": stations, "operators": operators, "agv": agv_out, "housing": housing, "kit": kit,
            "demand_per_h": demand, "group_change_prob": p_change}


def to_markdown(res: Mapping) -> str:
    st = res["stations"].copy()
    cols = ["rank", "station", "machines", "mean_time_s", "visits_per_order", "setup_per_unit_s", "cap_pure_per_h",
            "cap_setup_per_h", "cap_fail_per_h", "cap_effective_per_h", "util_at_demand"]
    lines = [f"수요 {res['demand_per_h']:.1f}대/h, 군 전환 확률(FIFO) {res['group_change_prob']:.3f}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in st[cols].itertuples(index=False):
        vals = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(vals) + " |")
    for r in st.itertuples(index=False):
        if r.note:
            lines.append(f"- {r.station}: {r.note}")
    o = res["operators"]
    lines += ["", f"작업자: 주문당 {o['operator_s_per_order']:.0f}s → 수요에서 {o['busy_operators_at_demand']:.2f}명 상시 점유"
                  f" / {o['operators']}명 (이용률 {o['util_at_demand'] or 0:.2f})"]
    a = res["agv"]
    needed = ", ".join(f"{k.split('batch')[1]}개 적재 {v:.2f}대" for k, v in a.items() if k.startswith("agv_needed"))
    lines += [f"AGV: 필요 대수(충전 가용률 {a['availability']:.3f} 반영) {needed} / 보유 {a['count']}대"]
    h = res["housing"]
    if h.get("enabled"):
        lines += [f"하우징: 리드타임 중 수요 {h['demand_during_lead']:.0f} vs s={h['reorder_point']:.0f} → "
                  f"공급 상한 {h['supply_cap_per_h']:.1f}대/h"]
    k = res["kit"]
    if k:
        lines += [f"키트: K={k['kit_slots']}, 투입→FA 체류 하한 {k['min_kit_residence_s'] / 60:.0f}분 → "
                  f"CONWIP 상한 {k['conwip_cap_per_h']:.1f}대/h"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2 정적 부하 계산")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        cfg = resolve_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(to_markdown(compute(cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
