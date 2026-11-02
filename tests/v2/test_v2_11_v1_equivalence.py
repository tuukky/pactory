"""V11: 중첩 일치 — 새 계층(작업자·AGV·고장·키트)을 무력화하면 검증된 v1 모델과 같은 결과.

(a) 결정론, (b) v1 확률 처리시간을 그대로 주입, (c) 우회 플래그가 아니라 퇴화 파라미터
(작업자 = 설비 수만큼 전 숙련·이동 0, AGV 다수·이동/상하차 0)로 실제 코드 경로를 통과.
"""

from __future__ import annotations

import copy

import pytest

from line_sim.engine import base_duration as v1_base_duration
from line_sim.engine import run_simulation as v1_run
from line_sim.entities import build_input_plan
from factory_sim.engine import run_simulation as v2_run

PROD_TO_SKU = {"P1": "A1", "P2": "B1", "P3": "C1"}
SETUP_S = 25.0
W = 12
TIMES = {"PREP": {"P1": 30, "P2": 35, "P3": 40}, "ASM": {"P1": 70, "P2": 85, "P3": 110},
         "TEST": {"P1": 50, "P2": 65, "P3": 60}, "PACK": {"P1": 20, "P2": 25, "P3": 30}}
CAPS = {"PREP": 8, "ASM": 4, "TEST": 2, "PACK": 3}
MACHINES = {"PREP": 1, "ASM": 2, "TEST": 1, "PACK": 1}


def v1_config(stochastic: bool) -> dict:
    names = list(TIMES)
    stations = {}
    for i, sid in enumerate(names):
        st = {"machines": MACHINES[sid], "input_buffer": {"id": f"Q_{sid}", "capacity": CAPS[sid]},
              "times_s": dict(TIMES[sid]), "route": {"next": names[i + 1] if i + 1 < len(names) else "SHIP"}}
        if sid == "ASM":
            st["setup_matrix_s"] = {a: {b: (0 if a == b else SETUP_S) for b in PROD_TO_SKU} for a in PROD_TO_SKU}
        stations[sid] = st
    return {
        "schema_version": 1, "model_name": "v11", "time_unit": "s",
        "orders": {"counts": {"P1": 30, "P2": 20, "P3": 14}, "order_ready_at_s": 0, "due_s": 28800},
        "release": {"min_interval_s": 0, "wip_limit": W, "sequence": {"policy": "random_permutation"}},
        "run": {"observation_window_s": [0, 28800], "time_limit_s": 57600},
        "stochastic": {"enabled": stochastic, "distribution": "lognormal", "cv": 0.3},
        "products": {p: {"fail_rate": 0.0, "recover_prob": 1.0} for p in PROD_TO_SKU},
        "line": {"first_station": "PREP", "stations": stations},
        "time_scale": {},
    }


def v2_config(v1cfg: dict, plan) -> dict:
    jobs = {s.id: s for s in plan.jobs}
    orders = [{"id": jid, "sku": PROD_TO_SKU[jobs[jid].product_type], "arrival_s": 0.0} for jid in plan.sequence]
    stations = {}
    for sid in TIMES:
        st = {"machines": MACHINES[sid], "times_s": {PROD_TO_SKU[p][0]: t for p, t in TIMES[sid].items()},
              "operator": "none", "input_buffer": {"capacity": CAPS[sid]}}
        if sid == "ASM":
            st["setup_group_change_s"] = SETUP_S
        stations[sid] = st
    return {
        "schema_version": 2, "model_name": "v11",
        "run": {"warmup_s": 0.0, "observation_s": 28800.0},
        "stochastic": {"enabled": False, "cv": 0.0},
        "products": {"groups": {"A": {"share": 0.5, "fail_rate": 0.0, "recover_prob": 1.0},
                                "B": {"share": 0.3, "fail_rate": 0.0, "recover_prob": 1.0},
                                "C": {"share": 0.2, "fail_rate": 0.0, "recover_prob": 1.0}},
                     "sku_factors": [1.0]},
        "demand": {"mode": "explicit", "rate_per_h": None, "due_offset_s": 28800, "explicit": orders},
        "release": {"kit_slots": W, "slot_release_at": "TERMINAL"},
        "housing": {"enabled": False},
        "routing": {"parts": {"X": {"from": "RM", "path": list(TIMES), "to": "SHIP"}}, "assembly": None,
                    "inspect": None, "ship_to": "SHIP"},
        "dispatch": {"t_max_s": 3600},
        "stations": stations,
        "failures": {"enabled": False, "distribution": "fixed", "mtbf_scale": 1.0, "classes": {}},
        "operators": {"mode": "unlimited", "walk_speed_mps": 1.2, "skills": {}},
        "agv": {"mode": "instant"},
        "layout": {"detour_factor": 1.0, "nodes": {}},
        "time_scale": {},
    }


def degenerate_layers(cfg: dict) -> dict:
    """작업자 풀(설비 수만큼, 전 숙련, 이동 0)과 AGV 함대(이동·상하차 0)를 켠 퇴화 설정."""
    out = copy.deepcopy(cfg)
    nodes = {n: [0.0, 0.0] for n in ("RM", "CHG", "PREP_IN")}
    for sid, st in out["stations"].items():
        st["operator"] = "full"
        for k in range(1, st["machines"] + 1):
            nodes[f"{sid}_{k}"] = [0.0, 0.0]
    n_ops = sum(st["machines"] for st in out["stations"].values())
    out["operators"] = {"mode": "pool", "walk_speed_mps": 1.2,
                        "skills": {f"OP{i:02d}": {"home": "RM", "skills": list(out["stations"])}
                                   for i in range(1, n_ops + 1)}}
    out["agv"] = {"mode": "fleet", "count": 50, "speed_mps": 1.0e9, "load_s": 0, "unload_s": 0, "capacity": 4,
                  "home_node": "CHG", "charge": {"after_busy_s": 0, "duration_s": 0, "node": "CHG"}}
    out["layout"] = {"detour_factor": 1.0, "nodes": nodes}
    return out


def _provider(v1cfg, plan):
    jobs = {s.id: s for s in plan.jobs}

    def provider(order_id, part, sid, visit):
        s = jobs[order_id]
        return v1_base_duration(v1cfg, sid, s.product_type, s.draws.z[sid])

    return provider


def _compare(r1, r2, *, per_machine: bool = True):
    t1 = {j.id: j.terminal_at for j in r1.jobs}
    t2 = {o.id: o.terminal_s for o in r2.orders}
    assert t1.keys() == t2.keys()
    for k in t1:
        assert t1[k] == pytest.approx(t2[k], abs=1e-9), k
    iv = r1.machine_intervals
    v1_tot = iv.groupby(["machine_id", "state"]).duration_s.sum().to_dict()
    st = r2.resource_states[r2.resource_states.resource_type == "machine"]
    v2_tot = {(r, s): v for r, s, v in zip(st.resource_id, st.state, st.full_s)}
    mapping = {"PROCESSING": "PROCESSING", "SETUP": "SETUP", "STARVED": "STARVED", "BLOCKED": "BLOCKED_DOWNSTREAM"}
    key = (lambda m: m) if per_machine else (lambda m: m.split("_")[0])
    agg1, agg2 = {}, {}
    for (mid, s), v in v1_tot.items():
        agg1[(key(mid), mapping[s])] = agg1.get((key(mid), mapping[s]), 0.0) + v
    for (mid, s), v in v2_tot.items():
        agg2[(key(mid), s)] = agg2.get((key(mid), s), 0.0) + v
    for k, v in agg1.items():
        assert agg2.get(k, 0.0) == pytest.approx(v, abs=1e-6), k
    for s in ("WAIT_OPERATOR", "BLOCKED_TRANSPORT", "DOWN"):
        assert sum(v for (m, st_), v in agg2.items() if st_ == s) == 0.0


def test_v11a_deterministic_matches_v1():
    v1cfg = v1_config(stochastic=False)
    plan = build_input_plan(v1cfg, 3)
    r1 = v1_run(v1cfg, 3, plan)
    r2 = v2_run(v2_config(v1cfg, plan), 3, check_invariants=True)
    assert r1.status == "complete" and r2.status == "complete"
    _compare(r1, r2)
    # 설비가 가져간 Job 순서도 같다.
    e1 = r1.events[r1.events.event_type == "BUFFER_EXIT"]
    r2f = v2_run(v2_config(v1cfg, plan), 3, detail="full")
    e2 = r2f.events[r2f.events.event_type == "BUFFER_EXIT"]
    assert list(zip(e1.job_id, e1.machine_id)) == list(zip(e2.order_id, e2.resource_id))


@pytest.mark.parametrize("seed", [1, 2])
def test_v11b_stochastic_durations_injected(seed):
    v1cfg = v1_config(stochastic=True)
    plan = build_input_plan(v1cfg, seed)
    r1 = v1_run(v1cfg, seed, plan)
    r2 = v2_run(v2_config(v1cfg, plan), seed, duration_provider=_provider(v1cfg, plan), check_invariants=True)
    _compare(r1, r2)


@pytest.mark.parametrize("stochastic,seed", [(False, 3), (True, 1)])
def test_v11c_degenerate_layers(stochastic, seed):
    v1cfg = v1_config(stochastic=stochastic)
    plan = build_input_plan(v1cfg, seed)
    r1 = v1_run(v1cfg, seed, plan)
    cfg = degenerate_layers(v2_config(v1cfg, plan))
    r2 = v2_run(cfg, seed, duration_provider=_provider(v1cfg, plan), check_invariants=True)
    _compare(r1, r2, per_machine=False)
    ops = r2.resource_states[r2.resource_states.resource_type == "operator"]
    assert ops[ops.state == "WALK"].full_s.sum() == 0.0
