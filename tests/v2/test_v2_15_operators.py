"""V15: 작업자 숙련·배정·도보, 한 작업자 동시 2설비 금지, 준비만/가공 전체 구분."""

from __future__ import annotations

import pytest

from v2_helpers import explicit_orders, single_line
from factory_sim.config import ConfigError, resolve_config
from factory_sim.engine import run_simulation


def _pool(cfg, skills, nodes):
    cfg["operators"] = {"mode": "pool", "walk_speed_mps": 1.2, "skills": skills}
    cfg["layout"] = {"detour_factor": 1.0, "nodes": nodes}
    return cfg


def _state(r, rid, state):
    s = r.resource_states
    return float(s[(s.resource_id == rid) & (s.state == state)].full_s.sum())


def _iv(r, rid, state):
    iv = r.intervals
    sel = iv[(iv.resource_id == rid) & (iv.state == state)]
    return [(float(a), float(b)) for a, b in zip(sel.start_s, sel.end_s)]


def test_single_skilled_operator_serves_two_machines():
    cfg = single_line({"M": 100.0, "N": 1.0}, orders=explicit_orders(["A1", "A1"]), machines={"M": 2},
                      operator={"M": "full"})
    _pool(cfg, {"OP1": {"home": "M_1", "skills": ["M"]}, "OP2": {"home": "N_1", "skills": ["N"]}},
          {"M_1": [0, 0], "M_2": [10, 0], "N_1": [20, 0]})
    r = run_simulation(cfg, seed=1, check_invariants=True, detail="full")
    walk = 10.0 / 1.2
    assert _iv(r, "M_1", "PROCESSING") == [(0.0, 100.0)]
    wait = _iv(r, "M_2", "WAIT_OPERATOR")
    assert len(wait) == 1 and wait[0][0] == 0.0 and wait[0][1] == pytest.approx(100.0 + walk)
    assert _iv(r, "M_2", "PROCESSING")[0][1] == pytest.approx(200.0 + walk)
    assert _state(r, "OP1", "WALK") == pytest.approx(walk)
    assert _state(r, "OP1", "PROCESS") == pytest.approx(200.0)
    assert _state(r, "OP2", "IDLE") == pytest.approx(r.sim_end_s)  # N은 작업자 불필요 → 배정 없음
    assigned = r.events[r.events.event_type == "OP_ASSIGN"]
    assert set(assigned.resource_id) == {"OP1"}
    # 한 작업자가 두 설비를 동시에 맡지 않는다: M 두 대의 가공 구간이 겹치지 않음.
    p1, p2 = _iv(r, "M_1", "PROCESSING"), _iv(r, "M_2", "PROCESSING")
    assert all(b1 <= a2 or b2 <= a1 for a1, b1 in p1 for a2, b2 in p2)


def test_setup_only_operator_needed_only_for_changeover():
    cfg = single_line({"S": 100.0}, orders=explicit_orders(["A1", "B1"]), operator={"S": "setup_only"},
                      setups={"S": 50.0})
    _pool(cfg, {"OP1": {"home": "S_1", "skills": ["S"]}}, {"S_1": [0, 0]})
    r = run_simulation(cfg, seed=1, check_invariants=True, detail="full")
    assert sorted(o.terminal_s for o in r.orders) == [100.0, 250.0]
    assert _state(r, "OP1", "SETUP") == 50.0
    assert _state(r, "OP1", "PROCESS") == 0.0


def test_full_operator_stays_during_repair():
    cfg = single_line({"M": 70.0}, orders=explicit_orders(["A1"]), operator={"M": "full"})
    _pool(cfg, {"OP1": {"home": "M_1", "skills": ["M"]}}, {"M_1": [0, 0]})
    cfg["stations"]["M"]["failure_class"] = "k"
    cfg["failures"] = {"enabled": True, "distribution": "fixed", "mtbf_scale": 1.0,
                       "classes": {"k": {"mtbf_proc_s": 30.0, "mttr_s": 20.0, "repair_cv": 0.0}}}
    r = run_simulation(cfg, seed=1, detail="full")
    # 30에 고장 → 50 재개 → 누적 가공 60(t=80)에 두 번째 고장 → 100 재개 → 110 완료
    assert r.orders[0].terminal_s == 110.0
    assert _state(r, "OP1", "WAIT_REPAIR") == 40.0
    assert _state(r, "OP1", "PROCESS") == 70.0


def test_unskilled_station_rejected():
    cfg = single_line({"M": 1.0}, orders=explicit_orders(["A1"]), operator={"M": "full"})
    _pool(cfg, {"OP1": {"home": "M_1", "skills": []}}, {"M_1": [0, 0]})
    with pytest.raises(ConfigError):
        resolve_config(cfg)
