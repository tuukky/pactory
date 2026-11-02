"""V08·V09: 가공시간 기준 고장 시계, 남은 시간 보존, 즉시 수리(정비 인력 없음)."""

from __future__ import annotations

import pytest

from v2_helpers import explicit_orders, single_line
from factory_sim.engine import run_simulation
from factory_sim.failures import FailureClock


def _cfg(orders, *, machines=1, mtbf=100.0, mttr=50.0, t=70.0):
    cfg = single_line({"M": t}, orders=orders, machines={"M": machines})
    cfg["stations"]["M"]["failure_class"] = "k"
    cfg["failures"] = {"enabled": True, "distribution": "fixed", "mtbf_scale": 1.0,
                       "classes": {"k": {"mtbf_proc_s": mtbf, "mttr_s": mttr, "repair_cv": 0.0}}}
    return cfg


def _iv(r, rid, state):
    iv = r.intervals
    sel = iv[(iv.resource_id == rid) & (iv.state == state)]
    return [(float(a), float(b)) for a, b in zip(sel.start_s, sel.end_s)]


def test_v08_failure_and_resume_hand_calc():
    r = run_simulation(_cfg(explicit_orders(["A1"] * 4)), seed=1, check_invariants=True, detail="full")
    # 1: 0~70(시계 30 남음) / 2: 70 시작, 100 고장~150, 남은 40 → 190 (새 시계 100 중 40 사용)
    # 3: 190 시작, 남은 시계 60 → 250 고장~300, 남은 10 → 310 / 4: 310~380 (시계 100 → 30 남음)
    assert sorted(o.terminal_s for o in r.orders) == [70.0, 190.0, 310.0, 380.0]
    assert _iv(r, "M_1", "DOWN") == [(100.0, 150.0), (250.0, 300.0)]


def test_v08_clock_runs_only_while_processing():
    # 1번 가공 0~70(시계 30 남음) → 70~500 굶주림(시계 정지) → 2번 시작 500, 530에 고장.
    r = run_simulation(_cfg(explicit_orders(["A1", "A1"], [0.0, 500.0])), seed=1, detail="full")
    assert _iv(r, "M_1", "DOWN") == [(530.0, 580.0)]
    assert sorted(o.terminal_s for o in r.orders) == [70.0, 620.0]


def test_v09_simultaneous_failures_repair_independently():
    r = run_simulation(_cfg(explicit_orders(["A1", "A1"]), machines=2, mtbf=50.0, mttr=100.0, t=60.0), seed=1,
                       check_invariants=True, detail="full")
    assert _iv(r, "M_1", "DOWN") == [(50.0, 150.0)]
    assert _iv(r, "M_2", "DOWN") == [(50.0, 150.0)]
    assert sorted(o.terminal_s for o in r.orders) == [160.0, 160.0]


def test_failure_boundary_equal_remaining_fails():
    clock = FailureClock(1, "M_1", {"mtbf_proc_s": 70.0, "mttr_s": 5.0, "repair_cv": 0.0}, 1.0, True, True)
    assert clock.ttf_left() == 70.0  # 남은 가공 70초와 같으면 끝 시점에 고장(D19)
    r = run_simulation(_cfg(explicit_orders(["A1"]), mtbf=70.0, mttr=5.0), seed=1, detail="full")
    assert _iv(r, "M_1", "DOWN") == [(70.0, 75.0)]
    assert r.orders[0].terminal_s == 75.0


def test_stochastic_failure_draws_match_means():
    clock = FailureClock(7, "R1_1", {"mtbf_proc_s": 1000.0, "mttr_s": 100.0, "repair_cv": 0.5}, 1.0, True)
    reps = [clock.next_repair() for _ in range(4000)]
    assert sum(reps) / len(reps) == pytest.approx(100.0, rel=0.05)
    ttfs = [d[1] for d in clock.draws_log]
    assert sum(ttfs) / len(ttfs) == pytest.approx(1000.0, rel=0.06)
