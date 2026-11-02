"""V10: AGV 운반 = 공차 이동 + 상차 + 적재 이동 + 하차, 묶음 적재 4, 목적지 예약, 충전."""

from __future__ import annotations

import pytest

from v2_helpers import explicit_orders, single_line
from factory_sim.engine import run_simulation


def _cfg(n, *, cap=4, x_cap=None):
    cfg = single_line({"X": 10.0}, orders=explicit_orders(["A1"] * n), caps={"X": x_cap})
    cfg["agv"] = {"mode": "fleet", "count": 1, "speed_mps": 1.0, "load_s": 20, "unload_s": 20, "capacity": cap,
                  "home_node": "CHG", "charge": {"after_busy_s": 1.0e9, "duration_s": 1800, "node": "CHG"}}
    # RM→X_IN 50m × 1.2 = 60m, CHG→RM 30m × 1.2 = 36m
    cfg["layout"] = {"detour_factor": 1.2,
                     "nodes": {"RM": [0, 0], "X_IN": [50, 0], "CHG": [-30, 0], "X_1": [55, 0]}}
    return cfg


def test_single_trip_hand_calc():
    r = run_simulation(_cfg(1), seed=1, check_invariants=True, detail="full")
    t = r.tables["agv_trips"]
    assert list(t.t_done) == [pytest.approx(36 + 20 + 60 + 20)]
    assert r.orders[0].terminal_s == pytest.approx(136 + 10)


def test_bundle_of_four():
    r = run_simulation(_cfg(6), seed=1, check_invariants=True, detail="full")
    t = r.tables["agv_trips"]
    assert list(t.n_items) == [4, 2]
    assert list(t.t_done) == [pytest.approx(136.0), pytest.approx(136.0 + 60 + 20 + 60 + 20)]
    s = r.resource_states
    assert s[(s.resource_type == "agv") & (s.state == "BLOCKED")].full_s.sum() == 0.0


def test_destination_reservation_limits_load():
    # X 입력 버퍼 2칸 → 예약 가능한 만큼만 싣는다. 하차가 막히는 일은 없다.
    r = run_simulation(_cfg(4, x_cap=2), seed=1, check_invariants=True, detail="full")
    t = r.tables["agv_trips"]
    assert max(t.n_items) <= 2
    assert all(o.outcome == "shipped" for o in r.orders)


def test_charging_after_busy_time():
    cfg = _cfg(6)
    cfg["agv"]["charge"] = {"after_busy_s": 100.0, "duration_s": 500.0, "node": "CHG"}
    r = run_simulation(cfg, seed=1, detail="full")
    # 1회차 136에 완료(가동 136 ≥ 100) → CHG 이동 96 → 232~732 충전 → 2회차 CHG→RM 36, 상차, 60m, 하차
    iv = r.intervals
    ch = iv[(iv.resource_id == "AGV1") & (iv.state == "CHARGING")]
    assert list(zip(ch.start_s, ch.end_s))[0] == (pytest.approx(232.0), pytest.approx(732.0))
    t = r.tables["agv_trips"]
    assert list(t.t_done) == [pytest.approx(136.0), pytest.approx(732.0 + 36 + 20 + 60 + 20)]
