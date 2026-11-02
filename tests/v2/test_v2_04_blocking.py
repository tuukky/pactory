"""V04': 유한 버퍼와 가공 후 막힘(blocking-after-service) — v1 V04와 같은 손계산."""

from __future__ import annotations

from v2_helpers import explicit_orders, single_line
from factory_sim.engine import run_simulation


def _cfg(cap):
    return single_line({"U": 5.0, "D": 20.0}, orders=explicit_orders(["A1"] * 5), caps={"U": None, "D": cap})


def _intervals(r, rid, state):
    iv = r.intervals
    sel = iv[(iv.resource_id == rid) & (iv.state == state)]
    return [(float(a), float(b)) for a, b in zip(sel.start_s, sel.end_s)]


def test_blocking_capacity_1():
    r = run_simulation(_cfg(1), seed=1, check_invariants=True, detail="full")
    assert sorted(o.terminal_s for o in r.orders) == [25.0, 45.0, 65.0, 85.0, 105.0]
    assert _intervals(r, "U_1", "BLOCKED_DOWNSTREAM") == [(15.0, 25.0), (30.0, 45.0), (50.0, 65.0)]
    ev = r.events
    assert ev[ev.buffer_id == "Q_D"].buffer_count.max() == 1


def test_blocking_capacity_2():
    r = run_simulation(_cfg(2), seed=1, check_invariants=True, detail="full")
    assert _intervals(r, "U_1", "BLOCKED_DOWNSTREAM") == [(20.0, 25.0), (30.0, 45.0)]
