"""V01': 결정론 직선 라인 처리량·완료시각 손계산."""

from __future__ import annotations

from v2_helpers import explicit_orders, single_line
from factory_sim.engine import run_simulation


def _ships(r):
    return sorted(o.terminal_s for o in r.orders if o.outcome == "shipped")


def test_bottleneck_paces_line():
    n = 20
    cfg = single_line({"A": 10.0, "B": 15.0, "C": 30.0}, orders=explicit_orders(["A1"] * n))
    r = run_simulation(cfg, seed=1, check_invariants=True)
    assert r.status == "complete"
    assert _ships(r) == [25.0 + 30.0 * k for k in range(1, n + 1)]


def test_group_times_and_time_scale():
    cfg = single_line({"A": {"A": 10.0, "B": 20.0}}, orders=explicit_orders(["A1", "B1"]), groups=("A", "B"))
    cfg["time_scale"] = {"A": 0.5}
    r = run_simulation(cfg, seed=1)
    assert _ships(r) == [5.0, 15.0]
