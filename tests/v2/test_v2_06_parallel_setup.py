"""V06': 병렬 설비별 품종군 준비 기억, 같은 군은 준비 0, 설비 첫 Job은 준비 0."""

from __future__ import annotations

from v2_helpers import explicit_orders, single_line
from factory_sim.engine import run_simulation


def _cfg(skus, n_asm=2):
    return single_line({"PREP": {"A": 30.0, "B": 35.0, "C": 40.0}, "ASM": {"A": 70.0, "B": 85.0, "C": 110.0}},
                       orders=explicit_orders(skus), machines={"ASM": n_asm}, setups={"ASM": 40.0})


def _iv(r, rid, state):
    iv = r.intervals
    sel = iv[(iv.resource_id == rid) & (iv.state == state)]
    return [(float(a), float(b)) for a, b in zip(sel.start_s, sel.end_s)]


def test_independent_setup_memory():
    r = run_simulation(_cfg(["A1", "C1", "C1"]), seed=1, check_invariants=True, detail="full")
    ships = {o.id: o.terminal_s for o in r.orders}
    assert ships == {1: 100.0, 2: 180.0, 3: 260.0}
    assert _iv(r, "ASM_2", "SETUP") == []
    assert _iv(r, "ASM_1", "SETUP") == [(110.0, 150.0)]
    exits = r.events[(r.events.event_type == "BUFFER_EXIT") & (r.events.buffer_id == "Q_ASM")]
    assert list(zip(exits.order_id, exits.resource_id)) == [(1, "ASM_1"), (2, "ASM_2"), (3, "ASM_1")]


def test_same_group_different_sku_no_setup():
    cfg = _cfg(["A1", "A1"], n_asm=1)
    cfg["products"]["sku_factors"] = [1.0, 1.0]
    cfg["demand"]["explicit"][1]["sku"] = "A2"
    r = run_simulation(cfg, seed=1, detail="full")
    assert _iv(r, "ASM_1", "SETUP") == []
