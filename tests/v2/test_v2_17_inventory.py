"""V17: 하우징 (s, Q) — 재고 수지, 재고 ≥ 0, 재고위치 기준 발주 시점, 리드타임 입고."""

from __future__ import annotations

import simpy

from v2_helpers import rev_a, short
from factory_sim.engine import run_simulation
from factory_sim.inventory import HousingInventory
from factory_sim.logging import EventLogger


def test_reorder_on_inventory_position():
    env = simpy.Environment()
    cfg = {"housing": {"enabled": True, "reorder_point": 3, "order_qty": 4, "lead_time_s": 10.0,
                       "initial_on_hand": 5}}
    inv = HousingInventory(env, cfg, EventLogger("t", False), (0, 100))
    inv.consume()          # 보유 4, 위치 4
    assert inv.on_order == 0
    inv.consume()          # 보유 3, 위치 3 ≤ s → Q 발주, 위치 7
    assert inv.on_order == 4 and inv.position() == 7
    inv.consume()          # 보유 2, 위치 6 → 추가 발주 없음
    assert inv.on_order == 4
    env.run(until=10.5)
    assert inv.on_hand == 6 and inv.on_order == 0 and inv.received == 4
    assert inv.initial + inv.received - inv.consumed == inv.on_hand


def test_rev_a_ledger_balance():
    r = run_simulation(short(rev_a(), 8, 40), seed=5, detail="full")
    led = r.tables["housing_ledger"]
    h = r.stats["housing"]
    assert (led.on_hand >= 0).all()
    assert h["initial"] + h["received"] - h["consumed"] == h["on_hand_end"]
    orders = led[led.kind == "order"]
    assert len(orders) >= 1
    s = r.config["housing"]["reorder_point"]
    q = r.config["housing"]["order_qty"]
    # 발주 직전 재고위치 ≤ s
    assert ((orders.position - q) <= s).all()
    rec = led[led.kind == "receive"]
    lead = r.config["housing"]["lead_time_s"]
    for t_rec, t_ord in zip(rec.time_s, orders.time_s):
        assert abs(t_rec - (t_ord + lead)) < 1e-6
