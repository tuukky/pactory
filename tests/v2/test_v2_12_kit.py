"""V12: 키트 동기화 — 교착 없음, 모터 1대 = 같은 주문 부품 3개, 슬롯 ≤ K, 결품 시 투입 정지."""

from __future__ import annotations

from v2_helpers import motor_config
from factory_sim.engine import run_simulation


def test_kit_pairs_same_order_and_bound():
    cfg = motor_config(60)
    cfg["release"]["kit_slots"] = 5
    cfg["time_scale"] = {"R1": 0.2, "R2": 0.2, "R3": 0.2, "R4": 0.2}  # 로터만 빠름
    r = run_simulation(cfg, seed=1, check_invariants=True, detail="full")
    assert r.status == "complete"
    assert r.stats["kit_max_occupied"] <= 5
    ev = r.events
    deliv = ev[ev.event_type == "KIT_DELIVER"]
    per_order = deliv.groupby("order_id").part.apply(sorted)
    assert all(p == ["HOUSING", "ROTOR", "STATOR"] for p in per_order)
    # 같은 슬롯 번호가 동시에 두 주문에 쓰이지 않는다
    rel = ev[ev.event_type == "ORDER_RELEASED"][["time_s", "order_id", "detail"]]
    take = ev[ev.event_type == "FA_TAKE"][["time_s", "order_id"]]
    spans = rel.merge(take, on="order_id", suffixes=("_rel", "_take"))
    for slot, g in spans.groupby("detail"):
        g = g.sort_values("time_s_rel")
        assert (g.time_s_rel.iloc[1:].to_numpy() >= g.time_s_take.iloc[:-1].to_numpy()).all()


def test_housing_stockout_stops_release():
    cfg = motor_config(12)
    cfg["housing"] = {"enabled": True, "reorder_point": 0, "order_qty": 5, "lead_time_s": 20000.0,
                      "initial_on_hand": 5}
    r = run_simulation(cfg, seed=1, check_invariants=True, detail="full")
    assert r.status == "complete"
    led = r.tables["housing_ledger"]
    assert (led.on_hand >= 0).all()
    released = sorted(o.released_s for o in r.orders)
    receipts = list(led[led.kind == "receive"].time_s)
    assert receipts[:2] == [20000.0, 40000.0]
    assert all(t == 0.0 for t in released[:5])
    assert all(t == 20000.0 for t in released[5:10])
    assert all(t == 40000.0 for t in released[10:])
    assert r.stats["release_blocked_stockout"]["full_area"] > 0
