"""V05'·V14: EOL 판정·재작업·같은 EOL 재검사 재순환, 예약 슬롯, 교착 없음."""

from __future__ import annotations

import pytest
import simpy

from v2_helpers import motor_config
from factory_sim.engine import run_simulation
from factory_sim.entities import Buffer, Machine, Order, PartJob
from factory_sim.logging import EventLogger
from factory_sim.randomness import fail_u, recover_u


def _all_groups(cfg, fail, recover):
    for g in cfg["products"]["groups"].values():
        g["fail_rate"], g["recover_prob"] = fail, recover
    return cfg


def _visits(r, sid):
    ev = r.events
    return ev[(ev.event_type == "PROCESS_START") & (ev.station_id == sid)].groupby("order_id").size()


@pytest.mark.parametrize("recover,outcome", [(1.0, "shipped"), (0.0, "scrapped")])
def test_v14_every_unit_rechecked_once(recover, outcome):
    cfg = _all_groups(motor_config(40), 1.0, recover)
    # 작은 버퍼·AGV 1대로 순환 막힘을 유도한다.
    cfg["stations"]["EOL"]["input_buffer"] = {"capacity": 3, "reserved_recheck": 1}
    for sid in ("FA", "EOL", "RW", "PK"):
        cfg["stations"][sid]["output_buffer"] = {"capacity": 1}
    cfg["agv"]["count"] = 1
    r = run_simulation(cfg, seed=1, check_invariants=True, detail="full")
    assert r.status == "complete"
    assert {o.outcome for o in r.orders} == {outcome}
    assert all(o.eol_visits == 2 and o.reworked for o in r.orders)
    assert set(_visits(r, "EOL")) == {2}
    assert set(_visits(r, "RW")) == {1}
    if outcome == "scrapped":
        assert len(_visits(r, "PK")) == 0


def test_v05_inspection_uses_keyed_draws():
    cfg = motor_config(60, deterministic=False)
    r = run_simulation(cfg, seed=4, check_invariants=True)
    g = cfg["products"]["groups"]
    for o in r.orders:
        assert o.outcome != "incomplete"
        failed = fail_u(4, o.id) < g[o.group]["fail_rate"]
        assert o.first_inspection == ("fail" if failed else "pass")
        if failed:
            rec = recover_u(4, o.id) < g[o.group]["recover_prob"]
            assert o.recheck == ("pass" if rec else "fail")
            assert o.outcome == ("shipped" if rec else "scrapped")
        else:
            assert o.outcome == "shipped" and o.eol_visits == 1


def _job(i):
    return PartJob(Order(i, "A1", "A", 0.0, 0.0), "MOTOR")


def test_reserved_slot_semantics():
    env = simpy.Environment()
    b = Buffer(env, "Q_EOL", "EOL", 3, EventLogger("t", False), reserved=1)
    # 신규품은 general 2칸만
    assert b.request_put(_job(1), None).triggered
    assert b.request_put(_job(2), None).triggered
    blocked = b.request_put(_job(3), None)
    assert not blocked.triggered
    assert b.free("general") == 0 and b.free("recheck") == 1
    # 재검사품은 예약 슬롯으로 들어간다
    slot = b.reserve("recheck")
    assert slot == "reserved"
    b.deliver(_job(4), slot)
    assert b.n_res == 1 and b.free("recheck") == 0
    m = Machine(env, "EOL_1", "EOL", 0, (0, 1), EventLogger("t", False), False)
    # 예약 슬롯이 비어도 막힌 신규 putter는 그 칸을 쓸 수 없다
    assert b.request_get(m, selector=lambda items, mach, now: 2).value.id == 4
    assert not blocked.triggered and b.n_res == 0
    # general 슬롯이 비면 막힌 신규 putter가 들어간다(FIFO)
    assert b.request_get(m).value.id == 1
    assert blocked.triggered
    assert [j.id for j in b.items] == [2, 3]


def test_v14_stress_high_conwip_no_deadlock():
    from v2_helpers import rev_a, short
    from factory_sim.config import apply_overrides
    cfg = apply_overrides(short(rev_a(), 2, 14), {
        "release.kit_slots": 200, "agv.count": 1,
        "products.groups.A.fail_rate": 0.5, "products.groups.B.fail_rate": 0.5, "products.groups.C.fail_rate": 0.5,
        "stations.EOL.input_buffer": {"capacity": 3, "reserved_recheck": 1},
        "stations.FA.output_buffer": {"capacity": 1}, "stations.EOL.output_buffer": {"capacity": 1},
        "stations.RW.output_buffer": {"capacity": 1}, "housing.initial_on_hand": 5000})
    r = run_simulation(cfg, seed=3, check_invariants=True, deadlock_window_s=14400.0)
    assert r.stats["shipped"] > 0
    assert any(o.recheck is not None for o in r.orders)
