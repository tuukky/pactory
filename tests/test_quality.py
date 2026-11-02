"""V05: inspection, single rework and scrap paths."""

from __future__ import annotations

import copy

import pytest

from conftest import BASELINE_PATH
from line_sim.config import load_yaml
from line_sim.engine import run_simulation


def _cfg(fail: float, recover: float, counts=None) -> dict:
    cfg = copy.deepcopy(load_yaml(BASELINE_PATH))
    if counts is not None:
        cfg["orders"]["counts"] = counts
    for p in cfg["products"].values():
        p["fail_rate"] = fail
        p["recover_prob"] = recover
    return cfg


SMALL = {"P1": 20, "P2": 12, "P3": 8}


def _entries(result, buffer_id):
    ev = result.events
    return ev[(ev.event_type == "BUFFER_ENTER") & (ev.buffer_id == buffer_id)]


def test_no_defects():
    r = run_simulation(_cfg(0.0, 0.85, SMALL), seed=7, check_invariants=True)
    assert len(_entries(r, "Q_REWORK")) == 0
    assert all(j.outcome.value == "shipped" and not j.reworked for j in r.jobs)


def test_all_fail_all_recover():
    r = run_simulation(_cfg(1.0, 1.0, SMALL), seed=7, check_invariants=True)
    assert r.status == "complete"
    rw = _entries(r, "Q_REWORK")
    assert rw.job_id.is_unique and len(rw) == len(r.jobs)
    for j in r.jobs:
        assert j.first_inspection == "fail" and j.reworked and j.rework_result == "recovered"
        assert j.outcome.value == "shipped"
        assert [op.station_id for op in j.operation_history] == ["PREP", "ASM", "TEST", "REWORK", "PACK"]


def test_all_fail_none_recover():
    r = run_simulation(_cfg(1.0, 0.0, SMALL), seed=7, check_invariants=True)
    assert r.status == "complete"
    assert all(j.outcome.value == "scrapped" for j in r.jobs)
    assert len(_entries(r, "Q_PACK")) == 0
    assert (r.events.event_type == "JOB_SHIPPED").sum() == 0


@pytest.mark.parametrize("seed", [100, 101])
def test_rework_at_most_once_and_ids_kept(seed):
    r = run_simulation(load_yaml(BASELINE_PATH), seed=seed)
    ev = r.events
    rw = _entries(r, "Q_REWORK")
    assert rw.job_id.is_unique
    ids = set(ev.job_id.dropna().astype(int))
    assert ids == {j.id for j in r.jobs}
    for j in r.jobs:
        assert sum(op.station_id == "REWORK" for op in j.operation_history) <= 1


def test_fail_draw_decides_inspection():
    cfg = load_yaml(BASELINE_PATH)
    r = run_simulation(cfg, seed=100)
    for j in r.jobs:
        expected = "fail" if j.draws.fail_u < cfg["products"][j.product_type]["fail_rate"] else "pass"
        assert j.first_inspection == expected
        if j.reworked:
            rec = j.draws.recover_u < cfg["products"][j.product_type]["recover_prob"]
            assert j.rework_result == ("recovered" if rec else "failed")
