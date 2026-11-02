"""V19: 디스패칭 FIFO/EDD/FAMILY 선택 순서 손계산, T_max 초과 시 강제 선택."""

from __future__ import annotations

import pytest

from v2_helpers import explicit_orders, single_line
from factory_sim.dispatch import Candidate, select
from factory_sim.engine import run_simulation

# (큐 진입, 순번, 납기, 군)
CANDS = [Candidate(10.0, 0, 900.0, "A"), Candidate(20.0, 1, 500.0, "B"), Candidate(30.0, 2, 700.0, "B"),
         Candidate(5.0, 3, 950.0, "C")]


def test_fifo():
    assert select("FIFO", CANDS, "B", 100.0, 3600.0) == 3


def test_edd():
    assert select("EDD", CANDS, "A", 100.0, 3600.0) == 1


def test_family_prefers_same_group_fifo():
    assert select("FAMILY", CANDS, "B", 100.0, 3600.0) == 1
    assert select("FAMILY", CANDS, "A", 100.0, 3600.0) == 0
    assert select("FAMILY", CANDS, "D", 100.0, 3600.0) == 3  # 같은 군 없음 → 전체 FIFO
    assert select("FAMILY", CANDS, None, 100.0, 3600.0) == 3


def test_family_t_max_override():
    # 대기 = now − 큐 진입. now=60, T_max=50 → 진입 5(대기 55), 10(대기 50)이 초과 → 그중 FIFO = 진입 5
    assert select("FAMILY", CANDS, "B", 60.0, 50.0) == 3
    # now=54.9: 진입 5만 대기 49.9 < 50 → 초과 없음 → 같은 군 B 중 FIFO = 1
    assert select("FAMILY", CANDS, "B", 54.9, 50.0) == 1
    # now=55: 진입 5가 정확히 50 → 초과(≥)로 강제 선택
    assert select("FAMILY", CANDS, "B", 55.0, 50.0) == 3
    # 초과 후보가 군이 달라도 우선한다
    assert select("FAMILY", [CANDS[0], CANDS[1]], "B", 60.0, 50.0) == 0
    assert select("FAMILY", [CANDS[0], CANDS[1]], "B", 59.0, 50.0) == 1


def test_unknown_rule():
    with pytest.raises(ValueError):
        select("LIFO", CANDS, None, 0.0, 1.0)


def _line(rule, skus):
    cfg = single_line({"P": 10.0, "M": 100.0}, orders=explicit_orders(skus), setups={"M": 50.0})
    cfg["stations"]["M"]["dispatch"] = rule
    return cfg


def test_family_in_engine_reduces_setups():
    skus = ["A1", "B1", "A1", "B1", "A1", "B1"]
    fifo = run_simulation(_line("FIFO", skus), seed=1, detail="full")
    fam = run_simulation(_line("FAMILY", skus), seed=1, detail="full")
    n_setup = lambda r: int((r.events.event_type == "SETUP_START").sum())  # noqa: E731
    assert n_setup(fam) < n_setup(fifo)
    order = fam.events[(fam.events.event_type == "BUFFER_EXIT") & (fam.events.buffer_id == "Q_M")].order_id
    # M: 1(A) 0~ / 대기열에서 A 우선 → 3, 5 다음 B
    assert list(order) == [1, 3, 5, 2, 4, 6]
