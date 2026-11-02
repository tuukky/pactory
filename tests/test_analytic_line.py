"""V01: A->B->C deterministic line against hand calculation."""

from __future__ import annotations

import math

import pytest

from conftest import BASELINE_PATH, abc_config
from line_sim.config import nominal_capacity, resolve_config
from line_sim.engine import run_simulation
from line_sim.metrics import compute_metrics


def _ship_times(result) -> list[float]:
    return sorted(j.terminal_at for j in result.jobs if j.outcome.value == "shipped")


def _completed_by(ship_times, t) -> int:
    return sum(1 for s in ship_times if s <= t)


@pytest.fixture(scope="module")
def abc30():
    return run_simulation(abc_config(30.0), seed=1)


@pytest.fixture(scope="module")
def abc25():
    return run_simulation(abc_config(25.0), seed=1)


def test_first_ship_and_interval(abc30):
    ships = _ship_times(abc30)
    assert ships[0] == 55.0
    gaps = {round(b - a, 9) for a, b in zip(ships, ships[1:])}
    assert gaps == {30.0}


@pytest.mark.parametrize("t,expected", [(55, 1), (84, 1), (85, 2), (3600, 119)])
def test_completed_count_formula(abc30, t, expected):
    ships = _ship_times(abc30)
    assert _completed_by(ships, t) == expected
    assert expected == math.floor((t - 55) / 30) + 1


def test_c25_still_bottleneck(abc25):
    ships = _ship_times(abc25)
    assert ships[0] == 50.0
    assert _completed_by(ships, 3600) == 143


def test_window_boundary_includes_exact_28800(abc25):
    # 50 + 25k = 28800 at k = 1150 -> 1151 shipments with "<=".
    # A run(until=28800) style loop would drop the event at exactly 28800 and give 1150.
    ships = _ship_times(abc25)
    assert 28800.0 in ships
    assert abc25.status == "incomplete_at_limit"
    m = compute_metrics(abc25)
    assert m["shift"]["good_shipped"] == 1151


def test_window_c30(abc30):
    m = compute_metrics(abc30)
    assert m["shift"]["good_shipped"] == math.floor((28800 - 55) / 30) + 1 == 959


def test_nominal_capacity_baseline():
    cap = nominal_capacity(resolve_config(BASELINE_PATH))
    assert round(cap["PREP"], 2) == 107.46
    assert round(cap["ASM"], 2) == 87.27
    assert round(cap["TEST"], 2) == 57.60
    assert round(cap["PACK"], 2) == 153.19
    assert round(cap["release"], 2) == 75.00
