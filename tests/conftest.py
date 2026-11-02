"""Shared configuration factories for the verification tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from line_sim.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "configs" / "baseline.yaml"
EXPERIMENTS_PATH = ROOT / "configs" / "experiments.yaml"


def serial_config(stage_times: dict[str, float], capacities: dict[str, int | None], n_orders: int, *,
                  time_limit_s: float = 57600.0, window=(0, 28800)) -> dict:
    """Single-product serial line, deterministic, no defects, interval 0, unlimited WIP."""
    names = list(stage_times)
    stations = {}
    for i, name in enumerate(names):
        nxt = names[i + 1] if i + 1 < len(names) else "SHIP"
        stations[name] = {
            "machines": 1,
            "input_buffer": {"id": f"Q_{name}", "capacity": capacities.get(name)},
            "times_s": {"X": stage_times[name]},
            "route": {"next": nxt},
        }
    return {
        "schema_version": 1,
        "model_name": "serial_test_line",
        "time_unit": "s",
        "orders": {"counts": {"X": n_orders}, "order_ready_at_s": 0, "due_s": window[1]},
        "release": {"min_interval_s": 0, "wip_limit": None, "sequence": {"policy": "random_permutation"}},
        "run": {"observation_window_s": list(window), "time_limit_s": time_limit_s},
        "stochastic": {"enabled": False, "distribution": "lognormal", "cv": 0.0},
        "products": {"X": {"fail_rate": 0.0, "recover_prob": 1.0}},
        "line": {"first_station": names[0], "stations": stations},
        "time_scale": {},
    }


def abc_config(c_time: float = 30.0, n_orders: int = 2000, time_limit_s: float = 28800.0) -> dict:
    """V01: A(10s) -> B(15s) -> C(c_time), one machine each, unlimited buffers."""
    return serial_config({"A": 10.0, "B": 15.0, "C": c_time}, {}, n_orders, time_limit_s=time_limit_s)


def blocking_config(q_d_capacity: int = 1) -> dict:
    """V04: U(5s) -> D(20s, ships), Q_U unlimited, Q_D finite, 5 orders."""
    return serial_config({"U": 5.0, "D": 20.0}, {"U": None, "D": q_d_capacity}, 5)


@pytest.fixture
def baseline_cfg() -> dict:
    return load_yaml(BASELINE_PATH)


def deterministic(cfg: dict, *, no_defects: bool = True) -> dict:
    out = copy.deepcopy(cfg)
    out["stochastic"]["enabled"] = False
    if no_defects:
        for p in out["products"].values():
            p["fail_rate"] = 0.0
    return out
