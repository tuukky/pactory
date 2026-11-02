"""V06: two parallel ASM machines with per-machine changeover memory."""

from __future__ import annotations

import pytest

from conftest import BASELINE_PATH, deterministic
from line_sim.config import load_yaml
from line_sim.engine import run_simulation


def _grouped(cfg, counts, order=("P1", "P2", "P3")):
    cfg["orders"]["counts"] = counts
    cfg["release"]["sequence"] = {"policy": "grouped", "order": list(order)}
    return cfg


def _intervals(r, machine_id, state):
    iv = r.machine_intervals
    sel = iv[(iv.machine_id == machine_id) & (iv.state == state)]
    return [(float(a), float(b), int(j)) for a, b, j in zip(sel.start_s, sel.end_s, sel.job_id)]


def test_independent_setup_memory():
    cfg = _grouped(deterministic(load_yaml(BASELINE_PATH)), {"P1": 1, "P2": 0, "P3": 2})
    r = run_simulation(cfg, seed=100, check_invariants=True)
    ev = r.events
    exits = ev[(ev.event_type == "BUFFER_EXIT") & (ev.buffer_id == "Q_ASM")]
    taken = {int(j): (m, float(t)) for j, m, t in zip(exits.job_id, exits.machine_id, exits.time_s)}
    assert taken[1] == ("ASM_1", 30.0)
    assert taken[2] == ("ASM_2", 88.0)
    assert taken[3] == ("ASM_1", 136.0)
    # ASM_2 had no previous product -> no setup; ASM_1 changes P1 -> P3 -> 40 s.
    assert _intervals(r, "ASM_2", "SETUP") == []
    assert _intervals(r, "ASM_1", "SETUP") == [(136.0, 176.0, 3)]
    assert (176.0, 286.0, 3) in _intervals(r, "ASM_1", "PROCESSING")


@pytest.mark.parametrize("counts,expected_setup", [
    ({"P1": 3, "P2": 0, "P3": 0}, []),
    ({"P1": 1, "P2": 0, "P3": 1}, [40.0]),
])
def test_single_machine_setup(counts, expected_setup):
    cfg = _grouped(deterministic(load_yaml(BASELINE_PATH)), counts)
    cfg["line"]["stations"]["ASM"]["machines"] = 1
    r = run_simulation(cfg, seed=100, check_invariants=True)
    setups = [b - a for a, b, _ in _intervals(r, "ASM_1", "SETUP")]
    assert setups == expected_setup


def test_parallel_overlap_allowed_but_not_within_machine():
    r = run_simulation(load_yaml(BASELINE_PATH), seed=100)
    iv = r.machine_intervals
    busy = iv[iv.state.isin(["SETUP", "PROCESSING", "BLOCKED"])]
    a1 = busy[busy.machine_id == "ASM_1"].sort_values("start_s")
    a2 = busy[busy.machine_id == "ASM_2"].sort_values("start_s")
    overlap = any(not (e1 <= s2 or e2 <= s1)
                  for s1, e1 in zip(a1.start_s, a1.end_s) for s2, e2 in zip(a2.start_s[:50], a2.end_s[:50]))
    assert overlap
    for g in (a1, a2):
        assert (g.start_s.iloc[1:].to_numpy() >= g.end_s.iloc[:-1].to_numpy()).all()
    # Each ASM job runs on exactly one machine: per-machine processing intervals never overlap.
    proc = iv[(iv.station_id == "ASM") & (iv.state == "PROCESSING")]
    assert proc.job_id.is_unique
