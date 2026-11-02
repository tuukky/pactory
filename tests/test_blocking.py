"""V04: finite buffer and blocking-after-service."""

from __future__ import annotations

from conftest import blocking_config
from line_sim.engine import run_simulation


def _blocked(result, machine_id):
    iv = result.machine_intervals
    sel = iv[(iv.machine_id == machine_id) & (iv.state == "BLOCKED")]
    return [(float(a), float(b)) for a, b in zip(sel.start_s, sel.end_s)]


def _ships(result):
    return sorted(j.terminal_at for j in result.jobs if j.outcome.value == "shipped")


def test_blocking_capacity_1():
    r = run_simulation(blocking_config(1), seed=1, check_invariants=True)
    assert _ships(r) == [25.0, 45.0, 65.0, 85.0, 105.0]
    blocked = _blocked(r, "U_1")
    assert blocked == [(15.0, 25.0), (30.0, 45.0), (50.0, 65.0)]
    assert sum(b - a for a, b in blocked) == 40.0

    ev = r.events
    q_d = ev[ev.buffer_id == "Q_D"]
    assert q_d.buffer_count.max() == 1

    # While U is blocked it never starts a new job.
    starts = ev[(ev.event_type == "PROCESS_START") & (ev.machine_id == "U_1")].time_s
    for a, b in blocked:
        assert not any(a <= t < b for t in starts)


def test_blocking_capacity_2():
    r = run_simulation(blocking_config(2), seed=1, check_invariants=True)
    assert _ships(r) == [25.0, 45.0, 65.0, 85.0, 105.0]
    blocked = _blocked(r, "U_1")
    assert blocked == [(20.0, 25.0), (30.0, 45.0)]
    assert sum(b - a for a, b in blocked) == 20.0


def test_blocked_job_not_counted_in_buffer():
    r = run_simulation(blocking_config(1), seed=1)
    ev = r.events
    # Job 3 finishes U at 15 and is blocked until 25: it must not enter Q_D before 25.
    enter = ev[(ev.event_type == "BUFFER_ENTER") & (ev.buffer_id == "Q_D")]
    t_enter = {int(j): float(t) for j, t in zip(enter.job_id, enter.time_s)}
    order = [j.id for j in sorted(r.jobs, key=lambda j: j.release_seq)]
    assert t_enter[order[2]] == 25.0
    at_20 = enter[enter.time_s <= 20.0]
    last = at_20.buffer_count.iloc[-1]
    assert last == 1
