"""V02 (mass conservation) and V03 (time conservation) on the baseline line."""

from __future__ import annotations

from collections import Counter

import pytest

from conftest import BASELINE_PATH, serial_config
from line_sim.config import resolve_config
from line_sim.engine import run_simulation
from line_sim.metrics import compute_metrics, job_window_residence_sum, state_time_in_window

TOL = 1e-6


@pytest.fixture(scope="module")
def base_run():
    return run_simulation(resolve_config(BASELINE_PATH), seed=100, check_invariants=True)


def test_invariants_and_status(base_run):
    assert base_run.status == "complete"
    m = compute_metrics(base_run)
    assert all(c["ok"] for c in m["checks"].values())


def test_event_replay_mass_balance(base_run):
    """Replay the log; at the end of every time group each job has exactly one location."""
    cfg = base_run.config
    caps = {st["input_buffer"]["id"]: st["input_buffer"]["capacity"] for st in cfg["line"]["stations"].values()}
    n = len(base_run.jobs)
    loc: dict[int, str] = {}
    last_count: dict[str, int] = {}
    ev = base_run.events.sort_values("event_seq")
    rows = list(ev.itertuples(index=False))

    def check(t):
        c = Counter(loc.values())
        assert "TRANSIT" not in c, f"t={t}: 투입 후 위치 미정 Job 존재"
        in_buf = sum(c[b] for b in caps)
        in_mach = sum(v for k, v in c.items() if k not in caps and k not in ("POOL", "SHIPPED", "SCRAPPED"))
        assert all(v <= 1 for k, v in c.items() if k not in caps and k not in ("POOL", "SHIPPED", "SCRAPPED"))
        assert n == c["POOL"] + in_buf + in_mach + c["SHIPPED"] + c["SCRAPPED"]
        for b in caps:
            assert c[b] == last_count.get(b, 0), f"t={t}: {b} 로그 점유와 위치 재생 불일치"

    for i, r in enumerate(rows):
        et = r.event_type
        if et == "ORDER_CREATED":
            loc[int(r.job_id)] = "POOL"
        elif et == "JOB_RELEASED":
            loc[int(r.job_id)] = "TRANSIT"
        elif et == "BUFFER_ENTER":
            loc[int(r.job_id)] = r.buffer_id
            last_count[r.buffer_id] = int(r.buffer_count)
        elif et == "BUFFER_EXIT":
            loc[int(r.job_id)] = r.machine_id
            last_count[r.buffer_id] = int(r.buffer_count)
        elif et == "JOB_SHIPPED":
            assert loc[int(r.job_id)] == r.machine_id
            loc[int(r.job_id)] = "SHIPPED"
        elif et == "JOB_SCRAPPED":
            assert loc[int(r.job_id)] == r.machine_id
            loc[int(r.job_id)] = "SCRAPPED"
        if i + 1 == len(rows) or rows[i + 1].time_s > r.time_s:
            check(r.time_s)


def test_one_terminal_event_per_job(base_run):
    ev = base_run.events
    term = ev[ev.event_type.isin(["JOB_SHIPPED", "JOB_SCRAPPED"])]
    assert term.job_id.is_unique
    assert len(term) == len(base_run.jobs)


def test_buffer_count_bounds(base_run):
    caps = {st["input_buffer"]["id"]: st["input_buffer"]["capacity"]
            for st in base_run.config["line"]["stations"].values()}
    ev = base_run.events[base_run.events.buffer_id.notna()]
    for bid, grp in ev.groupby("buffer_id"):
        assert grp.buffer_count.min() >= 0
        assert grp.buffer_count.max() <= caps[bid]


def test_state_time_sums(base_run):
    iv = base_run.machine_intervals
    for w0, w1 in ((0.0, 28800.0), (0.0, base_run.sim_end_s)):
        for mid, d in state_time_in_window(iv, w0, w1).items():
            assert abs(sum(d.values()) - (w1 - w0)) <= TOL, mid


def test_wip_area_identity(base_run):
    m = compute_metrics(base_run)
    job_sum = job_window_residence_sum(base_run.jobs_frame(), base_run.sim_end_s, 0.0, 28800.0)
    assert abs(m["shift"]["avg_wip"] * 28800.0 - job_sum) <= TOL


def test_job_time_monotonic(base_run):
    for j in base_run.jobs:
        assert j.admitted_at is not None and j.terminal_at is not None
        assert j.order_ready_at <= j.admitted_at <= j.terminal_at
        prev_depart = j.admitted_at
        for op in j.operation_history:
            assert op.buffer_enter_s == prev_depart
            seq = [op.buffer_enter_s, op.buffer_exit_s, op.setup_start_s, op.setup_end_s,
                   op.process_start_s, op.process_end_s, op.depart_s]
            seq = [t for t in seq if t is not None]
            assert seq == sorted(seq)
            assert op.process_end_s is not None and op.depart_s is not None
            prev_depart = op.depart_s
        assert prev_depart == j.terminal_at


def test_machine_intervals_contiguous(base_run):
    iv = base_run.machine_intervals
    for mid, grp in iv.groupby("machine_id"):
        g = grp.sort_values("start_s")
        assert g.start_s.iloc[0] == 0.0
        assert g.end_s.iloc[-1] == base_run.sim_end_s
        assert (g.duration_s > 0).all()
        assert (g.start_s.iloc[1:].to_numpy() == g.end_s.iloc[:-1].to_numpy()).all(), mid


def test_early_finish_still_covers_window():
    cfg = serial_config({"A": 10.0, "B": 20.0}, {}, 10)
    r = run_simulation(cfg, seed=1, check_invariants=True)
    assert r.status == "complete"
    assert r.completion_time_s < 28800
    assert r.sim_end_s == 28800
    for d in state_time_in_window(r.machine_intervals, 0.0, 28800.0).values():
        assert abs(sum(d.values()) - 28800.0) <= TOL


def test_unshipped_decomposition(base_run):
    s = compute_metrics(base_run)["shift"]
    u = s["unshipped"]
    assert u["total"] == s["orders_total"] - s["good_shipped"]
    assert u["total"] == u["unreleased"] + u["wip"] + u["scrapped"]


@pytest.mark.parametrize("station", ["ASM", "TEST", "PACK"])
def test_processing_intervals_per_job(base_run, station):
    """Each PROCESSING interval belongs to one job; per-job sums match jobs.csv process times."""
    iv = base_run.machine_intervals
    proc = iv[(iv.station_id == station) & (iv.state == "PROCESSING")]
    jobs = base_run.jobs_frame()
    visited = jobs[jobs[f"{station}_process_s"].notna()]
    for mid, grp in proc.groupby("machine_id"):
        expected = set(visited.loc[visited[f"{station}_machine"] == mid, "job_id"].astype(int))
        assert set(grp.job_id.astype(int)) == expected, mid
    sums = proc.groupby("job_id").duration_s.sum()
    for jid, d in zip(visited.job_id, visited[f"{station}_process_s"]):
        assert abs(sums.loc[jid] - d) <= TOL


def test_window_with_late_start(base_run):
    m = compute_metrics(base_run, window=(3600.0, 28800.0))
    s = m["shift"]
    u = s["unshipped"]
    assert u["total"] == s["orders_total"] - s["good_shipped"]
    assert u["total"] == u["unreleased"] + u["wip"] + u["scrapped"]
    assert all(c["ok"] for c in m["checks"].values())
    assert s["good_shipped_in_window"] <= s["good_shipped"]
    assert s["good_shipped"] == compute_metrics(base_run)["shift"]["good_shipped"]
