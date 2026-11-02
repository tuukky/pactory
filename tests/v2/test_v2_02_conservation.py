"""V02'·V03'·V18: 물량 보존, WIP 면적 = 주문 체류 합, 설비·작업자·AGV 상태시간 보존(기준안)."""

from __future__ import annotations

import pytest

from v2_helpers import rev_a, short
from factory_sim.engine import run_simulation
from factory_sim.metrics import compute_metrics, load_run_dir
from factory_sim.run import write_run_dir


@pytest.fixture(scope="module")
def rev_a_short():
    return run_simulation(short(rev_a()), seed=2, check_invariants=True, detail="full")


def test_all_checks_ok(rev_a_short):
    m = compute_metrics(rev_a_short)
    assert m["checks"]["all_ok"], m["checks"]
    assert m["factory"]["good_in_window"] > 0


def test_state_time_conservation_each_resource(rev_a_short):
    r = rev_a_short
    w0, w1 = r.window
    s = r.resource_states
    for rtype in ("machine", "operator", "agv"):
        sub = s[s.resource_type == rtype]
        assert sub.resource_id.nunique() == {"machine": 21, "operator": 8, "agv": 3}[rtype]
        for rid, g in sub.groupby("resource_id"):
            assert g.window_s.sum() == pytest.approx(w1 - w0, abs=1e-6), rid
            assert g.full_s.sum() == pytest.approx(r.sim_end_s, abs=1e-6), rid
    # 상태구간과 누적이 일치
    iv = r.intervals
    tot = iv.groupby(["resource_id", "state"]).duration_s.sum()
    full = s.set_index(["resource_id", "state"]).full_s
    for k, v in tot.items():
        assert full[k] == pytest.approx(v, abs=1e-6)


def test_mass_balance(rev_a_short):
    o = rev_a_short.orders_frame()
    st = rev_a_short.stats
    released = o.released_s.notna()
    terminal = o.terminal_s.notna()
    assert len(o) == st["arrived"]
    assert (~released).sum() == st["backlog_end"]
    assert (o.outcome == "shipped").sum() == st["shipped"]
    assert (o.outcome == "scrapped").sum() == st["scrapped"]
    assert len(o) == (~released).sum() + (released & ~terminal).sum() + terminal.sum()
    # 시각 순서: 도착 ≤ 투입 ≤ 키트 완료 ≤ FA 시작 ≤ 종결
    done = o[o.outcome == "shipped"]
    assert (done.arrival_s <= done.released_s).all()
    assert (done.released_s <= done.kit_complete_s).all()
    assert (done.kit_complete_s <= done.fa_start_s).all()
    assert (done.fa_start_s <= done.terminal_s).all()


def test_metrics_from_saved_dir_match(tmp_path):
    r = run_simulation(short(rev_a(), 2, 8), seed=3)
    m_mem = compute_metrics(r)
    m_dir = write_run_dir(r, tmp_path / "seed_3")
    assert m_mem == m_dir == compute_metrics(load_run_dir(tmp_path / "seed_3"))
