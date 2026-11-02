"""V07: reproducibility, common random numbers across scenarios and boundary handling."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd

from conftest import BASELINE_PATH, ROOT
from line_sim.config import apply_overrides, load_yaml, resolve_config
from line_sim.engine import run_simulation
from line_sim.entities import build_input_plan
from line_sim.metrics import compute_metrics, load_run_dir
from line_sim.run import write_run_dir

S5 = {"release.sequence": {"policy": "grouped", "order": ["P1", "P2", "P3"]}}


def test_same_seed_identical():
    cfg = resolve_config(BASELINE_PATH)
    a = run_simulation(cfg, 100, run_id="a")
    b = run_simulation(cfg, 100, run_id="b")
    pd.testing.assert_frame_equal(a.events.drop(columns="run_id"), b.events.drop(columns="run_id"))
    pd.testing.assert_frame_equal(a.jobs_frame().drop(columns="run_id"), b.jobs_frame().drop(columns="run_id"))
    pd.testing.assert_frame_equal(a.machine_intervals.drop(columns="run_id"),
                                  b.machine_intervals.drop(columns="run_id"))
    ma, mb = compute_metrics(a), compute_metrics(b)
    ma.pop("run_id"), mb.pop("run_id")
    assert ma == mb


def test_metrics_from_saved_dir_match(tmp_path):
    r = run_simulation(resolve_config(BASELINE_PATH), 101)
    m_mem = write_run_dir(r, tmp_path / "seed_101")
    m_dir = compute_metrics(load_run_dir(tmp_path / "seed_101"))
    assert m_mem == m_dir


_SNIPPET = (
    "import json,sys;from line_sim.config import resolve_config;from line_sim.entities import build_input_plan;"
    "p=build_input_plan(resolve_config(sys.argv[1]),100);"
    "print(json.dumps({'seq':list(p.sequence[:50]),'z':[s.draws.z['TEST'] for s in p.jobs[:50]],"
    "'f':[s.draws.fail_u for s in p.jobs[:50]]}))"
)


def test_draws_independent_of_hash_seed():
    outs = []
    for hs in ("1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT / "src"))
        out = subprocess.run([sys.executable, "-c", _SNIPPET, str(BASELINE_PATH)], env=env, capture_output=True,
                             text=True, check=True)
        outs.append(json.loads(out.stdout))
    assert outs[0] == outs[1]


def test_common_random_numbers_b0_vs_s5():
    base = resolve_config(BASELINE_PATH)
    s5 = resolve_config(BASELINE_PATH, S5)
    pb, ps = build_input_plan(base, 100), build_input_plan(s5, 100)
    assert [(s.id, s.product_type, s.draws) for s in pb.jobs] == [(s.id, s.product_type, s.draws) for s in ps.jobs]
    rb, rs = run_simulation(base, 100), run_simulation(s5, 100)
    ob = rb.orders_frame().sort_values("job_id").drop(columns=["run_id", "release_seq"]).reset_index(drop=True)
    os_ = rs.orders_frame().sort_values("job_id").drop(columns=["run_id", "release_seq"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ob, os_)
    # S5: products are released in contiguous groups P1 -> P2 -> P3.
    prods = [rs.jobs[jid - 1].product_type for jid in ps.sequence]
    assert prods == sorted(prods, key=["P1", "P2", "P3"].index)
    assert pb.sequence != ps.sequence


def test_time_scale_applies_to_processing_only():
    base = resolve_config(BASELINE_PATH)
    s3 = resolve_config(BASELINE_PATH, {"time_scale.ASM": 0.8})
    rb, rs = run_simulation(base, 100), run_simulation(s3, 100)
    orders = rb.orders_frame().set_index("job_id")
    jobs_s3 = rs.jobs_frame().set_index("job_id")
    for jid in orders.index[:100]:
        assert abs(jobs_s3.loc[jid, "ASM_process_s"] - orders.loc[jid, "ASM_base_s"] * 0.8) < 1e-9
    setups = rs.machine_intervals.query("state == 'SETUP'").duration_s.round(9).unique()
    assert set(setups) <= {20.0, 30.0, 40.0}
    pd.testing.assert_frame_equal(rb.orders_frame(), rs.orders_frame())


def test_release_interval_and_wip_limit():
    cfg = resolve_config(BASELINE_PATH)
    r = run_simulation(cfg, 100)
    rel = r.events[r.events.event_type == "JOB_RELEASED"].time_s.to_numpy()
    assert rel[0] == 0.0
    assert (rel[1:] - rel[:-1] >= 48 - 1e-9).all()
    m = compute_metrics(r)
    ev = r.events.sort_values("event_seq")
    delta = ev.event_type.map({"JOB_RELEASED": 1, "JOB_SHIPPED": -1, "JOB_SCRAPPED": -1}).fillna(0)
    assert delta.cumsum().max() <= 40
    assert m["shift"]["avg_wip"] <= 40


def test_time_limit_incomplete():
    cfg = apply_overrides(load_yaml(BASELINE_PATH),
                          {"run.time_limit_s": 20000, "run.observation_window_s": [0, 20000]})
    r = run_simulation(cfg, 100, check_invariants=True)
    assert r.status == "incomplete_at_limit"
    assert r.completion_time_s is None
    assert r.sim_end_s == 20000
    m = compute_metrics(r)
    assert m["full_run"]["completion_time_s"] is None
    assert m["full_run"]["line_time"] is None
    assert m["checks"]["mass_balance"]["ok"] and m["checks"]["time_conservation"]["ok"]
    assert any(j.outcome.value == "incomplete" for j in r.jobs)
