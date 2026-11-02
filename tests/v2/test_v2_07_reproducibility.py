"""V07'·V16: 재현성, PYTHONHASHSEED 무관, 공통 난수(고장·도착)가 자원 변경과 무관."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd

from v2_helpers import REV_A_PATH, ROOT, rev_a, short
from factory_sim.config import apply_overrides
from factory_sim.engine import run_simulation
from factory_sim.metrics import compute_metrics


def test_same_seed_identical():
    cfg = short(rev_a(), 2, 8)
    a = run_simulation(cfg, 6, run_id="a", detail="full")
    b = run_simulation(cfg, 6, run_id="b", detail="full")
    pd.testing.assert_frame_equal(a.events.drop(columns="run_id"), b.events.drop(columns="run_id"))
    pd.testing.assert_frame_equal(a.orders_frame().drop(columns="run_id"), b.orders_frame().drop(columns="run_id"))
    ma, mb = compute_metrics(a), compute_metrics(b)
    ma.pop("run_id"), mb.pop("run_id")
    assert ma == mb


def test_different_seed_differs():
    cfg = short(rev_a(), 2, 8)
    a, b = run_simulation(cfg, 6), run_simulation(cfg, 7)
    assert list(a.orders_frame().arrival_s[:20]) != list(b.orders_frame().arrival_s[:20])


_SNIPPET = (
    "import json,sys;from factory_sim.config import resolve_config;from factory_sim.release import generate_orders;"
    "from factory_sim.randomness import duration_z,failure_draw,fail_u;"
    "o=generate_orders(resolve_config(sys.argv[1]),8,36000.0);"
    "print(json.dumps({'a':[(x.arrival_s,x.sku) for x in o[:30]],'z':duration_z(8,3,'ROTOR','R1'),"
    "'f':failure_draw(8,'S3_2',4),'u':fail_u(8,9)}))"
)


def test_draws_independent_of_hash_seed():
    outs = []
    for hs in ("1", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT / "src"))
        out = subprocess.run([sys.executable, "-c", _SNIPPET, str(REV_A_PATH)], env=env, capture_output=True,
                             text=True, check=True)
        outs.append(json.loads(out.stdout))
    assert outs[0] == outs[1]


def test_v16_crn_failures_and_arrivals_independent_of_resources():
    base = short(rev_a(), 4, 20)
    r2 = run_simulation(apply_overrides(base, {"agv.count": 2}), 9)
    r3 = run_simulation(base, 9)
    o2, o3 = r2.orders_frame(), r3.orders_frame()
    pd.testing.assert_series_equal(o2.arrival_s, o3.arrival_s)
    pd.testing.assert_series_equal(o2.sku, o3.sku)
    d2, d3 = r2.stats["failure_draws"], r3.stats["failure_draws"]
    assert d2.keys() == d3.keys()
    for mid in d2:
        n = min(len(d2[mid]), len(d3[mid]))
        assert n >= 1
        assert d2[mid][:n] == d3[mid][:n], mid
    # 결과 자체는 달라도 된다(자원 차이), 하지만 같은 수요를 공유한다.
    assert r2.stats["arrived"] == r3.stats["arrived"]


def test_demand_rate_scales_same_draws():
    base = short(rev_a(), 2, 8)
    a = run_simulation(base, 5).orders_frame()
    b = run_simulation(apply_overrides(base, {"demand.rate_per_h": 36.0}), 5).orders_frame()
    n = len(a)
    assert list(b.sku[:n]) == list(a.sku)
    assert (abs(b.arrival_s[:n].to_numpy() * 2 - a.arrival_s.to_numpy()) < 1e-6).all()
