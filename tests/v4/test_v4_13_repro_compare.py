"""재현 비교: 정확 일치(A) 판정, 전체 dev 비교(B) 판정 규칙."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vision_guide.stats import compare_runs, exact_compare


def _df(seed=0, n_frames=40):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        for o in range(5):
            rows.append({"scene_id": 2, "im_id": f, "obj_id": o + 1, "symmetric": False, "diameter_mm": 100.0,
                         "code": "OK" if rng.random() < 0.85 else "LOW_VISIBILITY",
                         "fitness": rng.uniform(0.5, 1), "rmse_mm": rng.uniform(1, 4),
                         "add_mm": 5.0 if rng.random() < 0.4 else 80.0, "adi_mm": 3.0, "total_ms": 1.0})
    return pd.DataFrame(rows)


def test_exact_equal_ignores_time_and_order():
    a = _df()
    b = a.sample(frac=1.0, random_state=1).copy()
    b["total_ms"] = 99.0
    b.loc[b.index[0], "add_mm"] = np.nan
    a.loc[b.index[0], "add_mm"] = np.nan
    r = exact_compare(a, b)
    assert r["equal"] and r["same_targets"]


def test_exact_detects_mismatch():
    a = _df()
    b = a.copy()
    b.loc[3, "fitness"] += 1e-12
    b.loc[5, "code"] = "LOW_FITNESS"
    r = exact_compare(a, b)
    assert not r["equal"] and r["mismatch"]["fitness"] == 1 and r["mismatch"]["code"] == 1
    assert not exact_compare(a, a.iloc[1:])["same_targets"]


def test_compare_runs_pass_and_fail():
    a = _df()
    same = compare_runs(a, a.copy(), n_boot=500)
    assert same["pass"] and same["target_agreement"] == 1.0
    b = a.copy()
    b["add_mm"] = 5.0  # 정답률 크게 상승
    diff = compare_runs(a, b, n_boot=500)
    assert not diff["pass"] and not diff["delta_ok"]
    small = a.copy()
    small.loc[0, "code"] = "LOW_FITNESS" if small.loc[0, "code"] == "OK" else "OK"
    s = compare_runs(a, small, n_boot=500)
    assert s["delta_ok"] and s["target_agreement"] >= 0.99
