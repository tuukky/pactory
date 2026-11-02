"""통계: 지표 정의, 부트스트랩 시드 결정성, 같은 결과 CI ∋ 0, 알려진 차이 검출, 계수 쌍(H3 형식)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vision_guide.stats import add_correct_flags, by_visib, headline, paired_frame_bootstrap


def _df(n_frames=30, per=4, p_ok=0.8, p_cor=0.5, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_frames):
        for o in range(per):
            ok = rng.random() < p_ok
            cor = rng.random() < p_cor
            rows.append({"scene_id": 2, "im_id": f, "obj_id": o + 1, "symmetric": o == 0, "diameter_mm": 100.0,
                         "code": "OK" if ok else "LOW_FITNESS", "add_mm": 5.0 if cor else 50.0,
                         "adi_mm": 4.0 if cor else 40.0, "visib_fract": rng.uniform(0, 1),
                         "total_ms": 1000.0 + shift + rng.normal(0, 10), "region_hit": bool(rng.random() < 0.7)})
    return pd.DataFrame(rows)


def test_definitions():
    df = pd.DataFrame([
        {"scene_id": 1, "im_id": 1, "obj_id": 1, "symmetric": False, "diameter_mm": 100, "code": "OK", "add_mm": 5, "adi_mm": 5, "total_ms": 1},
        {"scene_id": 1, "im_id": 1, "obj_id": 2, "symmetric": False, "diameter_mm": 100, "code": "OK", "add_mm": 50, "adi_mm": 5, "total_ms": 2},
        {"scene_id": 1, "im_id": 2, "obj_id": 1, "symmetric": True, "diameter_mm": 100, "code": "OK", "add_mm": 50, "adi_mm": 5, "total_ms": 3},
        {"scene_id": 1, "im_id": 2, "obj_id": 2, "symmetric": False, "diameter_mm": 100, "code": "LOW_FITNESS", "add_mm": 5, "adi_mm": 5, "total_ms": 4},
    ])
    d = add_correct_flags(df)
    assert d["correct"].tolist() == [True, False, True, False]   # 대칭은 ADI, 거부는 오답
    assert d["false_accept"].tolist() == [False, True, False, False]
    h = headline(df)
    assert h["rate_add_s"] == 0.5 and h["false_accept_rate"] == 0.25 and abs(h["precision"] - 2 / 3) < 1e-12
    assert h["accepted_rate"] == 0.75 and h["median_total_ms"] == 2.5 and h["n_frames"] == 2


def test_by_visib_bins():
    df = _df()
    b = by_visib(df)
    assert b["n"].sum() == len(df) and len(b) == 4


def test_bootstrap_deterministic_and_identical():
    a = _df(seed=1)
    r1 = paired_frame_bootstrap({"a": a, "b": a.copy()}, [("a", "b")], n=2000, seed=5)
    r2 = paired_frame_bootstrap({"a": a, "b": a.copy()}, [("a", "b")], n=2000, seed=5)
    pd.testing.assert_frame_equal(r1, r2)
    assert (r1["estimate"].abs() < 1e-12).all()
    assert r1["ci_contains_zero"].all()


def test_bootstrap_detects_known_difference():
    a = _df(seed=2, p_cor=0.2)
    b = a.copy()
    b["add_mm"] = 5.0
    b["adi_mm"] = 4.0
    b["total_ms"] = a["total_ms"] - 300.0
    r = paired_frame_bootstrap({"a": a, "b": b}, [("a", "b")], n=2000, seed=0).set_index("metric")
    assert r.loc["rate_add_s", "estimate"] > 0.5 and r.loc["rate_add_s", "ci_low"] > 0
    assert r.loc["false_accept_rate", "ci_high"] < 0
    assert r.loc["median_total_ms", "ci_high"] < 0
    assert not r.loc["rate_add_s", "ci_contains_zero"]


def test_coef_pair_h3_form():
    a = _df(seed=3)
    r = paired_frame_bootstrap({"base": a, "v42": a.copy()}, [("base", "v42", 0.5)], ["false_accept_rate"], n=1000)
    fa = headline(a)["false_accept_rate"]
    assert abs(r.iloc[0]["estimate"] - 0.5 * fa) < 1e-12 and r.iloc[0]["coef"] == 0.5


def test_frame_set_mismatch_rejected():
    a = _df(seed=4)
    b = a[a.im_id != 0]
    try:
        paired_frame_bootstrap({"a": a, "b": b}, [("a", "b")], n=10)
    except ValueError:
        return
    raise AssertionError("프레임 집합이 다르면 거부해야 한다")
