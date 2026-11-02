"""§7 지표: ADD/ADD-S를 BOP toolkit 정의와 대조, 손계산 사례, 실패 코드 분해."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import spatial

from v3_helpers import rng
from vision_guide import metrics as m
from vision_guide.failures import FailureCode, failure_breakdown, majority_grasp_failure, other_share_ok
from vision_guide.geometry import make_T, random_rotation, rotation_about


# --- BOP toolkit bop_toolkit_lib/pose_error.py 원문 전사 (패키지 미설치 환경의 대조 기준) ---
def _bop_transform_pts_Rt(pts, R, t):
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def _bop_add(R_est, t_est, R_gt, t_gt, pts):
    pts_est = _bop_transform_pts_Rt(pts, R_est, t_est)
    pts_gt = _bop_transform_pts_Rt(pts, R_gt, t_gt)
    return np.linalg.norm(pts_est - pts_gt, axis=1).mean()


def _bop_adi(R_est, t_est, R_gt, t_gt, pts):
    pts_est = _bop_transform_pts_Rt(pts, R_est, t_est)
    pts_gt = _bop_transform_pts_Rt(pts, R_gt, t_gt)
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)
    return nn_dists.mean()


def _bop_re(R_est, R_gt):
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    return np.rad2deg(np.arccos(error_cos))


def _random_case(r):
    pts = r.normal(scale=0.05, size=(300, 3))
    Te = make_T(random_rotation(r), r.normal(size=3))
    Tg = make_T(random_rotation(r), r.normal(size=3))
    return pts, Te, Tg


def test_against_transcribed_bop_definitions():
    r = rng(0)
    for _ in range(20):
        pts, Te, Tg = _random_case(r)
        args = (Te[:3, :3], Te[:3, 3], Tg[:3, :3], Tg[:3, 3], pts)
        assert m.add(Te, Tg, pts) == pytest.approx(_bop_add(*args), rel=1e-12)
        assert m.adi(Te, Tg, pts) == pytest.approx(_bop_adi(*args), rel=1e-12)
        assert m.re_deg(Te[:3, :3], Tg[:3, :3]) == pytest.approx(_bop_re(Te[:3, :3], Tg[:3, :3]), abs=1e-9)
        assert m.te(Te[:3, 3], Tg[:3, 3]) == pytest.approx(np.linalg.norm(Te[:3, 3] - Tg[:3, 3]))


def test_against_bop_toolkit_if_installed():
    pe = pytest.importorskip("bop_toolkit_lib.pose_error")
    r = rng(1)
    pts, Te, Tg = _random_case(r)
    a = (Te[:3, :3], Te[:3, 3].reshape(3, 1), Tg[:3, :3], Tg[:3, 3].reshape(3, 1), pts)
    assert m.add(Te, Tg, pts) == pytest.approx(pe.add(*a))
    assert m.adi(Te, Tg, pts) == pytest.approx(pe.adi(*a))


def test_hand_cases():
    pts = rng(2).normal(size=(200, 3))
    Tg = np.eye(4)
    # 순수 병진 d → ADD = d, te = d
    Te = make_T(None, [0.003, 0.004, 0.0])
    assert m.add(Te, Tg, pts) == pytest.approx(0.005)
    assert m.te(Te[:3, 3], Tg[:3, 3]) == pytest.approx(0.005)
    # 대칭 변환만큼 돌린 대칭 물체: ADI = 0, ADD > 0, re_sym = 0
    ring = np.array([[np.cos(a), np.sin(a), 0.0] for a in np.linspace(0, 2 * np.pi, 3, endpoint=False)])
    S = make_T(rotation_about([0, 0, 1], 120.0))
    assert m.adi(S, Tg, ring) == pytest.approx(0.0, abs=1e-12)
    assert m.add(S, Tg, ring) > 1.0
    syms = [np.eye(4), S, S @ S]
    assert m.re_sym_deg(S[:3, :3], np.eye(3), syms) == pytest.approx(0.0, abs=1e-6)
    assert m.re_deg(S[:3, :3], np.eye(3)) == pytest.approx(120.0)
    # 대칭축 주위 회전은 축 오차 0
    assert m.axis_error_deg(rotation_about([0, 0, 1], 37.0), np.eye(3), [0, 0, 1]) == pytest.approx(0.0, abs=1e-6)
    assert m.axis_error_deg(rotation_about([1, 0, 0], 5.0), np.eye(3), [0, 0, 1]) == pytest.approx(5.0)
    assert m.is_correct(9.99, 100.0) and not m.is_correct(10.0, 100.0) and not m.is_correct(np.nan, 1.0)


def test_summarize_counts_rejections_as_wrong():
    df = pd.DataFrame({
        "obj_id": [1, 1, 2, 2], "symmetric": [False, False, True, True], "diameter_mm": [100.0] * 4,
        "add_mm": [5.0, 5.0, 50.0, 50.0], "adi_mm": [4.0, 4.0, 5.0, 20.0], "te_mm": [1, 2, 3, 4],
        "re_deg": [1, 1, 1, 1], "code": ["OK", "LOW_FITNESS", "OK", "OK"]})
    s = m.summarize(df).set_index("obj_id")
    assert s.loc["1", "rate_add_s"] == 0.5 and s.loc["1", "raw_rate_add_s"] == 1.0
    assert s.loc["2", "rate_add_s"] == 0.5 and s.loc["2", "rate_add"] == 0.0
    assert s.loc["ALL", "rate_add_s"] == 0.5 and s.loc["ALL", "n"] == 4


def test_summarize_stats_accepted_only_and_sym_rotation():
    df = pd.DataFrame({
        "obj_id": [1, 1, 2, 2], "symmetric": [False, False, True, True], "diameter_mm": [100.0] * 4,
        "add_mm": [5.0] * 4, "adi_mm": [4.0] * 4, "te_mm": [1.0, 100.0, 3.0, 5.0],
        "re_deg": [2.0, 90.0, 179.0, 170.0], "re_sym_deg": [2.0, 90.0, 1.0, 3.0],
        "code": ["OK", "LOW_FITNESS", "OK", "OK"]})
    s = m.summarize(df).set_index("obj_id")
    # 채택만: 거부된 te 100 mm는 median에서 빠지고 raw에는 들어간다
    assert s.loc["1", "median_te_mm"] == 1.0 and s.loc["1", "raw_median_te_mm"] == 50.5
    # 대칭 물체는 re_sym_deg, 비대칭은 re_deg
    assert s.loc["2", "median_re_deg"] == 2.0 and s.loc["1", "median_re_deg"] == 2.0
    assert s.loc["ALL", "median_re_deg"] == 2.0 and s.loc["ALL", "raw_median_re_deg"] == 2.5


def test_failure_breakdown_and_tie_rule():
    df = failure_breakdown(["OK", "OK", FailureCode.NO_CLUSTER, "OTHER"])
    assert set(df["code"]) == {c.value for c in FailureCode}
    assert df["n"].sum() == 4
    assert df.loc[df.code == "OK", "share"].item() == 0.5
    assert not other_share_ok(df) and other_share_ok(failure_breakdown(["OK"] * 20 + ["OTHER"]))
    assert majority_grasp_failure(2, 2) == FailureCode.UNREACHABLE
    assert majority_grasp_failure(1, 3) == FailureCode.APPROACH_LIMIT
    assert majority_grasp_failure(0, 0) == FailureCode.NO_GRASP
