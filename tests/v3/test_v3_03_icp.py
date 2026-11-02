"""§7 ICP: 알려진 변환 복원, 노이즈 0이면 오차 ≈ 0."""

from __future__ import annotations

import numpy as np

from v3_helpers import housing_model, pipeline_cfg, rng
from vision_guide.geometry import inv_T, make_T, rotation_about, rotation_between_deg, transform_points
from vision_guide.pose import refine_icp
from vision_guide.segment import to_pcd


def _case(noise_m: float, seed: int):
    model = housing_model()
    cfg = pipeline_cfg()
    tgt, _ = model.features(cfg)
    r = rng(seed)
    T_true = make_T(rotation_about(r.normal(size=3), 3.0), r.normal(size=3) * 0.003)  # 3°, ~5 mm
    src_pts = np.asarray(tgt.points)
    src = to_pcd(transform_points(T_true, src_pts) + r.normal(scale=noise_m, size=src_pts.shape) if noise_m else
                 transform_points(T_true, src_pts))
    T_est, fit, rmse = refine_icp(src, tgt, np.eye(4), cfg)  # T_tgt_src ≈ inv(T_true)
    D = T_est @ T_true
    return np.linalg.norm(D[:3, 3]), rotation_between_deg(D[:3, :3], np.eye(3)), fit, rmse


def test_icp_exact_recovery_without_noise():
    for seed in range(3):
        te, re, fit, rmse = _case(0.0, seed)
        assert te < 1e-6 and re < 1e-4
        assert fit == 1.0 and rmse < 1e-6


def test_icp_within_tolerance_with_noise():
    for seed in range(3):
        te, re, fit, _ = _case(0.001, seed)
        assert te < 0.001 and re < 0.5
        assert fit > 0.9


def test_inv_relation():
    T = make_T(rotation_about([0, 1, 0], 10.0), [0.01, 0, 0])
    assert np.allclose(inv_T(T) @ T, np.eye(4))
