"""§7 Guidance: 손계산 장면(물체 1개, 축 정렬)의 T_base_tcp, 접근·작업영역 검사, 오차 예산 성질."""

from __future__ import annotations

import numpy as np
import pytest

from v3_helpers import GRIPPER_PATH, camera
from vision_guide.failures import FailureCode
from vision_guide.geometry import inv_T, make_T, rotation_about
from vision_guide.guidance import (check_approach, load_gripper, max_handeye_rot_for_tolerance, mc_error_budget,
                                   select_grasp, tcp_error, tcp_target)


def test_hand_calculated_tcp():
    _, T_base_cam = camera()
    # 손계산: 카메라 (0.6, 0, 1.0), 시선 -z_base, 영상 위 = +x_base
    #   -> x_cam = -y_base, y_cam = -x_base, z_cam = -z_base
    assert np.allclose(T_base_cam, [[0, -1, 0, 0.6], [-1, 0, 0, 0.0], [0, 0, -1, 1.0], [0, 0, 0, 1]])
    # 똑바로 놓인 하우징 원점 base (0.65, 0.05, 0), 회전 없음 -> 카메라 좌표 (-0.05, -0.05, 1.0), R = R_base_cam^T
    T_cam_obj = np.array([[0, -1, 0, -0.05], [-1, 0, 0, -0.05], [0, 0, -1, 1.0], [0, 0, 0, 1.0]])
    assert np.allclose(T_cam_obj, inv_T(T_base_cam) @ make_T(None, [0.65, 0.05, 0.0]))
    grip, ws = load_gripper(GRIPPER_PATH)
    T = tcp_target(T_base_cam, T_cam_obj, grip.grasps[0])  # 윗면 파지: (0,0,0.08), x축 180°
    expect = np.array([[1, 0, 0, 0.65], [0, -1, 0, 0.05], [0, 0, -1, 0.08], [0, 0, 0, 1.0]])
    assert np.allclose(T, expect, atol=1e-12)
    ok, tilt = check_approach(T, grip)
    assert ok and tilt == pytest.approx(0.0, abs=1e-6)
    d = select_grasp(T_base_cam, [T_cam_obj], grip, ws)
    assert d.code == FailureCode.OK and d.grasp_idx == 0 and np.allclose(d.T_base_tcp, expect)


def test_flipped_tilted_and_out_of_workspace():
    _, T_base_cam = camera()
    grip, ws = load_gripper(GRIPPER_PATH)
    T_cam_base = inv_T(T_base_cam)
    flipped = T_cam_base @ make_T(rotation_about([1, 0, 0], 180.0), [0.6, 0.0, 0.08])
    d = select_grasp(T_base_cam, [flipped], grip, ws)
    assert d.code == FailureCode.OK and d.grasp_idx == 1 and d.tilt_deg == pytest.approx(0.0, abs=1e-6)
    tilted = T_cam_base @ make_T(rotation_about([1, 0, 0], 40.0), [0.6, 0.0, 0.0])
    d = select_grasp(T_base_cam, [tilted], grip, ws)
    assert d.code == FailureCode.APPROACH_LIMIT and d.n_approach == 2
    far = T_cam_base @ make_T(None, [1.5, 0.0, 0.0])
    assert select_grasp(T_base_cam, [far], grip, ws).code == FailureCode.UNREACHABLE
    assert select_grasp(T_base_cam, [], grip, ws).code == FailureCode.NO_GRASP
    # 대칭 후보 중 기울기 최소 선택
    small = T_cam_base @ make_T(rotation_about([1, 0, 0], 10.0), [0.6, 0.0, 0.0])
    d = select_grasp(T_base_cam, [tilted, small], grip, ws)
    assert d.sym_idx == 1 and d.tilt_deg == pytest.approx(10.0)


def test_tcp_error_and_budget_properties():
    _, T_base_cam = camera()
    grip, _ = load_gripper(GRIPPER_PATH)
    T_cam_obj = inv_T(T_base_cam) @ make_T(None, [0.6, 0.0, 0.0])
    G = grip.grasps[0]
    pe, ae = tcp_error(make_T(None, [0.001, 0, 0]), np.eye(4), grip.approach_axis_tcp)
    assert pe == pytest.approx(0.001) and ae == 0.0
    df = mc_error_budget(T_base_cam, T_cam_obj, G, grip, (0.0, 0.0), (0.0, 0.0), 50, 0)
    assert df.pos_err_mm.max() < 1e-9 and df.success.all()
    # Hand-eye 회전만: 카메라 원점 주위 회전 -> 위치 오차 RMS ≈ √2 · σ[rad] · 거리
    sig = 0.05
    df = mc_error_budget(T_base_cam, T_cam_obj, G, grip, (0.0, 0.0), (sig, 0.0), 4000, 1)
    h = df[df.source == "handeye"]
    dist = np.linalg.norm((T_cam_obj @ G)[:3, 3])
    rms = np.sqrt(np.mean(np.square(h.pos_err_mm))) * 1e-3
    assert rms == pytest.approx(np.sqrt(2) * np.radians(sig) * dist, rel=0.05)
    assert (df[df.source == "pose"].pos_err_mm == 0).all()
    best, grid = max_handeye_rot_for_tolerance(T_base_cam, T_cam_obj, G, grip, (0.0, 0.0), 0.0,
                                               [0.0, 0.05, 0.1, 0.2], 1000, 0, 0.95)
    assert grid.success.is_monotonic_decreasing
    assert best is not None and best < 0.2
