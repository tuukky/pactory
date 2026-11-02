"""§7 Hand-eye (eye-to-hand): 노이즈 0 가상 데이터에서 정답을 수치 오차 수준으로 복원."""

from __future__ import annotations

import numpy as np

from v3_helpers import camera, gripper_cfg, rng
from vision_guide import calib as c
from vision_guide.geometry import inv_T

EXACT = ["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"]


def _hs():
    intr, T_base_cam = camera()
    return c.handeye_settings(gripper_cfg(), T_base_cam, intr), T_base_cam


def test_eye_to_hand_data_is_consistent():
    hs, T_base_cam = _hs()
    d = c.generate_handeye_data(hs, 8, rng(0))
    for A, B in zip(d["T_base_tcp"], d["T_cam_board_true"]):
        assert np.allclose(inv_T(A) @ T_base_cam @ B, hs["T_tcp_board"], atol=1e-12)


def test_noise_free_recovery():
    hs, T_base_cam = _hs()
    for seed in range(3):
        d = c.generate_handeye_data(hs, 10, rng(seed))
        for m in EXACT:
            e_mm, e_deg = c.handeye_error(c.calibrate_handeye(d["T_base_tcp"], d["T_cam_board_meas"], m), T_base_cam)
            assert e_mm < 1e-6 and e_deg < 1e-6, (m, e_mm, e_deg)


def test_precondition_is_equivalent_and_fixes_tsai():
    """base 고정 회전(전처리)은 같은 문제다: PARK 결과가 전처리 유무와 무관.
    전처리 없이 TCP가 아래를 보는 자세를 넣으면 OpenCV TSAI가 퇴화(≈180°)한다 — 전처리를 둔 이유 (verification.md)."""
    hs, T_base_cam = _hs()
    d = c.generate_handeye_data(hs, 12, rng(5), rot_noise_deg=0.05, trans_noise_m=0.00025)
    a = c.calibrate_handeye(d["T_base_tcp"], d["T_cam_board_meas"], "PARK", precondition=True)
    b = c.calibrate_handeye(d["T_base_tcp"], d["T_cam_board_meas"], "PARK", precondition=False)
    assert np.allclose(a, b, atol=1e-9)
    d0 = c.generate_handeye_data(hs, 10, rng(0))
    _, e_raw = c.handeye_error(c.calibrate_handeye(d0["T_base_tcp"], d0["T_cam_board_meas"], "TSAI",
                                                   precondition=False), T_base_cam)
    _, e_pre = c.handeye_error(c.calibrate_handeye(d0["T_base_tcp"], d0["T_cam_board_meas"], "TSAI"), T_base_cam)
    assert e_pre < 1e-6 and e_raw > 1.0


def test_error_grows_with_noise_and_shrinks_with_poses():
    hs, _ = _hs()
    levels = [{"rot_deg": 0.0, "trans_m": 0.0}, {"rot_deg": 0.1, "trans_m": 0.0005}]
    df = c.handeye_curve(hs, levels, [5, 30], 12, seed=0, methods=["PARK"])
    med = df.groupby(["noise_level", "n_poses"]).err_deg.median()
    assert med[(0, 5)] < 1e-6 and med[(1, 5)] > 1e-3
    assert med[(1, 30)] < med[(1, 5)]


def test_end_to_end_checkerboard_pnp():
    intr, _ = camera()
    hs, _ = _hs()
    df = c.handeye_end_to_end(hs, intr, 6, 1, seed=0, pixel_noise=0.0, methods=["PARK"])
    r = df.iloc[0]
    assert r.n_detected == 6
    assert r.err_mm < 2.0 and r.err_deg < 0.1
