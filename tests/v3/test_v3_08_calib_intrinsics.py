"""합성 체커보드: 렌더링 코너 위치가 투영과 일치(픽셀 규약), 내부 파라미터 복원."""

from __future__ import annotations

import numpy as np

from v3_helpers import camera, gripper_cfg, rng
from vision_guide import calib as c
from vision_guide.geometry import project, transform_points


def _board():
    return c.Board(**gripper_cfg()["calibration"]["board"])


def test_rendered_corners_match_projection():
    intr, _ = camera()
    board = _board()
    for T in c.sample_board_poses(3, board, intr, rng(0)):
        img = c.render_checkerboard(board, intr, T)
        det = c.detect_corners(img, board)
        assert det is not None
        uv, _ = project(transform_points(T, c.board_points(board)), intr)
        # OpenCV 코너(픽셀 중심 = 정수) + 0.5 = 이 모듈 규약
        err = np.linalg.norm(det + 0.5 - uv, axis=1)
        assert np.median(err) < 0.1 and err.max() < 0.3


def test_intrinsics_recovered_noise_free():
    intr, _ = camera()
    board = _board()
    r = rng(1)
    imgs = [c.render_checkerboard(board, intr, T) for T in c.sample_board_poses(12, board, intr, r)]
    res = c.calibrate_intrinsics(imgs, board, (intr.width, intr.height))
    e = c.intrinsics_error(res["intr"], intr)
    assert res["n_used"] == 12
    assert res["rms_px"] < 0.15
    # 렌더링 앨리어싱(4x4 슈퍼샘플)과 k3 상관 때문에 완전 복원은 아니다: 0.3 % 이내
    assert abs(e["d_fx"]) < 2.0 and abs(e["d_fy"]) < 2.0
    assert abs(e["d_cx"]) < 2.0 and abs(e["d_cy"]) < 2.0


def test_pixel_noise_and_board_geometry():
    intr, _ = camera()
    board = _board()
    T = c.sample_board_poses(1, board, intr, rng(2))[0]
    a = c.render_checkerboard(board, intr, T)
    b = c.render_checkerboard(board, intr, T, pixel_noise=5.0, rng=rng(3))
    assert 4.0 < np.std(b.astype(float) - a.astype(float)) < 6.0
    assert set(np.unique(a)) >= {0, 128, 255}
    p = c.board_points(board)
    assert p.shape == (board.cols * board.rows, 3)
    assert np.allclose(p[1] - p[0], [board.square_m, 0, 0])
    assert np.allclose(p[board.cols] - p[0], [0, board.square_m, 0])
