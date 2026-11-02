"""§7 역투영·투영 왕복, SE(3) 성질, 픽셀 규약."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from v3_helpers import camera, rng
from vision_guide import geometry as g


def _points(r, n=500):
    """시야(영상) 안의 점. 왜곡 모델은 영상 밖에서 단조가 아니므로 시야 안으로 제한한다."""
    z = r.uniform(0.3, 2.0, n)
    return np.stack([r.uniform(-0.5, 0.5, n) * z, r.uniform(-0.38, 0.38, n) * z, z], axis=1)


def test_project_backproject_roundtrip_pinhole():
    intr, _ = camera()
    intr = intr.pinhole()
    p = _points(rng(1))
    uv, z = g.project(p, intr)
    back = g.backproject(uv, z, intr)
    assert np.max(np.abs(back - p)) < 1e-9


def test_project_backproject_roundtrip_distorted():
    intr, _ = camera()
    assert intr.has_distortion
    p = _points(rng(2))
    uv, z = g.project(p, intr)
    back = g.backproject(uv, z, intr)
    # 왜곡 역보정은 cv2 반복법 (50회, eps 1e-14)
    assert np.max(np.abs(back - p)) < 1e-9


def test_pixel_convention_matches_opencv_with_half_shift():
    intr, _ = camera()
    p = _points(rng(3), 50)
    uv, _ = g.project(p, intr)
    uv_cv, _ = cv2.projectPoints(p.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intr.K_opencv(), intr.dist_array())
    assert np.allclose(uv, uv_cv.reshape(-1, 2) + 0.5, atol=1e-9)
    # 주점 위 점은 영상 중심 (W/2, H/2)에 투영된다
    uv0, _ = g.project(np.array([[0.0, 0.0, 1.0]]), intr.pinhole())
    assert np.allclose(uv0, [[intr.width / 2, intr.height / 2]])
    rt = g.CameraIntrinsics.from_opencv(intr.K_opencv(), intr.width, intr.height, intr.dist)
    assert rt == intr


def test_depth_to_points_uses_pixel_centers():
    intr = g.CameraIntrinsics(100.0, 100.0, 2.0, 1.5, 4, 3)
    depth = np.full((3, 4), 2.0)
    depth[0, 0] = 0.0
    pts, idx = g.depth_to_points(depth, intr)
    assert len(pts) == 11
    k = np.where((idx[:, 0] == 1) & (idx[:, 1] == 1))[0][0]
    # 픽셀 (u=1, v=1) 중심 = (1.5, 1.5) → x = (1.5-2)/100*2, y = 0
    assert np.allclose(pts[k], [-0.01, 0.0, 2.0])


def test_se3_compose_inverse_orthonormal():
    r = rng(4)
    for _ in range(50):
        A = g.make_T(g.random_rotation(r), r.normal(size=3))
        B = g.make_T(g.random_rotation(r), r.normal(size=3))
        assert g.is_se3(A) and g.is_se3(g.compose(A, B))
        assert np.allclose(g.compose(A, g.inv_T(A)), np.eye(4), atol=1e-12)
        assert np.allclose(g.inv_T(g.compose(A, B)), g.compose(g.inv_T(B), g.inv_T(A)), atol=1e-12)
        p = r.normal(size=(10, 3))
        assert np.allclose(g.transform_points(g.compose(A, B), p),
                           g.transform_points(A, g.transform_points(B, p)), atol=1e-12)
    assert not g.is_se3(np.diag([1.0, 1.0, -1.0, 1.0]))


def test_rotation_angles_and_perturb():
    R = g.rotation_about([0, 0, 1], 30.0)
    assert g.rotation_angle_deg(R) == pytest.approx(30.0)
    assert g.rotation_between_deg(R, np.eye(3)) == pytest.approx(30.0)
    assert g.rotation_angle_deg(g.rotation_about([1, 1, 0], 1e-4)) == pytest.approx(1e-4, rel=1e-6)
    T = g.make_T(g.random_rotation(rng(5)), [1, 2, 3])
    assert np.allclose(g.perturb_T(T, 0.0, 0.0, rng(6)), T)
    # 오른쪽 섭동의 회전은 병진을 바꾸지 않는다
    Tp = g.perturb_T(T, 2.0, 0.0, rng(7), side="right")
    assert np.allclose(Tp[:3, 3], T[:3, 3]) and g.rotation_between_deg(Tp[:3, :3], T[:3, :3]) > 0


def test_look_at_and_camera_yaml():
    intr, T_base_cam = camera()
    assert g.is_se3(T_base_cam)
    # 카메라는 빈 중심을 내려다본다: 시선(+z_cam) = -z_base
    assert np.allclose(T_base_cam[:3, 2], [0, 0, -1])
    assert np.allclose(T_base_cam[:3, 3], [0.6, 0.0, 1.0])
    # 빈 중심은 영상 중심, 깊이 1.0 m
    p_cam = g.transform_points(g.inv_T(T_base_cam), [[0.6, 0.0, 0.0]])
    uv, z = g.project(p_cam, intr.pinhole())
    assert np.allclose(uv, [[320.0, 240.0]]) and z[0] == pytest.approx(1.0)
    # 영상 위쪽(-y_cam)은 base +x
    assert np.allclose(-T_base_cam[:3, 1], [1, 0, 0])
