"""§7 합성 Depth 역투영 점이 정답 메시 표면 위에 있음, 합성 CAD 성질."""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from v3_helpers import T_cam_floor, camera, small_intr
from vision_guide import cad
from vision_guide.geometry import depth_to_points, inv_T, make_T, rotation_about, transform_points


def _surface_distance(mesh, T_cam_obj, pts_cam):
    scene = o3d.t.geometry.RaycastingScene()
    v = transform_points(T_cam_obj, np.asarray(mesh.vertices)).astype(np.float32)
    scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    return scene.compute_distance(o3d.core.Tensor(pts_cam.astype(np.float32))).numpy()


def test_plane_depth_exact():
    intr = small_intr()
    floor = cad.make_bin_floor((5.0, 5.0))
    depth, inst = cad.render_depth([(floor, make_T(rotation_about([1, 0, 0], 180.0), [0, 0, 1.25]))], intr)
    assert (inst == 0).all()
    assert np.allclose(depth, 1.25, atol=1e-6)


def test_backprojected_depth_lies_on_mesh_surface():
    intr, T_base_cam = camera()
    sc = cad.make_simple_scene(3, intr.pinhole(), T_cam_floor(T_base_cam), seed=11)
    for k, (mesh, T) in enumerate(zip(sc.meshes, sc.T_cam_objs)):
        pts, idx = depth_to_points(sc.depth_m, sc.intr, mask=sc.inst == k + 1)
        assert len(pts) > 500
        d = _surface_distance(mesh, T, pts)
        # float32 레이캐스팅 정밀도 수준
        assert np.max(d) < 2e-5


def test_noise_and_dropout_applied():
    intr, T_base_cam = camera()
    a = cad.make_simple_scene(2, intr.pinhole(), T_cam_floor(T_base_cam), seed=3)
    b = cad.make_simple_scene(2, intr.pinhole(), T_cam_floor(T_base_cam), seed=3, noise_sigma_m=0.002,
                              dropout=0.15)
    valid = a.depth_m > 0
    dropped = valid & (b.depth_m == 0)
    assert dropped.sum() / valid.sum() == pytest.approx(0.15, abs=0.01)
    both = valid & (b.depth_m > 0)
    assert np.std(b.depth_m[both] - a.depth_m[both]) == pytest.approx(0.002, rel=0.05)


def test_housing_model_points_on_exterior_and_symmetry():
    spec = cad.HousingSpec()
    parts = cad.housing_parts(spec)
    mesh = cad.make_housing(spec)
    pts, nrm = cad.sample_model_points(parts, 4000, seed=0)
    assert len(pts) == 4000
    assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0)
    # 외부 표면만: 다른 부품 안쪽 점 없음 → 몸체 축 근처 내부(맞닿은 면)에 점이 없어야 한다
    r = np.linalg.norm(pts[:, :2], axis=1)
    inner_contact = (np.abs(pts[:, 2] - spec.body_h) < 1e-6) & (r < spec.body_r - 1e-3)
    assert not inner_contact.any()
    for S in cad.housing_symmetries(spec):
        moved = transform_points(S, mesh.vertices)
        # 대칭 변환 후 꼭짓점이 원래 메시 표면 위
        d = _surface_distance(mesh, np.eye(4), moved)
        assert d.max() < 1e-6
    assert len(cad.housing_symmetries(spec)) == spec.n_legs
    d = cad.model_diameter(mesh)
    assert spec.total_h < d < 2 * spec.leg_reach + spec.total_h


def test_simple_scene_objects_do_not_overlap_and_rest_on_floor():
    intr, T_base_cam = camera()
    Tf = T_cam_floor(T_base_cam)
    sc = cad.make_simple_scene(5, intr.pinhole(), Tf, seed=7)
    centers = [(inv_T(Tf) @ T)[:2, 3] for T in sc.T_cam_objs]
    for i in range(5):
        for j in range(i + 1, 5):
            assert np.linalg.norm(centers[i] - centers[j]) > 2 * 0.058
    for mesh, T in zip(sc.meshes, sc.T_cam_objs):
        z = transform_points(inv_T(Tf) @ T, mesh.vertices)[:, 2]
        assert z.min() == pytest.approx(0.0, abs=1e-9)
