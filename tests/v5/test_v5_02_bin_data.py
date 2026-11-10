"""PLAN_v5 S1 — 하우징 bin 데이터 규칙 (분할 시드, 깊이 후처리, ROI). 렌더 파일이 없어도 도는 단위 검사."""

from __future__ import annotations

import numpy as np
import pytest

from vision_guide import bin_v5 as B


@pytest.fixture(scope="module")
def cfg():
    return B.load_cfg()


def test_split_seeds_disjoint(cfg):
    seeds = {s: {sc["seed"] for sc in B.plan_scenes(cfg, s)} for s in ("dev", "eval", "train", "val")}
    for a in seeds:
        for b in seeds:
            if a < b:
                assert not seeds[a] & seeds[b], (a, b)


def test_dev_eval_fixed_camera(cfg):
    _, T = B.camera()
    for s in ("dev", "eval"):
        for sc in B.plan_scenes(cfg, s):
            assert len(sc["cam2world_cv"]) == 1
            np.testing.assert_allclose(np.asarray(sc["cam2world_cv"][0]), T)


def test_condition_seeds_unique(cfg):
    names = B.condition_names(cfg)
    seeds = {B.condition_seed(cfg, 60001, c) for c in names}
    assert len(seeds) == len(names)


def test_depth_condition_noise_and_dropout():
    rng = np.random.default_rng(0)
    depth = np.full((120, 160), 1.0)
    inst = -np.ones((120, 160), int)
    inst[30:90, 40:120] = 0
    d0 = B.apply_depth_condition(depth, inst, 0.0, 0.0, 1)
    np.testing.assert_array_equal(d0, depth)
    d = B.apply_depth_condition(depth, inst, 0.0, 0.30, 1)
    obj = inst >= 0
    frac = float((d[obj] == 0).mean())
    assert frac == pytest.approx(0.30, abs=0.01)
    assert (d[~obj] > 0).all()                    # 빈·바닥에는 결측 없음
    a = B.apply_depth_condition(depth, inst, 2.0, 0.05, 7)
    b = B.apply_depth_condition(depth, inst, 2.0, 0.05, 7)
    np.testing.assert_array_equal(a, b)           # 같은 시드면 같은 후처리
    sd = float(np.std(a[(a > 0) & ~obj] - 1.0))
    assert sd == pytest.approx(0.002, rel=0.1)
    _ = rng


def test_bin_roi_excludes_walls(cfg):
    intr, T = B.camera()
    from vision_guide.cad import render_depth
    import trimesh
    b = cfg["bin"]
    cx, cy, _ = b["center_m"]
    sx, sy = b["inner_size_xy_m"]
    floor = trimesh.creation.box(extents=[sx + 0.02, sy + 0.02, 0.01])
    wall = trimesh.creation.box(extents=[0.01, sy, b["wall_height_m"]])
    T_floor = np.eye(4); T_floor[:3, 3] = [cx, cy, -0.005]
    T_wall = np.eye(4); T_wall[:3, 3] = [cx + sx / 2 + 0.005, cy, b["wall_height_m"] / 2]
    from vision_guide.geometry import inv_T
    depth, ins = render_depth([(floor, inv_T(T) @ T_floor), (wall, inv_T(T) @ T_wall)], intr)
    roi = B.bin_roi_mask(depth, intr, T, cfg, {"margin_m": 0.008, "z_min_m": 0.003, "z_max_m": 0.3})
    assert not roi[ins == 1].any()                # 벽 픽셀은 모두 제외
    assert not roi[ins == 0].any()                # 바닥(z≈0)도 z_min 아래라 제외
    assert B.bin_roi_mask(depth, intr, T, cfg, None) is None
