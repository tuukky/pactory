"""픽셀 ↔ 점 대응: 역투영 → 투영 → floor 왕복, voxel 점이 마스크 안, 클러스터 RGB 회수."""

from __future__ import annotations

import numpy as np

from v4_helpers import intr_small
from vision_guide.color import cluster_hsv, pixels_of_points, rgb_to_hsv_u8
from vision_guide.geometry import depth_to_points
from vision_guide.segment import to_pcd


def _scene():
    intr = intr_small()
    depth = np.zeros((intr.height, intr.width))
    yy, xx = np.mgrid[0:intr.height, 0:intr.width]
    mask = (xx - 70) ** 2 + (yy - 50) ** 2 < 25 ** 2
    depth[mask] = 0.6 + 0.0005 * (xx[mask] - 70)
    return intr, depth, mask


def test_roundtrip_pixels():
    intr, depth, mask = _scene()
    pts, idx = depth_to_points(depth, intr)
    v, u = pixels_of_points(pts, intr)
    got = set(zip(v.tolist(), u.tolist()))
    want = set(map(tuple, idx.tolist()))
    assert got == want


def test_voxel_points_inside_mask():
    intr, depth, mask = _scene()
    pts, _ = depth_to_points(depth, intr)
    pcd = to_pcd(pts).voxel_down_sample(0.004)
    v, u = pixels_of_points(np.asarray(pcd.points), intr)
    assert len(v) > 50
    assert mask[v, u].mean() >= 0.99


def test_rgb_recovered():
    intr, depth, mask = _scene()
    rgb = np.zeros((intr.height, intr.width, 3), np.uint8)
    yy, xx = np.mgrid[0:intr.height, 0:intr.width]
    rgb[..., 0] = xx
    rgb[..., 1] = yy
    rgb[..., 2] = 200
    pts, idx = depth_to_points(depth, intr)
    v, u = pixels_of_points(pts, intr)
    assert np.array_equal(rgb[v, u, 0], u) and np.array_equal(rgb[v, u, 1], v)
    hsv = cluster_hsv(to_pcd(pts), rgb_to_hsv_u8(rgb), intr)
    assert hsv.shape == (len(idx), 3)
