"""합성 하우징 CAD와 간이 Depth 장면 (합성 표기 필수).

하우징은 원통 몸체 + 상단 장착 플랜지 + 하단 다리 n개로 된 **합성 CAD**다(실제 제품 도면 아님).
- 물체 좌표: 원점 = 몸체 축 위 바닥(다리 밑면) 중심, +z = 대칭축(플랜지 쪽).
- 대칭: 다리 n개 → z축 주위 C_n (항등 포함 n개).

간이 장면(make_simple_scene)은 BlenderProc 합성 데이터(4주차) 전에 파이프라인·Guidance를 시험하기 위한
numpy + Open3D 레이캐스팅 렌더러다. 물체는 빈 바닥에 똑바로(다리 아래) 또는 뒤집혀(플랜지 아래) 놓인다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.distance import pdist

from .geometry import (CameraIntrinsics, compose, make_T, normalized_rays, pixel_centers, rotation_about,
                       transform_points)

SYNTHETIC_LABEL = "합성 CAD (실제 제품 도면 아님)"


@dataclass(frozen=True)
class HousingSpec:
    """치수 m (합성 가정)."""

    body_r: float = 0.035
    body_h: float = 0.074
    flange_r: float = 0.050
    flange_t: float = 0.006
    n_legs: int = 3
    leg_w: float = 0.014
    leg_h: float = 0.012
    leg_reach: float = 0.058  # 다리 끝 반경
    sections: int = 64

    @property
    def total_h(self) -> float:
        return self.body_h + self.flange_t


def housing_parts(spec: HousingSpec | None = None) -> list[trimesh.Trimesh]:
    """볼록 부품 목록 (몸체, 플랜지, 다리들). 부품은 서로 맞닿거나 겹친다."""
    s = spec or HousingSpec()
    # 원통 분할 수를 다리 수의 배수로 맞춰 다각형 메시도 정확히 C_n 대칭이 되게 한다
    sec = int(np.ceil(s.sections / s.n_legs) * s.n_legs)
    body = trimesh.creation.cylinder(radius=s.body_r, height=s.body_h, sections=sec)
    body.apply_translation([0, 0, s.body_h / 2])
    flange = trimesh.creation.cylinder(radius=s.flange_r, height=s.flange_t, sections=sec)
    flange.apply_translation([0, 0, s.body_h + s.flange_t / 2])
    parts = [body, flange]
    r_in = s.body_r * 0.8  # 몸체 안쪽으로 겹쳐 틈이 없게
    for k in range(s.n_legs):
        leg = trimesh.creation.box(extents=[s.leg_reach - r_in, s.leg_w, s.leg_h])
        leg.apply_translation([(s.leg_reach + r_in) / 2, 0, s.leg_h / 2])
        leg.apply_transform(make_T(rotation_about([0, 0, 1], 360.0 * k / s.n_legs)))
        parts.append(leg)
    return parts


def make_housing(spec: HousingSpec | None = None) -> trimesh.Trimesh:
    """부품을 이어붙인 메시 (불리언 합집합 아님; 렌더링·레이캐스팅용)."""
    mesh = trimesh.util.concatenate(housing_parts(spec))
    mesh.metadata["label"] = SYNTHETIC_LABEL
    return mesh


def housing_symmetries(spec: HousingSpec | None = None) -> list[np.ndarray]:
    """C_n 대칭 4x4 목록 (항등 포함). T_obj_obj' 형태로 정답 자세에 오른쪽 곱한다."""
    s = spec or HousingSpec()
    return [make_T(rotation_about([0, 0, 1], 360.0 * k / s.n_legs)) for k in range(s.n_legs)]


def make_bin_floor(size_xy: tuple[float, float] = (0.60, 0.40), z: float = 0.0) -> trimesh.Trimesh:
    """빈 바닥 판 (윗면 = z)."""
    floor = trimesh.creation.box(extents=[size_xy[0], size_xy[1], 0.01])
    floor.apply_translation([0, 0, z - 0.005])
    return floor


def _inside_convex(part: trimesh.Trimesh, pts: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """볼록 부품 내부(경계 포함) 판정: 모든 면 평면의 안쪽."""
    n = part.face_normals
    d = np.einsum("ij,ij->i", n, part.triangles[:, 0, :])
    return np.all(pts @ n.T - d <= tol, axis=1)


def sample_model_points(mesh_or_parts, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """표면 균등 샘플 (점, 법선). 부품 목록을 주면 다른 부품 안(맞닿은 면 포함)에 있는 점은 버린다."""
    rng = np.random.default_rng(seed)
    parts = mesh_or_parts if isinstance(mesh_or_parts, (list, tuple)) else [mesh_or_parts]
    areas = np.array([p.area for p in parts])
    pts_all, nrm_all = [], []
    # 내부 점을 버리므로 넉넉히 뽑은 뒤 n개로 줄인다
    m = int(n * 1.6) + 10
    counts = rng.multinomial(m, areas / areas.sum())
    for i, (part, c) in enumerate(zip(parts, counts)):
        if c == 0:
            continue
        pts, fidx = trimesh.sample.sample_surface(part, int(c), seed=int(rng.integers(2**31)))
        nrm = part.face_normals[fidx]
        keep = np.ones(len(pts), dtype=bool)
        for j, other in enumerate(parts):
            if j != i and len(parts) > 1:
                keep &= ~_inside_convex(other, pts)
        pts_all.append(np.asarray(pts)[keep])
        nrm_all.append(np.asarray(nrm)[keep])
    pts = np.concatenate(pts_all)
    nrm = np.concatenate(nrm_all)
    if len(pts) > n:
        idx = rng.choice(len(pts), n, replace=False)
        pts, nrm = pts[idx], nrm[idx]
    return pts, nrm


def model_diameter(points_or_mesh) -> float:
    """두 점 사이 최대 거리 (볼록껍질 꼭짓점으로 계산)."""
    if isinstance(points_or_mesh, trimesh.Trimesh):
        pts = points_or_mesh.convex_hull.vertices
    else:
        pts = np.asarray(points_or_mesh, dtype=float)
        pts = trimesh.convex.convex_hull(pts).vertices if len(pts) > 4 else pts
    return float(pdist(pts).max())


# ---------------------------------------------------------------------------- 렌더링
def render_depth(meshes_with_T: list[tuple[trimesh.Trimesh, np.ndarray]], intr: CameraIntrinsics,
                 T_world_cam: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """레이캐스팅 z-깊이 영상 (m, 미검출 0)과 인스턴스 영상 (메시 순번, 미검출 -1).

    광선은 픽셀 중심 (u+0.5, v+0.5)에서 z=1 정규화 방향이므로 t_hit가 곧 z-깊이다.
    T_world_cam이 없으면 각 T는 T_cam_obj, 있으면 T_world_obj로 본다.
    """
    T_cam_world = np.eye(4) if T_world_cam is None else np.linalg.inv(T_world_cam)
    scene = o3d.t.geometry.RaycastingScene()
    ids = []
    for mesh, T in meshes_with_T:
        T_cam_obj = compose(T_cam_world, T)
        v = transform_points(T_cam_obj, np.asarray(mesh.vertices)).astype(np.float32)
        f = np.asarray(mesh.faces, dtype=np.uint32)
        ids.append(scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f)))
    rays_dir = normalized_rays(pixel_centers(intr), intr)
    rays = np.concatenate([np.zeros_like(rays_dir), rays_dir], axis=1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t = ans["t_hit"].numpy().astype(float)
    gid = ans["geometry_ids"].numpy().astype(np.int64)
    hit = np.isfinite(t)
    depth = np.where(hit, t * rays_dir[:, 2], 0.0).reshape(intr.height, intr.width)
    lookup = {int(g): k for k, g in enumerate(ids)}
    inst = np.full(gid.shape, -1, dtype=np.int64)
    inst[hit] = [lookup.get(int(g), -1) for g in gid[hit]] if hit.any() else []
    return depth, inst.reshape(intr.height, intr.width)


@dataclass
class SimpleScene:
    """간이 합성 장면 (합성). 인스턴스 영상에서 0 = 바닥, k+1 = k번째 물체."""

    depth_m: np.ndarray
    intr: CameraIntrinsics
    T_cam_objs: list[np.ndarray]
    meshes: list[trimesh.Trimesh]
    seed: int
    inst: np.ndarray = field(repr=False, default=None)
    T_cam_floor: np.ndarray = field(repr=False, default=None)
    flipped: list[bool] = field(default_factory=list)
    label: str = "합성 장면 (간이 레이캐스팅)"


def _rest_pose(spec: HousingSpec, flipped: bool, yaw_deg: float, xy: np.ndarray) -> np.ndarray:
    """바닥 좌표에서 물체 자세: 똑바로(다리 아래) 또는 뒤집힘(플랜지 아래)."""
    T_yaw = make_T(rotation_about([0, 0, 1], yaw_deg), [xy[0], xy[1], 0.0])
    if not flipped:
        return T_yaw
    return T_yaw @ make_T(rotation_about([1, 0, 0], 180.0), [0, 0, spec.total_h])


def make_simple_scene(n_objects: int, intr: CameraIntrinsics, T_cam_floor: np.ndarray, seed: int, *,
                      noise_sigma_m: float = 0.0, dropout: float = 0.0, spec: HousingSpec | None = None,
                      floor_size_xy: tuple[float, float] = (0.60, 0.40), flip_prob: float = 0.3,
                      margin_m: float = 0.005) -> SimpleScene:
    """빈 바닥에 하우징 n개를 겹치지 않게(외접원 거부 샘플링) 놓고 Depth를 렌더링한다.

    노이즈: 유효 픽셀에 N(0, σ²) (m). 결측: 유효 픽셀 중 비율 dropout을 무작위로 0으로.
    """
    s = spec or HousingSpec()
    rng = np.random.default_rng(seed)
    mesh = make_housing(s)
    r_obj = max(s.flange_r, s.leg_reach) + margin_m
    half = np.array(floor_size_xy) / 2 - r_obj
    centers: list[np.ndarray] = []
    for _ in range(n_objects):
        for _attempt in range(1000):
            c = rng.uniform(-half, half)
            if all(np.linalg.norm(c - o) >= 2 * r_obj for o in centers):
                centers.append(c)
                break
        else:
            raise RuntimeError(f"{n_objects}개를 겹치지 않게 놓지 못함 (seed={seed})")
    flipped = [bool(rng.random() < flip_prob) for _ in centers]
    T_cam_objs = [compose(T_cam_floor, _rest_pose(s, f, float(rng.uniform(0, 360)), c))
                  for f, c in zip(flipped, centers)]
    floor = make_bin_floor(floor_size_xy)
    depth, inst = render_depth([(floor, T_cam_floor)] + [(mesh, T) for T in T_cam_objs], intr)
    valid = depth > 0
    if noise_sigma_m > 0:
        depth = np.where(valid, depth + rng.normal(0.0, noise_sigma_m, depth.shape), 0.0)
    if dropout > 0:
        drop = valid & (rng.random(depth.shape) < dropout)
        depth = np.where(drop, 0.0, depth)
    return SimpleScene(depth, intr, T_cam_objs, [mesh] * n_objects, seed, inst, T_cam_floor, flipped)
