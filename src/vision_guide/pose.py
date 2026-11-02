"""자세 추정: FPFH + RANSAC 초기 정렬 → ICP(point-to-plane) → 신뢰도 → 대칭 후보.

정합 방향: source = 관측 클러스터, target = 모델 점(정확한 메시 법선) → 결과 T_obj_cam의 역이 T_cam_obj.
이렇게 하면 ICP fitness가 "관측 점 중 모델에 맞는 비율"이 된다(모델의 안 보이는 면에 영향받지 않음).

난수: ``o3d.utility.random.seed(seed)``로 고정한다. Open3D RANSAC은 멀티스레드라 스레드 스케줄에 따라
결과가 미세하게 달라질 수 있다(알려진 한계, docs/v3/model_spec.md).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d

from .failures import FailureCode
from .geometry import CameraIntrinsics, depth_to_points, inv_T, project, transform_points
from .segment import SegmentResult, preprocess, segment_scene, to_pcd

REG = o3d.pipelines.registration
TIMING_KEYS = ("backproject", "preprocess", "plane", "cluster", "fpfh", "ransac", "icp", "confidence", "symmetry")


@dataclass
class ObjectModel:
    """모델 점(물체 좌표, m), 법선, 직경(m), 대칭 4x4 목록(항등 포함), 대칭축(없으면 None)."""

    name: str
    points: np.ndarray
    normals: np.ndarray
    diameter: float
    symmetries: list[np.ndarray] = field(default_factory=lambda: [np.eye(4)])
    sym_axis: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_mesh(cls, name: str, mesh_or_parts, n_points: int = 6000, *, symmetries=None, sym_axis=None,
                  seed: int = 0) -> "ObjectModel":
        from .cad import model_diameter, sample_model_points

        pts, nrm = sample_model_points(mesh_or_parts, n_points, seed=seed)
        mesh = mesh_or_parts if not isinstance(mesh_or_parts, (list, tuple)) else None
        diam = model_diameter(mesh) if mesh is not None else model_diameter(pts)
        return cls(name, pts, nrm, diam, list(symmetries) if symmetries is not None else [np.eye(4)],
                   None if sym_axis is None else np.asarray(sym_axis, dtype=float))

    def features(self, cfg: Mapping[str, Any]) -> tuple[o3d.geometry.PointCloud, Any]:
        """voxel 다운샘플 모델 + FPFH (voxel 크기별 캐시)."""
        voxel = float(cfg["preprocess"]["voxel_m"])
        key = (voxel, float(cfg["fpfh"]["radius_factor"]), int(cfg["fpfh"]["max_nn"]))
        if key not in self._cache:
            pcd = to_pcd(self.points, self.normals).voxel_down_sample(voxel)
            n = np.asarray(pcd.normals)
            pcd.normals = o3d.utility.Vector3dVector(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
            self._cache[key] = (pcd, compute_fpfh(pcd, cfg))
        return self._cache[key]


@dataclass
class PoseResult:
    T_cam_obj: np.ndarray | None
    fitness: float
    rmse: float
    visible_ratio: float
    code: FailureCode
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: list[np.ndarray] = field(default_factory=list)
    n_cluster_points: int = 0

    @property
    def score(self) -> float:
        """후보 비교 점수 = fitness × 가시 비율 (가시 비율이 없으면 fitness)."""
        if self.T_cam_obj is None:
            return -1.0
        vis = self.visible_ratio if np.isfinite(self.visible_ratio) else 1.0
        return float(self.fitness * vis)

    @property
    def total_ms(self) -> float:
        return float(sum(self.timings_ms.values()))


def compute_fpfh(pcd: o3d.geometry.PointCloud, cfg: Mapping[str, Any]):
    voxel = float(cfg["preprocess"]["voxel_m"])
    return REG.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel * float(cfg["fpfh"]["radius_factor"]), max_nn=int(cfg["fpfh"]["max_nn"])))


def global_register(src: o3d.geometry.PointCloud, src_f, tgt: o3d.geometry.PointCloud, tgt_f,
                    cfg: Mapping[str, Any]):
    """FPFH 특징 대응 + RANSAC (point-to-point)."""
    rc = cfg["ransac"]
    dist = float(cfg["preprocess"]["voxel_m"]) * float(rc["distance_factor"])
    return REG.registration_ransac_based_on_feature_matching(
        src, tgt, src_f, tgt_f, bool(rc["mutual_filter"]), dist, REG.TransformationEstimationPointToPoint(False),
        int(rc["ransac_n"]),
        [REG.CorrespondenceCheckerBasedOnEdgeLength(float(rc["edge_length"])),
         REG.CorrespondenceCheckerBasedOnDistance(dist)],
        REG.RANSACConvergenceCriteria(int(rc["max_iteration"]), float(rc["confidence"])))


def refine_icp(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud, init: np.ndarray,
               cfg: Mapping[str, Any]) -> tuple[np.ndarray, float, float]:
    """거친 → 정밀 2단 point-to-plane ICP. (T_tgt_src, fitness, inlier RMSE) — fitness·RMSE는 정밀 단계 기준."""
    ic = cfg["icp"]
    est = REG.TransformationEstimationPointToPlane()
    r1 = REG.registration_icp(src, tgt, float(ic["coarse_max_corr_m"]), init, est,
                              REG.ICPConvergenceCriteria(max_iteration=int(ic["coarse_max_iter"])))
    r2 = REG.registration_icp(src, tgt, float(ic["fine_max_corr_m"]), r1.transformation, est,
                              REG.ICPConvergenceCriteria(max_iteration=int(ic["fine_max_iter"])))
    return np.asarray(r2.transformation), float(r2.fitness), float(r2.inlier_rmse)


def visible_ratio(T_cam_obj: np.ndarray, model: ObjectModel, depth_m: np.ndarray, intr: CameraIntrinsics,
                  vis_tol_m: float) -> float:
    """추정 자세에서 카메라를 향한 모델 점을 z-버퍼로 투영 → 관측 depth가 유효한 픽셀 중 |d − z| < tol 비율."""
    p = transform_points(T_cam_obj, model.points)
    n = model.normals @ T_cam_obj[:3, :3].T
    facing = np.einsum("ij,ij->i", n, p) < 0
    p = p[facing & (p[:, 2] > 0)]
    if len(p) == 0:
        return 0.0
    uv, z = project(p, intr)
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    u, v, z = u[inside], v[inside], z[inside]
    if len(z) == 0:
        return 0.0
    pix = v * intr.width + u
    order = np.lexsort((z, pix))
    pix, z = pix[order], z[order]
    first = np.ones(len(pix), dtype=bool)
    first[1:] = pix[1:] != pix[:-1]
    pix, z = pix[first], z[first]
    d = depth_m.reshape(-1)[pix]
    valid = d > 0
    if not valid.any():
        return 0.0
    return float(np.mean(np.abs(d[valid] - z[valid]) < vis_tol_m))


def symmetric_candidates(T_cam_obj: np.ndarray, symmetries: Sequence[np.ndarray]) -> list[np.ndarray]:
    """대칭으로 구분할 수 없는 자세 집합 {T · S}."""
    return [T_cam_obj @ S for S in symmetries]


def estimate_pose(cluster_pcd: o3d.geometry.PointCloud, model: ObjectModel, cfg: Mapping[str, Any], *,
                  depth_m: np.ndarray | None = None, intr: CameraIntrinsics | None = None,
                  seed: int = 0) -> PoseResult:
    t: dict[str, float] = {}
    n_pts = len(cluster_pcd.points)
    if n_pts < int(cfg["ransac"]["min_points"]):
        return PoseResult(None, 0.0, float("nan"), float("nan"), FailureCode.FEATURE_FAIL, t, [], n_pts)
    if not cluster_pcd.has_normals():
        pp = cfg["preprocess"]
        cluster_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=pp["normal_radius_m"],
                                                                          max_nn=int(pp["normal_max_nn"])))
        cluster_pcd.orient_normals_towards_camera_location(np.zeros(3))
    t0 = time.perf_counter()
    src_f = compute_fpfh(cluster_pcd, cfg)
    tgt, tgt_f = model.features(cfg)
    t["fpfh"] = (time.perf_counter() - t0) * 1e3
    cc = cfg["confidence"]
    use_vis = depth_m is not None and intr is not None
    best = None
    t["ransac"] = 0.0
    t["icp"] = 0.0
    t["confidence"] = 0.0
    for r in range(int(cfg["ransac"].get("n_restarts", 1))):
        o3d.utility.random.seed(int(seed) * 1000 + r)
        t0 = time.perf_counter()
        g = global_register(cluster_pcd, src_f, tgt, tgt_f, cfg)
        t["ransac"] += (time.perf_counter() - t0) * 1e3
        if len(g.correspondence_set) < 3:
            continue
        t0 = time.perf_counter()
        T, fit, rmse = refine_icp(cluster_pcd, tgt, np.asarray(g.transformation), cfg)
        t["icp"] += (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        T_cam_obj = inv_T(T)
        vis = visible_ratio(T_cam_obj, model, depth_m, intr, float(cc["vis_tol_m"])) if use_vis else float("nan")
        t["confidence"] += (time.perf_counter() - t0) * 1e3
        # 재시작 간 선택: fitness × 가시 비율 (평면 원판처럼 모양이 모호한 관측에서 잘못된 면 정합을 거른다)
        key = (fit * (vis if use_vis else 1.0), -rmse)
        if best is None or key > best[0]:
            best = (key, T_cam_obj, fit, rmse, vis)
    if best is None:
        return PoseResult(None, 0.0, float("nan"), float("nan"), FailureCode.FEATURE_FAIL, t, [], n_pts)
    _, T_cam_obj, fit, rmse, vis = best
    t0 = time.perf_counter()
    if fit < cc["min_fitness"]:
        code = FailureCode.LOW_FITNESS
    elif rmse > cc["max_rmse_m"]:
        code = FailureCode.HIGH_RMSE
    elif np.isfinite(vis) and vis < cc["min_visible_ratio"]:
        code = FailureCode.LOW_VISIBILITY
    else:
        code = FailureCode.OK
    t["confidence"] += (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    cands = symmetric_candidates(T_cam_obj, model.symmetries)
    t["symmetry"] = (time.perf_counter() - t0) * 1e3
    return PoseResult(T_cam_obj, fit, rmse, vis, code, t, cands, n_pts)


def pick_best(results: Sequence[PoseResult]) -> PoseResult:
    """채택(OK) 중 점수 최고, 없으면 전체 중 점수 최고 (그 코드로 실패 귀속)."""
    ok = [r for r in results if r.code == FailureCode.OK]
    pool = ok or list(results)
    return max(pool, key=lambda r: r.score)


def run_pipeline(depth_m: np.ndarray, intr: CameraIntrinsics, model: ObjectModel, cfg: Mapping[str, Any], *,
                 seed: int = 0, roi_mask: np.ndarray | None = None, max_candidates: int | None = None,
                 seg: str = "dbscan", segmentation: SegmentResult | None = None,
                 stop_at_first_ok: bool = False) -> tuple[list[PoseResult], SegmentResult | None]:
    """Depth → 자세 후보 목록 (위쪽 클러스터부터) 과 분할 결과.

    seg="dbscan": 평면 제거 + DBSCAN 클러스터 각각에 정합.
    seg="gt_mask": roi_mask(정답 가시 마스크) 점 전체를 클러스터 하나로 본다 (정합 상한 측정용).
    segmentation을 주면 역투영·분할을 다시 하지 않는다 (같은 프레임 여러 대상 평가용; 그 시간은 결과에 더한다).
    """
    o3d.utility.random.seed(int(seed))
    base_t: dict[str, float] = {}
    if segmentation is None:
        t0 = time.perf_counter()
        pts, _ = depth_to_points(depth_m, intr, roi_mask)
        base_t["backproject"] = (time.perf_counter() - t0) * 1e3
        if len(pts) == 0:
            return [PoseResult(None, 0.0, float("nan"), float("nan"), FailureCode.NO_POINTS, base_t)], None
        if seg == "gt_mask":
            t0 = time.perf_counter()
            pcd = preprocess(pts, cfg)
            base_t["preprocess"] = (time.perf_counter() - t0) * 1e3
            segmentation = SegmentResult([pcd] if len(pcd.points) else [], None, len(pts), {}, 1)
        elif seg == "dbscan":
            segmentation = segment_scene(pts, cfg, model.diameter)
        else:
            raise ValueError(f"seg must be 'dbscan' or 'gt_mask': {seg!r}")
        base_t.update(segmentation.timings_ms)
    else:
        base_t.update(segmentation.timings_ms)
    clusters = segmentation.clusters
    if not clusters:
        code = FailureCode.NO_CLUSTER if segmentation.n_input > 0 else FailureCode.NO_POINTS
        return [PoseResult(None, 0.0, float("nan"), float("nan"), code, base_t)], segmentation
    k = max_candidates if max_candidates is not None else int(cfg["cluster"]["max_candidates"])
    results = []
    for i, c in enumerate(clusters[:k]):
        r = estimate_pose(c, model, cfg, depth_m=depth_m, intr=intr, seed=seed + i)
        r.timings_ms = {**base_t, **r.timings_ms} if i == 0 else r.timings_ms
        results.append(r)
        if stop_at_first_ok and r.code == FailureCode.OK:
            break
    return results, segmentation
