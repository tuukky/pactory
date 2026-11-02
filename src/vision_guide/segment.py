"""전처리와 분할: ROI → voxel 다운샘플 → 이상점 제거 → 법선 → 평면 제거(RANSAC) → DBSCAN → 크기 필터."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import open3d as o3d


@dataclass
class SegmentResult:
    clusters: list[o3d.geometry.PointCloud]
    plane_model: np.ndarray | None
    n_input: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    n_raw_clusters: int = 0


def to_pcd(points: np.ndarray, normals: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=float).reshape(-1, 3)))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=float).reshape(-1, 3))
    return pcd


def preprocess(pts: np.ndarray, cfg: Mapping[str, Any]) -> o3d.geometry.PointCloud:
    """ROI(깊이 범위) → voxel → statistical outlier → 법선(카메라 원점 쪽으로 정렬)."""
    roi, pp = cfg["roi"], cfg["preprocess"]
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    pts = pts[(pts[:, 2] >= roi["z_min_m"]) & (pts[:, 2] <= roi["z_max_m"])]
    pcd = to_pcd(pts).voxel_down_sample(pp["voxel_m"])
    if len(pcd.points) > pp["outlier_nb"]:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=int(pp["outlier_nb"]), std_ratio=float(pp["outlier_std"]))
    if len(pcd.points) >= 3:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=pp["normal_radius_m"],
                                                                  max_nn=int(pp["normal_max_nn"])))
        pcd.orient_normals_towards_camera_location(np.zeros(3))
    return pcd


def remove_plane(pcd: o3d.geometry.PointCloud, cfg: Mapping[str, Any]
                 ) -> tuple[o3d.geometry.PointCloud, np.ndarray | None]:
    """가장 큰 평면이 min_inlier_frac 이상이면 제거한다. (남은 점, 평면 계수 또는 None)."""
    pc = cfg["plane"]
    n = len(pcd.points)
    if not pc.get("enabled", True) or n < 3:
        return pcd, None
    model, inliers = pcd.segment_plane(distance_threshold=pc["distance_m"], ransac_n=int(pc["ransac_n"]),
                                       num_iterations=int(pc["iterations"]))
    if len(inliers) < pc["min_inlier_frac"] * n:
        return pcd, None
    return pcd.select_by_index(inliers, invert=True), np.asarray(model)


def filter_by_extent(clusters: list[o3d.geometry.PointCloud], cfg: Mapping[str, Any], diameter: float | None
                     ) -> list[o3d.geometry.PointCloud]:
    """bbox 대각이 [min_extent_frac, max_extent_frac] × 모델 직경 안인 클러스터만 남긴다."""
    if diameter is None:
        return list(clusters)
    cc = cfg["cluster"]
    out = []
    for c in clusters:
        ext = float(np.linalg.norm(c.get_max_bound() - c.get_min_bound()))
        if cc["min_extent_frac"] * diameter <= ext <= cc["max_extent_frac"] * diameter:
            out.append(c)
    return out


def cluster(pcd: o3d.geometry.PointCloud, cfg: Mapping[str, Any], diameter: float | None = None
            ) -> tuple[list[o3d.geometry.PointCloud], int]:
    """DBSCAN + 점 수·크기 필터. 중심 z 오름차순(카메라에 가까운 = 위쪽 먼저). (클러스터, 필터 전 개수)."""
    cc = cfg["cluster"]
    if len(pcd.points) == 0:
        return [], 0
    labels = np.asarray(pcd.cluster_dbscan(eps=cc["eps_m"], min_points=int(cc["min_points"])))
    n_raw = int(labels.max() + 1) if len(labels) else 0
    out = []
    for k in range(n_raw):
        idx = np.nonzero(labels == k)[0]
        if len(idx) >= cc["min_cluster_points"]:
            out.append(pcd.select_by_index(idx))
    out = filter_by_extent(out, cfg, diameter)
    out.sort(key=lambda c: float(np.asarray(c.points)[:, 2].mean()))
    return out, n_raw


def segment_scene(points_cam: np.ndarray, cfg: Mapping[str, Any], diameter: float | None = None) -> SegmentResult:
    t = {}
    t0 = time.perf_counter()
    pcd = preprocess(points_cam, cfg)
    t["preprocess"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    rest, plane = remove_plane(pcd, cfg)
    t["plane"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    clusters, n_raw = cluster(rest, cfg, diameter)
    t["cluster"] = (time.perf_counter() - t0) * 1e3
    return SegmentResult(clusters, plane, len(points_cam), t, n_raw)
