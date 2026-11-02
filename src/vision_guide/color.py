"""v4.1 색 처리: 모델 정점 색 기준 분포, 클러스터 관측 색 분포, 유사도, 후보 순위, 색 일치 검사.

규약
- HSV는 OpenCV uint8 규약: H ∈ [0, 180) (2°/단위), S·V ∈ [0, 255].
- 히스토그램 = [유채색 nh×ns | 무채색 na], 전체 합 1. 유채색은 S ≥ s_min, 무채색은 나머지.
  H 구간 0은 hue 0을 중심으로 둔다 (idx = floor(((h + 90/nh) mod 180) / (180/nh))) → 빨강이 두 구간으로 갈리지 않는다.
  무채색은 V를 na 구간으로 나눈다 (na = 1이면 V를 쓰지 않음, 조명 영향 축소).
- H 원형 가우시안 평활 σ(구간 단위)는 S 행마다 원형으로 적용한다 (σ = 0이면 평활 없음).
- 기준 분포는 **모델 정점 색에서만** 만든다 (dev 영상에서 색을 학습하지 않는다, PLAN_v4 §4.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .geometry import CameraIntrinsics, project


@dataclass(frozen=True)
class HistSpec:
    h_bins: int = 18
    s_bins: int = 2
    s_min: int = 40
    achrom_v_bins: int = 1
    h_smooth_sigma: float = 1.0

    @classmethod
    def from_cfg(cls, c: Mapping[str, Any]) -> "HistSpec":
        return cls(int(c["h_bins"]), int(c["s_bins"]), int(c["s_min"]), int(c["achrom_v_bins"]),
                   float(c["h_smooth_sigma"]))

    @property
    def size(self) -> int:
        return self.h_bins * self.s_bins + self.achrom_v_bins


def rgb_to_hsv_u8(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (..., 3) → HSV uint8 같은 모양 (OpenCV, H ∈ [0,180))."""
    a = np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)
    shp = a.shape
    if a.size == 0:
        return a.copy()
    out = cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    return out.reshape(shp)


def _h_index(h: np.ndarray, nh: int) -> np.ndarray:
    w = 180.0 / nh
    return (np.floor(np.mod(h.astype(float) + w / 2.0, 180.0) / w).astype(np.int64)) % nh


def _circular_smooth(rows: np.ndarray, sigma: float) -> np.ndarray:
    """(ns, nh) 각 행을 H 방향 원형 가우시안으로 평활 (합 보존)."""
    if sigma <= 0:
        return rows
    r = int(np.ceil(3 * sigma))
    offs = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (offs / sigma) ** 2)
    k /= k.sum()
    out = np.zeros_like(rows)
    for o, w in zip(offs, k):
        out += w * np.roll(rows, o, axis=1)
    return out


def hs_histogram(hsv: np.ndarray, spec: HistSpec, weights: np.ndarray | None = None) -> np.ndarray:
    """HSV 점 (N,3) → 정규화 히스토그램 (spec.size,). 점이 없으면 0 벡터."""
    hsv = np.asarray(hsv).reshape(-1, 3)
    w = np.ones(len(hsv)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    nh, ns, na = spec.h_bins, spec.s_bins, spec.achrom_v_bins
    h, s, v = hsv[:, 0].astype(float), hsv[:, 1].astype(float), hsv[:, 2].astype(float)
    chrom = s >= spec.s_min
    rows = np.zeros((ns, nh))
    if chrom.any():
        hi = _h_index(h[chrom], nh)
        sw = (256.0 - spec.s_min) / ns
        si = np.clip(np.floor((s[chrom] - spec.s_min) / sw).astype(np.int64), 0, ns - 1)
        np.add.at(rows, (si, hi), w[chrom])
    rows = _circular_smooth(rows, spec.h_smooth_sigma)
    ach = np.zeros(na)
    if (~chrom).any():
        vi = np.clip(np.floor(v[~chrom] / (256.0 / na)).astype(np.int64), 0, na - 1)
        np.add.at(ach, vi, w[~chrom])
    hist = np.concatenate([rows.ravel(), ach])
    tot = hist.sum()
    return hist / tot if tot > 0 else hist


def sample_mesh_colors(mesh_colored, n_samples: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """표면 면적 비례 샘플 점(물체 좌표)과 정점 색 무게중심 보간 RGB uint8."""
    import trimesh
    pts, fidx = trimesh.sample.sample_surface(mesh_colored, int(n_samples), seed=int(seed))
    tri = mesh_colored.triangles[fidx]
    bary = trimesh.triangles.points_to_barycentric(tri, pts)
    vc = np.asarray(mesh_colored.visual.vertex_colors)[:, :3].astype(float)
    cols = np.einsum("ij,ijk->ik", bary, vc[mesh_colored.faces[fidx]])
    return np.asarray(pts), np.clip(np.round(cols), 0, 255).astype(np.uint8)


def model_color_hist(mesh_colored, spec: HistSpec, n_samples: int = 20000, seed: int = 0) -> np.ndarray:
    """모델 정점 색 기준 분포 (시드 고정)."""
    _, cols = sample_mesh_colors(mesh_colored, n_samples, seed)
    return hs_histogram(rgb_to_hsv_u8(cols), spec)


def similarity(p: np.ndarray, q: np.ndarray, kind: str = "bhattacharyya") -> float:
    """bhattacharyya = Σ√(pq) (1 = 같음), intersection = Σmin(p, q)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if kind == "bhattacharyya":
        return float(np.sum(np.sqrt(p * q)))
    if kind == "intersection":
        return float(np.sum(np.minimum(p, q)))
    raise ValueError(f"unknown similarity: {kind!r}")


def pixels_of_points(points_cam: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """카메라 좌표 점 → 픽셀 (v, u) (floor(project), 영상 안, 중복 제거, 정렬)."""
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    p = p[p[:, 2] > 0]
    if len(p) == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    uv, _ = project(p, intr)
    u = np.floor(uv[:, 0]).astype(np.int64)
    v = np.floor(uv[:, 1]).astype(np.int64)
    ok = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    pix = np.unique(v[ok] * intr.width + u[ok])
    return pix // intr.width, pix % intr.width


def cluster_hsv(cluster_pcd, hsv_img: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """클러스터 점이 덮는 픽셀의 HSV (n_px, 3)."""
    v, u = pixels_of_points(np.asarray(cluster_pcd.points), intr)
    return hsv_img[v, u]


def cluster_color_hist(cluster_pcd, rgb: np.ndarray, intr: CameraIntrinsics, spec: HistSpec,
                       hsv_img: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """(관측 히스토그램, 픽셀 수). hsv_img를 주면 RGB 변환을 다시 하지 않는다."""
    hsv_img = rgb_to_hsv_u8(rgb) if hsv_img is None else hsv_img
    px = cluster_hsv(cluster_pcd, hsv_img, intr)
    return hs_histogram(px, spec), int(len(px))


def rank_by_similarity(sims: Sequence[float]) -> list[tuple[int, float]]:
    """유사도 내림차순 (동률은 원래 순번 오름차순 = z순)."""
    order = sorted(range(len(sims)), key=lambda i: (-float(sims[i]), i))
    return [(i, float(sims[i])) for i in order]


def rank_clusters(clusters: Sequence, rgb: np.ndarray, intr: CameraIntrinsics, ref_hist: np.ndarray,
                  spec: HistSpec, kind: str = "bhattacharyya", min_px: int = 30,
                  hsv_img: np.ndarray | None = None) -> list[tuple[int, float]]:
    """클러스터를 기준 색 분포 유사도 내림차순으로 (원래 idx, sim). 픽셀 < min_px면 sim = 0."""
    hsv_img = rgb_to_hsv_u8(rgb) if hsv_img is None else hsv_img
    sims = []
    for c in clusters:
        px = cluster_hsv(c, hsv_img, intr)
        sims.append(similarity(hs_histogram(px, spec), ref_hist, kind) if len(px) >= min_px else 0.0)
    return rank_by_similarity(sims)


def _circ_dh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a.astype(float) - b.astype(float)) % 180.0
    return np.minimum(d, 180.0 - d)


def color_match_ratio(model_rgb: np.ndarray, obs_rgb: np.ndarray, cfg: Mapping[str, Any]
                      ) -> tuple[float, int]:
    """대응 픽셀 쌍의 색 일치 비율과 비교 픽셀 수.

    모델 유채색(S ≥ s_min_model): 관측 S ≥ s_min_obs 이고 원형 |ΔH| ≤ h_tol (OpenCV H 단위) 이면 일치.
    모델 무채색: 관측 S ≤ achrom_s_max_obs 이면 일치. 비교 픽셀이 없으면 (nan, 0).
    """
    m = rgb_to_hsv_u8(np.asarray(model_rgb, dtype=np.uint8).reshape(-1, 3))
    o = rgb_to_hsv_u8(np.asarray(obs_rgb, dtype=np.uint8).reshape(-1, 3))
    n = len(m)
    if n == 0:
        return float("nan"), 0
    chrom = m[:, 1] >= int(cfg["s_min_model"])
    ok = np.where(chrom, (o[:, 1] >= int(cfg["s_min_obs"])) & (_circ_dh(m[:, 0], o[:, 0]) <= float(cfg["h_tol"])),
                  o[:, 1] <= int(cfg["achrom_s_max_obs"]))
    return float(ok.mean()), int(n)
