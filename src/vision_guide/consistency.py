"""v4.1 신뢰도 검사 강화: 추정 자세 렌더링(깊이·색) → 깊이 일관성(자유공간 위반), 가시 비율 완화, 색 일치.

렌더링은 v3 ``cad.render_depth``와 같은 RaycastingScene 경로다. 다만 메시를 자세마다 다시 넣지 않도록
물체 좌표 메시로 장면을 한 번 만들고 광선을 물체 좌표로 옮긴다 (광선 방향 z_cam = 1이라 t_hit = z-깊이).

깊이 판정 (모델이 맞은 픽셀 중 관측 d > 0):
- match:     |d − z| < tol_match
- occluded:  d < z − tol_match (모델 앞에 다른 것이 있음 = 가림으로 설명됨)
- violation: d > z + tol_free (관측이 모델보다 뒤 = 모델이 빈 공간에 떠 있음)
violation_ratio = violation / valid. valid < min_valid_px면 판정하지 않는다 (nan).

실패 귀속 순서 (첫 실패 하나): LOW_FITNESS → HIGH_RMSE → LOW_VISIBILITY(완화 후) → DEPTH_INCONSISTENT → COLOR_MISMATCH.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d

from .color import color_match_ratio
from .failures import FailureCode
from .geometry import CameraIntrinsics, normalized_rays, project, transform_points


@dataclass
class RenderResult:
    depth: np.ndarray               # bbox 영역 z-깊이 (m, 미검출 0)
    colors: np.ndarray | None       # bbox 영역 RGB uint8 (미검출 0), 정점 색이 없으면 None
    bbox: tuple[int, int, int, int]  # (v0, v1, u0, u1) 반열림 구간

    @property
    def empty(self) -> bool:
        return self.depth.size == 0


class ModelRaycaster:
    """물체 좌표 메시(m)의 RaycastingScene (1회 생성·캐시)."""

    def __init__(self, mesh) -> None:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.uint32)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(f))
        vis = getattr(mesh, "visual", None)
        vc = getattr(vis, "vertex_colors", None) if vis is not None and getattr(vis, "kind", None) == "vertex" else None
        self.vertex_colors = None if vc is None else np.asarray(vc)[:, :3].astype(float)
        lo, hi = np.asarray(mesh.bounds, dtype=float)
        self.corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    def _bbox(self, T_cam_obj: np.ndarray, intr: CameraIntrinsics, pad_px: int) -> tuple[int, int, int, int]:
        c = transform_points(T_cam_obj, self.corners)
        if np.all(c[:, 2] <= 0):
            return 0, 0, 0, 0
        if np.any(c[:, 2] <= 1e-6):  # 카메라 평면을 가로지르면 전체 영상
            return 0, intr.height, 0, intr.width
        uv, _ = project(c, intr)
        u0 = int(np.clip(np.floor(uv[:, 0].min()) - pad_px, 0, intr.width))
        u1 = int(np.clip(np.ceil(uv[:, 0].max()) + pad_px + 1, 0, intr.width))
        v0 = int(np.clip(np.floor(uv[:, 1].min()) - pad_px, 0, intr.height))
        v1 = int(np.clip(np.ceil(uv[:, 1].max()) + pad_px + 1, 0, intr.height))
        return v0, v1, u0, u1

    def render(self, T_cam_obj: np.ndarray, intr: CameraIntrinsics, pad_px: int = 2) -> RenderResult:
        v0, v1, u0, u1 = self._bbox(T_cam_obj, intr, pad_px)
        h, w = max(v1 - v0, 0), max(u1 - u0, 0)
        if h == 0 or w == 0:
            return RenderResult(np.zeros((0, 0)), None, (v0, v1, u0, u1))
        vv, uu = np.mgrid[v0:v1, u0:u1]
        uv = np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5], axis=1).astype(float)
        d_cam = normalized_rays(uv, intr)                      # z = 1
        R, t = T_cam_obj[:3, :3], T_cam_obj[:3, 3]
        o_obj = -R.T @ t
        d_obj = d_cam @ R                                      # = (Rᵀ d)ᵀ
        rays = np.concatenate([np.broadcast_to(o_obj, d_obj.shape), d_obj], axis=1).astype(np.float32)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        th = ans["t_hit"].numpy().astype(float)
        hit = np.isfinite(th)
        depth = np.where(hit, th, 0.0).reshape(h, w)          # t_hit × d_z(=1) = z
        colors = None
        if self.vertex_colors is not None:
            cols = np.zeros((len(th), 3))
            if hit.any():
                pid = ans["primitive_ids"].numpy().astype(np.int64)[hit]
                bu = ans["primitive_uvs"].numpy().astype(float)[hit]
                f = self.faces[pid]
                vc = self.vertex_colors
                cols[hit] = ((1.0 - bu[:, :1] - bu[:, 1:2]) * vc[f[:, 0]] + bu[:, :1] * vc[f[:, 1]]
                             + bu[:, 1:2] * vc[f[:, 2]])
            colors = np.clip(np.round(cols), 0, 255).astype(np.uint8).reshape(h, w, 3)
        return RenderResult(depth, colors, (v0, v1, u0, u1))


@dataclass
class DepthCheck:
    n_valid: int
    n_match: int
    n_occluded: int
    n_violation: int
    violation_ratio: float
    explained_ratio: float
    match_ratio: float


def _crop(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    v0, v1, u0, u1 = bbox
    return img[v0:v1, u0:u1]


def depth_consistency(r: RenderResult, depth_obs: np.ndarray, tol_match_m: float, tol_free_m: float
                      ) -> DepthCheck:
    """렌더 depth vs 관측 depth (모델이 맞고 관측이 유효한 픽셀)."""
    if r.empty:
        return DepthCheck(0, 0, 0, 0, float("nan"), float("nan"), float("nan"))
    z = r.depth
    d = _crop(depth_obs, r.bbox)
    valid = (z > 0) & (d > 0)
    n = int(valid.sum())
    if n == 0:
        return DepthCheck(0, 0, 0, 0, float("nan"), float("nan"), float("nan"))
    zv, dv = z[valid], d[valid]
    match = np.abs(dv - zv) < tol_match_m
    occl = dv < zv - tol_match_m
    viol = dv > zv + tol_free_m
    nm, no, nv = int(match.sum()), int(occl.sum()), int(viol.sum())
    return DepthCheck(n, nm, no, nv, nv / n, (nm + no) / n, nm / n)


def match_pixels(r: RenderResult, depth_obs: np.ndarray, tol_match_m: float) -> tuple[np.ndarray, np.ndarray]:
    """깊이 match 픽셀의 (v, u) 절대 좌표."""
    if r.empty:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    d = _crop(depth_obs, r.bbox)
    m = (r.depth > 0) & (d > 0) & (np.abs(d - r.depth) < tol_match_m)
    vv, uu = np.nonzero(m)
    return vv + r.bbox[0], uu + r.bbox[2]


def compute_diag(T_cam_obj: np.ndarray, raycaster: ModelRaycaster, depth_m: np.ndarray, rgb: np.ndarray | None,
                 intr: CameraIntrinsics, checks: Mapping[str, Any], *,
                 tol_free_grid: Sequence[float] | None = None,
                 color_grid: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """검사에 필요한 원값 (게이트 미적용). grid를 주면 재생 조정용 변형도 함께 계산한다.

    키: n_valid, n_match, n_occluded, viol (설정 tol_free), color_ratio, color_n (설정 색 조건),
    viol@<tol>, color_ratio@<h_tol>_<achrom>, color_n@<h_tol>_<achrom>.
    """
    dc = checks["depth"]
    cm = checks["color_match"]
    r = raycaster.render(T_cam_obj, intr)
    tol_m = float(dc["tol_match_m"])
    base = depth_consistency(r, depth_m, tol_m, float(dc["tol_free_m"]))
    out = {"n_valid": base.n_valid, "n_match": base.n_match, "n_occluded": base.n_occluded,
           "viol": base.violation_ratio}
    for tf in (tol_free_grid or []):
        out[f"viol@{tf:g}"] = depth_consistency(r, depth_m, tol_m, float(tf)).violation_ratio
    mv, mu = match_pixels(r, depth_m, tol_m)
    have_color = rgb is not None and r.colors is not None
    model_rgb = r.colors[mv - r.bbox[0], mu - r.bbox[2]] if have_color and len(mv) else np.zeros((0, 3), np.uint8)
    obs_rgb = np.asarray(rgb)[mv, mu, :3] if have_color and len(mv) else np.zeros((0, 3), np.uint8)
    ratio, n = color_match_ratio(model_rgb, obs_rgb, cm) if have_color else (float("nan"), 0)
    out["color_ratio"], out["color_n"] = ratio, n
    for g in (color_grid or []):
        c = {**cm, **g}
        rr, nn = color_match_ratio(model_rgb, obs_rgb, c) if have_color else (float("nan"), 0)
        key = f"{int(c['h_tol'])}_{int(c['achrom_s_max_obs'])}"
        out[f"color_ratio@{key}"], out[f"color_n@{key}"] = rr, nn
    return out


def decide_code(v3_code: FailureCode, vis: float, diag: Mapping[str, Any], checks: Mapping[str, Any], *,
                viol_key: str = "viol", color_key: str = "color_ratio", color_n_key: str = "color_n"
                ) -> FailureCode:
    """v3 신뢰도 코드 + v4.1 검사 원값 → 최종 코드 (첫 실패 귀속). 재생 조정과 실시간이 같은 함수를 쓴다."""
    if v3_code not in (FailureCode.OK, FailureCode.LOW_VISIBILITY):
        return v3_code
    dc, cm, vr = checks["depth"], checks["color_match"], checks["visibility_relax"]
    viol = float(diag.get(viol_key, float("nan")))
    n_valid = int(diag.get("n_valid", 0))
    depth_judged = n_valid >= int(dc["min_valid_px"]) and np.isfinite(viol)
    if v3_code == FailureCode.LOW_VISIBILITY:
        relaxed = (bool(vr.get("enabled", False)) and depth_judged and viol <= float(dc["max_violation"])
                   and int(diag.get("n_match", 0)) >= int(vr["min_match_px"])
                   and np.isfinite(vis) and vis >= float(vr["min_vis_relaxed"]))
        if not relaxed:
            return FailureCode.LOW_VISIBILITY
    if depth_judged and viol > float(dc["max_violation"]):
        return FailureCode.DEPTH_INCONSISTENT
    ratio = float(diag.get(color_key, float("nan")))
    if int(diag.get(color_n_key, 0)) >= int(cm["min_px"]) and np.isfinite(ratio) and ratio < float(cm["min_match"]):
        return FailureCode.COLOR_MISMATCH
    return FailureCode.OK


def apply_v41_checks(pose, raycaster: ModelRaycaster, depth_m: np.ndarray, rgb: np.ndarray | None,
                     intr: CameraIntrinsics, cfg: Mapping[str, Any], *, compute_all: bool = False,
                     tol_free_grid: Sequence[float] | None = None,
                     color_grid: Sequence[Mapping[str, Any]] | None = None) -> tuple[FailureCode, dict]:
    """PoseResult(v3 코드) → (v4.1 최종 코드, 원값). compute_all이면 앞 단계 실패여도 원값을 계산한다."""
    if pose.T_cam_obj is None:
        return pose.code, {}
    if not compute_all and pose.code not in (FailureCode.OK, FailureCode.LOW_VISIBILITY):
        return pose.code, {}
    diag = compute_diag(pose.T_cam_obj, raycaster, depth_m, rgb, intr, cfg["checks"],
                        tol_free_grid=tol_free_grid, color_grid=color_grid)
    return decide_code(pose.code, pose.visible_ratio, diag, cfg["checks"]), diag
