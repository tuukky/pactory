"""캘리브레이션: 합성 체커보드 내부 파라미터, Hand-eye (eye-to-hand, AX=XB).

- 체커보드 렌더링: 픽셀 → 왜곡 보정 광선 → 보드 평면 교차 → 칸 색. 4x4 슈퍼샘플, 가우시안 밝기 노이즈, 흰 여백.
  OpenCV 코너 좌표(픽셀 중심 = 정수)는 ``CameraIntrinsics.from_opencv``로 이 모듈 규약(+0.5)으로 바꾼다.
- Hand-eye: 카메라 고정, 보드를 TCP에 고정(T_tcp_board 정답). cv2.calibrateHandEye의 gripper2base 자리에
  inv(T_base_tcp)를 넣으면 결과 cam2gripper가 T_base_cam이 된다:
  T_tcp_board = inv(T_base_tcp_i) · T_base_cam · T_cam_board_i (모든 i에서 일정).
- 노이즈 (a) 자세 직접: T_cam_board 측정에 축별 σ 가우시안(오른쪽 섭동). 로봇 자세는 정확하다고 가정.
  (b) 종단: 체커보드 렌더링 → 코너 검출 → solvePnP.
가상 로봇 자세·합성 영상이므로 실제 로봇 반복 정밀도·렌즈 특성을 반영하지 않는다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .geometry import (CameraIntrinsics, inv_T, make_T, normalized_rays, perturb_T, project, rotation_about,
                       rotation_angle_deg, rotation_between_deg, transform_points)

HANDEYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class Board:
    """내부 코너 cols x rows, 칸 크기 square_m. 보드 좌표 원점 = 첫 내부 코너, +z = 보드 뒤쪽."""

    cols: int
    rows: int
    square_m: float

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.cols - 1) * self.square_m / 2, (self.rows - 1) * self.square_m / 2, 0.0])


def board_points(board: Board) -> np.ndarray:
    """내부 코너 (N,3), 행 우선(x가 빠르게) — cv2.findChessboardCorners 순서와 같다."""
    jj, ii = np.mgrid[0:board.rows, 0:board.cols]
    return np.stack([ii.ravel() * board.square_m, jj.ravel() * board.square_m, np.zeros(ii.size)], axis=1)


def T_from_cfg(d: Mapping[str, Any]) -> np.ndarray:
    return make_T(Rotation.from_euler("xyz", d.get("rpy_deg", [0, 0, 0]), degrees=True).as_matrix(),
                  d.get("position_m", [0, 0, 0]))


# ---------------------------------------------------------------------------- 렌더링
@lru_cache(maxsize=4)
def _subpixel_rays(intr: CameraIntrinsics, ss: int) -> np.ndarray:
    """(H, W, ss*ss, 3) 서브픽셀 광선 (z=1). 왜곡 보정은 한 번만 계산해 캐시한다."""
    off = (np.arange(ss) + 0.5) / ss
    vv, uu = np.mgrid[0:intr.height, 0:intr.width]
    su = (uu[..., None, None] + off[None, None, None, :]).repeat(ss, axis=2)
    sv = (vv[..., None, None] + off[None, None, :, None]).repeat(ss, axis=3)
    uv = np.stack([su.ravel(), sv.ravel()], axis=1)
    return normalized_rays(uv, intr).reshape(intr.height, intr.width, ss * ss, 3).astype(np.float32)


def render_checkerboard(board: Board, intr: CameraIntrinsics, T_cam_board: np.ndarray, *, supersample: int = 4,
                        pixel_noise: float = 0.0, rng: np.random.Generator | None = None,
                        background: float = 128.0) -> np.ndarray:
    """8비트 회색 체커보드 영상. 체커 영역 바깥에 한 칸 폭 흰 여백, 그 밖은 회색 배경."""
    rays = _subpixel_rays(intr, supersample)
    R, t = T_cam_board[:3, :3].astype(np.float32), T_cam_board[:3, 3].astype(np.float32)
    n = R[:, 2]
    denom = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.dot(n, t) / denom
    P = rays * lam[..., None]
    q = (P - t) @ R  # 보드 좌표
    s = board.square_m
    x, y = q[..., 0], q[..., 1]
    in_front = np.isfinite(lam) & (lam > 0)
    checker = (x >= -s) & (x <= board.cols * s) & (y >= -s) & (y <= board.rows * s)
    margin = (x >= -2 * s) & (x <= (board.cols + 1) * s) & (y >= -2 * s) & (y <= (board.rows + 1) * s)
    black = ((np.floor(x / s) + np.floor(y / s)) % 2 == 0)
    val = np.full(x.shape, background)
    val = np.where(in_front & margin, 255.0, val)
    val = np.where(in_front & checker & black, 0.0, val)
    img = val.mean(axis=2)
    if pixel_noise > 0:
        rng = rng if rng is not None else np.random.default_rng()
        img = img + rng.normal(0.0, pixel_noise, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def sample_board_poses(n: int, board: Board, intr: CameraIntrinsics, rng: np.random.Generator, *,
                       distance_m: Sequence[float] = (0.35, 0.7), max_tilt_deg: float = 40.0,
                       max_roll_deg: float = 20.0, center_frac: float = 0.8, margin_px: float = 15.0
                       ) -> list[np.ndarray]:
    """카메라를 향한 보드 자세 T_cam_board n개 (모든 코너 + 여백이 영상 안). 보드 x ≈ 영상 +u."""
    out: list[np.ndarray] = []
    corners = board_points(board)
    s = board.square_m
    outline = np.array([[-2 * s, -2 * s, 0], [(board.cols + 1) * s, -2 * s, 0],
                        [-2 * s, (board.rows + 1) * s, 0], [(board.cols + 1) * s, (board.rows + 1) * s, 0]])
    for _ in range(100000):
        if len(out) == n:
            break
        roll = rotation_about([0, 0, 1], rng.uniform(-max_roll_deg, max_roll_deg))
        phi = rng.uniform(0, 2 * np.pi)
        tilt = rotation_about([np.cos(phi), np.sin(phi), 0.0], rng.uniform(0, max_tilt_deg))
        R = tilt @ roll
        d = rng.uniform(*distance_m)
        u = intr.cx + rng.uniform(-0.5, 0.5) * center_frac * intr.width
        v = intr.cy + rng.uniform(-0.5, 0.5) * center_frac * intr.height
        c_cam = np.array([(u - intr.cx) / intr.fx * d, (v - intr.cy) / intr.fy * d, d])
        T = make_T(R, c_cam - R @ board.center)
        uv, z = project(transform_points(T, np.vstack([corners, outline])), intr)
        if (z <= 0).any():
            continue
        if ((uv[:, 0] < margin_px) | (uv[:, 0] > intr.width - margin_px) | (uv[:, 1] < margin_px)
                | (uv[:, 1] > intr.height - margin_px)).any():
            continue
        out.append(T)
    if len(out) < n:
        raise RuntimeError("보드 자세를 충분히 뽑지 못했다")
    return out


# ---------------------------------------------------------------------------- 검출·내부 파라미터
def detect_corners(img: np.ndarray, board: Board) -> np.ndarray | None:
    """OpenCV 규약 코너 (N,2) 또는 None. 180° 순서 뒤집힘은 첫 행이 +u로 가도록 맞춘다."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCorners(img, (board.cols, board.rows), flags=flags)
    if not ok:
        return None
    c = cv2.cornerSubPix(img, c, (7, 7), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)).reshape(-1, 2)
    # 보드 자세 표본이 roll ±20° 이내라 보드 +x ≈ 영상 +u: 첫 행 방향이 -u면 뒤집는다
    # (TODO: 실제 보드에서는 ChArUco 등 방향이 정해진 패턴으로 대체)
    if c[board.cols - 1, 0] < c[0, 0]:
        c = c[::-1].copy()
    return c


def calibrate_intrinsics(images: Sequence[np.ndarray], board: Board, image_size: tuple[int, int]) -> dict:
    """Zhang 방법 (cv2.calibrateCamera, 왜곡 5계수). 결과 intr는 이 모듈 규약."""
    obj = board_points(board).astype(np.float32)
    objs, imgs = [], []
    for im in images:
        c = detect_corners(im, board)
        if c is not None:
            objs.append(obj)
            imgs.append(c.astype(np.float32).reshape(-1, 1, 2))
    if len(objs) < 3:
        raise RuntimeError(f"코너 검출 영상이 부족하다 ({len(objs)})")
    w, h = image_size
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, (w, h), None, None)
    per_view = []
    for o, i, rv, tv in zip(objs, imgs, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, rv, tv, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((p - i) ** 2, axis=2)))))
    intr = CameraIntrinsics.from_opencv(K, w, h, tuple(np.asarray(dist).ravel()[:5]))
    return {"rms_px": float(rms), "intr": intr, "K": intr.K(), "dist": np.asarray(dist).ravel(),
            "per_view_rms_px": per_view, "n_used": len(objs)}


def intrinsics_error(est: CameraIntrinsics, truth: CameraIntrinsics, grid: int = 20) -> dict:
    """fx·fy·cx·cy 오차(px)와 투영 지도 오차: 같은 광선을 두 모델로 투영한 픽셀 차이 (평균·최대)."""
    uu, vv = np.meshgrid(np.linspace(10, truth.width - 10, grid), np.linspace(10, truth.height - 10, grid))
    rays = normalized_rays(np.stack([uu.ravel(), vv.ravel()], axis=1), truth)
    uv_t, _ = project(rays, truth)
    uv_e, _ = project(rays, est)
    d = np.linalg.norm(uv_e - uv_t, axis=1)
    return {"d_fx": est.fx - truth.fx, "d_fy": est.fy - truth.fy, "d_cx": est.cx - truth.cx,
            "d_cy": est.cy - truth.cy, "map_err_mean_px": float(d.mean()), "map_err_max_px": float(d.max())}


# ---------------------------------------------------------------------------- Hand-eye
def handeye_settings(gcfg: Mapping[str, Any], T_base_cam: np.ndarray, intr: CameraIntrinsics | None = None) -> dict:
    """gripper.yaml calibration 절 → Hand-eye 데이터 생성 설정."""
    from .geometry import load_camera
    c = gcfg["calibration"]
    if intr is None:
        from pathlib import Path
        intr, _ = load_camera(Path(__file__).resolve().parents[2] / "configs" / "v3" / "camera.yaml")
    return {"T_base_cam": np.asarray(T_base_cam), "T_tcp_board": T_from_cfg(c["T_tcp_board"]),
            "board": Board(**c["board"]), "intr": intr, "sampling": dict(c["board_pose_sampling"])}


def generate_handeye_data(hs: Mapping[str, Any], n_poses: int, rng: np.random.Generator, *,
                          rot_noise_deg: float = 0.0, trans_noise_m: float = 0.0) -> dict:
    """가상 로봇 자세: 카메라를 향한 보드 자세를 뽑고 T_base_tcp = T_base_cam · T_cam_board · inv(T_tcp_board)."""
    sp = hs["sampling"]
    T_cam_board = sample_board_poses(n_poses, hs["board"], hs["intr"], rng, distance_m=sp["distance_m"],
                                     max_tilt_deg=sp["max_tilt_deg"], max_roll_deg=sp["max_roll_deg"],
                                     center_frac=sp["center_frac"])
    T_base_tcp = [hs["T_base_cam"] @ T @ inv_T(hs["T_tcp_board"]) for T in T_cam_board]
    meas = [perturb_T(T, rot_noise_deg, trans_noise_m, rng) for T in T_cam_board]
    return {"T_base_tcp": T_base_tcp, "T_cam_board_true": T_cam_board, "T_cam_board_meas": meas}


def calibrate_handeye(T_base_tcp_list: Sequence[np.ndarray], T_cam_board_list: Sequence[np.ndarray],
                      method: str = "PARK", precondition: bool = True) -> np.ndarray:
    """eye-to-hand: T_base_cam 추정.

    precondition: base 좌표를 첫 로봇 자세 방향으로 고정 회전한 base'(R_base'_tcp0 = I)에서 풀고 되돌린다
    (T_base'_tcp = W·T_base_tcp, 결과 W⁻¹·T_base'_cam). 수학적으로 같은 문제다. TCP가 아래를 보는 자세는
    gripper2base 자리 회전이 180° 근처라 OpenCV TSAI가 수치적으로 불안정했다 (verification.md).
    """
    W = make_T(np.asarray(T_base_tcp_list[0])[:3, :3].T) if precondition else np.eye(4)
    A = [inv_T(W @ T) for T in T_base_tcp_list]
    R, t = cv2.calibrateHandEye([a[:3, :3] for a in A], [a[:3, 3].reshape(3, 1) for a in A],
                                [b[:3, :3] for b in T_cam_board_list],
                                [b[:3, 3].reshape(3, 1) for b in T_cam_board_list], method=HANDEYE_METHODS[method])
    return inv_T(W) @ make_T(R, np.asarray(t).reshape(3))


def handeye_error(T_est: np.ndarray, T_true: np.ndarray) -> tuple[float, float]:
    """(병진 오차 mm, 회전 오차 °)."""
    return float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1e3), rotation_between_deg(T_est[:3, :3],
                                                                                            T_true[:3, :3])


def handeye_curve(hs: Mapping[str, Any], noise_levels: Sequence[Mapping[str, float]], pose_counts: Sequence[int],
                  n_trials: int, seed: int, methods: Sequence[str] = tuple(HANDEYE_METHODS)) -> pd.DataFrame:
    """노이즈 × 자세 수 × 방법 × 반복. 같은 (노이즈, 자세 수, 반복)의 데이터를 모든 방법이 공유한다 (CRN)."""
    rows = []
    for li, lv in enumerate(noise_levels):
        for n in pose_counts:
            for trial in range(n_trials):
                rng = np.random.default_rng([int(seed), li, int(n), trial])
                data = generate_handeye_data(hs, int(n), rng, rot_noise_deg=lv["rot_deg"], trans_noise_m=lv["trans_m"])
                for m in methods:
                    try:
                        T = calibrate_handeye(data["T_base_tcp"], data["T_cam_board_meas"], m)
                        e_mm, e_deg = handeye_error(T, hs["T_base_cam"])
                    except cv2.error:
                        e_mm, e_deg = float("nan"), float("nan")
                    rows.append({"noise_level": li, "rot_sigma_deg": lv["rot_deg"],
                                 "trans_sigma_mm": lv["trans_m"] * 1e3, "n_poses": int(n), "method": m,
                                 "trial": trial, "err_mm": e_mm, "err_deg": e_deg})
    return pd.DataFrame(rows)


def handeye_end_to_end(hs: Mapping[str, Any], intr: CameraIntrinsics, n_poses: int, n_trials: int, seed: int,
                       pixel_noise: float, methods: Sequence[str] = tuple(HANDEYE_METHODS)) -> pd.DataFrame:
    """체커보드 렌더링 → 코너 검출 → solvePnP(정답 내부 파라미터) → Hand-eye."""
    board = hs["board"]
    obj = board_points(board).astype(np.float32)
    rows = []
    for trial in range(n_trials):
        rng = np.random.default_rng([int(seed), 999, trial])
        data = generate_handeye_data(hs, n_poses, rng)
        A, B, pnp_rot, pnp_mm = [], [], [], []
        for T_tcp, T_true in zip(data["T_base_tcp"], data["T_cam_board_true"]):
            img = render_checkerboard(board, intr, T_true, pixel_noise=pixel_noise, rng=rng)
            c = detect_corners(img, board)
            if c is None:
                continue
            ok, rv, tv = cv2.solvePnP(obj, c.astype(np.float32), intr.K_opencv(), intr.dist_array(),
                                      flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            T_est = make_T(cv2.Rodrigues(rv)[0], tv.reshape(3))
            pnp_rot.append(rotation_between_deg(T_est[:3, :3], T_true[:3, :3]))
            pnp_mm.append(float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1e3))
            A.append(T_tcp)
            B.append(T_est)
        for m in methods:
            e_mm, e_deg = (float("nan"), float("nan"))
            if len(A) >= 3:
                try:
                    e_mm, e_deg = handeye_error(calibrate_handeye(A, B, m), hs["T_base_cam"])
                except cv2.error:  # handeye_curve와 같게 실패는 nan
                    e_mm, e_deg = float("nan"), float("nan")
            rows.append({"trial": trial, "method": m, "n_poses": n_poses, "n_detected": len(A),
                         "pixel_noise": pixel_noise, "pnp_med_rot_deg": float(np.median(pnp_rot)) if pnp_rot else np.nan,
                         "pnp_med_trans_mm": float(np.median(pnp_mm)) if pnp_mm else np.nan,
                         "err_mm": e_mm, "err_deg": e_deg})
    return pd.DataFrame(rows)


def handeye_reference_sigma(hs: Mapping[str, Any], hcfg: Mapping[str, Any], ref: Mapping[str, Any], n_trials: int,
                            seed: int) -> tuple[float, float, dict]:
    """기준 조건에서 Hand-eye 오차의 축별 σ ≈ RMS(크기)/√3 (등방 가우시안 가정). (rot_deg, trans_m, 정보)."""
    lv = hcfg["noise_levels"][int(ref["noise_level"])]
    df = handeye_curve(hs, [lv], [int(ref["n_poses"])], n_trials, seed + 7919, [ref["method"]])
    # cv2 실패(nan)는 RMS에서 빼고 개수를 남긴다
    n_failed = int((~np.isfinite(df.err_deg) | ~np.isfinite(df.err_mm)).sum())
    rms_deg = float(np.sqrt(np.nanmean(np.square(df.err_deg))))
    rms_mm = float(np.sqrt(np.nanmean(np.square(df.err_mm))))
    info = {"reference": dict(ref), "noise": dict(lv), "n_trials": n_trials, "n_failed": n_failed,
            "rms_err_deg": rms_deg, "rms_err_mm": rms_mm, "median_err_deg": float(df.err_deg.median()),
            "median_err_mm": float(df.err_mm.median())}
    if n_failed:
        info["warning"] = f"Hand-eye 기준 조건 {n_failed}/{n_trials}회 calibrateHandEye 실패 (σ는 성공 시행만으로 계산)"
        warnings.warn(info["warning"], RuntimeWarning, stacklevel=2)
    return rms_deg / np.sqrt(3), rms_mm / np.sqrt(3) * 1e-3, info


__all__ = ["Board", "board_points", "render_checkerboard", "sample_board_poses", "detect_corners",
           "calibrate_intrinsics", "intrinsics_error", "generate_handeye_data", "calibrate_handeye",
           "HANDEYE_METHODS", "handeye_error", "handeye_curve", "handeye_end_to_end", "handeye_reference_sigma",
           "handeye_settings", "T_from_cfg", "rotation_angle_deg"]
