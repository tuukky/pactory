"""카메라 모델, 투영·역투영, SE(3) 유틸.

규약
- 길이는 m, 각도는 °(함수 이름에 _deg) 또는 rad(내부).
- 변환 ``T_a_b``는 b 좌표의 점을 a 좌표로 옮기는 4x4 동차 행렬이다 (p_a = T_a_b · p_b).
- 카메라 좌표: +x 오른쪽, +y 아래, +z 시선(OpenCV와 같은 축).
- 픽셀 규약: 픽셀 (u, v)는 연속 좌표 [u, u+1) x [v, v+1)를 덮고 중심은 (u+0.5, v+0.5)이다.
  OpenCV/BOP는 픽셀 중심을 정수 좌표로 두므로 ``cx_cv = cx - 0.5``이다
  (``K_opencv()``/``from_opencv()``가 경계에서 변환한다).
- 왜곡 ``dist``는 OpenCV 순서 [k1, k2, p1, p2, k3]. 0이 아니면 project/backproject에 cv2로 적용한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------- 카메라 모델
@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: tuple[float, ...] = ()

    def K(self) -> np.ndarray:
        """이 모듈 규약(픽셀 중심 = +0.5)의 3x3 내부 행렬."""
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def K_opencv(self) -> np.ndarray:
        """OpenCV/BOP 규약(픽셀 중심 = 정수) 내부 행렬. cx, cy를 0.5 뺀다."""
        K = self.K()
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        return K

    def dist_array(self) -> np.ndarray:
        d = np.zeros(5)
        d[: len(self.dist)] = self.dist
        return d

    @property
    def has_distortion(self) -> bool:
        return any(abs(x) > 0 for x in self.dist)

    def pinhole(self) -> "CameraIntrinsics":
        """왜곡을 뺀 사본 (정합된 Depth 영상용)."""
        return replace(self, dist=())

    @classmethod
    def from_K(cls, K: np.ndarray, width: int, height: int, dist: Sequence[float] = ()) -> "CameraIntrinsics":
        """이 모듈 규약의 K에서 만든다."""
        K = np.asarray(K, dtype=float)
        return cls(float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), int(width), int(height),
                   tuple(float(x) for x in dist))

    @classmethod
    def from_opencv(cls, K_cv: np.ndarray, width: int, height: int,
                    dist: Sequence[float] = ()) -> "CameraIntrinsics":
        """OpenCV/BOP 규약 K에서 만든다 (cx, cy에 0.5를 더한다)."""
        K = np.array(K_cv, dtype=float).reshape(3, 3).copy()
        K[0, 2] += 0.5
        K[1, 2] += 0.5
        return cls.from_K(K, width, height, dist)


def load_camera(path: str | Path) -> tuple[CameraIntrinsics, np.ndarray]:
    """camera.yaml → (내부 파라미터, T_base_cam)."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    i = cfg["intrinsics"]
    intr = CameraIntrinsics(float(i["fx"]), float(i["fy"]), float(i["cx"]), float(i["cy"]), int(i["width"]),
                            int(i["height"]), tuple(float(x) for x in i.get("dist", ())))
    e = cfg["extrinsics"]
    T_base_cam = look_at(e["position_m"], e["look_at_m"], e["image_up"])
    return intr, T_base_cam


# ---------------------------------------------------------------------------- 투영
def project(points_cam: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """카메라 좌표 점 (N,3) → (연속 픽셀 좌표 uv (N,2), 깊이 z (N,))."""
    p = np.asarray(points_cam, dtype=float).reshape(-1, 3)
    z = p[:, 2].copy()
    if intr.has_distortion:
        uv, _ = cv2.projectPoints(p.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), intr.K_opencv(),
                                  intr.dist_array())
        uv = uv.reshape(-1, 2) + 0.5
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([intr.fx * p[:, 0] / z + intr.cx, intr.fy * p[:, 1] / z + intr.cy], axis=1)
    return uv, z


def normalized_rays(uv: np.ndarray, intr: CameraIntrinsics) -> np.ndarray:
    """연속 픽셀 좌표 → z=1 정규화 광선 (N,3). 왜곡이 있으면 cv2로 반복 보정한다."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    if intr.has_distortion:
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-14)
        xy = cv2.undistortPointsIter((uv - 0.5).reshape(-1, 1, 2), intr.K_opencv(), intr.dist_array(),
                                     None, None, crit).reshape(-1, 2)
    else:
        xy = np.stack([(uv[:, 0] - intr.cx) / intr.fx, (uv[:, 1] - intr.cy) / intr.fy], axis=1)
    return np.concatenate([xy, np.ones((len(xy), 1))], axis=1)


def backproject(uv: np.ndarray, depth: np.ndarray | float, intr: CameraIntrinsics) -> np.ndarray:
    """연속 픽셀 좌표 + z-깊이 → 카메라 좌표 점 (N,3)."""
    rays = normalized_rays(uv, intr)
    d = np.broadcast_to(np.asarray(depth, dtype=float).reshape(-1), (len(rays),))
    return rays * d[:, None]


def pixel_centers(intr: CameraIntrinsics) -> np.ndarray:
    """모든 픽셀 중심 (H*W, 2), 행 우선 순서 (v, u)."""
    vv, uu = np.mgrid[0:intr.height, 0:intr.width]
    return np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5], axis=1).astype(float)


def depth_to_points(depth_m: np.ndarray, intr: CameraIntrinsics,
                    mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Depth 영상(m, 결측 0) → (카메라 좌표 점 (N,3), 픽셀 인덱스 (N,2) = (v, u))."""
    depth_m = np.asarray(depth_m, dtype=float)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    v, u = np.nonzero(valid)
    uv = np.stack([u + 0.5, v + 0.5], axis=1).astype(float)
    pts = backproject(uv, depth_m[v, u], intr)
    return pts, np.stack([v, u], axis=1)


# ---------------------------------------------------------------------------- SE(3)
def make_T(R: np.ndarray | None = None, t: Sequence[float] | None = None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def compose(*Ts: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, dtype=float).reshape(-1, 3)
    return p @ T[:3, :3].T + T[:3, 3]


def is_se3(T: np.ndarray, tol: float = 1e-9) -> bool:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1], atol=tol):
        return False
    R = T[:3, :3]
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)


def rotation_angle_deg(R: np.ndarray) -> float:
    """회전 행렬의 회전각(°). trace 공식은 작은 각에서 부정확하므로 회전벡터 크기를 쓴다."""
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(np.asarray(R)[:3, :3]).as_rotvec())))


def rotation_between_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return rotation_angle_deg(np.asarray(R_a)[:3, :3] @ np.asarray(R_b)[:3, :3].T)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """SO(3) 균등 분포 회전 행렬."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    return Rotation.from_quat(q).as_matrix()


def rotation_about(axis: Sequence[float], angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return Rotation.from_rotvec(a * np.radians(angle_deg)).as_matrix()


def perturb_T(T: np.ndarray, rot_sigma_deg: float, trans_sigma_m: float, rng: np.random.Generator,
              side: str = "right") -> np.ndarray:
    """축별 σ의 등방 가우시안 오차를 더한다 (회전벡터 각 성분 ~ N(0, σ²), 병진 각 성분 ~ N(0, σ²)).

    side="right": T · ΔT — 회전은 자식 좌표 원점 주위, 병진은 자식 좌표로 표현 (기본).
    side="left":  ΔT · T — 회전은 부모 좌표 원점 주위.
    """
    dR = Rotation.from_rotvec(np.radians(rot_sigma_deg) * rng.standard_normal(3)).as_matrix()
    dT = make_T(dR, trans_sigma_m * rng.standard_normal(3))
    if side == "right":
        return T @ dT
    if side == "left":
        return dT @ T
    raise ValueError(f"side must be 'right' or 'left': {side!r}")


def look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float]) -> np.ndarray:
    """T_world_cam: 카메라가 eye에서 target을 보며, 영상 위쪽(-y_cam)이 up 쪽을 향한다."""
    eye = np.asarray(eye, dtype=float)
    z = np.asarray(target, dtype=float) - eye
    z /= np.linalg.norm(z)
    up = np.asarray(up, dtype=float)
    up_perp = up - np.dot(up, z) * z
    n = np.linalg.norm(up_perp)
    if n < 1e-9:
        raise ValueError("up 벡터가 시선과 평행하다")
    y = -up_perp / n
    x = np.cross(y, z)
    return make_T(np.stack([x, y, z], axis=1), eye)
