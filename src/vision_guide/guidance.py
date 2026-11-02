"""Guidance: 물체 자세 → 로봇 base 좌표 TCP 목표, 접근 방향·작업영역 검사, 오차 예산.

T_base_tcp = T_base_cam · T_cam_obj · T_obj_grasp

- 접근 기울기: TCP 접근축(R_base_tcp · a_tcp)과 base 연직 아래(0, 0, −1) 사이 각.
- 파지 선택: 대칭 후보 × 파지 목록 중 작업영역·기울기를 통과한 것에서 기울기 최소
  (동률이면 파지 순번, 대칭 순번이 작은 것). 모두 실패하면 failures.majority_grasp_failure 규칙.
- 오차 예산: 자세 오차(물체 원점 기준 오른쪽 섭동)와 Hand-eye 오차(카메라 원점 기준 오른쪽 섭동)를
  몬테카를로로 합성한다. 파지 성공은 위치·접근축 허용치(합성 가정)로 판정한다 — 파지 역학 아님.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .calib import T_from_cfg
from .failures import FailureCode, majority_grasp_failure
from .geometry import compose, inv_T, perturb_T, rotation_angle_deg

DOWN = np.array([0.0, 0.0, -1.0])


@dataclass
class Gripper:
    grasps: list[np.ndarray]           # T_obj_grasp
    approach_axis_tcp: np.ndarray
    max_tilt_deg: float
    tol_pos_m: float
    tol_axis_deg: float
    grasp_names: list[str] | None = None


@dataclass
class Workspace:
    lo: np.ndarray
    hi: np.ndarray

    def contains(self, p: np.ndarray) -> bool:
        p = np.asarray(p).reshape(3)
        return bool(np.all(p >= self.lo) and np.all(p <= self.hi))


@dataclass
class GraspDecision:
    T_base_tcp: np.ndarray | None
    grasp_idx: int | None
    sym_idx: int | None
    code: FailureCode
    tilt_deg: float = float("nan")
    n_unreachable: int = 0
    n_approach: int = 0


def load_gripper(path: str | Path) -> tuple[Gripper, Workspace]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    g = cfg["gripper"]
    grip = Gripper([T_from_cfg(x) for x in g["grasps"]], np.asarray(g["approach_axis_tcp"], dtype=float),
                   float(g["max_tilt_deg"]), float(g["tolerance"]["pos_m"]), float(g["tolerance"]["axis_deg"]),
                   [x.get("name", str(i)) for i, x in enumerate(g["grasps"])])
    w = cfg["workspace"]
    ws = Workspace(np.array([w["x_m"][0], w["y_m"][0], w["z_m"][0]]), np.array([w["x_m"][1], w["y_m"][1], w["z_m"][1]]))
    return grip, ws


def tcp_target(T_base_cam: np.ndarray, T_cam_obj: np.ndarray, T_obj_grasp: np.ndarray) -> np.ndarray:
    return compose(T_base_cam, T_cam_obj, T_obj_grasp)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def check_approach(T_base_tcp: np.ndarray, gripper: Gripper, down: Sequence[float] = DOWN) -> tuple[bool, float]:
    tilt = _angle_deg(T_base_tcp[:3, :3] @ gripper.approach_axis_tcp, np.asarray(down, dtype=float))
    return tilt <= gripper.max_tilt_deg, tilt


def check_workspace(T_base_tcp: np.ndarray, ws: Workspace) -> bool:
    return ws.contains(T_base_tcp[:3, 3])


def select_grasp(T_base_cam: np.ndarray, pose_candidates: Sequence[np.ndarray], gripper: Gripper,
                 ws: Workspace) -> GraspDecision:
    """작업영역 → 기울기 순으로 검사. 통과 후보 중 기울기 최소."""
    if not pose_candidates or not gripper.grasps:
        return GraspDecision(None, None, None, FailureCode.NO_GRASP)
    feasible = []
    n_unreach = n_appr = 0
    for si, T_cam_obj in enumerate(pose_candidates):
        for gi, G in enumerate(gripper.grasps):
            T = tcp_target(T_base_cam, T_cam_obj, G)
            if not check_workspace(T, ws):
                n_unreach += 1
                continue
            ok, tilt = check_approach(T, gripper)
            if not ok:
                n_appr += 1
                continue
            feasible.append((tilt, gi, si, T))
    if not feasible:
        return GraspDecision(None, None, None, majority_grasp_failure(n_unreach, n_appr), float("nan"),
                             n_unreach, n_appr)
    tilt, gi, si, T = min(feasible, key=lambda x: (x[0], x[1], x[2]))
    return GraspDecision(T, gi, si, FailureCode.OK, tilt, n_unreach, n_appr)


def tcp_error(T_est: np.ndarray, T_true: np.ndarray, approach_axis: Sequence[float]) -> tuple[float, float]:
    """(위치 오차 m, 접근축 각 오차 °)."""
    a = np.asarray(approach_axis, dtype=float)
    return (float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])),
            _angle_deg(T_est[:3, :3] @ a, T_true[:3, :3] @ a))


def tcp_error_sym(T_base_tcp_est: np.ndarray, T_base_cam: np.ndarray, T_cam_obj_true: np.ndarray,
                  symmetries: Sequence[np.ndarray], T_obj_grasp: np.ndarray,
                  approach_axis: Sequence[float]) -> tuple[float, float]:
    """정답 자세의 대칭 변형 중 위치 오차가 가장 작은 것과 비교."""
    errs = [tcp_error(T_base_tcp_est, tcp_target(T_base_cam, T_cam_obj_true @ S, T_obj_grasp), approach_axis)
            for S in symmetries]
    return min(errs, key=lambda e: (e[0], e[1]))


# ---------------------------------------------------------------------------- 오차 예산
def budget_scene(gcfg: Mapping[str, Any], T_base_cam: np.ndarray) -> tuple[np.ndarray, int]:
    """오차 예산 기준 장면: (T_cam_obj, 파지 순번)."""
    o = gcfg["error_budget"]["object"]
    return inv_T(T_base_cam) @ T_from_cfg(o), int(o["grasp"])


def _rotvec_angle_deg(sigma_deg: float, rng: np.random.Generator, n: int) -> np.ndarray:
    return np.degrees(np.linalg.norm(np.radians(sigma_deg) * rng.standard_normal((n, 3)), axis=1))


def mc_error_budget(T_base_cam: np.ndarray, T_cam_obj: np.ndarray, T_obj_grasp: np.ndarray, gripper: Gripper,
                    pose_sigma: tuple[float, float], he_sigma: tuple[float, float], n: int, seed: int
                    ) -> pd.DataFrame:
    """source ∈ {pose, handeye, both}별 TCP 오차 표본. pose·handeye 난수는 source 간 공유한다 (CRN)."""
    T_true = tcp_target(T_base_cam, T_cam_obj, T_obj_grasp)
    rp = np.random.default_rng([int(seed), 1])
    rh = np.random.default_rng([int(seed), 2])
    P = [perturb_T(T_cam_obj, pose_sigma[0], pose_sigma[1], rp) for _ in range(n)]
    H = [perturb_T(T_base_cam, he_sigma[0], he_sigma[1], rh) for _ in range(n)]
    rows = []
    for src in ("pose", "handeye", "both"):
        for i in range(n):
            Tc = P[i] if src in ("pose", "both") else T_cam_obj
            Th = H[i] if src in ("handeye", "both") else T_base_cam
            pe, ae = tcp_error(tcp_target(Th, Tc, T_obj_grasp), T_true, gripper.approach_axis_tcp)
            rows.append({"source": src, "i": i, "pos_err_mm": pe * 1e3, "axis_err_deg": ae,
                         "success": bool(pe <= gripper.tol_pos_m and ae <= gripper.tol_axis_deg)})
    return pd.DataFrame(rows)


def max_handeye_rot_for_tolerance(T_base_cam: np.ndarray, T_cam_obj: np.ndarray, T_obj_grasp: np.ndarray,
                                  gripper: Gripper, pose_sigma: tuple[float, float], he_trans_sigma_m: float,
                                  grid_deg: Sequence[float], n: int, seed: int, success_target: float = 0.95
                                  ) -> tuple[float | None, pd.DataFrame]:
    """Hand-eye 회전 σ 격자 탐색: 성공률 ≥ 목표를 만족하는 가장 큰 σ (그보다 작은 격자값도 모두 만족해야 함).

    반환 σ는 축별 σ(°)다. 회전각 크기의 중앙값은 표의 he_rot_med_angle_deg (≈ 1.54σ).
    """
    rows = []
    best: float | None = None
    broken = False
    for sig in sorted(grid_deg):
        df = mc_error_budget(T_base_cam, T_cam_obj, T_obj_grasp, gripper, pose_sigma, (sig, he_trans_sigma_m), n,
                             seed)
        b = df[df.source == "both"]
        rate = float(b.success.mean())
        med_ang = float(np.median(_rotvec_angle_deg(sig, np.random.default_rng([int(seed), 3]), n))) if sig else 0.0
        rows.append({"he_rot_sigma_deg": sig, "he_rot_med_angle_deg": med_ang, "success": rate,
                     "p95_pos_mm": float(b.pos_err_mm.quantile(0.95)), "p95_axis_deg": float(b.axis_err_deg.quantile(0.95))})
        if rate >= success_target and not broken:
            best = sig
        else:
            broken = True
    return best, pd.DataFrame(rows)


__all__ = ["Gripper", "Workspace", "GraspDecision", "load_gripper", "tcp_target", "check_approach",
           "check_workspace", "select_grasp", "tcp_error", "tcp_error_sym", "mc_error_budget",
           "max_handeye_rot_for_tolerance", "budget_scene", "rotation_angle_deg"]
