"""자세 정확도 지표: ADD, ADD-S(ADI), 회전·병진 오차, 성공 판정, 요약.

ADD/ADI/re/te 정의는 BOP toolkit ``bop_toolkit_lib.pose_error``와 같다 (입력만 4x4 행렬).
함수는 단위에 무관하다 (입력이 m이면 출력도 m). 표·요약은 mm·°로 낸다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geometry import transform_points


def add(T_est: np.ndarray, T_gt: np.ndarray, pts: np.ndarray) -> float:
    """평균 대응점 거리 (비대칭 물체)."""
    return float(np.linalg.norm(transform_points(T_est, pts) - transform_points(T_gt, pts), axis=1).mean())


def adi(T_est: np.ndarray, T_gt: np.ndarray, pts: np.ndarray) -> float:
    """ADD-S: 정답 점마다 추정 점 중 최근접 거리의 평균 (BOP adi와 같은 방향)."""
    pts_est = transform_points(T_est, pts)
    pts_gt = transform_points(T_gt, pts)
    d, _ = cKDTree(pts_est).query(pts_gt, k=1)
    return float(d.mean())


def add_or_adi(T_est: np.ndarray, T_gt: np.ndarray, pts: np.ndarray, symmetric: bool) -> float:
    """ADD(-S): 대칭 물체는 ADI, 그 외 ADD."""
    return adi(T_est, T_gt, pts) if symmetric else add(T_est, T_gt, pts)


def re_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """회전 오차 (BOP re: arccos((tr(R_e R_gᵀ) − 1)/2))."""
    tr = min(float(np.trace(np.asarray(R_est)[:3, :3] @ np.asarray(R_gt)[:3, :3].T)), 3.0)
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, 0.5 * (tr - 1.0))))))


def te(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(t_gt).reshape(3) - np.asarray(t_est).reshape(3)))


def re_sym_deg(R_est: np.ndarray, R_gt: np.ndarray, symmetries: Sequence[np.ndarray]) -> float:
    """대칭 변환 S에 대해 min_S re(R_e, R_g·R_S)."""
    R_gt = np.asarray(R_gt)[:3, :3]
    return min(re_deg(R_est, R_gt @ np.asarray(S)[:3, :3]) for S in symmetries)


def axis_error_deg(R_est: np.ndarray, R_gt: np.ndarray, axis_obj: Sequence[float]) -> float:
    """물체 좌표 축(예: 대칭축)이 카메라 좌표에서 이루는 각 차이 — 축 주위 회전을 제외한 오차."""
    a = np.asarray(axis_obj, dtype=float)
    a = a / np.linalg.norm(a)
    ae = np.asarray(R_est)[:3, :3] @ a
    ag = np.asarray(R_gt)[:3, :3] @ a
    return float(np.degrees(np.arccos(np.clip(np.dot(ae, ag), -1.0, 1.0))))


def is_correct(err: float, diameter: float, frac: float = 0.1) -> bool:
    """BOP 관례: 오차 < frac × 직경 이면 정답."""
    return bool(np.isfinite(err) and err < frac * diameter)


def _q(x: pd.Series, q: float) -> float:
    x = x.dropna()
    return float(np.quantile(x, q)) if len(x) else float("nan")


def summarize(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    """객체별 + 전체('ALL') 성공률·중앙값·p95.

    입력 열: obj_id, symmetric, diameter_mm, add_mm, adi_mm, te_mm, re_deg, code (+ 선택 re_sym_deg, total_ms).
    - rate_*: 파이프라인이 실패 코드를 낸 대상은 오답으로 센다 (BOP의 '추정 없음 = 오답').
    - raw_rate_add_s: 신뢰도 게이트를 무시하고 마지막 추정 자세로 판정 (임계값 진단용).
    - 회전 통계(*_re_deg)는 유효 회전 오차: 대칭 물체는 re_sym_deg, 그 외 re_deg (re_sym_deg 열이 없으면 re_deg).
    - median/p95 te·re는 채택(code == OK) 추정만, raw_median/raw_p95는 추정 자세가 있는 전체 대상.
    """
    d = df.copy()
    thr = frac * d["diameter_mm"]
    adds = np.where(d["symmetric"].astype(bool), d["adi_mm"], d["add_mm"])
    ok = d["code"] == "OK"
    d["c_add"] = ok & (d["add_mm"] < thr)
    d["c_adi"] = ok & (d["adi_mm"] < thr)
    d["c_add_s"] = ok & (adds < thr)
    d["c_raw_add_s"] = adds < thr
    sym = d["symmetric"].astype(bool)
    d["re_eff"] = np.where(sym, d["re_sym_deg"], d["re_deg"]) if "re_sym_deg" in d else d["re_deg"]
    rows = []
    groups = [(str(k), g) for k, g in d.groupby("obj_id")] + [("ALL", d)]
    for name, g in groups:
        ga = g[g["code"] == "OK"]  # 채택 추정만
        row = {"obj_id": name, "n": int(len(g)), "rate_add_s": float(g["c_add_s"].mean()),
               "rate_add": float(g["c_add"].mean()), "rate_adi": float(g["c_adi"].mean()),
               "raw_rate_add_s": float(g["c_raw_add_s"].mean()), "accepted": float((g["code"] == "OK").mean()),
               "median_te_mm": _q(ga["te_mm"], 0.5), "p95_te_mm": _q(ga["te_mm"], 0.95),
               "median_re_deg": _q(ga["re_eff"], 0.5), "p95_re_deg": _q(ga["re_eff"], 0.95),
               "raw_median_te_mm": _q(g["te_mm"], 0.5), "raw_p95_te_mm": _q(g["te_mm"], 0.95),
               "raw_median_re_deg": _q(g["re_eff"], 0.5), "raw_p95_re_deg": _q(g["re_eff"], 0.95)}
        if "total_ms" in g:
            row["median_total_ms"] = _q(g["total_ms"], 0.5)
            row["p95_total_ms"] = _q(g["total_ms"], 0.95)
        rows.append(row)
    return pd.DataFrame(rows)
