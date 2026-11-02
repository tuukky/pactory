"""워밍업 길이 확인: 시간당 양품 출하 수열의 MSER-5 (PLAN_v2 §9.1, D24).

MSER-5: 수열을 5개씩 묶은 평균 b_1..b_n 에 대해 d = 0..n/2 중
    MSER(d) = Σ_{j>d} (b_j − mean(b_{d+1..n}))² / (n − d)²
를 최소로 하는 d* 를 고른다. 워밍업 = d* × 5 시간.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import CSV_ENCODING

BATCH = 5


def hourly_good(orders: pd.DataFrame, horizon_s: float) -> np.ndarray:
    """[0, horizon) 를 1시간 구간으로 나눈 양품 출하 대수."""
    n_h = int(round(horizon_s / 3600.0))
    t = orders.loc[orders["outcome"] == "shipped", "terminal_s"].astype(float).to_numpy()
    t = t[(t >= 0) & (t < n_h * 3600.0)]
    return np.bincount((t // 3600.0).astype(int), minlength=n_h).astype(float)


def hourly_good_from_dir(run_dir: str | Path, horizon_s: float) -> np.ndarray:
    orders = pd.read_csv(Path(run_dir) / "orders.csv", encoding=CSV_ENCODING)
    return hourly_good(orders, horizon_s)


def mser5(x: np.ndarray | list[float], batch: int = BATCH) -> dict[str, Any]:
    """MSER-5 절단점. 반환: d_batches, d_hours(수열 단위 = 1h 가정), mser_curve."""
    x = np.asarray(x, dtype=float)
    n_b = len(x) // batch
    if n_b < 2:
        raise ValueError(f"MSER-{batch}: 묶음이 2개 미만 (길이 {len(x)})")
    b = x[: n_b * batch].reshape(n_b, batch).mean(axis=1)
    curve = []
    for d in range(0, n_b // 2 + 1):
        tail = b[d:]
        curve.append(float(((tail - tail.mean()) ** 2).sum() / (n_b - d) ** 2))
    d_star = int(np.argmin(curve))
    return {"d_batches": d_star, "d_hours": d_star * batch, "mser_curve": curve, "n_batches": n_b}


def decide_warmup_h(d_hours_by_seed: list[float], planned_h: float = 80.0, step_h: float = 8.0) -> dict[str, Any]:
    """D24: 시드별 d* 의 최댓값이 계획(80h) 이하면 계획 유지, 넘으면 8h 단위로 올림."""
    mx = float(max(d_hours_by_seed)) if d_hours_by_seed else 0.0
    if mx <= planned_h:
        return {"max_d_hours": mx, "warmup_h": planned_h, "changed": False}
    w = float(np.ceil(mx / step_h) * step_h)
    return {"max_d_hours": mx, "warmup_h": w, "changed": True}
