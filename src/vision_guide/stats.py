"""v4 지표·비교 통계: 정답/채택/오채택 플래그, 헤드라인 지표, 가시 구간 요약, 프레임 단위 짝지은 부트스트랩.

정의 (PLAN_v4 §6.1)
- 정답(correct): 채택(code == OK) 이고 ADD(-S) < frac × 직경. 실패 코드(추정 거부)는 오답.
- 오채택률(false_accept_rate): 채택했지만 오답인 대상 / 전체 대상.
- 채택 정밀도(precision): 정답 / 채택 (채택 0이면 nan).
- 대상 영역 확보율(region_hit_rate): 시도한 영역 중 점 50% 이상이 정답 가시 마스크 안인 것이 있던 대상 비율.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

FRAME_KEYS = ["scene_id", "im_id"]
TARGET_KEYS = ["scene_id", "im_id", "obj_id"]
BOOT_METRICS = ("rate_add_s", "false_accept_rate", "precision", "median_total_ms", "region_hit_rate")


def add_correct_flags(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    """correct, accepted, false_accept, add_s_mm 열을 더한 사본."""
    d = df.copy()
    sym = d["symmetric"].astype(bool)
    d["add_s_mm"] = np.where(sym, d["adi_mm"], d["add_mm"])
    d["accepted"] = d["code"] == "OK"
    under = np.asarray(d["add_s_mm"] < frac * d["diameter_mm"], dtype=bool)
    d["correct"] = d["accepted"] & under
    d["false_accept"] = d["accepted"] & ~d["correct"]
    return d


def _region_col(d: pd.DataFrame) -> str | None:
    for c in ("region_hit", "seg_hit"):
        if c in d:
            return c
    return None


def headline(df: pd.DataFrame, frac: float = 0.1) -> dict:
    d = add_correct_flags(df, frac)
    n = len(d)
    acc = int(d["accepted"].sum())
    rc = _region_col(d)
    out = {"n_targets": int(n), "n_frames": int(d[FRAME_KEYS].drop_duplicates().shape[0]),
           "rate_add_s": float(d["correct"].mean()) if n else float("nan"),
           "accepted_rate": float(d["accepted"].mean()) if n else float("nan"),
           "false_accept_rate": float(d["false_accept"].mean()) if n else float("nan"),
           "precision": float(d["correct"].sum() / acc) if acc else float("nan"),
           "n_accepted": acc, "n_correct": int(d["correct"].sum()), "n_false_accept": int(d["false_accept"].sum()),
           "median_total_ms": float(d["total_ms"].median()) if "total_ms" in d else float("nan"),
           "p95_total_ms": float(d["total_ms"].quantile(0.95)) if "total_ms" in d else float("nan")}
    if rc is not None:
        out["region_hit_rate"] = float(d[rc].astype(float).mean())
    return out


def by_visib(df: pd.DataFrame, bins: Sequence[float] = (0, .3, .6, .8, 1.0001), frac: float = 0.1) -> pd.DataFrame:
    """가시 비율 구간별 성공률·오채택률 (구간: (a, b], 첫 구간은 0 포함)."""
    d = add_correct_flags(df, frac)
    cut = pd.cut(d["visib_fract"], bins=list(bins), include_lowest=True)
    rows = []
    for iv, g in d.groupby(cut, observed=False):
        rows.append({"visib_bin": str(iv), "n": int(len(g)),
                     "rate_add_s": float(g["correct"].mean()) if len(g) else float("nan"),
                     "false_accept_rate": float(g["false_accept"].mean()) if len(g) else float("nan"),
                     "accepted_rate": float(g["accepted"].mean()) if len(g) else float("nan")})
    return pd.DataFrame(rows)


def _metric_from_counts(name: str, w: np.ndarray, agg: Mapping[str, np.ndarray]) -> np.ndarray:
    """w: (B, F) 프레임 재표본 횟수. agg: 프레임별 합계 배열 (F,)."""
    n = w @ agg["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "rate_add_s":
            return (w @ agg["correct"]) / n
        if name == "false_accept_rate":
            return (w @ agg["false_accept"]) / n
        if name == "precision":
            return (w @ agg["correct"]) / (w @ agg["accepted"])
        if name == "accepted_rate":
            return (w @ agg["accepted"]) / n
        if name == "region_hit_rate":
            return (w @ agg["region"]) / n
    raise KeyError(name)


def _frame_aggregates(d: pd.DataFrame, frames: list[tuple]) -> dict:
    g = d.groupby(FRAME_KEYS)
    idx = pd.MultiIndex.from_tuples(frames, names=FRAME_KEYS)
    rc = _region_col(d)
    agg = {"n": g.size().reindex(idx, fill_value=0).to_numpy(float),
           "correct": g["correct"].sum().reindex(idx, fill_value=0).to_numpy(float),
           "false_accept": g["false_accept"].sum().reindex(idx, fill_value=0).to_numpy(float),
           "accepted": g["accepted"].sum().reindex(idx, fill_value=0).to_numpy(float)}
    agg["region"] = (g[rc].sum().reindex(idx, fill_value=0).to_numpy(float) if rc is not None
                     else np.full(len(frames), np.nan))
    return agg


def _median_boot(d: pd.DataFrame, frames: list[tuple], w: np.ndarray) -> np.ndarray:
    pos = {f: i for i, f in enumerate(frames)}
    fi = np.array([pos[(a, b)] for a, b in zip(d["scene_id"], d["im_id"])])
    t = d["total_ms"].to_numpy(float)
    out = np.empty(len(w))
    for b in range(len(w)):
        rep = w[b][fi]
        out[b] = np.median(np.repeat(t, rep.astype(int))) if rep.sum() else np.nan
    return out


def paired_frame_bootstrap(dfs: Mapping[str, pd.DataFrame], pairs: Sequence[tuple],
                           metrics: Sequence[str] = BOOT_METRICS, n: int = 10000, seed: int = 20261030,
                           frac: float = 0.1) -> pd.DataFrame:
    """프레임 단위 짝지은 부트스트랩. pairs: (a, b) → b − a, 또는 (a, b, c) → b − c·a.

    같은 재표본 프레임 집합을 모든 버전에 쓴다 (프레임 집합은 모든 df에서 같아야 한다).
    반환 열: a, b, coef, metric, estimate, ci_low, ci_high, ci_contains_zero, n_boot.
    """
    ds = {k: add_correct_flags(v, frac) for k, v in dfs.items()}
    fsets = {k: set(map(tuple, v[FRAME_KEYS].drop_duplicates().to_numpy().tolist())) for k, v in ds.items()}
    first = next(iter(fsets.values()))
    if any(s != first for s in fsets.values()):
        raise ValueError("버전 간 프레임 집합이 다르다")
    frames = sorted(first)
    F = len(frames)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(F, np.full(F, 1.0 / F), size=n).astype(float)   # (B, F)
    ones = np.ones((1, F))
    aggs = {k: _frame_aggregates(v, frames) for k, v in ds.items()}
    cache: dict = {}

    def stat(k: str, name: str, ww: np.ndarray) -> np.ndarray:
        key = (k, name, ww.shape[0])
        if key not in cache:
            if name == "median_total_ms":
                cache[key] = _median_boot(ds[k], frames, ww) if "total_ms" in ds[k] else np.full(len(ww), np.nan)
            else:
                cache[key] = _metric_from_counts(name, ww, aggs[k])
        return cache[key]

    rows = []
    for pr in pairs:
        a, b = pr[0], pr[1]
        c = float(pr[2]) if len(pr) > 2 else 1.0
        for name in metrics:
            est = float(stat(b, name, ones)[0] - c * stat(a, name, ones)[0])
            diff = stat(b, name, w) - c * stat(a, name, w)
            diff = diff[np.isfinite(diff)]
            lo, hi = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))) if len(diff) else (np.nan,) * 2
            rows.append({"a": a, "b": b, "coef": c, "metric": name, "estimate": est, "ci_low": lo, "ci_high": hi,
                         "ci_contains_zero": bool(lo <= 0 <= hi) if np.isfinite(lo) else None,
                         "n_boot": int(len(diff))})
    return pd.DataFrame(rows)


def exact_compare(df_a: pd.DataFrame, df_b: pd.DataFrame, cols: Sequence[str] = ("code", "fitness", "rmse_mm", "add_mm")
                  ) -> dict:
    """대상별 열이 정확히 같은가 (NaN끼리는 같음, 시간 열 제외)."""
    a = df_a.set_index(TARGET_KEYS).sort_index()
    b = df_b.set_index(TARGET_KEYS).sort_index()
    out = {"same_targets": a.index.equals(b.index), "n": int(len(a)), "mismatch": {}}
    if not out["same_targets"]:
        out["equal"] = False
        return out
    for c in cols:
        x, y = a[c], b[c]
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            eq = (xv == yv) | (np.isnan(xv) & np.isnan(yv))
        else:
            eq = (x.astype(str).to_numpy() == y.astype(str).to_numpy())
        out["mismatch"][c] = int((~eq).sum())
    out["equal"] = all(v == 0 for v in out["mismatch"].values())
    return out


def compare_runs(df_ref: pd.DataFrame, df_new: pd.DataFrame, tol_pp: float = 3.0, frac: float = 0.1,
                 min_agree: float = 0.90, n_boot: int = 10000, seed: int = 20261030) -> dict:
    """재현 B 판정 (PLAN_v4 §7): 대상 집합 동일, ADD(-S)·채택률·오채택률 |Δ| ≤ tol_pp, 대상별 (채택, 정답)
    일치율 ≥ min_agree, 짝지은 부트스트랩 CI ∋ 0 (ADD(-S)·채택률·오채택률)."""
    ra, rb = add_correct_flags(df_ref, frac), add_correct_flags(df_new, frac)
    ka = set(map(tuple, ra[TARGET_KEYS].to_numpy().tolist()))
    kb = set(map(tuple, rb[TARGET_KEYS].to_numpy().tolist()))
    out: dict = {"same_targets": ka == kb, "n_ref": len(ra), "n_new": len(rb)}
    ha, hb = headline(df_ref, frac), headline(df_new, frac)
    out["ref"], out["new"] = ha, hb
    deltas = {k: 100.0 * (hb[k] - ha[k]) for k in ("rate_add_s", "accepted_rate", "false_accept_rate")}
    out["delta_pp"] = deltas
    out["delta_ok"] = all(abs(v) <= tol_pp for v in deltas.values())
    m = ra.set_index(TARGET_KEYS)[["accepted", "correct"]].join(
        rb.set_index(TARGET_KEYS)[["accepted", "correct"]], lsuffix="_ref", rsuffix="_new", how="inner")
    agree = ((m.accepted_ref == m.accepted_new) & (m.correct_ref == m.correct_new)).mean() if len(m) else float("nan")
    out["target_agreement"] = float(agree)
    out["agreement_ok"] = bool(agree >= min_agree)
    bs = paired_frame_bootstrap({"ref": df_ref, "new": df_new}, [("ref", "new")],
                                ("rate_add_s", "accepted_rate", "false_accept_rate"), n=n_boot, seed=seed, frac=frac)
    out["bootstrap"] = bs.to_dict(orient="records")
    out["ci_ok"] = bool(all(r["ci_contains_zero"] for r in out["bootstrap"]))
    out["pass"] = bool(out["same_targets"] and out["delta_ok"] and out["agreement_ok"] and out["ci_ok"])
    return out
