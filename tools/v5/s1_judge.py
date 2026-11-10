"""PLAN_v5 §4.4 S1 가설 판정 (eval-final 결과만 읽음, 규칙은 PLAN 그대로).

    python tools/v5/s1_judge.py results/v5/s1_bin/eval-final_<ts>
→ <dir>/s1_judgement.json
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(a: pd.Series, b: pd.Series) -> dict:
    """정확 McNemar (불일치 쌍 이항검정). a, b: 같은 index의 bool."""
    b01 = int(((~a) & b).sum())   # a 실패, b 성공
    b10 = int((a & (~b)).sum())
    n = b01 + b10
    p = binomtest(b01, n, 0.5).pvalue if n else 1.0
    return {"a_only": b10, "b_only": b01, "p_exact": float(p)}


def rate(df, col):
    k, n = int(df[col].sum()), len(df)
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def fa_rate(df):
    acc = df[df.accepted]
    k, n = int(acc.false_accept.sum()), len(acc)
    lo, hi = wilson(k, n)
    return {"false_accept": k, "accepted": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def main(d):
    d = Path(d)
    r = pd.DataFrame([json.loads(l) for l in (d / "pick_records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
    out = {"source": str(d), "n_records": len(r)}
    per = {}
    for (pl, cond), g in r.groupby(["pipeline", "condition"]):
        per.setdefault(pl, {})[cond] = {"accept": rate(g, "accepted"), "correct_pick": rate(g, "correct_pick"),
                                         "false_accept_of_accepted": fa_rate(g),
                                         "first_codes": dict(Counter(g.first_code)),
                                         "median_vision_ms": float(g.vision_ms.median())}
    out["per_condition"] = per

    # 짝 비교 (같은 장면·조건): B0 vs B1
    key = ["condition", "scene_id"]
    w = r.pivot_table(index=key, columns="pipeline", values=["correct_pick", "false_accept", "accepted"], aggfunc="first")
    out["paired_all"] = {m: mcnemar(w[m]["B0"].astype(bool), w[m]["B1"].astype(bool))
                         for m in ["correct_pick", "false_accept", "accepted"]}
    out["paired_by_condition"] = {c: {m: mcnemar(g[m]["B0"].astype(bool), g[m]["B1"].astype(bool))
                                      for m in ["correct_pick", "false_accept"]}
                                  for c, g in w.groupby(level=0)}

    # V1: 적재 1 → 20에서 B0 채택률↓·오채택률↑, B1 적재 20 오채택률 ≤ 10 %
    b0 = per["B0"]; b1 = per["B1"]
    v1_b0 = {"accept_n01": b0["n01_base"]["accept"] if "n01_base" in b0 else b0["n01"]["accept"],
             "accept_n20": b0["n20"]["accept"],
             "fa_n01": b0["n01"]["false_accept_of_accepted"], "fa_n20": b0["n20"]["false_accept_of_accepted"]}
    v1_b1 = b1["n20"]["false_accept_of_accepted"]
    out["V1"] = {"B0": v1_b0, "B1_n20_fa": v1_b1,
                 "B0_direction_as_predicted": (v1_b0["accept_n20"]["rate"] < v1_b0["accept_n01"]["rate"]
                                               and (v1_b0["fa_n20"]["rate"] or 0) > (v1_b0["fa_n01"]["rate"] or 0)),
                 "B1_point_le_10pct": (v1_b1["rate"] or 0) <= 0.10,
                 "B1_ci_upper_le_10pct": (v1_b1["ci95"][1] or 1) <= 0.10}
    # V2: 결측 30 %에서 채택률 ≤ 기준의 절반, 실패 코드 LOW_FITNESS 계열이 주
    v2 = {}
    for pl in ("B0", "B1"):
        base, d30 = per[pl]["base"]["accept"]["rate"], per[pl]["drop30"]["accept"]["rate"]
        codes = Counter(r[(r.pipeline == pl) & (r.condition == "drop30") & (~r.accepted)].first_code)
        v2[pl] = {"accept_base": base, "accept_drop30": d30, "halved": d30 <= base / 2,
                  "reject_codes_drop30": dict(codes),
                  "low_fitness_top": bool(codes) and codes.most_common(1)[0][0].startswith("LOW_FIT")}
    out["V2"] = v2
    # V3: 기준 조건 인식 시간 중앙값 B1 < B0
    out["V3"] = {"B0_ms": b0["base"]["median_vision_ms"], "B1_ms": b1["base"]["median_vision_ms"],
                 "B1_shorter": b1["base"]["median_vision_ms"] < b0["base"]["median_vision_ms"]}
    # 오채택 유형: 뒤집힘(축 오차 > 90°) vs 허용치 근처
    fa = r[r.false_accept]
    out["false_accept_types"] = {pl: {"flipped": int((g.tcp_axis_err_deg > 90).sum()),
                                      "near_tolerance": int((g.tcp_axis_err_deg <= 90).sum()),
                                      "by_condition": dict(Counter(g.condition))}
                                 for pl, g in fa.groupby("pipeline")}
    p = d / "s1_judgement.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("paired_all", "V1", "V2", "V3", "false_accept_types")}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
