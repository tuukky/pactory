"""PLAN_v5 §7.1 — S1(비전)·S2(MuJoCo 파지) eval 결과 → VP 셀 경험 분포 JSON.

    python tools/v5/build_vp_dist.py --s1 results/v5/s1_bin/eval-final_<ts> --s2 <mujo results/housing_pick/eval_...>
        --pipeline B1 --condition base --out configs/v5/vp_dist_measured.json

시도 1회 확률:
  p_reject = 1 − (S1 채택률)                         (채택 = 파지 가능한 후보를 냄)
  p_fail   = (S1 채택률) × (1 − S2 G-EST 성공률)      (채택했지만 MuJoCo 파지·놓기 실패)
시간 표본:
  vision_time_s       = S1 장면별 인식 시간 (채택 여부 무관)
  robot_time_s.success = S2 G-EST 성공 에피소드 cycle_s
  robot_time_s.fail    = S2 G-EST 실패 에피소드 cycle_s (실패까지 걸린 시간)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def build(s1_dir: Path, s2_dir: Path, pipeline: str, condition: str) -> dict:
    picks = pd.read_csv(s1_dir / "picks.csv")
    g = picks[(picks.pipeline == pipeline) & (picks.condition == condition)]
    if g.empty:
        raise ValueError(f"S1에 {pipeline}/{condition} 행이 없다")
    acc = float(g.accepted.mean())
    s2 = [json.loads(l) for l in (s2_dir / "pick_result.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    est = [r for r in s2 if r["group"] == "G-EST" and r["condition"] == condition and r["pipeline"] == pipeline]
    if not est:
        raise ValueError(f"S2에 G-EST/{condition}/{pipeline} 에피소드가 없다")
    ok = [r for r in est if r["success"]]
    bad = [r for r in est if not r["success"]]
    s_rate = len(ok) / len(est)
    return {
        "source": f"S1 {s1_dir.name} ({pipeline}, {condition}) + S2 {s2_dir.name} (G-EST {condition}). 합성 데이터·시뮬레이션 기반 경험 분포",
        "s1_dir": str(s1_dir), "s2_dir": str(s2_dir), "pipeline": pipeline, "condition": condition,
        "n_s1_scenes": int(len(g)), "s1_accept_rate": acc, "n_s2_episodes": len(est), "s2_success_rate": s_rate,
        "p_reject": 1.0 - acc, "p_fail": acc * (1.0 - s_rate),
        "vision_time_s": [float(x) / 1000.0 for x in g.vision_ms],
        "robot_time_s": {"success": [float(r["cycle_s"]) for r in ok], "fail": [float(r["cycle_s"]) for r in bad]},
        "s1_picks_sha256": hashlib.sha256((s1_dir / "picks.csv").read_bytes()).hexdigest(),
        "s2_results_sha256": hashlib.sha256((s2_dir / "pick_result.jsonl").read_bytes()).hexdigest(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--pipeline", default="B1")
    ap.add_argument("--condition", default="base")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = build(Path(a.s1), Path(a.s2), a.pipeline, a.condition)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in d.items() if not isinstance(v, (list, dict))}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
