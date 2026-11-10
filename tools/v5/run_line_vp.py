"""PLAN_v5 S3 — Rev.C + VP 셀 라인 영향 실험 (§7).

    python tools/v5/run_line_vp.py --phase dev      # 개발 시드 300–319 (검증·동작 확인)
    python tools/v5/run_line_vp.py --phase confirm  # 확인 시드 3000–3029, 1회 (잠금)

시나리오 (§7.2): BASE(Rev.C, VP 없음) / VP_MEAS(측정 분포) / VP_N20(적재 20 분포) /
VP_F{1,3,5,10,20,40}(1회 실패율 민감도, 시간 분포는 측정값). 수요 20.7·23.8대/h.
같은 시드에서 공통 난수(주문·고장·VP 시도 난수)로 BASE와 짝 비교 → 95% CI (t 분포).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "v5"))

VP_CFG = ROOT / "configs" / "v5" / "rev_C_vp.yaml"
DIST = {"VP_MEAS": "configs/v5/vp_dist_measured.json", "VP_N20": "configs/v5/vp_dist_measured_n20.json"}
SENS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]   # 60·80 %: dev에서 40 %까지 임계점 없음 (PLAN_v5 §15)
DEMANDS = {"d20.7": 20.7, "d23.8": 23.8}
PHASES = {"dev": list(range(300, 320)), "confirm": list(range(3000, 3030))}
OUT = ROOT / "results" / "v5" / "s3_line"
METRICS = ("demand_fulfil_ratio", "otd_rate", "lead_p90_lb_h")


def scenarios() -> dict[str, dict]:
    sc = {"BASE": {"vp": False, "ov": {}}}
    sc["VP_MEAS"] = {"vp": True, "ov": {"stations.VP.vision_pick.dist_file": DIST["VP_MEAS"]}}
    sc["VP_N20"] = {"vp": True, "ov": {"stations.VP.vision_pick.dist_file": DIST["VP_N20"]}}
    for f in SENS:
        sc[f"VP_F{int(round(f * 100)):02d}"] = {"vp": True, "ov": {"stations.VP.vision_pick.dist_file": DIST["VP_MEAS"],
                                                                     "stations.VP.vision_pick.attempt_fail_rate": f}}
    return sc


def config_for(name: str, demand: float) -> dict:
    from factory_sim.config import resolve_config
    import regress_revc as R
    s = scenarios()[name]
    if not s["vp"]:
        from factory_sim.config import apply_overrides
        return resolve_config(apply_overrides(R.rev_c_config(), {"demand.rate_per_h": demand}))
    return resolve_config(VP_CFG, {"demand.rate_per_h": demand, **s["ov"]})


def run_one(task: dict) -> dict:
    from factory_sim.engine import run_simulation
    from factory_sim.run import write_run_dir
    t0 = time.perf_counter()
    cfg = config_for(task["scenario"], task["demand"])
    res = run_simulation(cfg, task["seed"], run_id=f"seed_{task['seed']}", check_invariants=True)
    out = Path(task["out"]) / f"{task['scenario']}__{task['dkey']}" / f"seed_{task['seed']}"
    m = write_run_dir(res, out)
    f = m["factory"]
    vp = res.stats.get("vision_pick", {}).get("VP")
    st = m["stations"].get("VP") if isinstance(m.get("stations"), dict) else None
    row = {"scenario": task["scenario"], "demand": task["dkey"], "seed": task["seed"],
           "demand_fulfil_ratio": f["demand_fulfil_ratio"], "otd_rate": f["otd_rate"],
           "checks_ok": bool(m["checks"]["all_ok"]), "wall_s": time.perf_counter() - t0}
    coh = f.get("lead_time_arrival_cohort") or {}
    row["lead_p90_lb_h"] = (coh.get("p90_lower_bound_s") or float("nan")) / 3600.0   # §7.4 보조 지표 (납기 포화 보완)
    row["lead_mean_lb_h"] = (coh.get("mean_lower_bound_s") or float("nan")) / 3600.0
    if vp:
        iw = vp["in_window"]
        row.update(vp_housings=iw["housings"], vp_attempts=iw["attempts"], vp_manual=iw["manual"],
                   vp_manual_rate=iw["manual"] / iw["housings"] if iw["housings"] else None)
    if st:
        row.update(vp_util=st.get("PROCESSING"), vp_wait_operator=st.get("WAIT_OPERATOR"), vp_starved=st.get("STARVED"))
    # 작업자 대기 (FA 앞 굶주림)
    fa = m["stations"].get("FA") if isinstance(m.get("stations"), dict) else None
    if fa:
        row["fa_starved"] = fa.get("STARVED")
        row["fa_wait_operator"] = fa.get("WAIT_OPERATOR")
    return row


def paired(rows: list[dict], metric: str) -> list[dict]:
    from scipy import stats
    out = []
    by = {}
    for r in rows:
        by.setdefault((r["scenario"], r["demand"]), {})[r["seed"]] = r
    for (sc, dk), d in sorted(by.items()):
        vals = np.array([d[s][metric] for s in sorted(d)], float)
        n = len(vals)
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        t = float(stats.t.ppf(.975, n - 1)) if n > 1 else float("nan")
        row = {"scenario": sc, "demand": dk, "metric": metric, "n": n, "mean": mean, "ci_low": mean - t * se,
               "ci_high": mean + t * se}
        base = by.get(("BASE", dk))
        if sc != "BASE" and base:
            common = sorted(set(d) & set(base))
            diff = np.array([d[s][metric] - base[s][metric] for s in common], float)
            dse = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
            row.update(diff_mean=float(diff.mean()), diff_ci_low=float(diff.mean() - t * dse),
                       diff_ci_high=float(diff.mean() + t * dse))
        out.append(row)
    return out


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenarios")
    a = ap.parse_args(argv)
    for k, p in DIST.items():
        if not (ROOT / p).exists():
            raise FileNotFoundError(f"분포 파일 없음 ({k}): {p} — build_vp_dist.py로 먼저 만든다")
    lock = OUT / "confirm.lock"
    if a.phase == "confirm":
        if lock.exists():
            raise PermissionError(f"확인 실험은 이미 실행됨: {lock}")
        cfgtxt = VP_CFG.read_text(encoding="utf-8")
        if "vp_dist_dev_placeholder" in json.dumps(scenarios()):
            raise PermissionError("자리표시 분포로는 확인 실험을 하지 않는다")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"status": "started", "rev_C_vp_sha256": hashlib.sha256(cfgtxt.encode()).hexdigest(),
                                    "dist_sha256": {k: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for k, p in DIST.items()},
                                    "created_utc": datetime.now(timezone.utc).isoformat()}))
    names = a.scenarios.split(",") if a.scenarios else list(scenarios())
    exp = OUT / f"{a.phase}_{datetime.now():%Y%m%dT%H%M%S}"
    tasks = [{"scenario": s, "demand": dv, "dkey": dk, "seed": seed, "out": str(exp)}
             for s in names for dk, dv in DEMANDS.items() for seed in PHASES[a.phase]]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(run_one, tasks, chunksize=1))
    df = pd.DataFrame(rows)
    df.to_csv(exp / "runs.csv", index=False, encoding="utf-8-sig")
    summ = pd.DataFrame([r for mt in METRICS for r in paired(rows, mt)])
    summ.to_csv(exp / "summary.csv", index=False, encoding="utf-8-sig")
    extra = df.groupby(["scenario", "demand"]).agg({c: "mean" for c in df.columns if c.startswith(("vp_", "fa_"))}).reset_index() \
        if any(c.startswith("vp_") for c in df.columns) else pd.DataFrame()
    extra.to_csv(exp / "vp_summary.csv", index=False, encoding="utf-8-sig")
    meta = {"phase": a.phase, "seeds": PHASES[a.phase], "scenarios": names, "demands": DEMANDS, "n_runs": len(rows),
            "wall_s": time.time() - t0, "all_checks_ok": bool(df.checks_ok.all()),
            "rev_C_vp_sha256": hashlib.sha256(VP_CFG.read_bytes()).hexdigest(),
            "dist_sha256": {k: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for k, p in DIST.items()},
            "label": "합성 가정 기반 이산사건 시뮬레이션. 실제 공장 성능 아님"}
    (exp / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.phase == "confirm":
        st = json.loads(lock.read_text())
        st.update(status="done", out=str(exp.relative_to(ROOT)))
        lock.write_text(json.dumps(st))
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(extra.to_string(index=False))
    print("->", exp)


if __name__ == "__main__":
    main()
