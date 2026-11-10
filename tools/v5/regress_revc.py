"""PLAN_v5 V-R2 — Rev.C 확인 실험 재현 검사.

v5에서 factory_sim을 확장(VP 셀)한 뒤에도 기존 Rev.C 결과가 바뀌지 않았는지 확인한다.
동결 결과 ``results/v2/p2_final_confirm2_*/C__FINAL__p2_base/seed_N``과 같은 설정·시드로 다시 실행하고
``orders.csv``(바이트)와 ``metrics.json``의 factory 블록(값)을 비교한다.

    python tools/v5/regress_revc.py --seeds 2000-2029 --out results/v5/regress_revc.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FROZEN = sorted((ROOT / "results/v2").glob("p2_final_confirm2_*"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def rev_c_config() -> dict:
    from factory_sim.experiments import load_matrix, resolve_scenario_config, scenarios

    matrix = load_matrix(ROOT / "configs/v2/experiments.yaml")
    sc = scenarios(matrix, "p2_final")["FINAL"]
    return resolve_scenario_config(matrix, sc, "p2_base")


def check_seed(seed: int) -> dict:
    from factory_sim.engine import run_simulation
    from factory_sim.run import write_run_dir

    frozen_dir = FROZEN[-1] / "C__FINAL__p2_base" / f"seed_{seed}"
    with tempfile.TemporaryDirectory() as td:
        res = run_simulation(rev_c_config(), seed, run_id=f"seed_{seed}")
        metrics = write_run_dir(res, td)
        new_orders = _sha(Path(td) / "orders.csv")
    old_orders = _sha(frozen_dir / "orders.csv")
    old_factory = json.loads((frozen_dir / "metrics.json").read_text(encoding="utf-8"))["factory"]
    same_factory = json.dumps(old_factory, sort_keys=True) == json.dumps(metrics["factory"], sort_keys=True)
    return {"seed": seed, "orders_csv_identical": new_orders == old_orders, "factory_metrics_identical": same_factory,
            "demand_fulfil_ratio": metrics["factory"]["demand_fulfil_ratio"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", default="2000-2029")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if not FROZEN:
        print("동결 결과(results/v2/p2_final_confirm2_*)가 없음", file=sys.stderr)
        return 2
    seeds = _parse_seeds(a.seeds)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(check_seed, seeds))
    ok = all(r["orders_csv_identical"] and r["factory_metrics_identical"] for r in rows)
    report = {"check": "V-R2 Rev.C p2_base 재현", "frozen_dir": str(FROZEN[-1].relative_to(ROOT)),
              "seeds": seeds, "all_identical": ok, "rows": rows}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if a.out:
        Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / a.out).write_text(text, encoding="utf-8")
    print(f"V-R2: {'통과' if ok else '불일치'} ({sum(r['orders_csv_identical'] for r in rows)}/{len(rows)} 시드 orders.csv 일치)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
