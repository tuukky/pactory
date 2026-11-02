"""v2 단일 실행 CLI와 run 디렉토리 저장.

Usage:
    python -m factory_sim.run --config configs/v2/rev_A.yaml --seed 1 --output results/v2/demo --detail full
writes results/v2/demo/seed_1/.

summary: config_resolved.yaml, metadata.json, metrics.json, stats.json, resource_states.csv, orders.csv
full:    summary + events.csv, resource_intervals.csv, agv_trips.csv, batches.csv, housing_ledger.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import yaml

from line_sim import run as v1_run

from . import ENGINE_VERSION
from .config import ConfigError, resolve_config
from .engine import RunResult, run_simulation
from .metrics import CSV_ENCODING, compute_metrics, top_losses


def environment_info() -> dict[str, Any]:
    """v1과 같은 방식(패키지 버전, git 상태)에 v2 엔진 버전을 넣는다."""
    info = v1_run.environment_info()
    info["engine_version"] = ENGINE_VERSION
    return info


def collect_metadata(**kwargs: Any) -> dict[str, Any]:
    env_info = kwargs.pop("env_info", None)
    meta = v1_run.collect_metadata(env_info=env_info if env_info is not None else environment_info(), **kwargs)
    return meta


def write_run_dir(result: RunResult, out_dir: str | Path, *, experiment_id: str | None = None,
                  scenario: str | None = None, condition: str | None = None, wall_time_s: float | None = None,
                  command: str | None = None, env_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """run 파일을 모두 쓰고 metrics dict를 반환한다. full 상세는 result에 있을 때만 쓴다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(result.config, fh, sort_keys=False, allow_unicode=True)
    result.orders_frame().to_csv(out / "orders.csv", index=False, encoding=CSV_ENCODING)
    result.resource_states.to_csv(out / "resource_states.csv", index=False, encoding=CSV_ENCODING)
    (out / "stats.json").write_text(json.dumps(result.stats, ensure_ascii=False), encoding="utf-8")
    if result.events is not None:
        result.events.to_csv(out / "events.csv", index=False, encoding=CSV_ENCODING)
    if result.intervals is not None:
        result.intervals.to_csv(out / "resource_intervals.csv", index=False, encoding=CSV_ENCODING)
    for name, df in result.tables.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding=CSV_ENCODING)
    meta = collect_metadata(run_id=result.run_id, seed=result.seed, cfg=result.config, status=result.status,
                            experiment_id=experiment_id, scenario=scenario, condition=condition,
                            wall_time_s=wall_time_s, command=command, env_info=env_info)
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = compute_metrics(out)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_failed_run(out_dir: str | Path, *, run_id: str, seed: int, cfg: Mapping | None, exc: BaseException,
                     **kwargs: Any) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata(run_id=run_id, seed=seed, cfg=cfg, status="failed", error=repr(exc),
                            tb=traceback.format_exc(), **kwargs)
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x * 100:.1f}%" if pct else f"{x:.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="모터 공장 v2 시뮬레이션 단일 실행")
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True, help="출력 상위 디렉토리 (seed_<n>/ 하위에 저장)")
    parser.add_argument("--detail", choices=("summary", "full"), default="summary")
    parser.add_argument("--overwrite", action="store_true", help="기존 seed 디렉토리 덮어쓰기")
    args = parser.parse_args(argv)

    out_dir = Path(args.output) / f"seed_{args.seed}"
    if out_dir.exists() and not args.overwrite:
        print(f"오류: 출력 디렉토리가 이미 존재함: {out_dir} (--overwrite 사용)", file=sys.stderr)
        return 2
    try:
        cfg = resolve_config(args.config)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if out_dir.exists():
        shutil.rmtree(out_dir)  # 설정 검증 후에만
    command = "python -m factory_sim.run " + " ".join(argv if argv is not None else sys.argv[1:])
    run_id = f"seed_{args.seed}"
    t0 = time.perf_counter()
    try:
        result = run_simulation(cfg, args.seed, run_id=run_id, detail=args.detail)
        wall = time.perf_counter() - t0
        metrics = write_run_dir(result, out_dir, wall_time_s=wall, command=command)
    except Exception as exc:  # noqa: BLE001 - metadata에 기록
        write_failed_run(out_dir, run_id=run_id, seed=args.seed, cfg=cfg, exc=exc, command=command,
                         wall_time_s=time.perf_counter() - t0)
        print(f"실행 실패: {exc!r}", file=sys.stderr)
        return 1
    f, t = metrics["factory"], metrics["targets"]
    print(f"저장: {out_dir} (실행 {wall:.1f}s)")
    print(f"상태: {result.status} | 주간 양품 {_fmt(f['weekly_good'])} (수요 {_fmt(f['weekly_demand'])}) | "
          f"수요 충족 {_fmt(f['demand_fulfil_ratio'], True)} | 납기 충족 {_fmt(f['otd_rate'], True)} | "
          f"평균 리드타임 {_fmt((f['lead_time']['mean_s'] or 0) / 3600)}h | 목표 {t['all_ok']}")
    worst = sorted(((sid, ls[0]) for sid, ls in top_losses(metrics, 1).items()), key=lambda x: -x[1][1])[:5]
    print("상위 손실: " + ", ".join(f"{sid} {name} {v * 100:.0f}%" for sid, (name, v) in worst))
    print(f"검사: {'ok' if metrics['checks']['all_ok'] else '실패'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
