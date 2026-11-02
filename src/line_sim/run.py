"""Single-run CLI and run-directory writer.

Usage:
    python -m line_sim.run --config configs/baseline.yaml --seed 100 --output results/demo
writes results/demo/seed_100/.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import ENGINE_VERSION
from .config import ConfigError, config_hash, resolve_config
from .engine import RunResult, run_simulation
from .metrics import CSV_ENCODING, compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("simpy", "numpy", "pandas", "scipy", "PyYAML")


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=10)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def environment_info() -> dict[str, Any]:
    """Python/package versions and git state (query once per experiment and reuse)."""
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    commit = _git(["rev-parse", "HEAD"])
    porcelain = _git(["status", "--porcelain", "--", str(PROJECT_ROOT)])
    return {
        "engine_version": ENGINE_VERSION,
        "python": platform.python_version(),
        "packages": packages,
        "git_commit": None if commit is None else commit.strip(),
        "git_dirty": None if porcelain is None else bool(porcelain.strip()),
    }


def collect_metadata(*, run_id: str, seed: int, cfg: Mapping | None, status: str,
                     experiment_id: str | None = None, scenario: str | None = None, condition: str | None = None,
                     wall_time_s: float | None = None, command: str | None = None,
                     error: str | None = None, tb: str | None = None,
                     env_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    env = dict(env_info) if env_info is not None else environment_info()
    meta = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "scenario": scenario,
        "condition": condition,
        "seed": int(seed),
        "config_hash": None if cfg is None else config_hash(cfg),
        "status": status,
        "engine_version": env["engine_version"],
        "python": env["python"],
        "packages": env["packages"],
        "git_commit": env["git_commit"],
        "git_dirty": env["git_dirty"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_s": wall_time_s,
        "command": command if command is not None else " ".join(sys.argv),
    }
    if error is not None:
        meta["error"] = error
        meta["traceback"] = tb
    return meta


def write_run_dir(result: RunResult, out_dir: str | Path, *, experiment_id: str | None = None,
                  scenario: str | None = None, condition: str | None = None, wall_time_s: float | None = None,
                  command: str | None = None, env_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write all run files and return the metrics dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(result.config, fh, sort_keys=False, allow_unicode=True)
    result.orders_frame().to_csv(out / "orders.csv", index=False, encoding=CSV_ENCODING)
    result.events.to_csv(out / "events.csv", index=False, encoding=CSV_ENCODING)
    result.jobs_frame().to_csv(out / "jobs.csv", index=False, encoding=CSV_ENCODING)
    result.machine_intervals.to_csv(out / "machine_intervals.csv", index=False, encoding=CSV_ENCODING)
    metrics = compute_metrics(result)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = collect_metadata(run_id=result.run_id, seed=result.seed, cfg=result.config, status=result.status,
                            experiment_id=experiment_id, scenario=scenario, condition=condition,
                            wall_time_s=wall_time_s, command=command, env_info=env_info)
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_failed_run(out_dir: str | Path, *, run_id: str, seed: int, cfg: Mapping | None, exc: BaseException,
                     **kwargs: Any) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = collect_metadata(run_id=run_id, seed=seed, cfg=cfg, status="failed", error=repr(exc),
                            tb=traceback.format_exc(), **kwargs)
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="생산라인 시뮬레이션 단일 실행")
    parser.add_argument("--config", required=True, help="설정 YAML 경로")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True, help="출력 상위 디렉토리 (seed_<n>/ 하위에 저장)")
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
        shutil.rmtree(out_dir)  # only after the config loaded and validated
    command = "python -m line_sim.run " + " ".join(argv if argv is not None else sys.argv[1:])
    run_id = f"seed_{args.seed}"
    t0 = time.perf_counter()
    try:
        result = run_simulation(cfg, args.seed, run_id=run_id)
        metrics = write_run_dir(result, out_dir, wall_time_s=time.perf_counter() - t0, command=command)
    except Exception as exc:  # noqa: BLE001 - recorded in metadata
        write_failed_run(out_dir, run_id=run_id, seed=args.seed, cfg=cfg, exc=exc, command=command,
                         wall_time_s=time.perf_counter() - t0)
        print(f"실행 실패: {exc!r}", file=sys.stderr)
        return 1
    s = metrics["shift"]
    print(f"저장: {out_dir}")
    print(f"상태: {result.status} | 교대 내 양품 {s['good_shipped']}/{s['orders_total']} | "
          f"평균 WIP {s['avg_wip']:.2f} | 전체 종결시간 {metrics['full_run']['completion_time_s']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
