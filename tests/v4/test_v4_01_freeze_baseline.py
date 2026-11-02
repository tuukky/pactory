"""v4.0 기준선 동결: 동결본 = v3 pipeline_cfg("lmo"), v41은 선언한 v4 키만 다름, 새 실패 코드, 정리 항목."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

from v4_helpers import BASELINE_PATH, ROOT, V41_PATH, baseline_cfg, v41_cfg
from vision_guide.failures import (STAGE_OF, FailureCode, failure_breakdown, other_failure_share, other_share_ok)
from vision_guide.run import LMO_LABEL, SYNTH_BANNER, environment_info, pipeline_cfg, write_metadata

V4_KEYS = {"version", "frozen", "source", "color", "checks", "seg", "runtime"}


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in V4_KEYS}


def test_baseline_equals_v3_lmo_profile():
    base = baseline_cfg()
    v3 = pipeline_cfg("lmo")
    assert base["frozen"] is True and base["version"] == "baseline"
    assert _strip(base) == _strip(v3)
    assert base["color"]["enabled"] is False and base["checks"]["enabled"] is False
    assert base["seg"]["source"] == "dbscan" and base["runtime"]["omp_num_threads"] == 16


def test_baseline_file_matches_generator():
    from vision_guide.run_v4 import baseline_cfg_dict
    assert baseline_cfg() == baseline_cfg_dict()  # v3 pipeline.yaml이 바뀌면 실패 (동결 위반 감지)


def test_v41_differs_only_in_v4_keys():
    base, v41 = baseline_cfg(), v41_cfg()
    diff = {k for k in set(base) | set(v41) if base.get(k) != v41.get(k)}
    assert diff <= V4_KEYS
    assert v41["color"]["enabled"] and v41["checks"]["enabled"]


def test_v42_differs_only_in_v4_keys():
    from v4_helpers import v42_cfg
    base, v42 = baseline_cfg(), v42_cfg()
    diff = {k for k in set(base) | set(v42) if base.get(k) != v42.get(k)}
    assert diff <= V4_KEYS
    assert v42["seg"]["source"] == "yolo" and v42["version"] == "v42"


def test_new_failure_codes_and_stage():
    assert STAGE_OF[FailureCode.NO_DETECTION] == "segment"
    assert STAGE_OF[FailureCode.DEPTH_INCONSISTENT] == "confidence"
    assert STAGE_OF[FailureCode.COLOR_MISMATCH] == "confidence"
    assert set(STAGE_OF) == set(FailureCode)
    assert list(FailureCode)[-1] == FailureCode.OTHER


def test_other_failure_share():
    fb = failure_breakdown(["OK"] * 90 + ["OTHER"] * 2 + ["LOW_FITNESS"] * 8)
    assert abs(other_failure_share(fb) - 0.2) < 1e-12
    assert other_share_ok(fb) is True                      # 전체 대비 2%
    assert other_share_ok(fb, basis="failures") is False   # 실패 대비 20%
    assert other_failure_share(failure_breakdown(["OK"] * 3)) == 0.0


def test_metadata_label_and_threads(tmp_path):
    ns = argparse.Namespace(seed=0)
    write_metadata(tmp_path, "x", ns, [], 0.0)
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["label"] == SYNTH_BANNER
    write_metadata(tmp_path, "x", ns, [], 0.0, label=LMO_LABEL)
    meta = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert meta["label"] == LMO_LABEL and "합성 아님" in meta["label"]
    env = environment_info()
    assert "omp_num_threads" in env and "cv2_threads" in env


def test_run_v4_sets_omp_before_open3d():
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    env["PYTHONPATH"] = str(ROOT / "src")
    code = ("import sys, vision_guide.run_v4 as r, os; "
            "print(os.environ['OMP_NUM_THREADS'], r._O3D_PRELOADED)")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["16", "False"]


def test_configs_exist():
    assert BASELINE_PATH.exists() and V41_PATH.exists()
    _ = pd
