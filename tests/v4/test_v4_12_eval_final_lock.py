"""eval-final 잠금·동결 검사 (PLAN_v4 §6.4, 구현계획 §8). **실제 eval 프레임은 읽지 않는다** (tmp 경로, 평가 함수 mock).

- 미동결 설정·해시 불일치·OMP 불일치·확인 플래그 없음 → 거부, 잠금 미생성
- 정상 실행: 잠금 생성이 eval 예측·eval 프레임 첫 로드보다 먼저 (monkeypatch로 순서 기록)
- 두 번째 실행 거부 (잠금 존재), 잠금은 지우지 않음
"""

from __future__ import annotations

import argparse
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from v4_helpers import CONFIG_DIR_V4
from vision_guide import datasets as ds
from vision_guide import run_v4 as rv
from vision_guide.run_v4 import EvalFinalRefused, build_manifest, hypothesis_verdicts, verify_manifest


def _setup(tmp_path, monkeypatch, frozen=True):
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setattr(rv, "_o3d_preloaded", lambda: False)
    w = tmp_path / "best.pt"
    w.write_bytes(b"dummy yolo weights for test")
    sha = rv.sha256_file(w)
    cfgs = {}
    for v in rv.FINAL_VERSIONS:
        c = yaml.safe_load((CONFIG_DIR_V4 / f"lmo_{v}.yaml").read_text(encoding="utf-8"))
        c["frozen"] = frozen
        if v == "v42":
            c["seg"] = {**c["seg"], "weights": str(w), "weights_sha256": sha, "conf": 0.5}
        p = tmp_path / f"lmo_{v}.yaml"
        p.write_text(yaml.safe_dump(c, allow_unicode=True, sort_keys=False), encoding="utf-8")
        cfgs[v] = p
    man = tmp_path / "frozen_manifest.json"
    man.write_text(json.dumps(build_manifest(cfgs, w)), encoding="utf-8")
    args = argparse.Namespace(**{v: str(cfgs[v]) for v in rv.FINAL_VERSIONS}, manifest=str(man),
                              lock_path=str(tmp_path / "eval-final.lock"), o3d_init_seed=0, seed=0,
                              confirm_eval_once=True, out=str(tmp_path / "eval-final_out"))
    return args, cfgs, w


def _mock_eval(monkeypatch, tmp_path, log):
    lock = tmp_path / "eval-final.lock"

    def fake_iter(root, split_cfg, split="dev", allow_eval=False, **kw):
        log.append(("iter_targets", split, allow_eval, lock.exists()))
        return iter([])

    monkeypatch.setattr(ds, "iter_targets", fake_iter)
    monkeypatch.setattr(rv, "_eval_final_models", lambda ctx: log.append(("models", lock.exists())) or {})
    monkeypatch.setattr(rv, "_eval_final_outputs",
                        lambda ctx, acc: log.append(("outputs", sorted(acc))) or {"headlines": {}, "hypotheses": {}})

    def fake_predict(weights, pred_dir, lock_path, cfg42):
        log.append(("predict", lock.exists(), json.loads(lock.read_text(encoding="utf-8"))["status"]))

    return fake_predict


def test_refuse_unfrozen(tmp_path, monkeypatch):
    args, _, _ = _setup(tmp_path, monkeypatch, frozen=False)
    with pytest.raises(EvalFinalRefused, match="frozen"):
        rv.cmd_eval_final(args, predict_fn=lambda *a: pytest.fail("예측하면 안 됨"))
    assert not (tmp_path / "eval-final.lock").exists()


def test_refuse_hash_mismatch(tmp_path, monkeypatch):
    args, cfgs, w = _setup(tmp_path, monkeypatch)
    with open(cfgs["v41"], "a", encoding="utf-8") as fh:
        fh.write("# 동결 후 수정\n")
    with pytest.raises(EvalFinalRefused, match="해시 다름"):
        rv.cmd_eval_final(args, predict_fn=lambda *a: pytest.fail("예측하면 안 됨"))
    assert not (tmp_path / "eval-final.lock").exists()
    man = json.loads(open(args.manifest, encoding="utf-8").read())
    assert any("lmo_v41" in p for p in verify_manifest(man, cfgs, w))


def test_refuse_weights_changed(tmp_path, monkeypatch):
    args, _, w = _setup(tmp_path, monkeypatch)
    w.write_bytes(b"retrained after freeze")
    with pytest.raises(EvalFinalRefused):
        rv.cmd_eval_final(args, predict_fn=lambda *a: pytest.fail("예측하면 안 됨"))
    assert not (tmp_path / "eval-final.lock").exists()


def test_refuse_threads_and_confirm(tmp_path, monkeypatch):
    args, _, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with pytest.raises(EvalFinalRefused, match="OMP_NUM_THREADS"):
        rv.cmd_eval_final(args)
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setattr(rv, "_o3d_preloaded", lambda: True)
    with pytest.raises(EvalFinalRefused, match="open3d"):
        rv.cmd_eval_final(args)
    monkeypatch.setattr(rv, "_o3d_preloaded", lambda: False)
    args.confirm_eval_once = False
    with pytest.raises(EvalFinalRefused, match="confirm"):
        rv.cmd_eval_final(args)
    assert not (tmp_path / "eval-final.lock").exists()


def test_lock_before_eval_and_second_run_refused(tmp_path, monkeypatch):
    args, cfgs, w = _setup(tmp_path, monkeypatch)
    log = []
    pred = _mock_eval(monkeypatch, tmp_path, log)
    done = rv.cmd_eval_final(args, predict_fn=pred)
    names = [e[0] for e in log]
    assert names == ["predict", "models", "iter_targets", "outputs"]
    assert log[0][1] is True and log[0][2] == "started"                      # 잠금이 예측보다 먼저
    assert log[2] == ("iter_targets", "eval", True, True)                    # 잠금이 eval 프레임보다 먼저, 1회
    lock_path = tmp_path / "eval-final.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["status"] == "started" and lock["weights_sha256"] == rv.sha256_file(w)
    assert lock["omp_num_threads"] == "16" and lock["o3d_init_seed"] == 0
    assert set(lock["configs_sha256"]) == set(rv.FINAL_VERSIONS)
    assert (tmp_path / "eval-final.done.json").exists() and done["status"] == "finished"
    assert json.loads((tmp_path / "eval-final_out" / "lock.json").read_text(encoding="utf-8")) == lock
    # 두 번째 실행: 다른 출력 폴더여도 잠금이 있으면 거부, 아무것도 하지 않음
    shutil.rmtree(tmp_path / "eval-final_out")
    log.clear()
    with pytest.raises(EvalFinalRefused, match="잠금"):
        rv.cmd_eval_final(args, predict_fn=pred)
    assert log == [] and lock_path.exists()


def test_create_lock_exclusive(tmp_path):
    p = tmp_path / "x.lock"
    rv.create_lock(p, {"status": "started"})
    with pytest.raises(EvalFinalRefused):
        rv.create_lock(p, {"status": "started"})
    assert json.loads(p.read_text(encoding="utf-8")) == {"status": "started"}


def test_hypothesis_verdicts_rules():
    def row(a, b, m, est, lo, hi, coef=1.0):
        return {"a": a, "b": b, "coef": coef, "metric": m, "estimate": est, "ci_low": lo, "ci_high": hi}
    bs = pd.DataFrame([row("baseline", "v41", "rate_add_s", 0.05, 0.01, 0.09),
                       row("baseline", "v41", "false_accept_rate", -0.2, -0.3, -0.1),
                       row("v41", "v42", "rate_add_s", 0.03, -0.01, 0.07),
                       row("v41", "v42", "region_hit_rate", 0.1, 0.05, 0.15),
                       row("baseline", "v42", "false_accept_rate", -0.05, -0.12, 0.01, 0.5),
                       row("baseline", "v41", "median_total_ms", 2000.0, 1000.0, 3000.0)])
    h = hypothesis_verdicts(bs)
    assert h["H1"]["verdict"] == "지지"
    assert h["H2"]["verdict"] == "명확한 차이를 확인하지 못함" and h["H2"]["region_reason_verdict"] == "지지"
    assert h["H3"]["verdict"] == "명확한 차이를 확인하지 못함"           # 점추정은 만족, CI 상한 ≥ 0
    assert h["H4"]["verdict"].startswith("지지하지 않음")                 # 더 느림 (CI가 0 밖)
    bs.loc[4, ["estimate", "ci_high"]] = [0.01, 0.05]
    assert hypothesis_verdicts(bs)["H3"]["verdict"] == "지지하지 않음"
    bs.loc[4, ["estimate", "ci_low", "ci_high"]] = [-0.05, -0.1, -0.01]
    assert hypothesis_verdicts(bs)["H3"]["verdict"] == "지지"
    _ = np
