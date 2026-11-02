"""v4.2 예측 파일: npz 왕복, 점수 정렬, eval 예측 가드 (잠금·상태·해시·폴더), ultralytics 없이 import, conf 규칙."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from v4_helpers import ROOT
from vision_guide.seg_model import (check_eval_guard, conf_from_f1, load_frame_predictions, predict_split,
                                    save_frame_predictions, frame_pred_path, weights_sha256)


def test_roundtrip_sorted(tmp_path):
    rng = np.random.default_rng(0)
    masks = rng.random((3, 48, 64)) > 0.5
    save_frame_predictions(frame_pred_path(tmp_path, 2, 36), [1, 5, 1], [0.3, 0.9, 0.6], masks, 12.5)
    preds, ms = load_frame_predictions(tmp_path, 2, 36)
    assert ms == 12.5 and [p["score"] for p in preds] == pytest.approx([0.9, 0.6, 0.3])
    assert [p["obj_id"] for p in preds] == [5, 1, 1]
    assert np.array_equal(preds[0]["mask"], masks[1]) and np.array_equal(preds[2]["mask"], masks[0])
    save_frame_predictions(frame_pred_path(tmp_path, 2, 47), [], [], np.zeros((0, 48, 64), bool), 3.0)
    assert load_frame_predictions(tmp_path, 2, 47) == ([], 3.0)
    with pytest.raises(FileNotFoundError):
        load_frame_predictions(tmp_path, 2, 999)


def _lock(tmp_path, w, status="started", sha=None):
    p = tmp_path / "eval-final.lock"
    p.write_text(json.dumps({"status": status, "weights_sha256": sha or weights_sha256(w)}), encoding="utf-8")
    return p


def test_eval_guard(tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"dummy weights")
    out = tmp_path / "pred_eval"
    with pytest.raises(PermissionError):
        check_eval_guard(None, w, out)
    with pytest.raises(PermissionError):
        check_eval_guard(tmp_path / "nope.lock", w, out)
    with pytest.raises(PermissionError):
        check_eval_guard(_lock(tmp_path, w, status="finished"), w, out)
    with pytest.raises(PermissionError):
        check_eval_guard(_lock(tmp_path, w, sha="0" * 64), w, out)
    lk = _lock(tmp_path, w)
    assert check_eval_guard(lk, w, out)["status"] == "started"
    out.mkdir()
    with pytest.raises(PermissionError):
        check_eval_guard(lk, w, out)         # 이미 예측함 → 1회만


def test_predict_eval_refused_before_heavy_imports(tmp_path):
    w = tmp_path / "best.pt"
    w.write_bytes(b"x")
    mods = set(sys.modules)
    with pytest.raises(PermissionError):
        predict_split(w, "eval", tmp_path / "o", conf=0.05)
    assert "ultralytics" not in set(sys.modules) - mods
    assert not (tmp_path / "o").exists()


def test_import_without_ultralytics():
    code = ("import sys; import vision_guide.seg_model as s; "
            "print('ultralytics' in sys.modules, 'torch' in sys.modules, 'open3d' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False", "False"]


def test_conf_from_f1_rounding():
    px = np.linspace(0, 1, 1000)
    f1 = np.stack([np.exp(-((px - 0.43) / 0.2) ** 2), np.exp(-((px - 0.47) / 0.2) ** 2)])
    r = conf_from_f1(px, f1)
    assert abs(r["conf_raw"] - 0.45) < 0.01 and r["conf"] == 0.45
    assert conf_from_f1(px, np.exp(-((px - 0.62) / 0.1) ** 2))["conf"] == 0.6
