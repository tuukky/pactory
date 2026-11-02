"""v4.2 융합: NO_DETECTION, 클래스·conf·점수 순 마스크 선택, 처음 통과 채택, dbscan 대체 꺼짐, 예측 파일 → 프레임,
설정 키, 실제 정합 경로 (합성 하우징 장면)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from v3_helpers import T_cam_floor, camera, housing_model  # noqa: E402

from v4_helpers import baseline_cfg, fake_frame, v42_cfg  # noqa: E402
from vision_guide import cad  # noqa: E402
from vision_guide import pipeline_v4 as pv  # noqa: E402
from vision_guide.failures import FailureCode  # noqa: E402
from vision_guide.pipeline_v4 import (CandidateEval, FrameCache, V4Model, class_masks, erode_mask, method_of,  # noqa: E402
                                      prepare_frame, run_target, select_first_pass)
from vision_guide.pose import PoseResult  # noqa: E402
from vision_guide.seg_model import frame_pred_path, save_frame_predictions  # noqa: E402


def _cfg(**seg):
    c = v42_cfg()
    c["checks"] = {**c["checks"], "enabled": False}       # 융합 규칙만 시험 (검사는 test 04~06)
    c["seg"] = {**c["seg"], "conf": 0.5, "erode_px": 0, "max_instances_per_class": 3, **seg}
    c["ransac"] = {**c["ransac"], "n_restarts": 2}
    return c


def _scene():
    intr, T_base_cam = camera()
    sc = cad.make_simple_scene(2, intr.pinhole(), T_cam_floor(T_base_cam), seed=3)
    return sc


def _model():
    return V4Model(housing_model(), SimpleNamespace(obj_id=1))


def test_v42_config_and_method():
    c = v42_cfg()
    assert method_of(c) == "v42" and c["seg"]["source"] == "yolo" and c["seg"]["fallback_dbscan"] is False
    v41 = __import__("v4_helpers").v41_cfg()
    assert c["color"] == v41["color"] and c["checks"] == v41["checks"]    # v4.1 검사 값 그대로
    assert method_of(baseline_cfg()) == "baseline"


def test_class_masks_select():
    m = np.ones((4, 4), bool)
    preds = [{"obj_id": 1, "score": 0.6, "mask": m}, {"obj_id": 5, "score": 0.99, "mask": m},
             {"obj_id": 1, "score": 0.9, "mask": m}, {"obj_id": 1, "score": 0.4, "mask": m},
             {"obj_id": 1, "score": 0.7, "mask": m}]
    sel = class_masks(preds, 1, 0.5, 3)
    assert [p["score"] for p in sel] == [0.9, 0.7, 0.6]
    assert [p["score"] for p in class_masks(preds, 1, 0.5, 1)] == [0.9]
    assert class_masks(preds, 8, 0.0, 3) == [] and class_masks(None, 1, 0.0, 3) == []


def test_erode():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    assert erode_mask(m, 0).sum() == 100 and erode_mask(m, 2).sum() == 36


def test_no_detection_and_fallback_off():
    sc = _scene()
    fc = FrameCache(fake_frame(sc.depth_m, sc.intr), preds=[{"obj_id": 5, "score": 0.9, "mask": sc.inst == 1}],
                    seg_infer_ms=11.0)
    fc.seg_all = SimpleNamespace(clusters=["X"], n_input=1)   # dbscan 결과가 있어도 대체하지 않음 (플래그 꺼짐)
    tr = run_target("v42", fc, _model(), _cfg(), seed=0, seg="yolo")
    assert tr.code == FailureCode.NO_DETECTION and tr.candidates == [] and tr.best.T_cam_obj is None
    assert tr.timings_ms["seg_infer"] == 11.0
    # conf 미달도 NO_DETECTION
    fc.preds = [{"obj_id": 1, "score": 0.3, "mask": sc.inst == 1}]
    assert run_target("v42", fc, _model(), _cfg(), seed=0, seg="yolo").code == FailureCode.NO_DETECTION


def test_first_pass_in_score_order(monkeypatch):
    """점수 순 시도, 처음 OK에서 멈춤 (실시간). compute_all이면 모두 시도하지만 채택은 같다."""
    sc = _scene()
    calls = []

    def fake_est(c, om, cfg, *, depth_m, intr, seed):
        n = len(c.points)
        calls.append(n)
        ok = n > 300
        return PoseResult(np.eye(4), 0.9 if ok else 0.2, 0.001, 0.9, FailureCode.OK if ok else FailureCode.LOW_FITNESS)

    monkeypatch.setattr(pv, "estimate_pose", fake_est)
    small = np.zeros_like(sc.inst, bool)
    ys, xs = np.nonzero(sc.inst == 1)
    small[ys[:40], xs[:40]] = True                       # 점이 적은 엉뚱한 마스크 (점수 최고)
    preds = [{"obj_id": 1, "score": 0.95, "mask": small}, {"obj_id": 1, "score": 0.8, "mask": sc.inst == 1},
             {"obj_id": 1, "score": 0.7, "mask": sc.inst == 2}]
    fc = FrameCache(fake_frame(sc.depth_m, sc.intr), preds=preds, seg_infer_ms=5.0)
    tr = run_target("v42", fc, _model(), _cfg(), seed=0, seg="yolo")
    assert tr.code == FailureCode.OK and len(tr.candidates) == 2 and tr.best is tr.candidates[1].pose
    assert len(tr.tried_regions) == 2 and len(calls) == 2
    calls.clear()
    tr_all = run_target("v42", fc, _model(), _cfg(), seed=0, seg="yolo", compute_all=True)
    assert len(tr_all.candidates) == 3 and tr_all.best is tr_all.candidates[1].pose
    tr1 = run_target("v42", fc, _model(), _cfg(max_instances_per_class=1), seed=0, seg="yolo")
    assert tr1.code == FailureCode.LOW_FITNESS and len(tr1.candidates) == 1


def test_select_first_pass_rule():
    def cand(code, fit, rank):
        p = PoseResult(np.eye(4), fit, 0.001, 1.0, code)
        return CandidateEval(rank, "yolo", rank, float("nan"), p, {}, code)
    c = [cand(FailureCode.LOW_FITNESS, 0.95, 0), cand(FailureCode.OK, 0.7, 1), cand(FailureCode.OK, 0.9, 2)]
    best, code = select_first_pass(c)
    assert code == FailureCode.OK and best is c[1].pose            # 점수(fitness) 최고가 아니라 처음 통과
    best, code = select_first_pass([cand(FailureCode.LOW_FITNESS, 0.5, 0), cand(FailureCode.HIGH_RMSE, 0.8, 1)])
    assert code == FailureCode.HIGH_RMSE                             # 모두 실패 → pick_best 규칙으로 귀속


def test_prepare_frame_reads_predictions(tmp_path):
    sc = _scene()
    save_frame_predictions(frame_pred_path(tmp_path, 1, 0), [1], [0.8], (sc.inst == 1)[None], 7.5)
    fc = prepare_frame(fake_frame(sc.depth_m, sc.intr), _cfg(), method="v42", seg="yolo", preds_dir=tmp_path)
    assert fc.seg_all is None and fc.seg_infer_ms == 7.5 and len(fc.preds) == 1
    assert np.array_equal(fc.preds[0]["mask"], sc.inst == 1)


def test_real_registration_on_mask():
    """실제 정합 경로: 정답 마스크를 예측으로 주면 하우징 자세가 맞는다 (합성 장면)."""
    from vision_guide.metrics import adi
    sc = _scene()
    m = V4Model(housing_model(), SimpleNamespace(obj_id=1))
    fc = FrameCache(fake_frame(sc.depth_m, sc.intr), preds=[{"obj_id": 1, "score": 0.9, "mask": sc.inst == 1}])
    tr = run_target("v42", fc, m, _cfg(), seed=0, seg="yolo")
    assert tr.best.T_cam_obj is not None
    err = adi(tr.best.T_cam_obj, sc.T_cam_objs[0], m.om.points)
    assert err < 0.1 * m.om.diameter, err
