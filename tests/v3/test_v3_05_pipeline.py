"""파이프라인 종단: 간이 합성 장면에서 채택 자세의 정확도, 실패 코드."""

from __future__ import annotations

import numpy as np

from v3_helpers import T_cam_floor, camera, housing_model, pipeline_cfg
from vision_guide import cad
from vision_guide.failures import FailureCode
from vision_guide.metrics import adi
from vision_guide.pose import TIMING_KEYS, pick_best, run_pipeline


def _scene(n=3, seed=2, **kw):
    intr, T_base_cam = camera()
    return cad.make_simple_scene(n, intr.pinhole(), T_cam_floor(T_base_cam), seed=seed, **kw)


def test_accepted_poses_are_correct_on_clean_scene():
    model, cfg = housing_model(), pipeline_cfg()
    sc = _scene()
    results, seg = run_pipeline(sc.depth_m, sc.intr, model, cfg, seed=0)
    assert seg.plane_model is not None  # 빈 바닥 제거
    ok = [r for r in results if r.code == FailureCode.OK]
    assert len(ok) >= 1
    for r in ok:
        e = min(adi(r.T_cam_obj, T, model.points) for T in sc.T_cam_objs)
        assert e < 0.1 * model.diameter
        assert len(r.candidates) == 3
    assert set(results[0].timings_ms) >= {"backproject", "preprocess", "plane", "cluster", "fpfh", "ransac", "icp"}
    assert set(results[0].timings_ms) <= set(TIMING_KEYS)


def test_gt_mask_mode_single_cluster():
    model, cfg = housing_model(), pipeline_cfg()
    sc = _scene(seed=4)
    for k in range(len(sc.T_cam_objs)):
        results, _ = run_pipeline(sc.depth_m, sc.intr, model, cfg, seg="gt_mask", roi_mask=sc.inst == k + 1)
        assert len(results) == 1
        r = pick_best(results)
        if r.code == FailureCode.OK:
            assert adi(r.T_cam_obj, sc.T_cam_objs[k], model.points) < 0.1 * model.diameter


def test_failure_codes_for_empty_inputs():
    model, cfg = housing_model(), pipeline_cfg()
    intr, T_base_cam = camera()
    intr = intr.pinhole()
    empty = np.zeros((intr.height, intr.width))
    res, _ = run_pipeline(empty, intr, model, cfg)
    assert [r.code for r in res] == [FailureCode.NO_POINTS]
    floor_only, _ = cad.render_depth([(cad.make_bin_floor(), T_cam_floor(T_base_cam))], intr)
    res, _ = run_pipeline(floor_only, intr, model, cfg)
    assert [r.code for r in res] == [FailureCode.NO_CLUSTER]
