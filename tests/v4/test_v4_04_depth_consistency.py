"""깊이 일관성: Raycaster == cad.render_depth, 정답 위반 ≈ 0, 앞으로 옮긴 자세 위반 증가, 가림물은 occluded,
LM-O dev 정답 자세 위반 작음 (데이터 없으면 skip)."""

from __future__ import annotations

import numpy as np
import pytest

from v4_helpers import HAS_LMO, LMO_ROOT, colored_box, intr_small, pose_at
from vision_guide import cad
from vision_guide.consistency import ModelRaycaster, depth_consistency
from vision_guide.geometry import make_T, rotation_about


def _full(r, intr):
    full = np.zeros((intr.height, intr.width))
    v0, v1, u0, u1 = r.bbox
    full[v0:v1, u0:u1] = r.depth
    return full


def test_raycaster_matches_render_depth():
    intr = intr_small()
    box = colored_box()
    T = make_T(rotation_about([1, 1, 0], 35.0), [0.01, -0.005, 0.45])
    ref, _ = cad.render_depth([(box, T)], intr)
    r = ModelRaycaster(box).render(T, intr)
    assert np.max(np.abs(_full(r, intr) - ref)) < 1e-5
    assert r.colors is not None and (r.colors[r.depth > 0] == [220, 30, 30]).all()


def test_true_pose_no_violation_and_forward_shift():
    intr = intr_small()
    box = colored_box()
    T = make_T(rotation_about([0, 1, 0], 20.0), [0.0, 0.0, 0.5])
    plate = cad.make_bin_floor((0.5, 0.5))  # 물체 뒤 배경 평면 (z=0.53)
    obs, _ = cad.render_depth([(box, T), (plate, pose_at(0.53))], intr)
    rc = ModelRaycaster(box)
    good = depth_consistency(rc.render(T, intr), obs, 0.010, 0.015)
    assert good.n_valid > 100 and good.violation_ratio < 0.01 and good.match_ratio > 0.98
    T_fwd = T.copy()
    T_fwd[2, 3] -= 0.03      # 카메라 쪽으로 3 cm → 관측이 모델보다 뒤
    bad = depth_consistency(rc.render(T_fwd, intr), obs, 0.010, 0.015)
    assert bad.violation_ratio > 0.9


def test_occluder_counts_as_occluded():
    intr = intr_small()
    box = colored_box()
    T = pose_at(0.5)
    occ = cad.make_bin_floor((0.03, 0.2))
    obs, _ = cad.render_depth([(box, T), (occ, pose_at(0.40))], intr)
    d = depth_consistency(ModelRaycaster(box).render(T, intr), obs, 0.010, 0.015)
    assert d.n_occluded > 50 and d.violation_ratio < 0.01
    assert abs(d.explained_ratio - 1.0) < 1e-9


@pytest.mark.skipif(not HAS_LMO, reason="LM-O 데이터 없음")
@pytest.mark.xfail(strict=True, reason=(
    "사후 기록 (2026-10-30): 사전에 정한 기준 '정답 자세 위반 중앙값 ≤ 0.05 (tol_free 15 mm)'를 dev 첫 프레임에서 "
    "만족하지 못함 (중앙값 0.064, can 0.20). 위반은 실루엣 경계와 Kinect depth의 +3~5 mm 편향, can(금속)에서 큼. "
    "기준을 느슨하게 바꾸지 않고 실패로 남긴다 (docs/v4/verification.md §2). tol_free·max_violation은 dev 격자로 정한다."))
def test_lmo_dev_true_pose_violation_small():
    from vision_guide.datasets import iter_targets
    from vision_guide.pipeline_v4 import load_color_mesh
    from vision_guide.run import _bop_cfg
    cfg = _bop_cfg()
    fr = next(iter_targets(LMO_ROOT, cfg, "dev", limit=1, load_rgb=False))  # dev 첫 프레임만
    viols = []
    for g in fr.gts:
        rc = ModelRaycaster(load_color_mesh(LMO_ROOT, g["obj_id"]))
        d = depth_consistency(rc.render(g["T_cam_obj"], fr.intr), fr.depth_m, 0.010, 0.015)
        if d.n_valid >= 50:
            viols.append(d.violation_ratio)
    assert len(viols) >= 3
    assert float(np.median(viols)) <= 0.05
