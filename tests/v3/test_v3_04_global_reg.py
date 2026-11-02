"""§7 FPFH+RANSAC: 큰 초기 회전(최대 180°)에서 ICP 수렴 범위로 들어오는 비율.

낮은 sanity 하한만 둔다. 실제 비율은 synth-demo CLI 결과(global_reg.csv)로 기록한다. 결과를 보고 하한을 바꾸지 않는다.
"""

from __future__ import annotations

import numpy as np

from v3_helpers import housing_model, pipeline_cfg, rng
from vision_guide.geometry import make_T, random_rotation, transform_points
from vision_guide.metrics import re_sym_deg
from vision_guide.pose import compute_fpfh, global_register
from vision_guide.segment import preprocess

# ICP 수렴 범위 (합성 가정): 대칭 고려 회전 15° 이내, 병진 1 cm 이내
CONV_ROT_DEG = 15.0
CONV_TRANS_M = 0.01


def global_reg_trial(model, cfg, r) -> tuple[float, float]:
    T_true = make_T(random_rotation(r), r.uniform(-0.1, 0.1, 3) + [0, 0, 0.8])  # T_cam_obj
    pts = transform_points(T_true, model.points) + r.normal(scale=0.0005, size=model.points.shape)
    src = preprocess(pts, cfg)
    tgt, tgt_f = model.features(cfg)
    g = global_register(src, compute_fpfh(src, cfg), tgt, tgt_f, cfg)
    T_est = np.linalg.inv(np.asarray(g.transformation))  # T_cam_obj 추정
    re = re_sym_deg(T_est[:3, :3], T_true[:3, :3], model.symmetries)
    return re, float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))


def test_global_registration_enters_icp_basin():
    model, cfg = housing_model(), pipeline_cfg()
    r = rng(42)
    res = [global_reg_trial(model, cfg, r) for _ in range(12)]
    inside = [re < CONV_ROT_DEG and te < CONV_TRANS_M for re, te in res]
    print(f"global reg basin rate = {np.mean(inside):.2f} ({sum(inside)}/{len(inside)})")
    assert np.mean(inside) >= 0.5  # sanity 하한
