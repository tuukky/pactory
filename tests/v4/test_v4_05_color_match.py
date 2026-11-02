"""색 일치 검사: 유채색·무채색 판정, 원형 H, 최소 픽셀, decide_code 귀속."""

from __future__ import annotations

import numpy as np

from v4_helpers import v41_cfg
from vision_guide.color import color_match_ratio
from vision_guide.failures import FailureCode
from vision_guide.consistency import decide_code

CM = {"s_min_model": 40, "s_min_obs": 30, "achrom_s_max_obs": 90, "h_tol": 15, "min_match": 0.5, "min_px": 50}


def _rep(rgb, n=100):
    return np.tile(np.array(rgb, np.uint8), (n, 1))


def test_chromatic_and_achromatic():
    assert color_match_ratio(_rep((220, 30, 30)), _rep((200, 40, 35)), CM) == (1.0, 100)
    assert color_match_ratio(_rep((220, 30, 30)), _rep((30, 30, 220)), CM)[0] == 0.0
    assert color_match_ratio(_rep((220, 30, 30)), _rep((120, 110, 110)), CM)[0] == 0.0  # 관측 무채색
    assert color_match_ratio(_rep((128, 128, 128)), _rep((90, 95, 100)), CM)[0] == 1.0
    assert color_match_ratio(_rep((128, 128, 128)), _rep((20, 200, 20)), CM)[0] == 0.0
    r, n = color_match_ratio(np.zeros((0, 3)), np.zeros((0, 3)), CM)
    assert np.isnan(r) and n == 0


def test_hue_wraparound():
    # 빨강 H≈178 vs H≈2 → 원형 차 4 ≤ 15
    m = _rep((230, 20, 40))
    o = _rep((230, 40, 20))
    assert color_match_ratio(m, o, CM)[0] == 1.0


def _diag(**kw):
    d = {"n_valid": 500, "n_match": 400, "viol": 0.0, "color_ratio": 0.9, "color_n": 400}
    d.update(kw)
    return d


def test_decide_code_order():
    chk = v41_cfg()["checks"]
    mv = chk["depth"]["max_violation"]
    mm = chk["color_match"]["min_match"]
    assert decide_code(FailureCode.OK, 0.9, _diag(), chk) == FailureCode.OK
    assert decide_code(FailureCode.LOW_FITNESS, 0.9, _diag(viol=1.0), chk) == FailureCode.LOW_FITNESS
    assert decide_code(FailureCode.HIGH_RMSE, 0.9, _diag(), chk) == FailureCode.HIGH_RMSE
    # 깊이와 색 모두 실패 → 앞 단계(DEPTH) 귀속
    assert decide_code(FailureCode.OK, 0.9, _diag(viol=mv + 0.1, color_ratio=0.0), chk) == FailureCode.DEPTH_INCONSISTENT
    assert decide_code(FailureCode.OK, 0.9, _diag(color_ratio=mm - 0.1), chk) == FailureCode.COLOR_MISMATCH
    # 색 비교 픽셀 부족 → 판정 안 함
    assert decide_code(FailureCode.OK, 0.9, _diag(color_ratio=0.0, color_n=10), chk) == FailureCode.OK
    # 깊이 유효 픽셀 부족 → 깊이 판정 안 함
    assert decide_code(FailureCode.OK, 0.9, _diag(viol=1.0, n_valid=10), chk) == FailureCode.OK


def test_visibility_relax():
    chk = v41_cfg()["checks"]
    vr = chk["visibility_relax"]
    ok_vis = vr["min_vis_relaxed"] + 0.05
    assert decide_code(FailureCode.LOW_VISIBILITY, ok_vis, _diag(), chk) == FailureCode.OK
    assert decide_code(FailureCode.LOW_VISIBILITY, vr["min_vis_relaxed"] - 0.05, _diag(), chk) == FailureCode.LOW_VISIBILITY
    assert decide_code(FailureCode.LOW_VISIBILITY, ok_vis, _diag(n_match=vr["min_match_px"] - 1), chk) \
        == FailureCode.LOW_VISIBILITY
    assert decide_code(FailureCode.LOW_VISIBILITY, ok_vis, _diag(viol=0.99), chk) == FailureCode.LOW_VISIBILITY
    off = {**chk, "visibility_relax": {**vr, "enabled": False}}
    assert decide_code(FailureCode.LOW_VISIBILITY, ok_vis, _diag(), off) == FailureCode.LOW_VISIBILITY
