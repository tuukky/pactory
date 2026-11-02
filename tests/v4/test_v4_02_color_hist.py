"""색 분포: 단색 메시 유사도, H 원형(175 vs 3), 무채색 구간, 시드 재현."""

from __future__ import annotations

import numpy as np

from v4_helpers import colored_box
from vision_guide.color import (HistSpec, hs_histogram, model_color_hist, rank_by_similarity, rgb_to_hsv_u8,
                                similarity)


def _hsv_patch(rgb, n=500):
    return rgb_to_hsv_u8(np.tile(np.array(rgb, np.uint8), (n, 1)))


def test_same_color_max_other_low():
    spec = HistSpec()
    ref = model_color_hist(colored_box(rgb=(220, 30, 30)), spec, 5000, seed=1)
    for kind in ("bhattacharyya", "intersection"):
        same = similarity(hs_histogram(_hsv_patch((220, 30, 30)), spec), ref, kind)
        other = similarity(hs_histogram(_hsv_patch((30, 30, 220)), spec), ref, kind)
        gray = similarity(hs_histogram(_hsv_patch((128, 128, 128)), spec), ref, kind)
        assert same > 0.99
        assert other < 0.2 and gray < 0.05
        assert similarity(ref, ref, kind) > 0.999


def test_hue_circular():
    spec = HistSpec(h_bins=18, s_bins=1, s_min=40, achrom_v_bins=1, h_smooth_sigma=0.0)
    a = hs_histogram(np.array([[175, 200, 200]] * 10), spec)
    b = hs_histogram(np.array([[3, 200, 200]] * 10), spec)
    c = hs_histogram(np.array([[90, 200, 200]] * 10), spec)
    assert np.argmax(a) == np.argmax(b) == 0          # hue 0 중심 구간
    assert similarity(a, b) > 0.99 and similarity(a, c) < 0.01
    sm = HistSpec(h_bins=36, s_bins=1, s_min=40, achrom_v_bins=1, h_smooth_sigma=1.0)
    a2 = hs_histogram(np.array([[178, 200, 200]] * 10), sm)
    b2 = hs_histogram(np.array([[4, 200, 200]] * 10), sm)
    assert similarity(a2, b2) > 0.5                  # 평활도 원형으로 이어진다


def test_achromatic_bins():
    spec = HistSpec(h_bins=12, s_bins=2, s_min=40, achrom_v_bins=3, h_smooth_sigma=1.0)
    h = hs_histogram(np.array([[0, 10, 30], [0, 10, 128], [0, 10, 250], [0, 10, 250]]), spec)
    assert h.shape == (12 * 2 + 3,)
    assert np.allclose(h[:24], 0) and np.allclose(h[24:], [0.25, 0.25, 0.5])
    assert abs(h.sum() - 1) < 1e-12
    assert np.allclose(hs_histogram(np.zeros((0, 3)), spec), 0)


def test_seed_reproducible():
    spec = HistSpec()
    m = colored_box(rgb=(30, 200, 60))
    assert np.array_equal(model_color_hist(m, spec, 3000, seed=7), model_color_hist(m, spec, 3000, seed=7))


def test_rank_ties_keep_order():
    assert rank_by_similarity([0.5, 0.9, 0.5, 0.0]) == [(1, 0.9), (0, 0.5), (2, 0.5), (3, 0.0)]
