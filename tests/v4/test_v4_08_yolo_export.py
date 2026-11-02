"""v4.2-B YOLO-seg 변환 (합성): 폴리곤 ↔ 래스터 왕복, 여러 조각 병합, 클래스 매핑, data.yaml."""

from __future__ import annotations

import numpy as np

from v4_helpers import fake_synth_npz
from vision_guide.seg_model import (CLASS_IDS, CLASS_NAMES, export_yolo, mask_to_polygon, polygon_to_mask,
                                    write_data_yaml)


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def test_polygon_roundtrip_single():
    h, w = 120, 160
    yy, xx = np.mgrid[:h, :w]
    m = ((xx - 70) ** 2 / 30 ** 2 + (yy - 60) ** 2 / 20 ** 2) <= 1
    poly = mask_to_polygon(m)
    assert poly is not None and poly.min() >= 0 and poly.max() <= 1
    assert _iou(polygon_to_mask(poly, h, w), m) >= 0.9


def test_polygon_merges_pieces():
    h, w = 120, 160
    m = np.zeros((h, w), bool)
    m[10:40, 10:50] = True
    m[60:100, 90:150] = True
    m[5:7, 150:152] = True                     # 면적 4 < 16 → 버림
    poly = mask_to_polygon(m, min_area_px=16)
    r = polygon_to_mask(poly, h, w)
    assert r[20, 30] and r[80, 120]            # 두 조각 모두 덮음
    assert _iou(r, m) >= 0.9
    assert mask_to_polygon(np.zeros((h, w), bool)) is None


def test_export_yolo_class_mapping(tmp_path):
    from vision_guide.synth import finalize_scene
    raw = tmp_path / "raw.npz"
    meshes, intr = fake_synth_npz(raw, n_frames=2)
    finalize_scene(raw, meshes, tmp_path / "bop" / "train_synth")
    res = export_yolo(tmp_path / "bop" / "train_synth", tmp_path / "yolo", "train", min_px=10)
    assert res["n_images"] == 2 and res["n_instances"] == 4
    lab = sorted((tmp_path / "yolo" / "labels" / "train").glob("*.txt"))
    img = sorted((tmp_path / "yolo" / "images" / "train").glob("*.jpg"))
    assert len(lab) == 2 and [p.stem for p in lab] == [p.stem for p in img]
    lines = lab[0].read_text(encoding="utf-8").strip().splitlines()
    cls = [int(ln.split()[0]) for ln in lines]
    assert cls == [CLASS_IDS.index(1), CLASS_IDS.index(5)]   # obj 1 → 0, obj 5 → 1; 방해물 없음
    vals = np.array([float(v) for v in lines[0].split()[1:]]).reshape(-1, 2)
    assert vals.min() >= 0 and vals.max() <= 1
    # 높은 가시 기준이면 제외
    res2 = export_yolo(tmp_path / "bop" / "train_synth", tmp_path / "yolo2", "train", min_visib_fract=1.01, min_px=10)
    assert res2["n_instances"] == 0 and res2["n_images"] == 2
    dy = write_data_yaml(tmp_path / "yolo")
    txt = dy.read_text(encoding="utf-8")
    assert "train: images/train" in txt and "val: images/val" in txt
    assert all(f"{i}: {n}" in txt for i, n in enumerate(CLASS_NAMES))
    _ = intr
