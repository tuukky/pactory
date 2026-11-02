"""v4.2-A 합성 데이터 (합성): 장면 계획 결정성·시드 구간 분리, raw npz → BOP 정리 → 로더, mask_visib vs 정답 자세 레이캐스팅."""

from __future__ import annotations

import json

import numpy as np
import pytest

from v4_helpers import SYNTH_PILOT, fake_synth_npz
from vision_guide.datasets import load_frame
from vision_guide.synth import check_scene, finalize_scene, load_synth_config, n_scenes_for, plan_scenes


def test_plan_scenes_deterministic_and_disjoint():
    cfg = load_synth_config()
    a = plan_scenes(cfg, "train_synth", 250)
    assert a == plan_scenes(cfg, "train_synth", 250)
    v = plan_scenes(cfg, "val_synth", 25)
    p = plan_scenes(cfg, "pilot", 1)
    seeds = [s["seed"] for s in a], [s["seed"] for s in v], [s["seed"] for s in p]
    assert not (set(seeds[0]) & set(seeds[1])) and not (set(seeds[0]) & set(seeds[2])) and not (set(seeds[1]) & set(seeds[2]))
    assert all(s["n_frames"] == cfg["camera"]["views_per_scene"] for s in a)
    with pytest.raises(ValueError):
        plan_scenes(cfg, "eval", 1)


def test_scale_decision_recorded_before_generation():
    cfg = load_synth_config()
    assert cfg["frozen_before_generation"] is True
    dec = cfg["scale"]["scale_decision"]
    assert dec["choice"] in ("target", "fallback")
    # 사전 등록 규칙과 기록이 맞는지: t × 5,500 > budget → fallback
    want = "fallback" if dec["pilot_sec_per_image"] * 5500 > cfg["scale"]["budget_s"] else "target"
    assert dec["choice"] == want
    assert n_scenes_for(cfg, "train_synth") * cfg["camera"]["views_per_scene"] == cfg["scale"][want]["train_images"]


def test_finalize_and_loader(tmp_path):
    raw = tmp_path / "raw.npz"
    meshes, intr = fake_synth_npz(raw, n_frames=2)
    sdir = finalize_scene(raw, meshes, tmp_path / "bop" / "train_synth")
    for name in ("scene_gt.json", "scene_camera.json", "scene_gt_info.json", "scene_meta.json"):
        assert (sdir / name).exists()
    gt = json.loads((sdir / "scene_gt.json").read_text(encoding="utf-8"))
    assert [g["obj_id"] for g in gt["0"]] == [1, 5]           # 방해물(2)은 정답에 없음
    info = json.loads((sdir / "scene_gt_info.json").read_text(encoding="utf-8"))
    assert 0 < info["0"][1]["visib_fract"] < 1.0                # 파란 상자는 일부 가려짐
    fr = load_frame(tmp_path / "bop", "train_synth", 0, 1, load_rgb=True)
    assert fr.rgb.shape == (intr.height, intr.width, 3) and len(fr.gts) == 2
    z = np.load(raw)
    assert np.allclose(fr.gts[0]["T_cam_obj"], z["T_cam_obj"][1, 0], atol=1e-9)
    assert np.allclose(fr.depth_m, z["depth"][1], atol=1e-4)    # depth_scale 0.1 mm 양자화
    assert fr.masks_visib[0].sum() == int((z["inst"][1] == 0).sum())


def test_mask_visib_matches_raycast(tmp_path):
    raw = tmp_path / "raw.npz"
    meshes, _ = fake_synth_npz(raw, n_frames=2)
    finalize_scene(raw, meshes, tmp_path / "bop" / "train_synth")
    res = check_scene(tmp_path / "bop" / "train_synth", 0, meshes, n_frames=2, min_px=10)
    assert res["n"] == 4 and res["iou_min"] >= 0.99


@pytest.mark.skipif(not (SYNTH_PILOT / "000000" / "scene_gt_info.json").exists(), reason="합성 파일럿 데이터 없음")
def test_real_pilot_scene_iou():
    """BlenderProc 실제 출력 (파일럿 장면): 좌표 규약(Blender −Z 카메라, 0.5 px 픽셀 중심)이 맞으면 IoU ≥ 0.95."""
    from vision_guide.synth import load_target_meshes
    cfg = load_synth_config()
    res = check_scene(SYNTH_PILOT, 0, load_target_meshes(cfg), n_frames=5)
    assert res["n"] > 0 and res["iou_mean"] >= 0.95
