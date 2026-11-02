"""BOP 로더: 합성 BOP 픽스처(단위·규약·분할·대칭), 실데이터가 있으면 LM-O dev 정합성 확인."""

from __future__ import annotations

import json
import zipfile

import imageio.v3 as iio
import numpy as np
import pytest
import trimesh
import yaml

from v3_helpers import BOP_ROOT, DATASET_PATH
from vision_guide import cad
from vision_guide import datasets as ds
from vision_guide.geometry import make_T, rotation_about, transform_points

K = [500.0, 0.0, 32.0, 0.0, 500.0, 24.0, 0.0, 0.0, 1.0]


def _fixture(root):
    """3개 객체, 장면 1개, 영상 2장 (64x48)의 최소 BOP 폴더."""
    (root / "models_eval").mkdir(parents=True)
    box = trimesh.creation.box(extents=[40.0, 20.0, 10.0])  # mm
    for oid in (1, 2, 3):
        box.export(root / "models_eval" / f"obj_{oid:06d}.ply")
    sym = make_T(rotation_about([0, 0, 1], 180.0), [0.0, 0.0, 0.0]).ravel().tolist()
    info = {"1": {"diameter": 46.0},
            "2": {"diameter": 46.0, "symmetries_discrete": [sym]},
            "3": {"diameter": 46.0, "symmetries_continuous": [{"axis": [0, 0, 1], "offset": [5.0, 0, 0]}]}}
    (root / "models_eval" / "models_info.json").write_text(json.dumps(info))
    sdir = root / "test" / "000001"
    for sub in ("depth", "mask_visib", "rgb"):
        (sdir / sub).mkdir(parents=True)
    cams, gts, ginfo, targets = {}, {}, {}, []
    for im in (0, 1):
        cams[str(im)] = {"cam_K": K, "depth_scale": 0.1}
        iio.imwrite(sdir / "depth" / f"{im:06d}.png", np.full((48, 64), 10000, dtype=np.uint16))
        iio.imwrite(sdir / "rgb" / f"{im:06d}.png", np.zeros((48, 64, 3), dtype=np.uint8))
        m = np.zeros((48, 64), dtype=np.uint8)
        m[10:20, 10:30] = 255
        iio.imwrite(sdir / "mask_visib" / f"{im:06d}_{0:06d}.png", m)
        gts[str(im)] = [{"obj_id": 1, "cam_R_m2c": np.eye(3).ravel().tolist(), "cam_t_m2c": [10.0, -20.0, 1000.0]}]
        ginfo[str(im)] = [{"visib_fract": 0.8}]
        targets.append({"scene_id": 1, "im_id": im, "obj_id": 1, "inst_count": 1})
    (sdir / "scene_camera.json").write_text(json.dumps(cams))
    (sdir / "scene_gt.json").write_text(json.dumps(gts))
    (sdir / "scene_gt_info.json").write_text(json.dumps(ginfo))
    (root / "targets.json").write_text(json.dumps(targets))
    return {"split_dir": "test", "targets_file": "targets.json", "splits": {"dev": [[1, 0]], "eval": [[1, 1]]}}


def test_fixture_units_and_conventions(tmp_path):
    split = _fixture(tmp_path)
    frames = list(ds.iter_targets(tmp_path, split, "dev", load_rgb=True))
    assert len(frames) == 1
    f = frames[0]
    assert np.allclose(f.depth_m, 1.0)  # 10000 × 0.1 / 1000
    assert np.allclose(f.gts[0]["T_cam_obj"][:3, 3], [0.01, -0.02, 1.0])
    assert f.intr.cx == 32.0 and f.intr.fx == 500.0  # BOP K 그대로
    assert f.targets == [{"obj_id": 1, "inst_count": 1}]
    assert f.masks_visib[0].sum() == 200 and f.visib_fract == [0.8]
    assert f.rgb.shape == (48, 64, 3)


def test_eval_split_is_blocked(tmp_path):
    split = _fixture(tmp_path)
    with pytest.raises(ValueError):
        next(ds.iter_targets(tmp_path, split, "eval"))
    assert len(list(ds.iter_targets(tmp_path, split, "eval", allow_eval=True))) == 1


def test_models_and_symmetries(tmp_path):
    _fixture(tmp_path)
    infos = ds.load_models_info(tmp_path)
    m1 = ds.load_model(tmp_path, 1, info=infos[1])
    assert np.allclose(np.ptp(m1.points, axis=0), [0.04, 0.02, 0.01])
    assert m1.diameter == pytest.approx(0.046) and not m1.symmetric and len(m1.symmetries) == 1
    m2 = ds.load_model(tmp_path, 2, info=infos[2])
    assert m2.symmetric and len(m2.symmetries) == 2
    syms3 = ds.get_symmetry_transformations(infos[3], 0.01)
    assert len(syms3) == int(np.ceil(np.pi / 0.01)) - 1  # BOP 규약: i = 1..steps-1
    # 연속 대칭축(offset 5 mm = 0.005 m 지점, z방향) 위의 점은 움직이지 않는다
    on_axis = np.array([[0.005, 0.0, 0.3]])
    for S in syms3[::50]:
        assert np.allclose(transform_points(S, on_axis), on_axis, atol=1e-12)


def test_make_split_deterministic_disjoint():
    ids = [(2, i) for i in range(200)]
    a = ds.make_split(ids, 7, 50, 50)
    b = ds.make_split(list(reversed(ids)), 7, 50, 50)
    assert a == b
    assert len(a["dev"]) == 50 and len(a["eval"]) == 50 and not set(a["dev"]) & set(a["eval"])


def test_make_gapped_eval_deterministic_and_gapped():
    ids = [(2, i) for i in range(300)] + [(3, i) for i in range(50)]
    dev = ds.make_split(ids, 7, 50, 50)["dev"]
    a = ds.make_gapped_eval(ids, dev, 11, 40, 5)
    b = ds.make_gapped_eval(list(reversed(ids)), list(reversed(dev)), 11, 40, 5)
    assert a == b and len(a) == 40 and not set(a) & set(dev)
    assert ds.min_split_gap(dev, a) > 5
    # 다른 scene의 같은 im_id는 간격 계산에서 빠진다
    assert ds.min_split_gap([(2, 10)], [(3, 10)]) == float("inf")
    with pytest.raises(ValueError):
        ds.make_gapped_eval(ids, dev, 11, len(ids), 5)


def test_load_frame_forbid_ids(tmp_path):
    split = _fixture(tmp_path)
    with pytest.raises(ValueError):
        ds.load_frame(tmp_path, "test", 1, 1, forbid_ids={(1, 1)})
    assert ds.load_frame(tmp_path, "test", 1, 0, forbid_ids={(1, 1)}).im_id == 0
    # dev 목록에 eval id가 섞이면 iter_targets가 막는다
    bad = dict(split, splits={"dev": [[1, 0], [1, 1]], "eval": [[1, 1]]})
    with pytest.raises(ValueError):
        list(ds.iter_targets(tmp_path, bad, "dev"))


def test_zip_slip_rejected(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with zipfile.ZipFile(z) as zf, pytest.raises(ValueError):
        ds._safe_extract(zf, tmp_path / "out")


# ---------------------------------------------------------------------------- 실데이터 (없으면 skip)
LMO = BOP_ROOT / "lmo"
needs_lmo = pytest.mark.skipif(not (LMO / "test_targets_bop19.json").exists() or not (LMO / "test").exists(),
                               reason="LM-O 데이터 없음 (python -m vision_guide.run download-bop)")


def _bop_cfg():
    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["bop"]


@needs_lmo
def test_lmo_split_matches_dataset_yaml():
    cfg = _bop_cfg()
    targets = ds.load_targets(LMO, cfg["targets_file"])
    dev = [tuple(x) for x in cfg["splits"]["dev"]]
    ev = [tuple(x) for x in cfg["splits"]["eval"]]
    assert len(dev) == 50 and len(ev) == 50 and not set(dev) & set(ev)
    assert set(dev) | set(ev) <= set(targets)
    # dev = make_split의 dev (불변), eval = dev와 간격을 둔 재생성 목록 (2026-10-30)
    assert ds.make_split(list(targets), cfg["split_seed"], cfg["n_dev"], cfg["n_eval"])["dev"] == sorted(dev)
    assert ds.make_gapped_eval(list(targets), dev, cfg["split_seed"], cfg["n_eval"], cfg["eval_gap_im"]) == \
        sorted(ev)
    with pytest.raises(ValueError):
        next(ds.iter_targets(LMO, cfg, "eval"))


@needs_lmo
def test_lmo_dev_eval_min_gap():
    """같은 scene의 dev-eval 쌍은 모두 eval_gap_im보다 떨어져 있다 (인접 프레임 누설 방지)."""
    cfg = _bop_cfg()
    dev = [tuple(x) for x in cfg["splits"]["dev"]]
    ev = [tuple(x) for x in cfg["splits"]["eval"]]
    assert ds.min_split_gap(dev, ev) > cfg["eval_gap_im"]


@needs_lmo
def test_lmo_dev_gt_silhouette_and_depth_consistent():
    """정답 자세로 렌더링한 실루엣이 mask_visib와 맞고(규약·단위 확인), depth 차이가 작다."""
    cfg = _bop_cfg()
    infos = ds.load_models_info(LMO, cfg["models_dir"])
    for f in ds.iter_targets(LMO, cfg, "dev", limit=2):
        for g, m, vf in zip(f.gts, f.masks_visib, f.visib_fract):
            if vf < 0.3:
                continue
            mdl = ds.load_model(LMO, g["obj_id"], cfg["models_dir"], info=infos[g["obj_id"]])
            d, _ = cad.render_depth([(mdl.mesh, g["T_cam_obj"])], f.intr)
            assert ((d > 0) & m).sum() / m.sum() > 0.99
            sel = m & (d > 0) & (f.depth_m > 0)
            # LM-O 실측 depth는 정답 렌더보다 수 mm 멀다 (알려진 Kinect 편향). 단위 오류(×1000)는 여기서 걸린다.
            assert abs(np.median(f.depth_m[sel] - d[sel])) < 0.015
