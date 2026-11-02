"""v4 검증 테스트 공통 도우미 (conftest 없이 import, v3_helpers와 이름이 겹치지 않게)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from vision_guide.geometry import CameraIntrinsics, make_T

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR_V4 = ROOT / "configs" / "v4"
BASELINE_PATH = CONFIG_DIR_V4 / "lmo_baseline.yaml"
V41_PATH = CONFIG_DIR_V4 / "lmo_v41.yaml"
LMO_ROOT = ROOT / "data" / "v3" / "bop" / "lmo"
HAS_LMO = (LMO_ROOT / "test" / "000002").exists() and (LMO_ROOT / "models" / "obj_000001.ply").exists()


def intr_small() -> CameraIntrinsics:
    """빠른 테스트용 160x120 카메라 (합성 가정)."""
    return CameraIntrinsics(fx=160.0, fy=160.0, cx=80.0, cy=60.0, width=160, height=120)


def colored_box(extents=(0.06, 0.04, 0.03), rgb=(220, 30, 30)) -> trimesh.Trimesh:
    """단색 정점 색 상자 (m)."""
    m = trimesh.creation.box(extents=extents)
    m.visual.vertex_colors = np.tile(np.array([*rgb, 255], np.uint8), (len(m.vertices), 1))
    return m


def v41_cfg() -> dict:
    import yaml
    with open(V41_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def baseline_cfg() -> dict:
    import yaml
    with open(BASELINE_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fake_frame(depth_m, intr, rgb=None, im_id=0):
    return SimpleNamespace(depth_m=depth_m, intr=intr, rgb=rgb, im_id=im_id, scene_id=1)


def pose_at(z=0.5, x=0.0, y=0.0, R=None) -> np.ndarray:
    return make_T(R, [x, y, z])


V42_PATH = CONFIG_DIR_V4 / "lmo_v42.yaml"
SYNTH_PILOT = ROOT / "data" / "v4" / "synth" / "bop" / "pilot"


def v42_cfg() -> dict:
    import yaml
    with open(V42_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fake_synth_npz(path, n_frames: int = 2):
    """bproc_scene.py 출력과 같은 형식의 가짜 raw npz (합성). 대상 2개(obj 1 빨강, obj 5 파랑, 일부 가림) + 방해물 1개.

    렌더는 cad.render_depth (같은 레이캐스팅 규약) → finalize 결과 mask_visib vs 레이캐스팅 IoU ≈ 1."""
    import json
    from vision_guide import cad
    intr = intr_small()
    meshes = {1: colored_box((0.05, 0.04, 0.03), (220, 30, 30)), 5: colored_box((0.04, 0.04, 0.05), (30, 40, 220))}
    dis = colored_box((0.03, 0.03, 0.03), (30, 200, 30))
    slots = [(1, True, meshes[1]), (5, True, meshes[5]), (2, False, dis)]
    rgbs, depths, insts, Ts = [], [], [], []
    for f in range(n_frames):
        T = [pose_at(0.45 + 0.02 * f, x=-0.02), pose_at(0.5, x=0.02), pose_at(0.40, x=0.05, y=0.03)]
        depth, inst = cad.render_depth([(m, t) for (_, _, m), t in zip(slots, T)], intr)
        rgb = np.zeros((intr.height, intr.width, 3), np.uint8)
        for k, (_, _, m) in enumerate(slots):
            rgb[inst == k] = np.asarray(m.visual.vertex_colors)[0, :3]
        rgbs.append(rgb)
        depths.append(depth.astype(np.float32))
        insts.append(inst.astype(np.int16))
        Ts.append(np.stack(T))
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1.0]])
    meta = {"scene_id": 0, "split": "train_synth", "seed": 1, "slots": ["tgt_1", "tgt_5", "dis_2"]}
    np.savez_compressed(path, rgb=np.stack(rgbs), depth=np.stack(depths), inst=np.stack(insts),
                        slot_obj_ids=np.array([1, 5, 2], np.int32), slot_is_target=np.array([True, True, False]),
                        T_cam_obj=np.stack(Ts), cam2world=np.tile(np.eye(4), (n_frames, 1, 1)), K=K,
                        obj2world=np.stack([np.eye(4)] * 3), meta=json.dumps(meta))
    return meshes, intr
