"""v4.2-A 합성 학습 데이터 (호스트 쪽): 장면 계획 → BlenderProc 실행 → BOP 형식 정리 → 검사. **합성 데이터.**

흐름 (PLAN_v4 §5.1, 구현계획 §5)
1. ``configs/v4/train_synth.yaml`` (첫 생성 전 고정)에서 장면 목록을 결정적으로 만든다 (``plan_scenes``).
   train_synth / val_synth / pilot 시드 구간은 겹치지 않는다.
2. ``run_blenderproc``: 장면 ``scenes_per_process``개씩 Blender 프로세스 하나 (``bproc_scene.py``) → raw npz.
3. ``finalize_scene``: raw npz → BOP 폴더 (rgb jpg, depth png uint16 depth_scale 0.1, mask_visib(Blender 인스턴스),
   mask(대상 단독 레이캐스팅, ``consistency.ModelRaycaster`` = cad.render_depth 경로), scene_gt, scene_camera, scene_gt_info).
   방해물은 정답·마스크에 넣지 않는다 (배경). ``bproc.writer.write_bop``는 쓰지 않는다.
4. ``check_scene``: 정답 자세로 레이캐스팅한 가시 영역(렌더 depth와 일치하는 픽셀)과 mask_visib의 IoU
   → 좌표 규약(Blender −Z 카메라, 0.5 px 픽셀 중심) 오류를 잡는다.

LM-O test 영상(dev·eval)은 이 모듈이 읽지 않는다 (CAD 모델과 무작위 장면만).
하우징 bin-picking 합성 데이터(PLAN_v3 4주차)에도 같은 흐름을 재사용한다 (대상 모델 경로만 바꿔서).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTH_CONFIG = PROJECT_ROOT / "configs" / "v4" / "train_synth.yaml"
SYNTH_LABEL = "합성: BlenderProc 렌더링 (LM-O·LM CAD 모델만, 실측 영상 미사용)"
SPLITS = ("train_synth", "val_synth", "pilot")
BPROC_SCRIPT = Path(__file__).resolve().parent / "bproc_scene.py"


def load_synth_config(path: str | Path = SYNTH_CONFIG) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def n_scenes_for(cfg: Mapping[str, Any], split: str, decision: str | None = None) -> int:
    """규모 결정(target/fallback)에 따른 장면 수. pilot은 pilot.scenes."""
    sc = cfg["scale"]
    vps = int(cfg["camera"]["views_per_scene"])
    if split == "pilot":
        return int(sc["pilot"]["scenes"])
    dec = decision or (sc.get("scale_decision") or {}).get("choice")
    if dec not in ("target", "fallback"):
        raise ValueError("scale.scale_decision.choice가 기록되지 않았다 (파일럿 후 사전 등록 규칙으로 결정)")
    key = "train_images" if split == "train_synth" else "val_images"
    return int(np.ceil(int(sc[dec][key]) / vps))


def plan_scenes(cfg: Mapping[str, Any], split: str, n_scenes: int) -> list[dict]:
    """결정적 장면 목록 {scene_id, split, seed, n_frames}. 시드 = base + split offset + 장면 번호."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}: {split!r}")
    s = cfg["seeds"]
    off = {"train_synth": s["train_offset"], "val_synth": s["val_offset"], "pilot": s["pilot_offset"]}[split]
    lim = {"train_synth": s["val_offset"] - s["train_offset"], "val_synth": s["pilot_offset"] - s["val_offset"],
           "pilot": 10 ** 9}[split]
    if n_scenes > lim:
        raise ValueError(f"{split}: 장면 수 {n_scenes}가 시드 구간 {lim}을 넘는다 (구간 겹침)")
    vps = int(cfg["camera"]["views_per_scene"])
    return [{"scene_id": i, "split": split, "seed": int(s["base"]) + int(off) + i, "n_frames": vps}
            for i in range(int(n_scenes))]


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def raw_path(cfg: Mapping[str, Any], sc: Mapping[str, Any]) -> Path:
    return _abs(cfg["output"]["root"]) / cfg["output"]["raw_dir"] / f"{sc['split']}_{int(sc['scene_id']):06d}.npz"


def split_dir(cfg: Mapping[str, Any], split: str) -> Path:
    return _abs(cfg["output"]["root"]) / cfg["output"]["bop_dir"] / split


def make_job(cfg: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> dict:
    ob = json.loads(json.dumps(cfg["objects"]))
    for k in ("targets", "distractors"):
        ob[k]["root_abs"] = str(_abs(ob[k]["root"]))
    return {"objects": ob, "scene": cfg["scene"], "camera": cfg["camera"], "render": cfg["render"],
            "scenes": [{**dict(s), "raw_path": str(raw_path(cfg, s))} for s in scenes]}


def run_blenderproc(cfg: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]], *, blenderproc_exe: str | None = None,
                    blender_path: str | None = None, dry_run: bool = False, log_dir: str | Path | None = None,
                    on_chunk_done=None) -> list[Path]:
    """장면을 scenes_per_process개씩 Blender 프로세스로 렌더링 → raw npz 경로 목록 (이미 있는 장면은 건너뜀)."""
    tools = cfg["tools"]
    exe = blenderproc_exe or tools["blenderproc_exe"]
    bpath = blender_path or tools["blender_install_path"]
    per = int(tools["scenes_per_process"])
    rdir = _abs(cfg["output"]["root"]) / cfg["output"]["raw_dir"]
    rdir.mkdir(parents=True, exist_ok=True)
    ldir = Path(log_dir) if log_dir else rdir
    ldir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for c0 in range(0, len(scenes), per):
        chunk = list(scenes[c0:c0 + per])
        todo = [s for s in chunk if not raw_path(cfg, s).exists() and not _finalized(cfg, s)]
        if todo:
            job = make_job(cfg, todo)
            jpath = rdir / f"job_{todo[0]['split']}_{int(todo[0]['scene_id']):06d}.json"
            jpath.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
            cmd = [exe, "run", "--blender-install-path", str(bpath), str(BPROC_SCRIPT), "--job", str(jpath)]
            if dry_run:
                print(" ".join(cmd))
            else:
                log = ldir / f"bproc_{todo[0]['split']}_{int(todo[0]['scene_id']):06d}.log"
                t0 = time.time()
                with open(log, "w", encoding="utf-8", errors="replace") as fh:
                    r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
                print(f"[synth] chunk {todo[0]['split']} {todo[0]['scene_id']}..{todo[-1]['scene_id']}: "
                      f"exit {r.returncode}, {time.time() - t0:.0f}s (log {log})", flush=True)
                if r.returncode != 0:
                    raise RuntimeError(f"blenderproc 실패 (exit {r.returncode}), 로그: {log}")
        paths = [raw_path(cfg, s) for s in chunk]
        out.extend(paths)
        if on_chunk_done is not None and not dry_run:
            on_chunk_done(chunk)
    return out


def _finalized(cfg: Mapping[str, Any], sc: Mapping[str, Any]) -> bool:
    return (split_dir(cfg, sc["split"]) / f"{int(sc['scene_id']):06d}" / "scene_gt_info.json").exists()


# ---------------------------------------------------------------------------- BOP 정리
def load_target_meshes(cfg: Mapping[str, Any]) -> dict:
    """대상 물체 정점 색 메시 (m) — 단독 레이캐스팅(mask, visib_fract)과 검사용."""
    import trimesh
    root = _abs(cfg["objects"]["targets"]["root"])
    out = {}
    for oid in cfg["objects"]["targets"]["obj_ids"]:
        m = trimesh.load(root / f"obj_{int(oid):06d}.ply", process=False, force="mesh")
        m.apply_scale(float(cfg["objects"]["unit_scale"]))
        out[int(oid)] = m
    return out


def _full_render(raycaster, T, intr):
    """ModelRaycaster의 bbox 렌더를 전체 영상 z-깊이로 (미검출 0)."""
    r = raycaster.render(T, intr)
    z = np.zeros((intr.height, intr.width))
    if not r.empty:
        v0, v1, u0, u1 = r.bbox
        z[v0:v1, u0:u1] = r.depth
    return z


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [-1, -1, -1, -1]
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def finalize_scene(raw_npz: str | Path, meshes: Mapping[int, Any], out_split_dir: str | Path, depth_scale: float = 0.1,
                   jpg_quality: int = 95, raycasters: dict | None = None) -> Path:
    """raw npz → BOP 장면 폴더 (``out_split_dir/{scene:06d}``). 대상 슬롯만 정답에 넣는다."""
    import cv2

    from .consistency import ModelRaycaster
    from .geometry import CameraIntrinsics

    z = np.load(raw_npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    rgb, depth, inst = z["rgb"], z["depth"], z["inst"]
    T, K = z["T_cam_obj"], z["K"]
    oids, is_t = z["slot_obj_ids"], z["slot_is_target"]
    N, H, W = depth.shape
    intr = CameraIntrinsics.from_K(K, W, H)
    sdir = Path(out_split_dir) / f"{int(meta['scene_id']):06d}"
    for sub in ("rgb", "depth", "mask", "mask_visib"):
        (sdir / sub).mkdir(parents=True, exist_ok=True)
    rc = raycasters if raycasters is not None else {}
    tslots = [k for k in range(len(oids)) if bool(is_t[k])]
    scene_gt, scene_cam, scene_info = {}, {}, {}
    for f in range(N):
        cv2.imwrite(str(sdir / "rgb" / f"{f:06d}.jpg"), cv2.cvtColor(rgb[f], cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
        dmm = depth[f].astype(np.float64) * 1000.0 / depth_scale
        dmm[(dmm < 0) | (dmm > 65535)] = 0
        cv2.imwrite(str(sdir / "depth" / f"{f:06d}.png"), np.round(dmm).astype(np.uint16))
        scene_cam[str(f)] = {"cam_K": K.reshape(-1).tolist(), "depth_scale": float(depth_scale)}
        gts, infos = [], []
        for gi, k in enumerate(tslots):
            oid = int(oids[k])
            Tk = T[f, k]
            gts.append({"cam_R_m2c": Tk[:3, :3].reshape(-1).tolist(), "cam_t_m2c": (Tk[:3, 3] * 1000.0).tolist(),
                        "obj_id": oid})
            if oid not in rc:
                rc[oid] = ModelRaycaster(meshes[oid])
            amodal = _full_render(rc[oid], Tk, intr) > 0
            visib = inst[f] == k
            cv2.imwrite(str(sdir / "mask" / f"{f:06d}_{gi:06d}.png"), amodal.astype(np.uint8) * 255)
            cv2.imwrite(str(sdir / "mask_visib" / f"{f:06d}_{gi:06d}.png"), visib.astype(np.uint8) * 255)
            n_all, n_vis = int(amodal.sum()), int(visib.sum())
            infos.append({"bbox_obj": _bbox(amodal), "bbox_visib": _bbox(visib), "px_count_all": n_all,
                          "px_count_valid": int((amodal & (depth[f] > 0)).sum()), "px_count_visib": n_vis,
                          "visib_fract": float(n_vis / n_all) if n_all > 0 else 0.0})
        scene_gt[str(f)] = gts
        scene_info[str(f)] = infos
    for name, obj in (("scene_camera.json", scene_cam), ("scene_gt.json", scene_gt)):
        (sdir / name).write_text(json.dumps(obj), encoding="utf-8")
    (sdir / "scene_meta.json").write_text(json.dumps({**meta, "label": SYNTH_LABEL}, ensure_ascii=False),
                                          encoding="utf-8")
    # 완료 표시 = scene_gt_info.json (마지막에 쓴다)
    (sdir / "scene_gt_info.json").write_text(json.dumps(scene_info), encoding="utf-8")
    return sdir


def check_scene(split_dir_: str | Path, scene_id: int, meshes: Mapping[int, Any], n_frames: int = 5,
                tol_m: float = 0.003, min_px: int = 100) -> dict:
    """정답 자세 레이캐스팅 가시 영역(렌더 depth와 tol 이내로 일치) vs mask_visib IoU, BOP 로더로 읽기."""
    from .consistency import ModelRaycaster
    from .datasets import load_frame
    sdir = Path(split_dir_)
    rows = []
    rc = {}
    for f in range(int(n_frames)):
        fr = load_frame(sdir.parent, sdir.name, int(scene_id), f, load_rgb=True)
        for k, g in enumerate(fr.gts):
            oid = g["obj_id"]
            if oid not in rc:
                rc[oid] = ModelRaycaster(meshes[oid])
            z = _full_render(rc[oid], g["T_cam_obj"], fr.intr)
            pred = (z > 0) & (fr.depth_m > 0) & (np.abs(fr.depth_m - z) < tol_m)
            m = fr.masks_visib[k]
            union = int((pred | m).sum())
            if int(m.sum()) < min_px:
                continue
            rows.append({"im_id": f, "obj_id": oid, "n_visib": int(m.sum()),
                         "iou": float((pred & m).sum() / union) if union else float("nan")})
    ious = np.array([r["iou"] for r in rows], float)
    return {"scene_id": int(scene_id), "n": len(rows), "iou_mean": float(np.nanmean(ious)) if len(ious) else float("nan"),
            "iou_min": float(np.nanmin(ious)) if len(ious) else float("nan"),
            "iou_median": float(np.nanmedian(ious)) if len(ious) else float("nan"), "rows": rows,
            "rgb_shape": None if fr.rgb is None else list(fr.rgb.shape)}


# ---------------------------------------------------------------------------- 실행 (run_v4 synth-gen / synth-check)
def generate(cfg: Mapping[str, Any], split: str, n_scenes: int | None = None, *, dry_run: bool = False,
             keep_raw: bool | None = None, log_dir: str | Path | None = None) -> dict:
    """장면 계획 → 렌더 → 청크마다 BOP 정리 (raw 삭제). 이미 정리된 장면은 건너뛴다 (재개)."""
    n = n_scenes if n_scenes is not None else n_scenes_for(cfg, split)
    scenes = plan_scenes(cfg, split, n)
    meshes = load_target_meshes(cfg)
    rcs: dict = {}
    keep = bool(cfg["output"].get("keep_raw", False)) if keep_raw is None else keep_raw
    out_dir = split_dir(cfg, split)
    t0 = time.time()

    def done(chunk):
        for s in chunk:
            rp = raw_path(cfg, s)
            if rp.exists() and not _finalized(cfg, s):
                finalize_scene(rp, meshes, out_dir, float(cfg["render"]["depth_scale"]),
                               int(cfg["render"]["jpg_quality"]), rcs)
            if rp.exists() and not keep and _finalized(cfg, s):
                rp.unlink()
        print(f"[synth] {split}: {sum(_finalized(cfg, s) for s in scenes)}/{len(scenes)} scenes, "
              f"{time.time() - t0:.0f}s", flush=True)

    run_blenderproc(cfg, scenes, dry_run=dry_run, log_dir=log_dir, on_chunk_done=done)
    return {"split": split, "n_scenes": len(scenes), "n_images": len(scenes) * int(cfg["camera"]["views_per_scene"]),
            "wall_s": time.time() - t0, "out_dir": str(out_dir)}


if __name__ == "__main__":  # pragma: no cover
    print("run via: python -m vision_guide.run_v4 synth-gen|synth-check", file=sys.stderr)
    _ = os
