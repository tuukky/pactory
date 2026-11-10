"""PLAN_v5 S1 — 하우징 bin 합성 데이터 생성·검사·평가 (호스트, conda env ``n8n``).

    python -m vision_guide.bin_v5 export-mesh
    python -m vision_guide.bin_v5 render --split dev|eval|train|val [--limit N]
    python -m vision_guide.bin_v5 check --split dev [--limit N]          # V-S1 렌더 규약 검사
    python -m vision_guide.bin_v5 export-yolo                            # train·val → YOLO-seg 데이터
    python -m vision_guide.bin_v5 eval --split dev --pipeline B0|B1 [--conditions base,n20] [--preds DIR]
    python -m vision_guide.bin_v5 eval-final --pipeline B0,B1           # eval 1회 (잠금), handoff 작성

YOLO 학습·예측은 ``vision_guide.seg_bin_v5`` (env ``pactory-seg``)에서 한다.
모든 데이터는 합성 (BlenderProc, 합성 하우징 CAD)이다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = PROJECT_ROOT / "configs" / "v5" / "dataset_bin.yaml"
CAMERA_CFG = PROJECT_ROOT / "configs" / "v3" / "camera.yaml"
GRIPPER_CFG = PROJECT_ROOT / "configs" / "v3" / "gripper.yaml"
PIPE_CFG = {"B0": PROJECT_ROOT / "configs" / "v5" / "pipeline_bin_B0.yaml",
            "B1": PROJECT_ROOT / "configs" / "v5" / "pipeline_bin_B1.yaml"}
BPROC_SCRIPT = Path(__file__).resolve().parent / "bproc_bin_scene.py"
RESULTS = PROJECT_ROOT / "results" / "v5" / "s1_bin"
LABEL = "합성: BlenderProc 렌더링 (합성 하우징 CAD) — 실측·실제 장비 시험 아님"


# ---------------------------------------------------------------------------- 설정·공통
def _yaml(path: str | Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def file_sha256(*paths: str | Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def load_cfg(path: str | Path = CFG_PATH) -> dict:
    return _yaml(path)


def camera():
    from .geometry import load_camera
    intr, T_base_cam = load_camera(CAMERA_CFG)
    return intr.pinhole(), T_base_cam


def housing_model(n_points: int = 6000):
    """정합용 ObjectModel (synth-demo와 같은 방식) + 메시."""
    from . import cad
    from .pose import ObjectModel
    spec = cad.HousingSpec()
    mesh = cad.make_housing(spec)
    pts, nrm = cad.sample_model_points(cad.housing_parts(spec), int(n_points), seed=0)
    om = ObjectModel("housing", pts, nrm, cad.model_diameter(mesh), cad.housing_symmetries(spec),
                     np.array([0.0, 0.0, 1.0]))
    return om, mesh


def mesh_path(cfg: Mapping[str, Any]) -> Path:
    return _abs(cfg["output"]["root"]) / "housing_m.ply"


def export_mesh(cfg: Mapping[str, Any]) -> Path:
    _, mesh = housing_model()
    p = mesh_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(p)
    return p


def render_set_names(cfg: Mapping[str, Any]) -> list[str]:
    return list(cfg["renders"])


def condition_names(cfg: Mapping[str, Any]) -> list[str]:
    return list(cfg["conditions"])


# ---------------------------------------------------------------------------- 장면 계획
def _look_at_cv(eye, target, up) -> np.ndarray:
    from .geometry import look_at
    return look_at(eye, target, up)


def plan_scenes(cfg: Mapping[str, Any], split: str) -> list[dict]:
    """장면 목록. dev·eval: 렌더 세트마다 고정 카메라 1뷰. train·val: 무작위 적재·조명, 카메라 흔들림 여러 뷰."""
    _, T_base_cam = camera()
    sp = cfg["splits"][split]
    scenes = []
    if split in ("dev", "eval"):
        for ri, rs in enumerate(render_set_names(cfg)):
            r = cfg["renders"][rs]
            for i in range(int(sp["scenes_per_render"])):
                seed = int(sp["seed_base"]) + ri * 1000 + i
                scenes.append({"split": split, "render_set": rs, "scene_id": ri * 1000 + i, "seed": seed,
                               "n_housings": int(r["n_housings"]), "lighting": r["lighting"],
                               "cam2world_cv": [T_base_cam.tolist()]})
        return scenes
    cc = _yaml(CAMERA_CFG)["extrinsics"]
    jit = cfg["camera"]["train_jitter"]
    for i in range(int(sp["scenes"])):
        seed = int(sp["seed_base"]) + i
        rng = np.random.default_rng(seed)
        n = int(rng.integers(int(sp["n_housings"][0]), int(sp["n_housings"][1]) + 1))
        light = "side" if rng.random() < float(sp["p_side_light"]) else "base"
        cams = []
        for _v in range(int(jit["views_per_scene"])):
            eye = np.asarray(cc["position_m"], float) + rng.uniform(-1, 1, 3) * float(jit["position_m"])
            tgt = np.asarray(cc["look_at_m"], float) + rng.uniform(-1, 1, 3) * np.array([1.0, 1.0, 0.0]) * float(jit["look_at_m"])
            cams.append(_look_at_cv(eye, tgt, cc["image_up"]).tolist())
        scenes.append({"split": split, "render_set": "train_mix", "scene_id": i, "seed": seed, "n_housings": n,
                       "lighting": light, "cam2world_cv": cams})
    return scenes


def raw_path(cfg: Mapping[str, Any], sc: Mapping[str, Any]) -> Path:
    return _abs(cfg["output"]["root"]) / cfg["output"]["raw_dir"] / sc["split"] / f"{int(sc['scene_id']):06d}.npz"


def make_job(cfg: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> dict:
    intr, _ = camera()
    return {"housing_mesh": str(mesh_path(cfg)), "housing_material": cfg["housing"]["material"], "bin": cfg["bin"],
            "lighting": cfg["lighting"], "render": cfg["render"], "K": intr.K().tolist(), "width": intr.width,
            "height": intr.height, "scenes": [{**dict(s), "raw_path": str(raw_path(cfg, s))} for s in scenes]}


def blenderproc_cmd(tools: Mapping[str, Any]) -> list[str]:
    """``blenderproc.exe`` 런처는 Windows 앱 제어 정책이 막는다(2026-11-09 확인, WinError 4551).
    같은 env의 python으로 blenderproc 패키지의 command_line.py를 직접 실행한다 (BlenderProc이 허용하는 시작 방식)."""
    exe = Path(tools["blenderproc_exe"])
    py = exe.parent.parent / "python.exe"
    cli = exe.parent.parent / "Lib" / "site-packages" / "blenderproc" / "command_line.py"
    return [str(py), str(cli)]


def run_render(cfg: Mapping[str, Any], split: str, *, limit: int | None = None, dry_run: bool = False) -> list[Path]:
    if not mesh_path(cfg).exists():
        export_mesh(cfg)
    scenes = plan_scenes(cfg, split)
    if limit is not None:
        scenes = scenes[:limit]
    tools = cfg["tools"]
    per = int(tools["scenes_per_process"])
    rdir = raw_path(cfg, scenes[0]).parent
    rdir.mkdir(parents=True, exist_ok=True)
    out = []
    for c0 in range(0, len(scenes), per):
        chunk = scenes[c0:c0 + per]
        todo = [s for s in chunk if not raw_path(cfg, s).exists()]
        if todo:
            jpath = rdir / f"job_{split}_{int(todo[0]['scene_id']):06d}.json"
            jpath.write_text(json.dumps(make_job(cfg, todo), ensure_ascii=False), encoding="utf-8")
            cmd = blenderproc_cmd(tools) + ["run", "--blender-install-path", tools["blender_install_path"],
                                            str(BPROC_SCRIPT), "--job", str(jpath)]
            if dry_run:
                print(" ".join(cmd))
            else:
                log = rdir / f"bproc_{split}_{int(todo[0]['scene_id']):06d}.log"
                t0 = time.time()
                with open(log, "w", encoding="utf-8", errors="replace") as fh:
                    r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
                print(f"[bin_v5] render {split} {todo[0]['scene_id']}..{todo[-1]['scene_id']}: exit {r.returncode}, "
                      f"{time.time() - t0:.0f}s", flush=True)
                if r.returncode != 0:
                    raise RuntimeError(f"blenderproc 실패 (exit {r.returncode}), 로그: {log}")
        out.extend(raw_path(cfg, s) for s in chunk)
    return out


def load_raw(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in ("rgb", "depth", "inst", "T_base_obj", "cam2world_cv")}
    d["meta"] = json.loads(str(z["meta"]))
    return d


# ---------------------------------------------------------------------------- 깊이 조건 (호스트 후처리)
def condition_seed(cfg: Mapping[str, Any], scene_seed: int, cond: str) -> int:
    return int(scene_seed) * 100 + condition_names(cfg).index(cond)


def apply_depth_condition(depth: np.ndarray, inst: np.ndarray, noise_mm: float, dropout: float, seed: int) -> np.ndarray:
    """노이즈 N(0,σ²) (유효 픽셀) + 하우징 픽셀 결측 (저주파 잡음장 상위 p 분위, 덩어리 모양)."""
    import cv2
    rng = np.random.default_rng(int(seed))
    d = depth.astype(np.float64).copy()
    valid = d > 0
    if noise_mm > 0:
        d[valid] += rng.normal(0.0, noise_mm * 1e-3, int(valid.sum()))
    obj = (inst >= 0) & (d > 0)
    if dropout > 0 and obj.any():
        field = cv2.GaussianBlur(rng.standard_normal(d.shape).astype(np.float32), (0, 0), 6.0)
        thr = np.quantile(field[obj], 1.0 - float(dropout))
        d[obj & (field >= thr)] = 0.0
    return d


# ---------------------------------------------------------------------------- 정답·가시
def gt_cam_poses(raw: Mapping[str, Any], view: int = 0) -> list[np.ndarray]:
    from .geometry import inv_T
    T_cam_base = inv_T(np.asarray(raw["cam2world_cv"][view]))
    return [T_cam_base @ T for T in raw["T_base_obj"]]


def visibility(raw: Mapping[str, Any], mesh, intr, view: int = 0) -> list[dict]:
    """하우징별 보이는 픽셀 수 / 가림 없을 때 픽셀 수, 카메라 깊이."""
    from .cad import render_depth
    out = []
    inst = raw["inst"][view]
    for k, T in enumerate(gt_cam_poses(raw, view)):
        dep, _ = render_depth([(mesh, T)], intr)
        full = int((dep > 0).sum())
        vis = int((inst == k).sum())
        out.append({"k": k, "px_visib": vis, "px_full": full, "visib_fract": vis / full if full else 0.0,
                    "z_cam_m": float(T[2, 3])})
    return out


def check_scene(raw: Mapping[str, Any], mesh, intr, view: int = 0, tol_m: float = 0.002) -> dict:
    """V-S1: 정답 자세로 레이캐스팅한 하우징 depth ↔ Blender depth (하우징 픽셀)."""
    from .cad import render_depth
    Ts = gt_cam_poses(raw, view)
    dep, ins = render_depth([(mesh, T) for T in Ts], intr)
    inst = raw["inst"][view]
    bd = raw["depth"][view]
    m = (inst >= 0) & (bd > 0)
    agree = m & (ins == inst) & (np.abs(dep - bd) <= tol_m)
    ious = []
    for k in range(len(Ts)):
        a, b = inst == k, ins == k
        u = (a | b).sum()
        if a.sum() >= 200:
            ious.append(float((a & b).sum() / u))
    return {"px_housing": int(m.sum()), "agree_frac": float(agree.sum() / max(1, m.sum())),
            "min_iou_200px": float(min(ious)) if ious else None, "n_iou": len(ious)}


# ---------------------------------------------------------------------------- 평가 (B0 / B1)
def pipeline_cfg(name: str) -> dict:
    return _yaml(PIPE_CFG[name])


def bin_roi_mask(depth: np.ndarray, intr, T_base_cam: np.ndarray, cfg: Mapping[str, Any],
                 roi: Mapping[str, Any] | None) -> np.ndarray | None:
    """PLAN_v3 파이프라인 2단계 'ROI 자르기(빈 내부)': 역투영 점을 base 좌표로 옮겨 빈 내부(벽에서 margin 안쪽,
    바닥 위 z 범위)만 남기는 픽셀 마스크. roi가 없으면 None (전체 영상)."""
    if not roi:
        return None
    from .geometry import pixel_centers, normalized_rays
    rays = normalized_rays(pixel_centers(intr), intr).reshape(intr.height, intr.width, 3)
    p_cam = rays * depth[..., None]
    p = p_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    b = cfg["bin"]
    cx, cy, cz = b["center_m"]
    hx, hy = b["inner_size_xy_m"][0] / 2 - float(roi["margin_m"]), b["inner_size_xy_m"][1] / 2 - float(roi["margin_m"])
    z = p[..., 2] - cz
    return ((depth > 0) & (np.abs(p[..., 0] - cx) <= hx) & (np.abs(p[..., 1] - cy) <= hy)
            & (z >= float(roi["z_min_m"])) & (z <= float(roi["z_max_m"])))


def _seg_preds(preds_dir: str | Path | None, scene_id: int, view: int = 0):
    from .seg_model import load_frame_predictions
    return load_frame_predictions(preds_dir, scene_id, view)


def _codes_summary(codes: Sequence[str]) -> str:
    return ";".join(codes)


_WORKER: dict = {}


def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    from .consistency import ModelRaycaster
    from .guidance import load_gripper
    intr, T_base_cam = camera()
    om, mesh = housing_model()
    gripper, ws = load_gripper(GRIPPER_CFG)
    _WORKER.update(intr=intr, T_base_cam=T_base_cam, om=om, mesh=mesh, gripper=gripper, ws=ws,
                   raycaster=ModelRaycaster(mesh), cfg=load_cfg())


def eval_scene(task: Mapping[str, Any]) -> dict:
    """한 장면 × 한 조건 × 한 파이프라인 → 집기 결정 1개 (dataset_bin.yaml pick_rule)."""
    if not _WORKER:
        _worker_init()
    W = _WORKER
    from .failures import FailureCode
    from .guidance import select_grasp, tcp_error_sym
    from .metrics import adi
    from .pipeline_v4 import erode_mask, gt_mask_cluster
    from .pose import estimate_pose, run_pipeline
    cfg, pcfg = W["cfg"], task["pcfg"]
    intr, T_base_cam, om = W["intr"], W["T_base_cam"], W["om"]
    raw = load_raw(task["raw"])
    cond = cfg["conditions"][task["condition"]]
    depth = apply_depth_condition(raw["depth"][0], raw["inst"][0], float(cond["noise_mm"]), float(cond["dropout"]),
                                  condition_seed(cfg, raw["meta"]["seed"], task["condition"]))
    vis = visibility(raw, W["mesh"], intr)
    T_gt = gt_cam_poses(raw)
    seed = int(pcfg.get("seed", 0)) + int(raw["meta"]["scene_id"])
    tried: list[dict] = []
    pick = None
    elapsed = 0.0
    t_wall = time.perf_counter()
    roi = bin_roi_mask(depth, intr, T_base_cam, cfg, pcfg.get("roi_bin"))
    if task["pipeline"] == "B0":
        results, _ = run_pipeline(depth, intr, om, pcfg, seed=seed, roi_mask=roi)
        for i, r in enumerate(results):
            elapsed += r.total_ms
            dec = select_grasp(T_base_cam, r.candidates, W["gripper"], W["ws"]) if r.code == FailureCode.OK else None
            code = r.code.value if dec is None else (r.code.value if dec.code == FailureCode.OK else dec.code.value)
            tried.append({"rank": i, "code": code})
            if dec is not None and dec.code == FailureCode.OK:
                pick = (r, dec, elapsed)
                break
    else:
        from .consistency import apply_v41_checks
        sg = pcfg["seg"]
        preds, infer_ms = _seg_preds(task["preds"], raw["meta"]["scene_id"])
        elapsed += float(infer_ms)
        cand = [p for p in preds if float(p["score"]) >= float(sg["conf"])]
        if roi is not None:
            cand = [{**p, "mask": p["mask"] & roi} for p in cand if (p["mask"] & roi).any()]
        fr = SimpleNamespace(depth_m=depth, intr=intr)
        # 위쪽(카메라에 가까운) 마스크부터: 마스크 안 유효 깊이 중앙값 오름차순
        keyed = []
        for j, p in enumerate(cand):
            dz = depth[p["mask"] & (depth > 0)]
            keyed.append((float(np.median(dz)) if len(dz) else np.inf, j, p))
        keyed.sort(key=lambda t: (t[0], t[1]))
        if not keyed:
            tried.append({"rank": 0, "code": FailureCode.NO_DETECTION.value})
        for rank, (_, _, p) in enumerate(keyed[:int(sg["max_instances"])]):
            pcd, t, n_in = gt_mask_cluster(fr, pcfg, erode_mask(p["mask"], int(sg.get("erode_px", 0))))
            elapsed += sum(t.values())
            if pcd is None or len(pcd.points) == 0:
                tried.append({"rank": rank, "code": FailureCode.NO_POINTS.value})
                continue
            r = estimate_pose(pcd, om, pcfg, depth_m=depth, intr=intr, seed=seed + rank)
            t0 = time.perf_counter()
            code, _diag = apply_v41_checks(r, W["raycaster"], depth, None, intr, pcfg)
            elapsed += r.total_ms + (time.perf_counter() - t0) * 1e3
            dec = select_grasp(T_base_cam, r.candidates, W["gripper"], W["ws"]) if code == FailureCode.OK else None
            c = code.value if dec is None else (code.value if dec.code == FailureCode.OK else dec.code.value)
            tried.append({"rank": rank, "code": c})
            if dec is not None and dec.code == FailureCode.OK:
                pick = (r, dec, elapsed)
                break
    wall_ms = (time.perf_counter() - t_wall) * 1e3
    visible = [v for v in vis if v["px_visib"] > 0]
    top = min((v for v in visible if v["visib_fract"] >= 0.5), key=lambda v: v["z_cam_m"], default=None)
    row = {"split": task["split"], "pipeline": task["pipeline"], "condition": task["condition"],
           "render_set": raw["meta"]["render_set"], "scene_id": int(raw["meta"]["scene_id"]),
           "scene_seed": int(raw["meta"]["seed"]), "n_housings": int(raw["meta"]["n_housings"]),
           "n_tried": len(tried), "codes": _codes_summary([t["code"] for t in tried]),
           "first_code": tried[0]["code"] if tried else "NONE", "accepted": pick is not None,
           "vision_ms": float(pick[2] if pick else elapsed), "wall_ms": wall_ms,
           "top_gt_k": None if top is None else int(top["k"])}
    rec = {"T_base_obj_top_gt": None if top is None else np.asarray(raw["T_base_obj"][top["k"]]).tolist()}
    if pick is not None:
        r, dec, _ = pick
        errs = [adi(r.T_cam_obj, T, om.points) for T in T_gt]
        k = int(np.argmin(errs))
        G = W["gripper"].grasps[dec.grasp_idx]
        pe, ae = tcp_error_sym(dec.T_base_tcp, T_base_cam, T_gt[k], om.symmetries, G, W["gripper"].approach_axis_tcp)
        ok = bool(pe <= W["gripper"].tol_pos_m and ae <= W["gripper"].tol_axis_deg)
        T_cam_est = r.candidates[dec.sym_idx] if r.candidates else r.T_cam_obj
        row.update(match_k=k, match_visib_fract=vis[k]["visib_fract"], adi_mm=float(errs[k] * 1e3),
                   adds_correct=bool(errs[k] < 0.1 * om.diameter), tcp_pos_err_mm=float(pe * 1e3),
                   tcp_axis_err_deg=float(ae), correct_pick=ok, false_accept=not ok,
                   grasp_name=W["gripper"].grasp_names[dec.grasp_idx], tilt_deg=float(dec.tilt_deg))
        rec.update(T_base_obj_gt=np.asarray(raw["T_base_obj"][k]).tolist(),
                   T_base_obj_est=(T_base_cam @ T_cam_est).tolist(), T_base_tcp_est=np.asarray(dec.T_base_tcp).tolist(),
                   grasp_idx=int(dec.grasp_idx), fitness=float(r.fitness), rmse_m=float(r.rmse),
                   visible_ratio=float(r.visible_ratio))
    else:
        row.update(correct_pick=False, false_accept=False)
    row["_handoff"] = rec
    return row


def run_eval(split: str, pipelines: Sequence[str], conditions: Sequence[str] | None, out: Path, *,
             preds: Mapping[str, str | None] | None = None, workers: int = 8, limit: int | None = None) -> list[dict]:
    import pandas as pd
    cfg = load_cfg()
    conds = list(conditions) if conditions else condition_names(cfg)
    scenes = plan_scenes(cfg, split)
    tasks = []
    for pl in pipelines:
        pcfg = pipeline_cfg(pl)
        for c in conds:
            rs = cfg["conditions"][c]["render"]
            sel = [s for s in scenes if s["render_set"] == rs]
            if limit is not None:
                sel = sel[:limit]
            for s in sel:
                tasks.append({"split": split, "pipeline": pl, "condition": c, "raw": str(raw_path(cfg, s)),
                              "pcfg": pcfg, "preds": (preds or {}).get(pl)})
    missing = [t["raw"] for t in tasks if not Path(t["raw"]).exists()]
    if missing:
        raise FileNotFoundError(f"렌더가 없는 장면 {len(missing)}개 (예: {missing[0]})")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        rows = list(ex.map(eval_scene, tasks, chunksize=1))
    out.mkdir(parents=True, exist_ok=True)
    handoff = [{**{k: r[k] for k in ("split", "pipeline", "condition", "scene_id", "scene_seed", "render_set",
                                        "accepted", "correct_pick", "false_accept", "vision_ms")},
                **{k: r.get(k) for k in ("tcp_pos_err_mm", "tcp_axis_err_deg", "grasp_name", "first_code", "codes")},
                **r["_handoff"]} for r in rows]
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_handoff"} for r in rows])
    df.to_csv(out / "picks.csv", index=False, encoding="utf-8-sig")
    with open(out / "pick_records.jsonl", "w", encoding="utf-8") as fh:
        for h in handoff:
            fh.write(json.dumps(h, ensure_ascii=False) + "\n")
    summ = summarize(df)
    summ.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    meta = {"command": f"bin_v5 eval {split}", "pipelines": list(pipelines), "conditions": conds,
            "n_tasks": len(tasks), "wall_s": time.time() - t0, "created_utc": datetime.now(timezone.utc).isoformat(),
            "configs": {pl: str(PIPE_CFG[pl].relative_to(PROJECT_ROOT)) for pl in pipelines},
            "config_sha256": {pl: file_sha256(PIPE_CFG[pl]) for pl in pipelines},
            "dataset_sha256": file_sha256(CFG_PATH), "source_sha256": file_sha256(Path(__file__)),
            "preds": dict(preds or {}), "label": LABEL}
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summ.to_string(index=False))
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def summarize(df):
    import pandas as pd
    rows = []
    for (pl, c), g in df.groupby(["pipeline", "condition"], sort=False):
        n = len(g)
        na = int(g.accepted.sum())
        nf = int(g.false_accept.sum())
        nc = int(g.correct_pick.sum())
        lo, hi = wilson(nf, na) if na else (float("nan"), float("nan"))
        rows.append({"pipeline": pl, "condition": c, "n_scenes": n, "accepted": na, "accept_rate": na / n,
                     "correct_pick": nc, "correct_pick_rate": nc / n, "false_accept": nf,
                     "false_accept_rate": nf / na if na else float("nan"), "fa_ci_low": lo, "fa_ci_high": hi,
                     "median_vision_ms": float(g.vision_ms.median()), "p95_vision_ms": float(g.vision_ms.quantile(0.95)),
                     "first_codes": json.dumps(g.first_code.value_counts().to_dict())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- YOLO 데이터
def export_yolo(cfg: Mapping[str, Any], out_root: Path, min_px: int = 64, min_visib: float = 0.1) -> dict:
    import cv2
    from .seg_model import mask_to_polygon
    intr, _ = camera()
    _, mesh = housing_model()
    stats = {}
    for split, yname in (("train", "train"), ("val", "val")):
        (out_root / "images" / yname).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / yname).mkdir(parents=True, exist_ok=True)
        n_img = n_inst = 0
        for sc in plan_scenes(cfg, split):
            raw = load_raw(raw_path(cfg, sc))
            for v in range(len(raw["rgb"])):
                stem = f"{split}_{int(sc['scene_id']):06d}_{v:02d}"
                vis = visibility(raw, mesh, intr, view=v)
                lines = []
                for info in vis:
                    if info["px_visib"] < min_px or info["visib_fract"] < min_visib:
                        continue
                    poly = mask_to_polygon(raw["inst"][v] == info["k"])
                    if poly is None:
                        continue
                    lines.append("0 " + " ".join(f"{x:.6f}" for x in np.clip(poly, 0, 1).reshape(-1)))
                cv2.imwrite(str(out_root / "images" / yname / f"{stem}.jpg"), raw["rgb"][v][..., ::-1],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out_root / "labels" / yname / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                       encoding="utf-8")
                n_img += 1
                n_inst += len(lines)
        stats[split] = {"images": n_img, "instances": n_inst}
    (out_root / "data.yaml").write_text(
        "# 합성(BlenderProc) 하우징 bin YOLO-seg 데이터 (bin_v5 export-yolo)\n"
        f"path: {out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: housing\n",
        encoding="utf-8")
    (out_root / "export_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


# ---------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m vision_guide.bin_v5")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export-mesh")
    s = sub.add_parser("render")
    s.add_argument("--split", required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("check")
    s.add_argument("--split", default="dev")
    s.add_argument("--limit", type=int)
    sub.add_parser("export-yolo")
    s = sub.add_parser("eval")
    s.add_argument("--split", default="dev", choices=["dev"])
    s.add_argument("--pipeline", default="B0")
    s.add_argument("--conditions")
    s.add_argument("--preds")
    s.add_argument("--out")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--limit", type=int)
    s = sub.add_parser("eval-final")
    s.add_argument("--pipeline", default="B0,B1")
    s.add_argument("--preds", help="B1 eval 예측 폴더")
    s.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    cfg = load_cfg()
    if a.cmd == "export-mesh":
        print(export_mesh(cfg))
    elif a.cmd == "render":
        run_render(cfg, a.split, limit=a.limit, dry_run=a.dry_run)
    elif a.cmd == "check":
        intr, _ = camera()
        _, mesh = housing_model()
        rows = []
        for sc in plan_scenes(cfg, a.split)[:a.limit]:
            raw = load_raw(raw_path(cfg, sc))
            rows.append({"scene_id": sc["scene_id"], **check_scene(raw, mesh, intr)})
        out = RESULTS / f"check_{a.split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = all(r["agree_frac"] >= 0.98 and (r["min_iou_200px"] is None or r["min_iou_200px"] >= 0.98) for r in rows)
        out.write_text(json.dumps({"check": "V-S1", "all_ok": ok, "rows": rows}, indent=1), encoding="utf-8")
        print(json.dumps({"V-S1": "통과" if ok else "실패", "n": len(rows),
                          "min_agree": min(r["agree_frac"] for r in rows)}, ensure_ascii=False))
    elif a.cmd == "export-yolo":
        print(export_yolo(cfg, _abs("data/v5/yolo")))
    elif a.cmd == "eval":
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out = Path(a.out) if a.out else RESULTS / f"eval-dev_{a.pipeline}_{ts}"
        pls = a.pipeline.split(",")
        run_eval("dev", pls, a.conditions.split(",") if a.conditions else None, _abs(out),
                 preds={p: a.preds for p in pls if p == "B1"}, workers=a.workers, limit=a.limit)
    elif a.cmd == "eval-final":
        eval_final(a.pipeline.split(","), a.preds, workers=a.workers)


def eval_final(pipelines: Sequence[str], preds_b1: str | None, *, workers: int = 8) -> Path:
    """eval 1회: 설정 동결 확인 → 잠금 → 실행. 잠금이 있으면 다시 실행하지 않는다."""
    lock = RESULTS / "eval-final.lock"
    if lock.exists():
        raise PermissionError(f"eval-final은 이미 실행됨 (잠금 {lock})")
    for pl in pipelines:
        if not pipeline_cfg(pl).get("frozen"):
            raise PermissionError(f"{pl} 설정이 동결되지 않음 (frozen: true 필요)")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"status": "started", "pipelines": list(pipelines),
                                "config_sha256": {pl: file_sha256(PIPE_CFG[pl]) for pl in pipelines},
                                "created_utc": datetime.now(timezone.utc).isoformat()}, indent=1), encoding="utf-8")
    out = RESULTS / f"eval-final_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    run_eval("eval", pipelines, None, out, preds={"B1": preds_b1}, workers=workers)
    st = json.loads(lock.read_text(encoding="utf-8"))
    st.update(status="done", out=str(out.relative_to(PROJECT_ROOT)))
    lock.write_text(json.dumps(st, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    main()
