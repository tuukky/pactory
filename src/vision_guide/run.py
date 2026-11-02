"""v3 CLI: 데이터 받기 / 시각화 / 평가 / 캘리브레이션 / Guidance 오차 예산 / 간이 합성 데모.

Usage:
    python -m vision_guide.run download-bop
    python -m vision_guide.run show-bop
    python -m vision_guide.run eval-bop --split dev --seg dbscan
    python -m vision_guide.run calib-intrinsics
    python -m vision_guide.run calib-handeye
    python -m vision_guide.run guidance-budget
    python -m vision_guide.run synth-demo --n 3

출력은 results/v3/<cmd>_<UTC 시각>/ 와 metadata.json (패키지 버전, git 해시, 시드, 설정 해시, CPU).
평가 세트(eval)는 이 CLI로 평가하지 않는다 (4주차 임계값 동결 후 최종 1회, 별도 승인).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from . import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "v3"
RESULTS_DIR = PROJECT_ROOT / "results" / "v3"
CSV_ENCODING = "utf-8-sig"
PACKAGES = ("open3d", "opencv-python", "trimesh", "numpy", "scipy", "pandas", "PyYAML", "matplotlib", "imageio")
SYNTH_BANNER = "합성 데이터·합성 CAD 기반 / 실제 설비 성능 검증 아님"
LMO_LABEL = "실측: BOP LM-O test_bop19 (합성 아님)"


# ---------------------------------------------------------------------------- 공통
def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def file_hash(*paths: str | Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor()


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_info() -> dict[str, Any]:
    pk = {}
    for name in PACKAGES:
        try:
            pk[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            pk[name] = None
    porcelain = _git(["status", "--porcelain", "--", "."])
    return {"vision_guide_version": __version__, "python": platform.python_version(), "packages": pk,
            "git_commit": _git(["rev-parse", "HEAD"]), "git_dirty": None if porcelain is None else bool(porcelain),
            "cpu": _cpu_name(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "cv2_threads": _cv2_threads()}


def _cv2_threads() -> int | None:
    """cv2.getNumThreads() (cv2를 못 읽으면 None)."""
    try:
        import cv2
        return int(cv2.getNumThreads())
    except Exception:  # noqa: BLE001
        return None


def make_out_dir(cmd: str, out: str | None) -> Path:
    if out:
        d = Path(out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        d = RESULTS_DIR / f"{cmd}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_metadata(out: Path, cmd: str, args: argparse.Namespace, configs: list[Path], wall_s: float,
                   extra: Mapping[str, Any] | None = None, label: str | None = None) -> None:
    """label이 None이면 합성 배너(SYNTH_BANNER). 실측 데이터 평가는 label을 넘긴다 (v4.0 정리)."""
    meta = {"command": cmd, "argv": sys.argv[1:], "args": {k: v for k, v in vars(args).items() if k != "func"},
            "seed": getattr(args, "seed", None), "config_files": [str(p.relative_to(PROJECT_ROOT)) for p in configs],
            "config_hash": file_hash(*configs) if configs else None, "wall_time_s": round(wall_s, 3),
            "created_utc": datetime.now(timezone.utc).isoformat(), "label": SYNTH_BANNER if label is None else label,
            "env": environment_info()}
    if extra:
        meta.update(extra)
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o: Any):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else v
    return out


def pipeline_cfg(profile: str | None = None) -> dict:
    """pipeline.yaml (+ profiles.<profile> 덮어쓰기)."""
    cfg = load_yaml(CONFIG_DIR / "pipeline.yaml")
    profiles = cfg.pop("profiles", {}) or {}
    if profile:
        cfg = deep_merge(cfg, profiles[profile])
        cfg["profile"] = profile
    return cfg


def _bop_cfg() -> dict:
    return load_yaml(CONFIG_DIR / "dataset.yaml")["bop"]


def _bop_root(cfg: Mapping[str, Any]) -> Path:
    return PROJECT_ROOT / cfg["root"]


# ---------------------------------------------------------------------------- download-bop / show-bop
def cmd_download_bop(args: argparse.Namespace) -> None:
    from .datasets import download_bop
    path = download_bop(args.name, PROJECT_ROOT / "data" / "v3" / "bop", force=args.force)
    print(f"downloaded: {path}")


def _save_points_png(pts: np.ndarray, colors: np.ndarray | None, png: Path, title: str) -> str:
    """Open3D 오프스크린 캡처를 시도하고 안 되면 matplotlib 산점도로 저장한다. 사용한 방식을 반환."""
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        vis = o3d.visualization.Visualizer()
        if not vis.create_window(visible=False, width=800, height=600):
            raise RuntimeError("create_window 실패")
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, -1.0, 0.0])
        vis.poll_events()
        vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        vis.destroy_window()
        if img.size == 0 or float(img.std()) < 1e-6:
            raise RuntimeError("빈 캡처")
        import imageio.v3 as iio
        iio.imwrite(png, (np.clip(img, 0, 1) * 255).astype(np.uint8))
        return "open3d_offscreen"
    except Exception:  # noqa: BLE001 - 오프스크린 불가 환경 대비
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = np.random.default_rng(0).choice(len(pts), min(len(pts), 40000), replace=False)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(pts[sel, 0], -pts[sel, 1], c=colors[sel] if colors is not None else pts[sel, 2], s=0.5)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x_cam [m]")
        ax.set_ylabel("-y_cam [m]")
        fig.savefig(png, dpi=120)
        plt.close(fig)
        return "matplotlib_fallback"


def cmd_show_bop(args: argparse.Namespace) -> None:
    import open3d as o3d

    from .datasets import iter_targets, load_frame, load_targets
    from .geometry import depth_to_points
    t0 = time.perf_counter()
    cfg = _bop_cfg()
    root = _bop_root(cfg)
    if args.scene is not None and args.im is not None:
        sid, iid = args.scene, args.im
        # 평가 세트 프레임은 시각화로도 보지 않는다
        if [sid, iid] in cfg["splits"]["eval"]:
            raise SystemExit("평가 세트 프레임은 볼 수 없다")
        frame = load_frame(root, cfg["split_dir"], sid, iid, load_rgb=True,
                           targets=load_targets(root, cfg["targets_file"]).get((sid, iid), []),
                           forbid_ids={(int(a), int(b)) for a, b in cfg["splits"]["eval"]})
    else:
        frame = next(iter_targets(root, cfg, "dev", load_rgb=True))
    out = make_out_dir("show-bop", args.out)
    pts, idx = depth_to_points(frame.depth_m, frame.intr)
    colors = frame.rgb[idx[:, 0], idx[:, 1], :3] / 255.0 if frame.rgb is not None else None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out / "frame.ply"), pcd)
    how = _save_points_png(pts, colors, out / "frame_points.png",
                           f"LM-O scene {frame.scene_id} im {frame.im_id} (dev)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    if frame.rgb is not None:
        axs[0].imshow(frame.rgb)
    axs[0].set_title("RGB")
    im = axs[1].imshow(np.where(frame.depth_m > 0, frame.depth_m, np.nan), cmap="viridis")
    fig.colorbar(im, ax=axs[1], label="depth [m]")
    axs[1].set_title("Depth")
    lab = np.zeros(frame.depth_m.shape)
    for k, m in enumerate(frame.masks_visib):
        lab[m] = k + 1
    axs[2].imshow(np.where(lab > 0, lab, np.nan), cmap="tab10")
    axs[2].set_title("GT mask_visib")
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "frame_panels.png", dpi=110)
    plt.close(fig)
    info = {"scene_id": frame.scene_id, "im_id": frame.im_id, "n_points": int(len(pts)),
            "depth_range_m": [float(frame.depth_m[frame.depth_m > 0].min()), float(frame.depth_m.max())],
            "gt_objects": [g["obj_id"] for g in frame.gts], "targets": frame.targets, "png_method": how,
            "intrinsics": frame.intr.__dict__}
    _json(out / "frame_info.json", info)
    write_metadata(out, "show-bop", args, [CONFIG_DIR / "dataset.yaml"], time.perf_counter() - t0, label=LMO_LABEL)
    print(json.dumps(info, ensure_ascii=False, default=str))
    print(f"-> {out}")


# ---------------------------------------------------------------------------- eval-bop
def bop_object_model(root: Path, obj_id: int, cfg_bop: Mapping[str, Any], cfg_pipe: Mapping[str, Any], infos):
    """BOP 모델 → (정합용 ObjectModel, 지표용 BopModel)."""
    from .cad import sample_model_points
    from .datasets import load_model
    from .pose import ObjectModel
    bm = load_model(root, obj_id, cfg_bop["models_dir"], cfg_bop["max_sym_disc_step"], info=infos[obj_id])
    pts, nrm = sample_model_points(bm.mesh, int(cfg_pipe["model"]["n_points"]), seed=obj_id)
    return ObjectModel(f"lmo_{obj_id}", pts, nrm, bm.diameter, bm.symmetries), bm


def _mask_frac(cluster, mask: np.ndarray, intr) -> float:
    """클러스터 점을 영상에 투영했을 때 마스크 안에 드는 비율."""
    from .geometry import project
    uv, _ = project(np.asarray(cluster.points), intr)
    u = np.clip(np.floor(uv[:, 0]).astype(int), 0, intr.width - 1)
    v = np.clip(np.floor(uv[:, 1]).astype(int), 0, intr.height - 1)
    return float(mask[v, u].mean()) if len(u) else 0.0


def cmd_eval_bop(args: argparse.Namespace) -> None:
    from .datasets import iter_targets, load_models_info
    from .failures import failure_breakdown, other_failure_share, other_share_ok
    from .geometry import depth_to_points
    from .metrics import add, adi, re_deg, re_sym_deg, summarize, te
    from .pose import TIMING_KEYS, pick_best, run_pipeline
    from .segment import SegmentResult, filter_by_extent, segment_scene

    if args.split != "dev":
        raise SystemExit("이 단계에서는 dev 분할만 평가한다 (평가 세트 누설 금지)")
    t_start = time.perf_counter()
    cfg_b = _bop_cfg()
    cfg_p = pipeline_cfg(cfg_b["name"])
    root = _bop_root(cfg_b)
    infos = load_models_info(root, cfg_b["models_dir"])
    models = {oid: bop_object_model(root, oid, cfg_b, cfg_p, infos) for oid in cfg_b["objects"]}
    out = make_out_dir(f"eval-bop_{args.seg}", args.out)
    rows, trows = [], []
    for n_frame, fr in enumerate(iter_targets(root, cfg_b, "dev", limit=args.limit)):
        seg_all = None
        seg_t: dict[str, float] = {}
        if args.seg == "dbscan":
            t0 = time.perf_counter()
            pts, _ = depth_to_points(fr.depth_m, fr.intr)
            seg_t["backproject"] = (time.perf_counter() - t0) * 1e3
            seg_all = segment_scene(pts, cfg_p, None)
            seg_t.update(seg_all.timings_ms)
        for tg in fr.targets:
            oid = tg["obj_id"]
            om, bm = models[oid]
            gt_ks = [k for k, g in enumerate(fr.gts) if g["obj_id"] == oid]  # LM-O: 객체당 1개
            if not gt_ks:  # 대상이 gts에 없으면 OTHER로 귀속 (평가 계속)
                print(f"[OTHER] scene {fr.scene_id} im {fr.im_id} obj {oid}: 대상이 scene_gt에 없음", file=sys.stderr)
                rows.append({"scene_id": fr.scene_id, "im_id": fr.im_id, "obj_id": oid, "symmetric": bm.symmetric,
                             "visib_fract": np.nan, "diameter_mm": bm.diameter * 1e3, "code": "OTHER",
                             "n_candidates": 0, "fitness": np.nan, "rmse_mm": np.nan, "visible_ratio": np.nan,
                             "total_ms": 0.0, "add_mm": np.nan, "adi_mm": np.nan, "te_mm": np.nan,
                             "re_deg": np.nan, "re_sym_deg": np.nan})
                trows.append({"scene_id": fr.scene_id, "im_id": fr.im_id, "obj_id": oid,
                              **{k: 0.0 for k in TIMING_KEYS}})
                continue
            gt_k = gt_ks[0]
            T_gt = fr.gts[gt_k]["T_cam_obj"]
            seed = int(args.seed) + fr.im_id * 100 + oid
            in_mask: list[float] = []
            try:
                if args.seg == "dbscan":
                    clusters = filter_by_extent(seg_all.clusters, cfg_p, om.diameter)
                    seg = SegmentResult(clusters, seg_all.plane_model, seg_all.n_input, dict(seg_t),
                                        seg_all.n_raw_clusters)
                    results, _ = run_pipeline(fr.depth_m, fr.intr, om, cfg_p, seed=seed, segmentation=seg)
                    k_max = int(cfg_p["cluster"]["max_candidates"])
                    in_mask = [_mask_frac(c, fr.masks_visib[gt_k], fr.intr) for c in clusters[:k_max]]
                else:
                    results, _ = run_pipeline(fr.depth_m, fr.intr, om, cfg_p, seed=seed, seg="gt_mask",
                                              roi_mask=fr.masks_visib[gt_k])
                best = pick_best(results)
                code = best.code.value
            except Exception as exc:  # noqa: BLE001 - 모든 실패를 범주에 귀속 (OTHER)
                results, best, code, in_mask = [], None, "OTHER", []
                print(f"[OTHER] scene {fr.scene_id} im {fr.im_id} obj {oid}: {exc!r}", file=sys.stderr)
            timing = {k: 0.0 for k in TIMING_KEYS}
            for r in results:
                for k, v in r.timings_ms.items():
                    timing[k] += v
            T_e = None if best is None else best.T_cam_obj
            diam = bm.diameter
            row = {"scene_id": fr.scene_id, "im_id": fr.im_id, "obj_id": oid, "symmetric": bm.symmetric,
                   "visib_fract": fr.visib_fract[gt_k], "diameter_mm": diam * 1e3, "code": code,
                   "n_candidates": len(results), "fitness": np.nan if best is None else best.fitness,
                   "rmse_mm": np.nan if best is None else best.rmse * 1e3,
                   "visible_ratio": np.nan if best is None else best.visible_ratio,
                   "total_ms": float(sum(timing.values()))}
            if args.seg == "dbscan":
                # 분할 진단: 후보 클러스터 중 점의 50% 이상이 정답 가시 마스크 안인 것이 있었는가
                row["seg_hit"] = bool(in_mask and max(in_mask) >= 0.5)
                bi = next((i for i, r in enumerate(results) if r is best), None)
                row["best_cluster_in_mask"] = in_mask[bi] if bi is not None and bi < len(in_mask) else np.nan
            if T_e is not None:
                row.update(add_mm=add(T_e, T_gt, bm.points) * 1e3, adi_mm=adi(T_e, T_gt, bm.points) * 1e3,
                           te_mm=te(T_e[:3, 3], T_gt[:3, 3]) * 1e3, re_deg=re_deg(T_e[:3, :3], T_gt[:3, :3]),
                           re_sym_deg=re_sym_deg(T_e[:3, :3], T_gt[:3, :3], bm.symmetries))
            else:
                row.update(add_mm=np.nan, adi_mm=np.nan, te_mm=np.nan, re_deg=np.nan, re_sym_deg=np.nan)
            rows.append(row)
            trows.append({"scene_id": fr.scene_id, "im_id": fr.im_id, "obj_id": oid, **timing})
        print(f"frame {n_frame + 1}: scene {fr.scene_id} im {fr.im_id} targets {len(fr.targets)} "
              f"elapsed {time.perf_counter() - t_start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_target.csv", index=False, encoding=CSV_ENCODING)
    tdf = pd.DataFrame(trows)
    tdf.to_csv(out / "timings.csv", index=False, encoding=CSV_ENCODING)
    fb = failure_breakdown(df["code"])
    fb.to_csv(out / "failures.csv", index=False, encoding=CSV_ENCODING)
    summ = summarize(df, cfg_b["success_frac"])
    summ.to_csv(out / "summary_by_object.csv", index=False, encoding=CSV_ENCODING)
    stage = {k: {"median_ms": float(tdf[k].median()), "p95_ms": float(tdf[k].quantile(0.95))} for k in TIMING_KEYS}
    allrow = summ[summ.obj_id == "ALL"].iloc[0].to_dict()
    summary = {
        "dataset": "LM-O (BOP test_bop19)", "split": "dev", "seg": args.seg,
        "primary_seg": cfg_b["primary_seg"], "is_primary_result": args.seg == cfg_b["primary_seg"],
        "note": ("gt_mask: 정답 가시 마스크로 분할 → 정합 단계 상한. 전체 파이프라인 성능이 아님."
                 if args.seg == "gt_mask" else "dbscan: 평면 제거 + DBSCAN 전체 파이프라인 (주 결과)"),
        "n_frames": int(df[["scene_id", "im_id"]].drop_duplicates().shape[0]), "n_targets": int(len(df)),
        "success_criterion": f"ADD(-S) < {cfg_b['success_frac']} x diameter; 실패 코드(추정 거부)는 오답",
        "overall": allrow, "stage_timing_ms": stage,
        "other_share_ok": other_share_ok(fb), "other_failure_share": other_failure_share(fb),
        "pipeline_frozen": bool(cfg_p.get("frozen", False)),
        "pipeline_profile": cfg_p.get("profile"), "pipeline_resolved": cfg_p,
    }
    _json(out / "summary.json", summary)
    write_metadata(out, "eval-bop", args, [CONFIG_DIR / "dataset.yaml", CONFIG_DIR / "pipeline.yaml"],
                   time.perf_counter() - t_start, label=LMO_LABEL)
    print(summ.to_string(index=False))
    print(fb[fb.n > 0].to_string(index=False))
    print(f"-> {out}")


def cmd_resummarize(args: argparse.Namespace) -> None:
    """eval-bop 결과 폴더의 per_target.csv로 요약만 다시 계산 (재평가 없음). 이전 요약은 *_prev로 보존."""
    from .metrics import summarize
    cfg_b = _bop_cfg()
    for d in args.dirs:
        out = Path(d)
        df = pd.read_csv(out / "per_target.csv", encoding=CSV_ENCODING)
        old = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        for name in ("summary.json", "summary_by_object.csv"):
            prev = out / name.replace(".", "_prev.", 1)
            if not prev.exists():  # 최초 요약만 보존 (반복 실행해도 덮지 않음)
                prev.write_bytes((out / name).read_bytes())
        summ = summarize(df, cfg_b["success_frac"])
        summ.to_csv(out / "summary_by_object.csv", index=False, encoding=CSV_ENCODING)
        old["overall"] = summ[summ.obj_id == "ALL"].iloc[0].to_dict()
        old["resummarized"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "reason": "W2: te·re 통계를 채택(OK) 추정만으로, 대칭 물체 회전은 re_sym_deg (raw_*는 전체 추정)",
                               "prev": "summary_prev.json, summary_by_object_prev.csv"}
        _json(out / "summary.json", old)
        print(summ.to_string(index=False))
        print(f"-> {out}")


# ---------------------------------------------------------------------------- calib-intrinsics
def cmd_calib_intrinsics(args: argparse.Namespace) -> None:
    from .calib import Board, calibrate_intrinsics, intrinsics_error, render_checkerboard, sample_board_poses
    from .geometry import load_camera
    t0 = time.perf_counter()
    intr, _ = load_camera(CONFIG_DIR / "camera.yaml")
    gcfg = load_yaml(CONFIG_DIR / "gripper.yaml")["calibration"]
    board = Board(**gcfg["board"])
    out = make_out_dir("calib-intrinsics", args.out)
    rows = []
    for noise in args.pixel_noise:
        for trial in range(args.trials):
            rng = np.random.default_rng([args.seed, int(noise * 1000), trial])
            poses = sample_board_poses(args.views, board, intr, rng)
            imgs = [render_checkerboard(board, intr, T, supersample=4, pixel_noise=noise, rng=rng) for T in poses]
            res = calibrate_intrinsics(imgs, board, (intr.width, intr.height))
            err = intrinsics_error(res["intr"], intr)
            rows.append({"pixel_noise": noise, "trial": trial, "n_views": args.views, "n_used": res["n_used"],
                         "rms_px": res["rms_px"], **err})
            if trial == 0 and noise == args.pixel_noise[0]:
                import imageio.v3 as iio
                iio.imwrite(out / "sample_view.png", imgs[0])
    df = pd.DataFrame(rows)
    df.to_csv(out / "intrinsics_trials.csv", index=False, encoding=CSV_ENCODING)
    summ = df.groupby("pixel_noise").agg(["median", "max"]).drop(columns=["trial"], level=0)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(out / "summary.csv", index=False, encoding=CSV_ENCODING)
    _json(out / "summary.json", {"truth": intr.__dict__, "board": gcfg["board"],
                                 "by_noise": summ.reset_index().to_dict(orient="records")})
    write_metadata(out, "calib-intrinsics", args, [CONFIG_DIR / "camera.yaml", CONFIG_DIR / "gripper.yaml"],
                   time.perf_counter() - t0)
    print(summ.to_string())
    print(f"-> {out}")


# ---------------------------------------------------------------------------- calib-handeye
def cmd_calib_handeye(args: argparse.Namespace) -> None:
    from .calib import handeye_curve, handeye_end_to_end, handeye_settings
    from .geometry import load_camera
    t0 = time.perf_counter()
    intr, T_base_cam = load_camera(CONFIG_DIR / "camera.yaml")
    gcfg = load_yaml(CONFIG_DIR / "gripper.yaml")
    hs = handeye_settings(gcfg, T_base_cam)
    ccfg = gcfg["calibration"]["handeye"]
    out = make_out_dir("calib-handeye", args.out)
    n_trials = args.trials if args.trials is not None else int(ccfg["n_trials"])
    df = handeye_curve(hs, ccfg["noise_levels"], ccfg["pose_counts"], n_trials, args.seed, ccfg["methods"])
    df.to_csv(out / "handeye_trials.csv", index=False, encoding=CSV_ENCODING)
    curve = (df.groupby(["noise_level", "rot_sigma_deg", "trans_sigma_mm", "n_poses", "method"])
             .agg(med_err_mm=("err_mm", "median"), p95_err_mm=("err_mm", lambda x: x.quantile(0.95)),
                  med_err_deg=("err_deg", "median"), p95_err_deg=("err_deg", lambda x: x.quantile(0.95)),
                  rms_err_deg=("err_deg", lambda x: float(np.sqrt(np.mean(np.square(x))))),
                  rms_err_mm=("err_mm", lambda x: float(np.sqrt(np.mean(np.square(x))))))
             .reset_index())
    curve.to_csv(out / "handeye_curve.csv", index=False, encoding=CSV_ENCODING)
    zero = df[df.noise_level == 0]
    e2e = handeye_end_to_end(hs, intr, int(ccfg["end_to_end"]["n_poses"]), int(args.e2e_trials),
                             args.seed, float(ccfg["end_to_end"]["pixel_noise"]), ccfg["methods"])
    e2e.to_csv(out / "handeye_end_to_end.csv", index=False, encoding=CSV_ENCODING)
    _plot_handeye(curve, out / "handeye_curve.png")
    summary = {"noise0_max_err_mm": float(zero.err_mm.max()), "noise0_max_err_deg": float(zero.err_deg.max()),
               "noise0_by_method": zero.groupby("method")[["err_mm", "err_deg"]].max().to_dict(orient="index"),
               "noise_levels": ccfg["noise_levels"], "pose_counts": ccfg["pose_counts"], "n_trials": n_trials,
               "end_to_end": e2e.groupby("method")[["err_mm", "err_deg"]].median().to_dict(orient="index"),
               "end_to_end_trials": int(args.e2e_trials),
               "noise_model": "T_cam_board 측정에 축별 σ 가우시안 (오른쪽 섭동). 로봇 자세는 정확하다고 가정."}
    _json(out / "summary.json", summary)
    write_metadata(out, "calib-handeye", args, [CONFIG_DIR / "camera.yaml", CONFIG_DIR / "gripper.yaml"],
                   time.perf_counter() - t0)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=_json_default))
    print(f"-> {out}")


def _plot_handeye(curve: pd.DataFrame, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    levels = sorted(curve.noise_level.unique())
    fig, axs = plt.subplots(2, len(levels), figsize=(4 * len(levels), 7), squeeze=False)
    for j, lv in enumerate(levels):
        c = curve[curve.noise_level == lv]
        for m, g in c.groupby("method"):
            axs[0, j].plot(g.n_poses, g.med_err_deg, marker="o", label=m)
            axs[1, j].plot(g.n_poses, g.med_err_mm, marker="o", label=m)
        r = c.iloc[0]
        axs[0, j].set_title(f"noise σ={r.rot_sigma_deg}°/{r.trans_sigma_mm}mm")
        axs[0, j].set_ylabel("median rot err [°]")
        axs[1, j].set_ylabel("median trans err [mm]")
        axs[1, j].set_xlabel("robot poses")
        for a in axs[:, j]:
            a.set_yscale("symlog", linthresh=1e-6)
            a.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Hand-eye (eye-to-hand) error vs robot poses - SYNTHETIC virtual data")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------- guidance-budget
def cmd_guidance_budget(args: argparse.Namespace) -> None:
    from .calib import handeye_reference_sigma, handeye_settings
    from .geometry import load_camera
    from .guidance import budget_scene, load_gripper, max_handeye_rot_for_tolerance, mc_error_budget
    t0 = time.perf_counter()
    intr, T_base_cam = load_camera(CONFIG_DIR / "camera.yaml")
    gcfg = load_yaml(CONFIG_DIR / "gripper.yaml")
    gripper, ws = load_gripper(CONFIG_DIR / "gripper.yaml")
    eb = gcfg["error_budget"]
    out = make_out_dir("guidance-budget", args.out)
    hs = handeye_settings(gcfg, T_base_cam)
    he_rot, he_trans, he_info = handeye_reference_sigma(hs, gcfg["calibration"]["handeye"], eb["handeye_reference"],
                                                        int(args.he_trials), args.seed)
    if he_info.get("n_failed"):
        print(f"[경고] {he_info['warning']}", file=sys.stderr)
    if args.pose_sigma_from:
        ps = json.loads((Path(args.pose_sigma_from) / "summary.json").read_text(encoding="utf-8"))
        if not isinstance(ps.get("pose_sigma_empirical"), Mapping):
            raise SystemExit(f"{args.pose_sigma_from}/summary.json에 pose_sigma_empirical이 없다 "
                             "(synth-demo에서 채택된 추정이 없었음) — --pose-sigma-from 없이 실행하거나 synth-demo를 다시 돌린다")
        pose_rot, pose_trans = ps["pose_sigma_empirical"]["rot_deg"], ps["pose_sigma_empirical"]["trans_m"]
        pose_src = f"synth-demo 경험값 ({args.pose_sigma_from})"
    else:
        pose_rot, pose_trans = float(eb["pose_sigma"]["rot_deg"]), float(eb["pose_sigma"]["trans_m"])
        pose_src = "gripper.yaml error_budget.pose_sigma (합성 가정)"
    T_cam_obj, grasp_idx = budget_scene(gcfg, T_base_cam)
    n = int(args.samples if args.samples else eb["n_samples"])
    df = mc_error_budget(T_base_cam, T_cam_obj, gripper.grasps[grasp_idx], gripper, (pose_rot, pose_trans),
                         (he_rot, he_trans), n, args.seed)
    df.to_csv(out / "budget_samples.csv", index=False, encoding=CSV_ENCODING)
    bsum = (df.groupby("source").agg(success=("success", "mean"), med_pos_mm=("pos_err_mm", "median"),
                                     p95_pos_mm=("pos_err_mm", lambda x: x.quantile(0.95)),
                                     med_axis_deg=("axis_err_deg", "median"),
                                     p95_axis_deg=("axis_err_deg", lambda x: x.quantile(0.95)),
                                     msq_pos_mm2=("pos_err_mm", lambda x: float(np.mean(np.square(x)))))
            .reset_index())
    both = float(bsum.loc[bsum.source == "both", "msq_pos_mm2"].iloc[0])
    bsum["share_of_both_msq"] = bsum["msq_pos_mm2"] / both
    bsum.to_csv(out / "budget_summary.csv", index=False, encoding=CSV_ENCODING)
    grid = [float(x) for x in eb["handeye_rot_grid_deg"]]
    max_sig, gdf = max_handeye_rot_for_tolerance(T_base_cam, T_cam_obj, gripper.grasps[grasp_idx], gripper,
                                                 (pose_rot, pose_trans), he_trans, grid, n, args.seed,
                                                 float(eb["success_target"]))
    gdf.to_csv(out / "handeye_rot_grid.csv", index=False, encoding=CSV_ENCODING)
    # 보조 분석 (사후 추가, 주 결과 아님): Hand-eye 병진 오차 0으로 두고 회전만의 한계를 본다.
    max_sig_rot, gdf_rot = max_handeye_rot_for_tolerance(T_base_cam, T_cam_obj, gripper.grasps[grasp_idx], gripper,
                                                         (pose_rot, pose_trans), 0.0, grid, n, args.seed,
                                                         float(eb["success_target"]))
    gdf_rot.to_csv(out / "handeye_rot_grid_rot_only.csv", index=False, encoding=CSV_ENCODING)
    tol = gripper.tol_pos_m * 1e3
    if max_sig_rot is None:
        sentence_rot = "보조(Hand-eye 병진 0): 자세 오차만으로도 허용치를 만족하지 못한다."
    else:
        r2 = gdf_rot[gdf_rot.he_rot_sigma_deg == max_sig_rot].iloc[0]
        sentence_rot = (f"보조(Hand-eye 병진 오차 0 가정): TCP 위치 허용치 ±{tol:.0f} mm를 {eb['success_target']:.0%} 이상 "
                        f"지키려면 Hand-eye 회전 오차가 축별 σ {max_sig_rot:.3f}° (회전각 중앙값 "
                        f"{r2.he_rot_med_angle_deg:.3f}°) 이하여야 한다.")
    if max_sig is None:
        sentence = (f"자세 오차와 Hand-eye 병진 오차만으로도 TCP 허용치 ±{tol:.0f} mm 성공률 "
                    f"{eb['success_target']:.0%}를 만족하지 못한다 (격자 최소값 {grid[0]}°에서도 미달).")
    else:
        row = gdf[gdf.he_rot_sigma_deg == max_sig].iloc[0]
        sentence = (f"TCP 위치 허용치 ±{tol:.0f} mm(접근축 ±{gripper.tol_axis_deg:.0f}°)를 "
                    f"{eb['success_target']:.0%} 이상 지키려면 Hand-eye 회전 오차가 축별 σ {max_sig:.3f}° "
                    f"(회전각 중앙값 {row.he_rot_med_angle_deg:.3f}°) 이하여야 한다 "
                    f"[합성 가정: 자세 σ {pose_rot}°/{pose_trans * 1e3:.2f} mm, Hand-eye 병진 σ {he_trans * 1e3:.3f} mm].")
    summary = {"sentence": sentence, "max_handeye_rot_sigma_deg": max_sig,
               "supplementary_rot_only": {"sentence": sentence_rot, "max_handeye_rot_sigma_deg": max_sig_rot,
                                          "note": "사후 추가 보조 분석. Hand-eye 병진 σ=0. 주 결과는 sentence."}, "success_target": eb["success_target"],
               "tolerance": {"pos_mm": tol, "axis_deg": gripper.tol_axis_deg},
               "pose_sigma": {"rot_deg": pose_rot, "trans_m": pose_trans, "source": pose_src},
               "handeye_sigma": {"rot_deg": he_rot, "trans_m": he_trans, **he_info},
               "budget": bsum.to_dict(orient="records"), "grasp_idx": grasp_idx, "n_samples": n,
               "camera_object_distance_m": float(np.linalg.norm(T_cam_obj[:3, 3]))}
    _json(out / "summary.json", summary)
    write_metadata(out, "guidance-budget", args, [CONFIG_DIR / "camera.yaml", CONFIG_DIR / "gripper.yaml"],
                   time.perf_counter() - t0)
    print(bsum.to_string(index=False))
    print(sentence)
    print(sentence_rot)
    print(f"-> {out}")


# ---------------------------------------------------------------------------- synth-demo
def cmd_synth_demo(args: argparse.Namespace) -> None:
    from . import cad
    from .geometry import (depth_to_points, inv_T, load_camera, make_T, random_rotation, rotation_angle_deg,
                           transform_points)
    from .guidance import load_gripper, select_grasp, tcp_error_sym
    from .metrics import adi, axis_error_deg, re_sym_deg, te
    from .pose import ObjectModel, compute_fpfh, global_register, run_pipeline
    from .segment import preprocess
    import open3d as o3d

    t_start = time.perf_counter()
    intr, T_base_cam = load_camera(CONFIG_DIR / "camera.yaml")
    intr = intr.pinhole()  # Depth는 정합 영상 (왜곡 없음) 가정
    ccfg = load_yaml(CONFIG_DIR / "camera.yaml")
    cfg_p = pipeline_cfg()
    gripper, ws = load_gripper(CONFIG_DIR / "gripper.yaml")
    spec = cad.HousingSpec()
    pts, nrm = cad.sample_model_points(cad.housing_parts(spec), int(cfg_p["model"]["n_points"]), seed=0)
    model = ObjectModel("housing", pts, nrm, cad.model_diameter(cad.make_housing(spec)),
                        cad.housing_symmetries(spec), np.array([0.0, 0.0, 1.0]))
    T_cam_floor = inv_T(T_base_cam) @ make_T(None, ccfg["bin"]["center_m"])
    out = make_out_dir("synth-demo", args.out)
    rows, render_rows = [], []
    for s in range(args.n):
        seed = args.seed + s
        sc = cad.make_simple_scene(args.objects, intr, T_cam_floor, seed, noise_sigma_m=args.noise_mm * 1e-3,
                                   dropout=args.dropout, floor_size_xy=tuple(ccfg["bin"]["size_xy_m"]))
        # §7 합성 데이터 확인: 노이즈 없는 렌더 depth 역투영 점 ↔ 정답 자세 메시 표면 거리
        clean = cad.make_simple_scene(args.objects, intr, T_cam_floor, seed,
                                      floor_size_xy=tuple(ccfg["bin"]["size_xy_m"]))
        for k, T in enumerate(clean.T_cam_objs):
            p, _ = depth_to_points(clean.depth_m, intr, clean.inst == k + 1)
            scene = o3d.t.geometry.RaycastingScene()
            v = transform_points(T, clean.meshes[k].vertices).astype(np.float32)
            scene.add_triangles(o3d.core.Tensor(v), o3d.core.Tensor(np.asarray(clean.meshes[k].faces, np.uint32)))
            d = scene.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy() if len(p) else np.array([np.nan])
            render_rows.append({"scene": s, "obj": k, "n_px": len(p), "max_dist_mm": float(np.max(d) * 1e3),
                                "median_dist_mm": float(np.median(d) * 1e3)})
        t0 = time.perf_counter()
        results, seg = run_pipeline(sc.depth_m, intr, model, cfg_p, seed=seed)
        wall = (time.perf_counter() - t0) * 1e3
        for i, r in enumerate(results):
            row = {"scene": s, "seed": seed, "cluster": i, "code": r.code.value, "fitness": r.fitness,
                   "rmse_mm": r.rmse * 1e3, "visible_ratio": r.visible_ratio, "total_ms": r.total_ms,
                   "scene_wall_ms": wall}
            if r.T_cam_obj is not None:
                errs = [adi(r.T_cam_obj, T, model.points) for T in sc.T_cam_objs]
                k = int(np.argmin(errs))
                Tg = sc.T_cam_objs[k]
                dec = select_grasp(T_base_cam, r.candidates, gripper, ws)
                row.update(match_obj=k, flipped=sc.flipped[k], adi_mm=errs[k] * 1e3,
                           te_mm=te(r.T_cam_obj[:3, 3], Tg[:3, 3]) * 1e3,
                           re_sym_deg=re_sym_deg(r.T_cam_obj[:3, :3], Tg[:3, :3], model.symmetries),
                           axis_err_deg=axis_error_deg(r.T_cam_obj[:3, :3], Tg[:3, :3], model.sym_axis),
                           correct=errs[k] < 0.1 * model.diameter, grasp_code=dec.code.value,
                           grasp_idx=dec.grasp_idx, tilt_deg=dec.tilt_deg)
                if dec.T_base_tcp is not None:
                    pe, ae = tcp_error_sym(dec.T_base_tcp, T_base_cam, Tg, model.symmetries,
                                           gripper.grasps[dec.grasp_idx], gripper.approach_axis_tcp)
                    row.update(tcp_pos_err_mm=pe * 1e3, tcp_axis_err_deg=ae,
                               grasp_success=bool(pe <= gripper.tol_pos_m and ae <= gripper.tol_axis_deg))
            rows.append(row)
        if s == 0:
            _plot_synth(sc, results, model, intr, out / "scene0_overlay.png")
    df = pd.DataFrame(rows)
    df.to_csv(out / "poses.csv", index=False, encoding=CSV_ENCODING)
    rdf = pd.DataFrame(render_rows)
    rdf.to_csv(out / "render_check.csv", index=False, encoding=CSV_ENCODING)
    # 전역 정합 수렴 범위 비율 (§7 FPFH+RANSAC): 전체 모델 점, 균등 무작위 회전(최대 180°)
    gr = []
    rng = np.random.default_rng(args.seed)
    tgt, tgt_f = model.features(cfg_p)
    from .geometry import random_rotation
    for i in range(args.global_trials):
        o3d.utility.random.seed(args.seed + i)
        T_true = make_T(random_rotation(rng), rng.uniform(-0.1, 0.1, 3) + [0, 0, 0.8])
        p = transform_points(T_true, model.points) + rng.normal(scale=0.0005, size=model.points.shape)
        src = preprocess(p, cfg_p)
        t0 = time.perf_counter()
        g = global_register(src, compute_fpfh(src, cfg_p), tgt, tgt_f, cfg_p)
        dt = (time.perf_counter() - t0) * 1e3
        T_est = np.linalg.inv(np.asarray(g.transformation))
        rs = re_sym_deg(T_est[:3, :3], T_true[:3, :3], model.symmetries)
        tt = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
        gr.append({"trial": i, "init_rot_deg": rotation_angle_deg(T_true[:3, :3]), "re_sym_deg": rs, "te_mm": tt * 1e3, "in_basin": bool(rs < 15.0 and tt < 0.01), "ms": dt})
    gdf = pd.DataFrame(gr)
    gdf.to_csv(out / "global_reg.csv", index=False, encoding=CSV_ENCODING)
    ok = df[(df.code == "OK")] if len(df) else df
    acc = ok[ok.get("correct", pd.Series(dtype=bool)) == True] if len(ok) else ok  # noqa: E712
    # 자세 오차 경험 σ: 채택 & ADD-S 정답 중 대칭축이 뒤집히지 않은 것(축 오차 < 10°).
    # 회전 σ = RMS(대칭축 오차)/√2 (축 기울기는 두 성분), 병진 σ = RMS(te)/√3.
    # 대칭축 주위 회전(yaw)은 윗면·바닥면 파지 TCP 위치·접근축에 영향이 없어 제외한다. 뒤집힘은 별도 비율로 보고.
    pose_sigma = None
    fine = acc[acc.axis_err_deg < 10.0] if len(acc) else acc
    if len(fine):
        pose_sigma = {"rot_deg": float(np.sqrt(np.mean(np.square(fine.axis_err_deg))) / np.sqrt(2)),
                      "trans_m": float(np.sqrt(np.mean(np.square(fine.te_mm))) / np.sqrt(3) * 1e-3),
                      "n": int(len(fine)), "definition": "accepted & ADD-S correct & axis_err<10deg"}
    summary = {
        "label": "합성 간이 장면 (numpy+Open3D 레이캐스팅, BlenderProc 아님) / 합성 CAD",
        "n_scenes": args.n, "objects_per_scene": args.objects, "noise_mm": args.noise_mm, "dropout": args.dropout,
        "n_clusters": int(len(df)), "accepted": int((df.code == "OK").sum()),
        "accepted_correct": int(len(acc)), "false_accept": int(len(ok) - len(acc)),
        "objects_total": args.n * args.objects,
        "objects_with_correct_accept": int(acc.groupby(["scene", "match_obj"]).ngroups) if len(acc) else 0,
        "grasp_success_of_accepted": float(ok["grasp_success"].mean()) if "grasp_success" in ok and len(ok) else None,
        "median_scene_wall_ms": float(df.groupby("scene").scene_wall_ms.first().median()) if len(df) else None,
        "median_cluster_ms": float(df.total_ms.median()) if len(df) else None,
        "pose_sigma_empirical": pose_sigma,
        "accepted_axis_flipped": int((acc.axis_err_deg >= 10.0).sum()) if len(acc) else 0,
        "adds_correct_but_grasp_fail": int((acc["grasp_success"] == False).sum())  # noqa: E712
        if len(acc) and "grasp_success" in acc else 0,
        "median_re_sym_deg_accepted": float(acc.re_sym_deg.median()) if len(acc) else None,
        "render_check_max_dist_mm": float(rdf.max_dist_mm.max()),
        "global_reg_basin_rate": float(gdf.in_basin.mean()), "global_reg_trials": int(len(gdf)),
        "global_reg_median_ms": float(gdf.ms.median()),
        "failure_codes": df.code.value_counts().to_dict() if len(df) else {},
    }
    _json(out / "summary.json", summary)
    write_metadata(out, "synth-demo", args, [CONFIG_DIR / "camera.yaml", CONFIG_DIR / "pipeline.yaml",
                                             CONFIG_DIR / "gripper.yaml"], time.perf_counter() - t_start)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=_json_default))
    print(f"-> {out}")


def _plot_synth(sc, results, model, intr, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .geometry import project, transform_points
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].imshow(np.where(sc.depth_m > 0, sc.depth_m, np.nan), cmap="viridis")
    axs[0].set_title("synthetic depth (simple scene)")
    axs[1].imshow(np.where(sc.depth_m > 0, sc.depth_m, np.nan), cmap="gray")
    for r in results:
        if r.T_cam_obj is None:
            continue
        uv, _ = project(transform_points(r.T_cam_obj, model.points[::5]), intr)
        axs[1].scatter(uv[:, 0], uv[:, 1], s=0.3, c="lime" if r.code.value == "OK" else "red")
    axs[1].set_title("estimated poses (green = accepted, red = rejected)")
    for a in axs:
        a.axis("off")
    fig.suptitle("SYNTHETIC data / synthetic CAD - not a real equipment test")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m vision_guide.run", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download-bop", help="BOP 데이터셋 받기 (data/v3/bop/)")
    s.add_argument("--name", default="lmo")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_download_bop)

    s = sub.add_parser("show-bop", help="dev 프레임 1장 Point Cloud PLY + PNG")
    s.add_argument("--scene", type=int)
    s.add_argument("--im", type=int)
    s.add_argument("--out")
    s.set_defaults(func=cmd_show_bop)

    s = sub.add_parser("eval-bop", help="LM-O dev 평가 (ADD/ADD-S)")
    s.add_argument("--split", default="dev", choices=["dev"])
    s.add_argument("--seg", default="dbscan", choices=["dbscan", "gt_mask"])
    s.add_argument("--limit", type=int, default=None, help="프레임 수 제한 (기본 전체 dev)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_eval_bop)

    s = sub.add_parser("resummarize", help="eval-bop 결과 폴더 요약 재계산 (per_target.csv만 사용)")
    s.add_argument("dirs", nargs="+")
    s.set_defaults(func=cmd_resummarize)

    s = sub.add_parser("calib-intrinsics", help="합성 체커보드 내부 파라미터 캘리브레이션")
    s.add_argument("--views", type=int, default=20)
    s.add_argument("--trials", type=int, default=5)
    s.add_argument("--pixel-noise", type=float, nargs="+", default=[0.0, 2.0, 5.0], help="밝기 노이즈 σ (0~255)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_calib_intrinsics)

    s = sub.add_parser("calib-handeye", help="Hand-eye (eye-to-hand) 오차 곡선")
    s.add_argument("--trials", type=int, default=None, help="조건별 반복 (기본 gripper.yaml)")
    s.add_argument("--e2e-trials", type=int, default=5, help="체커보드 렌더링+solvePnP 종단 반복")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_calib_handeye)

    s = sub.add_parser("guidance-budget", help="TCP 오차 예산 몬테카를로")
    s.add_argument("--samples", type=int, default=None)
    s.add_argument("--he-trials", type=int, default=200, help="Hand-eye 기준 조건 σ 추정 반복")
    s.add_argument("--pose-sigma-from", default=None, help="synth-demo 결과 폴더 (경험 자세 σ 사용)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_guidance_budget)

    s = sub.add_parser("synth-demo", help="간이 합성 장면 파이프라인 + Guidance")
    s.add_argument("--n", type=int, default=3, help="장면 수")
    s.add_argument("--objects", type=int, default=5)
    s.add_argument("--noise-mm", type=float, default=1.0)
    s.add_argument("--dropout", type=float, default=0.05)
    s.add_argument("--global-trials", type=int, default=50)
    s.add_argument("--seed", type=int, default=100)
    s.add_argument("--out")
    s.set_defaults(func=cmd_synth_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
