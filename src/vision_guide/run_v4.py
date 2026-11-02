"""v4 CLI (v4.0 기준선 동결·재현, v4.1 색 순위·신뢰도 강화, v4.2 합성 데이터·YOLO-seg 융합, 동결·eval-final).

Usage:
    python -m vision_guide.run_v4 freeze-baseline
    python -m vision_guide.run_v4 eval-dev --config configs/v4/lmo_baseline.yaml [--seg dbscan|gt_mask] [--limit N]
    python -m vision_guide.run_v4 eval-dev --config configs/v4/lmo_v41.yaml --dump-candidates --compute-all --top-k 20
    python -m vision_guide.run_v4 eval-dev --config configs/v4/lmo_v42.yaml [--preds <seg-predict_dev 폴더>]
    python -m vision_guide.run_v4 color-rank-diag --config configs/v4/lmo_v41.yaml
    python -m vision_guide.run_v4 replay-tune --dump results/v4/eval-dev_v41_<ts> --k-min 5
    python -m vision_guide.run_v4 replay-v42 --dump <eval-dev_v42 덤프 폴더들> --ks 1 3
    python -m vision_guide.run_v4 compare-runs <ref_dir> <new_dir> [--exact]
    python -m vision_guide.run_v4 synth-gen --split pilot|val_synth|train_synth      (합성, BlenderProc)
    python -m vision_guide.run_v4 synth-check --split train_synth
    python -m vision_guide.run_v4 freeze-manifest
    python -m vision_guide.run_v4 eval-final --confirm-eval-once                       (한 번만; 잠금 파일)

스레드: Open3D RANSAC 결과는 스레드 수에 따라 달라지고, 16스레드에서는 같은 시드여도 실행마다 흔들린다
(verification §2.1). 이 모듈은 open3d를 import하기 **전에** OMP_NUM_THREADS를 정한다 (환경변수가 없으면 16).
실제 값은 metadata.json에 기록한다. 처리시간 측정 실행은 다른 무거운 작업(pytest, 설치, 학습, 렌더링, 다른 평가)과
동시에 돌리지 않는다. eval 분할은 eval-final 안에서만, 잠금 파일을 만든 뒤에 읽는다.
"""

from __future__ import annotations

import os
import sys

_O3D_PRELOADED = "open3d" in sys.modules
os.environ.setdefault("OMP_NUM_THREADS", "16")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Mapping  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from .run import (CONFIG_DIR, CSV_ENCODING, LMO_LABEL, PROJECT_ROOT, _bop_cfg, _bop_root, _json,  # noqa: E402
                  _mask_frac, file_hash, load_yaml, pipeline_cfg, write_metadata)

CONFIG_DIR_V4 = PROJECT_ROOT / "configs" / "v4"
RESULTS_DIR_V4 = PROJECT_ROOT / "results" / "v4"
V4_KEYS = ("version", "frozen", "source", "color", "checks", "seg", "runtime")


# ---------------------------------------------------------------------------- 공통
def runtime_info() -> dict:
    import cv2
    return {"omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "cv2_threads": int(cv2.getNumThreads()),
            "open3d_preloaded_before_env": _O3D_PRELOADED}


def _set_cv2_threads(cfg: Mapping[str, Any]) -> None:
    import cv2
    cv2.setNumThreads(int((cfg.get("runtime", {}) or {}).get("cv2_threads", 1)))


def _check_threads(cfg: Mapping[str, Any]) -> None:
    want = (cfg.get("runtime", {}) or {}).get("omp_num_threads")
    have = os.environ.get("OMP_NUM_THREADS")
    if _O3D_PRELOADED:
        print("[경고] open3d가 OMP_NUM_THREADS 설정 전에 import됨 — 스레드 수가 고정되지 않았을 수 있다", file=sys.stderr)
    if want is not None and str(want) != str(have):
        print(f"[경고] OMP_NUM_THREADS={have} (설정 runtime.omp_num_threads={want}). 재현 검증 A 등 의도된 경우만 허용",
              file=sys.stderr)


def load_v4_cfg(path: str | Path) -> dict:
    cfg = load_yaml(path)
    cfg.setdefault("color", {"enabled": False})
    cfg.setdefault("checks", {"enabled": False})
    cfg.setdefault("seg", {"source": "dbscan"})
    return cfg


def make_out_dir(name: str, out: str | None) -> Path:
    if out:
        d = Path(out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        d = RESULTS_DIR_V4 / f"{name}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------- freeze-baseline
def baseline_cfg_dict() -> dict:
    """pipeline.yaml 기본값 + profiles.lmo 덮어쓰기 결과 + v4 플래그(모두 꺼짐)."""
    cfg = pipeline_cfg("lmo")
    cfg["frozen"] = True
    cfg["version"] = "baseline"
    cfg["source"] = {"from": "configs/v3/pipeline.yaml (+ profiles.lmo)",
                     "sha256_16": file_hash(CONFIG_DIR / "pipeline.yaml")}
    cfg["color"] = {"enabled": False}
    cfg["checks"] = {"enabled": False}
    cfg["seg"] = {"source": "dbscan"}
    cfg["runtime"] = {"omp_num_threads": 16, "cv2_threads": 1}
    return cfg


def cmd_freeze_baseline(args: argparse.Namespace) -> None:
    path = Path(args.out) if args.out else CONFIG_DIR_V4 / "lmo_baseline.yaml"
    if path.exists() and not args.force:
        raise SystemExit(f"이미 있다 (동결본은 덮지 않는다): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ("# v4.0 기준선 동결본 (freeze-baseline이 생성, 손으로 고치지 않는다).\n"
            "# = configs/v3/pipeline.yaml 기본값 + profiles.lmo. v4 기능 플래그는 모두 꺼짐.\n"
            "# frozen: true 는 v4 동결 표시 (v3 pipeline.yaml의 frozen 키를 덮는다; 계산에는 쓰이지 않는다).\n")
    path.write_text(head + yaml.safe_dump(baseline_cfg_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"-> {path}")


# ---------------------------------------------------------------------------- eval-dev
def _nan_row_metrics() -> dict:
    return dict(add_mm=np.nan, adi_mm=np.nan, te_mm=np.nan, re_deg=np.nan, re_sym_deg=np.nan)


def _pose_metrics(T_e, T_gt, bm) -> dict:
    from .metrics import add, adi, re_deg, re_sym_deg, te
    if T_e is None:
        return _nan_row_metrics()
    return dict(add_mm=add(T_e, T_gt, bm.points) * 1e3, adi_mm=adi(T_e, T_gt, bm.points) * 1e3,
                te_mm=te(T_e[:3, 3], T_gt[:3, 3]) * 1e3, re_deg=re_deg(T_e[:3, :3], T_gt[:3, :3]),
                re_sym_deg=re_sym_deg(T_e[:3, :3], T_gt[:3, :3], bm.symmetries))


def tune_grids() -> dict:
    """§6 dev 격자 (거칠게). 덤프에서 원값을 미리 계산할 변형."""
    return {"tol_free_m": [0.010, 0.015, 0.020, 0.030],
            "color_variants": [{"h_tol": h, "achrom_s_max_obs": a} for h in (10, 15, 20) for a in (60, 90)]}


def eval_target(method: str, fc, m, cfg: Mapping[str, Any], fr, oid: int, *, seg: str, seed: int, success_frac: float,
                compute_all: bool = False, top_k: int | None = None, grids: Mapping[str, Any] | None = None,
                dump: bool = False) -> tuple[dict, dict, list, dict | None]:
    """대상 하나 평가 → (per_target 행, timings 행, 후보 덤프 행들, 분할 품질 행). eval-dev와 eval-final 공용."""
    from .pipeline_v4 import TIMING_KEYS_V4, run_target
    grids = grids or {"tol_free_m": None, "color_variants": None}
    bm = m.bm
    gt_ks = [k for k, g in enumerate(fr.gts) if g["obj_id"] == oid]
    base = {"scene_id": fr.scene_id, "im_id": fr.im_id, "obj_id": oid, "symmetric": bm.symmetric}
    keys3 = {k: base[k] for k in ("scene_id", "im_id", "obj_id")}
    if not gt_ks:
        print(f"[OTHER] scene {fr.scene_id} im {fr.im_id} obj {oid}: 대상이 scene_gt에 없음", file=sys.stderr)
        row = {**base, "visib_fract": np.nan, "diameter_mm": bm.diameter * 1e3, "code": "OTHER",
               "n_candidates": 0, "fitness": np.nan, "rmse_mm": np.nan, "visible_ratio": np.nan,
               "total_ms": 0.0, **_nan_row_metrics()}
        return row, {**keys3, **{k: 0.0 for k in TIMING_KEYS_V4}}, [], None
    gt_k = gt_ks[0]
    T_gt = fr.gts[gt_k]["T_cam_obj"]
    roi = fr.masks_visib[gt_k] if seg == "gt_mask" else None
    try:
        tr = run_target(method, fc, m, cfg, seed=seed, seg=seg, roi_mask=roi, compute_all=compute_all,
                        top_k=top_k, tol_free_grid=grids["tol_free_m"], color_grid=grids["color_variants"])
        best, code, cands, timing = tr.best, tr.code.value, tr.candidates, tr.timings_ms
        in_mask = [_mask_frac(c, fr.masks_visib[gt_k], fr.intr) for c in tr.tried_regions]
    except Exception as exc:  # noqa: BLE001 - 모든 실패를 범주에 귀속 (OTHER)
        print(f"[OTHER] scene {fr.scene_id} im {fr.im_id} obj {oid}: {exc!r}", file=sys.stderr)
        tr, best, code, cands, in_mask = None, None, "OTHER", [], []
        timing = {k: 0.0 for k in TIMING_KEYS_V4}
    row = {**base, "visib_fract": fr.visib_fract[gt_k], "diameter_mm": bm.diameter * 1e3, "code": code,
           "n_candidates": len(cands), "fitness": np.nan if best is None else best.fitness,
           "rmse_mm": np.nan if best is None else best.rmse * 1e3,
           "visible_ratio": np.nan if best is None else best.visible_ratio,
           "total_ms": float(sum(timing.values()))}
    qrow = None
    if seg in ("dbscan", "yolo"):
        row["seg_hit"] = bool(in_mask and max(in_mask) >= 0.5)
        row["region_hit"] = row["seg_hit"]
        row["n_regions"] = 0 if tr is None else tr.n_regions
        bi = next((i for i, c in enumerate(cands) if c.pose is best), None)
        row["best_cluster_in_mask"] = in_mask[bi] if bi is not None and bi < len(in_mask) else np.nan
    if seg == "yolo":
        q = seg_quality_row(fc.preds, oid, fr.masks_visib[gt_k], float(cfg["seg"]["conf"]))
        row.update(q)
        qrow = {**keys3, "visib_fract": fr.visib_fract[gt_k], **q}
    if method in ("v41", "v42"):
        bc = next((c for c in cands if c.pose is best), None)
        row["color_sim"] = np.nan if bc is None else bc.color_sim
        for k in ("viol", "color_ratio", "n_valid"):
            row[k] = np.nan if bc is None else bc.diag.get(k, np.nan)
    row.update(_pose_metrics(None if best is None else best.T_cam_obj, T_gt, bm))
    trow = {**keys3, **{k: float(timing.get(k, 0.0)) for k in TIMING_KEYS_V4}}
    crows = []
    if dump:
        for ci, c in enumerate(cands):
            pm = _pose_metrics(c.pose.T_cam_obj, T_gt, bm)
            adds = pm["adi_mm"] if bm.symmetric else pm["add_mm"]
            crows.append({**base, "diameter_mm": bm.diameter * 1e3, "idx": c.idx, "rank": c.rank,
                          "color_sim": c.color_sim, "v3_code": c.pose.code.value, "code": c.code.value,
                          "fitness": c.pose.fitness, "rmse_mm": c.pose.rmse * 1e3,
                          "visible_ratio": c.pose.visible_ratio, "score": c.pose.score,
                          "has_pose": c.pose.T_cam_obj is not None, **pm, "add_s_mm": adds,
                          "correct_pose": bool(np.isfinite(adds) and adds < success_frac * bm.diameter * 1e3),
                          "in_mask": in_mask[ci] if seg in ("dbscan", "yolo") and ci < len(in_mask) else np.nan,
                          "cand_ms": c.pose.total_ms, **c.diag})
    return row, trow, crows, qrow


def write_run_outputs(out: Path, rows: list, trows: list, crows: list, qrows: list, success_frac: float,
                      summary_extra: Mapping[str, Any], verbose: bool = True) -> dict:
    """per_target·timings·candidates·seg_quality·failures·summary_by_object·summary_by_visib CSV + summary.json."""
    from .failures import failure_breakdown, other_failure_share, other_share_ok
    from .metrics import summarize
    from .pipeline_v4 import TIMING_KEYS_V4
    from .stats import by_visib, headline
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_target.csv", index=False, encoding=CSV_ENCODING)
    tdf = pd.DataFrame(trows)
    tdf.to_csv(out / "timings.csv", index=False, encoding=CSV_ENCODING)
    if crows:
        pd.DataFrame(crows).to_csv(out / "candidates.csv", index=False, encoding=CSV_ENCODING)
    seg_q = None
    if qrows:
        qdf = pd.DataFrame(qrows)
        qdf.to_csv(out / "seg_quality.csv", index=False, encoding=CSV_ENCODING)
        seg_q = seg_quality_summary(qdf)
    fb = failure_breakdown(df["code"])
    fb.to_csv(out / "failures.csv", index=False, encoding=CSV_ENCODING)
    summ = summarize(df, success_frac)
    summ.to_csv(out / "summary_by_object.csv", index=False, encoding=CSV_ENCODING)
    by_visib(df, frac=success_frac).to_csv(out / "summary_by_visib.csv", index=False, encoding=CSV_ENCODING)
    stage = {k: {"median_ms": float(tdf[k].median()), "p95_ms": float(tdf[k].quantile(0.95))} for k in TIMING_KEYS_V4}
    head = headline(df, success_frac)
    summary = {"dataset": "LM-O (BOP test_bop19)", "label": LMO_LABEL, **summary_extra,
               "headline": head, "overall": summ[summ.obj_id == "ALL"].iloc[0].to_dict(),
               "stage_timing_ms": stage, "other_share_ok": other_share_ok(fb),
               "other_failure_share": other_failure_share(fb), "seg_quality": seg_q,
               "runtime": runtime_info()}
    _json(out / "summary.json", summary)
    if verbose:
        print(summ.to_string(index=False))
        print(fb[fb.n > 0].to_string(index=False))
        print(json.dumps(head, ensure_ascii=False))
    return summary


def cmd_eval_dev(args: argparse.Namespace) -> None:
    from .datasets import iter_targets, load_models_info
    from .pipeline_v4 import build_v4_model, method_of, prepare_frame

    t_start = time.perf_counter()
    if args.o3d_init_seed is not None:
        # Open3D 전역 난수는 프로세스 시작 시 시드가 없다(매번 다름) → v3 첫 프레임 분할(segment_plane)이 실행마다 다르다.
        # 재현 검증 A에서만 v3 실행과 같은 초기 시드를 준다 (기본 None = v3와 같은 동작).
        import open3d as o3d
        o3d.utility.random.seed(int(args.o3d_init_seed))
    cfg_path = Path(args.config).resolve()
    cfg = load_v4_cfg(cfg_path)
    _check_threads(cfg)
    _set_cv2_threads(cfg)
    method = method_of(cfg)
    seg = args.seg or cfg["seg"].get("source", "dbscan")
    if seg not in ("dbscan", "gt_mask", "yolo"):
        raise SystemExit(f"seg must be dbscan/gt_mask/yolo: {seg}")
    if seg == "yolo" and method != "v42":
        raise SystemExit("seg=yolo는 v42 설정에서만")
    preds = None
    if seg == "yolo":
        preds = _abs_path(args.preds or ((cfg["seg"].get("predictions") or {}).get("dev")))
        if preds is None or not (preds / "frames").exists():
            raise SystemExit(f"dev 예측 폴더가 없다: {preds} (seg_model predict --split dev)")
        _check_pred_weights(preds, cfg)
    cfg_b = _bop_cfg()
    root = _bop_root(cfg_b)
    infos = load_models_info(root, cfg_b["models_dir"])
    models = {oid: build_v4_model(root, oid, cfg_b, cfg, infos) for oid in cfg_b["objects"]}
    grids = tune_grids() if args.compute_all else {"tol_free_m": None, "color_variants": None}
    name = (f"eval-dev_{cfg.get('version', method)}_{seg}" if seg not in ("dbscan", "yolo")
            else f"eval-dev_{cfg.get('version', method)}")
    out = make_out_dir(name, args.out)
    load_rgb = method in ("v41", "v42")
    rows, trows, crows, qrows = [], [], [], []
    for n_frame, fr in enumerate(iter_targets(root, cfg_b, "dev", limit=args.limit, load_rgb=load_rgb)):
        fc = prepare_frame(fr, cfg, method=method, seg=seg, seed=int(args.seed), preds_dir=preds)
        for tg in fr.targets:
            oid = tg["obj_id"]
            row, trow, cr, q = eval_target(method, fc, models[oid], cfg, fr, oid, seg=seg,
                                           seed=int(args.seed) + fr.im_id * 100 + oid,
                                           success_frac=cfg_b["success_frac"], compute_all=args.compute_all,
                                           top_k=args.top_k, grids=grids, dump=args.dump_candidates)
            rows.append(row)
            trows.append(trow)
            crows.extend(cr)
            if q is not None:
                qrows.append(q)
        print(f"frame {n_frame + 1}: scene {fr.scene_id} im {fr.im_id} targets {len(fr.targets)} "
              f"elapsed {time.perf_counter() - t_start:.0f}s", flush=True)
    note = {"gt_mask": "gt_mask: 정답 가시 마스크 → 정합 + 이 버전 게이트의 상한 (참고 기준, 가설 아님)",
            "yolo": "YOLO-seg 마스크(합성 학습) → 정합 + v4.1 검사 (v4.2 전체 파이프라인)"}.get(
        seg, "dbscan 분할 전체 파이프라인")
    write_run_outputs(out, rows, trows, crows, qrows, cfg_b["success_frac"], {
        "split": "dev", "seg": seg, "method": method, "version": cfg.get("version"), "config": _rel(cfg_path),
        "config_frozen": bool(cfg.get("frozen")), "is_reference_only": seg == "gt_mask", "note": note,
        "dump": {"candidates": bool(crows), "compute_all": bool(args.compute_all), "top_k": args.top_k},
        "predictions": None if preds is None else _rel(preds), "cfg_resolved": cfg})
    write_metadata(out, "run_v4 eval-dev", args, [CONFIG_DIR / "dataset.yaml", cfg_path], time.perf_counter() - t_start,
                   extra={"runtime": runtime_info(), "method": method, "seg": seg}, label=LMO_LABEL)
    print(f"-> {out}")


# ---------------------------------------------------------------------------- v4.2 보조
def _abs_path(p) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _check_pred_weights(pred_dir: Path, cfg: Mapping[str, Any]) -> None:
    """예측 폴더의 가중치 해시가 설정 seg.weights_sha256과 같은지 (다르면 거부)."""
    want = (cfg.get("seg") or {}).get("weights_sha256")
    meta = pred_dir / "metadata.json"
    if want and meta.exists():
        have = json.loads(meta.read_text(encoding="utf-8")).get("weights_sha256")
        if have != want:
            raise SystemExit(f"예측 가중치 해시 {have} != 설정 {want}")


def seg_quality_row(preds, obj_id: int, gt_mask: np.ndarray, conf: float) -> dict:
    """대상 하나의 분할 품질 (진단): conf 이상 같은 클래스 마스크 수, 정답 가시 마스크와 최고 IoU, 검출(IoU ≥ 0.5)."""
    g = np.asarray(gt_mask, bool)
    sel = [p for p in (preds or []) if int(p["obj_id"]) == int(obj_id) and float(p["score"]) >= conf]
    ious = []
    for p in sel:
        m = np.asarray(p["mask"], bool)
        u = int((m | g).sum())
        ious.append(float((m & g).sum() / u) if u else 0.0)
    best = max(ious) if ious else 0.0
    return {"n_class_masks": len(sel), "seg_best_iou": best, "seg_top1_iou": ious[0] if ious else 0.0,
            "seg_detected": bool(best >= 0.5)}


def seg_quality_summary(q: pd.DataFrame) -> dict:
    out = {"n_targets": int(len(q)), "detection_rate_iou50": float(q["seg_detected"].mean()),
           "any_class_mask_rate": float((q["n_class_masks"] > 0).mean()),
           "mean_best_iou": float(q["seg_best_iou"].mean()), "median_best_iou": float(q["seg_best_iou"].median()),
           "mean_top1_iou": float(q["seg_top1_iou"].mean())}
    out["by_object"] = {int(k): {"n": int(len(g)), "detection_rate_iou50": float(g["seg_detected"].mean()),
                                 "mean_best_iou": float(g["seg_best_iou"].mean())} for k, g in q.groupby("obj_id")}
    return out


# ---------------------------------------------------------------------------- color-rank-diag
def color_grid() -> list[dict]:
    """§6 색 격자 (나열 순서 = 동률 처리 순서)."""
    out = []
    for hb, sb, smin, av, sig, kind in itertools.product((12, 18, 36), (1, 2), (30, 45, 60), (1, 3), (0.0, 1.0),
                                                        ("bhattacharyya", "intersection")):
        out.append({"h_bins": hb, "s_bins": sb, "s_min": smin, "achrom_v_bins": av, "h_smooth_sigma": sig,
                    "similarity": kind})
    return out


DIAG_KS = (3, 5, 8, 20)


def choose_color_setting(df: pd.DataFrame, ks=(3, 5, 8), keep: float = 0.95) -> dict:
    """사전 등록 규칙: recall@k / recall@20 ≥ keep 인 설정이 있는 최소 k → 그 k에서 recall 최고 (동률은 격자 순서)."""
    for k in ks:
        ok = df[df[f"ratio@{k}"] >= keep]
        if len(ok):
            best = ok.sort_values([f"recall@{k}", "grid_idx"], ascending=[False, True]).iloc[0]
            return {"k_min": int(k), "grid_idx": int(best.grid_idx), "row": best.to_dict()}
    best = df.sort_values([f"recall@{ks[-1]}", "grid_idx"], ascending=[False, True]).iloc[0]
    return {"k_min": 20, "grid_idx": int(best.grid_idx), "row": best.to_dict()}


def cmd_color_rank_diag(args: argparse.Namespace) -> None:
    from .color import HistSpec, cluster_hsv, hs_histogram, model_color_hist, rank_by_similarity, rgb_to_hsv_u8, similarity
    from .datasets import iter_targets, load_models_info
    from .pipeline_v4 import load_color_mesh, prepare_frame
    from .run import bop_object_model
    from .segment import filter_by_extent

    t_start = time.perf_counter()
    cfg_path = Path(args.config).resolve()
    cfg = load_v4_cfg(cfg_path)
    _check_threads(cfg)
    _set_cv2_threads(cfg)
    cfg_b = _bop_cfg()
    root = _bop_root(cfg_b)
    infos = load_models_info(root, cfg_b["models_dir"])
    diam = {oid: bop_object_model(root, oid, cfg_b, cfg, infos)[0].diameter for oid in cfg_b["objects"]}
    meshes = {oid: load_color_mesh(root, oid, cfg["color"].get("models_dir", "models")) for oid in cfg_b["objects"]}
    grid = color_grid()
    min_px = int(cfg["color"].get("min_px", 30))
    n_samp = int(cfg["color"].get("model_samples", 20000))
    refs = {}
    for gi, g in enumerate(grid):
        spec = HistSpec.from_cfg(g)
        key = (spec.h_bins, spec.s_bins, spec.s_min, spec.achrom_v_bins, spec.h_smooth_sigma)
        if key not in refs:
            refs[key] = {oid: model_color_hist(meshes[oid], spec, n_samp, seed=oid) for oid in cfg_b["objects"]}
    out = make_out_dir("color-rank-diag", args.out)
    targets = []  # (key, in_mask list, cluster hsv list)
    for n_frame, fr in enumerate(iter_targets(root, cfg_b, "dev", limit=args.limit, load_rgb=True)):
        fc = prepare_frame(fr, cfg, method="v41", seg="dbscan", seed=int(args.seed))
        hsv = rgb_to_hsv_u8(fr.rgb)
        for tg in fr.targets:
            oid = tg["obj_id"]
            gt_ks = [k for k, g in enumerate(fr.gts) if g["obj_id"] == oid]
            if not gt_ks:
                continue
            cl = filter_by_extent(fc.seg_all.clusters, cfg, diam[oid])
            inm = [_mask_frac(c, fr.masks_visib[gt_ks[0]], fr.intr) for c in cl]
            targets.append({"scene_id": fr.scene_id, "im_id": fr.im_id, "obj_id": oid,
                            "visib_fract": fr.visib_fract[gt_ks[0]], "in_mask": inm,
                            "hsv": [cluster_hsv(c, hsv, fr.intr) for c in cl]})
        print(f"frame {n_frame + 1}: elapsed {time.perf_counter() - t_start:.0f}s", flush=True)
    n_t = len(targets)
    z_rec = {k: float(np.mean([any(x >= 0.5 for x in t["in_mask"][:k]) for t in targets])) for k in DIAG_KS}
    any_hit = float(np.mean([any(x >= 0.5 for x in t["in_mask"]) for t in targets]))
    rows, trows = [], []
    for gi, g in enumerate(grid):
        spec = HistSpec.from_cfg(g)
        ref = refs[(spec.h_bins, spec.s_bins, spec.s_min, spec.achrom_v_bins, spec.h_smooth_sigma)]
        hits = {k: [] for k in DIAG_KS}
        for t in targets:
            sims = [similarity(hs_histogram(px, spec), ref[t["obj_id"]], g["similarity"]) if len(px) >= min_px else 0.0
                    for px in t["hsv"]]
            order = [i for i, _ in rank_by_similarity(sims)]
            first = next((r for r, i in enumerate(order) if t["in_mask"][i] >= 0.5), None)
            for k in DIAG_KS:
                hits[k].append(first is not None and first < k)
            if gi == 0 or args.per_target_all:
                trows.append({"grid_idx": gi, "scene_id": t["scene_id"], "im_id": t["im_id"], "obj_id": t["obj_id"],
                              "n_clusters": len(order), "target_rank": np.nan if first is None else first})
        row = {"grid_idx": gi, **g, **{f"recall@{k}": float(np.mean(hits[k])) for k in DIAG_KS}}
        for k in DIAG_KS[:-1]:
            row[f"ratio@{k}"] = row[f"recall@{k}"] / row["recall@20"] if row["recall@20"] > 0 else np.nan
        rows.append(row)
    gdf = pd.DataFrame(rows)
    gdf.to_csv(out / "color_rank_grid.csv", index=False, encoding=CSV_ENCODING)
    pd.DataFrame(trows).to_csv(out / "target_ranks.csv", index=False, encoding=CSV_ENCODING)
    choice = choose_color_setting(gdf)
    summary = {"n_targets": n_t, "grid_size": len(grid), "zorder_recall": {str(k): v for k, v in z_rec.items()},
               "any_target_cluster_rate": any_hit, "rule": "ratio@k = recall@k / recall@20 ≥ 0.95 인 최소 k, 그 k에서 recall 최고",
               "choice": choice, "label": LMO_LABEL}
    _json(out / "summary.json", summary)
    write_metadata(out, "run_v4 color-rank-diag", args, [CONFIG_DIR / "dataset.yaml", cfg_path],
                   time.perf_counter() - t_start, extra={"runtime": runtime_info()}, label=LMO_LABEL)
    print(gdf.sort_values("recall@5", ascending=False).head(10).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(f"-> {out}")


# ---------------------------------------------------------------------------- replay-tune
def gate_grid(k_values) -> list[dict]:
    out = []
    for k, tf, mv, ht, mcm, ach, mvr, mmp in itertools.product(
            k_values, (0.010, 0.015, 0.020, 0.030), (0.05, 0.1, 0.15, 0.2, 0.3), (10, 15, 20), (0.3, 0.4, 0.5, 0.6),
            (60, 90), (0.2, 0.3), (50, 100)):
        out.append({"top_k": k, "tol_free_m": tf, "max_violation": mv, "h_tol": ht, "min_color_match": mcm,
                    "achrom_s_max_obs": ach, "min_vis_relaxed": mvr, "min_match_px": mmp})
    return out


def _codes_vec(c: pd.DataFrame, g: Mapping[str, Any], checks: Mapping[str, Any]) -> np.ndarray:
    """consistency.decide_code의 벡터판 (True = 최종 OK). 테스트가 스칼라판과 같은지 확인한다."""
    v3 = c["v3_code"].to_numpy()
    vis = c["visible_ratio"].to_numpy(float)
    viol = c[f"viol@{g['tol_free_m']:g}"].to_numpy(float)
    key = f"{int(g['h_tol'])}_{int(g['achrom_s_max_obs'])}"
    cr = c[f"color_ratio@{key}"].to_numpy(float)
    cn = c[f"color_n@{key}"].fillna(0).to_numpy(float)
    n_valid = c["n_valid"].fillna(0).to_numpy(float)
    n_match = c["n_match"].fillna(0).to_numpy(float)
    dc, cm, vr = checks["depth"], checks["color_match"], checks["visibility_relax"]
    judged = (n_valid >= int(dc["min_valid_px"])) & np.isfinite(viol)
    with np.errstate(invalid="ignore"):
        relaxed = (bool(vr.get("enabled", True)) & judged & (viol <= g["max_violation"]) & (n_match >= g["min_match_px"])
                   & np.isfinite(vis) & (vis >= g["min_vis_relaxed"]))
        pass_vis = (v3 == "OK") | ((v3 == "LOW_VISIBILITY") & relaxed)
        depth_bad = judged & (viol > g["max_violation"])
        color_bad = (cn >= int(cm["min_px"])) & np.isfinite(cr) & (cr < g["min_color_match"])
    return pass_vis & ~depth_bad & ~color_bad


def replay_select(c: pd.DataFrame, ok: np.ndarray, k: int, n_targets: int) -> tuple[np.ndarray, np.ndarray]:
    """대상별 채택 결과 (accepted, correct) — select_candidate와 같은 규칙. c['t']는 대상 번호."""
    sel = c["rank"].to_numpy() < k
    t = c["t"].to_numpy()[sel]
    okk = ok[sel].astype(int)
    score = c["score"].to_numpy(float)[sel]
    rank = c["rank"].to_numpy()[sel]
    corr = c["correct_pose"].to_numpy(bool)[sel]
    order = np.lexsort((-rank, score, okk, t))     # 대상별 마지막 = OK 우선, 점수 최고, 동률은 앞 순위
    t_o = t[order]
    last = np.ones(len(t_o), bool)
    last[:-1] = t_o[1:] != t_o[:-1]
    idx = order[last]
    acc = np.zeros(n_targets, bool)
    cor = np.zeros(n_targets, bool)
    acc[t[idx]] = okk[idx] == 1
    cor[t[idx]] = (okk[idx] == 1) & corr[idx]
    return acc, cor


def cmd_replay_tune(args: argparse.Namespace) -> None:
    t_start = time.perf_counter()
    d = Path(args.dump)
    c = pd.read_csv(d / "candidates.csv", encoding=CSV_ENCODING)
    pt = pd.read_csv(d / "per_target.csv", encoding=CSV_ENCODING)
    keys = ["scene_id", "im_id", "obj_id"]
    pt = pt.reset_index(drop=True)
    pt["t"] = np.arange(len(pt))
    c = c.merge(pt[keys + ["t"]], on=keys, how="left")
    n_t = len(pt)
    cfg = load_v4_cfg(args.config)
    checks = cfg["checks"]
    ks = [k for k in (3, 5, 8, 20) if k >= int(args.k_min)]
    grid = gate_grid(ks)
    ref = {}
    for k in ks:
        acc, cor = replay_select(c, (c["v3_code"] == "OK").to_numpy(), k, n_t)
        ref[k] = (acc, cor)
    rows = []
    for gi, g in enumerate(grid):
        ok = _codes_vec(c, g, checks)
        acc, cor = replay_select(c, ok, int(g["top_k"]), n_t)
        r_acc, r_cor = ref[int(g["top_k"])]
        rows.append({"grid_idx": gi, **g, "rate_add_s": cor.mean(), "false_accept_rate": (acc & ~cor).mean(),
                     "accepted_rate": acc.mean(), "precision": cor.sum() / acc.sum() if acc.sum() else np.nan,
                     "lost_correct_rate": (r_cor & ~cor).mean(), "gained_correct_rate": (cor & ~r_cor).mean(),
                     "ref_rate_add_s": r_cor.mean(), "ref_false_accept_rate": (r_acc & ~r_cor).mean()})
    gdf = pd.DataFrame(rows)
    gdf["objective"] = gdf["rate_add_s"] - gdf["false_accept_rate"]
    feas = gdf[gdf["lost_correct_rate"] <= float(args.max_lost) + 1e-12]
    best = feas.sort_values(["objective", "top_k", "rate_add_s", "false_accept_rate", "grid_idx"],
                            ascending=[False, True, False, True, True]).iloc[0]
    out = make_out_dir("replay-tune", args.out)
    gdf.to_csv(out / "gate_grid.csv", index=False, encoding=CSV_ENCODING)
    ref_rows = [{"top_k": k, "rate_add_s": ref[k][1].mean(), "false_accept_rate": (ref[k][0] & ~ref[k][1]).mean(),
                 "accepted_rate": ref[k][0].mean()} for k in ks]
    summary = {"dump": _rel(d), "n_targets": n_t, "grid_size": len(grid), "k_values": ks,
               "rule": "objective = ADD(-S) − 오채택률 최대, 제약 lost_correct_rate ≤ max_lost (같은 k, v3 게이트만 대비), "
                       "동률: 작은 k → 높은 ADD(-S) → 낮은 오채택률 → 격자 순서",
               "max_lost": float(args.max_lost), "n_feasible": int(len(feas)), "choice": best.to_dict(),
               "reference_v3_gates_only": ref_rows,
               "top10": feas.sort_values(["objective", "top_k", "rate_add_s", "false_accept_rate", "grid_idx"],
                                         ascending=[False, True, False, True, True]).head(10).to_dict(orient="records")}
    _json(out / "summary.json", summary)
    write_metadata(out, "run_v4 replay-tune", args, [Path(args.config).resolve()], time.perf_counter() - t_start,
                   extra={"runtime": runtime_info()}, label=LMO_LABEL)
    print(json.dumps({"choice": summary["choice"], "reference": ref_rows}, ensure_ascii=False, indent=1, default=str))
    print(f"-> {out}")


# ---------------------------------------------------------------------------- compare-runs
def cmd_compare_runs(args: argparse.Namespace) -> None:
    from .stats import compare_runs, exact_compare
    a = pd.read_csv(Path(args.ref) / "per_target.csv", encoding=CSV_ENCODING)
    b = pd.read_csv(Path(args.new) / "per_target.csv", encoding=CSV_ENCODING)
    res = {"ref": _rel(Path(args.ref)), "new": _rel(Path(args.new))}
    if args.exact:
        res["exact"] = exact_compare(a, b)
    res["compare"] = compare_runs(a, b, tol_pp=float(args.tol_pp))
    txt = json.dumps(res, ensure_ascii=False, indent=1, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt, encoding="utf-8")
    print(txt)


# ---------------------------------------------------------------------------- synth-gen / synth-check (합성)
def cmd_synth_gen(args: argparse.Namespace) -> None:
    from .synth import SYNTH_LABEL, generate, load_synth_config
    t_start = time.perf_counter()
    cfg = load_synth_config(args.synth_config)
    out = make_out_dir(f"synth-gen_{args.split}", args.out)
    res = generate(cfg, args.split, args.n_scenes, dry_run=args.dry_run, log_dir=out / "logs")
    res["sec_per_image"] = res["wall_s"] / max(res["n_images"], 1)
    res["label"] = SYNTH_LABEL
    _json(out / "summary.json", res)
    write_metadata(out, "run_v4 synth-gen", args, [Path(args.synth_config).resolve()], time.perf_counter() - t_start,
                   extra={"label_note": SYNTH_LABEL})
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"-> {out}")


def cmd_synth_check(args: argparse.Namespace) -> None:
    from .synth import SYNTH_LABEL, check_scene, load_synth_config, load_target_meshes, split_dir
    t_start = time.perf_counter()
    cfg = load_synth_config(args.synth_config)
    meshes = load_target_meshes(cfg)
    sd = split_dir(cfg, args.split)
    scenes = sorted(int(p.name) for p in sd.iterdir() if (p / "scene_gt_info.json").exists())
    rng = np.random.default_rng(int(args.seed))
    pick = scenes if len(scenes) <= args.n_scenes else sorted(rng.choice(scenes, args.n_scenes, replace=False).tolist())
    out = make_out_dir(f"synth-check_{args.split}", args.out)
    res = [check_scene(sd, s, meshes, n_frames=args.n_frames) for s in pick]
    rows = [{"scene_id": r["scene_id"], **x} for r in res for x in r["rows"]]
    pd.DataFrame(rows).to_csv(out / "mask_iou.csv", index=False, encoding=CSV_ENCODING)
    ious = np.array([x["iou"] for x in rows], float)
    summ = {"split": args.split, "n_scenes_total": len(scenes), "scenes_checked": pick, "n_instances": int(len(ious)),
            "iou_mean": float(np.nanmean(ious)) if len(ious) else None,
            "iou_median": float(np.nanmedian(ious)) if len(ious) else None,
            "iou_p05": float(np.nanpercentile(ious, 5)) if len(ious) else None,
            "frac_iou_ge_0.95": float(np.mean(ious >= 0.95)) if len(ious) else None,
            "criterion": "정답 자세 레이캐스팅 가시 영역 vs mask_visib IoU 평균 ≥ 0.95 (구현계획 §4 test 07)",
            "label": SYNTH_LABEL}
    summ["pass"] = bool(summ["iou_mean"] is not None and summ["iou_mean"] >= 0.95)
    _json(out / "summary.json", summ)
    write_metadata(out, "run_v4 synth-check", args, [Path(args.synth_config).resolve()], time.perf_counter() - t_start)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print(f"-> {out}")


# ---------------------------------------------------------------------------- replay-v42
def replay_v42_select(c: pd.DataFrame, pt: pd.DataFrame, k: int) -> pd.DataFrame:
    """대상별 v4.2 채택 재생: 순위 < k 후보 중 처음 최종 OK (select_first_pass와 같은 규칙).

    반환: per_target 키 + accepted, correct, region_hit (시도 영역 중 in_mask ≥ 0.5)."""
    keys = ["scene_id", "im_id", "obj_id"]
    cc = c[c["rank"] < k].sort_values(keys + ["rank"])
    ok = cc[cc["code"] == "OK"].groupby(keys, as_index=False).first()
    reg = cc.groupby(keys)["in_mask"].max().reset_index()
    d = pt[keys + ["symmetric", "diameter_mm", "visib_fract"]].merge(ok[keys + ["correct_pose"]], on=keys, how="left")
    d = d.merge(reg, on=keys, how="left")
    d["accepted"] = d["correct_pose"].notna()
    d["correct"] = d["correct_pose"].fillna(False).astype(bool)
    d["region_hit"] = d["in_mask"].fillna(0) >= 0.5
    return d


def cmd_replay_v42(args: argparse.Namespace) -> None:
    rows = []
    for dd in args.dump:
        d = Path(dd)
        c = pd.read_csv(d / "candidates.csv", encoding=CSV_ENCODING)
        pt = pd.read_csv(d / "per_target.csv", encoding=CSV_ENCODING)
        cfg = json.loads((d / "summary.json").read_text(encoding="utf-8"))["cfg_resolved"]
        for k in args.ks:
            r = replay_v42_select(c, pt, int(k))
            n = len(r)
            acc, cor = int(r["accepted"].sum()), int(r["correct"].sum())
            rows.append({"dump": _rel(d), "erode_px": int(cfg["seg"].get("erode_px", 0)), "max_instances_per_class": int(k),
                         "n_targets": n, "rate_add_s": cor / n, "false_accept_rate": (acc - cor) / n,
                         "accepted_rate": acc / n, "precision": cor / acc if acc else np.nan,
                         "region_hit_rate": float(r["region_hit"].mean())})
    gdf = pd.DataFrame(rows)
    gdf["objective"] = gdf["rate_add_s"] - gdf["false_accept_rate"]
    best = gdf.sort_values(["objective", "erode_px", "max_instances_per_class"],
                           ascending=[False, True, True]).iloc[0].to_dict()
    out = make_out_dir("replay-v42", args.out)
    gdf.to_csv(out / "fusion_grid.csv", index=False, encoding=CSV_ENCODING)
    summary = {"rule": "objective = ADD(-S) − 오채택률 최대; 동률: 작은 erode_px → 작은 max_instances_per_class "
                       "(tuning_log §3.1 사전 등록)", "grid": rows, "choice": best, "label": LMO_LABEL}
    _json(out / "summary.json", summary)
    print(gdf.to_string(index=False))
    print(json.dumps({"choice": best}, ensure_ascii=False, indent=1, default=str))
    print(f"-> {out}")


# ---------------------------------------------------------------------------- freeze-manifest / eval-final
# eval-final: 세 버전(baseline, v41, v42)을 같은 eval 50프레임에서 **한 번만** 평가한다 (PLAN_v4 §6.4, 구현계획 §8).
# 실행 전 검사 → 잠금 파일 배타 생성(open(path, "x")) → (잠금 하에서만) eval 예측 → eval 프레임 1회 순회 → 결과.
# 잠금은 지우지 않는다. 중간에 실패해도 잠금이 남아 다시 실행할 수 없다 (재개 없음).
FINAL_VERSIONS = ("baseline", "v41", "v42")
DEFAULT_FINAL_CONFIGS = {v: CONFIG_DIR_V4 / f"lmo_{v}.yaml" for v in FINAL_VERSIONS}
MANIFEST_PATH = CONFIG_DIR_V4 / "frozen_manifest.json"
EVAL_LOCK_PATH = RESULTS_DIR_V4 / "eval-final.lock"
PACTORY_SEG_PYTHON = r"C:\Users\CEO\anaconda3\envs\pactory-seg\python.exe"
# dev 실행 간 잡음 (verification §2.3, 같은 코드·설정·16스레드 R1 vs R2) — eval 차이와 함께 보고한다.
DEV_NOISE_BAND = {"source": "results/v4/repro-B_compare_R1_R2.json (dev 366대상, OMP 16, 같은 설정 2회)",
                  "rate_add_s_pp": 1.37, "accepted_rate_pp": 0.55, "false_accept_rate_pp": 0.82,
                  "target_agreement": 0.803}


class EvalFinalRefused(SystemExit):
    """eval-final 실행 전 검사 실패 (아무것도 읽지·쓰지 않고 종료)."""


def sha256_file(path: str | Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(p: str | Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def manifest_paths(configs: Mapping[str, Path], weights: Path) -> list[Path]:
    """동결 대상: 세 설정, dataset.yaml, pipeline.yaml(기준선 출처), 학습·합성 설정, best.pt, src/vision_guide/*.py, requirements_v4."""
    srcs = sorted((PROJECT_ROOT / "src" / "vision_guide").glob("*.py"))
    extra = [CONFIG_DIR / "dataset.yaml", CONFIG_DIR / "pipeline.yaml", CONFIG_DIR_V4 / "train_synth.yaml",
             CONFIG_DIR_V4 / "train_seg.yaml", PROJECT_ROOT / "requirements_v4.txt"]
    return [*[Path(configs[v]) for v in FINAL_VERSIONS], *extra, Path(weights), *srcs]


def build_manifest(configs: Mapping[str, Path], weights: Path) -> dict:
    files = {_key(p): sha256_file(p) for p in manifest_paths(configs, weights)}
    combined = sha256_text(json.dumps(files, sort_keys=True))
    code = sha256_text(json.dumps({k: v for k, v in files.items() if k.startswith("src/vision_guide/")},
                                  sort_keys=True))
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "configs": {v: _key(configs[v]) for v in FINAL_VERSIONS}, "weights": _key(weights),
            "weights_sha256": files[_key(weights)], "code_sha256": code, "files": files,
            "combined_sha256": combined}


def sha256_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def verify_manifest(manifest: Mapping[str, Any], configs: Mapping[str, Path], weights: Path) -> list[str]:
    """동결 매니페스트와 현재 파일이 같은지. 다른 점 목록 (빈 목록 = 일치)."""
    probs = []
    if dict(manifest.get("configs", {})) != {v: _key(configs[v]) for v in FINAL_VERSIONS}:
        probs.append("매니페스트의 설정 경로가 다르다")
    if manifest.get("weights") != _key(weights):
        probs.append("매니페스트의 가중치 경로가 다르다")
    want = dict(manifest.get("files", {}))
    try:
        cur = {_key(p): sha256_file(p) for p in manifest_paths(configs, weights)}
    except FileNotFoundError as exc:
        return probs + [f"파일 없음: {exc.filename}"]
    for k in sorted(set(want) | set(cur)):
        if k not in cur:
            probs.append(f"파일 없음: {k}")
        elif k not in want:
            probs.append(f"매니페스트에 없는 파일: {k}")
        elif want[k] != cur[k]:
            probs.append(f"해시 다름: {k}")
    return probs


def _load_final_cfgs(configs: Mapping[str, Path]) -> dict:
    return {v: load_v4_cfg(configs[v]) for v in FINAL_VERSIONS}


def _v42_weights(cfg42: Mapping[str, Any]) -> Path:
    w = (cfg42.get("seg") or {}).get("weights")
    if not w:
        raise EvalFinalRefused("v42 설정에 seg.weights가 없다")
    return _abs_path(w)


def cmd_freeze_manifest(args: argparse.Namespace) -> None:
    configs = {v: Path(getattr(args, v)).resolve() for v in FINAL_VERSIONS}
    cfgs = _load_final_cfgs(configs)
    bad = [v for v in FINAL_VERSIONS if cfgs[v].get("frozen") is not True]
    if bad:
        raise SystemExit(f"frozen: true가 아닌 설정: {bad}")
    weights = _v42_weights(cfgs["v42"])
    wsha = sha256_file(weights)
    if cfgs["v42"]["seg"].get("weights_sha256") != wsha:
        raise SystemExit("v42 seg.weights_sha256이 가중치 파일과 다르다")
    path = Path(args.out) if args.out else MANIFEST_PATH
    if path.exists() and not args.force:
        raise SystemExit(f"이미 있다: {path} (--force로만 다시 만든다; eval-final 전까지만)")
    man = build_manifest(configs, weights)
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("configs", "weights", "weights_sha256", "code_sha256", "combined_sha256")},
                     ensure_ascii=False, indent=1))
    print(f"-> {path}")


def _o3d_preloaded() -> bool:
    return bool(_O3D_PRELOADED)


def eval_final_preflight(configs: Mapping[str, Path], manifest_path: Path, lock_path: Path, out_dir: Path,
                         confirm: bool) -> dict:
    """실행 전 검사 (모두 통과해야 잠금을 만든다). 실패하면 EvalFinalRefused (이유 전부)."""
    probs: list[str] = []
    if not confirm:
        probs.append("--confirm-eval-once 가 없다")
    if Path(lock_path).exists():
        raise EvalFinalRefused(f"잠금 파일이 이미 있다 (eval-final은 한 번만): {lock_path}")
    cfgs = _load_final_cfgs(configs)
    for v in FINAL_VERSIONS:
        if cfgs[v].get("frozen") is not True:
            probs.append(f"{v}: frozen: true 가 아니다")
        if cfgs[v].get("version") != v:
            probs.append(f"{v}: version 키가 {cfgs[v].get('version')!r}")
    from .pipeline_v4 import method_of
    for v in FINAL_VERSIONS:
        if method_of(cfgs[v]) != v:
            probs.append(f"{v}: 설정이 가리키는 방법이 {method_of(cfgs[v])}")
    if (cfgs["v42"].get("seg") or {}).get("fallback_dbscan"):
        probs.append("v42: fallback_dbscan이 켜져 있다 (주 결과에서는 끈다)")
    weights = None
    try:
        weights = _v42_weights(cfgs["v42"])
    except SystemExit as exc:
        probs.append(str(exc))
    man = None
    if not Path(manifest_path).exists():
        probs.append(f"동결 매니페스트가 없다: {manifest_path}")
    elif weights is not None:
        man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        probs += verify_manifest(man, configs, weights)
        if weights.exists():
            wsha = sha256_file(weights)
            if wsha != cfgs["v42"]["seg"].get("weights_sha256") or wsha != man.get("weights_sha256"):
                probs.append("가중치 해시가 설정·매니페스트와 다르다")
    omp = os.environ.get("OMP_NUM_THREADS")
    for v in FINAL_VERSIONS:
        want = (cfgs[v].get("runtime") or {}).get("omp_num_threads")
        if str(want) != str(omp):
            probs.append(f"{v}: OMP_NUM_THREADS={omp} != runtime.omp_num_threads={want}")
    if _o3d_preloaded():
        probs.append("open3d가 OMP_NUM_THREADS 설정 전에 import됐다")
    if Path(out_dir).exists():
        probs.append(f"출력 폴더가 이미 있다: {out_dir}")
    if probs:
        raise EvalFinalRefused("eval-final 거부:\n  - " + "\n  - ".join(probs))
    return {"cfgs": cfgs, "manifest": man, "weights": weights}


def create_lock(lock_path: Path, content: Mapping[str, Any]) -> None:
    """배타 생성 (이미 있으면 FileExistsError → 거부). eval 프레임을 읽기 전에 부른다."""
    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(lock_path, "x", encoding="utf-8") as fh:
            fh.write(json.dumps(content, ensure_ascii=False, indent=1))
    except FileExistsError:
        raise EvalFinalRefused(f"잠금 파일이 이미 있다 (eval-final은 한 번만): {lock_path}") from None


def default_eval_predict(weights: Path, pred_dir: Path, lock_path: Path, cfg42: Mapping[str, Any]) -> None:
    """pactory-seg subprocess로 eval 예측 (seg_model이 잠금·해시·폴더 미존재를 다시 검사)."""
    import subprocess
    sg = cfg42["seg"]
    exe = os.environ.get("PACTORY_SEG_PYTHON", sg.get("python_exe") or PACTORY_SEG_PYTHON)
    cmd = [exe, "-m", "vision_guide.seg_model", "predict", "--weights", str(weights), "--split", "eval",
           "--lock", str(lock_path), "--out", str(pred_dir), "--conf", str(sg.get("predict_conf_floor", 0.05)),
           "--iou", str(sg.get("iou", 0.7)), "--imgsz", str(sg.get("imgsz", 640))]
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    r = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"eval 예측 실패 (exit {r.returncode}). 잠금은 남는다")


def _eval_final_models(ctx: dict) -> dict:
    from .datasets import load_models_info
    from .pipeline_v4 import build_v4_model
    cfg_b, root = ctx["cfg_b"], ctx["root"]
    infos = load_models_info(root, cfg_b["models_dir"])
    return {v: {oid: build_v4_model(root, oid, cfg_b, ctx["cfgs"][v], infos) for oid in cfg_b["objects"]}
            for v in FINAL_VERSIONS}


# 실행 단위: (이름, 버전 설정, 방법, seg). 참고 기준(gt_mask)은 가설이 아니다.
FINAL_RUNS = (("baseline", "baseline", "baseline", "dbscan"), ("v41", "v41", "v41", "dbscan"),
              ("v42", "v42", "v42", "yolo"), ("gt_mask_baseline", "baseline", "baseline", "gt_mask"),
              ("gt_mask_v41", "v41", "v41", "gt_mask"))


def _eval_final_loop(ctx: dict, models: dict) -> dict:
    """eval 프레임을 **한 번** 순회. 프레임마다 FINAL_RUNS 순서 (같은 프로세스·스레드)."""
    from . import datasets as _ds
    from .pipeline_v4 import prepare_frame
    if not Path(ctx["lock_path"]).exists():
        raise RuntimeError("잠금 없이 eval 프레임을 읽으려 했다")
    import open3d as o3d
    o3d.utility.random.seed(int(ctx["o3d_init_seed"]))
    acc = {name: {"rows": [], "trows": [], "qrows": []} for name, *_ in FINAL_RUNS}
    t0 = time.perf_counter()
    frames = _ds.iter_targets(ctx["root"], ctx["cfg_b"], "eval", allow_eval=True, load_rgb=True)
    for n_frame, fr in enumerate(frames):
        for name, ver, method, seg in FINAL_RUNS:
            cfg = ctx["cfgs"][ver]
            fc = prepare_frame(fr, cfg, method=method, seg=seg, seed=int(ctx["seed"]),
                               preds_dir=ctx["pred_dir"] if seg == "yolo" else None)
            for tg in fr.targets:
                oid = tg["obj_id"]
                row, trow, _, q = eval_target(method, fc, models[ver][oid], cfg, fr, oid, seg=seg,
                                              seed=int(ctx["seed"]) + fr.im_id * 100 + oid,
                                              success_frac=ctx["cfg_b"]["success_frac"])
                acc[name]["rows"].append(row)
                acc[name]["trows"].append(trow)
                if q is not None:
                    acc[name]["qrows"].append(q)
        print(f"eval frame {n_frame + 1}: im {fr.im_id} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    return acc


def hypothesis_verdicts(bs: pd.DataFrame) -> dict:
    """H1~H4 판정 (verification.md 사전 등록 규칙). CI ∋ 0 → "명확한 차이를 확인하지 못함"."""
    def get(a, b, metric, coef=1.0):
        r = bs[(bs.a == a) & (bs.b == b) & (bs.metric == metric) & (np.isclose(bs.coef, coef))]
        return None if r.empty else r.iloc[0].to_dict()

    def comp(r, sign):  # sign +1: b가 커야 함, −1: 작아야 함
        if r is None or not np.isfinite(r["ci_low"]):
            return "판정 불가"
        dir_ok = r["estimate"] * sign > 0
        excl = (r["ci_low"] > 0) or (r["ci_high"] < 0)
        if not excl:
            return "명확한 차이를 확인하지 못함"
        return "지지" if dir_ok else "지지하지 않음 (반대 방향)"

    def combine(parts):
        if all(p == "지지" for p in parts):
            return "지지"
        if any(p.startswith("지지하지 않음") for p in parts):
            return "지지하지 않음"
        return "명확한 차이를 확인하지 못함"

    h1a, h1b = get("baseline", "v41", "rate_add_s"), get("baseline", "v41", "false_accept_rate")
    h2, h2r = get("v41", "v42", "rate_add_s"), get("v41", "v42", "region_hit_rate")
    h3 = get("baseline", "v42", "false_accept_rate", 0.5)
    h4 = get("baseline", "v41", "median_total_ms")
    p1 = [comp(h1a, +1), comp(h1b, -1)]
    if h3 is None:
        v3 = "판정 불가"
    elif h3["estimate"] > 0:
        v3 = "지지하지 않음"
    elif h3["ci_high"] < 0:
        v3 = "지지"
    else:
        v3 = "명확한 차이를 확인하지 못함"
    return {
        "H1": {"statement": "v4.1은 기준선보다 ADD(-S) 성공률이 높고 오채택률이 낮다", "parts": p1,
               "verdict": combine(p1), "evidence": [h1a, h1b]},
        "H2": {"statement": "v4.2는 v4.1보다 ADD(-S) 성공률이 높다 (주된 이유: 대상 영역 확보율 증가)",
               "verdict": comp(h2, +1), "region_hit_rate_diff": h2r, "region_reason_verdict": comp(h2r, +1),
               "evidence": [h2]},
        "H3": {"statement": "v4.2의 오채택률은 기준선의 절반 이하다",
               "rule": "점추정 FA_v42 ≤ 0.5·FA_base 이고 (FA_v42 − 0.5·FA_base) CI 상한 < 0 → 지지",
               "verdict": v3, "evidence": [h3]},
        "H4": {"statement": "v4.1의 처리시간 중앙값은 기준선보다 짧다", "verdict": comp(h4, -1), "evidence": [h4]},
        "dev_noise_band": DEV_NOISE_BAND}


def _eval_final_outputs(ctx: dict, acc: dict) -> dict:
    from .stats import BOOT_METRICS, paired_frame_bootstrap
    out = Path(ctx["out_dir"])
    frac = ctx["cfg_b"]["success_frac"]
    heads = {}
    for name, ver, method, seg in FINAL_RUNS:
        d = out / ("reference" if name.startswith("gt_mask") else "versions") / name
        summ = write_run_outputs(d, acc[name]["rows"], acc[name]["trows"], [], acc[name]["qrows"], frac, {
            "split": "eval", "seg": seg, "method": method, "version": ver, "config": _rel(ctx["configs"][ver]),
            "config_frozen": True, "is_reference_only": seg == "gt_mask",
            "note": "참고 기준, 가설 아님" if seg == "gt_mask" else "eval-final 주 결과"}, verbose=False)
        heads[name] = summ["headline"]
    if acc["v42"]["qrows"]:
        (out / "seg").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(acc["v42"]["qrows"]).to_csv(out / "seg" / "seg_quality.csv", index=False, encoding=CSV_ENCODING)
    dfs = {v: pd.DataFrame(acc[v]["rows"]) for v in FINAL_VERSIONS}
    cdir = out / "comparison"
    cdir.mkdir(parents=True, exist_ok=True)
    bs = paired_frame_bootstrap(dfs, [("baseline", "v41"), ("v41", "v42"), ("baseline", "v42")], BOOT_METRICS,
                                n=10000, seed=20261030, frac=frac)
    h3 = paired_frame_bootstrap(dfs, [("baseline", "v42", 0.5)], ["false_accept_rate"], n=10000, seed=20261030,
                                frac=frac)
    bs = pd.concat([bs, h3], ignore_index=True)
    bs.to_csv(cdir / "paired_bootstrap.csv", index=False, encoding=CSV_ENCODING)
    hyp = hypothesis_verdicts(bs)
    _json(cdir / "hypotheses.json", hyp)
    lines = ["# eval-final 비교 (LM-O eval 50프레임, 실측, 1회 평가)", "",
             "| 실행 | ADD(-S) | 오채택률 | 정밀도 | 채택률 | 처리시간 중앙값 (s) | 대상 영역 확보율 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in heads:
        h = heads[name]
        lines.append(f"| {name} | {100 * h['rate_add_s']:.1f}% | {100 * h['false_accept_rate']:.1f}% | "
                     f"{100 * h['precision']:.1f}% | {100 * h['accepted_rate']:.1f}% | {h['median_total_ms'] / 1e3:.2f} | "
                     f"{100 * h.get('region_hit_rate', float('nan')):.1f}% |")
    lines += ["", "gt_mask_* 는 참고 기준(가설 아님).", "", "## 가설 판정", ""]
    lines += [f"- {k}: **{v['verdict']}** — {v['statement']}" for k, v in hyp.items() if k.startswith("H")]
    lines += ["", f"dev 실행 간 잡음 (같은 설정 16스레드 2회): ADD(-S) {DEV_NOISE_BAND['rate_add_s_pp']}%p, "
              f"오채택률 {DEV_NOISE_BAND['false_accept_rate_pp']}%p, 대상별 일치율 {DEV_NOISE_BAND['target_agreement']:.1%}."]
    (cdir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"headlines": heads, "hypotheses": {k: v["verdict"] for k, v in hyp.items() if k.startswith("H")}}


def cmd_eval_final(args: argparse.Namespace, *, predict_fn=None) -> dict:
    """eval-final (1회). 검사 → 잠금 → eval 예측 → eval 1회 순회 → 결과. 잠금은 지우지 않는다."""
    t_start = time.perf_counter()
    configs = {v: Path(getattr(args, v)).resolve() for v in FINAL_VERSIONS}
    lock_path = Path(args.lock_path).resolve()
    out_dir = Path(args.out).resolve() if args.out else \
        RESULTS_DIR_V4 / f"eval-final_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    pre = eval_final_preflight(configs, Path(args.manifest), lock_path, out_dir, bool(args.confirm_eval_once))
    cfgs, man, weights = pre["cfgs"], pre["manifest"], pre["weights"]
    pred_dir = out_dir / "seg" / "predictions"
    lock = {"status": "started", "created_utc": datetime.now(timezone.utc).isoformat(),
            "configs_sha256": {v: man["files"][_key(configs[v])] for v in FINAL_VERSIONS},
            "weights_sha256": man["weights_sha256"], "code_sha256": man["code_sha256"],
            "dataset_sha256": man["files"][_key(CONFIG_DIR / "dataset.yaml")],
            "manifest_combined_sha256": man["combined_sha256"], "out_dir": str(out_dir), "pred_dir": str(pred_dir),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "o3d_init_seed": int(args.o3d_init_seed),
            "seed": int(args.seed)}
    create_lock(lock_path, lock)                       # ← eval 프레임·예측보다 먼저
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "seg").mkdir(parents=True, exist_ok=True)
    _set_cv2_threads(cfgs["baseline"])
    (predict_fn or default_eval_predict)(weights, pred_dir, lock_path, cfgs["v42"])
    cfg_b = _bop_cfg()
    ctx = {"cfgs": cfgs, "configs": configs, "cfg_b": cfg_b, "root": _bop_root(cfg_b), "lock_path": lock_path,
           "out_dir": out_dir, "pred_dir": pred_dir, "seed": int(args.seed), "o3d_init_seed": int(args.o3d_init_seed)}
    models = _eval_final_models(ctx)
    acc = _eval_final_loop(ctx, models)
    res = _eval_final_outputs(ctx, acc)
    write_metadata(out_dir, "run_v4 eval-final", args, [CONFIG_DIR / "dataset.yaml"], time.perf_counter() - t_start,
                   extra={"runtime": runtime_info(), "lock": lock, "configs": {v: _key(p) for v, p in configs.items()}},
                   label=LMO_LABEL)
    done = {"status": "finished", "finished_utc": datetime.now(timezone.utc).isoformat(), "out_dir": str(out_dir),
            "wall_s": time.perf_counter() - t_start, **res}
    _json(lock_path.with_name(lock_path.stem + ".done.json"), done)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"-> {out_dir}")
    return done


# ---------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m vision_guide.run_v4", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("freeze-baseline", help="configs/v4/lmo_baseline.yaml 생성 (기존 파일은 덮지 않음)")
    s.add_argument("--out")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_freeze_baseline)

    s = sub.add_parser("eval-dev", help="LM-O dev 평가 (dev만)")
    s.add_argument("--config", required=True)
    s.add_argument("--seg", choices=["dbscan", "gt_mask", "yolo"], default=None)
    s.add_argument("--preds", default=None, help="v42 dev 예측 폴더 (기본: 설정 seg.predictions.dev)")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--top-k", type=int, default=None, help="v4.1 상위 k 덮어쓰기 (덤프용 20)")
    s.add_argument("--dump-candidates", action="store_true")
    s.add_argument("--compute-all", action="store_true", help="게이트와 무관하게 모든 검사 원값·격자 변형 계산")
    s.add_argument("--o3d-init-seed", type=int, default=None, help="재현 검증 A 전용: 시작 시 Open3D 전역 난수 시드")
    s.add_argument("--out")
    s.set_defaults(func=cmd_eval_dev)

    s = sub.add_parser("color-rank-diag", help="분할 + 색 순위만: 격자별 대상 클러스터 top-k 포함률")
    s.add_argument("--config", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--per-target-all", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_color_rank_diag)

    s = sub.add_parser("replay-tune", help="후보 덤프 재생으로 게이트 격자 평가")
    s.add_argument("--dump", required=True)
    s.add_argument("--config", default=str(CONFIG_DIR_V4 / "lmo_v41.yaml"))
    s.add_argument("--k-min", type=int, default=3)
    s.add_argument("--max-lost", type=float, default=0.02)
    s.add_argument("--out")
    s.set_defaults(func=cmd_replay_tune)

    s = sub.add_parser("compare-runs", help="두 eval-dev 결과 비교 (재현 검증)")
    s.add_argument("ref")
    s.add_argument("new")
    s.add_argument("--exact", action="store_true")
    s.add_argument("--tol-pp", type=float, default=3.0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_compare_runs)

    synth_cfg = str(CONFIG_DIR_V4 / "train_synth.yaml")
    s = sub.add_parser("synth-gen", help="합성 학습 데이터 생성 (BlenderProc, 합성)")
    s.add_argument("--split", choices=["pilot", "train_synth", "val_synth"], required=True)
    s.add_argument("--n-scenes", type=int, default=None, help="기본: train_synth.yaml scale_decision")
    s.add_argument("--synth-config", default=synth_cfg)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_synth_gen)

    s = sub.add_parser("synth-check", help="합성 장면 마스크 vs 정답 자세 레이캐스팅 IoU, BOP 로더 검사")
    s.add_argument("--split", choices=["pilot", "train_synth", "val_synth"], required=True)
    s.add_argument("--n-scenes", type=int, default=10)
    s.add_argument("--n-frames", type=int, default=5)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--synth-config", default=synth_cfg)
    s.add_argument("--out")
    s.set_defaults(func=cmd_synth_check)

    s = sub.add_parser("replay-v42", help="v4.2 후보 덤프 재생 (max_instances_per_class k: 점수 순 처음 통과)")
    s.add_argument("--dump", required=True, nargs="+")
    s.add_argument("--ks", type=int, nargs="+", default=[1, 3])
    s.add_argument("--out")
    s.set_defaults(func=cmd_replay_v42)

    s = sub.add_parser("freeze-manifest", help="세 설정 동결 확인 + 파일 해시 매니페스트 (configs/v4/frozen_manifest.json)")
    for v in FINAL_VERSIONS:
        s.add_argument(f"--{v}", default=str(DEFAULT_FINAL_CONFIGS[v]))
    s.add_argument("--out")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_freeze_manifest)

    s = sub.add_parser("eval-final", help="세 버전 eval 1회 평가 (동결·매니페스트·잠금 검사). 한 번만 실행된다")
    for v in FINAL_VERSIONS:
        s.add_argument(f"--{v}", default=str(DEFAULT_FINAL_CONFIGS[v]))
    s.add_argument("--manifest", default=str(MANIFEST_PATH))
    s.add_argument("--lock-path", default=str(EVAL_LOCK_PATH))
    s.add_argument("--o3d-init-seed", type=int, default=0, help="시작 시 Open3D 전역 난수 시드 (팀장 결정: 0)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--confirm-eval-once", action="store_true")
    s.add_argument("--out")
    s.set_defaults(func=cmd_eval_final)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
