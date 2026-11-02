"""v4 실행부: 기준선(v3 경로 그대로) / v4.1(색 순위 + 신뢰도 강화) / v4.2(YOLO-seg 마스크 + v4.1 정합·검사).

기준선 경로는 v3 ``eval-bop``과 **같은 함수·같은 호출 순서**로 계산한다 (Open3D 전역 난수 소비까지 같게).
- 프레임마다 depth_to_points → segment_scene(전역 난수 재시드 없음, v3와 같음)
- 대상마다 filter_by_extent → run_pipeline(segmentation=...) → pick_best

v4.1 경로 (설정 플래그로만 켠다):
- 프레임 분할 전에 ``o3d.utility.random.seed(seed + im_id)``로 재시드한다. 상위 k가 달라도 같은 프레임의 분할이
  같아야 덤프 재생(replay-tune) = 실시간 실행이 되기 때문이다 (기준선 경로에는 적용하지 않는다).
- extent 필터 통과 클러스터를 색 유사도 내림차순으로 정렬 → 상위 k만 estimate_pose (시드 = seed + 원래 클러스터 idx)
- v3 신뢰도 코드 + 깊이 일관성·색 일치·가시 완화 (consistency.decide_code) → 채택은 pick_best와 같은 규칙.

v4.2 경로 (``seg.source: yolo``, PLAN_v4 §5.3):
- 프레임 예측 파일(``seg_model`` 산출: 클래스·점수·마스크·GPU 추론 시간)을 읽는다. 대상 클래스 마스크 중 conf 이상을
  점수 순으로 최대 ``max_instances_per_class``개.
- 마스크 → erode_px 침식 → depth_to_points(mask) → preprocess → 클러스터 하나 (gt_mask 경로와 같음, 평면 제거 없음)
  → estimate_pose (시드 = seed + 순위) → v4.1 검사 (값 그대로). 처음 통과한 마스크를 채택한다.
  모두 실패하면 pick_best 규칙(점수 최고)으로 실패 코드를 귀속한다.
- 대상 클래스 마스크가 없으면 ``NO_DETECTION`` (성공률에서는 오답, 오채택 아님).
  dbscan 대체는 ``seg.fallback_dbscan`` 플래그(기본 꺼짐)로만, 주 결과·eval-final에서는 쓰지 않는다.
- 처리시간: 프레임 GPU 추론 시간(seg_infer, 워밍업 제외)을 대상마다 더한다 (v3가 프레임 분할 시간을 더하는 것과 같다).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import open3d as o3d

from .failures import FailureCode
from .pose import TIMING_KEYS, ObjectModel, PoseResult, estimate_pose, pick_best, run_pipeline

TIMING_KEYS_V4 = TIMING_KEYS + ("color", "consistency", "seg_infer")
METHODS = ("baseline", "v41", "v42")


@dataclass
class V4Model:
    om: ObjectModel
    bm: Any                       # datasets.BopModel (지표용, models_eval)
    color_mesh: Any = None        # 정점 색 메시 (models, m)
    ref_hist: np.ndarray | None = None
    raycaster: Any = None


@dataclass
class FrameCache:
    frame: Any
    seg_all: Any = None           # SegmentResult (dbscan)
    seg_t: dict = field(default_factory=dict)
    hsv: np.ndarray | None = None
    preds: list | None = None     # v4.2 예측 (Part B)
    seg_infer_ms: float = 0.0
    color_ms: float = 0.0         # 프레임 HSV 변환 시간 (대상마다 더한다)


@dataclass
class CandidateEval:
    idx: int                      # 원래 클러스터 idx (extent 필터 후 z순)
    source: str                   # dbscan / gt_mask
    rank: int                     # 시도 순서
    color_sim: float
    pose: PoseResult
    diag: dict
    code: FailureCode


@dataclass
class TargetResult:
    best: PoseResult | None
    code: FailureCode
    candidates: list[CandidateEval]
    timings_ms: dict
    n_regions: int                # extent 필터 통과 영역 수
    tried_regions: list = field(default_factory=list)  # 시도한 영역 (dbscan: 클러스터, 대상 영역 확보율 진단용)


def load_color_mesh(root: str | Path, obj_id: int, models_dir: str = "models"):
    """정점 색 메시 (mm → m). LM-O 정점 색은 models에만 있다."""
    import trimesh
    mesh = trimesh.load(Path(root) / models_dir / f"obj_{obj_id:06d}.ply", process=False, force="mesh")
    mesh.apply_scale(1e-3)
    return mesh


def build_v4_model(root: str | Path, obj_id: int, cfg_bop: Mapping[str, Any], cfg: Mapping[str, Any], infos,
                   color_models_dir: str = "models") -> V4Model:
    """정합·지표 모델은 v3 bop_object_model 그대로. 색·검사가 켜졌을 때만 색 메시를 만든다."""
    from .run import bop_object_model
    om, bm = bop_object_model(Path(root), obj_id, cfg_bop, cfg, infos)
    m = V4Model(om, bm)
    col, chk = cfg.get("color", {}) or {}, cfg.get("checks", {}) or {}
    if col.get("enabled") or chk.get("enabled"):
        m.color_mesh = load_color_mesh(root, obj_id, col.get("models_dir", color_models_dir))
    if col.get("enabled"):
        from .color import HistSpec, model_color_hist
        m.ref_hist = model_color_hist(m.color_mesh, HistSpec.from_cfg(col["hist"]), int(col["model_samples"]),
                                      seed=obj_id)
    if chk.get("enabled"):
        from .consistency import ModelRaycaster
        m.raycaster = ModelRaycaster(m.color_mesh)
    return m


def method_of(cfg: Mapping[str, Any]) -> str:
    if (cfg.get("seg", {}) or {}).get("source") == "yolo":
        return "v42"
    return "v41" if (cfg.get("color", {}) or {}).get("enabled") or (cfg.get("checks", {}) or {}).get("enabled") \
        else "baseline"


def prepare_frame(frame, cfg: Mapping[str, Any], *, method: str, seg: str = "dbscan", seed: int = 0,
                  preds_dir: str | Path | None = None) -> FrameCache:
    """프레임 공통 계산 (분할 1회, HSV 1회)."""
    from .geometry import depth_to_points
    from .segment import segment_scene
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}: {method!r}")
    fc = FrameCache(frame)
    if method == "v42" and seg == "yolo":
        from .seg_model import load_frame_predictions
        if preds_dir is None:
            raise ValueError("v42: 예측 폴더(preds_dir)가 필요하다")
        fc.preds, fc.seg_infer_ms = load_frame_predictions(preds_dir, frame.scene_id, frame.im_id)
        if not (cfg.get("seg", {}) or {}).get("fallback_dbscan", False):
            return fc
        seg = "dbscan"   # 대체 경로용 분할 (플래그가 켜졌을 때만)
    if seg == "dbscan":
        if method != "baseline":
            o3d.utility.random.seed(int(seed) + int(frame.im_id))
        t0 = time.perf_counter()
        pts, _ = depth_to_points(frame.depth_m, frame.intr)
        fc.seg_t["backproject"] = (time.perf_counter() - t0) * 1e3
        fc.seg_all = segment_scene(pts, cfg, None)
        fc.seg_t.update(fc.seg_all.timings_ms)
    if method in ("v41", "v42") and (cfg.get("color", {}) or {}).get("enabled") and frame.rgb is not None:
        from .color import rgb_to_hsv_u8
        t0 = time.perf_counter()
        fc.hsv = rgb_to_hsv_u8(frame.rgb)
        fc.color_ms = (time.perf_counter() - t0) * 1e3
    return fc


def select_candidate(cands: Sequence[CandidateEval]) -> tuple[PoseResult | None, FailureCode]:
    """pick_best와 같은 규칙: 최종 OK 중 점수(fitness × 가시) 최고, 없으면 전체 최고 (그 코드로 귀속)."""
    if not cands:
        return None, FailureCode.NO_CLUSTER
    ok = [c for c in cands if c.code == FailureCode.OK]
    pool = ok or list(cands)
    best = max(pool, key=lambda c: c.pose.score)
    return best.pose, best.code


def select_first_pass(cands: Sequence[CandidateEval]) -> tuple[PoseResult | None, FailureCode]:
    """v4.2 규칙: 시도 순서(점수 순)로 처음 최종 OK인 후보. 없으면 select_candidate 규칙으로 귀속."""
    ok = sorted((c for c in cands if c.code == FailureCode.OK), key=lambda c: c.rank)
    if ok:
        return ok[0].pose, ok[0].code
    return select_candidate(cands)


def class_masks(preds: Sequence[Mapping[str, Any]] | None, obj_id: int, conf: float, max_n: int) -> list[dict]:
    """대상 클래스 예측 중 conf 이상, 점수 내림차순 최대 max_n개 (동률은 원래 순서)."""
    sel = [p for p in (preds or []) if int(p["obj_id"]) == int(obj_id) and float(p["score"]) >= float(conf)]
    sel = sorted(enumerate(sel), key=lambda t: (-float(t[1]["score"]), t[0]))
    return [p for _, p in sel[:int(max_n)]]


def erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    if int(erode_px) <= 0:
        return np.asarray(mask, bool)
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(erode_px) + 1, 2 * int(erode_px) + 1))
    return cv2.erode(np.asarray(mask, np.uint8), k).astype(bool)


def _sum_timings(base: Mapping[str, float], poses: Sequence[PoseResult]) -> dict:
    t = {k: 0.0 for k in TIMING_KEYS_V4}
    for k, v in base.items():
        t[k] = t.get(k, 0.0) + float(v)
    for p in poses:
        for k, v in p.timings_ms.items():
            t[k] = t.get(k, 0.0) + float(v)
    return t


def gt_mask_cluster(frame, cfg: Mapping[str, Any], roi_mask: np.ndarray):
    """run_pipeline(seg="gt_mask")와 같은 단일 클러스터 (역투영 → preprocess)와 시간."""
    from .geometry import depth_to_points
    from .segment import preprocess
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    pts, _ = depth_to_points(frame.depth_m, frame.intr, roi_mask)
    t["backproject"] = (time.perf_counter() - t0) * 1e3
    if len(pts) == 0:
        return None, t, 0
    t0 = time.perf_counter()
    pcd = preprocess(pts, cfg)
    t["preprocess"] = (time.perf_counter() - t0) * 1e3
    return pcd, t, len(pts)


def run_target(method: str, fc: FrameCache, m: V4Model, cfg: Mapping[str, Any], *, seed: int,
               seg: str = "dbscan", roi_mask: np.ndarray | None = None, compute_all: bool = False,
               top_k: int | None = None, tol_free_grid: Sequence[float] | None = None,
               color_grid: Sequence[Mapping[str, Any]] | None = None) -> TargetResult:
    """대상 하나. method="baseline"은 v3 eval-bop과 같은 호출, "v41"은 색 순위 + 강화 검사."""
    from .segment import SegmentResult, filter_by_extent
    fr = fc.frame
    if method == "baseline":
        if seg == "dbscan":
            clusters = filter_by_extent(fc.seg_all.clusters, cfg, m.om.diameter)
            sr = SegmentResult(clusters, fc.seg_all.plane_model, fc.seg_all.n_input, dict(fc.seg_t),
                               fc.seg_all.n_raw_clusters)
            results, _ = run_pipeline(fr.depth_m, fr.intr, m.om, cfg, seed=seed, segmentation=sr)
            n_regions = len(clusters)
        else:
            results, _ = run_pipeline(fr.depth_m, fr.intr, m.om, cfg, seed=seed, seg="gt_mask", roi_mask=roi_mask)
            n_regions = 1
        best = pick_best(results)
        cands = [CandidateEval(i, seg, i, float("nan"), r, {}, r.code) for i, r in enumerate(results)]
        tried = clusters[:int(cfg["cluster"]["max_candidates"])] if seg == "dbscan" else []
        return TargetResult(best, best.code, cands, _sum_timings({}, results), n_regions, tried)
    if method == "v42" and seg == "yolo":
        return _run_target_v42(fc, m, cfg, seed=seed, compute_all=compute_all, top_k=top_k)
    if method not in ("v41", "v42"):
        raise ValueError(f"unknown method: {method!r}")
    col, chk = cfg.get("color", {}) or {}, cfg.get("checks", {}) or {}
    extra = {"color": 0.0, "consistency": 0.0}
    if seg == "dbscan":
        o3d.utility.random.seed(int(seed))
        clusters = filter_by_extent(fc.seg_all.clusters, cfg, m.om.diameter)
        base_t = dict(fc.seg_t)
        n_regions = len(clusters)
        if not clusters:
            code = FailureCode.NO_CLUSTER if fc.seg_all.n_input > 0 else FailureCode.NO_POINTS
            p = PoseResult(None, 0.0, float("nan"), float("nan"), code, {})
            return TargetResult(p, code, [], _sum_timings(base_t, []), 0)
        k = int(top_k if top_k is not None else (col.get("top_k") if col.get("enabled")
                                                  else cfg["cluster"]["max_candidates"]))
        if col.get("enabled") and fc.hsv is not None:
            from .color import HistSpec, rank_clusters
            t0 = time.perf_counter()
            ranked = rank_clusters(clusters, fr.rgb, fr.intr, m.ref_hist, HistSpec.from_cfg(col["hist"]),
                                   col["similarity"], int(col["min_px"]), hsv_img=fc.hsv)
            extra["color"] = (time.perf_counter() - t0) * 1e3 + fc.color_ms
        else:
            ranked = [(i, float("nan")) for i in range(len(clusters))]
        todo = [(i, s, clusters[i]) for i, s in ranked[:k]]
    else:
        o3d.utility.random.seed(int(seed))
        pcd, base_t, n_in = gt_mask_cluster(fr, cfg, roi_mask)
        n_regions = 1
        if pcd is None or len(pcd.points) == 0:
            code = FailureCode.NO_POINTS if n_in == 0 else FailureCode.NO_CLUSTER
            p = PoseResult(None, 0.0, float("nan"), float("nan"), code, {})
            return TargetResult(p, code, [], _sum_timings(base_t, []), 0 if pcd is None else 1)
        todo = [(0, float("nan"), pcd)]
    cands: list[CandidateEval] = []
    for rank, (idx, sim, c) in enumerate(todo):
        r = estimate_pose(c, m.om, cfg, depth_m=fr.depth_m, intr=fr.intr, seed=seed + idx)
        code, diag = r.code, {}
        if chk.get("enabled") and m.raycaster is not None:
            from .consistency import apply_v41_checks
            t0 = time.perf_counter()
            code, diag = apply_v41_checks(r, m.raycaster, fr.depth_m, fr.rgb, fr.intr, cfg, compute_all=compute_all,
                                          tol_free_grid=tol_free_grid, color_grid=color_grid)
            extra["consistency"] += (time.perf_counter() - t0) * 1e3
        cands.append(CandidateEval(idx, seg, rank, sim, r, diag, code))
    best, code = select_candidate(cands)
    t = _sum_timings({**base_t, **extra}, [c.pose for c in cands])
    return TargetResult(best, code, cands, t, n_regions, [c for _, _, c in todo] if seg == "dbscan" else [])


def _run_target_v42(fc: FrameCache, m: V4Model, cfg: Mapping[str, Any], *, seed: int, compute_all: bool = False,
                    top_k: int | None = None) -> TargetResult:
    """v4.2: 대상 클래스 마스크(점수 순) → 마스크 영역 점군 → 정합 + v4.1 검사 → 처음 통과 채택."""
    fr = fc.frame
    sg, chk = cfg.get("seg", {}) or {}, cfg.get("checks", {}) or {}
    obj_id = int(getattr(m.bm, "obj_id"))
    max_n = int(top_k if top_k is not None else sg.get("max_instances_per_class", 3))
    masks = class_masks(fc.preds, obj_id, float(sg["conf"]), max_n)
    base_t = {"seg_infer": float(fc.seg_infer_ms)}
    o3d.utility.random.seed(int(seed))
    if not masks:
        if sg.get("fallback_dbscan", False) and fc.seg_all is not None:
            tr = run_target("v41", fc, m, cfg, seed=seed, compute_all=compute_all)
            tr.timings_ms["seg_infer"] = tr.timings_ms.get("seg_infer", 0.0) + float(fc.seg_infer_ms)
            return tr
        p = PoseResult(None, 0.0, float("nan"), float("nan"), FailureCode.NO_DETECTION, {})
        return TargetResult(p, FailureCode.NO_DETECTION, [], _sum_timings(base_t, []), 0)
    cands: list[CandidateEval] = []
    tried = []
    extra = {"consistency": 0.0, "backproject": 0.0, "preprocess": 0.0}
    for rank, pr in enumerate(masks):
        pcd, t, n_in = gt_mask_cluster(fr, cfg, erode_mask(pr["mask"], int(sg.get("erode_px", 0))))
        for k, v in t.items():
            extra[k] = extra.get(k, 0.0) + v
        if pcd is None or len(pcd.points) == 0:
            continue
        r = estimate_pose(pcd, m.om, cfg, depth_m=fr.depth_m, intr=fr.intr, seed=seed + rank)
        code, diag = r.code, {}
        if chk.get("enabled") and m.raycaster is not None:
            from .consistency import apply_v41_checks
            t0 = time.perf_counter()
            code, diag = apply_v41_checks(r, m.raycaster, fr.depth_m, fr.rgb, fr.intr, cfg, compute_all=compute_all)
            extra["consistency"] += (time.perf_counter() - t0) * 1e3
        diag = {**diag, "mask_score": float(pr["score"])}
        cands.append(CandidateEval(rank, "yolo", rank, float("nan"), r, diag, code))
        tried.append(pcd)
        if code == FailureCode.OK and not compute_all:
            break
    if not cands:
        p = PoseResult(None, 0.0, float("nan"), float("nan"), FailureCode.NO_POINTS, {})
        return TargetResult(p, FailureCode.NO_POINTS, [], _sum_timings({**base_t, **extra}, []), len(masks))
    best, code = select_first_pass(cands)
    t = _sum_timings({**base_t, **extra}, [c.pose for c in cands])
    return TargetResult(best, code, cands, t, len(masks), tried)
