"""v4.2-B 인스턴스 세그멘테이션 (YOLO11-seg) 학습·추론 래퍼. 학습 데이터는 합성(BlenderProc)만.

실행 env: ``pactory-seg`` (torch + ultralytics, open3d 없음). 그래서 이 모듈은 vision_guide의 다른 모듈
(open3d를 쓰는 geometry 등)을 import하지 않고, ultralytics·torch는 함수 안에서만 import한다
(n8n env의 테스트가 ultralytics 없이 변환·예측 파일 입출력을 검사할 수 있게).

CLI (pactory-seg):
    python -m vision_guide.seg_model export-yolo
    python -m vision_guide.seg_model train [--out results/v4/seg-train_<ts>]
    python -m vision_guide.seg_model val-synth --weights <best.pt>
    python -m vision_guide.seg_model predict --weights <best.pt> --split dev [--out ...]
    python -m vision_guide.seg_model predict --weights <best.pt> --split eval --lock results/v4/eval-final.lock --out <dir>
        (eval은 eval-final이 잠금을 만든 뒤 subprocess로만 호출한다. 잠금 상태·가중치 해시·출력 폴더 미존재를 검사)

예측 파일: ``<out>/frames/{scene:06d}_{im:06d}.npz`` (obj_ids, scores, masks=np.packbits, shape, infer_ms) +
``predictions.json`` (프레임 목록·시간) + ``metadata.json``.
라이선스: ultralytics 코드·가중치 AGPL-3.0 (공개 시 고지).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLASS_IDS = [1, 5, 6, 8, 9, 10, 11, 12]
CLASS_NAMES = ["ape", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]
SEG_CONFIG = PROJECT_ROOT / "configs" / "v4" / "train_seg.yaml"
SYNTH_CONFIG = PROJECT_ROOT / "configs" / "v4" / "train_synth.yaml"
DATASET_CONFIG = PROJECT_ROOT / "configs" / "v3" / "dataset.yaml"
RESULTS_DIR_V4 = PROJECT_ROOT / "results" / "v4"
LMO_LABEL = "실측: BOP LM-O test_bop19 (합성 아님)"
SYNTH_TRAIN_LABEL = "합성 데이터(BlenderProc)로 학습한 모델"


def _yaml(path: str | Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def weights_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------- 마스크 ↔ 폴리곤
def _merge_segments(segs: list[np.ndarray]) -> np.ndarray:
    """여러 조각을 최근접점 쌍으로 이은 폴리곤 하나 (폭 0 다리로 왕복 → 채우면 합집합)."""
    segs = sorted(segs, key=lambda s: -len(s))
    merged = segs[0]
    for s in segs[1:]:
        d = ((merged[:, None, :] - s[None, :, :]) ** 2).sum(-1)
        a, b = np.unravel_index(int(np.argmin(d)), d.shape)
        ring = np.roll(s, -int(b), axis=0)
        merged = np.concatenate([merged[:a + 1], ring, ring[:1], merged[a:]], axis=0)
    return merged


def mask_to_polygon(mask: np.ndarray, approx_eps_px: float = 1.0, min_area_px: float = 16) -> np.ndarray | None:
    """이진 마스크 → 정규화 폴리곤 (K,2) in [0,1] (x/w, y/h). 조각이 없으면 None.

    findContours(RETR_EXTERNAL) → 면적 ≥ min_area_px 조각 → approxPolyDP(eps) → 여러 조각은 최근접점 쌍 연결.
    """
    import cv2
    m = np.ascontiguousarray(mask.astype(np.uint8))
    h, w = m.shape
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    segs = []
    for c in cs:
        if len(c) < 3 or cv2.contourArea(c) + 0.5 * len(c) < min_area_px:  # 면적 ≈ 내부 + 경계 절반 (픽셀 수 근사)
            continue
        a = cv2.approxPolyDP(c, float(approx_eps_px), True).reshape(-1, 2).astype(float)
        if len(a) >= 3:
            segs.append(a)
    if not segs:
        return None
    poly = _merge_segments(segs) if len(segs) > 1 else segs[0]
    return poly / np.array([w, h], float)


def polygon_to_mask(poly_norm: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2
    m = np.zeros((h, w), np.uint8)
    pts = np.round(np.asarray(poly_norm, float) * np.array([w, h], float)).astype(np.int32)
    cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 1)
    return m.astype(bool)


# ---------------------------------------------------------------------------- 변환
def export_yolo(bop_split_dir: str | Path, out_dir: str | Path, split: str, obj_ids: Sequence[int] = CLASS_IDS,
                min_visib_fract: float = 0.1, min_px: int = 64, approx_eps_px: float = 1.0,
                min_piece_area_px: float = 16) -> dict:
    """합성 BOP 분할 → YOLO-seg (images/{split}/*.jpg 복사, labels/{split}/*.txt). 대상 외 물체는 배경."""
    import cv2
    src = Path(bop_split_dir)
    out = Path(out_dir)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cls_of = {int(o): i for i, o in enumerate(obj_ids)}
    n_img = n_inst = n_skip = 0
    per_cls = {i: 0 for i in range(len(obj_ids))}
    for sdir in sorted(p for p in src.iterdir() if (p / "scene_gt_info.json").exists()):
        gt = json.loads((sdir / "scene_gt.json").read_text(encoding="utf-8"))
        info = json.loads((sdir / "scene_gt_info.json").read_text(encoding="utf-8"))
        for im in sorted(gt, key=int):
            stem = f"{src.name}_{sdir.name}_{int(im):06d}"
            img = next((sdir / "rgb" / f"{int(im):06d}.{e}" for e in ("jpg", "png")
                        if (sdir / "rgb" / f"{int(im):06d}.{e}").exists()), None)
            if img is None:
                continue
            lines = []
            for k, g in enumerate(gt[im]):
                oid = int(g["obj_id"])
                inf = info[im][k]
                if oid not in cls_of or inf["visib_fract"] < min_visib_fract or inf["px_count_visib"] < min_px:
                    n_skip += 1
                    continue
                mask = cv2.imread(str(sdir / "mask_visib" / f"{int(im):06d}_{k:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
                poly = mask_to_polygon(mask, approx_eps_px, min_piece_area_px)
                if poly is None:
                    n_skip += 1
                    continue
                lines.append(f"{cls_of[oid]} " + " ".join(f"{v:.6f}" for v in np.clip(poly, 0, 1).reshape(-1)))
                per_cls[cls_of[oid]] += 1
            shutil.copyfile(img, out / "images" / split / f"{stem}{img.suffix}")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                encoding="utf-8")
            n_img += 1
            n_inst += len(lines)
    return {"split": split, "src": str(src), "n_images": n_img, "n_instances": n_inst, "n_skipped": n_skip,
            "per_class": {CLASS_NAMES[i]: c for i, c in per_cls.items()}}


def write_data_yaml(out_dir: str | Path, train: str = "train", val: str = "val") -> Path:
    out = Path(out_dir)
    lines = [f"path: {out.resolve().as_posix()}", f"train: images/{train}", f"val: images/{val}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    p = out / "data.yaml"
    p.write_text("# 합성(BlenderProc) LM-O 8종 YOLO-seg 데이터 (seg_model export-yolo 생성)\n" + "\n".join(lines) + "\n",
                 encoding="utf-8")
    return p


# ---------------------------------------------------------------------------- 학습·검증 (ultralytics)
def train(data_yaml: str | Path, train_cfg: Mapping[str, Any], out_dir: str | Path) -> Path:
    from ultralytics import YOLO
    out = Path(out_dir).resolve()   # 상대 경로면 ultralytics가 runs/ 아래에 둔다
    model = YOLO(str(_abs(train_cfg["model"])))
    t = dict(train_cfg["train"])
    model.train(data=str(data_yaml), project=str(out.parent), name=out.name, exist_ok=True, **t)
    best = out / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"best.pt가 없다: {best}")
    return best


def conf_from_f1(px: np.ndarray, f1: np.ndarray, step: float = 0.05) -> dict:
    """클래스 평균 F1 곡선 최대 conf → step 단위 반올림."""
    f = np.asarray(f1, float)
    mean = f.mean(axis=0) if f.ndim == 2 else f
    i = int(np.argmax(mean))
    c = float(px[i])
    return {"conf_raw": c, "f1_max": float(mean[i]), "conf": float(np.round(np.round(c / step) * step, 4))}


def val_synth(weights: str | Path, data_yaml: str | Path, imgsz: int = 640, device: int | str = 0) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(weights))
    r = m.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=8, device=device, plots=False, verbose=False)
    seg, box = r.seg, r.box
    px = np.asarray(getattr(r.seg, "px", np.linspace(0, 1, 1000)))
    out = {"mask_map50": float(seg.map50), "mask_map50_95": float(seg.map), "box_map50": float(box.map50),
           "box_map50_95": float(box.map), "mask_map50_95_per_class": dict(zip(CLASS_NAMES, map(float, seg.maps))),
           "fusion_conf": conf_from_f1(px, np.asarray(seg.f1_curve)), "weights_sha256": weights_sha256(weights),
           "label": "합성 검증 세트 (val_synth)"}
    return out


# ---------------------------------------------------------------------------- 예측 파일
def save_frame_predictions(path: str | Path, obj_ids, scores, masks, infer_ms: float) -> None:
    masks = np.asarray(masks, bool)
    shape = masks.shape[1:] if masks.ndim == 3 else (0, 0)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, obj_ids=np.asarray(obj_ids, np.int32), scores=np.asarray(scores, np.float32),
                        masks=np.packbits(masks.reshape(len(masks), -1), axis=1) if len(masks) else np.zeros((0, 0), np.uint8),
                        shape=np.asarray(shape, np.int32), infer_ms=np.float64(infer_ms))


def frame_pred_path(pred_dir: str | Path, scene_id: int, im_id: int) -> Path:
    return Path(pred_dir) / "frames" / f"{int(scene_id):06d}_{int(im_id):06d}.npz"


def load_frame_predictions(pred_dir: str | Path, scene_id: int, im_id: int) -> tuple[list[dict], float]:
    """({obj_id, score, mask} 목록 (점수 내림차순), infer_ms). 파일이 없으면 FileNotFoundError."""
    z = np.load(frame_pred_path(pred_dir, scene_id, im_id), allow_pickle=False)
    h, w = (int(x) for x in z["shape"])
    n = len(z["obj_ids"])
    out = []
    for i in range(n):
        m = np.unpackbits(z["masks"][i], count=h * w).reshape(h, w).astype(bool)
        out.append({"obj_id": int(z["obj_ids"][i]), "score": float(z["scores"][i]), "mask": m})
    out.sort(key=lambda d: -d["score"])
    return out, float(z["infer_ms"])


# ---------------------------------------------------------------------------- 예측 (dev / eval 가드)
def _split_ids(split: str) -> list[tuple[int, int]]:
    cfg = _yaml(DATASET_CONFIG)["bop"]
    return [(int(s), int(i)) for s, i in cfg["splits"][split]]


def check_eval_guard(lock_path: str | Path | None, weights: str | Path, out_dir: str | Path) -> dict:
    """eval 예측 허용 조건: 잠금 파일 존재 + status started + 가중치 해시 일치 + 출력 폴더 미존재."""
    if lock_path is None:
        raise PermissionError("eval 예측은 eval-final 잠금(--lock) 없이 할 수 없다")
    lp = Path(lock_path)
    if not lp.exists():
        raise PermissionError(f"잠금 파일이 없다: {lp}")
    lock = json.loads(lp.read_text(encoding="utf-8"))
    if lock.get("status") != "started":
        raise PermissionError(f"잠금 상태가 started가 아니다: {lock.get('status')!r}")
    if lock.get("weights_sha256") != weights_sha256(weights):
        raise PermissionError("가중치 해시가 잠금 기록과 다르다")
    if Path(out_dir).exists():
        raise PermissionError(f"eval 예측 폴더가 이미 있다 (1회만): {out_dir}")
    return lock


def predict_split(weights: str | Path, split: str, out_dir: str | Path, *, conf: float, iou: float = 0.7,
                  imgsz: int = 640, device: int | str = 0, lock_path: str | Path | None = None,
                  limit: int | None = None) -> Path:
    """LM-O dev(또는 잠금 하의 eval) 프레임 RGB → 예측 파일. 첫 프레임 1회 워밍업 시간은 제외."""
    if split not in ("dev", "eval"):
        raise ValueError(split)
    if split == "eval":                      # 가드가 먼저 (무거운 import·프레임 읽기 전)
        check_eval_guard(lock_path, weights, out_dir)
    import cv2
    import torch
    from ultralytics import YOLO
    ds = _yaml(DATASET_CONFIG)["bop"]
    root = _abs(ds["root"]) / ds["split_dir"]
    ids = _split_ids(split)
    if split == "dev":
        ev = set(_split_ids("eval"))
        if any(i in ev for i in ids):
            raise ValueError("dev 목록에 eval 프레임이 섞여 있다")
    if limit is not None:
        ids = ids[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=split == "dev")
    model = YOLO(str(weights))
    t_start = time.perf_counter()
    rows = []
    warm = False
    for scene_id, im_id in ids:
        sdir = root / f"{scene_id:06d}" / "rgb"
        p = next(sdir / f"{im_id:06d}.{e}" for e in ("png", "jpg") if (sdir / f"{im_id:06d}.{e}").exists())
        img = cv2.imread(str(p))  # BGR (ultralytics 규약)
        if not warm:
            model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, retina_masks=True, verbose=False)
            warm = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(img, conf=conf, iou=iou, imgsz=imgsz, device=device, retina_masks=True, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if r.masks is not None and len(r.boxes):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            sc = r.boxes.conf.cpu().numpy().astype(float)
            mk = r.masks.data.cpu().numpy() > 0.5
        else:
            cls, sc, mk = np.zeros(0, int), np.zeros(0), np.zeros((0, img.shape[0], img.shape[1]), bool)
        oids = [CLASS_IDS[c] for c in cls]
        save_frame_predictions(frame_pred_path(out, scene_id, im_id), oids, sc, mk, ms)
        rows.append({"scene_id": scene_id, "im_id": im_id, "n_pred": len(oids), "infer_ms": ms})
    meta = {"command": "seg_model predict", "split": split, "weights": str(weights),
            "weights_sha256": weights_sha256(weights), "conf_floor": conf, "iou": iou, "imgsz": imgsz,
            "retina_masks": True, "n_frames": len(rows), "wall_s": time.perf_counter() - t_start,
            "median_infer_ms": float(np.median([r["infer_ms"] for r in rows])) if rows else None,
            "created_utc": datetime.now(timezone.utc).isoformat(), "label": LMO_LABEL,
            "model_label": SYNTH_TRAIN_LABEL, "env": _env()}
    (out / "predictions.json").write_text(json.dumps({"frames": rows}, indent=1), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _env() -> dict:
    d: dict = {"python": sys.version.split()[0]}
    try:
        import torch
        d.update(torch=torch.__version__, cuda=torch.version.cuda,
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        import ultralytics
        d["ultralytics"] = ultralytics.__version__
    except Exception:  # noqa: BLE001
        pass
    return d


# ---------------------------------------------------------------------------- CLI
def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m vision_guide.seg_model")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("export-yolo")
    s.add_argument("--synth-config", default=str(SYNTH_CONFIG))
    s.add_argument("--seg-config", default=str(SEG_CONFIG))
    s = sub.add_parser("train")
    s.add_argument("--seg-config", default=str(SEG_CONFIG))
    s.add_argument("--out")
    s = sub.add_parser("val-synth")
    s.add_argument("--weights", required=True)
    s.add_argument("--seg-config", default=str(SEG_CONFIG))
    s.add_argument("--out")
    s = sub.add_parser("predict")
    s.add_argument("--weights", required=True)
    s.add_argument("--split", choices=["dev", "eval"], required=True)
    s.add_argument("--conf", type=float, default=None, help="기본: train_seg.yaml predict_conf_floor")
    s.add_argument("--iou", type=float, default=0.7)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--lock")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--out")
    a = p.parse_args(argv)
    seg_cfg = _yaml(a.seg_config) if hasattr(a, "seg_config") else _yaml(SEG_CONFIG)
    if a.cmd == "export-yolo":
        syn = _yaml(a.synth_config)
        bop = _abs(syn["output"]["root"]) / syn["output"]["bop_dir"]
        yroot = _abs(syn["output"]["yolo"]["root"])
        ex = seg_cfg["export"]
        res = [export_yolo(bop / sp, yroot, ys, CLASS_IDS, ex["min_visib_fract"], ex["min_px"], ex["approx_eps_px"],
                           ex["min_piece_area_px"]) for sp, ys in (("train_synth", "train"), ("val_synth", "val"))]
        dy = write_data_yaml(yroot)
        (yroot / "export_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=1), f"\n-> {dy}")
    elif a.cmd == "train":
        out = Path(a.out) if a.out else RESULTS_DIR_V4 / f"seg-train_{_ts()}"
        t0 = time.perf_counter()
        best = train(_abs(seg_cfg["data"]), seg_cfg, out)
        meta = {"command": "seg_model train", "best": str(best), "best_sha256": weights_sha256(best),
                "wall_s": time.perf_counter() - t0, "seg_config": str(a.seg_config), "env": _env(),
                "label": SYNTH_TRAIN_LABEL, "license": "ultralytics AGPL-3.0"}
        (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(meta, ensure_ascii=False, indent=1))
    elif a.cmd == "val-synth":
        res = val_synth(a.weights, _abs(seg_cfg["data"]), int(seg_cfg["train"]["imgsz"]))
        out = Path(a.out) if a.out else Path(a.weights).resolve().parents[1]
        (out / "val_synth.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=1))
    elif a.cmd == "predict":
        conf = float(a.conf if a.conf is not None else seg_cfg["predict_conf_floor"])
        out = Path(a.out) if a.out else RESULTS_DIR_V4 / f"seg-predict_{a.split}_{_ts()}"
        d = predict_split(a.weights, a.split, out, conf=conf, iou=a.iou, imgsz=a.imgsz, lock_path=a.lock,
                          limit=a.limit)
        print(f"-> {d}")


if __name__ == "__main__":
    main()
