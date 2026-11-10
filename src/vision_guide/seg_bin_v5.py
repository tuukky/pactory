"""PLAN_v5 S1 B1 — 하우징 전용 YOLO11n-seg 학습·예측 (env ``pactory-seg``: torch + ultralytics, open3d 없음).

    python -m vision_guide.seg_bin_v5 train                      # data/v5/yolo (bin_v5 export-yolo) → results/v5/s1_bin/seg-train
    python -m vision_guide.seg_bin_v5 val --weights <best.pt>    # 합성 val Mask F1 → 융합 conf (0.05 단위)
    python -m vision_guide.seg_bin_v5 predict --weights <best.pt> --split dev
    python -m vision_guide.seg_bin_v5 predict --weights <best.pt> --split eval   # B1 설정 동결(frozen) + 가중치 해시 일치일 때만

예측 파일 형식은 v4 seg_model과 같다 (``frames/{scene:06d}_{view:06d}.npz``). 학습 데이터는 합성만.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .seg_model import conf_from_f1, frame_pred_path, save_frame_predictions, weights_sha256

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CFG = ROOT / "configs" / "v5" / "train_seg_housing.yaml"
B1_CFG = ROOT / "configs" / "v5" / "pipeline_bin_B1.yaml"
RAW = ROOT / "data" / "v5" / "bin" / "raw"
OUT = ROOT / "results" / "v5" / "s1_bin"


def _yaml(p):
    import yaml
    return yaml.safe_load(Path(p).read_text(encoding="utf-8"))


def _runtime_data_yaml(cfg) -> Path:
    """data.yaml의 path를 현재 실행 환경(Windows 또는 WSL /mnt/c) 경로로 다시 쓴 사본.

    Windows 앱 제어 정책이 torchvision 네이티브 모듈(_C.pyd)을 차단해(2026-11-09, WinError 4551)
    학습·예측은 WSL2(같은 torch·ultralytics 버전, 같은 GPU)에서 한다."""
    src = ROOT / cfg["data"]
    txt = src.read_text(encoding="utf-8")
    lines = [f"path: {src.parent.as_posix()}" if l.startswith("path:") else l for l in txt.splitlines()]
    out = src.parent / "data_runtime.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def cmd_train(a):
    from ultralytics import YOLO
    cfg = _yaml(TRAIN_CFG)
    out = OUT / "seg-train"
    t0 = time.time()
    model = YOLO(str(ROOT / cfg["model"]))
    model.train(data=str(_runtime_data_yaml(cfg)), project=str(out.parent), name=out.name, exist_ok=True, **cfg["train"])
    best = out / "weights" / "best.pt"
    meta = {"best": str(best), "best_sha256": weights_sha256(best), "wall_s": time.time() - t0,
            "train_cfg_sha256": weights_sha256(TRAIN_CFG), "label": "합성 데이터(BlenderProc 하우징 bin)로 학습",
            "license": "ultralytics AGPL-3.0"}
    (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(meta)


def cmd_val(a):
    from ultralytics import YOLO
    cfg = _yaml(TRAIN_CFG)
    m = YOLO(a.weights)
    r = m.val(data=str(_runtime_data_yaml(cfg)), split="val", imgsz=640, batch=8, device=0, plots=False, verbose=False)
    px = np.asarray(getattr(r.seg, "px", np.linspace(0, 1, 1000)))
    out = {"mask_map50": float(r.seg.map50), "mask_map50_95": float(r.seg.map), "box_map50": float(r.box.map50),
           "fusion_conf": conf_from_f1(px, np.asarray(r.seg.f1_curve)), "weights_sha256": weights_sha256(a.weights),
           "label": "합성 검증 세트 (val)"}
    (OUT / "seg-train" / "val_synth.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)


def cmd_predict(a):
    import torch
    from ultralytics import YOLO
    split = a.split
    if split == "eval":
        b1 = _yaml(B1_CFG)
        if not b1.get("frozen"):
            raise PermissionError("eval 예측은 B1 설정 동결(frozen: true) 뒤에만")
        if b1["seg"]["weights_sha256"] != weights_sha256(a.weights):
            raise PermissionError("가중치 해시가 B1 설정과 다르다")
    out = OUT / f"seg-predict_{split}"
    if split == "eval" and out.exists():
        raise PermissionError(f"eval 예측은 1회만: {out}")
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    files = sorted((RAW / split).glob("*.npz"))
    rows, warm = [], False
    t_start = time.perf_counter()
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        img = np.ascontiguousarray(z["rgb"][0][..., ::-1])   # RGB → BGR (ultralytics 규약)
        if not warm:
            model.predict(img, conf=0.05, iou=0.7, imgsz=640, device=0, retina_masks=True, verbose=False)
            warm = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(img, conf=0.05, iou=0.7, imgsz=640, device=0, retina_masks=True, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if r.masks is not None and len(r.boxes):
            sc = r.boxes.conf.cpu().numpy().astype(float)
            mk = r.masks.data.cpu().numpy() > 0.5
        else:
            sc, mk = np.zeros(0), np.zeros((0, img.shape[0], img.shape[1]), bool)
        save_frame_predictions(frame_pred_path(out, meta["scene_id"], 0), [1] * len(sc), sc, mk, ms)
        rows.append({"scene_id": meta["scene_id"], "n_pred": int(len(sc)), "infer_ms": ms})
    meta = {"split": split, "weights": a.weights, "weights_sha256": weights_sha256(a.weights), "n_frames": len(rows),
            "median_infer_ms": float(np.median([r["infer_ms"] for r in rows])) if rows else None,
            "wall_s": time.perf_counter() - t_start, "created_utc": datetime.now(timezone.utc).isoformat(),
            "label": "합성 하우징 bin (BlenderProc) 예측"}
    (out / "predictions.json").write_text(json.dumps({"frames": rows}, indent=1), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(meta)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m vision_guide.seg_bin_v5")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    s = sub.add_parser("val"); s.add_argument("--weights", required=True)
    s = sub.add_parser("predict"); s.add_argument("--weights", required=True); s.add_argument("--split", required=True)
    a = ap.parse_args(argv)
    {"train": cmd_train, "val": cmd_val, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
