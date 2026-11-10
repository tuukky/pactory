"""v5 보고서·발표용 그림 (eval 결과만 읽음) → docs/v5/fig/

    python tools/v5/make_figures.py results/v5/s1_bin/eval-final_<ts> [scene_id ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "v5" / "bin" / "raw" / "eval"
PRED = ROOT / "results" / "v5" / "s1_bin" / "seg-predict_eval"
FIG = ROOT / "docs" / "v5" / "fig"


def _edge(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    e = np.zeros_like(m)
    e[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    return e


def bin_scene(rec_dir: Path, scene_id: int, cond: str = "base", conf: float = 0.6):
    from vision_guide.seg_model import load_frame_predictions
    raw = next(f for f in sorted(RAW.glob("*.npz")) if json.loads(str(np.load(f)["meta"]))["scene_id"] == scene_id)
    z = np.load(raw)
    rgb, inst, T = z["rgb"][0].copy(), z["inst"][0], z["T_base_obj"]
    recs = [json.loads(l) for l in (rec_dir / "pick_records.jsonl").read_text(encoding="utf-8").splitlines()]
    rec = next(r for r in recs if r["pipeline"] == "B1" and r["condition"] == cond and r["scene_id"] == scene_id)
    preds, _ = load_frame_predictions(PRED, scene_id, 0)
    out = rgb.astype(float)
    for p in preds:
        if float(p["score"]) < conf:
            continue
        mk = p["mask"]
        out[mk] = out[mk] * .78 + np.array([56, 189, 248]) * .22
        out[_edge(mk)] = [56, 189, 248]
    if rec.get("T_base_obj_gt") is not None:
        g = np.asarray(rec["T_base_obj_gt"])[:3, 3]
        k = int(np.argmin(np.linalg.norm(T[:, :3, 3] - g, axis=1)))
        tm = inst == k
        out[tm] = out[tm] * .45 + np.array([16, 185, 129]) * .55
        e = _edge(tm)
        for _ in range(2):
            e = e | np.roll(e, 1, 0) | np.roll(e, -1, 0) | np.roll(e, 1, 1) | np.roll(e, -1, 1)
        out[e] = [16, 185, 129]
    ys, xs = np.where(inst >= -1)  # 전체
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8)).crop((150, 0, 490, 480))
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"bin_{cond}_{scene_id}.png"
    img.save(p)
    Image.fromarray(rgb).save(FIG / f"bin_{cond}_{scene_id}_rgb.png")
    print(p, "B1:", rec["correct_pick"], rec["first_code"], "n_pred≥conf:", sum(float(q["score"]) >= conf for q in preds))


if __name__ == "__main__":
    d = Path(sys.argv[1])
    for s in (sys.argv[2:] or ["base:1000"]):
        c, i = s.split(":")
        bin_scene(d, int(i), c)
