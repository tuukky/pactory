"""v4 실행부: 플래그 끄면 run_target("baseline") == v3 run_pipeline, 2색 장면 색 순위, 첫 실패 귀속,
select_candidate 실시간 == 재생(replay-tune 벡터판)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from v3_helpers import T_cam_floor, camera, housing_model  # noqa: E402

from v4_helpers import baseline_cfg, colored_box, fake_frame, intr_small, pose_at, v41_cfg  # noqa: E402
from vision_guide import cad  # noqa: E402
from vision_guide.color import HistSpec, model_color_hist, rank_clusters  # noqa: E402
from vision_guide.consistency import decide_code  # noqa: E402
from vision_guide.failures import FailureCode  # noqa: E402
from vision_guide.geometry import depth_to_points  # noqa: E402
from vision_guide.pipeline_v4 import CandidateEval, FrameCache, V4Model, prepare_frame, run_target, select_candidate  # noqa: E402
from vision_guide.pose import PoseResult, pick_best, run_pipeline  # noqa: E402
from vision_guide.run_v4 import _codes_vec, replay_select  # noqa: E402
from vision_guide.segment import SegmentResult, filter_by_extent, preprocess, segment_scene  # noqa: E402


_EQ_SCRIPT = r"""
import sys
sys.path[:0] = [{v3!r}, {v4!r}]
import numpy as np, open3d as o3d
from v3_helpers import T_cam_floor, camera, housing_model
from v4_helpers import baseline_cfg, fake_frame
from vision_guide import cad
from vision_guide.geometry import depth_to_points
from vision_guide.pipeline_v4 import V4Model, prepare_frame, run_target
from vision_guide.pose import pick_best, run_pipeline
from vision_guide.segment import SegmentResult, filter_by_extent, segment_scene
cfg = baseline_cfg()
cfg["ransac"] = {{**cfg["ransac"], "n_restarts": 2}}
model = housing_model()
intr, T_base_cam = camera()
sc = cad.make_simple_scene(3, intr.pinhole(), T_cam_floor(T_base_cam), seed=2)
o3d.utility.random.seed(123)                      # v3 eval-bop 순서
pts, _ = depth_to_points(sc.depth_m, sc.intr)
seg_all = segment_scene(pts, cfg, None)
cl = filter_by_extent(seg_all.clusters, cfg, model.diameter)
sr = SegmentResult(cl, seg_all.plane_model, seg_all.n_input, dict(seg_all.timings_ms), seg_all.n_raw_clusters)
res, _ = run_pipeline(sc.depth_m, sc.intr, model, cfg, seed=7, segmentation=sr)
ref = pick_best(res)
o3d.utility.random.seed(123)                      # v4 기준선 경로
fc = prepare_frame(fake_frame(sc.depth_m, sc.intr), cfg, method="baseline")
tr = run_target("baseline", fc, V4Model(model, None), cfg, seed=7)
assert tr.code == ref.code and len(tr.candidates) == len(res)
assert tr.best.fitness == ref.fitness and tr.best.rmse == ref.rmse
assert np.array_equal(tr.best.T_cam_obj, ref.T_cam_obj)
assert [c.pose.code for c in tr.candidates] == [r.code for r in res]
assert [c.pose.fitness for c in tr.candidates] == [r.fitness for r in res]
print("EQUAL", len(res))
"""


def test_baseline_target_equals_v3_call():
    """1스레드(결정적)에서 v4 기준선 경로 == v3 호출. 16스레드 RANSAC은 같은 프로세스에서도 결과가 흔들린다."""
    import os
    import subprocess
    here = Path(__file__).resolve().parent
    code = _EQ_SCRIPT.format(v3=str(here.parent / "v3"), v4=str(here))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": str(here.parents[1] / "src")}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.startswith("EQUAL")


def test_two_color_scene_rank():
    intr = intr_small()
    red, blue = colored_box(rgb=(220, 30, 30)), colored_box(rgb=(30, 40, 220))
    Ta, Tb = pose_at(0.5, x=-0.06), pose_at(0.5, x=0.06)
    depth, inst = cad.render_depth([(blue, Ta), (red, Tb)], intr)  # 0 = 파랑, 1 = 빨강
    rgb = np.zeros((intr.height, intr.width, 3), np.uint8)
    rgb[inst == 0] = (30, 40, 220)
    rgb[inst == 1] = (220, 30, 30)
    cfg = baseline_cfg()
    clusters = [preprocess(depth_to_points(depth, intr, inst == k)[0], cfg) for k in (0, 1)]
    spec = HistSpec()
    ref = model_color_hist(red, spec, 4000, seed=0)
    ranked = rank_clusters(clusters, rgb, intr, ref, spec, "bhattacharyya", min_px=5)
    assert ranked[0][0] == 1 and ranked[0][1] > 0.9 and ranked[1][1] < 0.3
    # min_px보다 작으면 sim 0
    ranked2 = rank_clusters(clusters, rgb, intr, ref, spec, "bhattacharyya", min_px=10 ** 6)
    assert [s for _, s in ranked2] == [0.0, 0.0]


def _cand(code, fit, vis, rank):
    p = PoseResult(np.eye(4), fit, 0.001, vis, code)
    return CandidateEval(rank, "dbscan", rank, 0.5, p, {}, code)


def test_select_candidate_rule():
    c = [_cand(FailureCode.LOW_FITNESS, 0.9, 0.9, 0), _cand(FailureCode.OK, 0.7, 0.8, 1),
         _cand(FailureCode.OK, 0.8, 0.8, 2)]
    best, code = select_candidate(c)
    assert code == FailureCode.OK and best is c[2].pose
    allfail = [_cand(FailureCode.LOW_VISIBILITY, 0.9, 0.3, 0), _cand(FailureCode.DEPTH_INCONSISTENT, 0.9, 0.9, 1)]
    best, code = select_candidate(allfail)
    assert code == FailureCode.DEPTH_INCONSISTENT
    assert select_candidate([]) == (None, FailureCode.NO_CLUSTER)


def test_replay_equals_realtime_rule():
    """무작위 후보에서 decide_code + select_candidate (실시간) == _codes_vec + replay_select (재생)."""
    rng = np.random.default_rng(0)
    checks = v41_cfg()["checks"]
    g = {"top_k": 5, "tol_free_m": 0.015, "max_violation": 0.15, "h_tol": 15, "min_color_match": 0.5,
         "achrom_s_max_obs": 90, "min_vis_relaxed": 0.2, "min_match_px": 100}
    rows = []
    v3c = ["OK", "LOW_VISIBILITY", "LOW_FITNESS", "HIGH_RMSE", "FEATURE_FAIL"]
    for t in range(60):
        for r in range(int(rng.integers(1, 9))):
            code = v3c[int(rng.choice(5, p=[0.4, 0.3, 0.1, 0.1, 0.1]))]
            has = code != "FEATURE_FAIL"
            fit, vis = float(rng.uniform(0.3, 1)), float(rng.uniform(0, 1))
            rows.append({"t": t, "rank": r, "v3_code": code, "visible_ratio": vis if has else np.nan,
                         "score": fit * vis if has else -1.0, "fitness": fit, "correct_pose": bool(rng.random() < 0.4),
                         "n_valid": float(rng.integers(0, 600)), "n_match": float(rng.integers(0, 300)),
                         "viol@0.015": float(rng.uniform(0, 0.4)), "color_ratio@15_90": float(rng.uniform(0, 1)),
                         "color_n@15_90": float(rng.integers(0, 200))})
    c = pd.DataFrame(rows)
    ok = _codes_vec(c, g, checks)
    acc, cor = replay_select(c, ok, g["top_k"], 60)
    chk = {**checks, "depth": {**checks["depth"], "max_violation": g["max_violation"]},
           "color_match": {**checks["color_match"], "min_match": g["min_color_match"]},
           "visibility_relax": {**checks["visibility_relax"], "min_vis_relaxed": g["min_vis_relaxed"],
                                "min_match_px": g["min_match_px"]}}
    for t in range(60):
        sub = c[(c.t == t) & (c["rank"] < g["top_k"])]
        cands = []
        for _, r in sub.iterrows():
            v3 = FailureCode(r.v3_code)
            diag = {"n_valid": r.n_valid, "n_match": r.n_match, "viol": r["viol@0.015"],
                    "color_ratio": r["color_ratio@15_90"], "color_n": r["color_n@15_90"]}
            has = v3 != FailureCode.FEATURE_FAIL
            p = PoseResult(np.eye(4) if has else None, r.fitness if has else 0.0, 0.001,
                           r.visible_ratio, v3)
            p.correct = bool(r.correct_pose)
            code = decide_code(v3, r.visible_ratio, diag, chk) if has else v3
            cands.append(CandidateEval(int(r["rank"]), "dbscan", int(r["rank"]), 0.0, p, diag, code))
            assert bool(code == FailureCode.OK) == bool(ok[r.name])
        best, code = select_candidate(cands)
        assert acc[t] == (code == FailureCode.OK)
        assert cor[t] == (code == FailureCode.OK and best.correct)
    _ = FrameCache
