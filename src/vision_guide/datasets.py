"""BOP 형식 데이터 로더 (LM-O 우선).

경계 변환 규약
- 길이: BOP는 mm → 로더에서 m로 바꾼다 (모델 점, cam_t_m2c, 대칭 offset).
- Depth: ``imread(UNCHANGED) * depth_scale / 1000`` (depth_scale은 scene_camera.json 값) → m.
- 내부 파라미터: BOP cam_K를 이 모듈 규약(픽셀 중심 = +0.5) 그대로 쓴다 (``CameraIntrinsics.from_K``).
  근거: LM-O dev 프레임에서 정답 자세로 레이캐스팅한 실루엣과 BOP mask_visib의 일치율이
  그대로 쓸 때 0.999, OpenCV 규약으로 보고 +0.5 할 때 0.95~0.99였다 (BOP 렌더러는 OpenGL 픽셀 중심 규약).
  tests/v3/test_v3_07_datasets.py가 이 일치율을 검사한다.

개발/평가 분할: configs/v3/dataset.yaml의 im_id 목록. 평가 세트는 ``allow_eval=True`` 없이 읽을 수 없다.
dev = ``make_split``의 dev, eval = ``make_gapped_eval`` (dev와 |im_id 차| > eval_gap_im).
``load_frame(forbid_ids=...)``에 eval id 집합을 넘기면 실수로 eval 프레임을 읽을 때 ValueError.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import trimesh

from .geometry import CameraIntrinsics, make_T

HF_URL = "https://huggingface.co/datasets/bop-benchmark/{name}/resolve/main/{file}"
LMO_FILES = {"base": "lmo_base.zip", "models": "lmo_models.zip", "test": "lmo_test_bop19.zip"}
DATASET_FILES = {"lmo": LMO_FILES, "lm": {"models": "lm_models.zip"}}  # lm: v4 합성 방해물 모델만


# ---------------------------------------------------------------------------- 다운로드
def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """zip-slip 검사: 모든 항목이 dest 안에 풀리는지 확인한 뒤 압축 해제."""
    dest = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if dest != target and dest not in target.parents:
            raise ValueError(f"zip 항목이 대상 폴더 밖을 가리킨다: {info.filename}")
    zf.extractall(dest)


def download_bop(name: str, root: str | Path, parts: Sequence[str] = ("base", "models", "test"), *,
                 force: bool = False) -> Path:
    """HF bop-benchmark에서 zip을 받아 root/_zips에 두고 root/name에 푼다. 데이터셋 폴더(root/name) 반환.

    lmo_base.zip은 항목이 ``lmo/``로 시작하고 models·test zip은 ``models/``, ``test/``로 시작하므로
    항목 접두어를 보고 풀 위치를 정한다.
    """
    files = DATASET_FILES[name]
    root = Path(root)
    zdir = root / "_zips"
    zdir.mkdir(parents=True, exist_ok=True)
    for part in parts:
        fname = files[part]
        zpath = zdir / fname
        if force or not zpath.exists():
            tmp = zpath.with_suffix(".part")
            with urllib.request.urlopen(HF_URL.format(name=name, file=fname), timeout=60) as resp, \
                    open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            tmp.replace(zpath)
        marker = root / f".extracted_{fname}"
        if force or not marker.exists():
            with zipfile.ZipFile(zpath) as zf:
                prefixed = all(i.filename.startswith(f"{name}/") for i in zf.infolist())
                _safe_extract(zf, root if prefixed else root / name)
            marker.write_text("ok", encoding="utf-8")
    return root / name


# ---------------------------------------------------------------------------- 모델
@dataclass
class BopModel:
    obj_id: int
    points: np.ndarray        # 꼭짓점 (m)
    normals: np.ndarray       # 꼭짓점 법선
    diameter: float           # m (models_info)
    symmetries: list[np.ndarray]
    symmetric: bool
    mesh: trimesh.Trimesh | None = field(default=None, repr=False)


def load_models_info(ds_root: str | Path, models_dir: str = "models_eval") -> dict[int, dict]:
    path = Path(ds_root) / models_dir / "models_info.json"
    with open(path, "r", encoding="utf-8") as fh:
        return {int(k): v for k, v in json.load(fh).items()}


def _rotation_matrix(angle: float, axis: np.ndarray) -> np.ndarray:
    """Rodrigues (bop_toolkit transform.rotation_matrix와 같은 결과)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def get_symmetry_transformations(info: Mapping[str, Any], max_sym_disc_step: float = 0.01,
                                 unit_scale: float = 1e-3) -> list[np.ndarray]:
    """bop_toolkit misc.get_symmetry_transformations와 같은 규약의 4x4 목록 (병진은 unit_scale 배, 기본 mm→m).

    이산 대칭(항등 포함) × 연속 대칭 이산화(i = 1..steps-1). 연속 대칭이 있으면 BOP 원본처럼
    회전 0 단계는 넣지 않는다 (원본 규약 그대로).
    """
    disc = [np.eye(4)]
    for sym in info.get("symmetries_discrete", []):
        S = np.array(sym, dtype=float).reshape(4, 4)
        S[:3, 3] *= unit_scale
        disc.append(S)
    cont = []
    for sym in info.get("symmetries_continuous", []):
        axis = np.array(sym["axis"], dtype=float)
        offset = np.array(sym["offset"], dtype=float).reshape(3) * unit_scale
        steps = int(np.ceil(np.pi / max_sym_disc_step))
        step = 2.0 * np.pi / steps
        for i in range(1, steps):
            R = _rotation_matrix(i * step, axis)
            cont.append(make_T(R, -R @ offset + offset))
    if not cont:
        return disc
    return [make_T(C[:3, :3] @ D[:3, :3], C[:3, :3] @ D[:3, 3] + C[:3, 3]) for D in disc for C in cont]


def load_model(ds_root: str | Path, obj_id: int, models_dir: str = "models_eval",
               max_sym_disc_step: float = 0.01, info: Mapping[str, Any] | None = None) -> BopModel:
    root = Path(ds_root) / models_dir
    info = info if info is not None else load_models_info(ds_root, models_dir)[obj_id]
    mesh = trimesh.load(root / f"obj_{obj_id:06d}.ply", process=False, force="mesh")
    mesh.apply_scale(1e-3)
    pts = np.asarray(mesh.vertices, dtype=float)
    nrm = np.asarray(mesh.vertex_normals, dtype=float)
    syms = get_symmetry_transformations(info, max_sym_disc_step)
    symmetric = bool(info.get("symmetries_discrete") or info.get("symmetries_continuous"))
    return BopModel(obj_id, pts, nrm, float(info["diameter"]) * 1e-3, syms, symmetric, mesh)


# ---------------------------------------------------------------------------- 프레임
@dataclass
class BopFrame:
    scene_id: int
    im_id: int
    depth_m: np.ndarray
    rgb: np.ndarray | None
    intr: CameraIntrinsics
    gts: list[dict]                 # {obj_id, T_cam_obj (m), inst}
    masks_visib: list[np.ndarray]   # gt 인스턴스 순서
    targets: list[dict] = field(default_factory=list)  # test_targets 항목 {obj_id, inst_count}
    visib_fract: list[float] = field(default_factory=list)


def make_split(im_ids: Sequence[tuple[int, int]] | Sequence[int], seed: int, n_dev: int, n_eval: int
               ) -> dict[str, list]:
    """정렬된 목록을 시드 고정으로 섞어 앞 n_dev = dev, 다음 n_eval = eval. 겹치지 않는다.

    dataset.yaml의 dev 목록은 이 함수로 만들었다. eval은 인접 프레임 누설 때문에 ``make_gapped_eval``로 다시 뽑는다.
    """
    ids = sorted(im_ids)
    if n_dev + n_eval > len(ids):
        raise ValueError("n_dev + n_eval이 전체보다 크다")
    perm = np.random.default_rng(seed).permutation(len(ids))
    pick = [ids[i] for i in perm]
    return {"dev": sorted(pick[:n_dev]), "eval": sorted(pick[n_dev:n_dev + n_eval])}


def make_gapped_eval(all_ids: Sequence[tuple[int, int]], dev_ids: Sequence[tuple[int, int]], seed: int,
                     n_eval: int, gap: int) -> list[tuple[int, int]]:
    """dev와 시간적으로 떨어진 eval 선택: 같은 scene에서 모든 dev 프레임과 |im_id 차| > gap 인 후보 중 seed로 n_eval장.

    연속 시퀀스에서 인접 프레임이 dev/eval에 나뉘어 들어가는 누설을 막는다 (dev는 그대로 둔다).
    """
    dev = {(int(s), int(i)) for s, i in dev_ids}
    cand = sorted((int(s), int(i)) for s, i in all_ids
                  if (int(s), int(i)) not in dev
                  and all(not (ds_ == int(s) and abs(di - int(i)) <= gap) for ds_, di in dev))
    if n_eval > len(cand):
        raise ValueError(f"간격 조건을 만족하는 후보({len(cand)})가 n_eval({n_eval})보다 적다")
    perm = np.random.default_rng(seed).permutation(len(cand))
    return sorted(cand[i] for i in perm[:n_eval])


def min_split_gap(dev_ids: Sequence[tuple[int, int]], eval_ids: Sequence[tuple[int, int]]) -> float:
    """같은 scene의 dev-eval 쌍 중 최소 |im_id 차| (같은 scene 쌍이 없으면 inf)."""
    gaps = [abs(int(a[1]) - int(b[1])) for a in dev_ids for b in eval_ids if int(a[0]) == int(b[0])]
    return float(min(gaps)) if gaps else float("inf")


def load_targets(ds_root: str | Path, targets_file: str) -> dict[tuple[int, int], list[dict]]:
    with open(Path(ds_root) / targets_file, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    out: dict[tuple[int, int], list[dict]] = {}
    for t in items:
        out.setdefault((int(t["scene_id"]), int(t["im_id"])), []).append(
            {"obj_id": int(t["obj_id"]), "inst_count": int(t["inst_count"])})
    return out


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_frame(ds_root: str | Path, split_dir: str, scene_id: int, im_id: int, *, load_rgb: bool = True,
               targets: list[dict] | None = None, _cache: dict | None = None,
               forbid_ids: set | None = None) -> BopFrame:
    import imageio.v3 as iio

    # 평가 세트 가드: 금지 id이면 파일을 열기 전에 막는다
    if forbid_ids is not None and (int(scene_id), int(im_id)) in forbid_ids:
        raise ValueError(f"평가 세트 프레임은 읽을 수 없다: scene {scene_id} im {im_id}")

    sdir = Path(ds_root) / split_dir / f"{scene_id:06d}"
    cache = _cache if _cache is not None else {}
    if sdir not in cache:
        cache[sdir] = (_read_json(sdir / "scene_camera.json"), _read_json(sdir / "scene_gt.json"),
                       _read_json(sdir / "scene_gt_info.json") if (sdir / "scene_gt_info.json").exists() else {})
    cams, gts_all, info_all = cache[sdir]
    cam = cams[str(im_id)]
    depth_raw = iio.imread(sdir / "depth" / f"{im_id:06d}.png")
    depth_m = depth_raw.astype(float) * float(cam.get("depth_scale", 1.0)) / 1000.0
    h, w = depth_m.shape
    intr = CameraIntrinsics.from_K(np.array(cam["cam_K"]).reshape(3, 3), w, h)
    rgb = None
    if load_rgb:
        for ext in ("png", "jpg"):
            p = sdir / "rgb" / f"{im_id:06d}.{ext}"
            if p.exists():
                rgb = iio.imread(p)
                break
    gts, masks, vis = [], [], []
    infos = info_all.get(str(im_id), [])
    for k, g in enumerate(gts_all.get(str(im_id), [])):
        T = make_T(np.array(g["cam_R_m2c"]).reshape(3, 3), np.array(g["cam_t_m2c"]).reshape(3) * 1e-3)
        gts.append({"obj_id": int(g["obj_id"]), "T_cam_obj": T, "inst": k})
        mp = sdir / "mask_visib" / f"{im_id:06d}_{k:06d}.png"
        masks.append(iio.imread(mp) > 0 if mp.exists() else np.zeros((h, w), dtype=bool))
        vis.append(float(infos[k]["visib_fract"]) if k < len(infos) else float("nan"))
    return BopFrame(scene_id, im_id, depth_m, rgb, intr, gts, masks, list(targets or []), vis)


def iter_targets(ds_root: str | Path, split_cfg: Mapping[str, Any], split: str = "dev",
                 allow_eval: bool = False, *, load_rgb: bool = False, limit: int | None = None
                 ) -> Iterator[BopFrame]:
    """분할(dev/eval)에 속한 프레임을 차례로 낸다 (frame.targets = 평가 대상 객체).

    split_cfg: dataset.yaml의 bop 절 (split_dir, targets_file, splits: {dev: [[scene, im], ...], eval: ...}).
    """
    if split not in ("dev", "eval"):
        raise ValueError(f"split must be 'dev' or 'eval': {split!r}")
    if split == "eval" and not allow_eval:
        raise ValueError("평가 세트는 최종 1회만 쓴다: allow_eval=True 없이 eval을 읽을 수 없다")
    targets = load_targets(ds_root, split_cfg["targets_file"])
    ids = [tuple(x) for x in split_cfg["splits"][split]]
    # dev를 읽을 때는 eval id를 금지 목록으로 넘긴다
    forbid = None if split == "eval" else {(int(s), int(i)) for s, i in split_cfg["splits"].get("eval", [])}
    cache: dict = {}
    for n, (scene_id, im_id) in enumerate(ids):
        if limit is not None and n >= limit:
            break
        yield load_frame(ds_root, split_cfg["split_dir"], int(scene_id), int(im_id), load_rgb=load_rgb,
                         targets=targets.get((int(scene_id), int(im_id)), []), _cache=cache,
                         forbid_ids=forbid)
