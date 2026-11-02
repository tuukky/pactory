"""v3 검증 테스트 공통 도우미.

tests/conftest.py(v1)·tests/v2/v2_helpers.py와 모듈 이름이 겹치지 않도록 conftest를 두지 않고 이 파일을 import한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_guide.geometry import CameraIntrinsics, load_camera

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "v3"
CAMERA_PATH = CONFIG_DIR / "camera.yaml"
PIPELINE_PATH = CONFIG_DIR / "pipeline.yaml"
GRIPPER_PATH = CONFIG_DIR / "gripper.yaml"
DATASET_PATH = CONFIG_DIR / "dataset.yaml"
BOP_ROOT = ROOT / "data" / "v3" / "bop"


def camera() -> tuple[CameraIntrinsics, np.ndarray]:
    return load_camera(CAMERA_PATH)


def small_intr() -> CameraIntrinsics:
    """빠른 렌더링 테스트용 320x240 카메라 (합성 가정)."""
    return CameraIntrinsics(fx=307.5, fy=307.5, cx=160.0, cy=120.0, width=320, height=240)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def T_cam_floor(T_base_cam: np.ndarray, center=(0.6, 0.0, 0.0)) -> np.ndarray:
    """빈 바닥 좌표(base와 같은 방향, 원점 = 빈 중심) → 카메라 좌표."""
    from vision_guide.geometry import inv_T, make_T
    return inv_T(T_base_cam) @ make_T(None, center)


def pipeline_cfg() -> dict:
    import yaml
    with open(PIPELINE_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def housing_model():
    """합성 하우징 ObjectModel (외부 표면 샘플, C3 대칭, 대칭축 +z)."""
    from vision_guide import cad
    from vision_guide.pose import ObjectModel
    spec = cad.HousingSpec()
    pts, nrm = cad.sample_model_points(cad.housing_parts(spec), 6000, seed=0)
    return ObjectModel("housing", pts, nrm, cad.model_diameter(cad.make_housing(spec)),
                       cad.housing_symmetries(spec), np.array([0.0, 0.0, 1.0]))


def gripper_cfg() -> dict:
    import yaml
    with open(GRIPPER_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
