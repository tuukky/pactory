"""v2 검증 테스트용 설정 생성기.

tests/conftest.py(v1)와 모듈 이름이 겹치지 않도록 conftest를 두지 않고 이 파일을 import한다.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping

from factory_sim.config import load_config

ROOT = Path(__file__).resolve().parents[2]
REV_A_PATH = ROOT / "configs" / "v2" / "rev_A.yaml"


def rev_a() -> dict:
    return load_config(REV_A_PATH)


def short(cfg: dict, warmup_h: float = 4, observation_h: float = 16) -> dict:
    out = copy.deepcopy(cfg)
    out["run"] = {"warmup_s": warmup_h * 3600.0, "observation_s": observation_h * 3600.0}
    return out


def explicit_orders(skus: Iterable[str], arrivals: Iterable[float] | None = None) -> list[dict]:
    skus = list(skus)
    arrivals = list(arrivals) if arrivals is not None else [0.0] * len(skus)
    return [{"id": i + 1, "sku": s, "arrival_s": float(t)} for i, (s, t) in enumerate(zip(skus, arrivals))]


def single_line(stages: Mapping[str, Any], *, orders: list[dict], caps: Mapping[str, int | None] | None = None,
                machines: Mapping[str, int] | None = None, setups: Mapping[str, float] | None = None,
                operator: Mapping[str, str] | None = None, groups: Iterable[str] = ("A", "B", "C"),
                kit_slots: int = 100000, observation_s: float = 1.0e6) -> dict:
    """단일 경로 RM → 스테이션들 → SHIP. 결정론, 고장 없음, instant 운반, 작업자 무제한."""
    caps = caps or {}
    machines = machines or {}
    setups = setups or {}
    operator = operator or {}
    groups = list(groups)
    stations: dict[str, Any] = {}
    for sid, t in stages.items():
        times = dict(t) if isinstance(t, Mapping) else {g: float(t) for g in groups}
        st = {"machines": machines.get(sid, 1), "times_s": times, "operator": operator.get(sid, "none"),
              "failure_class": None, "input_buffer": {"capacity": caps.get(sid)}}
        if setups.get(sid):
            st["setup_group_change_s"] = setups[sid]
        stations[sid] = st
    share = 1.0 / len(groups)
    return {
        "schema_version": 2,
        "model_name": "v2_test_line",
        "run": {"warmup_s": 0.0, "observation_s": float(observation_s)},
        "stochastic": {"enabled": False, "cv": 0.0},
        "products": {"groups": {g: {"share": share, "fail_rate": 0.0, "recover_prob": 1.0} for g in groups},
                     "sku_factors": [1.0]},
        "demand": {"mode": "explicit", "rate_per_h": None, "due_offset_s": 1.0e6, "explicit": orders},
        "release": {"kit_slots": kit_slots, "slot_release_at": "TERMINAL"},
        "housing": {"enabled": False},
        "routing": {"parts": {"X": {"from": "RM", "path": list(stages), "to": "SHIP"}},
                    "assembly": None, "inspect": None, "ship_to": "SHIP"},
        "dispatch": {"t_max_s": 3600},
        "stations": stations,
        "failures": {"enabled": False, "distribution": "fixed", "mtbf_scale": 1.0, "classes": {}},
        "operators": {"mode": "unlimited", "walk_speed_mps": 1.2, "skills": {}},
        "agv": {"mode": "instant"},
        "layout": {"detour_factor": 1.0, "nodes": {}},
        "time_scale": {},
    }


def motor_config(n_orders: int, *, skus: Iterable[str] | None = None, arrivals: Iterable[float] | None = None,
                 deterministic: bool = True, failures: bool = False, observation_s: float = 2.0e6) -> dict:
    """Rev.A 공장 구조 + explicit 주문 (작은 검증용)."""
    cfg = rev_a()
    cycle = ["A1", "B2", "C3", "A4"]
    skus = list(skus) if skus is not None else [cycle[i % 4] for i in range(n_orders)]
    cfg["demand"] = {"mode": "explicit", "rate_per_h": None, "due_offset_s": 115200.0,
                     "explicit": explicit_orders(skus, arrivals)}
    cfg["run"] = {"warmup_s": 0.0, "observation_s": float(observation_s)}
    cfg["stochastic"]["enabled"] = not deterministic
    cfg["failures"]["enabled"] = failures
    cfg["housing"]["initial_on_hand"] = max(cfg["housing"]["initial_on_hand"], n_orders + 1)
    return cfg
