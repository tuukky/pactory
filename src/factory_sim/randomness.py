"""키 기반 난수 (공통 난수, CRN).

모든 추출은 ``SeedSequence(entropy=seed, spawn_key=(DOMAIN, *key))``로 만든 독립 스트림에서
나온다. 문자열 키는 v1과 같은 방식(SHA-256 앞 8바이트)으로 정수화하므로 PYTHONHASHSEED와 무관하다.

- 처리시간: (order_id, part, op, "dur" | "dur2")
- 품질: (order_id, "MOTOR", "EOL", "fail_u") / (order_id, "MOTOR", "RW", "recover_u")
- 고장: (machine_id, k, "ttf" | "repair") — 설비 k번째 고장은 인원·AGV·규칙과 무관하게 같다.
- 도착: ("arrivals", "gap" | "group" | "sku") — 시드에만 의존, 개정 간 같은 수요를 공유한다.

배정·배차·디스패칭은 규칙 기반이라 난수를 쓰지 않는다.
"""

from __future__ import annotations

import numpy as np

from line_sim.randomness import lognormal_from_z, op_code  # noqa: F401 - re-export

DOMAIN = 0x4653  # "FS"


def _code(x: int | str) -> int:
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return int(x)
    return op_code(str(x))


def rng(seed: int, *key: int | str) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=(DOMAIN, *(_code(k) for k in key)))
    return np.random.Generator(np.random.PCG64(ss))


def duration_z(seed: int, order_id: int, part: str, op: str, visit: int = 1) -> float:
    kind = "dur" if visit == 1 else f"dur{visit}"
    return float(rng(seed, order_id, part, op, kind).standard_normal())


def fail_u(seed: int, order_id: int) -> float:
    return float(rng(seed, order_id, "MOTOR", "EOL", "fail_u").random())


def recover_u(seed: int, order_id: int) -> float:
    return float(rng(seed, order_id, "MOTOR", "RW", "recover_u").random())


def failure_draw(seed: int, machine_id: str, k: int) -> tuple[float, float]:
    """(단위 지수 표본, 수리시간 표준정규 z). ttf = unit_exp × MTBF × mtbf_scale."""
    e = float(rng(seed, machine_id, k, "ttf").standard_exponential())
    z = float(rng(seed, machine_id, k, "repair").standard_normal())
    return e, z


def arrival_draws(seed: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(단위 지수 간격, 군 선택 균등난수, SKU 선택 균등난수) 각 n개.

    간격 = E_i / λ 이므로 도착률을 바꿔도 같은 E_i·SKU 표본을 쓴다.
    """
    gaps = rng(seed, "arrivals", "gap").standard_exponential(n)
    ug = rng(seed, "arrivals", "group").random(n)
    us = rng(seed, "arrivals", "sku").random(n)
    return gaps, ug, us
