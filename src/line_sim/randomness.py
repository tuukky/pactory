"""Keyed random streams: every draw maps to (seed, job_id, operation, draw_kind).

Never uses Python's built-in ``hash()`` (salted per process). Unknown operation
names are encoded with the first 8 bytes of their SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

DOMAIN = 0x4C53  # "LS"
OP_CODES = {"ORDER": 0, "PREP": 1, "ASM": 2, "TEST": 3, "REWORK": 4, "PACK": 5}
DRAW_CODES = {"permutation": 0, "duration_z": 1, "fail_u": 2, "recover_u": 3}

ORDER_JOB_ID = 0  # job_id slot used for order-level (non-job) draws


def op_code(op: str) -> int:
    if op in OP_CODES:
        return OP_CODES[op]
    return int.from_bytes(hashlib.sha256(op.encode("utf-8")).digest()[:8], "big")


def rng_for(seed: int, job_id: int, op: str, kind: str) -> np.random.Generator:
    ss = np.random.SeedSequence(entropy=int(seed),
                                spawn_key=(DOMAIN, int(job_id), op_code(op), DRAW_CODES[kind]))
    return np.random.Generator(np.random.PCG64(ss))


def order_rng(seed: int) -> np.random.Generator:
    return rng_for(seed, ORDER_JOB_ID, "ORDER", "permutation")


@dataclass(frozen=True)
class JobDraws:
    """Pre-generated draws for one job; identical across scenarios for the same seed."""

    z: Mapping[str, float]  # station_id -> standard normal draw for duration
    fail_u: float
    recover_u: float


def quality_ops(stations: Mapping[str, Mapping]) -> tuple[str, str]:
    """Operation names used as keys for the fail / recover draws."""
    inspect = next((s for s, st in stations.items() if (st.get("route") or {}).get("type") == "inspect"), "TEST")
    rework = next((s for s, st in stations.items() if (st.get("route") or {}).get("type") == "rework"), "REWORK")
    return inspect, rework


def draw_job(seed: int, job_id: int, stations: Mapping[str, Mapping]) -> JobDraws:
    z = {sid: float(rng_for(seed, job_id, sid, "duration_z").standard_normal()) for sid in stations}
    inspect_op, rework_op = quality_ops(stations)
    fail_u = float(rng_for(seed, job_id, inspect_op, "fail_u").random())
    recover_u = float(rng_for(seed, job_id, rework_op, "recover_u").random())
    return JobDraws(z=z, fail_u=fail_u, recover_u=recover_u)


def lognormal_from_z(mean: float, cv: float, z: float, enabled: bool = True) -> float:
    """Lognormal sample with given mean and CV built from a standard normal ``z``.

    sigma^2 = ln(1 + cv^2), mu = ln(mean) - sigma^2 / 2, value = exp(mu + sigma * z).
    Returns ``mean`` exactly in deterministic mode, for cv == 0 or mean == 0.
    """
    if not enabled or cv == 0 or mean == 0:
        return float(mean)
    s = math.sqrt(math.log1p(cv * cv))
    return float(mean * math.exp(s * z - s * s / 2.0))
