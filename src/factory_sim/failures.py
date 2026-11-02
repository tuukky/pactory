"""설비 고장: 가공시간 기준 고장 시계, 즉시 수리, 남은 가공시간 보존.

정비 인력은 모델링하지 않는다(PLAN_v2 §3.4). 고장이 나면 바로 수리가 시작되고
DOWN 시간 = 수리시간 표본이다. 고장 시계는 PROCESSING 시간에만 흐르므로 인터럽트 없이
가공 시작 시점에 고장 시점을 계산할 수 있다.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping

import simpy

from .entities import Machine, MachineState
from .logging import EventType
from .randomness import failure_draw, lognormal_from_z


class FailureClock:
    def __init__(self, seed: int, machine_id: str, cls_cfg: Mapping | None, mtbf_scale: float, enabled: bool,
                 deterministic: bool = False):
        self.seed = int(seed)
        self.machine_id = machine_id
        self.enabled = bool(enabled and cls_cfg is not None)
        self.cfg = cls_cfg or {}
        self.mtbf_scale = float(mtbf_scale)
        self.deterministic = bool(deterministic)
        self.k = 0
        self.draws_log: list[tuple[int, float, float]] = []  # (k, ttf, repair) — V16
        self._left = math.inf
        self._repair = 0.0
        if self.enabled:
            self._load(1)

    def _load(self, k: int) -> None:
        self.k = k
        mtbf = float(self.cfg["mtbf_proc_s"]) * self.mtbf_scale
        mttr = float(self.cfg["mttr_s"])
        if self.deterministic:
            ttf, rep = mtbf, mttr
        else:
            e, z = failure_draw(self.seed, self.machine_id, k)
            ttf = e * mtbf
            rep = lognormal_from_z(mttr, float(self.cfg["repair_cv"]), z)
        self._left = ttf
        self._repair = rep
        self.draws_log.append((k, ttf, rep))

    def ttf_left(self) -> float:
        return self._left

    def consume(self, dt: float) -> None:
        if self.enabled:
            self._left -= dt

    def next_repair(self) -> float:
        """현재(k번째) 고장의 수리시간을 반환하고 k+1번째 고장 간격을 불러온다."""
        rep = self._repair
        self._load(self.k + 1)
        return rep


def process_with_failures(env: simpy.Environment, m: Machine, clock: FailureClock, duration: float, logger,
                          *, on_down: Callable[[], None] | None = None, on_up: Callable[[], None] | None = None,
                          job=None):
    """가공 duration을 진행한다. 남은 고장 간격 ≤ 남은 가공이면 그 시점에 DOWN → 수리 → 재개.

    고장이 비활성이면 v1과 같이 ``env.timeout(duration)``을 정확히 한 번 yield한다.
    반환값: 이 가공 중 DOWN 시간 합.
    """
    remaining = float(duration)
    down = 0.0
    if not clock.enabled:
        yield env.timeout(remaining)
        return down
    while clock.ttf_left() <= remaining:
        ttf = clock.ttf_left()
        if ttf > 0:
            yield env.timeout(ttf)
        clock.consume(ttf)
        remaining -= ttf
        repair = clock.next_repair()
        m.set_state(MachineState.DOWN)
        logger.log(env.now, EventType.FAIL_START, job=job, resource_id=m.id, station_id=m.station_id)
        if on_down is not None:
            on_down()
        if repair > 0:
            yield env.timeout(repair)
        down += repair
        logger.log(env.now, EventType.REPAIR_END, job=job, resource_id=m.id, station_id=m.station_id)
        m.set_state(MachineState.PROCESSING)
        if on_up is not None:
            on_up()
    yield env.timeout(remaining)
    clock.consume(remaining)
    return down
