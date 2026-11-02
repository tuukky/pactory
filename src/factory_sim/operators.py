"""공유 작업자 풀: 숙련 매트릭스, 최근접 유휴 작업자 배정, 도보 이동.

- 요청 큐는 (요청 시각, 설비 order_index) 순 FIFO.
- 배정: 큐 앞에서부터, 그 공정 숙련이 있는 IDLE 작업자 중 가장 가까운 사람(동률이면 ID 순).
- 도보 중 작업자는 WALK, 설비는 WAIT_OPERATOR. 이동 0초면 yield하지 않는다.
- 작업자는 한 번에 한 설비만 맡는다. 설비가 Job을 확보한 뒤에만 요청하므로 순환 대기가 없다.
- ``mode: unlimited``(V11)는 즉시 None을 돌려준다(작업자 계층 무력화).
"""

from __future__ import annotations

from typing import Mapping

import simpy

from .entities import Machine, OperatorState, StateTracker
from .layout import Layout
from .logging import EventLogger, EventType


class Operator(StateTracker):
    resource_type = "operator"

    def __init__(self, env, op_id: str, home: str, skills, window, logger, record):
        self.init_tracker(env, op_id, OperatorState.IDLE, window, logger, record)
        self.location = home
        self.skills = frozenset(skills)
        self.machine: Machine | None = None


class OperatorPool:
    def __init__(self, env: simpy.Environment, cfg: Mapping, layout: Layout, logger: EventLogger,
                 window: tuple[float, float], record: bool):
        self.env = env
        self.logger = logger
        self.layout = layout
        oc = cfg["operators"]
        self.unlimited = oc["mode"] == "unlimited"
        self.speed = float(oc.get("walk_speed_mps") or 1.0)
        self.operators: list[Operator] = []
        if not self.unlimited:
            for op_id in sorted(oc["skills"]):
                s = oc["skills"][op_id]
                self.operators.append(Operator(env, op_id, s["home"], s["skills"], window, logger, record))
        self._queue: list[tuple[float, int, Machine, simpy.Event]] = []
        self._matching = False
        self.assignments = 0

    def request(self, machine: Machine) -> simpy.Event:
        ev = self.env.event()
        if self.unlimited:
            ev.succeed(None)
            return ev
        self._queue.append((self.env.now, machine.order_index, machine, ev))
        self._queue.sort(key=lambda q: (q[0], q[1]))
        self._match()
        return ev

    def _match(self) -> None:
        if self._matching:
            return
        self._matching = True
        try:
            i = 0
            while i < len(self._queue):
                _, _, m, ev = self._queue[i]
                node = m.id
                best = None
                for op in self.operators:
                    if op.state != OperatorState.IDLE or m.station_id not in op.skills:
                        continue
                    key = (self.layout.distance(op.location, node), op.id)
                    if best is None or key < best[0]:
                        best = (key, op)
                if best is None:
                    i += 1
                    continue
                del self._queue[i]
                self._assign(best[1], m, ev)
        finally:
            self._matching = False

    def _assign(self, op: Operator, m: Machine, ev: simpy.Event) -> None:
        now = self.env.now
        op.machine = m
        m.operator = op
        self.assignments += 1
        tt = self.layout.travel_time(op.location, m.id, self.speed)
        self.logger.log(now, EventType.OP_ASSIGN, resource_id=op.id, station_id=m.station_id, detail=m.id)
        if tt <= 0:
            op.location = m.id
            op.set_state(OperatorState.SETUP)  # 곧 machine이 SETUP/PROCESS로 바꾼다
            ev.succeed(op)
            return
        op.set_state(OperatorState.WALK)
        self.logger.log(now, EventType.OP_WALK_START, resource_id=op.id, node_from=op.location, node_to=m.id)
        self.env.process(self._walk(op, m, ev, tt))

    def _walk(self, op: Operator, m: Machine, ev: simpy.Event, tt: float):
        yield self.env.timeout(tt)
        op.location = m.id
        self.logger.log(self.env.now, EventType.OP_ARRIVE, resource_id=op.id, node_to=m.id)
        op.set_state(OperatorState.SETUP)
        ev.succeed(op)

    def set_state(self, op: Operator | None, state: str) -> None:
        if op is not None:
            op.set_state(state)

    def release(self, op: Operator | None) -> None:
        if op is None:
            return
        m = op.machine
        op.machine = None
        if m is not None:
            m.operator = None
        op.set_state(OperatorState.IDLE)
        self.logger.log(self.env.now, EventType.OP_RELEASE, resource_id=op.id,
                        station_id=None if m is None else m.station_id)
        self._match()
