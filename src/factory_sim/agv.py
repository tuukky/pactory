"""AGV 물류: 목적지 예약 배차, 묶음 적재, 거리 기반 이동, 상·하차, 충전.

- 운반 요청 = (출력 버퍼, 목적지) 쌍. 미배정 품목이 있고 목적지 여유(예약 차감) ≥ 1일 때만 배차한다
  (D1: 하차 막힘·교착 방지). 요청 순서는 가장 오래된 품목의 진입 시각 FIFO.
- 배차: 픽업 지점에 가장 가까운 IDLE AGV(동률이면 ID). 적재 = min(용량, 대기 수, 목적지 여유).
- 상차 시점에 같은 목적지로 새로 들어온 품목(꺼낸 자리로 들어온 막힌 품목 포함)이 있으면 용량·여유 안에서
  더 싣는다. 기다리지는 않는다.
- 상·하차 시간은 정차 1회당(D16). 충전: 누적 가동(이동+상하차) ≥ after_busy_s면 운반을 마친 뒤
  충전 노드로 가서 duration_s 동안 충전(D15).
- 출력 버퍼에 막힌 설비는 품목이 운반 대기(배정됨 또는 목적지 여유 있음)면 BLOCKED_TRANSPORT,
  아니면 BLOCKED_DOWNSTREAM으로 분류하고 변화가 있을 때마다 다시 분류한다.
"""

from __future__ import annotations

from typing import Callable, Mapping

import simpy

from .entities import AgvState, MachineState, OutEntry, OutputBuffer, StateTracker
from .layout import Layout
from .logging import EventLogger, EventType
from .routing import Target


class Agv(StateTracker):
    resource_type = "agv"

    def __init__(self, env, agv_id: str, node: str, window, logger, record):
        self.init_tracker(env, agv_id, AgvState.IDLE, window, logger, record)
        self.location = node
        self.busy_acc = 0.0


class AgvFleet:
    def __init__(self, env: simpy.Environment, cfg: Mapping, layout: Layout, logger: EventLogger,
                 window: tuple[float, float], record: bool, *, free_fn: Callable[[Target], float],
                 reserve_fn: Callable[[Target], str | None], deliver_fn: Callable[[OutEntry], None]):
        self.env = env
        self.layout = layout
        self.logger = logger
        a = cfg["agv"]
        self.speed = float(a["speed_mps"])
        self.load_s = float(a["load_s"])
        self.unload_s = float(a["unload_s"])
        self.capacity = int(a["capacity"])
        ch = a["charge"]
        self.charge_after = float(ch["after_busy_s"])
        self.charge_s = float(ch["duration_s"])
        self.charge_node = ch["node"]
        self.agvs = [Agv(env, f"AGV{i}", a["home_node"], window, logger, record)
                     for i in range(1, int(a["count"]) + 1)]
        self.out_buffers: list[OutputBuffer] = []
        self.free_fn = free_fn
        self.reserve_fn = reserve_fn
        self.deliver_fn = deliver_fn
        self.trips: list[tuple] = []
        self._dispatching = False
        self._again = False

    def add_output(self, ob: OutputBuffer) -> None:
        self.out_buffers.append(ob)
        ob.on_enter = self.notify

    # -- 배차 --------------------------------------------------------------------------------
    def notify(self) -> None:
        """출력 버퍼 유입, 목적지 공간 변화, AGV 유휴 시 호출."""
        if self._dispatching:
            self._again = True
            return
        self._dispatching = True
        try:
            self._again = True
            while self._again:
                self._again = False
                while self._dispatch_one():
                    pass
            self._reclassify()
        finally:
            self._dispatching = False

    def _dispatch_one(self) -> bool:
        idle = [v for v in self.agvs if v.state == AgvState.IDLE]
        if not idle:
            return False
        best = None  # (oldest enter_s, buffer index, target)
        for bi, ob in enumerate(self.out_buffers):
            seen: set = set()
            for e in ob.entries:
                if e.claimed or e.target.key in seen:
                    continue
                seen.add(e.target.key)
                if self.free_fn(e.target) < 1:
                    continue
                key = (e.enter_s, bi)
                if best is None or key < best[0]:
                    best = (key, ob, e.target)
        if best is None:
            return False
        _, ob, target = best
        agv = min(idle, key=lambda v: (self.layout.distance(v.location, ob.node), v.id))
        load = self._claim(ob, target, self.capacity)
        self.logger.log(self.env.now, EventType.AGV_ASSIGN, resource_id=agv.id, node_from=ob.node,
                        node_to=target.node, detail=str(len(load)))
        agv.set_state(AgvState.TRAVEL_EMPTY)  # 0초 이동이면 같은 시각에 바로 바뀐다
        self.env.process(self._trip(agv, ob, target, load))
        return True

    def _claim(self, ob: OutputBuffer, target: Target, n_max: int) -> list[OutEntry]:
        out = []
        for e in ob.entries:
            if len(out) >= n_max:
                break
            if e.claimed or e.target.key != target.key:
                continue
            if self.free_fn(target) < 1:
                break
            e.slot = self.reserve_fn(target)
            e.claimed = True
            out.append(e)
        return out

    def _move(self, agv: Agv, dest: str, state: str):
        tt = self.layout.travel_time(agv.location, dest, self.speed)
        if tt > 0:
            agv.set_state(state)
            self.logger.log(self.env.now, EventType.AGV_DEPART, resource_id=agv.id, node_from=agv.location,
                            node_to=dest)
            yield self.env.timeout(tt)
            if state in AgvState.BUSY:
                agv.busy_acc += tt
            self.logger.log(self.env.now, EventType.AGV_ARRIVE, resource_id=agv.id, node_to=dest)
        agv.location = dest

    def _trip(self, agv: Agv, ob: OutputBuffer, target: Target, load: list[OutEntry]):
        env = self.env
        t_assign = env.now
        yield from self._move(agv, ob.node, AgvState.TRAVEL_EMPTY)
        t_pick = env.now
        # 상차: 그사이 같은 목적지로 들어온 품목을 더 싣는다. 꺼낸 자리에 막혀 있던 품목이 들어오면
        # 같은 정차에서 용량·목적지 여유 안에서 이어서 싣는다(묶음 효율). 그동안 다른 AGV 배차는 미룬다.
        load += self._claim(ob, target, self.capacity - len(load))
        held, self._dispatching = self._dispatching, True
        try:
            ob.remove(load)
            while len(load) < self.capacity:
                extra = self._claim(ob, target, self.capacity - len(load))
                if not extra:
                    break
                ob.remove(extra)
                load += extra
        finally:
            self._dispatching = held
        if self.load_s > 0:
            agv.set_state(AgvState.LOAD)
            yield env.timeout(self.load_s)
            agv.busy_acc += self.load_s
        self.logger.log(env.now, EventType.AGV_LOAD, resource_id=agv.id, node_from=ob.node, detail=str(len(load)))
        self.notify()  # 출력 버퍼 공간 변화 반영(막힘 재분류)
        t_depart = env.now
        yield from self._move(agv, target.node, AgvState.TRAVEL_LOADED)
        t_arrive = env.now
        if self.unload_s > 0:
            agv.set_state(AgvState.UNLOAD)
            yield env.timeout(self.unload_s)
            agv.busy_acc += self.unload_s
        self.logger.log(env.now, EventType.AGV_UNLOAD, resource_id=agv.id, node_to=target.node,
                        detail=str(len(load)))
        for e in load:
            self.deliver_fn(e)
        self.trips.append((agv.id, ob.id, target.node, len(load), t_assign, t_pick, t_depart, t_arrive, env.now))
        if self.charge_s > 0 and agv.busy_acc >= self.charge_after:
            yield from self._move(agv, self.charge_node, AgvState.TO_CHARGE)
            agv.set_state(AgvState.CHARGING)
            yield env.timeout(self.charge_s)
            agv.busy_acc = 0.0
        agv.set_state(AgvState.IDLE)
        self.notify()

    # -- 막힘 분류 ---------------------------------------------------------------------------
    def _reclassify(self) -> None:
        for ob in self.out_buffers:
            if not ob.has_blocked():
                continue
            transport = any(e.claimed or self.free_fn(e.target) >= 1 for e in ob.entries)
            state = MachineState.BLOCKED_TRANSPORT if transport else MachineState.BLOCKED_DOWNSTREAM
            for m in ob.blocked_machines():
                m.set_state(state)

    def classify(self, ob: OutputBuffer) -> str:
        transport = any(e.claimed or self.free_fn(e.target) >= 1 for e in ob.entries)
        return MachineState.BLOCKED_TRANSPORT if transport else MachineState.BLOCKED_DOWNSTREAM
