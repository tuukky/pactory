"""바니시 경화 오븐(배치 설비).

- 오븐이 형성 중이면 입력 버퍼 품목을 바로 챔버(max_size)로 싣는다(D4). 입력 버퍼는 가동 중 대기열이다.
- 시작 조건: 챔버 가득, 또는 형성 시작(빈 챔버에 첫 품목 적재, 대기품 일괄 적재 포함) 후 max_wait_s 경과.
- 가공 process_s 고정(× time_scale). 고장 시 배치 전체가 함께 멈췄다가 재개한다.
- 완료 후 출력 버퍼로 하나씩 하역한다. 출력이 차면 막히고, 다 비워야 다음 형성을 시작한다.
- 상태: 빈 챔버 대기 STARVED, 형성 중 STARVED(detail=batch_forming).
- 같은 스테이션에 오븐이 여러 대면(2단계 설계 후보 '오븐 +1') 한 번에 한 대만 형성한다(스테이션별 형성 토큰).
  여러 대가 동시에 형성하면 도착품이 나뉘어 배치가 작아지므로, 형성 중인 오븐이 가공을 시작해야 다음 오븐이
  형성을 시작한다. 토큰 대기 중 상태는 STARVED. 1대일 때는 토큰이 항상 즉시 확보되어 동작이 바뀌지 않는다.
"""

from __future__ import annotations

from typing import Mapping

import simpy

from .entities import Buffer, Machine, MachineState
from .failures import FailureClock, process_with_failures
from .logging import EventType

BATCH_FORMING = "batch_forming"


def oven_process(ctx, m: Machine, st: Mapping, in_buf: Buffer, clock: FailureClock):
    env, log = ctx.env, ctx.logger.log
    b = st["batch"]
    cap = int(b["max_size"])
    proc = float(b["process_s"]) * float(ctx.scale.get(m.station_id, 1.0))
    max_wait = float(b["max_wait_s"])
    target = ctx.graph.next_target(m.station_id)
    token = ctx.oven_tokens.setdefault(m.station_id, simpy.Resource(env, capacity=1))
    while True:
        treq = token.request()
        if not treq.triggered:
            m.set_state(MachineState.STARVED)
            yield treq
        ev = in_buf.request_get(m)
        if ev.triggered:
            job = ev.value
        else:
            m.set_state(MachineState.STARVED)
            job = yield ev
        chamber = [job]
        m.held = chamber
        # 최대 대기는 '형성 시작'(빈 챔버에 첫 품목을 실은 시각) 기준이다(팀장 결정 C1).
        # 입력 도착 시각 기준이면 가동 중 쌓인 대기품 때문에 하역 직후 이미 만료되어 배치가 작게 굳는다.
        deadline = env.now + max_wait
        while len(chamber) < cap:
            # 이미 대기 중인 품목은 바로 싣는다(0초). 더 기다릴지는 그다음에 판단한다.
            if in_buf.items:
                chamber.append(in_buf.request_get(m).value)
                continue
            if env.now >= deadline:
                break
            m.set_state(MachineState.STARVED, BATCH_FORMING)
            ev = in_buf.request_get(m)
            yield ev | env.timeout(deadline - env.now)
            if ev.triggered:
                chamber.append(ev.value)
            else:
                in_buf.cancel_get(ev)
                break
        start = env.now
        token.release(treq)  # 가공 시작 → 다른 오븐이 형성 가능
        m.set_state(MachineState.PROCESSING)
        for j in chamber:
            j.current_op.process_start_s = start
        log(start, EventType.BATCH_START, resource_id=m.id, station_id=m.station_id, detail=str(len(chamber)))
        down = yield from process_with_failures(env, m, clock, proc, ctx.logger)
        end = env.now
        ctx.batches.append((m.id, start, end, len(chamber), down))
        log(end, EventType.BATCH_END, resource_id=m.id, station_id=m.station_id, detail=str(len(chamber)))
        for j in chamber:
            op = j.current_op
            op.process_end_s = end
            op.down_s = down
        while chamber:
            j = chamber.pop(0)
            m.job = j
            yield from ctx.send(m, j, target)
            j.current_op.depart_s = env.now
        m.job = None
        m.held = []
