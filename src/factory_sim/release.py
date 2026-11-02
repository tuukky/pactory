"""주문 도착과 키트 슬롯 예약 투입 (공장 전체 CONWIP).

- 주문 도착: 가동시간 기준 포아송 과정(사전 생성) 또는 explicit 목록. 백로그에 쌓인다.
- 투입(원자적, yield 없음): 백로그가 있고 빈 키트 슬롯이 있고 하우징 재고가 있으면
  슬롯 예약 → 재고 차감 → 부품 Job 생성 → 원자재/하우징 픽업 지점(또는 instant면 첫 공정)으로.
- 막히면 슬롯 해제·입고·(instant) 첫 버퍼 공간 이벤트를 ``any_of``로 기다린다(v1 투입 제어기 구조).
- 투입 순서는 도착순 FIFO(D22).
"""

from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

import numpy as np

from .config import groups, split_sku
from .entities import Order, PartJob, TimeWeighted
from .logging import EventType
from .randomness import arrival_draws


def generate_orders(cfg: Mapping, seed: int, horizon: float) -> list[Order]:
    """도착 목록(호라이즌까지). poisson은 시드에만 의존하는 같은 표본을 쓴다."""
    dem = cfg["demand"]
    due_off = float(dem["due_offset_s"])
    out: list[Order] = []
    if dem["mode"] == "explicit":
        for o in dem["explicit"]:
            g, _ = split_sku(o["sku"])
            t = float(o["arrival_s"])
            out.append(Order(int(o["id"]), str(o["sku"]), g, t, t + due_off))
        out.sort(key=lambda o: o.arrival_s)  # 안정 정렬: 같은 시각은 목록 순서
        return out
    lam = float(dem["rate_per_h"]) / 3600.0
    grp = groups(cfg)
    shares = np.cumsum([float(cfg["products"]["groups"][g]["share"]) for g in grp])
    n_sku = len(cfg["products"]["sku_factors"])
    n = int(horizon * lam * 1.5 + 50)
    while True:
        gaps, ug, us = arrival_draws(seed, n)
        t = np.cumsum(gaps / lam)
        if t[-1] > horizon:
            break
        n *= 2
    for i in range(n):
        ti = float(t[i])
        if ti > horizon:
            break
        gi = min(int(np.searchsorted(shares, ug[i], side="right")), len(grp) - 1)
        si = min(int(us[i] * n_sku), n_sku - 1)
        g = grp[gi]
        out.append(Order(i + 1, f"{g}{si + 1}", g, ti, ti + due_off))
    return out


class ReleaseController:
    def __init__(self, ctx, orders: Sequence[Order]):
        self.ctx = ctx
        self.env = ctx.env
        self.orders = list(orders)
        self.backlog: deque[Order] = deque()
        self.arrived = 0
        self.released = 0
        self._arrival_waiters: list = []
        w = ctx.window
        self.backlog_tw = TimeWeighted(w)
        self.blocked_k = TimeWeighted(w)
        self.blocked_stockout = TimeWeighted(w)
        self.instant = ctx.fleet is None
        self.part_names = list(ctx.graph.parts)

    # -- 도착 ------------------------------------------------------------------------------
    def arrivals(self):
        env = self.env
        for o in self.orders:
            if o.arrival_s > env.now:
                yield env.timeout(o.arrival_s - env.now)
            self.backlog.append(o)
            self.arrived += 1
            self.ctx.logger.log(env.now, EventType.ORDER_ARRIVED, order=o)
            self.backlog_tw.update(env.now, len(self.backlog))
            waiters, self._arrival_waiters = self._arrival_waiters, []
            for w in waiters:
                w.succeed()
            self._indicators()

    def _first_buffers(self):
        g = self.ctx.graph
        return [self.ctx.buffers[g.parts[p]["path"][0]] for p in self.part_names if g.parts[p]["path"]]

    def _can_admit(self) -> bool:
        ctx = self.ctx
        if ctx.kit.free_slots() < 1 or not ctx.inventory.available():
            return False
        if self.instant:
            return all(b.has_space() for b in self._first_buffers())
        return True

    def _indicators(self) -> None:
        now = self.env.now
        has = bool(self.backlog)
        stock = self.ctx.inventory.available()
        self.blocked_stockout.update(now, 1.0 if (has and not stock) else 0.0)
        self.blocked_k.update(now, 1.0 if (has and stock and self.ctx.kit.free_slots() < 1) else 0.0)

    # -- 투입 ------------------------------------------------------------------------------
    def run(self):
        env, ctx = self.env, self.ctx
        while True:
            if not self.backlog:
                if self.arrived >= len(self.orders):
                    return
                ev = env.event()
                self._arrival_waiters.append(ev)
                yield ev
                continue
            if self._can_admit():
                self._admit(self.backlog.popleft())
                continue
            self._indicators()
            events = [ctx.kit.wait_release()]
            if self.instant:
                events += [b.wait_space() for b in self._first_buffers()]
            if ctx.inventory.enabled:
                events.append(ctx.inventory.wait_restock())
            yield env.any_of(events)

    def _admit(self, order: Order) -> None:
        ctx, now = self.ctx, self.env.now
        ctx.kit.reserve(order)
        order.released_s = now
        self.released += 1
        ctx.wip.add(now, 1)
        self.backlog_tw.update(now, len(self.backlog))
        ctx.logger.log(now, EventType.ORDER_RELEASED, order=order, detail=str(order.slot_id))
        g = ctx.graph
        for p in self.part_names:
            if g.source_node(p) == "HW":
                ctx.inventory.consume()
            job = PartJob(order, p, location=g.source_node(p))
            target = g.first_target(p)
            if self.instant:
                if target.kind == "kit":
                    ctx.kit.deliver(job)
                else:
                    ctx.buffers[target.station_id].request_put(job, None)
            else:
                ctx.src_out[g.source_node(p)].request_put(job, None, target, target.cls)
        self._indicators()
