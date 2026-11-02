"""하우징 구매품 재고: 재고위치(보유 + 미착) 기준 (s, Q) 발주, 고정 리드타임 입고 (D21)."""

from __future__ import annotations

from typing import Mapping

import simpy

from .entities import TimeWeighted
from .logging import EventLogger, EventType


class HousingInventory:
    def __init__(self, env: simpy.Environment, cfg: Mapping, logger: EventLogger, window: tuple[float, float]):
        self.env = env
        self.logger = logger
        h = cfg.get("housing") or {}
        self.enabled = bool(h.get("enabled"))
        self.s = int(h.get("reorder_point", 0))
        self.q = int(h.get("order_qty", 1))
        self.lead = float(h.get("lead_time_s", 0.0))
        self.initial = int(h.get("initial_on_hand", 0))
        self.on_hand = self.initial
        self.on_order = 0
        self.received = 0
        self.consumed = 0
        self.min_on_hand = self.on_hand
        self.ledger: list[tuple[float, str, int, int, int]] = []  # (시각, 종류, 수량, 보유, 재고위치)
        self._waiters: list[simpy.Event] = []
        self.level = TimeWeighted(window, self.on_hand)

    def available(self) -> bool:
        return (not self.enabled) or self.on_hand > 0

    def position(self) -> int:
        return self.on_hand + self.on_order

    def consume(self) -> None:
        if not self.enabled:
            return
        now = self.env.now
        self.on_hand -= 1
        self.consumed += 1
        self.min_on_hand = min(self.min_on_hand, self.on_hand)
        self.level.update(now, self.on_hand)
        self.ledger.append((now, "consume", 1, self.on_hand, self.position()))
        while self.position() <= self.s:
            self.on_order += self.q
            self.ledger.append((now, "order", self.q, self.on_hand, self.position()))
            self.logger.log(now, EventType.HOUSING_ORDER, detail=str(self.q))
            self.env.process(self._receive(self.q))

    def _receive(self, qty: int):
        if self.lead > 0:
            yield self.env.timeout(self.lead)
        now = self.env.now
        self.on_order -= qty
        self.on_hand += qty
        self.received += qty
        self.level.update(now, self.on_hand)
        self.ledger.append((now, "receive", qty, self.on_hand, self.position()))
        self.logger.log(now, EventType.HOUSING_RECEIVE, detail=str(qty))
        waiters, self._waiters = self._waiters, []
        for w in waiters:
            w.succeed()

    def wait_restock(self) -> simpy.Event:
        ev = self.env.event()
        self._waiters.append(ev)
        return ev
