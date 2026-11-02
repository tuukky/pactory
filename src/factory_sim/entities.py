"""시뮬레이션 객체: 주문·부품 Job, 상태 추적, 입력·출력 버퍼, 키트 영역, 시간가중 집계."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import simpy

from .logging import EventLogger, EventType


class MachineState:
    PROCESSING = "PROCESSING"
    SETUP = "SETUP"
    STARVED = "STARVED"
    BLOCKED_DOWNSTREAM = "BLOCKED_DOWNSTREAM"
    BLOCKED_TRANSPORT = "BLOCKED_TRANSPORT"
    WAIT_OPERATOR = "WAIT_OPERATOR"
    DOWN = "DOWN"
    ALL = (PROCESSING, SETUP, STARVED, BLOCKED_DOWNSTREAM, BLOCKED_TRANSPORT, WAIT_OPERATOR, DOWN)


class OperatorState:
    IDLE = "IDLE"
    WALK = "WALK"
    SETUP = "SETUP"
    PROCESS = "PROCESS"
    WAIT_REPAIR = "WAIT_REPAIR"
    ALL = (IDLE, WALK, SETUP, PROCESS, WAIT_REPAIR)


class AgvState:
    IDLE = "IDLE"
    TRAVEL_EMPTY = "TRAVEL_EMPTY"
    LOAD = "LOAD"
    TRAVEL_LOADED = "TRAVEL_LOADED"
    UNLOAD = "UNLOAD"
    BLOCKED = "BLOCKED"  # 목적지 예약 배차(D1)라 진입하지 않음. 0초 집계를 검증한다.
    TO_CHARGE = "TO_CHARGE"
    CHARGING = "CHARGING"
    ALL = (IDLE, TRAVEL_EMPTY, LOAD, TRAVEL_LOADED, UNLOAD, BLOCKED, TO_CHARGE, CHARGING)
    BUSY = (TRAVEL_EMPTY, LOAD, TRAVEL_LOADED, UNLOAD)


class Outcome:
    SHIPPED = "shipped"
    SCRAPPED = "scrapped"
    INCOMPLETE = "incomplete"


# ---------------------------------------------------------------------------- jobs
@dataclass(slots=True)
class OperationRecord:
    station_id: str
    buffer_enter_s: float
    machine_id: str | None = None
    buffer_exit_s: float | None = None
    setup_s: float = 0.0
    process_start_s: float | None = None
    process_end_s: float | None = None
    depart_s: float | None = None
    wait_operator_s: float = 0.0
    down_s: float = 0.0


@dataclass(slots=True)
class Order:
    id: int
    sku: str
    group: str
    arrival_s: float
    due_s: float
    released_s: float | None = None
    kit_complete_s: float | None = None
    fa_start_s: float | None = None
    terminal_s: float | None = None
    outcome: str = Outcome.INCOMPLETE
    eol_visits: int = 0
    reworked: bool = False
    first_inspection: str | None = None  # "pass" | "fail"
    recheck: str | None = None  # "pass" | "fail"
    slot_id: int | None = None


@dataclass(slots=True)
class PartJob:
    """한 주문의 부품(ROTOR/STATOR/HOUSING) 또는 조립 이후 모터(MOTOR)."""

    order: Order
    part: str
    location: str = "RM"
    slot_cls: str = "general"  # 입력 버퍼에서 차지한 슬롯 종류
    is_recheck: bool = False
    queue_enter_s: float = 0.0
    visits: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    @property
    def id(self) -> int:
        return self.order.id

    @property
    def group(self) -> str:
        return self.order.group

    @property
    def current_op(self) -> OperationRecord | None:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------- time accounting
class StateTracker:
    """자원 상태구간 추적. 구간을 닫을 때 [w0, w1]와 [0, ∞) 누적을 온라인으로 더한다."""

    resource_type = "resource"

    def init_tracker(self, env: simpy.Environment, rid: str, initial: str, window: tuple[float, float],
                     logger: EventLogger, record: bool, station_id: str | None = None) -> None:
        self.env = env
        self.id = rid
        self.station_id = station_id
        self.logger = logger
        self.state = initial
        self.detail: str | None = None
        self.state_since = env.now
        self.w0, self.w1 = window
        self.win: dict[str, float] = {}
        self.full: dict[str, float] = {}
        self.win_detail: dict[tuple[str, str], float] = {}
        self.intervals: list[tuple] | None = [] if record else None

    def set_state(self, new: str, detail: str | None = None) -> None:
        if new == self.state and detail == self.detail:
            return
        now = self.env.now
        self._close(now)
        if self.logger.enabled:
            self.logger.log(now, EventType.STATE_CHANGE, resource_id=self.id, station_id=self.station_id,
                            from_state=self.state, to_state=new, detail=detail)
        self.state = new
        self.detail = detail
        self.state_since = now

    def _close(self, end: float) -> None:
        s = self.state_since
        if end <= s:
            return
        st = self.state
        self.full[st] = self.full.get(st, 0.0) + (end - s)
        a = s if s > self.w0 else self.w0
        b = end if end < self.w1 else self.w1
        if b > a:
            self.win[st] = self.win.get(st, 0.0) + (b - a)
            if self.detail is not None:
                key = (st, self.detail)
                self.win_detail[key] = self.win_detail.get(key, 0.0) + (b - a)
        if self.intervals is not None:
            self.intervals.append((st, s, end, self.detail))

    def close(self, end: float) -> None:
        self._close(end)
        self.state_since = end


class TimeWeighted:
    """계단함수 값의 [w0, w1] 적분과 최댓값(창 안에서 양의 시간 동안 유지된 값)."""

    __slots__ = ("w0", "w1", "value", "last_t", "area", "full_area", "max")

    def __init__(self, window: tuple[float, float], initial: float = 0.0, t0: float = 0.0):
        self.w0, self.w1 = window
        self.value = float(initial)
        self.last_t = float(t0)
        self.area = 0.0
        self.full_area = 0.0
        self.max = float("-inf")

    def _advance(self, now: float) -> None:
        if now > self.last_t:
            self.full_area += self.value * (now - self.last_t)
            a = self.last_t if self.last_t > self.w0 else self.w0
            b = now if now < self.w1 else self.w1
            if b > a:
                self.area += self.value * (b - a)
                if self.value > self.max:
                    self.max = self.value
            self.last_t = now

    def update(self, now: float, value: float) -> None:
        self._advance(now)
        self.value = float(value)

    def add(self, now: float, delta: float) -> None:
        self.update(now, self.value + delta)

    def finish(self, end: float) -> None:
        self._advance(end)

    def as_dict(self) -> dict[str, Any]:
        span = self.w1 - self.w0
        return {"area": self.area, "avg": self.area / span if span > 0 else None,
                "max": None if self.max == float("-inf") else self.max, "full_area": self.full_area}


# ---------------------------------------------------------------------------- machine
class Machine(StateTracker):
    resource_type = "machine"

    def __init__(self, env: simpy.Environment, machine_id: str, station_id: str, order_index: int,
                 window: tuple[float, float], logger: EventLogger, record: bool):
        self.init_tracker(env, machine_id, MachineState.STARVED, window, logger, record, station_id)
        self.order_index = order_index
        self.last_group: str | None = None
        self.job: PartJob | None = None
        self.held: list[PartJob] = []  # 오븐 챔버
        self.operator = None


Selector = Callable[[Sequence[PartJob], Machine, float], int]


# ---------------------------------------------------------------------------- input buffer
class Buffer:
    """유한 입력 버퍼, 가공 후 막힘(blocking-after-service). v1 Buffer 의미를 그대로 따른다.

    - ``items``는 대기 중인 Job만 담는다. 막힌 putter의 Job은 상류 설비가 들고 있다.
    - 빈 버퍼에 도착하면 (대기 시작, 설비 order_index)가 가장 앞선 getter에게 바로 넘긴다.
    - 슬롯이 비면 같은 호출 안에서 가장 오래된 (호환되는) putter로 채운다.
    - 슬롯 클래스: general(신규) / reserved(재검사 전용). 재검사품은 reserved를 먼저 쓰고,
      없으면 general을 쓴다. 신규품은 general만 쓴다.
    - AGV 배달은 ``reserve()``로 슬롯을 먼저 예약하고 ``deliver()``로 넣는다(막히지 않음).
    """

    def __init__(self, env: simpy.Environment, buffer_id: str, station_id: str, capacity: int | None,
                 logger: EventLogger, reserved: int = 0, on_exit: Callable[[PartJob], None] | None = None):
        self.env = env
        self.id = buffer_id
        self.station_id = station_id
        self.logger = logger
        self.reserved = int(reserved)
        self.gen_cap = None if capacity is None else int(capacity) - self.reserved
        self.items: deque[PartJob] = deque()
        self.n_gen = 0
        self.n_res = 0
        self.inbound = {"general": 0, "reserved": 0}
        self._getters: list[tuple[float, int, Machine, simpy.Event, Selector | None]] = []
        self._putters: deque[tuple[float, int, Machine | None, PartJob, str, simpy.Event]] = deque()
        self._put_seq = 0
        self._space_waiters: list[simpy.Event] = []
        self.listeners: list[Callable[[], None]] = []
        self.on_exit = on_exit
        self.max_count = 0

    def __len__(self) -> int:
        return len(self.items)

    # -- 용량 ------------------------------------------------------------------------
    def _slot_for(self, cls: str, count_inbound: bool) -> str | None:
        """cls('general' | 'recheck') 품목이 들어갈 슬롯 종류, 없으면 None."""
        if cls == "recheck":
            used = self.n_res + (self.inbound["reserved"] if count_inbound else 0)
            if used < self.reserved:
                return "reserved"
        used = self.n_gen + (self.inbound["general"] if count_inbound else 0)
        if self.gen_cap is None or used < self.gen_cap:
            return "general"
        return None

    def has_space(self, cls: str = "general") -> bool:
        return self._slot_for(cls, count_inbound=True) is not None

    def free(self, cls: str = "general") -> float:
        """예약을 뺀 남은 슬롯 수 (무제한이면 inf)."""
        out = 0.0
        if cls == "recheck":
            out += max(0, self.reserved - self.n_res - self.inbound["reserved"])
        if self.gen_cap is None:
            return float("inf")
        return out + max(0, self.gen_cap - self.n_gen - self.inbound["general"])

    def reserve(self, cls: str) -> str:
        slot = self._slot_for(cls, count_inbound=True)
        if slot is None:
            raise RuntimeError(f"{self.id}: 예약할 슬롯 없음")
        self.inbound[slot] += 1
        return slot

    def wait_space(self) -> simpy.Event:
        ev = self.env.event()
        self._space_waiters.append(ev)
        return ev

    # -- 내부 이동 (yield 없음) --------------------------------------------------------------
    def _enter(self, job: PartJob, src: Machine | None, slot: str) -> None:
        now = self.env.now
        self.items.append(job)
        job.slot_cls = slot
        if slot == "reserved":
            self.n_res += 1
        else:
            self.n_gen += 1
        job.location = self.id
        job.queue_enter_s = now
        job.history.append(OperationRecord(self.station_id, now))
        if src is not None:
            src.job = None
        n = len(self.items)
        if n > self.max_count:
            self.max_count = n
        if self.logger.enabled:
            self.logger.log(now, EventType.BUFFER_ENTER, job=job, resource_id=None if src is None else src.id,
                            station_id=self.station_id, buffer_id=self.id, buffer_count=n)

    def _exit_to(self, machine: Machine, idx: int = 0) -> PartJob:
        now = self.env.now
        if idx == 0:
            job = self.items.popleft()
        else:
            job = self.items[idx]
            del self.items[idx]
        if job.slot_cls == "reserved":
            self.n_res -= 1
        else:
            self.n_gen -= 1
        job.location = machine.id
        op = job.current_op
        op.buffer_exit_s = now
        op.machine_id = machine.id
        machine.job = job
        if self.logger.enabled:
            self.logger.log(now, EventType.BUFFER_EXIT, job=job, resource_id=machine.id,
                            station_id=self.station_id, buffer_id=self.id, buffer_count=len(self.items))
        if self.on_exit is not None:
            self.on_exit(job)
        return job

    def _handoff_or_enter(self, job: PartJob, src: Machine | None, slot: str) -> bool:
        """getter가 있으면 바로 넘기고 True."""
        if self._getters:
            idx = min(range(len(self._getters)), key=lambda i: self._getters[i][:2])
            _, _, machine, gev, _ = self._getters.pop(idx)
            self._enter(job, src, slot)
            gev.succeed(self._exit_to(machine, len(self.items) - 1))
            return True
        self._enter(job, src, slot)
        return False

    def _after_space_freed(self, freed: str) -> None:
        # 비워진 슬롯을 가장 오래된 호환 putter로 채운다 (general 슬롯은 모든 putter, reserved는 재검사만).
        for i, (_, _, src, job, cls, pev) in enumerate(self._putters):
            if freed == "general" or cls == "recheck":
                del self._putters[i]
                self._enter(job, src, freed)
                pev.succeed()
                break
        else:
            waiters, self._space_waiters = self._space_waiters, []
            for w in waiters:
                w.succeed()
        for cb in self.listeners:
            cb()

    # -- 공개 API ------------------------------------------------------------------------
    def request_put(self, job: PartJob, src: Machine | None, cls: str = "general") -> simpy.Event:
        ev = self.env.event()
        if self._getters:
            self._handoff_or_enter(job, src, "general")
            ev.succeed()
        else:
            slot = self._slot_for(cls, count_inbound=True)
            if slot is not None:
                self._enter(job, src, slot)
                ev.succeed()
            else:
                self._putters.append((self.env.now, self._put_seq, src, job, cls, ev))
                self._put_seq += 1
        return ev

    def deliver(self, job: PartJob, slot: str) -> None:
        """AGV가 예약 슬롯으로 배달 (막히지 않음)."""
        self.inbound[slot] -= 1
        self._handoff_or_enter(job, None, slot)

    def request_get(self, machine: Machine, selector: Selector | None = None) -> simpy.Event:
        ev = self.env.event()
        if self.items:
            idx = 0 if (selector is None or len(self.items) == 1) else selector(self.items, machine, self.env.now)
            freed = self.items[idx].slot_cls
            ev.succeed(self._exit_to(machine, idx))
            self._after_space_freed(freed)
        else:
            self._getters.append((self.env.now, machine.order_index, machine, ev, selector))
        return ev

    def cancel_get(self, ev: simpy.Event) -> None:
        self._getters = [g for g in self._getters if g[3] is not ev]


# ---------------------------------------------------------------------------- output buffer
class OutEntry:
    __slots__ = ("job", "target", "cls", "enter_s", "claimed", "slot")

    def __init__(self, job: PartJob, target: Any, cls: str, enter_s: float):
        self.job = job
        self.target = target
        self.cls = cls
        self.enter_s = enter_s
        self.claimed = False
        self.slot: str | None = None


class OutputBuffer:
    """AGV 픽업 대기 버퍼. 가득 차면 상류 설비가 막힌다(putter FIFO)."""

    def __init__(self, env: simpy.Environment, buffer_id: str, station_id: str | None, node: str,
                 capacity: int | None, logger: EventLogger):
        self.env = env
        self.id = buffer_id
        self.station_id = station_id
        self.node = node
        self.capacity = capacity
        self.logger = logger
        self.entries: list[OutEntry] = []
        self._putters: deque[tuple[Machine, OutEntry, simpy.Event]] = deque()
        self.on_enter: Callable[[], None] | None = None
        self.max_count = 0

    def __len__(self) -> int:
        return len(self.entries)

    def has_blocked(self) -> bool:
        return bool(self._putters)

    def blocked_machines(self) -> list[Machine]:
        return [m for m, _, _ in self._putters if m is not None]

    def _enter(self, entry: OutEntry, src: Machine | None) -> None:
        now = self.env.now
        entry.enter_s = now
        self.entries.append(entry)
        entry.job.location = self.id
        if src is not None:
            src.job = None
        n = len(self.entries)
        if n > self.max_count:
            self.max_count = n
        if self.logger.enabled:
            self.logger.log(now, EventType.BUFFER_ENTER, job=entry.job,
                            resource_id=None if src is None else src.id, station_id=self.station_id,
                            buffer_id=self.id, buffer_count=n)

    def request_put(self, job: PartJob, src: Machine | None, target: Any, cls: str) -> simpy.Event:
        ev = self.env.event()
        entry = OutEntry(job, target, cls, self.env.now)
        if self.capacity is None or len(self.entries) < self.capacity:
            self._enter(entry, src)
            ev.succeed()
            if self.on_enter is not None:
                self.on_enter()
        else:
            self._putters.append((src, entry, ev))
        return ev

    def remove(self, entries: list[OutEntry]) -> None:
        """AGV 상차: 품목을 꺼내고 빈 슬롯을 막힌 putter로 채운다."""
        ids = {id(e) for e in entries}
        self.entries = [e for e in self.entries if id(e) not in ids]
        if self.logger.enabled:
            for e in entries:
                self.logger.log(self.env.now, EventType.BUFFER_EXIT, job=e.job, station_id=self.station_id,
                                buffer_id=self.id, buffer_count=len(self.entries))
        added = False
        while self._putters and (self.capacity is None or len(self.entries) < self.capacity):
            src, entry, pev = self._putters.popleft()
            self._enter(entry, src)
            pev.succeed()
            added = True
        if added and self.on_enter is not None:
            self.on_enter()


# ---------------------------------------------------------------------------- kit area
class KitArea:
    """주문별 키트 슬롯 K개. 부품은 자기 주문 슬롯으로만 배달된다."""

    def __init__(self, env: simpy.Environment, k: int, parts: Sequence[str], window: tuple[float, float],
                 logger: EventLogger, on_complete: Callable[[Order, list[PartJob]], None]):
        self.env = env
        self.k = int(k)
        self.parts = tuple(parts)
        self.logger = logger
        self.on_complete = on_complete
        self._free = list(range(self.k - 1, -1, -1))  # pop()이 가장 작은 번호
        self.slots: dict[int, tuple[Order, list[PartJob]]] = {}
        self._waiters: list[simpy.Event] = []
        self.occupied = TimeWeighted(window)
        self.max_occupied = 0

    def free_slots(self) -> int:
        return len(self._free)

    def reserve(self, order: Order) -> int:
        slot = self._free.pop()
        self.slots[slot] = (order, [])
        order.slot_id = slot
        n = self.k - len(self._free)
        self.occupied.update(self.env.now, n)
        self.max_occupied = max(self.max_occupied, n)
        return slot

    def deliver(self, job: PartJob) -> None:
        order, have = self.slots[job.order.slot_id]
        if order is not job.order:
            raise RuntimeError(f"키트 슬롯 불일치: 주문 {job.order.id}")
        have.append(job)
        job.location = f"KIT:{order.slot_id}"
        self.logger.log(self.env.now, EventType.KIT_DELIVER, job=job, detail=str(order.slot_id))
        if len(have) == len(self.parts):
            order.kit_complete_s = self.env.now
            self.logger.log(self.env.now, EventType.KIT_COMPLETE, order=order, detail=str(order.slot_id))
            self.on_complete(order, have)

    def release(self, order: Order) -> None:
        slot = order.slot_id
        del self.slots[slot]
        self._free.append(slot)
        self._free.sort(reverse=True)
        self.occupied.update(self.env.now, self.k - len(self._free))
        waiters, self._waiters = self._waiters, []
        for w in waiters:
            w.succeed()

    def wait_release(self) -> simpy.Event:
        ev = self.env.event()
        self._waiters.append(ev)
        return ev
