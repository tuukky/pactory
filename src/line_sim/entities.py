"""Core simulation objects: jobs, buffers, machines, WIP tracker and the input plan."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import simpy

from .config import product_types
from .logging import EventLogger, EventType
from .randomness import JobDraws, draw_job, order_rng


class MachineState(str, Enum):
    STARVED = "STARVED"
    SETUP = "SETUP"
    PROCESSING = "PROCESSING"
    BLOCKED = "BLOCKED"
    DOWN = "DOWN"  # TODO: optional breakdown feature (PLAN §15) not implemented; never entered.


class Outcome(str, Enum):
    SHIPPED = "shipped"
    SCRAPPED = "scrapped"
    INCOMPLETE = "incomplete"


ORDER_POOL = "ORDER_POOL"


@dataclass
class OperationRecord:
    station_id: str
    buffer_id: str
    buffer_enter_s: float
    machine_id: str | None = None
    buffer_exit_s: float | None = None
    setup_start_s: float | None = None
    setup_end_s: float | None = None
    process_start_s: float | None = None
    process_end_s: float | None = None
    depart_s: float | None = None


@dataclass(frozen=True)
class JobSpec:
    """Immutable order line fixed before any scenario is run."""

    id: int
    product_type: str
    order_ready_at: float
    draws: JobDraws


@dataclass
class Job:
    id: int
    product_type: str
    order_ready_at: float
    draws: JobDraws
    release_seq: int
    admitted_at: float | None = None
    terminal_at: float | None = None
    outcome: Outcome = Outcome.INCOMPLETE
    first_inspection: str | None = None  # "pass" | "fail"
    reworked: bool = False
    rework_result: str | None = None  # "recovered" | "failed"
    location: str = ORDER_POOL
    operation_history: list[OperationRecord] = field(default_factory=list)

    @property
    def current_op(self) -> OperationRecord | None:
        return self.operation_history[-1] if self.operation_history else None


@dataclass(frozen=True)
class InputPlan:
    jobs: tuple[JobSpec, ...]  # canonical order: job_id 1..N
    sequence: tuple[int, ...]  # release order as job ids


def build_input_plan(cfg: Mapping, seed: int) -> InputPlan:
    """Fix job ids in canonical product order, then derive the release sequence.

    Job ids are never reassigned by the sequence policy, so draws keyed by job id
    are identical across sequencing scenarios.
    """
    stations = cfg["line"]["stations"]
    ready = float(cfg["orders"].get("order_ready_at_s", 0))
    specs: list[JobSpec] = []
    jid = 1
    for p in product_types(cfg):
        for _ in range(cfg["orders"]["counts"][p]):
            specs.append(JobSpec(jid, p, ready, draw_job(seed, jid, stations)))
            jid += 1
    perm = [int(i) for i in order_rng(seed).permutation(len(specs))]
    seq_cfg = cfg["release"]["sequence"]
    if seq_cfg["policy"] == "grouped":
        rank = {p: i for i, p in enumerate(seq_cfg["order"])}
        perm = sorted(perm, key=lambda i: rank[specs[i].product_type])  # stable sort
    return InputPlan(jobs=tuple(specs), sequence=tuple(specs[i].id for i in perm))


class WipTracker:
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.count = 0
        self._waiters: list[simpy.Event] = []

    def increment(self) -> None:
        self.count += 1

    def decrement(self) -> None:
        self.count -= 1
        waiters, self._waiters = self._waiters, []
        for ev in waiters:
            ev.succeed()

    def wait_decrease(self) -> simpy.Event:
        ev = self.env.event()
        self._waiters.append(ev)
        return ev


class Machine:
    def __init__(self, env: simpy.Environment, machine_id: str, station_id: str, order_index: int,
                 logger: EventLogger):
        self.env = env
        self.id = machine_id
        self.station_id = station_id
        self.order_index = order_index
        self.logger = logger
        self.state = MachineState.STARVED
        self.state_since = env.now
        self.last_product: str | None = None
        self.job: Job | None = None
        self.intervals: list[tuple[str, float, float, int | None, str | None]] = []
        self._state_job: tuple[int | None, str | None] = (None, None)

    def set_state(self, new: MachineState, now: float) -> None:
        """Change state; a new interval also starts when the held job changes within the same state."""
        job_key = (None, None) if self.job is None else (self.job.id, self.job.product_type)
        if new == self.state and job_key == self._state_job:
            return
        self._close_interval(now)
        self.logger.log(now, EventType.STATE_CHANGE, job=self.job, machine_id=self.id,
                        station_id=self.station_id, from_state=self.state.value, to_state=new.value)
        self.state = new
        self.state_since = now
        self._state_job = (None, None) if self.job is None else (self.job.id, self.job.product_type)

    def _close_interval(self, end: float) -> None:
        if end > self.state_since:
            jid, prod = self._state_job
            self.intervals.append((self.state.value, self.state_since, end, jid, prod))

    def close(self, end: float) -> None:
        """Close the last state interval at ``end`` (may be later than env.now)."""
        self._close_interval(end)
        self.state_since = end


class Buffer:
    """Finite FIFO input buffer with blocking-after-service (not simpy.Store).

    ``items`` holds only waiting jobs (len <= capacity). A job whose put is
    pending stays held by its source machine and is not counted here.
    Waiting machines are served by (wait start time, machine order index).
    """

    def __init__(self, env: simpy.Environment, buffer_id: str, station_id: str, capacity: int | None,
                 logger: EventLogger):
        self.env = env
        self.id = buffer_id
        self.station_id = station_id
        self.capacity = capacity
        self.logger = logger
        self.items: deque[Job] = deque()
        self._getters: list[tuple[float, int, Machine, simpy.Event]] = []
        self._putters: deque[tuple[float, int, Machine | None, Job, simpy.Event]] = deque()
        self._put_seq = 0
        self._space_waiters: list[simpy.Event] = []

    def __len__(self) -> int:
        return len(self.items)

    def has_space(self) -> bool:
        return self.capacity is None or len(self.items) < self.capacity

    def wait_space(self) -> simpy.Event:
        ev = self.env.event()
        self._space_waiters.append(ev)
        return ev

    # -- internal transfers (no yields) -------------------------------------------------
    def _enter(self, job: Job, src: Machine | None) -> None:
        now = self.env.now
        self.items.append(job)
        job.location = self.id
        job.operation_history.append(OperationRecord(self.station_id, self.id, now))
        if src is not None:
            src.job = None
        self.logger.log(now, EventType.BUFFER_ENTER, job=job, machine_id=None if src is None else src.id,
                        station_id=self.station_id, buffer_id=self.id, buffer_count=len(self.items))

    def _exit_to(self, machine: Machine) -> Job:
        now = self.env.now
        job = self.items.popleft()
        job.location = machine.id
        op = job.current_op
        op.buffer_exit_s = now
        op.machine_id = machine.id
        machine.job = job
        self.logger.log(now, EventType.BUFFER_EXIT, job=job, machine_id=machine.id,
                        station_id=self.station_id, buffer_id=self.id, buffer_count=len(self.items))
        return job

    # -- public API ---------------------------------------------------------------------
    def request_put(self, job: Job, src: Machine | None) -> simpy.Event:
        ev = self.env.event()
        if self._getters:
            # Buffer is empty: hand the job straight to the highest-priority waiting machine.
            idx = min(range(len(self._getters)), key=lambda i: self._getters[i][:2])
            _, _, machine, gev = self._getters.pop(idx)
            self._enter(job, src)
            gev.succeed(self._exit_to(machine))
            ev.succeed()
        elif self.has_space():
            self._enter(job, src)
            ev.succeed()
        else:
            self._putters.append((self.env.now, self._put_seq, src, job, ev))
            self._put_seq += 1
        return ev

    def request_get(self, machine: Machine) -> simpy.Event:
        ev = self.env.event()
        if self.items:
            ev.succeed(self._exit_to(machine))
            if self._putters:
                # Fill the freed slot from the oldest blocked putter in the same call.
                _, _, src, job, pev = self._putters.popleft()
                self._enter(job, src)
                pev.succeed()
            else:
                waiters, self._space_waiters = self._space_waiters, []
                for w in waiters:
                    w.succeed()
        else:
            self._getters.append((self.env.now, machine.order_index, machine, ev))
        return ev
