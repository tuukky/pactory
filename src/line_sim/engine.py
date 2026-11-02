"""Discrete-event engine (SimPy) for the multi-product line."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import simpy

from .config import TERMINAL_TARGETS, config_hash, resolve_config
from .entities import (
    Buffer, InputPlan, Job, Machine, MachineState, Outcome, WipTracker, build_input_plan,
)
from .logging import EventLogger, EventType
from .randomness import lognormal_from_z
from .release import ReleaseController

STATUS_COMPLETE = "complete"
STATUS_INCOMPLETE = "incomplete_at_limit"

INTERVAL_COLUMNS = ["run_id", "machine_id", "station_id", "state", "start_s", "end_s", "duration_s",
                    "job_id", "product_type"]


class InvariantError(RuntimeError):
    """Raised when a conservation invariant is violated or the model deadlocks."""


@dataclass
class RunResult:
    run_id: str
    status: str
    seed: int
    config: dict
    config_hash: str
    input_plan: InputPlan
    events: pd.DataFrame
    jobs: list[Job]
    machine_intervals: pd.DataFrame
    sim_end_s: float
    completion_time_s: float | None

    @property
    def window(self) -> tuple[float, float]:
        w = self.config["run"]["observation_window_s"]
        return float(w[0]), float(w[1])

    def jobs_frame(self) -> pd.DataFrame:
        return build_jobs_frame(self.run_id, self.config, self.jobs)

    def orders_frame(self) -> pd.DataFrame:
        return build_orders_frame(self.run_id, self.config, self.input_plan)


class _Context:
    def __init__(self, env: simpy.Environment, cfg: Mapping, logger: EventLogger, wip: WipTracker):
        self.env = env
        self.cfg = cfg
        self.logger = logger
        self.wip = wip
        self.buffers: dict[str, Buffer] = {}
        self.machines: list[Machine] = []
        self.terminal_count = 0
        self.shipped = 0
        self.scrapped = 0
        self.last_terminal_at: float | None = None
        stoch = cfg["stochastic"]
        self.stochastic = bool(stoch["enabled"])
        self.cv = float(stoch["cv"])
        self.scale = dict(cfg.get("time_scale") or {})


def base_duration(cfg: Mapping, station_id: str, job_product: str, z: float) -> float:
    """Sampled (or fixed) processing time before ``time_scale`` is applied."""
    mean = float(cfg["line"]["stations"][station_id]["times_s"][job_product])
    stoch = cfg["stochastic"]
    return lognormal_from_z(mean, float(stoch["cv"]), z, bool(stoch["enabled"]))


def _terminate(ctx: _Context, m: Machine, job: Job, target: str) -> None:
    now = ctx.env.now
    m.job = None
    job.terminal_at = now
    if target == "SHIP":
        job.outcome, job.location = Outcome.SHIPPED, "SHIPPED"
        ctx.shipped += 1
        etype = EventType.JOB_SHIPPED
    else:
        job.outcome, job.location = Outcome.SCRAPPED, "SCRAPPED"
        ctx.scrapped += 1
        etype = EventType.JOB_SCRAPPED
    ctx.terminal_count += 1
    ctx.last_terminal_at = now
    ctx.logger.log(now, etype, job=job, machine_id=m.id, station_id=m.station_id)
    ctx.wip.decrement()


def _route(ctx: _Context, m: Machine, job: Job, route: Mapping) -> str:
    now, log = ctx.env.now, ctx.logger.log
    rtype = route.get("type", "next")
    pc = ctx.cfg["products"][job.product_type]
    if rtype == "inspect":
        failed = job.draws.fail_u < pc["fail_rate"]
        job.first_inspection = "fail" if failed else "pass"
        log(now, EventType.INSPECT_FAIL if failed else EventType.INSPECT_PASS, job=job, machine_id=m.id,
            station_id=m.station_id)
        return route["on_fail"] if failed else route["on_pass"]
    if rtype == "rework":
        recovered = job.draws.recover_u < pc["recover_prob"]
        job.rework_result = "recovered" if recovered else "failed"
        log(now, EventType.REWORK_RECOVERED if recovered else EventType.REWORK_FAILED, job=job,
            machine_id=m.id, station_id=m.station_id)
        return route["on_pass"] if recovered else route["on_fail"]
    return route["next"]


def _machine_process(ctx: _Context, m: Machine, station_cfg: Mapping, in_buf: Buffer):
    env, log = ctx.env, ctx.logger.log
    sid = m.station_id
    matrix = station_cfg.get("setup_matrix_s")
    route = station_cfg["route"]
    scale = float(ctx.scale.get(sid, 1.0))
    while True:
        ev = in_buf.request_get(m)
        if ev.triggered:
            job = ev.value
        else:
            m.set_state(MachineState.STARVED, env.now)
            job = yield ev
        op = job.current_op
        p = job.product_type
        setup = float(matrix[m.last_product][p]) if (matrix and m.last_product is not None) else 0.0
        if route.get("type") == "rework":
            job.reworked = True
        if setup > 0:
            m.set_state(MachineState.SETUP, env.now)
            op.setup_start_s = env.now
            log(env.now, EventType.SETUP_START, job=job, machine_id=m.id, station_id=sid)
            yield env.timeout(setup)
            op.setup_end_s = env.now
            log(env.now, EventType.SETUP_END, job=job, machine_id=m.id, station_id=sid)
        m.last_product = p
        duration = base_duration(ctx.cfg, sid, p, job.draws.z[sid]) * scale
        m.set_state(MachineState.PROCESSING, env.now)
        op.process_start_s = env.now
        log(env.now, EventType.PROCESS_START, job=job, machine_id=m.id, station_id=sid)
        yield env.timeout(duration)
        op.process_end_s = env.now
        log(env.now, EventType.PROCESS_END, job=job, machine_id=m.id, station_id=sid)

        target = _route(ctx, m, job, route)
        if target in TERMINAL_TARGETS:
            op.depart_s = env.now
            _terminate(ctx, m, job, target)
            continue
        pe = ctx.buffers[target].request_put(job, m)
        if not pe.triggered:
            m.set_state(MachineState.BLOCKED, env.now)
            yield pe
        op.depart_s = env.now


def _check_invariants(ctx: _Context, controller: ReleaseController, n_orders: int) -> None:
    unreleased = n_orders - controller.released
    if n_orders != unreleased + ctx.wip.count + ctx.shipped + ctx.scrapped:
        raise InvariantError(f"t={ctx.env.now}: 물량 보존 위반 (주문 {n_orders} != 미투입 {unreleased} + "
                             f"WIP {ctx.wip.count} + 양품 {ctx.shipped} + 폐기 {ctx.scrapped})")
    in_buffers = sum(len(b) for b in ctx.buffers.values())
    held = sum(1 for m in ctx.machines if m.job is not None)
    if ctx.wip.count != in_buffers + held:
        raise InvariantError(f"t={ctx.env.now}: WIP {ctx.wip.count} != 버퍼 {in_buffers} + 설비 {held}")
    for b in ctx.buffers.values():
        if b.capacity is not None and len(b) > b.capacity:
            raise InvariantError(f"t={ctx.env.now}: {b.id} 용량 초과 ({len(b)} > {b.capacity})")


def run_simulation(config: Mapping | str, seed: int, input_plan: InputPlan | None = None, *,
                   run_id: str | None = None, check_invariants: bool = False) -> RunResult:
    """Run one replication until every order is terminal or ``run.time_limit_s`` is passed.

    A manual ``env.step()`` loop is used instead of ``env.run(until=...)`` so that
    events scheduled exactly at the time limit are processed.
    """
    cfg = resolve_config(config)
    plan = input_plan if input_plan is not None else build_input_plan(cfg, seed)
    run_id = run_id or f"seed_{seed}"
    env = simpy.Environment()
    logger = EventLogger(run_id)
    wip = WipTracker(env)
    ctx = _Context(env, cfg, logger, wip)

    seq_index = {jid: i + 1 for i, jid in enumerate(plan.sequence)}
    jobs = {s.id: Job(s.id, s.product_type, s.order_ready_at, s.draws, seq_index[s.id]) for s in plan.jobs}
    sequence = [jobs[jid] for jid in plan.sequence]

    stations = cfg["line"]["stations"]
    for sid, st in stations.items():
        b = st["input_buffer"]
        ctx.buffers[sid] = Buffer(env, b["id"], sid, b.get("capacity"), logger)
    order_index = 0
    for sid, st in stations.items():
        for k in range(1, st["machines"] + 1):
            m = Machine(env, f"{sid}_{k}", sid, order_index, logger)
            order_index += 1
            ctx.machines.append(m)
            env.process(_machine_process(ctx, m, st, ctx.buffers[sid]))

    rel = cfg["release"]
    controller = ReleaseController(env, sequence, ctx.buffers[cfg["line"]["first_station"]], wip,
                                   rel["min_interval_s"], rel.get("wip_limit"), logger)
    ready = float(cfg["orders"].get("order_ready_at_s", 0))

    def starter():
        if ready > env.now:
            yield env.timeout(ready - env.now)
        for job in sorted(jobs.values(), key=lambda j: j.id):
            logger.log(env.now, EventType.ORDER_CREATED, job=job)
        yield from controller.run()

    env.process(starter())

    n_orders = len(jobs)
    limit = float(cfg["run"]["time_limit_s"])
    while ctx.terminal_count < n_orders:
        t = env.peek()
        if t == math.inf:
            raise InvariantError(f"t={env.now}: 진행할 이벤트가 없음 (교착), 종결 {ctx.terminal_count}/{n_orders}")
        if t > limit:
            break
        env.step()
        if check_invariants and env.peek() > env.now:
            _check_invariants(ctx, controller, n_orders)
    if ctx.terminal_count == n_orders:
        # Drain remaining same-time events so machine states are final.
        while env.peek() == env.now:
            env.step()
    if check_invariants:
        _check_invariants(ctx, controller, n_orders)

    complete = ctx.terminal_count == n_orders
    status = STATUS_COMPLETE if complete else STATUS_INCOMPLETE
    completion = ctx.last_terminal_at if complete else None
    w1 = float(cfg["run"]["observation_window_s"][1])
    sim_end = max(completion if complete else limit, w1)
    for m in ctx.machines:
        m.close(sim_end)
    logger.log(sim_end, EventType.SIM_END)

    rows = []
    for m in ctx.machines:
        for state, start, end, jid, prod in m.intervals:
            rows.append((run_id, m.id, m.station_id, state, start, end, end - start, jid, prod))
    intervals = pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
    intervals["job_id"] = intervals["job_id"].astype("Int64")

    return RunResult(run_id=run_id, status=status, seed=int(seed), config=cfg, config_hash=config_hash(cfg),
                     input_plan=plan, events=logger.to_frame(), jobs=sorted(jobs.values(), key=lambda j: j.id),
                     machine_intervals=intervals, sim_end_s=float(sim_end), completion_time_s=completion)


def _setup_stations(cfg: Mapping) -> set[str]:
    return {sid for sid, st in cfg["line"]["stations"].items() if st.get("setup_matrix_s")}


def _diff(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else b - a


def build_jobs_frame(run_id: str, cfg: Mapping, jobs: list[Job]) -> pd.DataFrame:
    stations = list(cfg["line"]["stations"])
    setup_st = _setup_stations(cfg)
    w1 = float(cfg["run"]["observation_window_s"][1])
    rows: list[dict[str, Any]] = []
    for j in jobs:
        ops = {op.station_id: op for op in j.operation_history}
        row: dict[str, Any] = {
            "run_id": run_id, "job_id": j.id, "product_type": j.product_type, "release_seq": j.release_seq,
            "order_ready_at_s": j.order_ready_at, "admitted_at_s": j.admitted_at, "terminal_at_s": j.terminal_at,
            "outcome": j.outcome.value, "location_at_end": j.location, "first_inspection": j.first_inspection,
            "reworked": j.reworked, "rework_result": j.rework_result,
        }
        total_wait = total_blocked = 0.0
        for sid in stations:
            op = ops.get(sid)
            row[f"{sid}_machine"] = None if op is None else op.machine_id
            wait = None if op is None else _diff(op.buffer_enter_s, op.buffer_exit_s)
            row[f"{sid}_wait_s"] = wait
            if sid in setup_st:
                if op is None or op.process_start_s is None:
                    row[f"{sid}_setup_s"] = None
                else:
                    row[f"{sid}_setup_s"] = _diff(op.setup_start_s, op.setup_end_s) or 0.0
            row[f"{sid}_process_s"] = None if op is None else _diff(op.process_start_s, op.process_end_s)
            blocked = None if op is None else _diff(op.process_end_s, op.depart_s)
            row[f"{sid}_blocked_s"] = blocked
            total_wait += wait or 0.0
            total_blocked += blocked or 0.0
        row["total_wait_s"] = total_wait
        row["total_blocked_s"] = total_blocked
        row["line_time_s"] = _diff(j.admitted_at, j.terminal_at)
        row["order_age_s"] = _diff(j.order_ready_at, j.terminal_at)
        row["shipped_in_shift"] = bool(j.outcome == Outcome.SHIPPED and j.terminal_at is not None
                                       and j.terminal_at <= w1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_orders_frame(run_id: str, cfg: Mapping, plan: InputPlan) -> pd.DataFrame:
    """Input order list with base (unscaled) duration samples and quality draws."""
    stations = list(cfg["line"]["stations"])
    seq_index = {jid: i + 1 for i, jid in enumerate(plan.sequence)}
    rows = []
    for s in sorted(plan.jobs, key=lambda s: seq_index[s.id]):
        row = {"run_id": run_id, "release_seq": seq_index[s.id], "job_id": s.id, "product_type": s.product_type,
               "order_ready_at_s": s.order_ready_at}
        for sid in stations:
            row[f"{sid}_base_s"] = base_duration(cfg, sid, s.product_type, s.draws.z[sid])
        row["fail_u"] = s.draws.fail_u
        row["recover_u"] = s.draws.recover_u
        rows.append(row)
    return pd.DataFrame(rows)
