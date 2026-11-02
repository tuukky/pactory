"""v2 이산사건 엔진 (SimPy): 서브라인 → 키트 합류 → 최종조립 → 검사·재검사 → 포장 → 출하.

자원 확보 순서는 Job 확보(=설비 점유) → 작업자 요청으로 고정한다(PLAN_v2 §7.1).
v1과 같이 수동 ``env.step()`` 루프로 호라이즌과 같은 시각의 이벤트까지 처리한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pandas as pd
import simpy

from .agv import AgvFleet
from .batch import oven_process
from .config import config_hash, horizon_s, resolve_config, split_sku, window_s
from .dispatch import make_selector
from .entities import (
    AgvState, Buffer, KitArea, Machine, MachineState, OperatorState, Order, OutEntry, OutputBuffer, Outcome,
    PartJob, TimeWeighted,
)
from .failures import FailureClock, process_with_failures
from .inventory import HousingInventory
from .layout import Layout
from .logging import EventLogger, EventType
from .operators import OperatorPool
from .randomness import duration_z, fail_u, lognormal_from_z, recover_u
from .release import ReleaseController, generate_orders
from .routing import ProcessGraph, Target

STATUS_COMPLETE = "complete"  # explicit 주문이 모두 종결
STATUS_HORIZON = "complete_at_horizon"  # 호라이즌에서 절단(D23)

RESOURCE_COLUMNS = ["run_id", "resource_type", "resource_id", "station_id", "state", "window_s", "full_s"]
INTERVAL_COLUMNS = ["run_id", "resource_type", "resource_id", "station_id", "state", "start_s", "end_s",
                    "duration_s", "detail"]

DurationProvider = Callable[[int, str, str, int], float]


class InvariantError(RuntimeError):
    """보존 불변식 위반."""


class DeadlockError(InvariantError):
    """WIP가 있는데 진행 이벤트가 없음."""


@dataclass
class RunResult:
    run_id: str
    status: str
    seed: int
    config: dict
    config_hash: str
    orders: list[Order]
    resource_states: pd.DataFrame
    stats: dict
    sim_end_s: float
    intervals: pd.DataFrame | None = None
    events: pd.DataFrame | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)  # full 상세: agv_trips, batches, housing_ledger

    @property
    def window(self) -> tuple[float, float]:
        return window_s(self.config)

    def orders_frame(self) -> pd.DataFrame:
        return build_orders_frame(self.run_id, self.orders)


class _Context:
    def __init__(self, env: simpy.Environment, cfg: Mapping, seed: int, logger: EventLogger, record: bool,
                 duration_provider: DurationProvider | None):
        self.env = env
        self.oven_tokens: dict[str, simpy.Resource] = {}  # 스테이션별 오븐 형성 토큰 (batch.py)
        self.cfg = cfg
        self.seed = int(seed)
        self.logger = logger
        self.record = record
        self.window = window_s(cfg)
        self.graph = ProcessGraph.from_cfg(cfg)
        self.layout = Layout.from_cfg(cfg)
        self.scale = dict(cfg.get("time_scale") or {})
        st = cfg["stochastic"]
        self.stochastic = bool(st["enabled"])
        self.cv = float(st["cv"])
        self.sku_factors = [float(f) for f in cfg["products"]["sku_factors"]]
        self.duration_provider = duration_provider
        self.buffers: dict[str, Buffer] = {}
        self.outs: dict[str, OutputBuffer] = {}
        self.src_out: dict[str, OutputBuffer] = {}
        self.machines: list[Machine] = []
        self.clocks: dict[str, FailureClock] = {}
        self.batches: list[tuple] = []
        self.wip = TimeWeighted(self.window)
        self.shipped = 0
        self.scrapped = 0
        self.terminal_count = 0
        self.last_progress_s = 0.0
        self.last_terminal_s: float | None = None
        self.kit: KitArea | None = None
        self.kit_queue: Buffer | None = None
        self.pool: OperatorPool | None = None
        self.fleet: AgvFleet | None = None
        self.inventory: HousingInventory | None = None
        self.release: ReleaseController | None = None
        self.groups_cfg = cfg["products"]["groups"]

    # -- 처리시간 ------------------------------------------------------------------------
    def duration(self, job: PartJob, sid: str) -> float:
        visit = job.visits.get(sid, 0) + 1
        job.visits[sid] = visit
        scale = float(self.scale.get(sid, 1.0))
        if self.duration_provider is not None:
            return float(self.duration_provider(job.order.id, job.part, sid, visit)) * scale
        g, idx = split_sku(job.order.sku)
        mean = float(self.cfg["stations"][sid]["times_s"][g]) * self.sku_factors[idx]
        if not self.stochastic or self.cv == 0 or mean == 0:
            return mean * scale
        z = duration_z(self.seed, job.order.id, job.part, sid, visit)
        return lognormal_from_z(mean, self.cv, z) * scale

    # -- 종결 ----------------------------------------------------------------------------
    def terminate(self, job: PartJob, kind: str, m: Machine | None) -> None:
        now = self.env.now
        o = job.order
        if m is not None:
            m.job = None
        o.terminal_s = now
        if kind in ("ship", "dock"):
            o.outcome = Outcome.SHIPPED
            job.location = "SHIPPED"
            self.shipped += 1
            et = EventType.ORDER_SHIPPED
        else:
            o.outcome = Outcome.SCRAPPED
            job.location = "SCRAPPED"
            self.scrapped += 1
            et = EventType.ORDER_SCRAPPED
        self.terminal_count += 1
        self.last_terminal_s = now
        self.last_progress_s = now
        self.logger.log(now, et, job=job, resource_id=None if m is None else m.id,
                        station_id=None if m is None else m.station_id)
        self.wip.add(now, -1)
        if self.cfg["release"]["slot_release_at"] == "TERMINAL":
            self.kit.release(o)

    # -- 이동 ----------------------------------------------------------------------------
    def send(self, m: Machine, job: PartJob, target: Target):
        """가공을 마친 job을 target으로 보낸다(막히면 그동안 BLOCKED_*)."""
        kind = target.kind
        if kind in ("ship", "scrap"):
            self.terminate(job, kind, m)
            return
        if target.transport and self.fleet is not None:
            ob = self.outs[m.station_id]
            pe = ob.request_put(job, m, target, target.cls)
            if not pe.triggered:
                m.set_state(self.fleet.classify(ob))
                yield pe
            return
        if kind == "kit":
            m.job = None
            self.kit.deliver(job)
            return
        if kind == "dock":
            self.terminate(job, "dock", m)
            return
        pe = self.buffers[target.station_id].request_put(job, m, target.cls)
        if not pe.triggered:
            m.set_state(MachineState.BLOCKED_DOWNSTREAM)
            yield pe

    # -- AGV 배달 콜백 ---------------------------------------------------------------------
    def free_for(self, target: Target) -> float:
        if target.kind == "station":
            return self.buffers[target.station_id].free(target.cls)
        return math.inf

    def reserve_for(self, target: Target) -> str | None:
        if target.kind == "station":
            return self.buffers[target.station_id].reserve(target.cls)
        return None

    def deliver(self, entry: OutEntry) -> None:
        t = entry.target
        self.last_progress_s = self.env.now
        if t.kind == "station":
            self.buffers[t.station_id].deliver(entry.job, entry.slot)
        elif t.kind == "kit":
            self.kit.deliver(entry.job)
        else:
            self.terminate(entry.job, "dock", None)

    # -- 키트 완료 → 모터 Job ----------------------------------------------------------------
    def on_kit_complete(self, order: Order, parts: list[PartJob]) -> None:
        motor = PartJob(order, "MOTOR", location="KIT")
        self.kit_queue.request_put(motor, None)

    def on_fa_take(self, job: PartJob) -> None:
        o = job.order
        o.fa_start_s = self.env.now
        self.logger.log(self.env.now, EventType.FA_TAKE, job=job, detail=str(o.slot_id))
        if self.cfg["release"]["slot_release_at"] == "FA_START":
            self.kit.release(o)

    # -- 검사·재작업 경로 --------------------------------------------------------------------
    def route_after(self, m: Machine, job: PartJob) -> Target:
        g, sid, now = self.graph, m.station_id, self.env.now
        o = job.order
        if sid == g.inspect:
            o.eol_visits += 1
            gc = self.groups_cfg[o.group]
            if job.is_recheck:
                passed = recover_u(self.seed, o.id) < float(gc["recover_prob"])
                o.recheck = "pass" if passed else "fail"
                et = EventType.RECHECK_PASS if passed else EventType.RECHECK_FAIL
            else:
                passed = not (fail_u(self.seed, o.id) < float(gc["fail_rate"]))
                o.first_inspection = "pass" if passed else "fail"
                et = EventType.INSPECT_PASS if passed else EventType.INSPECT_FAIL
            self.logger.log(now, et, job=job, resource_id=m.id, station_id=sid)
            return g.next_target(sid, passed=passed, recheck=job.is_recheck)
        if sid == g.rework:
            o.reworked = True
            job.is_recheck = True
        return g.next_target(sid)


# ---------------------------------------------------------------------------- machine process
def _machine_process(ctx: _Context, m: Machine, st: Mapping, in_buf: Buffer, selector, clock: FailureClock):
    env, log = ctx.env, ctx.logger.log
    sid = m.station_id
    setup_s = float(st.get("setup_group_change_s", 0) or 0)
    mode = st.get("operator", "none")
    pool = ctx.pool
    while True:
        ev = in_buf.request_get(m, selector)
        if ev.triggered:
            job = ev.value
        else:
            m.set_state(MachineState.STARVED)
            job = yield ev
        rec = job.current_op
        grp = job.order.group
        setup = setup_s if (setup_s > 0 and m.last_group is not None and m.last_group != grp) else 0.0
        op = None
        if mode == "full" or (mode == "setup_only" and setup > 0):
            rq = pool.request(m)
            if rq.triggered:
                op = rq.value
            else:
                t0 = env.now
                m.set_state(MachineState.WAIT_OPERATOR)
                op = yield rq
                rec.wait_operator_s += env.now - t0
        if setup > 0:
            m.set_state(MachineState.SETUP)
            if op is not None:
                op.set_state(OperatorState.SETUP)
            log(env.now, EventType.SETUP_START, job=job, resource_id=m.id, station_id=sid)
            yield env.timeout(setup)
            rec.setup_s = setup
            log(env.now, EventType.SETUP_END, job=job, resource_id=m.id, station_id=sid)
            if mode == "setup_only":
                pool.release(op)
                op = None
        m.last_group = grp
        duration = ctx.duration(job, sid)
        m.set_state(MachineState.PROCESSING)
        if op is not None:
            op.set_state(OperatorState.PROCESS)
        rec.process_start_s = env.now
        log(env.now, EventType.PROCESS_START, job=job, resource_id=m.id, station_id=sid)
        if op is not None:
            on_down = (lambda o=op: o.set_state(OperatorState.WAIT_REPAIR))
            on_up = (lambda o=op: o.set_state(OperatorState.PROCESS))
        else:
            on_down = on_up = None
        rec.down_s = yield from process_with_failures(env, m, clock, duration, ctx.logger, on_down=on_down,
                                                      on_up=on_up, job=job)
        rec.process_end_s = env.now
        ctx.last_progress_s = env.now
        log(env.now, EventType.PROCESS_END, job=job, resource_id=m.id, station_id=sid)
        if op is not None:
            pool.release(op)
        target = ctx.route_after(m, job)
        yield from ctx.send(m, job, target)
        rec.depart_s = env.now


# ---------------------------------------------------------------------------- build / run
def _build(ctx: _Context) -> None:
    env, cfg, logger, graph = ctx.env, ctx.cfg, ctx.logger, ctx.graph
    t_max = float(cfg["dispatch"]["t_max_s"])
    stations = cfg["stations"]

    ctx.inventory = HousingInventory(env, cfg, logger, ctx.window)
    parts = list(graph.parts)
    ctx.kit = KitArea(env, cfg["release"]["kit_slots"], parts, ctx.window, logger, ctx.on_kit_complete)
    ctx.pool = OperatorPool(env, cfg, ctx.layout, logger, ctx.window, ctx.record)

    for sid, st in stations.items():
        if sid == graph.assembly:
            ctx.kit_queue = Buffer(env, "Q_KIT", sid, None, logger, on_exit=ctx.on_fa_take)
            ctx.buffers[sid] = ctx.kit_queue
        else:
            ib = st["input_buffer"]
            ctx.buffers[sid] = Buffer(env, f"Q_{sid}", sid, ib.get("capacity"), logger,
                                      reserved=int(ib.get("reserved_recheck", 0) or 0))

    if cfg["agv"]["mode"] == "fleet":
        ctx.fleet = AgvFleet(env, cfg, ctx.layout, logger, ctx.window, ctx.record, free_fn=ctx.free_for,
                             reserve_fn=ctx.reserve_for, deliver_fn=ctx.deliver)
        for src in sorted({graph.source_node(p) for p in parts}):
            ob = OutputBuffer(env, f"OUT_{src}", None, src, None, logger)
            ctx.src_out[src] = ob
            ctx.fleet.add_output(ob)
        for sid in graph.transport_out_stations():
            cap = stations[sid]["output_buffer"]["capacity"]  # validate가 필수로 검사
            ob = OutputBuffer(env, f"OUT_{sid}", sid, f"{sid}_OUT", cap, logger)
            ctx.outs[sid] = ob
            ctx.fleet.add_output(ob)
        # AGV가 배달하는 입력 버퍼는 공간이 비면 배차를 다시 본다.
        fed = {graph.parts[p]["path"][0] for p in parts if graph.parts[p]["path"]}
        if graph.assembly:
            fed |= {graph.inspect, graph.rework, graph.pass_to}
        for sid in fed:
            ctx.buffers[sid].listeners.append(ctx.fleet.notify)

    fcfg = cfg["failures"]
    classes = fcfg.get("classes") or {}
    order_index = 0
    for sid, st in stations.items():
        selector = make_selector(graph.dispatch_rule(sid), t_max)
        for k in range(1, st["machines"] + 1):
            m = Machine(env, f"{sid}_{k}", sid, order_index, ctx.window, logger, ctx.record)
            order_index += 1
            ctx.machines.append(m)
            clock = FailureClock(ctx.seed, m.id, classes.get(st.get("failure_class")), fcfg["mtbf_scale"],
                                 fcfg["enabled"], deterministic=fcfg["distribution"] == "fixed")
            ctx.clocks[m.id] = clock
            if st.get("kind", "machine") == "batch_oven":
                env.process(oven_process(ctx, m, st, ctx.buffers[sid], clock))
            else:
                env.process(_machine_process(ctx, m, st, ctx.buffers[sid], selector, clock))


def _check_invariants(ctx: _Context) -> None:
    rel, now = ctx.release, ctx.env.now
    wip = int(round(ctx.wip.value))
    if rel.arrived != len(rel.backlog) + wip + ctx.shipped + ctx.scrapped:
        raise InvariantError(f"t={now}: 물량 보존 위반 (도착 {rel.arrived} != 백로그 {len(rel.backlog)} + "
                             f"WIP {wip} + 출하 {ctx.shipped} + 폐기 {ctx.scrapped})")
    for b in ctx.buffers.values():
        if (b.gen_cap is not None and b.n_gen > b.gen_cap) or b.n_res > b.reserved:
            raise InvariantError(f"t={now}: {b.id} 용량 초과")
        if b.inbound["general"] < 0 or b.inbound["reserved"] < 0:
            raise InvariantError(f"t={now}: {b.id} 예약 음수")
    for ob in ctx.outs.values():
        if ob.capacity is not None and len(ob) > ob.capacity:
            raise InvariantError(f"t={now}: {ob.id} 용량 초과")
    occ = ctx.kit.k - ctx.kit.free_slots()
    if occ > ctx.kit.k or occ != wip and ctx.cfg["release"]["slot_release_at"] == "TERMINAL":
        raise InvariantError(f"t={now}: 키트 슬롯 {occ} (K={ctx.kit.k}, WIP {wip})")
    if ctx.inventory.enabled and ctx.inventory.on_hand < 0:
        raise InvariantError(f"t={now}: 하우징 재고 음수")


def run_simulation(config: Mapping | str, seed: int, *, run_id: str | None = None, check_invariants: bool = False,
                   detail: str = "summary", duration_provider: DurationProvider | None = None,
                   deadlock_window_s: float = 86400.0) -> RunResult:
    """1회 실행. ``detail="full"``이면 이벤트 로그·상태구간·운반/배치/재고 원장을 함께 반환한다."""
    cfg = resolve_config(config)
    if detail not in ("summary", "full"):
        raise ValueError("detail: summary | full")
    full = detail == "full"
    run_id = run_id or f"seed_{seed}"
    env = simpy.Environment()
    logger = EventLogger(run_id, enabled=full)
    ctx = _Context(env, cfg, seed, logger, full, duration_provider)
    _build(ctx)
    horizon = horizon_s(cfg)
    orders = generate_orders(cfg, seed, horizon)
    ctx.release = ReleaseController(ctx, orders)
    env.process(ctx.release.arrivals())
    env.process(ctx.release.run())

    explicit = cfg["demand"]["mode"] == "explicit"
    n_orders = len(orders)
    steps = 0
    while True:
        t = env.peek()
        if t > horizon:
            break
        env.step()
        steps += 1
        if check_invariants and env.peek() > env.now:
            _check_invariants(ctx)
        if explicit and ctx.terminal_count == n_orders:
            while env.peek() == env.now:
                env.step()
            break
        if steps % 4096 == 0 and ctx.wip.value > 0 and env.now - ctx.last_progress_s > deadlock_window_s:
            raise DeadlockError(f"t={env.now}: {deadlock_window_s}s 동안 진행 없음 (WIP {ctx.wip.value:.0f})")
    if check_invariants:
        _check_invariants(ctx)
    complete = explicit and ctx.terminal_count == n_orders
    if not complete and env.peek() == math.inf and ctx.wip.value > 0:
        raise DeadlockError(f"t={env.now}: 진행할 이벤트가 없음 (교착), WIP {ctx.wip.value:.0f}")
    w1 = ctx.window[1]
    sim_end = max(ctx.last_terminal_s or 0.0, w1) if complete else horizon
    status = STATUS_COMPLETE if complete else STATUS_HORIZON
    logger.log(sim_end, EventType.SIM_END)
    return _collect(ctx, run_id, seed, cfg, orders, status, sim_end, full)


def _trackers(ctx: _Context):
    out = [("machine", m) for m in ctx.machines]
    out += [("operator", o) for o in ctx.pool.operators]
    if ctx.fleet is not None:
        out += [("agv", a) for a in ctx.fleet.agvs]
    return out


_STATES = {"machine": MachineState.ALL, "operator": OperatorState.ALL, "agv": AgvState.ALL}


def _collect(ctx: _Context, run_id: str, seed: int, cfg: dict, orders: list[Order], status: str,
             sim_end: float, full: bool) -> RunResult:
    rows, ivs = [], []
    for rtype, r in _trackers(ctx):
        r.close(sim_end)
        for s in _STATES[rtype]:
            rows.append((run_id, rtype, r.id, r.station_id, s, r.win.get(s, 0.0), r.full.get(s, 0.0)))
        if r.intervals is not None:
            ivs += [(run_id, rtype, r.id, r.station_id, s, a, b, b - a, d) for s, a, b, d in r.intervals]
    states = pd.DataFrame(rows, columns=RESOURCE_COLUMNS)

    rel, inv, kit = ctx.release, ctx.inventory, ctx.kit
    for tw in (ctx.wip, rel.backlog_tw, rel.blocked_k, rel.blocked_stockout, kit.occupied, inv.level):
        tw.finish(sim_end)
    forming = {m.id: m.win_detail.get((MachineState.STARVED, "batch_forming"), 0.0) for m in ctx.machines
               if m.win_detail}
    stats: dict[str, Any] = {
        "sim_end_s": sim_end,
        "wip": ctx.wip.as_dict(),
        "backlog": rel.backlog_tw.as_dict(),
        "release_blocked_k": rel.blocked_k.as_dict(),
        "release_blocked_stockout": rel.blocked_stockout.as_dict(),
        "kit_occupied": kit.occupied.as_dict(),
        "kit_slots": kit.k,
        "kit_max_occupied": kit.max_occupied,
        "housing": {"enabled": inv.enabled, "initial": inv.initial, "received": inv.received,
                    "consumed": inv.consumed, "on_hand_end": inv.on_hand, "on_order_end": inv.on_order,
                    "min_on_hand": inv.min_on_hand, "level": inv.level.as_dict(),
                    "orders_placed": sum(1 for r in inv.ledger if r[1] == "order")},
        "buffer_max": {b.id: b.max_count for b in ctx.buffers.values()},
        "out_buffer_max": {ob.id: ob.max_count for ob in list(ctx.outs.values()) + list(ctx.src_out.values())},
        "batch_forming_s": forming,
        "batches": _batch_summary(ctx.batches, ctx.window),
        "agv_trips": _trip_summary(ctx.fleet, ctx.window),
        "operator_assignments": ctx.pool.assignments,
        "failure_draws": {mid: [list(d) for d in c.draws_log] for mid, c in ctx.clocks.items() if c.enabled},
        "arrived": rel.arrived, "released": rel.released, "backlog_end": len(rel.backlog),
        "shipped": ctx.shipped, "scrapped": ctx.scrapped,
    }
    res = RunResult(run_id=run_id, status=status, seed=int(seed), config=cfg, config_hash=config_hash(cfg),
                    orders=orders, resource_states=states, stats=stats, sim_end_s=float(sim_end))
    if full:
        iv = pd.DataFrame(ivs, columns=INTERVAL_COLUMNS)
        res.intervals = iv
        res.events = ctx.logger.to_frame()
        res.tables["batches"] = pd.DataFrame(ctx.batches, columns=["machine_id", "start_s", "end_s", "size",
                                                                   "down_s"])
        res.tables["housing_ledger"] = pd.DataFrame(inv.ledger, columns=["time_s", "kind", "qty", "on_hand",
                                                                         "position"])
        trips = [] if ctx.fleet is None else ctx.fleet.trips
        res.tables["agv_trips"] = pd.DataFrame(trips, columns=["agv_id", "pickup", "dest", "n_items", "t_assign",
                                                               "t_arrive_pick", "t_depart", "t_arrive_dest",
                                                               "t_done"])
    return res


def _batch_summary(batches, window) -> dict:
    w0, w1 = window
    sizes = [b[3] for b in batches if w0 <= b[1] < w1]
    return {"n": len(batches), "n_in_window": len(sizes),
            "mean_size_in_window": (sum(sizes) / len(sizes)) if sizes else None,
            "max_size": max((b[3] for b in batches), default=None)}


def _trip_summary(fleet, window) -> dict:
    if fleet is None:
        return {"n": 0, "n_in_window": 0, "mean_load_in_window": None}
    w0, w1 = window
    loads = [t[3] for t in fleet.trips if w0 <= t[8] < w1]
    return {"n": len(fleet.trips), "n_in_window": len(loads),
            "mean_load_in_window": (sum(loads) / len(loads)) if loads else None}


def build_orders_frame(run_id: str, orders: list[Order]) -> pd.DataFrame:
    rows = [(run_id, o.id, o.sku, o.group, o.arrival_s, o.due_s, o.released_s, o.kit_complete_s, o.fa_start_s,
             o.terminal_s, o.outcome, o.eol_visits, o.reworked, o.first_inspection, o.recheck)
            for o in orders]
    return pd.DataFrame(rows, columns=["run_id", "order_id", "sku", "group", "arrival_s", "due_s", "released_s",
                                       "kit_complete_s", "fa_start_s", "terminal_s", "outcome", "eol_visits",
                                       "reworked", "first_inspection", "recheck"])
