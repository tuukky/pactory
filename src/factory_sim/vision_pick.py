"""PLAN_v5 S3 — 비전 피킹 셀(VP) 스테이션 (``kind: vision_pick``).

HOUSING 경로 ``HW → VP → KIT``에서 하우징 1개마다:
  시도 k = 1..N:
    촬영(고정) + 인식 시간(S1 경험 분포)
    결과(시도마다 독립, S1·S2 경험 확률): 인식 거부 → 빈 흔들기(reject_extra_s) 후 재촬영 / 채택+파지 실패 → 재촬영 /
    채택+파지 성공 → 끝. 흔들기로 빈 상태가 바뀐다고 보고 시도를 독립으로 둔다 (한계: 실제로는 상관이 있을 수 있음)
    채택된 시도는 팔 동작 시간(S2 경험 분포, 성공·실패 따로)을 더한다.
  N회 모두 실패 → 작업자 수동 처리 (작업자 풀, 숙련 목록에 VP가 있는 작업자, 합성 가정 시간).
설비 고장은 v2 설비와 같은 FailureClock (가공 시간 동안만 흐름).

난수: (run seed, 주문 id, "VP", 시도 번호)로 정해지는 독립 흐름 → 같은 시드·같은 주문이면 후보 설정이 달라도
같은 균등난수를 쓴다 (공통 난수, 민감도 곡선의 짝 비교).
분포 파일 형식 (``vision_pick.dist_file``, JSON):
    {"vision_time_s": [..], "robot_time_s": {"success": [..], "fail": [..]},
     "p_reject": float, "p_fail": float, "source": "..."}   # 시도 1회 기준 확률
민감도: ``vision_pick.attempt_fail_rate``를 주면 (p_reject + p_fail) = 그 값이 되게 둘의 비율을 유지해 스케일한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .entities import MachineState, OperatorState
from .failures import process_with_failures
from .logging import EventType
from .randomness import rng

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dist(vp: Mapping[str, Any]) -> dict:
    p = Path(vp["dist_file"])
    p = p if p.is_absolute() else PROJECT_ROOT / p
    d = json.loads(p.read_text(encoding="utf-8"))
    pr, pf = float(d["p_reject"]), float(d["p_fail"])
    if vp.get("attempt_fail_rate") is not None:
        f = float(vp["attempt_fail_rate"])
        tot = pr + pf
        share = pr / tot if tot > 0 else 0.5
        pr, pf = f * share, f * (1 - share)
    if not (0 <= pr and 0 <= pf and pr + pf <= 1):
        raise ValueError(f"VP 확률 범위 오류: p_reject={pr}, p_fail={pf}")
    return {"vision": np.asarray(d["vision_time_s"], float), "robot_ok": np.asarray(d["robot_time_s"]["success"], float),
            "robot_fail": np.asarray(d["robot_time_s"]["fail"] or d["robot_time_s"]["success"], float),
            "p_reject": pr, "p_fail": pf, "source": d.get("source", "")}


def new_stats() -> dict:
    return {"housings": 0, "attempts": 0, "rejects": 0, "grasp_fails": 0, "manual": 0, "manual_s": 0.0,
            "vp_busy_s": 0.0, "in_window": {"housings": 0, "attempts": 0, "manual": 0}}


def _sample(arr: np.ndarray, g: np.random.Generator) -> float:
    return float(arr[int(g.integers(len(arr)))]) if len(arr) else 0.0


def vision_pick_process(ctx, m, st: Mapping[str, Any], in_buf, selector, clock):
    """VP 기계 1대의 프로세스 (engine._machine_process와 같은 입출력 규약)."""
    env, log = ctx.env, ctx.logger.log
    sid = m.station_id
    vp = st["vision_pick"]
    dist = ctx.vp_dists.setdefault(sid, load_dist(vp))
    stats = ctx.vp_stats.setdefault(sid, new_stats())
    capture = float(vp["capture_s"])
    n_max = int(vp["max_attempts"])
    manual_s = float(vp["manual_s"])
    w0, w1 = ctx.window
    while True:
        ev = in_buf.request_get(m, selector)
        if ev.triggered:
            job = ev.value
        else:
            m.set_state(MachineState.STARVED)
            job = yield ev
        rec = job.current_op
        t_start = env.now
        rec.process_start_s = env.now
        log(env.now, EventType.PROCESS_START, job=job, resource_id=m.id, station_id=sid)
        done = False
        attempts = 0
        busy = 0.0
        down_total = 0.0
        for k in range(1, n_max + 1):
            g = rng(ctx.seed, job.order.id, "VP", sid, k)
            u = float(g.random())
            t = capture + _sample(dist["vision"], g)
            attempts += 1
            if u < dist["p_reject"]:
                stats["rejects"] += 1
                t += float(vp.get("reject_extra_s", 0.0))   # 집을 후보 없음 → 빈 흔들기 후 재촬영 (합성 가정)
            elif u < dist["p_reject"] + dist["p_fail"]:
                stats["grasp_fails"] += 1
                t += _sample(dist["robot_fail"], g)
            else:
                t += _sample(dist["robot_ok"], g)
                done = True
            m.set_state(MachineState.PROCESSING)
            down_total += yield from process_with_failures(env, m, clock, t, ctx.logger, job=job)
            busy += t
            if done:
                break
        if not done:
            # 작업자 수동 처리: 기계는 작업자를 기다리고(WAIT_OPERATOR), 작업자가 수동으로 키트로 보낸다.
            t0 = env.now
            rq = ctx.pool.request(m)
            if rq.triggered:
                op = rq.value
            else:
                m.set_state(MachineState.WAIT_OPERATOR)
                op = yield rq
                rec.wait_operator_s += env.now - t0
            op.set_state(OperatorState.PROCESS)
            m.set_state(MachineState.PROCESSING)
            yield env.timeout(manual_s)
            ctx.pool.release(op)
            stats["manual"] += 1
            stats["manual_s"] += manual_s
            if w0 <= t_start < w1:
                stats["in_window"]["manual"] += 1
        stats["housings"] += 1
        stats["attempts"] += attempts
        stats["vp_busy_s"] += busy
        if w0 <= t_start < w1:
            stats["in_window"]["housings"] += 1
            stats["in_window"]["attempts"] += attempts
        rec.down_s = down_total
        rec.process_end_s = env.now
        ctx.last_progress_s = env.now
        log(env.now, EventType.PROCESS_END, job=job, resource_id=m.id, station_id=sid)
        target = ctx.route_after(m, job)
        yield from ctx.send(m, job, target)
        rec.depart_s = env.now
