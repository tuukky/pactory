"""디스패칭 규칙(대기열에서 다음 Job 선택). 순수 함수 — 난수 없음.

- FIFO: (큐 진입 시각, 순번) 최소
- EDD: (납기, 큐 진입 시각, 순번) 최소
- FAMILY: 대기 ≥ T_max인 후보가 있으면 그중 FIFO(굶주림 방지). 없으면 직전 군과 같은 군 중 FIFO,
  그것도 없으면 전체 FIFO.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

from .config import DISPATCH_RULES


class Candidate(NamedTuple):
    queue_enter_s: float
    seq: int
    due_s: float
    group: str


def _fifo(idx: Sequence[int], cands: Sequence[Candidate]) -> int:
    return min(idx, key=lambda i: (cands[i].queue_enter_s, cands[i].seq))


def select(rule: str, cands: Sequence[Candidate], last_group: str | None, now: float, t_max_s: float) -> int:
    if not cands:
        raise ValueError("빈 후보 목록")
    all_idx = range(len(cands))
    if rule == "FIFO":
        return _fifo(all_idx, cands)
    if rule == "EDD":
        return min(all_idx, key=lambda i: (cands[i].due_s, cands[i].queue_enter_s, cands[i].seq))
    if rule == "FAMILY":
        overdue = [i for i in all_idx if now - cands[i].queue_enter_s >= t_max_s]
        if overdue:
            return _fifo(overdue, cands)
        same = [i for i in all_idx if cands[i].group == last_group]
        return _fifo(same if same else all_idx, cands)
    raise ValueError(f"알 수 없는 규칙 {rule!r} ({DISPATCH_RULES})")


def make_selector(rule: str, t_max_s: float):
    """Buffer.request_get용 selector. FIFO면 None(버퍼 순서 = 진입 순서)."""
    if rule == "FIFO":
        return None

    def selector(items, machine, now) -> int:
        cands = [Candidate(j.queue_enter_s, i, j.order.due_s, j.order.group) for i, j in enumerate(items)]
        return select(rule, cands, machine.last_group, now, t_max_s)

    return selector
