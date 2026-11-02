"""v2 이벤트 로그.

이 모듈 이름은 패키지 안의 ``logging``이다. ``from factory_sim.logging import ...``로 쓴다.
실험(summary)에서는 로그를 끄며, 비활성 로거의 ``log()``는 즉시 반환한다.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd

EVENT_COLUMNS = [
    "run_id", "event_seq", "time_s", "event_type", "order_id", "part", "sku", "group",
    "resource_id", "station_id", "buffer_id", "buffer_count", "from_state", "to_state",
    "node_from", "node_to", "detail",
]


class EventType(str, Enum):
    ORDER_ARRIVED = "ORDER_ARRIVED"
    ORDER_RELEASED = "ORDER_RELEASED"
    BUFFER_ENTER = "BUFFER_ENTER"
    BUFFER_EXIT = "BUFFER_EXIT"
    SETUP_START = "SETUP_START"
    SETUP_END = "SETUP_END"
    PROCESS_START = "PROCESS_START"
    PROCESS_END = "PROCESS_END"
    INSPECT_PASS = "INSPECT_PASS"
    INSPECT_FAIL = "INSPECT_FAIL"
    RECHECK_PASS = "RECHECK_PASS"
    RECHECK_FAIL = "RECHECK_FAIL"
    KIT_DELIVER = "KIT_DELIVER"
    KIT_COMPLETE = "KIT_COMPLETE"
    FA_TAKE = "FA_TAKE"
    AGV_ASSIGN = "AGV_ASSIGN"
    AGV_DEPART = "AGV_DEPART"
    AGV_ARRIVE = "AGV_ARRIVE"
    AGV_LOAD = "AGV_LOAD"
    AGV_UNLOAD = "AGV_UNLOAD"
    OP_ASSIGN = "OP_ASSIGN"
    OP_WALK_START = "OP_WALK_START"
    OP_ARRIVE = "OP_ARRIVE"
    OP_RELEASE = "OP_RELEASE"
    FAIL_START = "FAIL_START"
    REPAIR_END = "REPAIR_END"
    BATCH_START = "BATCH_START"
    BATCH_END = "BATCH_END"
    HOUSING_ORDER = "HOUSING_ORDER"
    HOUSING_RECEIVE = "HOUSING_RECEIVE"
    ORDER_SHIPPED = "ORDER_SHIPPED"
    ORDER_SCRAPPED = "ORDER_SCRAPPED"
    STATE_CHANGE = "STATE_CHANGE"
    SIM_END = "SIM_END"


class EventLogger:
    """전역 증가 ``event_seq``로 이벤트를 모은다. ``buffer_count``는 이벤트 직후 점유 수."""

    def __init__(self, run_id: str, enabled: bool = True):
        self.run_id = run_id
        self.enabled = bool(enabled)
        self._rows: list[tuple] = []
        self._seq = 0

    def log(self, time_s: float, event_type: EventType, *, job: Any = None, resource_id: str | None = None,
            station_id: str | None = None, buffer_id: str | None = None, buffer_count: int | None = None,
            from_state: str | None = None, to_state: str | None = None, node_from: str | None = None,
            node_to: str | None = None, detail: str | None = None, order: Any = None) -> None:
        if not self.enabled:
            return
        o = order if order is not None else (None if job is None else job.order)
        self._rows.append((
            self.run_id, self._seq, float(time_s), event_type.value,
            None if o is None else o.id,
            None if job is None else job.part,
            None if o is None else o.sku,
            None if o is None else o.group,
            resource_id, station_id, buffer_id, buffer_count, from_state, to_state, node_from, node_to, detail,
        ))
        self._seq += 1

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=EVENT_COLUMNS)
        df["order_id"] = df["order_id"].astype("Int64")
        df["buffer_count"] = df["buffer_count"].astype("Int64")
        return df
