"""Event log for one simulation run.

NOTE: this module is named ``logging`` inside the ``line_sim`` package. Import it
as ``from line_sim import logging as event_log`` (or ``from line_sim.logging import ...``).
It does not use the standard library ``logging`` module.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd

EVENT_COLUMNS = [
    "run_id", "event_seq", "time_s", "event_type", "job_id", "product_type",
    "machine_id", "station_id", "buffer_id", "buffer_count", "from_state", "to_state",
]


class EventType(str, Enum):
    ORDER_CREATED = "ORDER_CREATED"
    JOB_RELEASED = "JOB_RELEASED"
    BUFFER_ENTER = "BUFFER_ENTER"
    BUFFER_EXIT = "BUFFER_EXIT"
    SETUP_START = "SETUP_START"
    SETUP_END = "SETUP_END"
    PROCESS_START = "PROCESS_START"
    PROCESS_END = "PROCESS_END"
    INSPECT_PASS = "INSPECT_PASS"
    INSPECT_FAIL = "INSPECT_FAIL"
    REWORK_RECOVERED = "REWORK_RECOVERED"
    REWORK_FAILED = "REWORK_FAILED"
    JOB_SHIPPED = "JOB_SHIPPED"
    JOB_SCRAPPED = "JOB_SCRAPPED"
    STATE_CHANGE = "STATE_CHANGE"
    SIM_END = "SIM_END"


class EventLogger:
    """Collects events with a globally increasing ``event_seq``.

    ``buffer_count`` is the buffer occupancy immediately after the event.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._rows: list[tuple] = []
        self._seq = 0

    def log(self, time_s: float, event_type: EventType, *, job: Any = None, machine_id: str | None = None,
            station_id: str | None = None, buffer_id: str | None = None, buffer_count: int | None = None,
            from_state: str | None = None, to_state: str | None = None) -> None:
        self._rows.append((
            self.run_id, self._seq, float(time_s), event_type.value,
            None if job is None else job.id,
            None if job is None else job.product_type,
            machine_id, station_id, buffer_id, buffer_count, from_state, to_state,
        ))
        self._seq += 1

    def __len__(self) -> int:
        return len(self._rows)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self._rows, columns=EVENT_COLUMNS)
        df["job_id"] = df["job_id"].astype("Int64")
        df["buffer_count"] = df["buffer_count"].astype("Int64")
        return df
