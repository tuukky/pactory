"""Single release controller: minimum interval, global WIP limit and first-buffer space."""

from __future__ import annotations

from typing import Sequence

import simpy

from .entities import Buffer, Job, WipTracker
from .logging import EventLogger, EventType


class ReleaseController:
    """Admits jobs in sequence order.

    A job is admitted atomically (no yield in between) only when
    (1) at least ``min_interval_s`` has passed since the last actual release,
    (2) a WIP slot is free and (3) the first buffer has space. Waiting jobs stay
    in the external order pool. There is no catch-up batch release.
    """

    def __init__(self, env: simpy.Environment, sequence: Sequence[Job], first_buffer: Buffer,
                 wip: WipTracker, min_interval_s: float, wip_limit: int | None, logger: EventLogger):
        self.env = env
        self.sequence = list(sequence)
        self.first_buffer = first_buffer
        self.wip = wip
        self.min_interval_s = float(min_interval_s)
        self.wip_limit = wip_limit
        self.logger = logger
        self.released = 0
        self.last_release: float | None = None

    def _can_admit(self) -> bool:
        wip_ok = self.wip_limit is None or self.wip.count < self.wip_limit
        return wip_ok and self.first_buffer.has_space()

    def run(self):
        env = self.env
        for job in self.sequence:
            while True:
                if self.last_release is not None and env.now < self.last_release + self.min_interval_s:
                    yield env.timeout(self.last_release + self.min_interval_s - env.now)
                    continue
                if self._can_admit():
                    break
                yield env.any_of([self.wip.wait_decrease(), self.first_buffer.wait_space()])
            # Atomic admission.
            self.wip.increment()
            job.admitted_at = env.now
            self.last_release = env.now
            self.released += 1
            self.logger.log(env.now, EventType.JOB_RELEASED, job=job)
            self.first_buffer.request_put(job, None)
