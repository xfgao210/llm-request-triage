"""Reference scheduling callback.

FCFS lives here as the simplest end-to-end check of the callback hook.
The team's baseline schedulers (length-aware, static SLO-aware) implement
the same :class:`~llm_triage.simulation.callback.SchedulerCallback` protocol.
"""

from __future__ import annotations

from .callback import SimState
from .request import Action


class FCFSCallback:
    """First-Come, First-Served: admit the oldest request whenever it fits.

    Never defers or drops.  The engine enforces capacity, so when the batch
    or KV-cache is full the head of the queue simply waits.
    """

    def decide(self, state: SimState) -> Action:
        return Action.ADMIT
