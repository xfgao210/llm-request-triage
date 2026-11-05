"""Shortest-estimated-service-time baseline."""

from dataclasses import dataclass, field

from .base import AdmitWhenFeasible, CompositeScheduler, Decision, SchedulerSnapshot
from .selectors import ShortestEstimatedServiceSelector


@dataclass(frozen=True)
class LengthAwareScheduler:
    """Select the shortest visible service estimate, never the true length."""

    _scheduler: CompositeScheduler = field(
        default_factory=lambda: CompositeScheduler(
            selector=ShortestEstimatedServiceSelector(),
            policy=AdmitWhenFeasible(),
        ),
        init=False,
        repr=False,
    )

    def decide(self, snapshot: SchedulerSnapshot) -> Decision | None:
        return self._scheduler.decide(snapshot)

