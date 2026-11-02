"""First-come, first-served baseline."""

from dataclasses import dataclass, field

from .base import AdmitWhenFeasible, CompositeScheduler, Decision, SchedulerSnapshot
from .selectors import FCFSSelector


@dataclass(frozen=True)
class FCFSScheduler:
    _scheduler: CompositeScheduler = field(
        default_factory=lambda: CompositeScheduler(
            selector=FCFSSelector(), policy=AdmitWhenFeasible()
        ),
        init=False,
        repr=False,
    )

    def decide(self, snapshot: SchedulerSnapshot) -> Decision | None:
        return self._scheduler.decide(snapshot)

