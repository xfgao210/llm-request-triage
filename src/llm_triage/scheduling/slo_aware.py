"""Static SLO-aware scheduling baseline."""

from dataclasses import dataclass, field

from .base import (
    Action,
    CandidateView,
    CompositeScheduler,
    Decision,
    SchedulerSnapshot,
)
from .selectors import LowestEstimatedSlackSelector


@dataclass(frozen=True)
class _StaticSLORule:
    """Drop estimated-infeasible work, otherwise admit when feasible."""

    def act(self, candidate: CandidateView, snapshot: SchedulerSnapshot) -> Action:
        if candidate.estimated_slack_s(snapshot.now_s) < 0:
            return Action.DROP
        return Action.ADMIT if snapshot.resources.can_admit else Action.DEFER


@dataclass(frozen=True)
class StaticSLOAwareScheduler:
    _scheduler: CompositeScheduler = field(
        default_factory=lambda: CompositeScheduler(
            selector=LowestEstimatedSlackSelector(), policy=_StaticSLORule()
        ),
        init=False,
        repr=False,
    )

    def decide(self, snapshot: SchedulerSnapshot) -> Decision | None:
        return self._scheduler.decide(snapshot)

