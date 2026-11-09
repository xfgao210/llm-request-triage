"""Static SLO-aware scheduling baseline."""

from dataclasses import dataclass

from llm_triage.simulation import Action, RequestView, SimState


@dataclass(frozen=True)
class StaticSLOAwareScheduler:
    """Prioritize least slack and drop requests estimated to be infeasible."""

    def priority(self, candidate: RequestView) -> tuple[float, float, str]:
        return (
            candidate.estimated_slack_s(),
            candidate.arrival_time_s,
            str(candidate.request_id),
        )

    def decide(self, snapshot: SimState) -> Action:
        if snapshot.candidate.estimated_slack_s(snapshot.now_s) < 0:
            return Action.DROP
        return Action.ADMIT
