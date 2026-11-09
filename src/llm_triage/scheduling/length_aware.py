"""Shortest-estimated-service-time baseline."""

from dataclasses import dataclass

from llm_triage.simulation import Action, RequestView, SimState


@dataclass(frozen=True)
class LengthAwareScheduler:
    """Select the shortest visible service estimate, never the true length."""

    def priority(self, candidate: RequestView) -> tuple[float, float, str]:
        return (
            candidate.estimated_service_s,
            candidate.arrival_time_s,
            str(candidate.request_id),
        )

    def decide(self, snapshot: SimState) -> Action:
        return Action.ADMIT if snapshot.candidate_fits() else Action.DEFER
