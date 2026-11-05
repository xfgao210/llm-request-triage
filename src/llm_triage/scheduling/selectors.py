"""Deterministic candidate-selection rules."""

from __future__ import annotations

from dataclasses import dataclass

from .base import CandidateView, SchedulerSnapshot


def _require_candidates(snapshot: SchedulerSnapshot) -> tuple[CandidateView, ...]:
    if not snapshot.candidates:
        raise ValueError("Cannot select from an empty candidate set")
    return snapshot.candidates


@dataclass(frozen=True)
class FCFSSelector:
    def select(self, snapshot: SchedulerSnapshot) -> CandidateView:
        return min(
            _require_candidates(snapshot),
            key=lambda candidate: candidate.arrival_time_s,
        )


@dataclass(frozen=True)
class ShortestEstimatedServiceSelector:
    def select(self, snapshot: SchedulerSnapshot) -> CandidateView:
        return min(
            _require_candidates(snapshot),
            key=lambda candidate: (
                candidate.estimated_service_s,
                candidate.arrival_time_s,
                candidate.request_id,
            ),
        )


@dataclass(frozen=True)
class LowestEstimatedSlackSelector:
    def select(self, snapshot: SchedulerSnapshot) -> CandidateView:
        return min(
            _require_candidates(snapshot),
            key=lambda candidate: (
                candidate.estimated_slack_s(snapshot.now_s),
                candidate.arrival_time_s,
                candidate.request_id,
            ),
        )

