"""Structural interfaces for simulator integration; no backend dependency."""
from typing import Protocol
from llm_triage.types import CandidateView, Decision, SchedulerSnapshot, Action


class CandidateSelector(Protocol):
    def select(self, snapshot: SchedulerSnapshot) -> str: ...


class TriagePolicy(Protocol):
    def act(self, candidate: CandidateView, snapshot: SchedulerSnapshot) -> Action: ...


class Scheduler(Protocol):
    def decide(self, snapshot: SchedulerSnapshot) -> Decision: ...
