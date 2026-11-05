"""Environment request truth and re-exports of the canonical scheduling types."""
from dataclasses import dataclass
from llm_triage.scheduling.base import (
    Action, CandidateView, Decision, ResourceView, SchedulerSnapshot,
)


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_time_s: float
    input_tokens: int
    true_output_tokens: int
    deadline_s: float
