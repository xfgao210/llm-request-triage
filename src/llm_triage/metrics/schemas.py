"""Typed schemas used at the evaluation-engine boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class EventType(str, Enum):
    """Request lifecycle events emitted by a simulator or real backend."""

    ARRIVAL = "arrival"
    DECISION = "decision"
    SERVICE_START = "service_start"
    FIRST_TOKEN = "first_token"
    COMPLETION = "completion"
    DROP = "drop"
    EPISODE_END = "episode_end"


class RequestStatus(str, Enum):
    """Terminal request states used by policy-independent metrics."""

    COMPLETED = "completed"
    DROPPED = "dropped"
    UNFINISHED = "unfinished"


@dataclass(frozen=True)
class LifecycleEvent:
    """A timestamped fact emitted by the runtime.

    Required details:
      * ARRIVAL: ``deadline_s`` and optionally ``input_tokens``.
      * DECISION: optionally ``action`` and ``decision_overhead_s``.

    Any event may include a nonnegative ``queue_length`` diagnostic.
    """

    event_type: EventType
    timestamp_s: float
    request_id: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestRecord:
    """Final lifecycle record for one request."""

    request_id: str
    arrival_time_s: float
    deadline_s: float
    status: RequestStatus
    input_tokens: int | None = None
    decision_time_s: float | None = None
    service_start_time_s: float | None = None
    first_token_time_s: float | None = None
    completion_time_s: float | None = None

    @property
    def slo_met(self) -> bool:
        return (
            self.status is RequestStatus.COMPLETED
            and self.completion_time_s is not None
            and self.completion_time_s <= self.deadline_s
        )

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_time_s is None:
            return None
        return self.first_token_time_s - self.arrival_time_s

    @property
    def completion_latency_s(self) -> float | None:
        if self.completion_time_s is None:
            return None
        return self.completion_time_s - self.arrival_time_s

    @property
    def service_time_s(self) -> float | None:
        if self.service_start_time_s is None or self.completion_time_s is None:
            return None
        return self.completion_time_s - self.service_start_time_s


@dataclass(frozen=True)
class CollectedRun:
    """Request records plus event-level run diagnostics."""

    request_records: tuple[RequestRecord, ...]
    episode_end_s: float
    max_queue_length: int | None = None
    decision_overheads_s: tuple[float, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    """Complete result returned by the evaluation engine."""

    request_records: Sequence[RequestRecord]
    summary: Mapping[str, float | int | None]

