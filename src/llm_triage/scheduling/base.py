"""Shared, leakage-safe scheduling contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CandidateView:
    """Policy-visible request features.

    True output length is intentionally absent. The environment may retain it,
    but non-oracle schedulers must use ``estimated_output_tokens`` and
    ``estimated_service_s``.
    """

    request_id: str
    arrival_time_s: float
    input_tokens: int
    estimated_output_tokens: float
    deadline_s: float
    estimated_service_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        _finite_nonnegative(self.arrival_time_s, "arrival_time_s")
        if isinstance(self.input_tokens, bool) or not isinstance(self.input_tokens, int):
            raise ValueError("input_tokens must be an integer")
        if self.input_tokens <= 0:
            raise ValueError("input_tokens must be positive")
        _finite_positive(self.estimated_output_tokens, "estimated_output_tokens")
        deadline = _finite_number(self.deadline_s, "deadline_s")
        if deadline < self.arrival_time_s:
            raise ValueError("deadline_s must not precede arrival_time_s")
        _finite_positive(self.estimated_service_s, "estimated_service_s")

    def estimated_slack_s(self, now_s: float) -> float:
        """Return time to deadline after estimated service is deducted."""

        now = _finite_nonnegative(now_s, "now_s")
        return self.deadline_s - now - self.estimated_service_s


@dataclass(frozen=True)
class ResourceView:
    """Sanitized resource state relevant to an admission decision."""

    can_admit: bool
    active_batch_size: int = 0
    kv_cache_available_tokens: int | None = None
    prefill_utilization: float = 0.0
    decode_utilization: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.can_admit, bool):
            raise ValueError("can_admit must be a boolean")
        if isinstance(self.active_batch_size, bool) or not isinstance(
            self.active_batch_size, int
        ):
            raise ValueError("active_batch_size must be an integer")
        if self.active_batch_size < 0:
            raise ValueError("active_batch_size must be nonnegative")
        if self.kv_cache_available_tokens is not None:
            if isinstance(self.kv_cache_available_tokens, bool) or not isinstance(
                self.kv_cache_available_tokens, int
            ):
                raise ValueError("kv_cache_available_tokens must be an integer")
            if self.kv_cache_available_tokens < 0:
                raise ValueError("kv_cache_available_tokens must be nonnegative")
        _utilization(self.prefill_utilization, "prefill_utilization")
        _utilization(self.decode_utilization, "decode_utilization")


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Complete information made available for one scheduling decision."""

    now_s: float
    candidates: tuple[CandidateView, ...]
    resources: ResourceView

    def __post_init__(self) -> None:
        _finite_nonnegative(self.now_s, "now_s")
        if len({candidate.request_id for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("Candidate request IDs must be unique")
        if any(candidate.arrival_time_s > self.now_s for candidate in self.candidates):
            raise ValueError("Scheduler snapshot contains a future arrival")


class Action(str, Enum):
    ADMIT = "admit"
    DEFER = "defer"
    DROP = "drop"


@dataclass(frozen=True)
class Decision:
    request_id: str
    action: Action

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if not isinstance(self.action, Action):
            raise ValueError("action must be an Action")


@runtime_checkable
class CandidateSelector(Protocol):
    def select(self, snapshot: SchedulerSnapshot) -> CandidateView:
        """Choose one candidate from a nonempty snapshot."""


@runtime_checkable
class TriagePolicy(Protocol):
    def act(
        self, candidate: CandidateView, snapshot: SchedulerSnapshot
    ) -> Action:
        """Choose Admit, Defer, or Drop for the selected candidate."""


@runtime_checkable
class Scheduler(Protocol):
    def decide(self, snapshot: SchedulerSnapshot) -> Decision | None:
        """Return no decision only when the candidate set is empty."""


@dataclass(frozen=True)
class CompositeScheduler:
    """Compose a candidate selector and triage policy."""

    selector: CandidateSelector
    policy: TriagePolicy

    def decide(self, snapshot: SchedulerSnapshot) -> Decision | None:
        if not snapshot.candidates:
            return None
        candidate = self.selector.select(snapshot)
        if candidate not in snapshot.candidates:
            raise ValueError("Selector returned a candidate outside the snapshot")
        return Decision(candidate.request_id, self.policy.act(candidate, snapshot))


@dataclass(frozen=True)
class AdmitWhenFeasible:
    """Admit selected work when resources permit, otherwise defer it."""

    def act(self, candidate: CandidateView, snapshot: SchedulerSnapshot) -> Action:
        del candidate
        return Action.ADMIT if snapshot.resources.can_admit else Action.DEFER


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_nonnegative(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _finite_positive(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _utilization(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return number
