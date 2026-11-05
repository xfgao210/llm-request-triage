"""Leakage-safe inputs to transition and terminal reward calculations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class RewardEvent(str, Enum):
    """Observable outcome that has completed before reward calculation."""

    ON_TIME_COMPLETION = "on_time_completion"
    LATE_COMPLETION = "late_completion"
    DROP = "drop"
    DEFER = "defer"


@dataclass(frozen=True)
class RewardTransition:
    """Facts available after one reward-bearing transition.

    The schema deliberately excludes true output length, future arrivals, and
    future service outcomes. Late-completion waste is normalized by the
    request's SLO window so reward scale is comparable across SLO classes.
    """

    event: RewardEvent
    service_time_s: float | None = None
    request_slo_s: float | None = None
    defer_duration_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event, RewardEvent):
            raise ValueError("event must be a RewardEvent")
        if self.event is RewardEvent.LATE_COMPLETION:
            _require_nonnegative(self.service_time_s, "service_time_s")
            _require_positive(self.request_slo_s, "request_slo_s")
            _require_absent(self.defer_duration_s, "defer_duration_s")
        elif self.event is RewardEvent.DEFER:
            _require_positive(self.defer_duration_s, "defer_duration_s")
            _require_absent(self.service_time_s, "service_time_s")
            _require_absent(self.request_slo_s, "request_slo_s")
        else:
            _require_absent(self.service_time_s, "service_time_s")
            _require_absent(self.request_slo_s, "request_slo_s")
            _require_absent(self.defer_duration_s, "defer_duration_s")

    @property
    def normalized_wasted_service(self) -> float:
        if self.event is not RewardEvent.LATE_COMPLETION:
            return 0.0
        assert self.service_time_s is not None
        assert self.request_slo_s is not None
        return self.service_time_s / self.request_slo_s


@dataclass(frozen=True)
class EpisodeOutcome:
    """Aggregate terminal facts for an episode."""

    arrivals: int
    completed: int
    slo_met: int
    dropped: int
    unfinished: int
    duration_s: float

    def __post_init__(self) -> None:
        for name in ("arrivals", "completed", "slo_met", "dropped", "unfinished"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        _require_positive(self.duration_s, "duration_s")
        if self.completed + self.dropped + self.unfinished != self.arrivals:
            raise ValueError("Episode terminal counts must sum to arrivals")
        if self.slo_met > self.completed:
            raise ValueError("slo_met cannot exceed completed")

    @property
    def slo_attainment(self) -> float:
        return self.slo_met / self.arrivals if self.arrivals else 0.0

    @property
    def completion_fraction(self) -> float:
        return self.completed / self.arrivals if self.arrivals else 0.0

    @property
    def unfinished_fraction(self) -> float:
        return self.unfinished / self.arrivals if self.arrivals else 0.0

    @property
    def throughput_rps(self) -> float:
        return self.completed / self.duration_s

    @property
    def goodput_rps(self) -> float:
        return self.slo_met / self.duration_s

    @property
    def dropped_all(self) -> bool:
        return self.arrivals > 0 and self.dropped == self.arrivals


def _require_absent(value: object, name: str) -> None:
    if value is not None:
        raise ValueError(f"{name} is not valid for this reward event")


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_nonnegative(value: object, name: str) -> float:
    number = _require_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _require_positive(value: object, name: str) -> float:
    number = _require_number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number

