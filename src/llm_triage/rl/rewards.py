"""Pluggable, leakage-safe reward shaping for request triage."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping, Protocol, Set, runtime_checkable

from llm_triage.simulation import (
    Action,
    EventType,
    RequestOutcome,
    RequestState,
    Transition,
    TriageKernel,
)


@dataclass(frozen=True)
class RewardConfig:
    """Nonnegative magnitudes for signed reward components.

    Values are fixed within one run for reproducibility, but callers may pass
    a different configuration to every training or evaluation experiment.
    """

    success: float = 1.0
    late: float = 1.0
    wasted_service: float = 0.5
    drop: float = 0.2
    defer_per_second: float = 0.02
    terminal_attainment: float = 1.0
    terminal_completion: float = 0.5
    terminal_unfinished: float = 0.5
    terminal_drop_all: float = 2.0
    invalid_action_penalty: float = 0.1

    def __post_init__(self) -> None:
        for reward_field in fields(self):
            value = getattr(self, reward_field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{reward_field.name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{reward_field.name} must be finite and nonnegative"
                )


@dataclass(frozen=True)
class RewardBreakdown:
    """Signed reward components for logging and ablation analysis."""

    success: float = 0.0
    late: float = 0.0
    wasted_service: float = 0.0
    drop: float = 0.0
    defer: float = 0.0
    terminal_attainment: float = 0.0
    terminal_completion: float = 0.0
    terminal_unfinished: float = 0.0
    terminal_drop_all: float = 0.0
    invalid_action: float = 0.0

    @property
    def total(self) -> float:
        return sum(getattr(self, reward_field.name) for reward_field in fields(self))

    def as_dict(self) -> dict[str, float]:
        values = {
            reward_field.name: getattr(self, reward_field.name)
            for reward_field in fields(self)
        }
        values["total"] = self.total
        return values

    def __add__(self, other: object) -> "RewardBreakdown":
        if not isinstance(other, RewardBreakdown):
            return NotImplemented
        return RewardBreakdown(
            **{
                reward_field.name: (
                    getattr(self, reward_field.name)
                    + getattr(other, reward_field.name)
                )
                for reward_field in fields(self)
            }
        )


@runtime_checkable
class RewardModel(Protocol):
    """Environment-facing contract for scoring one simulator transition."""

    def reward(self, transition: Transition) -> float:
        """Consume one canonical transition and return its scalar reward."""
        ...


class AsymmetricRewardModel:
    """Reward useful completions and penalize avoidable GPU waste.

    The model consumes immutable outcomes emitted by the canonical simulator.
    It never reads true output length or future workload state.  A defer action
    is charged immediately for exactly one configured retry interval, so later
    service or capacity waits cannot accumulate against the earlier action.
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        *,
        defer_retry_s: float = 0.01,
    ) -> None:
        self.config = config or RewardConfig()
        self.defer_retry_s = self._validate_defer_retry(defer_retry_s)
        self.reset()

    @staticmethod
    def _validate_defer_retry(value: float) -> float:
        retry = float(value)
        if not math.isfinite(retry) or retry <= 0:
            raise ValueError("defer_retry_s must be finite and positive")
        return retry

    def bind(
        self,
        kernel: TriageKernel,
        *,
        defer_retry_s: float | None = None,
    ) -> None:
        """Reset episode accounting and synchronize the kernel retry interval.

        ``kernel`` is accepted for the environment integration contract; the
        model deliberately obtains outcome truth only from each ``Transition``.
        """

        del kernel
        if defer_retry_s is not None:
            self.defer_retry_s = self._validate_defer_retry(defer_retry_s)
        self.reset()

    def reset(self) -> None:
        """Clear state used to calculate cumulative terminal objectives."""

        self._scored_requests: Set[str] = set()
        self._late_completions: Set[str] = set()
        self.last_breakdown = RewardBreakdown()
        self.cumulative_breakdown = RewardBreakdown()

    def reward(
        self,
        transition: Transition,
        *,
        invalid_action: bool = False,
    ) -> float:
        """Consume one transition and return its scalar reward."""

        return self.reward_breakdown(
            transition, invalid_action=invalid_action
        ).total

    def reward_breakdown(
        self,
        transition: Transition,
        *,
        invalid_action: bool = False,
    ) -> RewardBreakdown:
        """Consume one transition and return all signed reward components."""

        cfg = self.config
        breakdown = RewardBreakdown(
            invalid_action=(
                -cfg.invalid_action_penalty if invalid_action else 0.0
            )
        )

        action = Action.coerce(transition.decision.action)
        request_id = str(transition.decision.request_id)
        if action is Action.DROP:
            breakdown += RewardBreakdown(drop=-cfg.drop)
            self._scored_requests.add(request_id)
        elif action is Action.DEFER:
            breakdown += RewardBreakdown(
                defer=-cfg.defer_per_second * self.defer_retry_s
            )

        completion_ids = {
            str(event.request_id)
            for event in transition.events
            if event.event_type is EventType.COMPLETION
            and event.request_id is not None
        }

        for outcome in transition.outcomes:
            outcome_id = str(outcome.request_id)
            if (
                outcome.state is RequestState.VIOLATED
                and outcome_id in completion_ids
            ):
                self._late_completions.add(outcome_id)
            if outcome_id in self._scored_requests:
                continue
            if outcome.state is RequestState.COMPLETED:
                breakdown += RewardBreakdown(success=cfg.success)
                self._scored_requests.add(outcome_id)
            elif outcome.state is RequestState.VIOLATED:
                breakdown += self._violation_reward(outcome)
                self._scored_requests.add(outcome_id)

        if transition.done and transition.total_requests > 0:
            total = transition.total_requests
            served = transition.completed_count + len(self._late_completions)
            unfinished = max(0, total - served - transition.dropped_count)
            breakdown += RewardBreakdown(
                terminal_attainment=(
                    cfg.terminal_attainment * transition.slo_attainment
                ),
                terminal_completion=(
                    cfg.terminal_completion * served / total
                ),
                terminal_unfinished=(
                    -cfg.terminal_unfinished * unfinished / total
                ),
                terminal_drop_all=(
                    -cfg.terminal_drop_all
                    if transition.dropped_count == total
                    else 0.0
                ),
            )

        self.last_breakdown = breakdown
        self.cumulative_breakdown += breakdown
        return breakdown

    def _violation_reward(self, outcome: RequestOutcome) -> RewardBreakdown:
        slo_window = max(1e-12, outcome.deadline_s - outcome.arrival_time_s)
        return RewardBreakdown(
            late=-self.config.late,
            wasted_service=(
                -self.config.wasted_service
                * outcome.gpu_time_s
                / slo_window
            ),
        )

    def describe(self) -> Mapping[str, float]:
        """Return resolved weights for experiment provenance."""

        return {
            reward_field.name: float(getattr(self.config, reward_field.name))
            for reward_field in fields(self.config)
        }


__all__ = [
    "AsymmetricRewardModel",
    "RewardBreakdown",
    "RewardConfig",
    "RewardModel",
]
