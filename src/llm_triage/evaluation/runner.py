"""Run a policy matrix through a simulator-independent executor boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Protocol, runtime_checkable

from llm_triage.metrics import EvaluationResult, LifecycleEvent, evaluate_events
from llm_triage.scheduling import Scheduler


@dataclass(frozen=True)
class EvaluationConfig:
    measurement_window_s: float
    window_start_s: float = 0.0
    episode_end_s: float | None = None

    def __post_init__(self) -> None:
        if not _is_number(self.measurement_window_s):
            raise ValueError("measurement_window_s must be numeric")
        if not math.isfinite(self.measurement_window_s) or self.measurement_window_s <= 0:
            raise ValueError("measurement_window_s must be finite and positive")
        if not _is_number(self.window_start_s):
            raise ValueError("window_start_s must be numeric")
        if not math.isfinite(self.window_start_s) or self.window_start_s < 0:
            raise ValueError("window_start_s must be finite and nonnegative")
        if self.episode_end_s is not None:
            if not _is_number(self.episode_end_s):
                raise ValueError("episode_end_s must be numeric")
            if not math.isfinite(self.episode_end_s) or self.episode_end_s < 0:
                raise ValueError("episode_end_s must be finite and nonnegative")


@runtime_checkable
class PolicyExecutor(Protocol):
    """Adapter implemented by Vikas's simulator or another execution backend.

    Implementations must reset to the same workload and configuration for each
    call so all policies receive paired inputs.
    """

    def run(
        self, policy_name: str, scheduler: Scheduler
    ) -> Iterable[LifecycleEvent]:
        """Execute one policy and emit a chronological lifecycle stream."""


def evaluate_policy_matrix(
    policies: Mapping[str, Scheduler],
    executor: PolicyExecutor,
    config: EvaluationConfig,
) -> dict[str, EvaluationResult]:
    """Evaluate each named policy through one shared backend adapter."""

    if not policies:
        raise ValueError("At least one policy is required")
    results: dict[str, EvaluationResult] = {}
    for policy_name, scheduler in policies.items():
        if not isinstance(policy_name, str) or not policy_name.strip():
            raise ValueError("Policy names must be nonempty strings")
        if not isinstance(scheduler, Scheduler):
            raise TypeError(f"Policy {policy_name!r} does not implement Scheduler")
        events = executor.run(policy_name, scheduler)
        results[policy_name] = evaluate_events(
            events,
            measurement_window_s=config.measurement_window_s,
            window_start_s=config.window_start_s,
            episode_end_s=config.episode_end_s,
        )
    return results


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))
