"""Pluggable reward model interface and default asymmetric reward configuration.

This module implements ``TECH_DESIGN.md`` §6.7 / §7.4:
- :class:`RewardModel`: replaceable protocol consumed by :class:`~llm_triage.rl.environment.TriageEnv`.
- :class:`RewardConfig` and :class:`AsymmetricRewardModel`: default reward
  implementation providing positive reward for on-time completions, asymmetric
  penalties for SLO violations/wasted GPU time versus proactive drops, bounded
  defer costs, and an episode-level SLO attainment term.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Protocol, Set, Union, runtime_checkable

from llm_triage.simulation import (
    Action,
    EventType,
    Request,
    RequestOutcome,
    RequestState,
    Transition,
    TriageKernel,
)


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the asymmetric triage reward function.

    Matches the parameter names used in ``contributions/hengyi/rl_env.py``
    while adding ``invalid_action_penalty`` for ``invalid_action_mode="penalize"``.
    """

    success: float = 1.0
    late: float = 0.1
    wasted_service: float = 0.2
    drop: float = 0.05
    defer_per_second: float = 0.02
    terminal_attainment: float = 1.0
    invalid_action_penalty: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "success",
            "late",
            "wasted_service",
            "drop",
            "defer_per_second",
            "terminal_attainment",
            "invalid_action_penalty",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be a finite nonnegative number, got {val!r}")


@runtime_checkable
class RewardModel(Protocol):
    """Protocol for replaceable reward shaping components (``TECH_DESIGN.md`` §7.4)."""

    def reward(self, transition: Transition) -> float:
        """Compute scalar step reward for *transition*."""
        ...


class AsymmetricRewardModel:
    """Default asymmetric reward model wired to :class:`Transition` outcomes.

    Works both standalone (purely from :class:`Transition`) and when bound to
    a :class:`TriageKernel`. Accounts for:
    - Proactive drop penalty (``-config.drop``) when ``Action.DROP`` is taken.
    - Defer waiting cost (``-config.defer_per_second * defer_retry_s``) when
      ``Action.DEFER`` is taken.
    - On-time completion reward (``+config.success``) and late/violated
      penalties (``-config.late - config.wasted_service * gpu_time_s / slo_budget``)
      for every request reaching ``COMPLETED`` or ``VIOLATED`` during the
      kernel step.
    - Terminal episode attainment bonus
      (``+config.terminal_attainment * slo_attainment``) at episode end.
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        *,
        defer_retry_s: float = 0.01,
    ) -> None:
        self.config = config or RewardConfig()
        self.defer_retry_s = float(defer_retry_s)
        self._kernel: Optional[TriageKernel] = None
        self._scored_requests: Set[Union[int, str]] = set()
        self._request_map: Dict[str, Request] = {}

    def bind(
        self,
        kernel: TriageKernel,
        *,
        defer_retry_s: Optional[float] = None,
    ) -> None:
        """Attach the reward model to *kernel* at ``reset()`` time."""
        self._kernel = kernel
        if defer_retry_s is not None:
            self.defer_retry_s = float(defer_retry_s)
        self._scored_requests = set()
        self._request_map = {str(r.request_id): r for r in kernel.requests}

    def reset(self) -> None:
        """Reset per-episode reward accounting."""
        self._scored_requests = set()
        if self._kernel is not None:
            self._request_map = {str(r.request_id): r for r in self._kernel.requests}

    def _score_request_outcome(self, outcome: Union[Request, RequestOutcome]) -> float:
        cfg = self.config
        if outcome.state is RequestState.COMPLETED:
            return cfg.success
        if outcome.state is RequestState.VIOLATED:
            arrival = getattr(outcome, "arrival_time_s", getattr(outcome, "arrival_time", 0.0))
            deadline = getattr(outcome, "deadline_s", getattr(outcome, "slo_deadline", arrival + 1.0))
            slo_budget = max(1e-3, float(deadline) - float(arrival))
            return -cfg.late - cfg.wasted_service * (float(outcome.gpu_time_s) / slo_budget)
        return 0.0

    def reward(
        self,
        transition: Transition,
        *,
        invalid_action: bool = False,
    ) -> float:
        """Compute scalar step reward for *transition*."""
        cfg = self.config
        total = 0.0

        if invalid_action:
            total -= cfg.invalid_action_penalty

        action = Action.coerce(transition.decision.action)
        if action is Action.DROP:
            total -= cfg.drop
            self._scored_requests.add(str(transition.decision.request_id))
        elif action is Action.DEFER:
            total -= cfg.defer_per_second * self.defer_retry_s

        if transition.outcomes:
            for outcome in transition.outcomes:
                key = str(outcome.request_id)
                if key in self._scored_requests:
                    continue
                if outcome.state in (RequestState.COMPLETED, RequestState.VIOLATED):
                    self._scored_requests.add(key)
                    total += self._score_request_outcome(outcome)
        elif self._kernel is not None:
            if not self._request_map and self._kernel.requests:
                self._request_map = {str(r.request_id): r for r in self._kernel.requests}
            for req in self._kernel.requests:
                key = str(req.request_id)
                if key in self._scored_requests:
                    continue
                if req.state in (RequestState.COMPLETED, RequestState.VIOLATED):
                    self._scored_requests.add(key)
                    total += self._score_request_outcome(req)
        else:
            # Standalone fallback for hand-constructed Transition objects without outcomes
            for ev in transition.events:
                if ev.event_type is EventType.COMPLETION and ev.request_id is not None:
                    if ev.request_id in self._scored_requests:
                        continue
                    self._scored_requests.add(ev.request_id)
                    details = ev.details or {}
                    slo_met = details.get("slo_met")
                    if slo_met is None and "deadline_s" in details:
                        slo_met = float(ev.timestamp_s) <= float(details["deadline_s"])  # type: ignore[arg-type]
                    if slo_met is False:
                        arrival = float(details.get("arrival_time_s", 0.0))  # type: ignore[arg-type]
                        deadline = float(details.get("deadline_s", ev.timestamp_s))  # type: ignore[arg-type]
                        gpu_time = float(details.get("gpu_time_s", 0.0))  # type: ignore[arg-type]
                        slo_budget = max(1e-3, deadline - arrival)
                        total += -cfg.late - cfg.wasted_service * (gpu_time / slo_budget)
                    else:
                        total += cfg.success

        if transition.done:
            if transition.total_requests > 0:
                total += cfg.terminal_attainment * float(transition.slo_attainment)
            elif self._kernel is not None and self._kernel.requests:
                attainment = self._kernel.completed_count / len(self._kernel.requests)
                total += cfg.terminal_attainment * attainment

        return float(total)


__all__ = [
    "AsymmetricRewardModel",
    "RewardConfig",
    "RewardModel",
]
