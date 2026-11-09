"""Stateful reward adapter for canonical simulator transitions.

The simulator deliberately keeps policy observations free of realized output
length.  This adapter derives rewards from the lifecycle events emitted after
an action, so the Gymnasium wrapper does not need a second request-state model
or access to hidden request truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from llm_triage.simulation import Action, EventType, LifecycleEvent, Transition

from .rewards import AsymmetricRewardModel, RewardBreakdown
from .transitions import EpisodeOutcome, RewardEvent, RewardTransition


@dataclass
class _RequestLedger:
    arrival_time_s: float
    deadline_s: float
    service_start_s: float | None = None
    terminal: bool = False


@runtime_checkable
class RewardModel(Protocol):
    """Environment-facing contract for scoring simulator transitions.

    Implementations may keep episode-local accounting, but the environment
    only depends on this transition-level method.  Stateful implementations
    must be reset before the first transition of each episode.
    """

    def reward(self, transition: Transition) -> float:
        """Consume one canonical simulator transition and return its reward."""
        ...


class KernelRewardAdapter(RewardModel):
    """Convert one kernel transition into a scalar reward and breakdown.

    This is the canonical :class:`RewardModel` implementation used by the RL
    environment.  It composes the pure :class:`AsymmetricRewardModel` scoring
    policy with the episode accounting needed to interpret kernel lifecycle
    events.

    Call :meth:`reset` with the events emitted by ``TriageKernel.reset()``,
    then call either :meth:`reward` or :meth:`reward_breakdown` exactly once
    for every subsequent transition.  Component-level values remain available
    through :attr:`last_breakdown` for training diagnostics.
    """

    def __init__(self, model: AsymmetricRewardModel | None = None) -> None:
        self.model = model or AsymmetricRewardModel()
        self.reset()

    def reset(self, initial_events: Iterable[LifecycleEvent] = ()) -> None:
        """Clear episode state and ingest reset-time arrival events."""

        self._requests: dict[str, _RequestLedger] = {}
        self._pending_defers: dict[str, float] = {}
        self._completed = 0
        self._slo_met = 0
        self._dropped = 0
        self._episode_start_s: float | None = None
        self._last_time_s = 0.0
        self._terminal = False
        self.last_breakdown = RewardBreakdown()
        self.cumulative_breakdown = RewardBreakdown()
        self.outcome: EpisodeOutcome | None = None

        for event in initial_events:
            if event.event_type is not EventType.ARRIVAL:
                raise ValueError("reset events may only contain arrivals")
            self._record_arrival(event)
            self._last_time_s = max(self._last_time_s, event.timestamp_s)

    def reward(self, transition: Transition) -> float:
        """Consume a transition and return its scalar reward."""

        return self.reward_breakdown(transition).total

    def reward_breakdown(self, transition: Transition) -> RewardBreakdown:
        """Consume a transition and return all signed reward components."""

        if self._terminal:
            raise RuntimeError("reward adapter must be reset after episode termination")
        if transition.previous_time_s < self._last_time_s - 1e-12:
            raise ValueError("transitions must be consumed in chronological order")
        if transition.now_s < transition.previous_time_s:
            raise ValueError("transition time cannot move backward")

        breakdown = RewardBreakdown()
        for event in transition.events:
            if event.timestamp_s < self._last_time_s - 1e-12:
                raise ValueError("lifecycle events must be chronological")
            self._last_time_s = max(self._last_time_s, event.timestamp_s)

            if event.event_type is EventType.ARRIVAL:
                self._record_arrival(event)
            elif event.event_type is EventType.DECISION:
                breakdown += self._record_decision(event)
            elif event.event_type is EventType.SERVICE_START:
                ledger = self._ledger(event)
                if ledger.service_start_s is None:
                    ledger.service_start_s = event.timestamp_s
            elif event.event_type is EventType.COMPLETION:
                breakdown += self._record_completion(event)
            elif event.event_type is EventType.DROP:
                breakdown += self._record_drop(event)
            elif event.event_type is EventType.EPISODE_END:
                breakdown += self._record_episode_end(event)

        if transition.done != self._terminal:
            raise ValueError("transition.done disagrees with EPISODE_END events")

        # Charge elapsed waiting time on every environment step.  This keeps
        # defer cost bounded by simulated time even when another candidate is
        # selected before the deferred request reappears.
        if not self._terminal:
            breakdown += self._charge_pending_defers(transition.now_s)

        self._last_time_s = max(self._last_time_s, transition.now_s)
        self.last_breakdown = breakdown
        self.cumulative_breakdown += breakdown
        return breakdown

    def _record_arrival(self, event: LifecycleEvent) -> None:
        request_id = self._request_id(event)
        if request_id in self._requests:
            raise ValueError(f"duplicate arrival for request {request_id!r}")
        deadline = event.details.get("deadline_s")
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
            raise ValueError("arrival event must contain numeric deadline_s")
        deadline_s = float(deadline)
        if deadline_s <= event.timestamp_s:
            raise ValueError("reward normalization requires a positive SLO window")
        self._requests[request_id] = _RequestLedger(
            arrival_time_s=event.timestamp_s,
            deadline_s=deadline_s,
        )
        if self._episode_start_s is None:
            self._episode_start_s = event.timestamp_s
        else:
            self._episode_start_s = min(self._episode_start_s, event.timestamp_s)

    def _record_decision(self, event: LifecycleEvent) -> RewardBreakdown:
        request_id = self._request_id(event)
        self._ledger(event)
        breakdown = self._resolve_defer(request_id, event.timestamp_s)
        try:
            action = Action.coerce(event.details.get("action"))
        except (TypeError, ValueError) as exc:
            raise ValueError("decision event must contain a valid action") from exc
        if action is Action.DEFER:
            self._pending_defers[request_id] = event.timestamp_s
        return breakdown

    def _record_completion(self, event: LifecycleEvent) -> RewardBreakdown:
        request_id = self._request_id(event)
        ledger = self._ledger(event)
        self._mark_terminal(request_id, ledger)
        self._completed += 1

        if event.timestamp_s <= ledger.deadline_s:
            self._slo_met += 1
            reward_transition = RewardTransition(RewardEvent.ON_TIME_COMPLETION)
        else:
            if ledger.service_start_s is None:
                raise ValueError("completion event requires a service-start event")
            reward_transition = RewardTransition(
                RewardEvent.LATE_COMPLETION,
                service_time_s=event.timestamp_s - ledger.service_start_s,
                request_slo_s=ledger.deadline_s - ledger.arrival_time_s,
            )
        return self.model.transition_reward(reward_transition)

    def _record_drop(self, event: LifecycleEvent) -> RewardBreakdown:
        request_id = self._request_id(event)
        ledger = self._ledger(event)
        self._mark_terminal(request_id, ledger)
        self._dropped += 1
        return self.model.transition_reward(RewardTransition(RewardEvent.DROP))

    def _record_episode_end(self, event: LifecycleEvent) -> RewardBreakdown:
        if self._terminal:
            raise ValueError("episode end may only be emitted once")
        defer_reward = self._charge_pending_defers(event.timestamp_s, close=True)
        arrivals = len(self._requests)
        unfinished = arrivals - self._completed - self._dropped
        start = self._episode_start_s if self._episode_start_s is not None else event.timestamp_s
        self.outcome = EpisodeOutcome(
            arrivals=arrivals,
            completed=self._completed,
            slo_met=self._slo_met,
            dropped=self._dropped,
            unfinished=unfinished,
            duration_s=max(0.0, event.timestamp_s - start),
        )
        self._terminal = True
        return defer_reward + self.model.episode_reward(self.outcome)

    def _charge_pending_defers(
        self, timestamp_s: float, *, close: bool = False
    ) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        for request_id, start_s in tuple(self._pending_defers.items()):
            duration_s = timestamp_s - start_s
            if duration_s < -1e-12:
                raise ValueError("defer duration cannot be negative")
            if duration_s > 0:
                breakdown += self.model.transition_reward(
                    RewardTransition(
                        RewardEvent.DEFER,
                        defer_duration_s=duration_s,
                    )
                )
            if close:
                del self._pending_defers[request_id]
            else:
                self._pending_defers[request_id] = timestamp_s
        return breakdown

    def _resolve_defer(self, request_id: str, timestamp_s: float) -> RewardBreakdown:
        start_s = self._pending_defers.pop(request_id, None)
        if start_s is None or timestamp_s == start_s:
            return RewardBreakdown()
        return self.model.transition_reward(
            RewardTransition(
                RewardEvent.DEFER,
                defer_duration_s=timestamp_s - start_s,
            )
        )

    def _mark_terminal(self, request_id: str, ledger: _RequestLedger) -> None:
        if ledger.terminal:
            raise ValueError(f"duplicate terminal event for request {request_id!r}")
        if request_id in self._pending_defers:
            raise ValueError("terminal event occurred before deferred time was resolved")
        ledger.terminal = True

    def _ledger(self, event: LifecycleEvent) -> _RequestLedger:
        request_id = self._request_id(event)
        try:
            return self._requests[request_id]
        except KeyError as exc:
            raise ValueError(f"event precedes arrival for request {request_id!r}") from exc

    @staticmethod
    def _request_id(event: LifecycleEvent) -> str:
        if event.request_id is None:
            raise ValueError(f"{event.event_type.value} event requires request_id")
        return str(event.request_id)
