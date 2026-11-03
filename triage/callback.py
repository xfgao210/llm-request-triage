"""Scheduling callback protocol, simulator snapshots, and lifecycle events.

This module is the **interface contract** between the simulator engine and
every scheduling policy (FCFS, length-aware / SLO-aware baselines, the RL
agent), as well as the lifecycle event stream consumed by the evaluation
engine (``TECH_DESIGN.md`` §7.2–§7.5).

Two scheduling interfaces are supported directly:

1. **Per-candidate callback** (:class:`SchedulerCallback`): implements
   ``decide(state: SimState) -> Action`` (and optionally
   ``priority(view: RequestView) -> float``).
2. **Snapshot-to-Decision scheduler** (:class:`Scheduler`): implements
   ``decide(snapshot: SimState) -> Decision | None``, selecting a candidate
   from ``snapshot.candidates`` and returning ``Decision(request_id, action)``
   (matching ``llm_triage.scheduling.Scheduler`` on teammate branches).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, Tuple, Union, runtime_checkable

from triage.request import Action, RequestState


# ---------------------------------------------------------------------------
# Policy-visible request and resource views (no ground-truth output length!)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RequestView:
    """Read-only projection of a request visible to schedulers.

    Also aliased as :class:`CandidateView` and exposes all fields and methods
    of ``TECH_DESIGN.md`` §7.2 (``arrival_time_s``, ``deadline_s``,
    ``estimated_output_tokens``, ``estimated_service_s``,
    ``estimated_slack_s(now_s)``) so ``llm_triage`` selectors and policies can
    inspect it directly without an adapter wrapper.

    .. warning::
        The ground-truth ``output_tokens`` is deliberately absent.  Policies
        must decide without knowing how long the response will be.
    """

    request_id: Union[int, str]
    arrival_time: float
    input_tokens: int
    slo_deadline: float
    slo_slack: float        # deadline − now (negative once overdue)
    time_in_queue: float    # now − arrival_time
    state: RequestState
    tokens_generated: int   # tokens produced so far (0 while queued / after preemption)
    defer_count: int
    preemption_count: int
    estimated_output_tokens: float = 144.0
    estimated_service_s: float = 0.1

    # ---- TECH_DESIGN.md §7.2 compatibility properties --------------------

    @property
    def arrival_time_s(self) -> float:
        return self.arrival_time

    @property
    def deadline_s(self) -> float:
        return self.slo_deadline

    def estimated_slack_s(self, now_s: float | None = None) -> float:
        """Time to deadline after estimated service time is deducted."""
        if now_s is None:
            return self.slo_slack - self.estimated_service_s
        return self.slo_deadline - float(now_s) - self.estimated_service_s


# Alias for TECH_DESIGN.md §7.2
CandidateView = RequestView


@dataclass(frozen=True)
class ResourceView:
    """Sanitized resource state relevant to an admission decision.

    Exposes both ``can_admit`` (used by ``origin/evaluation-engine``) and
    ``admission_available`` (used by ``origin/hengyi/week1-data-fcfs``).
    """

    can_admit: bool
    active_batch_size: int = 0
    kv_cache_available_tokens: int | None = None
    kv_cache_available: float | None = None
    prefill_utilization: float = 0.0
    decode_utilization: float = 0.0

    @property
    def admission_available(self) -> bool:
        return self.can_admit


@dataclass(frozen=True)
class Decision:
    """Combined candidate selection and triage action (``TECH_DESIGN.md`` §7.3)."""

    request_id: Union[int, str]
    action: Action

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", Action.coerce(self.action))


# ---------------------------------------------------------------------------
# Simulator state snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimState:
    """Immutable snapshot handed to the scheduler at each decision point.

    Simultaneously satisfies :class:`SimState` (via ``candidate``,
    ``queue_requests``, etc.) and ``SchedulerSnapshot`` from
    ``TECH_DESIGN.md`` §7.2 (via ``now_s``, ``candidates``, ``resources``).
    """

    now: float

    # Request being triaged (selected by priority or head of queue)
    candidate: RequestView

    # Eligible waiting requests in arrival order
    queue_length: int
    queue_requests: Tuple[RequestView, ...]

    # Requests resident on the GPU: decoding plus those admitted earlier in
    # this scheduling round (about to be prefilled)
    active_count: int
    active_requests: Tuple[RequestView, ...]
    pending_prefill_count: int
    pending_prefill_tokens: int

    # Hardware
    max_batch_size: int
    batch_utilization: float  # active_count / max_batch_size
    kv_block_size: int
    kv_blocks_used: int
    kv_blocks_total: int
    kv_utilization: float     # kv_blocks_used / kv_blocks_total

    # Cumulative counters
    completed_count: int
    violated_count: int
    dropped_count: int
    preempted_count: int

    @property
    def batch_full(self) -> bool:
        return self.active_count >= self.max_batch_size

    @property
    def kv_blocks_free(self) -> int:
        return self.kv_blocks_total - self.kv_blocks_used

    def candidate_fits(self, view: RequestView | None = None) -> bool:
        """Would ADMIT succeed right now for *view* (defaults to ``candidate``)?"""
        target = self.candidate if view is None else view
        blocks = -(-target.input_tokens // self.kv_block_size)
        return not self.batch_full and blocks <= self.kv_blocks_free

    # ---- SchedulerSnapshot (TECH_DESIGN.md §7.2) compatibility -----------

    @property
    def now_s(self) -> float:
        return self.now

    @property
    def candidates(self) -> Tuple[RequestView, ...]:
        return self.queue_requests

    @property
    def resources(self) -> ResourceView:
        any_fits = any(self.candidate_fits(v) for v in self.queue_requests)
        prefill_util = (
            min(1.0, self.pending_prefill_count / self.max_batch_size)
            if self.max_batch_size > 0
            else 1.0
        )
        decode_count = max(0, self.active_count - self.pending_prefill_count)
        decode_util = (
            min(1.0, decode_count / self.max_batch_size)
            if self.max_batch_size > 0
            else 1.0
        )
        return ResourceView(
            can_admit=any_fits,
            active_batch_size=self.active_count,
            kv_cache_available_tokens=self.kv_blocks_free * self.kv_block_size,
            kv_cache_available=1.0 - self.kv_utilization,
            prefill_utilization=prefill_util,
            decode_utilization=decode_util,
        )


SchedulerSnapshot = SimState


# ---------------------------------------------------------------------------
# Lifecycle events (TECH_DESIGN.md §7.5)
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Request lifecycle events emitted by the simulator."""

    ARRIVAL = "arrival"
    DECISION = "decision"
    SERVICE_START = "service_start"
    FIRST_TOKEN = "first_token"
    COMPLETION = "completion"
    DROP = "drop"
    EPISODE_END = "episode_end"


@dataclass(frozen=True)
class LifecycleEvent:
    """A timestamped lifecycle fact emitted by the simulation kernel."""

    event_type: EventType
    timestamp_s: float
    request_id: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    """Result of applying one scheduling decision to :class:`~triage.engine.TriageKernel`."""

    decision: Decision
    previous_time_s: float
    now_s: float
    events: Tuple[LifecycleEvent, ...]
    done: bool


# ---------------------------------------------------------------------------
# Scheduler protocols
# ---------------------------------------------------------------------------

@runtime_checkable
class SchedulerCallback(Protocol):
    """Per-candidate callback interface.

    ``decide(state)`` returns either an :class:`Action` (applied to
    ``state.candidate``) or a :class:`Decision` (identifying any request in
    ``state.candidates`` and the action to apply).
    """

    def decide(self, state: SimState) -> Union[Action, Decision, None]:
        """Decide what to do at the current scheduling point."""
        ...
