"""Iteration-level LLM serving simulator and steppable TriageKernel.

This module implements both:

1. :class:`TriageKernel` (``TECH_DESIGN.md`` §6.4 / §7.4): the authoritative,
   steppable request-state machine (`reset()`, `snapshot()`, `apply(decision)`,
   `is_terminal()`, `events()`) used by baseline evaluation and the Week 2
   Gymnasium RL environment wrapper.
2. :func:`simulate`: the convenience runner that drives :class:`TriageKernel`
   to completion with any :class:`~llm_triage.simulation.callback.SchedulerCallback` or
   ``llm_triage.scheduling.Scheduler`` and returns a :class:`SimResult`.
3. :class:`SimulatorExecutor`: the ``PolicyExecutor`` adapter
   (``run(policy_name, scheduler) -> Iterable[LifecycleEvent]``) consumed by
   ``evaluate_policy_matrix`` in the evaluation engine.

Execution model (one GPU iteration at a time, vLLM-style):

- **Schedule**: absorb arrivals up to ``now``, auto-resume any previously
  admitted requests that were preempted when capacity frees up, and expose
  the remaining eligible queue to the policy one decision at a time.
- **Prefill**: when one or more requests are admitted (and the GPU did not
  just run a prefill while decode requests are waiting), run one batched
  prefill iteration over the new prompts.  Prefill emits each request's first
  token (``first_token_time`` / ``FIRST_TOKEN`` event) and immediately updates
  KV block allocations.
- **Decode**: otherwise, advance all running requests by one token in
  lockstep.  KV blocks are reserved prior to the step: any sequence that has
  outgrown total hardware KV capacity is terminated first; if free blocks are
  still insufficient, the newest running request is preempted (recompute:
  blocks freed, progress reset, queued for automatic re-admission without
  re-running admission triage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .callback import (
    Decision,
    EventType,
    LifecycleEvent,
    RequestView,
    SchedulerCallback,
    SimState,
    Transition,
)
from .hardware import HardwareConfig, KVCacheManager
from .profile import ServiceProfile
from .request import (
    Action,
    Request,
    RequestState,
    _check_finite,
    _check_int,
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IterationRecord:
    """One GPU iteration (or idle gap), for utilization analysis."""

    start: float
    end: float
    kind: str              # "prefill" | "decode" | "idle"
    batch_size: int        # sequences processed in this iteration
    tokens: int            # prompt tokens (prefill) or context tokens (decode)
    kv_blocks_used: int    # after the iteration
    queue_length: int      # at iteration start


@dataclass
class SimResult:
    """Outcome of a simulation run."""

    requests: List[Request]
    total_time: float
    hardware: HardwareConfig
    profile: ServiceProfile
    iterations: List[IterationRecord] = field(default_factory=list)
    events: List[LifecycleEvent] = field(default_factory=list)
    preemptions: int = 0

    def by_id(self) -> Dict[Union[int, str], Request]:
        return {r.request_id: r for r in self.requests}

    def _with(self, state: RequestState) -> List[Request]:
        return [r for r in self.requests if r.state == state]

    @property
    def completed(self) -> List[Request]:
        return self._with(RequestState.COMPLETED)

    @property
    def violated(self) -> List[Request]:
        return self._with(RequestState.VIOLATED)

    @property
    def dropped(self) -> List[Request]:
        return self._with(RequestState.DROPPED)

    @property
    def slo_attainment(self) -> float:
        """Requests finished within SLO / *all* requests (drops count against)."""
        return len(self.completed) / len(self.requests) if self.requests else 0.0

    @property
    def drop_rate(self) -> float:
        return len(self.dropped) / len(self.requests) if self.requests else 0.0

    @property
    def goodput(self) -> float:
        """SLO-meeting completions per simulated second."""
        return len(self.completed) / self.total_time if self.total_time > 0 else 0.0

    @property
    def gpu_busy_time(self) -> float:
        return sum(it.end - it.start for it in self.iterations if it.kind != "idle")

    @property
    def wasted_gpu_time(self) -> float:
        """GPU seconds spent on requests that did not meet their SLO."""
        return sum(r.gpu_time_s for r in self.requests if r.state != RequestState.COMPLETED)

    def summary(self) -> str:
        rows = [
            ("Total requests", f"{len(self.requests)}"),
            ("Completed (SLO met)", f"{len(self.completed)}"),
            ("Violated (SLO missed)", f"{len(self.violated)}"),
            ("Dropped", f"{len(self.dropped)}"),
            ("Preemptions", f"{self.preemptions}"),
            ("SLO attainment", f"{self.slo_attainment:.2%}"),
            ("Drop rate", f"{self.drop_rate:.2%}"),
            ("Goodput (req/s)", f"{self.goodput:.3f}"),
            ("GPU busy time (s)", f"{self.gpu_busy_time:.3f}"),
            ("Wasted GPU time (s)", f"{self.wasted_gpu_time:.3f}"),
            ("Simulated time (s)", f"{self.total_time:.3f}"),
        ]
        return "\n".join(f"{k:<24} {v}" for k, v in rows)


# ---------------------------------------------------------------------------
# Steppable TriageKernel (TECH_DESIGN.md §6.4 / §7.4)
# ---------------------------------------------------------------------------

class TriageKernel:
    """Authoritative, steppable request-state machine.

    Usage (step-by-step, e.g. from a Gymnasium environment)::

        kernel = TriageKernel(hardware=hw, profile=profile)
        kernel.reset(workload)
        while not kernel.is_terminal():
            snap = kernel.snapshot()
            decision = policy_decide(snap)
            transition = kernel.apply(decision)

    At every decision point ``kernel.snapshot()`` returns a :class:`SimState`
    containing only the currently eligible candidates (those not deferred earlier
    in the same scheduling pass and not waiting past their ``defer_retry_s``
    cooldown).  Calling ``kernel.apply(decision)`` applies one decision for
    one candidate and automatically advances the simulation clock through any
    subsequent GPU prefill/decode iterations or idle gaps until the next
    decision point or episode termination.
    """

    def __init__(
        self,
        hardware: Optional[HardwareConfig] = None,
        profile: Optional[ServiceProfile] = None,
        *,
        defer_retry_s: float = 0.01,
        max_defers: Optional[int] = 64,
        expire_overdue_in_queue: bool = False,
        estimated_output_tokens: Optional[float] = None,
        service_estimator: Optional[Callable[[ int, float], float]] = None,
        priority_fn: Optional[Callable[[RequestView], object]] = None,
        record_iterations: bool = True,
    ) -> None:
        self.hw = hardware or HardwareConfig()
        self.profile = profile or ServiceProfile()
        self.defer_retry_s = _check_finite(defer_retry_s, "defer_retry_s", strictly_positive=True)
        if max_defers is not None:
            _check_int(max_defers, "max_defers", min_val=1)
        self.max_defers = max_defers
        self.expire_overdue_in_queue = bool(expire_overdue_in_queue)
        self.estimated_output_tokens = (
            self.profile.default_estimated_output_tokens
            if estimated_output_tokens is None
            else _check_finite(
                estimated_output_tokens,
                "estimated_output_tokens",
                strictly_positive=True,
            )
        )
        self._service_estimator = service_estimator
        self._priority_fn = priority_fn
        self.record_iterations = bool(record_iterations)

        # Initialized in reset()
        self.requests: List[Request] = []
        self.kv = KVCacheManager(self.hw)
        self.now: float = 0.0
        self._pending: List[Request] = []
        self._next: int = 0
        self.queue: List[Request] = []
        self._preempted_queue: List[Request] = []
        self._admitted_batch: List[Request] = []
        self.running: List[Request] = []
        self._deferred_in_pass: Set[Union[int, str]] = set()
        self._defer_until: Dict[Union[int, str], float] = {}
        self._last_iter_kind: Optional[str] = None
        self._episode_ended: bool = False

        self._preempted_count_before_completion: int = 0
        self.completed_count: int = 0
        self.violated_count: int = 0
        self.dropped_count: int = 0
        self.preempted_count: int = 0
        self.iterations: List[IterationRecord] = []
        self._events: List[LifecycleEvent] = []
        self._emitted_service_start: Set[Union[int, str]] = set()
        self._emitted_first_token: Set[Union[int, str]] = set()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, requests: Sequence[object]) -> Optional[SimState]:
        """Reset the kernel with a fresh copy of *requests* and advance to the
        first scheduling decision point (or termination if *requests* is empty)."""
        copied = [Request.from_any(r) for r in requests]
        # Lifecycle events serialize request IDs as strings. Reject collisions
        # such as 1 and "1" before they become indistinguishable downstream.
        ids = [str(r.request_id) for r in copied]
        if len(set(ids)) != len(ids):
            raise ValueError("request_ids must be unique after string normalization")

        self.requests = copied
        self.kv = KVCacheManager(self.hw)
        self.now = 0.0
        # Sort by arrival time, preserving input order for ties
        indexed = list(enumerate(self.requests))
        indexed.sort(key=lambda pair: (pair[1].arrival_time, pair[0]))
        self._pending = [r for _, r in indexed]
        self._next = 0
        self.queue = []
        self._preempted_queue = []
        self._admitted_batch = []
        self.running = []
        self._deferred_in_pass = set()
        self._defer_until = {}
        self._last_iter_kind = None
        self._episode_ended = False

        self._preempted_count_before_completion = 0
        self.completed_count = 0
        self.violated_count = 0
        self.dropped_count = 0
        self.preempted_count = 0
        self.iterations = []
        self._events = []
        self._emitted_service_start = set()
        self._emitted_first_token = set()

        self._advance_until_decision_or_done()
        return None if self.is_terminal() else self.snapshot()

    def is_terminal(self) -> bool:
        """``True`` when all requests have reached a terminal state."""
        return self._episode_ended

    def events(self) -> Tuple[LifecycleEvent, ...]:
        """Return all chronological lifecycle events emitted so far."""
        return tuple(self._events)

    def snapshot(self) -> SimState:
        """Build the policy-visible :class:`SimState` at the current decision point."""
        eligible = self._eligible_candidates()
        if not eligible:
            raise RuntimeError("No eligible candidates at current state; check is_terminal()")
        cand = self._pick_candidate(eligible)
        active = self.running + self._admitted_batch
        return SimState(
            now=self.now,
            candidate=self._view(cand),
            queue_length=len(eligible),
            queue_requests=tuple(self._view(r) for r in eligible),
            active_count=len(active),
            active_requests=tuple(self._view(r) for r in active),
            pending_prefill_count=len(self._admitted_batch),
            pending_prefill_tokens=sum(r.input_tokens for r in self._admitted_batch),
            max_batch_size=self.hw.max_batch_size,
            batch_utilization=len(active) / self.hw.max_batch_size,
            kv_block_size=self.hw.kv_block_size,
            kv_blocks_used=self.kv.blocks_used,
            kv_blocks_total=self.kv.total_blocks,
            kv_utilization=self.kv.utilization,
            completed_count=self.completed_count,
            violated_count=self.violated_count,
            dropped_count=self.dropped_count,
            preempted_count=self.preempted_count,
        )

    def apply(
        self,
        decision: Union[Action, Decision, str, int],
        *,
        decision_overhead_s: float = 0.0,
    ) -> Transition:
        """Apply one scheduling decision and advance to the next decision point."""
        if self._episode_ended:
            raise RuntimeError("Cannot call apply() on a terminated episode")
        eligible = self._eligible_candidates()
        if not eligible:
            raise RuntimeError("No eligible candidates to apply a decision to")

        prev_now = self.now
        event_start_idx = len(self._events)

        if isinstance(decision, Decision):
            target_id = decision.request_id
            action = Action.coerce(decision.action)
            cand = next(
                (r for r in eligible if r.request_id == target_id or str(r.request_id) == str(target_id)),
                None,
            )
            if cand is None:
                raise ValueError(f"Decision request_id {target_id!r} is not in eligible candidates")
        else:
            cand = self._pick_candidate(eligible)
            action = Action.coerce(decision)
            target_id = cand.request_id

        resolved_decision = Decision(request_id=target_id, action=action)
        cand.decision_time = self.now
        self._emit(
            EventType.DECISION,
            cand.request_id,
            action=action.value,
            decision_overhead_s=max(0.0, float(decision_overhead_s)),
            queue_length=len(self.queue) + len(self._preempted_queue),
        )

        if action is Action.ADMIT:
            if self._can_admit_request(cand):
                self.queue.remove(cand)
                self._defer_until.pop(cand.request_id, None)
                self._stage_for_prefill(cand)
            else:
                # Physical capacity blocked this specific candidate; skip it for
                # the rest of this scheduling pass without charging a defer.
                self._deferred_in_pass.add(cand.request_id)
        elif action is Action.DROP:
            self.queue.remove(cand)
            self._defer_until.pop(cand.request_id, None)
            self._terminate(cand, RequestState.DROPPED)
            self._emit(
                EventType.DROP,
                cand.request_id,
                queue_length=len(self.queue) + len(self._preempted_queue),
            )
        else:  # Action.DEFER
            cand.defer_count += 1
            if self.max_defers is not None and cand.defer_count >= self.max_defers:
                # Bounded defer limit reached: expire the request as unserved
                # (VIOLATED; no DROP or COMPLETION event so RequestCollector
                # records it as UNFINISHED / slo_met=False).
                self.queue.remove(cand)
                self._defer_until.pop(cand.request_id, None)
                self._terminate(cand, RequestState.VIOLATED)
            else:
                self._deferred_in_pass.add(cand.request_id)
                self._defer_until[cand.request_id] = self.now + self.defer_retry_s

        self._advance_until_decision_or_done()
        new_events = tuple(self._events[event_start_idx:])
        return Transition(
            decision=resolved_decision,
            previous_time_s=prev_now,
            now_s=self.now,
            events=new_events,
            done=self._episode_ended,
        )

    def result(self) -> SimResult:
        """Build a :class:`SimResult` from the current state of the kernel."""
        return SimResult(
            requests=self.requests,
            total_time=self.now,
            hardware=self.hw,
            profile=self.profile,
            iterations=list(self.iterations),
            events=list(self._events),
            preemptions=self.preempted_count,
        )

    # -- internal helpers --------------------------------------------------

    def _emit(
        self,
        event_type: EventType,
        request_id: Union[int, str, None] = None,
        *, timestamp_s: Optional[float] = None,
        **details: object,
    ) -> None:
        str_id = None if request_id is None else str(request_id)
        self._events.append(
            LifecycleEvent(
                event_type=event_type,
                timestamp_s=self.now if timestamp_s is None else timestamp_s,
                request_id=str_id,
                details=details,
            )
        )

    def _has_pending_arrivals(self) -> bool:
        return self._next < len(self._pending)

    def _absorb_arrivals(self) -> None:
        while self._has_pending_arrivals() and self._pending[self._next].arrival_time <= self.now:
            req = self._pending[self._next]
            req.state = RequestState.QUEUED
            self.queue.append(req)
            self._next += 1
            self._emit(
                EventType.ARRIVAL,
                req.request_id,
                timestamp_s=req.arrival_time,
                deadline_s=req.slo_deadline,
                input_tokens=req.input_tokens,
                queue_length=len(self.queue) + len(self._preempted_queue),
            )

    def _terminate(self, req: Request, state: RequestState) -> None:
        req.state = state
        req.completion_time = self.now
        self.kv.release(req)
        if state is RequestState.COMPLETED:
            self.completed_count += 1
        elif state is RequestState.VIOLATED:
            self.violated_count += 1
        else:
            self.dropped_count += 1

    def _finish_served(self, req: Request) -> None:
        self._preempted_count_before_completion = 0
        ok = self.now <= req.slo_deadline
        self._terminate(req, RequestState.COMPLETED if ok else RequestState.VIOLATED)
        self._emit(
            EventType.COMPLETION,
            req.request_id,
            queue_length=len(self.queue) + len(self._preempted_queue),
        )

    def _estimate_service_s(self, input_tokens: int) -> float:
        if self._service_estimator is not None:
            return float(self._service_estimator(input_tokens, self.estimated_output_tokens))
        return self.profile.estimate_service_time(input_tokens, self.estimated_output_tokens)

    def _view(self, req: Request) -> RequestView:
        return RequestView(
            request_id=req.request_id,
            arrival_time=req.arrival_time,
            input_tokens=req.input_tokens,
            slo_deadline=req.slo_deadline,
            slo_slack=req.slo_deadline - self.now,
            time_in_queue=self.now - req.arrival_time,
            state=req.state,
            tokens_generated=req.tokens_generated,
            defer_count=req.defer_count,
            preemption_count=req.preemption_count,
            estimated_output_tokens=self.estimated_output_tokens,
            estimated_service_s=self._estimate_service_s(req.input_tokens),
        )

    def _pick_candidate(self, eligible: Sequence[Request]) -> Request:
        if self._priority_fn is None:
            return eligible[0]
        prio = self._priority_fn
        return min(eligible, key=lambda r: prio(self._view(r)))

    def _record(self, start: float, kind: str, batch: int, tokens: int, qlen: int) -> None:
        if self.record_iterations:
            self.iterations.append(
                IterationRecord(start, self.now, kind, batch, tokens, self.kv.blocks_used, qlen)
            )

    def _active_count(self) -> int:
        return len(self.running) + len(self._admitted_batch)

    def _can_admit_request(self, req: Request) -> bool:
        if self._active_count() >= self.hw.max_batch_size:
            return False
        # Prefill emits the first output token (context_tokens = input_tokens + 1
        # if output_tokens > 1, or input_tokens during prefill). Reserve enough
        # blocks for the prompt; _run_prefill will grow by 1 token if needed and
        # kept in batch, so check input_tokens here.
        return self.kv.can_allocate(req.input_tokens)

    def _stage_for_prefill(self, req: Request) -> None:
        self.kv.allocate(req, req.input_tokens)
        req.state = RequestState.PREFILLING
        if req.admit_time is None:
            req.admit_time = self.now
        if req.request_id not in self._emitted_service_start:
            self._emitted_service_start.add(req.request_id)
            self._emit(
                EventType.SERVICE_START,
                req.request_id,
                queue_length=len(self.queue) + len(self._preempted_queue),
            )
        self._admitted_batch.append(req)

    def _prune_infeasible_and_expired(self) -> None:
        """Remove requests from the queue that can never fit in hardware KV
        cache, and (if ``expire_overdue_in_queue`` is enabled) overdue requests."""
        for r in list(self.queue):
            if not self.kv.fits_ever(r.input_tokens):
                self.queue.remove(r)
                self._defer_until.pop(r.request_id, None)
                self._terminate(r, RequestState.VIOLATED)
            elif self.expire_overdue_in_queue and self.now >= r.slo_deadline:
                self.queue.remove(r)
                self._defer_until.pop(r.request_id, None)
                self._terminate(r, RequestState.VIOLATED)

    def _resume_preempted(self) -> None:
        """Automatically re-admit preempted requests in FIFO order once a
        running sequence has completed and freed KV blocks, without re-running
        the admission triage callback."""
        if self.running and self._preempted_count_before_completion > 0:
            return
        while self._preempted_queue:
            req = self._preempted_queue[0]
            if not self.kv.fits_ever(req.input_tokens):
                self._preempted_queue.pop(0)
                self._terminate(req, RequestState.VIOLATED)
                continue
            if not self._can_admit_request(req):
                break
            self._preempted_queue.pop(0)
            self._stage_for_prefill(req)

    def _should_interleave_decode(self) -> bool:
        """Prevent consecutive prefill iterations from starving active decode
        requests during a traffic burst."""
        return bool(self.running) and self._last_iter_kind == "prefill"

    def _eligible_candidates(self) -> List[Request]:
        """Return queued requests eligible for a policy decision right now."""
        if self._should_interleave_decode():
            return []
        if self._preempted_queue:
            return []
        if self._active_count() >= self.hw.max_batch_size or self.kv.blocks_free == 0:
            return []
        eligible: List[Request] = []
        for r in self.queue:
            if r.request_id in self._deferred_in_pass:
                continue
            cooldown = self._defer_until.get(r.request_id)
            if cooldown is not None and self.now < cooldown - 1e-12:
                continue
            eligible.append(r)
        return eligible

    def _advance_until_decision_or_done(self) -> None:
        """Advance simulation state through prefill/decode/idle steps until
        either a policy decision is needed or all requests are terminal."""
        while True:
            self._absorb_arrivals()
            self._prune_infeasible_and_expired()

            if not self._should_interleave_decode():
                self._resume_preempted()

            eligible = self._eligible_candidates()
            if eligible:
                return  # Pause for callback / step(action)

            # Current scheduling pass is complete; execute one GPU iteration or idle.
            self._deferred_in_pass.clear()

            if self._admitted_batch and not self._should_interleave_decode():
                self._run_prefill(self._admitted_batch)
                self._admitted_batch = []
            elif self.running:
                self._run_decode()
            elif self._admitted_batch:
                self._run_prefill(self._admitted_batch)
                self._admitted_batch = []
            elif self.queue or self._preempted_queue or self._has_pending_arrivals():
                self._idle()
            else:
                self._episode_ended = True
                self._emit(EventType.EPISODE_END, None)
                return

    # -- GPU iterations ----------------------------------------------------

    def _run_prefill(self, batch: List[Request]) -> None:
        start = self.now
        qlen = len(self.queue) + len(self._preempted_queue)
        prompts = [r.input_tokens for r in batch]
        dur = self.profile.prefill_iteration_time(prompts)
        self.now += dur
        # Preserve real arrival times before emitting iteration-end events.
        self._absorb_arrivals()
        total = sum(prompts)
        for r in batch:
            r.gpu_time_s += dur * r.input_tokens / total
            r.state = RequestState.DECODING
            r.tokens_generated = 1  # prefill emits the first token
            if r.first_token_time is None:
                r.first_token_time = self.now
            if r.request_id not in self._emitted_first_token:
                self._emitted_first_token.add(r.request_id)
                self._emit(
                    EventType.FIRST_TOKEN,
                    r.request_id,
                    queue_length=len(self.queue) + len(self._preempted_queue),
                )
            if r.tokens_generated >= r.output_tokens:
                self._finish_served(r)
            else:
                self.running.append(r)

        # Synchronize KV block allocations immediately after prefill so the
        # next scheduling pass sees accurate free blocks.
        self._reserve_running_kv()
        self._last_iter_kind = "prefill"
        self._record(start, "prefill", len(batch), total, qlen)

    def _preempt(self, req: Request) -> None:
        """Recompute-style preemption: free KV, drop progress, and queue for
        automatic re-admission ahead of new arrivals."""
        self.running.remove(req)
        self.kv.release(req)
        req.state = RequestState.QUEUED
        req.tokens_generated = 0
        req.preemption_count += 1
        self.preempted_count += 1
        self._preempted_count_before_completion += 1
        self._preempted_queue.insert(0, req)

    def _reserve_running_kv(self) -> None:
        """Ensure every running sequence has KV blocks allocated for its
        current ``context_tokens``.

        1. First terminate any sequence whose ``context_tokens`` exceeds the
           entire hardware KV-cache (so it never causes innocent sequences to
           be preempted).
        2. If total blocks needed still exceed free blocks, preempt the most
           recently admitted sequence until the remainder fits.
        """
        for r in list(self.running):
            if not self.kv.fits_ever(r.context_tokens):
                self.running.remove(r)
                self._terminate(r, RequestState.VIOLATED)

        while self.running:
            need = sum(self.kv.extra_blocks_needed(r, r.context_tokens) for r in self.running)
            if need <= self.kv.blocks_free:
                break
            if len(self.running) == 1:
                self._terminate(self.running.pop(), RequestState.VIOLATED)
                break
            self._preempt(self.running[-1])

        for r in self.running:
            self.kv.update(r, r.context_tokens)

    def _run_decode(self) -> None:
        start = self.now
        qlen = len(self.queue) + len(self._preempted_queue)
        self._reserve_running_kv()
        if not self.running:
            self._last_iter_kind = "decode"
            return
        contexts = [r.context_tokens for r in self.running]
        dur = self.profile.decode_iteration_time(contexts)
        self.now += dur
        self._absorb_arrivals()
        share = dur / len(self.running)
        finished: List[Request] = []
        for r in self.running:
            r.gpu_time_s += share
            r.tokens_generated += 1
            if r.tokens_generated >= r.output_tokens:
                finished.append(r)
        for r in finished:
            self.running.remove(r)
            self._finish_served(r)

        # Synchronize KV allocations immediately so the next scheduling pass
        # sees exact KV usage after token generation.
        self._reserve_running_kv()
        self._last_iter_kind = "decode"
        self._record(start, "decode", len(contexts), sum(contexts), qlen)

    def _idle(self) -> None:
        start = self.now
        qlen = len(self.queue) + len(self._preempted_queue)
        candidates_times: List[float] = []
        if self._has_pending_arrivals():
            candidates_times.append(self._pending[self._next].arrival_time)
        for r in self.queue:
            cooldown = self._defer_until.get(r.request_id)
            if cooldown is not None and cooldown > self.now:
                candidates_times.append(cooldown)
        if candidates_times:
            target = min(candidates_times)
        else:
            target = self.now + self.defer_retry_s
        self.now = max(self.now + 1e-12, target)
        self._last_iter_kind = "idle"
        self._record(start, "idle", 0, 0, qlen)


# ---------------------------------------------------------------------------
# PolicyExecutor adapter for evaluation-engine integration
# ---------------------------------------------------------------------------

class SimulatorExecutor:
    """Adapter implementing ``PolicyExecutor`` (``run(policy_name, scheduler)``)
    so ``evaluate_policy_matrix`` can run any ``Scheduler`` on this simulator
    and collect the resulting :class:`LifecycleEvent` stream."""

    def __init__(
        self,
        requests: Sequence[object],
        hardware: Optional[HardwareConfig] = None,
        profile: Optional[ServiceProfile] = None,
        *,
        defer_retry_s: float = 0.01,
        max_defers: Optional[int] = 64,
        expire_overdue_in_queue: bool = False,
        estimated_output_tokens: Optional[float] = None,
        service_estimator: Optional[Callable[[int, float], float]] = None,
    ) -> None:
        self.requests = tuple(Request.from_any(r) for r in requests)
        self.hardware = hardware or HardwareConfig()
        self.profile = profile or ServiceProfile()
        self.defer_retry_s = defer_retry_s
        self.max_defers = max_defers
        self.expire_overdue_in_queue = expire_overdue_in_queue
        self.estimated_output_tokens = estimated_output_tokens
        self.service_estimator = service_estimator
        self.results: Dict[str, SimResult] = {}

    def run(
        self,
        policy_name: str,
        scheduler: SchedulerCallback,
    ) -> Iterable[LifecycleEvent]:
        res = simulate(
            self.requests,
            scheduler,
            self.hardware,
            self.profile,
            defer_retry_s=self.defer_retry_s,
            max_defers=self.max_defers,
            expire_overdue_in_queue=self.expire_overdue_in_queue,
            estimated_output_tokens=self.estimated_output_tokens,
            service_estimator=self.service_estimator,
            record_iterations=False,
        )
        self.results[policy_name] = res
        return res.events


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    workload: Sequence[object],
    callback: SchedulerCallback,
    hardware: Optional[HardwareConfig] = None,
    profile: Optional[ServiceProfile] = None,
    *,
    defer_retry_s: float = 0.01,
    max_defers: Optional[int] = 64,
    expire_overdue_in_queue: bool = False,
    estimated_output_tokens: Optional[float] = None,
    service_estimator: Optional[Callable[[int, float], float]] = None,
    record_iterations: bool = True,
) -> SimResult:
    """Run one simulation episode to completion.

    Parameters
    ----------
    workload:
        Requests in any order (either :class:`~llm_triage.simulation.request.Request` or
        duck-typed ``llm_triage`` request objects).  Copied internally so the
        input objects are never mutated.
    callback:
        Scheduling policy implementing ``decide(state) -> Action | Decision``
        (and optionally ``priority(view) -> float``).
    hardware:
        GPU constraints.  Defaults to :class:`HardwareConfig`.
    profile:
        Timing model.  Defaults to the RTX 5070 / Qwen2.5-0.5B calibration.
    defer_retry_s:
        Minimum simulation time before a deferred request is re-offered to the
        policy.
    max_defers:
        Maximum times a single request may be deferred before being dropped
        (``None`` disables the bound).
    expire_overdue_in_queue:
        If ``True``, automatically evict queued requests as ``VIOLATED`` once
        ``now >= slo_deadline``.  Defaults to ``False`` so naive policies like
        FCFS serve overdue work late unless the policy itself drops them.
    estimated_output_tokens:
        Policy-visible output token estimate exposed on each
        :class:`~llm_triage.simulation.callback.RequestView`.
    service_estimator:
        Optional ``(input_tokens, estimated_output_tokens) -> float`` function
        for policy-visible service time estimation.
    record_iterations:
        Keep a per-iteration trace in ``SimResult.iterations``.
    """
    priority_fn = getattr(callback, "priority", None)
    kernel = TriageKernel(
        hardware=hardware,
        profile=profile,
        defer_retry_s=defer_retry_s,
        max_defers=max_defers,
        expire_overdue_in_queue=expire_overdue_in_queue,
        estimated_output_tokens=estimated_output_tokens,
        service_estimator=service_estimator,
        priority_fn=priority_fn,
        record_iterations=record_iterations,
    )
    kernel.reset(workload)
    while not kernel.is_terminal():
        snap = kernel.snapshot()
        t0 = time.perf_counter()
        raw_decision = callback.decide(snap)
        overhead_s = time.perf_counter() - t0
        if raw_decision is None:
            raise RuntimeError("Scheduler returned None on a non-empty snapshot")
        if hasattr(raw_decision, "request_id") and hasattr(raw_decision, "action"):
            decision: Union[Action, Decision] = Decision(
                request_id= getattr(raw_decision, "request_id"),
                action=Action.coerce(getattr(raw_decision, "action")),
            )
        else:
            decision = Action.coerce(raw_decision)
        kernel.apply(decision, decision_overhead_s=overhead_s)
    return kernel.result()
