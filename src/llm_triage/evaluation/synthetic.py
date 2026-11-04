"""Deterministic reference executor for end-to-end development and demos.

This is intentionally a small single-server model, not the high-fidelity
simulator used for final claims. It exercises the same scheduler, lifecycle,
metrics, and reporting interfaces that the real simulator adapter will use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

from llm_triage.metrics import EventType, LifecycleEvent
from llm_triage.scheduling import (
    Action,
    CandidateView,
    ResourceView,
    Scheduler,
    SchedulerSnapshot,
)


@dataclass(frozen=True)
class SyntheticRequest:
    """Environment truth for a synthetic request."""

    request_id: str
    arrival_time_s: float
    input_tokens: int
    true_output_tokens: int
    deadline_s: float


@dataclass(frozen=True)
class SyntheticConfig:
    duration_s: float = 30.0
    request_count: int = 60
    seed: int = 42
    slo_s: float = 3.0
    burst_multiplier: float = 3.0
    prefill_tokens_per_s: float = 2000.0
    decode_tokens_per_s: float = 400.0
    estimated_output_tokens: float = 144.0
    defer_seconds: float = 0.25
    max_defer_decisions: int = 10_000

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "slo_s",
            "burst_multiplier",
            "prefill_tokens_per_s",
            "decode_tokens_per_s",
            "estimated_output_tokens",
            "defer_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("request_count", "max_defer_decisions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def generate_synthetic_workload(
    config: SyntheticConfig,
) -> tuple[SyntheticRequest, ...]:
    """Generate a paired workload with a concentrated middle burst."""

    rng = random.Random(config.seed)
    base_interval = config.duration_s / config.request_count
    now = 0.0
    requests: list[SyntheticRequest] = []
    burst_start = config.request_count // 3
    burst_end = 2 * config.request_count // 3

    for index in range(config.request_count):
        if index:
            in_burst = burst_start <= index < burst_end
            multiplier = config.burst_multiplier if in_burst else 1.0
            jitter = rng.uniform(0.75, 1.25)
            now += base_interval * jitter / multiplier
        # The final clamp protects very small request counts with large jitter.
        arrival = min(now, math.nextafter(config.duration_s, 0.0))
        requests.append(
            SyntheticRequest(
                request_id=f"synthetic-{index:04d}",
                arrival_time_s=arrival,
                input_tokens=rng.randint(128, 1024),
                true_output_tokens=rng.randint(32, 256),
                deadline_s=arrival + config.slo_s,
            )
        )
    return tuple(requests)


class SyntheticExecutor:
    """Run schedulers against a deterministic, non-preemptive single server."""

    def __init__(
        self,
        requests: Iterable[SyntheticRequest],
        config: SyntheticConfig,
    ) -> None:
        self.requests = tuple(
            sorted(requests, key=lambda row: (row.arrival_time_s, row.request_id))
        )
        self.config = config
        self._validate_requests()

    def run(
        self, policy_name: str, scheduler: Scheduler
    ) -> Iterable[LifecycleEvent]:
        del policy_name
        events: list[LifecycleEvent] = []
        queue: list[SyntheticRequest] = []
        cursor = 0
        now = 0.0
        defer_decisions = 0

        def emit_arrivals_through(timestamp_s: float) -> None:
            nonlocal cursor
            while (
                cursor < len(self.requests)
                and self.requests[cursor].arrival_time_s <= timestamp_s
            ):
                request = self.requests[cursor]
                queue.append(request)
                cursor += 1
                events.append(
                    LifecycleEvent(
                        EventType.ARRIVAL,
                        request.arrival_time_s,
                        request.request_id,
                        {
                            "deadline_s": request.deadline_s,
                            "input_tokens": request.input_tokens,
                            "queue_length": len(queue),
                        },
                    )
                )

        while cursor < len(self.requests) or queue:
            if not queue:
                now = max(now, self.requests[cursor].arrival_time_s)
            emit_arrivals_through(now)
            snapshot = self._snapshot(now, queue)
            decision = scheduler.decide(snapshot)
            if decision is None:
                raise RuntimeError("Scheduler returned no decision for a nonempty queue")
            selected_index = next(
                (
                    index
                    for index, request in enumerate(queue)
                    if request.request_id == decision.request_id
                ),
                None,
            )
            if selected_index is None:
                raise ValueError("Scheduler selected a request outside the active queue")
            request = queue[selected_index]
            events.append(
                LifecycleEvent(
                    EventType.DECISION,
                    now,
                    request.request_id,
                    {"action": decision.action.value, "decision_overhead_s": 0.0},
                )
            )

            if decision.action is Action.DEFER:
                defer_decisions += 1
                if defer_decisions > self.config.max_defer_decisions:
                    raise RuntimeError("Scheduler exceeded the bounded defer limit")
                end = now + self.config.defer_seconds
                emit_arrivals_through(end)
                now = end
                continue

            defer_decisions = 0
            queue.pop(selected_index)
            if decision.action is Action.DROP:
                events.append(
                    LifecycleEvent(
                        EventType.DROP,
                        now,
                        request.request_id,
                        {"queue_length": len(queue)},
                    )
                )
                continue

            if decision.action is not Action.ADMIT:
                raise ValueError(f"Unsupported scheduler action: {decision.action}")
            events.append(
                LifecycleEvent(
                    EventType.SERVICE_START,
                    now,
                    request.request_id,
                    {"queue_length": len(queue)},
                )
            )
            prefill_s = request.input_tokens / self.config.prefill_tokens_per_s
            first_token_s = now + prefill_s + 1.0 / self.config.decode_tokens_per_s
            completion_s = (
                now
                + prefill_s
                + request.true_output_tokens / self.config.decode_tokens_per_s
            )
            emit_arrivals_through(first_token_s)
            events.append(
                LifecycleEvent(
                    EventType.FIRST_TOKEN,
                    first_token_s,
                    request.request_id,
                    {"queue_length": len(queue)},
                )
            )
            emit_arrivals_through(completion_s)
            events.append(
                LifecycleEvent(
                    EventType.COMPLETION,
                    completion_s,
                    request.request_id,
                    {"queue_length": len(queue)},
                )
            )
            now = completion_s

        events.append(
            LifecycleEvent(EventType.EPISODE_END, max(now, self.config.duration_s))
        )
        return events

    def _snapshot(
        self, now_s: float, queue: list[SyntheticRequest]
    ) -> SchedulerSnapshot:
        candidates = tuple(
            CandidateView(
                request_id=request.request_id,
                arrival_time_s=request.arrival_time_s,
                input_tokens=request.input_tokens,
                estimated_output_tokens=self.config.estimated_output_tokens,
                deadline_s=request.deadline_s,
                estimated_service_s=(
                    request.input_tokens / self.config.prefill_tokens_per_s
                    + self.config.estimated_output_tokens
                    / self.config.decode_tokens_per_s
                ),
            )
            for request in queue
        )
        return SchedulerSnapshot(
            now_s=now_s,
            candidates=candidates,
            resources=ResourceView(
                can_admit=True,
                active_batch_size=0,
                kv_cache_available_tokens=None,
            ),
        )

    def _validate_requests(self) -> None:
        if len({request.request_id for request in self.requests}) != len(self.requests):
            raise ValueError("Synthetic request IDs must be unique")
        for request in self.requests:
            if not isinstance(request.request_id, str) or not request.request_id.strip():
                raise ValueError("Synthetic request IDs must be nonempty strings")
            numeric = (
                request.arrival_time_s,
                request.deadline_s,
                request.input_tokens,
                request.true_output_tokens,
            )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in numeric
            ):
                raise ValueError("Synthetic request fields must be finite numbers")
            if not 0 <= request.arrival_time_s < self.config.duration_s:
                raise ValueError("Synthetic arrival is outside the measurement window")
            if request.input_tokens <= 0 or request.true_output_tokens <= 0:
                raise ValueError("Synthetic token counts must be positive")
            if request.deadline_s < request.arrival_time_s:
                raise ValueError("Synthetic deadline precedes arrival")

