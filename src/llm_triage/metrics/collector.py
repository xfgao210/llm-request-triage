"""Build final request records from a chronological lifecycle event stream."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .schemas import (
    CollectedRun,
    EvaluationResult,
    EventType,
    LifecycleEvent,
    RequestRecord,
    RequestStatus,
)
from .summary import summarize_records


@dataclass
class _MutableRecord:
    request_id: str
    arrival_time_s: float
    deadline_s: float
    input_tokens: int | None
    decision_time_s: float | None = None
    service_start_time_s: float | None = None
    first_token_time_s: float | None = None
    completion_time_s: float | None = None
    status: RequestStatus | None = None

    def freeze(self) -> RequestRecord:
        if self.status is None:
            raise RuntimeError(f"Request {self.request_id!r} is not terminal")
        return RequestRecord(
            request_id=self.request_id,
            arrival_time_s=self.arrival_time_s,
            deadline_s=self.deadline_s,
            input_tokens=self.input_tokens,
            decision_time_s=self.decision_time_s,
            service_start_time_s=self.service_start_time_s,
            first_token_time_s=self.first_token_time_s,
            completion_time_s=self.completion_time_s,
            status=self.status,
        )


class RequestCollector:
    """Collect a validated, chronological lifecycle event stream."""

    def __init__(self) -> None:
        self._records: dict[str, _MutableRecord] = {}
        self._last_timestamp_s = -math.inf
        self._episode_end_s: float | None = None
        self._max_queue_length: int | None = None
        self._decision_overheads_s: list[float] = []

    def record(self, event: LifecycleEvent) -> None:
        """Apply one event, rejecting inconsistent lifecycle transitions."""

        timestamp = _finite_nonnegative(event.timestamp_s, "timestamp_s")
        if self._episode_end_s is not None:
            raise ValueError("Cannot record events after EPISODE_END")
        if timestamp < self._last_timestamp_s:
            raise ValueError("Lifecycle events must be chronological")
        self._last_timestamp_s = timestamp
        self._record_diagnostics(event)

        if event.event_type is EventType.EPISODE_END:
            if event.request_id is not None:
                raise ValueError("EPISODE_END must not identify a request")
            self._finish_pending(timestamp)
            self._episode_end_s = timestamp
            return

        request_id = _required_request_id(event)
        if event.event_type is EventType.ARRIVAL:
            self._arrival(request_id, timestamp, event)
            return

        record = self._records.get(request_id)
        if record is None:
            raise ValueError(f"Event for unknown request {request_id!r}")
        if record.status is not None:
            raise ValueError(f"Request {request_id!r} is already terminal")

        if event.event_type is EventType.DECISION:
            record.decision_time_s = timestamp
        elif event.event_type is EventType.SERVICE_START:
            if record.service_start_time_s is not None:
                raise ValueError(f"Request {request_id!r} started service twice")
            record.service_start_time_s = timestamp
        elif event.event_type is EventType.FIRST_TOKEN:
            if record.service_start_time_s is None:
                raise ValueError("FIRST_TOKEN requires SERVICE_START")
            if record.first_token_time_s is not None:
                raise ValueError(f"Request {request_id!r} emitted FIRST_TOKEN twice")
            record.first_token_time_s = timestamp
        elif event.event_type is EventType.COMPLETION:
            if record.service_start_time_s is None:
                raise ValueError("COMPLETION requires SERVICE_START")
            record.completion_time_s = timestamp
            record.status = RequestStatus.COMPLETED
        elif event.event_type is EventType.DROP:
            if record.service_start_time_s is not None:
                raise ValueError("A request cannot be dropped after service starts")
            record.status = RequestStatus.DROPPED
        else:  # pragma: no cover - defensive against future enum additions
            raise ValueError(f"Unsupported event type: {event.event_type}")

    def collect(self, episode_end_s: float | None = None) -> CollectedRun:
        """Finalize pending requests and return immutable records.

        ``episode_end_s`` is required when the event stream did not contain an
        explicit ``EPISODE_END`` event.
        """

        if self._episode_end_s is None:
            if episode_end_s is None:
                raise ValueError(
                    "episode_end_s is required without an EPISODE_END event"
                )
            end = _finite_nonnegative(episode_end_s, "episode_end_s")
            if end < self._last_timestamp_s:
                raise ValueError("episode_end_s precedes the last lifecycle event")
            self._finish_pending(end)
            self._episode_end_s = end
        elif episode_end_s is not None and episode_end_s != self._episode_end_s:
            raise ValueError("episode_end_s conflicts with EPISODE_END")

        records = tuple(
            record.freeze()
            for record in sorted(
                self._records.values(),
                key=lambda row: (row.arrival_time_s, row.request_id),
            )
        )
        return CollectedRun(
            request_records=records,
            episode_end_s=self._episode_end_s,
            max_queue_length=self._max_queue_length,
            decision_overheads_s=tuple(self._decision_overheads_s),
        )

    def _arrival(
        self,
        request_id: str,
        timestamp: float,
        event: LifecycleEvent,
    ) -> None:
        if request_id in self._records:
            raise ValueError(f"Duplicate arrival for request {request_id!r}")
        try:
            raw_deadline = event.details["deadline_s"]
        except KeyError as error:
            raise ValueError("ARRIVAL requires details['deadline_s']") from error
        deadline = _finite_number(raw_deadline, "deadline_s")
        if deadline < timestamp:
            raise ValueError("deadline_s must not precede arrival")

        input_tokens: int | None = None
        if "input_tokens" in event.details:
            raw_tokens = event.details["input_tokens"]
            if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int):
                raise ValueError("input_tokens must be an integer")
            if raw_tokens <= 0:
                raise ValueError("input_tokens must be positive")
            input_tokens = raw_tokens

        self._records[request_id] = _MutableRecord(
            request_id=request_id,
            arrival_time_s=timestamp,
            deadline_s=deadline,
            input_tokens=input_tokens,
        )

    def _record_diagnostics(self, event: LifecycleEvent) -> None:
        if "queue_length" in event.details:
            raw_length = event.details["queue_length"]
            if isinstance(raw_length, bool) or not isinstance(raw_length, int):
                raise ValueError("queue_length must be an integer")
            if raw_length < 0:
                raise ValueError("queue_length must be nonnegative")
            if self._max_queue_length is None:
                self._max_queue_length = raw_length
            else:
                self._max_queue_length = max(self._max_queue_length, raw_length)

        if "decision_overhead_s" in event.details:
            if event.event_type is not EventType.DECISION:
                raise ValueError("decision_overhead_s is valid only on DECISION")
            overhead = _finite_nonnegative(
                event.details["decision_overhead_s"], "decision_overhead_s"
            )
            self._decision_overheads_s.append(overhead)

    def _finish_pending(self, timestamp_s: float) -> None:
        for record in self._records.values():
            if record.status is None:
                record.status = RequestStatus.UNFINISHED


def evaluate_events(
    events: Iterable[LifecycleEvent],
    *,
    measurement_window_s: float,
    window_start_s: float = 0.0,
    episode_end_s: float | None = None,
) -> EvaluationResult:
    """Collect events and calculate a complete run summary."""

    collector = RequestCollector()
    for event in events:
        collector.record(event)
    run = collector.collect(episode_end_s=episode_end_s)
    summary = summarize_records(
        run.request_records,
        measurement_window_s=measurement_window_s,
        window_start_s=window_start_s,
        max_queue_length=run.max_queue_length,
        decision_overheads_s=run.decision_overheads_s,
        episode_end_s=run.episode_end_s,
    )
    return EvaluationResult(request_records=run.request_records, summary=summary)


def _required_request_id(event: LifecycleEvent) -> str:
    if not isinstance(event.request_id, str) or not event.request_id.strip():
        raise ValueError(f"{event.event_type.value} requires a nonempty request_id")
    return event.request_id


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

