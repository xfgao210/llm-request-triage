"""Policy-independent summary metrics for request lifecycle records."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable, Sequence

from .schemas import RequestRecord, RequestStatus


def summarize_records(
    records: Iterable[RequestRecord],
    *,
    measurement_window_s: float,
    window_start_s: float = 0.0,
    max_queue_length: int | None = None,
    decision_overheads_s: Sequence[float] = (),
    episode_end_s: float | None = None,
) -> dict[str, float | int | None]:
    """Validate request records and calculate one policy-run summary.

    The input cohort must contain requests arriving inside the half-open
    measurement window ``[window_start_s, window_start_s +
    measurement_window_s)``. Dropped and unfinished requests remain in the SLO
    denominator.
    """

    duration = _finite_positive(measurement_window_s, "measurement_window_s")
    window_start = _finite_nonnegative(window_start_s, "window_start_s")
    window_end = window_start + duration
    rows = tuple(records)
    _validate_records(rows, window_start, window_end)
    _validate_diagnostics(max_queue_length, decision_overheads_s, episode_end_s)

    completed = tuple(row for row in rows if row.status is RequestStatus.COMPLETED)
    dropped = tuple(row for row in rows if row.status is RequestStatus.DROPPED)
    unfinished = tuple(row for row in rows if row.status is RequestStatus.UNFINISHED)
    on_time = tuple(row for row in completed if row.slo_met)
    within_window = tuple(
        row
        for row in completed
        if row.completion_time_s is not None
        and window_start <= row.completion_time_s < window_end
    )
    good_within_window = tuple(row for row in within_window if row.slo_met)

    ttfts = tuple(row.ttft_s for row in rows if row.ttft_s is not None)
    completion_latencies = tuple(
        row.completion_latency_s
        for row in completed
        if row.completion_latency_s is not None
    )
    service_times = tuple(
        row.service_time_s for row in completed if row.service_time_s is not None
    )
    wasted_service_times = tuple(
        row.service_time_s
        for row in completed
        if not row.slo_met and row.service_time_s is not None
    )
    overheads = tuple(float(value) for value in decision_overheads_s)
    n = len(rows)

    summary: dict[str, float | int | None] = {
        "arrivals": n,
        "completed": len(completed),
        "dropped": len(dropped),
        "unfinished": len(unfinished),
        "slo_met": len(on_time),
        "slo_attainment": _rate(len(on_time), n),
        "drop_rate": _rate(len(dropped), n),
        "unfinished_rate": _rate(len(unfinished), n),
        "first_token_count": len(ttfts),
        "mean_ttft_s": _mean(ttfts),
        "p50_ttft_s": _percentile(ttfts, 0.50),
        "p95_ttft_s": _percentile(ttfts, 0.95),
        "p99_ttft_s": _percentile(ttfts, 0.99),
        "mean_completion_latency_s": _mean(completion_latencies),
        "p50_completion_latency_s": _percentile(completion_latencies, 0.50),
        "p95_completion_latency_s": _percentile(completion_latencies, 0.95),
        "p99_completion_latency_s": _percentile(completion_latencies, 0.99),
        "window_completed": len(within_window),
        "window_throughput_rps": len(within_window) / duration,
        "window_goodput_rps": len(good_within_window) / duration,
        "total_service_s": sum(service_times),
        "wasted_service_s": sum(wasted_service_times),
        "max_queue_length": max_queue_length,
        "scheduler_decisions": len(overheads),
        "total_scheduler_overhead_s": sum(overheads),
        "mean_scheduler_overhead_ms": (
            1000.0 * fmean(overheads) if overheads else None
        ),
        "p95_scheduler_overhead_ms": (
            1000.0 * _percentile(overheads, 0.95) if overheads else None
        ),
        "measurement_window_s": duration,
        "window_start_s": window_start,
        "episode_end_s": episode_end_s,
    }
    return summary


def _validate_records(
    records: Sequence[RequestRecord], window_start: float, window_end: float
) -> None:
    seen: set[str] = set()
    for row in records:
        if not isinstance(row.request_id, str) or not row.request_id.strip():
            raise ValueError("request_id must be a nonempty string")
        if row.request_id in seen:
            raise ValueError(f"Duplicate request_id {row.request_id!r}")
        seen.add(row.request_id)

        arrival = _finite_nonnegative(row.arrival_time_s, "arrival_time_s")
        deadline = _finite_number(row.deadline_s, "deadline_s")
        if not window_start <= arrival < window_end:
            raise ValueError(
                f"Request {row.request_id!r} arrived outside the measurement window"
            )
        if deadline < arrival:
            raise ValueError("deadline_s must not precede arrival_time_s")
        if row.input_tokens is not None:
            if isinstance(row.input_tokens, bool) or not isinstance(
                row.input_tokens, int
            ):
                raise ValueError("input_tokens must be an integer")
            if row.input_tokens <= 0:
                raise ValueError("input_tokens must be positive")

        timestamps = [
            row.decision_time_s,
            row.service_start_time_s,
            row.first_token_time_s,
            row.completion_time_s,
        ]
        for timestamp in timestamps:
            if timestamp is not None:
                value = _finite_nonnegative(timestamp, "request timestamp")
                if value < arrival:
                    raise ValueError("Request timestamp precedes arrival")

        if (
            row.decision_time_s is not None
            and row.service_start_time_s is not None
            and row.service_start_time_s < row.decision_time_s
        ):
            raise ValueError("service_start_time_s precedes decision_time_s")
        if (
            row.first_token_time_s is not None
            and row.service_start_time_s is None
        ):
            raise ValueError("first_token_time_s requires service_start_time_s")
        if (
            row.first_token_time_s is not None
            and row.service_start_time_s is not None
            and row.first_token_time_s < row.service_start_time_s
        ):
            raise ValueError("first_token_time_s precedes service_start_time_s")
        if (
            row.completion_time_s is not None
            and row.service_start_time_s is None
        ):
            raise ValueError("completion_time_s requires service_start_time_s")
        if (
            row.completion_time_s is not None
            and row.service_start_time_s is not None
            and row.completion_time_s < row.service_start_time_s
        ):
            raise ValueError("completion_time_s precedes service_start_time_s")
        if (
            row.first_token_time_s is not None
            and row.completion_time_s is not None
            and row.completion_time_s < row.first_token_time_s
        ):
            raise ValueError("completion_time_s precedes first_token_time_s")

        if row.status is RequestStatus.COMPLETED:
            if row.completion_time_s is None:
                raise ValueError("Completed requests require completion_time_s")
        elif row.status in (RequestStatus.DROPPED, RequestStatus.UNFINISHED):
            if row.completion_time_s is not None:
                raise ValueError(
                    "Dropped or unfinished requests cannot have completion_time_s"
                )
        else:
            raise ValueError(f"Unknown request status: {row.status!r}")


def _validate_diagnostics(
    max_queue_length: int | None,
    decision_overheads_s: Sequence[float],
    episode_end_s: float | None,
) -> None:
    if max_queue_length is not None:
        if isinstance(max_queue_length, bool) or not isinstance(
            max_queue_length, int
        ):
            raise ValueError("max_queue_length must be an integer")
        if max_queue_length < 0:
            raise ValueError("max_queue_length must be nonnegative")
    for overhead in decision_overheads_s:
        _finite_nonnegative(overhead, "decision_overhead_s")
    if episode_end_s is not None:
        _finite_nonnegative(episode_end_s, "episode_end_s")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


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

