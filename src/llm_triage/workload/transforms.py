"""Explicit timestamp alignment with optional arrival-rate scaling."""
from dataclasses import dataclass
import math

from llm_triage.types import Request


@dataclass(frozen=True)
class ReplayWindow:
    requests: tuple[Request, ...]
    horizon_s: float
    source_origin_s: float
    time_scale: float


def replay_window(rows, *, slo_s=5.0, target_rps=None):
    """Shift the first arrival to zero, preserving ties and input order.

    Native mode preserves source spacing and adds one mean interarrival interval
    after the last arrival. Scaled mode uses N / target_rps as the horizon and
    (N-1) / target_rps as the last arrival. Thus every arrival is inside [0, H).
    Deadlines are completion deadlines, not TTFT deadlines.
    """
    if not math.isfinite(slo_s) or slo_s <= 0 or len(rows) < 2:
        raise ValueError("Need a positive finite SLO and at least two requests")
    if target_rps is not None and (not math.isfinite(target_rps) or target_rps <= 0):
        raise ValueError("Target RPS must be positive and finite")
    if len({r.source_row for r in rows}) != len(rows):
        raise ValueError("Duplicate source request IDs")
    if any(not math.isfinite(r.timestamp_s) or r.timestamp_s < 0 or r.input_tokens <= 0
           or r.true_output_tokens <= 0 for r in rows):
        raise ValueError("Invalid source request")
    if any(a.timestamp_s > b.timestamp_s for a, b in zip(rows, rows[1:])):
        raise ValueError("Replay rows must be chronological")
    origin = rows[0].timestamp_s
    span = rows[-1].timestamp_s - origin
    if span <= 0:
        raise ValueError("Replay window must have positive time span")
    scale = 1.0 if target_rps is None else (len(rows) - 1) / target_rps / span
    horizon = span * len(rows) / (len(rows) - 1) if target_rps is None else len(rows) / target_rps
    if not math.isfinite(scale) or not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Replay scale or horizon overflow")
    requests = tuple(Request(str(r.source_row), (r.timestamp_s - origin) * scale,
                             r.input_tokens, r.true_output_tokens,
                             (r.timestamp_s - origin) * scale + slo_s) for r in rows)
    if any(not math.isfinite(r.deadline_s) or r.arrival_time_s >= horizon for r in requests):
        raise ValueError("Replay deadlines or arrival bounds overflow")
    return ReplayWindow(requests, horizon, origin, scale)
