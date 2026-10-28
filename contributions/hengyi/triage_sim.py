"""Dependency-free queue prototype. Not a calibrated LLM serving simulator."""
from dataclasses import dataclass
from enum import Enum
import heapq
import math
import random


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival: float
    input_tokens: int
    output_tokens: int  # Environment only; never passed to a policy.
    deadline: float


@dataclass(frozen=True)
class Observation:
    now: float
    request_id: int
    input_tokens: int
    estimated_output_tokens: float
    slack: float
    queue_length: int
    server_busy: bool
    estimated_service_s: float


class Action(str, Enum):
    ADMIT = "admit"
    DEFER = "defer"
    DROP = "drop"


def fcfs(observation):
    return Action.DEFER if observation.server_busy else Action.ADMIT


def expired_only(observation):
    """Drop only already-expired queue heads, at dispatch time."""
    return Action.DROP if observation.slack <= 0 else fcfs(observation)


def static_slo(observation):
    """Use visible prompt length and a fixed output estimate, never the true length."""
    if observation.slack < observation.estimated_service_s:
        return Action.DROP
    return fcfs(observation)


def generate_workload(seed=42, duration=120.0, base_rate=1.2, burst_multiplier=3.0, slo=5.0):
    if any(not math.isfinite(x) or x <= 0 for x in [duration, base_rate, burst_multiplier, slo]):
        raise ValueError("Duration, rate, multiplier and SLO must be positive")
    # Separate streams keep arrival times independent of token sampling.
    arrivals = random.Random(seed)
    lengths = random.Random(seed + 1)
    requests = []
    for start, end, rate in [(0, duration / 3, base_rate),
                             (duration / 3, 2 * duration / 3, base_rate * burst_multiplier),
                             (2 * duration / 3, duration, base_rate)]:
        now = start
        while True:
            now += arrivals.expovariate(rate)
            if now >= end:
                break
            requests.append(Request(len(requests), now, lengths.randint(256, 1024),
                                    lengths.randint(32, 256), now + slo))
    return requests


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def simulate(requests, policy=fcfs, horizon=120.0, prefill_rate=2000.0, decode_rate=400.0,
             estimated_output_tokens=144.0):
    if any(not math.isfinite(x) or x <= 0 for x in [horizon, prefill_rate, decode_rate, estimated_output_tokens]):
        raise ValueError("Horizon and token rates must be positive")
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("Request IDs must be unique")
    if any(any(not math.isfinite(v) for v in [r.arrival, r.deadline, r.input_tokens, r.output_tokens]) or
           r.arrival < 0 or r.arrival >= horizon or r.input_tokens <= 0 or
           r.output_tokens <= 0 or r.deadline < r.arrival for r in requests):
        raise ValueError("Invalid request or arrival outside measurement window")
    records = {r.request_id: dict(request_id=r.request_id, arrival=r.arrival,
               input_tokens=r.input_tokens, output_tokens=r.output_tokens,
               deadline=r.deadline, start=None, completion=None,
               latency=None, wait=None, decision_time=None,
               status="pending", slo_met=False) for r in requests}
    events = [(r.arrival, 1, r.request_id) for r in requests]
    heapq.heapify(events)
    lookup = {r.request_id: r for r in requests}
    from collections import deque
    queue = deque()
    busy = False
    trace = [dict(time=0.0, queue_length=0, server_busy=0)]
    last_event = 0.0
    while events:
        now = events[0][0]
        # Resolve all simultaneous events before dispatching.
        while events and events[0][0] == now:
            _, kind, rid = heapq.heappop(events)
            r = lookup[rid]
            rec = records[rid]
            if kind == 0:
                busy = False
                rec.update(completion=now, latency=now-r.arrival,
                           status="completed", slo_met=now <= r.deadline)
            else:
                queue.append(rid)
        while queue and not busy:
            rid = queue[0]
            r = lookup[rid]
            observation = Observation(now, rid, r.input_tokens, estimated_output_tokens,
                                      r.deadline-now, len(queue), busy,
                                      r.input_tokens/prefill_rate + estimated_output_tokens/decode_rate)
            action = Action(policy(observation))
            if action == Action.DEFER:
                break
            queue.popleft()
            records[rid]['decision_time'] = now
            if action == Action.DROP:
                records[rid]["status"] = "dropped"
                continue
            busy = True
            records[rid].update(start=now, wait=now-r.arrival)
            completion = now + r.input_tokens/prefill_rate + r.output_tokens/decode_rate
            heapq.heappush(events, (completion, 0, rid))
        trace.append(dict(time=now, queue_length=len(queue), server_busy=int(busy)))
        last_event = now
    for rid in queue:
        records[rid]["status"] = "deferred"
    rows = list(records.values())
    return rows, trace, summarize_records(rows, trace, horizon, last_event)


def summarize_records(rows, trace, horizon, last_event):
    """Shared accounting for callback simulation and interactive environment."""
    completed = [r for r in rows if r["status"] == "completed"]
    within = [r for r in completed if r["completion"] <= horizon]
    latencies = [r["latency"] for r in completed]
    elapsed = max(horizon, last_event)
    n = len(rows)
    metrics = dict(arrivals=n, completed=len(completed),
                   dropped=sum(r["status"] == "dropped" for r in rows),
                   deferred=sum(r["status"] == "deferred" for r in rows),
                   slo_met=sum(r["slo_met"] for r in rows),
                   slo_attainment=sum(r["slo_met"] for r in rows)/n if n else 0.0,
                   drop_rate=sum(r["status"] == "dropped" for r in rows)/n if n else 0.0,
                   window_completed=len(within),
                   window_throughput_rps=len(within)/horizon,
                   window_goodput_rps=sum(r["slo_met"] for r in within)/horizon,
                   drained_throughput_rps=len(completed)/elapsed,
                   p95_latency_s=percentile(latencies, .95),
                   p99_latency_s=percentile(latencies, .99),
                   max_queue_length=max(t["queue_length"] for t in trace),
                   wasted_service_s=sum(r['completion']-r['start'] for r in completed
                                        if not r['slo_met']),
                   total_service_s=sum(r['completion']-r['start'] for r in completed),
                   measurement_window_s=horizon, drain_end_s=elapsed)
    return metrics
