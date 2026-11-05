"""Deterministic synthetic workloads for local smoke tests and demos."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from llm_triage.simulation import Request


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


def generate_synthetic_workload(config: SyntheticConfig) -> tuple[Request, ...]:
    """Generate a paired workload with a concentrated middle burst."""

    rng = random.Random(config.seed)
    base_interval = config.duration_s / config.request_count
    now = 0.0
    requests: list[Request] = []
    burst_start = config.request_count // 3
    burst_end = 2 * config.request_count // 3

    for index in range(config.request_count):
        if index:
            in_burst = burst_start <= index < burst_end
            multiplier = config.burst_multiplier if in_burst else 1.0
            now += base_interval * rng.uniform(0.75, 1.25) / multiplier
        arrival = min(now, math.nextafter(config.duration_s, 0.0))
        requests.append(
            Request(
                request_id=f"synthetic-{index:04d}",
                arrival_time=arrival,
                input_tokens=rng.randint(128, 1024),
                output_tokens=rng.randint(32, 256),
                slo_deadline=arrival + config.slo_s,
            )
        )
    return tuple(requests)


__all__ = ["SyntheticConfig", "generate_synthetic_workload"]
