"""Minimal Week 1 contracts following TECH_DESIGN.md section 7.

Only Request contains environment truth. Scheduler snapshots must be constructed
from CandidateView, not from Request. Resource feasibility belongs to the kernel.
"""
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_time_s: float
    input_tokens: int
    true_output_tokens: int
    deadline_s: float


@dataclass(frozen=True)
class CandidateView:
    request_id: str
    arrival_time_s: float
    input_tokens: int
    estimated_output_tokens: float
    estimated_service_s: float
    deadline_s: float


@dataclass(frozen=True)
class ResourceView:
    active_batch_size: int = 0
    kv_cache_available: float | None = None
    prefill_utilization: float = 0.0
    decode_utilization: float = 0.0
    # Kernel checks actual resource limits; None above means not modeled.
    admission_available: bool = True


@dataclass(frozen=True)
class SchedulerSnapshot:
    now_s: float
    # Chronological enqueue order breaks tied arrival times deterministically.
    candidates: tuple[CandidateView, ...]
    resources: ResourceView


class Action(str, Enum):
    ADMIT = "admit"
    DEFER = "defer"
    DROP = "drop"


@dataclass(frozen=True)
class Decision:
    request_id: str
    action: Action
