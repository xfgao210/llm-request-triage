"""Input validation helpers and request data model for the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Union


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_int(value: object, name: str, *, min_val: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    return value


def _check_finite(value: object, name: str, *, min_val: float | None = None, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if strictly_positive and fval <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    if min_val is not None and fval < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    return fval


# ---------------------------------------------------------------------------
# Scheduling action and request state
# ---------------------------------------------------------------------------

class Action(str, Enum):
    """Scheduling action returned by a callback or policy.

    Accepts string values (``"admit"``, ``"defer"``, ``"drop"``), integer
    indices (``0``, ``1``, ``2``), or foreign enum instances with matching
    string ``.value`` or ``.name``.
    """

    ADMIT = "admit"
    DEFER = "defer"
    DROP = "drop"

    @classmethod
    def coerce(cls, value: object) -> "Action":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
        if not isinstance(value, bool) and isinstance(value, int):
            mapping = {0: cls.ADMIT, 1: cls.DEFER, 2: cls.DROP}
            if value in mapping:
                return mapping[value]
        if isinstance(value, Enum):
            if isinstance(value.value, str):
                return cls.coerce(value.value)
            return cls.coerce(value.name)
        raise ValueError(f"Cannot coerce {value!r} to Action")


class RequestState(Enum):
    """Lifecycle state of a request inside the simulator."""

    QUEUED = auto()
    PREFILLING = auto()
    DECODING = auto()
    COMPLETED = auto()  # Finished within its SLO deadline
    DROPPED = auto()    # Proactively rejected by the scheduler before service
    VIOLATED = auto()   # Missed its SLO (finished late, expired, or infeasible)


TERMINAL_STATES = frozenset(
    {RequestState.COMPLETED, RequestState.DROPPED, RequestState.VIOLATED}
)


# ---------------------------------------------------------------------------
# Request data model
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """A single inference request.

    ``output_tokens`` (aliased as ``true_output_tokens``) is the *ground-truth*
    number of tokens the model will generate.  The simulator engine uses it to
    know when decoding finishes, but it is **never** exposed to scheduling
    callbacks — they only see the fields surfaced through
    :class:`~triage.callback.RequestView` / :class:`~triage.callback.CandidateView`.
    """

    request_id: Union[int, str]
    arrival_time: float
    input_tokens: int
    output_tokens: int   # Ground truth — NEVER exposed to schedulers
    slo_deadline: float  # Absolute time: arrival_time + slo_budget

    # ---- mutable state (written by the simulator engine) ----
    state: RequestState = field(default=RequestState.QUEUED)
    decision_time: float | None = field(default=None, repr=False)
    admit_time: float | None = field(default=None, repr=False)        # first admission (SERVICE_START)
    first_token_time: float | None = field(default=None, repr=False)  # TTFT = this - arrival
    completion_time: float | None = field(default=None, repr=False)   # time a terminal state was reached
    tokens_generated: int = field(default=0, repr=False)
    kv_blocks_allocated: int = field(default=0, repr=False)
    defer_count: int = field(default=0, repr=False)
    preemption_count: int = field(default=0, repr=False)
    gpu_time_s: float = field(default=0.0, repr=False)  # attributed share of GPU iteration time

    def __post_init__(self) -> None:
        if isinstance(self.request_id, bool) or not isinstance(self.request_id, (int, str)):
            raise ValueError("request_id must be an int or nonempty str")
        if isinstance(self.request_id, str) and not self.request_id.strip():
            raise ValueError("request_id string must not be empty")
        _check_finite(self.arrival_time, f"request {self.request_id} arrival_time", min_val=0.0)
        _check_int(self.input_tokens, f"request {self.request_id} input_tokens", min_val=1)
        _check_int(self.output_tokens, f"request {self.request_id} output_tokens", min_val=1)
        _check_finite(self.slo_deadline, f"request {self.request_id} slo_deadline")
        if self.slo_deadline < self.arrival_time:
            raise ValueError(f"request {self.request_id}: slo_deadline precedes arrival_time")

    # ---- compatibility aliases (TECH_DESIGN.md §7.1) ---------------------

    @property
    def arrival_time_s(self) -> float:
        return self.arrival_time

    @property
    def true_output_tokens(self) -> int:
        return self.output_tokens

    @property
    def deadline_s(self) -> float:
        return self.slo_deadline

    @classmethod
    def from_any(cls, obj: object) -> "Request":
        """Convert either a :class:`Request` or a duck-typed ``llm_triage``
        request object (with ``arrival_time_s``, ``true_output_tokens``,
        ``deadline_s``) into a fresh :class:`Request`."""
        if isinstance(obj, cls):
            return obj.fresh_copy()
        request_id = getattr(obj, "request_id")
        arrival = getattr(obj, "arrival_time", getattr(obj, "arrival_time_s", None))
        input_tokens = getattr(obj, "input_tokens")
        output_tokens = getattr(
            obj, "output_tokens", getattr(obj, "true_output_tokens", None)
        )
        deadline = getattr(obj, "slo_deadline", getattr(obj, "deadline_s", None))
        return cls(
            request_id=request_id,
            arrival_time=arrival,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            slo_deadline=deadline,
        )

    @property
    def context_tokens(self) -> int:
        """Tokens whose KV entries are currently resident in cache."""
        return self.input_tokens + self.tokens_generated

    def fresh_copy(self) -> "Request":
        """Copy with only the immutable fields, so all simulation state is reset."""
        return Request(
            request_id=self.request_id,
            arrival_time=self.arrival_time,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            slo_deadline=self.slo_deadline,
        )
