"""Observation builder and ground-truth output-length masking for the RL environment.

This module implements ``TECH_DESIGN.md`` §6.5 / §7.2:
- Converts internal :class:`~llm_triage.simulation.SimState` (or
  ``SchedulerSnapshot``) objects into fixed-dimension, normalized numerical
  observation vectors for RL agents.
- Enforces strict non-leakage invariants: ground-truth ``output_tokens``
  (``true_output_tokens``) and future arrivals are never accessible in
  policy-visible views or observation features.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional, Sequence, Tuple

from llm_triage.simulation import CandidateView, RequestView, SchedulerSnapshot, SimState

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional fallback for stdlib-only environments
    np = None  # type: ignore[assignment]


ObservationPreset = Literal["full", "compact_7"]

FULL_FEATURE_NAMES: Tuple[str, ...] = (
    "candidate_slack_ratio",
    "candidate_estimated_slack_ratio",
    "candidate_estimated_service_ratio",
    "candidate_input_tokens_log",
    "candidate_wait_ratio",
    "candidate_defer_ratio",
    "candidate_fits",
    "queue_length_log",
    "queue_min_slack_ratio",
    "queue_mean_slack_ratio",
    "batch_utilization",
    "kv_cache_utilization",
    "prefill_utilization",
    "decode_utilization",
    "episode_progress",
)

COMPACT_7_FEATURE_NAMES: Tuple[str, ...] = (
    "slack/SLO",
    "estimated_service/SLO",
    "log1p(input)/10",
    "log1p(queue)/5",
    "estimated_output/1000",
    "now/horizon",
    "defers/max_defers",
)

_FORBIDDEN_TRUTH_ATTRS: Tuple[str, ...] = (
    "output_tokens",
    "true_output_tokens",
)


def _clip(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _to_float_array(values: Sequence[float]):
    """Convert a sequence of floats to a 1D float32 numpy array (or tuple fallback)."""
    if np is not None:
        return np.asarray(values, dtype=np.float32)
    return tuple(float(v) for v in values)


def assert_view_fields_only(view: object) -> None:
    """Reject any object carrying ground-truth output token attributes."""
    for attr in _FORBIDDEN_TRUTH_ATTRS:
        if hasattr(view, attr) or attr in getattr(view, "__dict__", {}):
            raise ValueError(
                f"Object {view!r} leaked forbidden ground-truth attribute {attr!r}"
            )


def assert_no_ground_truth_leakage(snapshot: object) -> None:
    """Verify that *snapshot* exposes neither ground-truth output lengths nor
    future un-arrived requests.

    Works on both :class:`~llm_triage.simulation.SimState` and generic
    ``SchedulerSnapshot`` objects.
    """
    for attr in _FORBIDDEN_TRUTH_ATTRS:
        if hasattr(snapshot, attr):
            raise ValueError(f"Snapshot leaked forbidden ground-truth attribute {attr!r}")

    now_s = float(getattr(snapshot, "now_s", getattr(snapshot, "now", 0.0)))
    views_list = []
    if hasattr(snapshot, "candidate") and getattr(snapshot, "candidate") is not None:
        views_list.append(getattr(snapshot, "candidate"))
    if hasattr(snapshot, "queue_requests"):
        views_list.extend(getattr(snapshot, "queue_requests"))
    elif hasattr(snapshot, "candidates"):
        views_list.extend(getattr(snapshot, "candidates"))
    if hasattr(snapshot, "active_requests"):
        views_list.extend(getattr(snapshot, "active_requests"))

    for view in views_list:
        assert_view_fields_only(view)
        arrival = float(getattr(view, "arrival_time_s", getattr(view, "arrival_time", 0.0)))
        if arrival > now_s + 1e-9:
            raise ValueError(
                f"RequestView {getattr(view, 'request_id', None)!r} has future "
                f"arrival_time_s {arrival} > now_s {now_s}"
            )


def sanitize_request_view(view: RequestView) -> RequestView:
    """Return a clean :class:`RequestView` containing only policy-visible fields."""
    assert_view_fields_only(view)
    if type(view) is RequestView:
        return view
    return RequestView(
        request_id=view.request_id,
        arrival_time=float(getattr(view, "arrival_time", getattr(view, "arrival_time_s", 0.0))),
        input_tokens=int(view.input_tokens),
        slo_deadline=float(getattr(view, "slo_deadline", getattr(view, "deadline_s", 0.0))),
        slo_slack=float(getattr(view, "slo_slack", 0.0)),
        time_in_queue=float(getattr(view, "time_in_queue", 0.0)),
        state=getattr(view, "state", None),  # type: ignore[arg-type]
        tokens_generated=int(getattr(view, "tokens_generated", 0)),
        defer_count=int(getattr(view, "defer_count", 0)),
        preemption_count=int(getattr(view, "preemption_count", 0)),
        estimated_output_tokens=float(view.estimated_output_tokens),
        estimated_service_s=float(view.estimated_service_s),
    )


def sanitize_snapshot(snapshot: SimState) -> SimState:
    """Validate and return a sanitized :class:`SimState` with ground-truth checks."""
    assert_no_ground_truth_leakage(snapshot)
    if (
        type(snapshot) is SimState
        and type(snapshot.candidate) is RequestView
        and all(type(v) is RequestView for v in snapshot.queue_requests)
        and all(type(v) is RequestView for v in snapshot.active_requests)
    ):
        return snapshot
    return SimState(
        now=float(snapshot.now),
        candidate=sanitize_request_view(snapshot.candidate),
        queue_length=int(snapshot.queue_length),
        queue_requests=tuple(sanitize_request_view(v) for v in snapshot.queue_requests),
        active_count=int(snapshot.active_count),
        active_requests=tuple(sanitize_request_view(v) for v in snapshot.active_requests),
        pending_prefill_count=int(snapshot.pending_prefill_count),
        pending_prefill_tokens=int(snapshot.pending_prefill_tokens),
        max_batch_size=int(snapshot.max_batch_size),
        batch_utilization=float(snapshot.batch_utilization),
        kv_block_size=int(snapshot.kv_block_size),
        kv_blocks_used=int(snapshot.kv_blocks_used),
        kv_blocks_total=int(snapshot.kv_blocks_total),
        kv_utilization=float(snapshot.kv_utilization),
        completed_count=int(snapshot.completed_count),
        violated_count=int(snapshot.violated_count),
        dropped_count=int(snapshot.dropped_count),
        preempted_count=int(snapshot.preempted_count),
    )


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for converting a :class:`SimState` into a feature vector."""

    preset: ObservationPreset = "full"
    horizon_s: float = 120.0
    max_defers: int = 64
    min_slo_s: float = 1e-3
    slack_clip_min: float = -5.0
    slack_clip_max: float = 2.0
    service_clip_max: float = 10.0
    wait_clip_max: float = 5.0
    horizon_clip_max: float = 5.0
    log_token_scale: float = 10.0
    log_queue_scale: float = 5.0

    def __post_init__(self) -> None:
        if self.preset not in ("full", "compact_7"):
            raise ValueError(f"preset must be 'full' or 'compact_7', got {self.preset!r}")
        for name in (
            "horizon_s",
            "min_slo_s",
            "service_clip_max",
            "wait_clip_max",
            "horizon_clip_max",
            "log_token_scale",
            "log_queue_scale",
        ):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be a finite positive number, got {val!r}")
        if isinstance(self.max_defers, bool) or not isinstance(self.max_defers, int) or self.max_defers < 1:
            raise ValueError(f"max_defers must be a positive int, got {self.max_defers!r}")
        if not math.isfinite(self.slack_clip_min) or not math.isfinite(self.slack_clip_max):
            raise ValueError("slack_clip_min and slack_clip_max must be finite")
        if self.slack_clip_min >= self.slack_clip_max:
            raise ValueError("slack_clip_min must be < slack_clip_max")


class ObservationBuilder:
    """Builds fixed-size policy-visible observation vectors from :class:`SimState`
    or ``SchedulerSnapshot``."""

    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config = config or ObservationConfig()
        if self.config.preset == "full":
            self._feature_names = FULL_FEATURE_NAMES
            self._low = (
                self.config.slack_clip_min,  # candidate_slack_ratio
                self.config.slack_clip_min,  # candidate_estimated_slack_ratio
                0.0,                         # candidate_estimated_service_ratio
                0.0,                         # candidate_input_tokens_log
                0.0,                         # candidate_wait_ratio
                0.0,                         # candidate_defer_ratio
                0.0,                         # candidate_fits
                0.0,                         # queue_length_log
                self.config.slack_clip_min,  # queue_min_slack_ratio
                self.config.slack_clip_min,  # queue_mean_slack_ratio
                0.0,                         # batch_utilization
                0.0,                         # kv_cache_utilization
                0.0,                         # prefill_utilization
                0.0,                         # decode_utilization
                0.0,                         # episode_progress
            )
            self._high = (
                self.config.slack_clip_max,    # candidate_slack_ratio
                self.config.slack_clip_max,    # candidate_estimated_slack_ratio
                self.config.service_clip_max,  # candidate_estimated_service_ratio
                5.0,                           # candidate_input_tokens_log
                self.config.wait_clip_max,     # candidate_wait_ratio
                1.0,                           # candidate_defer_ratio
                1.0,                           # candidate_fits
                5.0,                           # queue_length_log
                self.config.slack_clip_max,    # queue_min_slack_ratio
                self.config.slack_clip_max,    # queue_mean_slack_ratio
                1.0,                           # batch_utilization
                1.0,                           # kv_cache_utilization
                1.0,                           # prefill_utilization
                1.0,                           # decode_utilization
                self.config.horizon_clip_max,  # episode_progress
            )
        else:
            self._feature_names = COMPACT_7_FEATURE_NAMES
            self._low = (
                self.config.slack_clip_min,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            self._high = (
                self.config.slack_clip_max,
                self.config.service_clip_max,
                5.0,
                5.0,
                10.0,
                self.config.horizon_clip_max,
                1.0,
            )

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Ordered names of features in the observation vector."""
        return self._feature_names

    @property
    def dim(self) -> int:
        """Dimensionality of the observation vector."""
        return len(self._feature_names)

    @property
    def low(self):
        """Lower bounds per feature as a float32 array."""
        return _to_float_array(self._low)

    @property
    def high(self):
        """Upper bounds per feature as a float32 array."""
        return _to_float_array(self._high)

    def zeros(self):
        """Return an all-zero terminal observation vector."""
        return _to_float_array((0.0,) * self.dim)

    def _slo_budget(self, view: object) -> float:
        arrival = float(getattr(view, "arrival_time_s", getattr(view, "arrival_time", 0.0)))
        deadline = float(getattr(view, "deadline_s", getattr(view, "slo_deadline", arrival + 1.0)))
        return max(self.config.min_slo_s, deadline - arrival)

    def _raw_slack(self, view: object, now_s: float) -> float:
        if hasattr(view, "slo_slack"):
            return float(getattr(view, "slo_slack"))
        deadline = float(getattr(view, "deadline_s", getattr(view, "slo_deadline", now_s)))
        return deadline - now_s

    def _slack_ratio(self, view: object, now_s: float) -> float:
        budget = self._slo_budget(view)
        return _clip(
            self._raw_slack(view, now_s) / budget,
            self.config.slack_clip_min,
            self.config.slack_clip_max,
        )

    def build(
        self,
        snapshot: Optional[SchedulerSnapshot],
        *,
        candidate: Optional[CandidateView] = None,
    ):
        """Convert *snapshot* into a 1D float32 observation vector.

        Parameters
        ----------
        snapshot:
            A :class:`~llm_triage.simulation.SimState` or ``SchedulerSnapshot``
            (or ``None`` when the episode is terminal).
        candidate:
            Optional candidate override (used when composed inside
            ``TriagePolicy.act(candidate, snapshot)``). Defaults to
            ``snapshot.candidate`` or ``snapshot.candidates[0]``.
        """
        if snapshot is None:
            return self.zeros()

        assert_no_ground_truth_leakage(snapshot)
        if candidate is not None:
            assert_view_fields_only(candidate)

        now_s = float(getattr(snapshot, "now_s", getattr(snapshot, "now", 0.0)))
        queue_views = getattr(
            snapshot,
            "queue_requests",
            getattr(snapshot, "candidates", ()),
        )
        cand = candidate if candidate is not None else getattr(snapshot, "candidate", None)
        if cand is None:
            if not queue_views:
                return self.zeros()
            cand = queue_views[0]

        cfg = self.config
        slo_budget = self._slo_budget(cand)
        raw_slack = self._raw_slack(cand, now_s)
        est_service_s = float(getattr(cand, "estimated_service_s", 0.1))
        est_out_tokens = float(getattr(cand, "estimated_output_tokens", 144.0))
        inp_tokens = int(getattr(cand, "input_tokens", 1))
        defer_count = int(getattr(cand, "defer_count", 0))
        arrival_s = float(getattr(cand, "arrival_time_s", getattr(cand, "arrival_time", now_s)))
        wait_s = float(getattr(cand, "time_in_queue", max(0.0, now_s - arrival_s)))
        queue_len = int(getattr(snapshot, "queue_length", len(queue_views)))

        slack_ratio = _clip(raw_slack / slo_budget, cfg.slack_clip_min, cfg.slack_clip_max)
        est_service_ratio = _clip(est_service_s / slo_budget, 0.0, cfg.service_clip_max)
        input_log = _clip(math.log1p(inp_tokens) / cfg.log_token_scale, 0.0, 5.0)
        queue_log = _clip(math.log1p(queue_len) / cfg.log_queue_scale, 0.0, 5.0)
        defer_ratio = _clip(defer_count / max(1, cfg.max_defers), 0.0, 1.0)
        progress = _clip(now_s / cfg.horizon_s, 0.0, cfg.horizon_clip_max)

        if cfg.preset == "compact_7":
            est_out_scaled = _clip(est_out_tokens / 1000.0, 0.0, 10.0)
            return _to_float_array(
                (
                    slack_ratio,
                    est_service_ratio,
                    input_log,
                    queue_log,
                    est_out_scaled,
                    progress,
                    defer_ratio,
                )
            )

        if hasattr(cand, "estimated_slack_s"):
            est_slack_s = float(cand.estimated_slack_s(now_s))
        else:
            est_slack_s = raw_slack - est_service_s
        est_slack_ratio = _clip(
            est_slack_s / slo_budget,
            cfg.slack_clip_min,
            cfg.slack_clip_max,
        )
        wait_ratio = _clip(wait_s / slo_budget, 0.0, cfg.wait_clip_max)

        res = getattr(snapshot, "resources", None)
        if hasattr(snapshot, "candidate_fits"):
            fits = 1.0 if snapshot.candidate_fits(cand) else 0.0  # type: ignore[arg-type]
        elif res is not None:
            fits = 1.0 if bool(getattr(res, "can_admit", True)) else 0.0
        else:
            fits = 1.0

        if queue_views:
            queue_slacks = [self._slack_ratio(v, now_s) for v in queue_views]
            q_min_slack = min(queue_slacks)
            q_mean_slack = sum(queue_slacks) / len(queue_slacks)
        else:
            q_min_slack = slack_ratio
            q_mean_slack = slack_ratio

        batch_util = _clip(float(getattr(snapshot, "batch_utilization", 0.0)), 0.0, 1.0)
        kv_util = _clip(float(getattr(snapshot, "kv_utilization", 0.0)), 0.0, 1.0)
        prefill_util = _clip(float(getattr(res, "prefill_utilization", 0.0)) if res is not None else 0.0, 0.0, 1.0)
        decode_util = _clip(float(getattr(res, "decode_utilization", 0.0)) if res is not None else 0.0, 0.0, 1.0)

        return _to_float_array(
            (
                slack_ratio,
                est_slack_ratio,
                est_service_ratio,
                input_log,
                wait_ratio,
                defer_ratio,
                fits,
                queue_log,
                q_min_slack,
                q_mean_slack,
                batch_util,
                kv_util,
                prefill_util,
                decode_util,
                progress,
            )
        )


__all__ = [
    "COMPACT_7_FEATURE_NAMES",
    "FULL_FEATURE_NAMES",
    "ObservationBuilder",
    "ObservationConfig",
    "ObservationPreset",
    "assert_no_ground_truth_leakage",
    "assert_view_fields_only",
    "sanitize_request_view",
    "sanitize_snapshot",
]
