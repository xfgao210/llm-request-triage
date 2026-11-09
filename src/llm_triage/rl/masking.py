"""Action space definitions, feasibility masking, and invalid-action handling.

This module implements ``TECH_DESIGN.md`` §6.7:
- Discrete action space of size 3: ``(Action.ADMIT, Action.DEFER, Action.DROP)``.
- Boolean action mask marking invalid actions using only policy-visible state
  (never leaking ground-truth output lengths or future arrivals).
- Configurable handling for masked actions passed to ``step(action)``:
  ``"raise"`` (default), ``"fallback"``, or ``"penalize"``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional, Sequence, Tuple

from llm_triage.rl.observation import assert_no_ground_truth_leakage, assert_view_fields_only
from llm_triage.simulation import Action, CandidateView, SchedulerSnapshot

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional fallback for stdlib-only environments
    np = None  # type: ignore[assignment]


ACTIONS: Tuple[Action, ...] = (Action.ADMIT, Action.DEFER, Action.DROP)
ACTION_TO_INDEX = {
    Action.ADMIT: 0,
    Action.DEFER: 1,
    Action.DROP: 2,
}

InvalidActionMode = Literal["raise", "fallback", "penalize"]
VALID_INVALID_ACTION_MODES: Tuple[InvalidActionMode, ...] = (
    "raise",
    "fallback",
    "penalize",
)


def _to_bool_array(values: Sequence[bool]):
    """Convert a sequence of booleans to a 1D bool numpy array (or tuple fallback)."""
    if np is not None:
        return np.asarray(values, dtype=bool)
    return tuple(bool(v) for v in values)


def _extract_candidate(snapshot: object, candidate: Optional[CandidateView]) -> Optional[object]:
    if candidate is not None:
        return candidate
    if hasattr(snapshot, "candidate") and getattr(snapshot, "candidate") is not None:
        return getattr(snapshot, "candidate")
    candidates = getattr(snapshot, "candidates", getattr(snapshot, "queue_requests", ()))
    return candidates[0] if candidates else None


def _candidate_slack_s(target: object, now_s: float) -> float:
    if hasattr(target, "slo_slack"):
        return float(getattr(target, "slo_slack"))
    deadline = float(getattr(target, "deadline_s", getattr(target, "slo_deadline", now_s)))
    return deadline - now_s


@dataclass(frozen=True)
class ActionMaskConfig:
    """Configuration for action feasibility masking.

    Parameters
    ----------
    max_defers:
        Maximum allowed defers per request before ``Action.DEFER`` is masked out
        (``None`` means unbounded).
    defer_retry_s:
        Per-request defer cooldown in seconds.
    require_positive_slack_for_defer:
        When ``True`` (default), ``Action.DEFER`` is masked out once the
        candidate's remaining ``slo_slack <= defer_retry_s``.
    """

    max_defers: int | None = 64
    defer_retry_s: float = 0.01
    require_positive_slack_for_defer: bool = True

    def __post_init__(self) -> None:
        if self.max_defers is not None:
            if isinstance(self.max_defers, bool) or not isinstance(self.max_defers, int) or self.max_defers < 0:
                raise ValueError(f"max_defers must be a nonnegative int or None, got {self.max_defers!r}")
        if (
            isinstance(self.defer_retry_s, bool)
            or not isinstance(self.defer_retry_s, (int, float))
            or not math.isfinite(self.defer_retry_s)
            or self.defer_retry_s <= 0
        ):
            raise ValueError(f"defer_retry_s must be a finite positive number, got {self.defer_retry_s!r}")


def compute_action_mask(
    snapshot: Optional[SchedulerSnapshot],
    config: Optional[ActionMaskConfig] = None,
    *,
    candidate: Optional[CandidateView] = None,
):
    """Return a boolean action mask of length 3 for ``(ADMIT, DEFER, DROP)``.

    Rules:
    - Terminal / empty state (``snapshot is None`` or no candidates): ``[False, False, False]``.
    - ``ADMIT`` (index 0): ``True`` iff the candidate fits within current batch
      and KV-cache capacity (``snapshot.candidate_fits(target)`` or
      ``snapshot.resources.can_admit``).
    - ``DEFER`` (index 1): ``True`` iff ``target.defer_count < max_defers``
      (when bounded) and, if ``require_positive_slack_for_defer`` is enabled,
      ``target.slo_slack > defer_retry_s``.
    - ``DROP`` (index 2): ``True`` whenever a candidate is present.
    """
    if snapshot is None:
        return _to_bool_array((False, False, False))

    assert_no_ground_truth_leakage(snapshot)
    if candidate is not None:
        assert_view_fields_only(candidate)

    target = _extract_candidate(snapshot, candidate)
    if target is None:
        return _to_bool_array((False, False, False))

    cfg = config or ActionMaskConfig()
    now_s = float(getattr(snapshot, "now_s", getattr(snapshot, "now", 0.0)))

    if hasattr(snapshot, "candidate_fits"):
        can_admit = bool(snapshot.candidate_fits(target))  # type: ignore[arg-type]
    elif hasattr(snapshot, "resources") and getattr(snapshot, "resources") is not None:
        can_admit = bool(getattr(snapshot.resources, "can_admit", True))
    else:
        can_admit = True

    defer_count = int(getattr(target, "defer_count", 0))
    slo_slack = _candidate_slack_s(target, now_s)

    can_defer = True
    if cfg.max_defers is not None and defer_count >= cfg.max_defers:
        can_defer = False
    elif cfg.require_positive_slack_for_defer and slo_slack <= cfg.defer_retry_s:
        can_defer = False

    can_drop = True
    return _to_bool_array((can_admit, can_defer, can_drop))


def resolve_action(
    raw_action: object,
    mask: Sequence[bool],
    snapshot: SchedulerSnapshot,
    *,
    mode: InvalidActionMode = "raise",
    candidate: Optional[CandidateView] = None,
) -> Tuple[Action, Action, bool]:
    """Coerce *raw_action* and validate it against *mask* under *mode*.

    Returns
    -------
    tuple[Action, Action, bool]
        ``(requested_action, executed_action, was_invalid)``.
    """
    if mode not in VALID_INVALID_ACTION_MODES:
        raise ValueError(
            f"invalid_action_mode must be one of {VALID_INVALID_ACTION_MODES}, got {mode!r}"
        )

    # Support numpy integer / 0-d array / int / str / Action
    if np is not None and isinstance(raw_action, np.ndarray):
        if raw_action.ndim == 0 or raw_action.size == 1:
            raw_action = int(raw_action.item())
    elif np is not None and isinstance(raw_action, np.integer):
        raw_action = int(raw_action)

    requested = Action.coerce(raw_action)
    idx = ACTION_TO_INDEX[requested]
    if len(mask) != 3:
        raise ValueError(f"Action mask must have length 3, got {len(mask)}")

    if bool(mask[idx]):
        return requested, requested, False

    if mode == "raise":
        raise ValueError(
            f"Invalid or masked action {requested.value!r} (index={idx}, "
            f"mask={[bool(m) for m in mask]})"
        )

    # Deterministic fallback when mode is "fallback" or "penalize":
    # - Masked ADMIT (capacity temporarily full for this candidate) -> DEFER if valid, else DROP.
    # - Masked DEFER (defer limit reached or slack <= defer_retry_s) -> ADMIT if valid and not overdue, else DROP.
    now_s = float(getattr(snapshot, "now_s", getattr(snapshot, "now", 0.0)))
    target = _extract_candidate(snapshot, candidate)
    slo_slack = _candidate_slack_s(target, now_s) if target is not None else -1.0

    if requested is Action.ADMIT:
        executed = Action.DEFER if bool(mask[ACTION_TO_INDEX[Action.DEFER]]) else Action.DROP
    elif requested is Action.DEFER:
        if bool(mask[ACTION_TO_INDEX[Action.ADMIT]]) and slo_slack >= 0.0:
            executed = Action.ADMIT
        else:
            executed = Action.DROP
    else:
        executed = Action.DROP

    return requested, executed, True


__all__ = [
    "ACTIONS",
    "ACTION_TO_INDEX",
    "ActionMaskConfig",
    "InvalidActionMode",
    "VALID_INVALID_ACTION_MODES",
    "compute_action_mask",
    "resolve_action",
]
