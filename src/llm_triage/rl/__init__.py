"""Gymnasium RL environment wrapper, observation builder, action masking, and reward interfaces."""

from .environment import (
    ACTIONS,
    ACTION_TO_INDEX,
    CandidateSelectorSpec,
    RLPolicyCallback,
    TriageEnv,
    VALID_SELECTOR_NAMES,
    resolve_priority_fn,
    run_policy,
)
from .masking import (
    ActionMaskConfig,
    InvalidActionMode,
    VALID_INVALID_ACTION_MODES,
    compute_action_mask,
    resolve_action,
)
from .observation import (
    COMPACT_7_FEATURE_NAMES,
    FULL_FEATURE_NAMES,
    ObservationBuilder,
    ObservationConfig,
    ObservationPreset,
    assert_no_ground_truth_leakage,
    assert_view_fields_only,
    sanitize_request_view,
    sanitize_snapshot,
)
from .rewards import (
    AsymmetricRewardModel,
    RewardBreakdown,
    RewardConfig,
    RewardModel,
)
from llm_triage.simulation import RequestOutcome, Transition

__all__ = [
    "ACTIONS",
    "ACTION_TO_INDEX",
    "ActionMaskConfig",
    "AsymmetricRewardModel",
    "COMPACT_7_FEATURE_NAMES",
    "CandidateSelectorSpec",
    "FULL_FEATURE_NAMES",
    "InvalidActionMode",
    "ObservationBuilder",
    "ObservationConfig",
    "ObservationPreset",
    "RLPolicyCallback",
    "RequestOutcome",
    "RewardBreakdown",
    "RewardConfig",
    "RewardModel",
    "Transition",
    "TriageEnv",
    "VALID_INVALID_ACTION_MODES",
    "VALID_SELECTOR_NAMES",
    "assert_no_ground_truth_leakage",
    "assert_view_fields_only",
    "compute_action_mask",
    "resolve_action",
    "resolve_priority_fn",
    "run_policy",
    "sanitize_request_view",
    "sanitize_snapshot",
]
