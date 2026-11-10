"""Gymnasium-compatible RL environment wrapper around TriageKernel.

This module implements ``TECH_DESIGN.md`` §6.7 / §7.4 and Vikas's Week 2
deliverable:
- Wraps :class:`~llm_triage.simulation.TriageKernel` in a Gymnasium-compatible
  episodic environment (:class:`TriageEnv`) without duplicating request-state
  transition logic.
- Exposes a normalized observation space (queue slack, service estimates,
  queue statistics, active batch & KV-cache utilization) and a 3-action
  discrete action space (``ADMIT=0``, ``DEFER=1``, ``DROP=2``) with action
  feasibility masking and configurable invalid-action handling.
- Strictly masks ground-truth ``output_tokens`` and future arrivals from
  policy observations.
- Provides :class:`RLPolicyCallback` implementing both ``SchedulerCallback`` /
  ``Scheduler`` (``decide(snapshot) -> Decision | None``) and ``TriagePolicy``
  (``act(candidate, snapshot) -> Action``) so any trained RL agent can be
  evaluated through ``simulate()``, ``SimulatorExecutor``, ``CompositeScheduler``,
  or ``evaluate_policy_matrix()``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from llm_triage.metrics import evaluate_events
from llm_triage.rl.masking import (
    ACTIONS,
    ACTION_TO_INDEX,
    ActionMaskConfig,
    InvalidActionMode,
    VALID_INVALID_ACTION_MODES,
    compute_action_mask,
    resolve_action,
)
from llm_triage.rl.observation import (
    ObservationBuilder,
    ObservationConfig,
    ObservationPreset,
    assert_no_ground_truth_leakage,
    assert_view_fields_only,
    sanitize_request_view,
    sanitize_snapshot,
)
from llm_triage.rl.rewards import AsymmetricRewardModel, RewardConfig, RewardModel
from llm_triage.simulation import (
    Action,
    CandidateView,
    Decision,
    HardwareConfig,
    LifecycleEvent,
    Request,
    RequestView,
    ResourceView,
    SchedulerSnapshot,
    SimResult,
    SimState,
    Transition,
    TriageKernel,
)

try:
    from llm_triage.simulation import ServiceProfile
except ImportError:  # pragma: no cover - compatibility with pre-consolidation layout
    from llm_triage.service import ServiceProfile

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional fallback for stdlib-only environments
    np = None  # type: ignore[assignment]

try:
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces

    _GymEnvBase = gym.Env
except ImportError:  # pragma: no cover - lightweight fallback when gymnasium is absent
    gym = None  # type: ignore[assignment]
    gym_spaces = None  # type: ignore[assignment]

    class _GymEnvBase:  # type: ignore[no-redef]
        """Minimal Gymnasium Env base fallback when gymnasium is not installed."""

        metadata: Dict[str, object] = {"render_modes": []}
        action_space: object = None
        observation_space: object = None


# ---------------------------------------------------------------------------
# Fallback Space objects when gymnasium is not installed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FallbackDiscrete:
    """Minimal ``gymnasium.spaces.Discrete`` stand-in for stdlib-only environments."""

    n: int

    def contains(self, x: object) -> bool:
        if isinstance(x, bool):
            return False
        if np is not None and isinstance(x, np.integer):
            x = int(x)
        return isinstance(x, int) and 0 <= x < self.n

    def sample(
        self,
        mask: Optional[Sequence[bool]] = None,
        rng: Optional[random.Random] = None,
    ) -> int:
        gen = rng or random
        if mask is not None:
            valid = [i for i, ok in enumerate(mask) if bool(ok)]
            if not valid:
                return 0
            return gen.choice(valid)
        return gen.randrange(self.n)


@dataclass(frozen=True)
class _FallbackBox:
    """Minimal ``gymnasium.spaces.Box`` stand-in for stdlib-only environments."""

    low: object
    high: object
    shape: Tuple[int, ...]
    dtype: object = float

    def contains(self, x: object) -> bool:
        try:
            vals = list(x)  # type: ignore[arg-type]
        except TypeError:
            return False
        if (len(vals),) != self.shape:
            return False
        lows = list(self.low)  # type: ignore[arg-type]
        highs = list(self.high)  # type: ignore[arg-type]
        return all(
            math.isfinite(float(v)) and float(lo) - 1e-5 <= float(v) <= float(hi) + 1e-5
            for v, lo, hi in zip(vals, lows, highs)
        )


def _make_spaces(builder: ObservationBuilder) -> Tuple[object, object]:
    if gym_spaces is not None and np is not None:
        action_space = gym_spaces.Discrete(len(ACTIONS))
        obs_space = gym_spaces.Box(
            low=np.asarray(builder.low, dtype=np.float32),
            high=np.asarray(builder.high, dtype=np.float32),
            shape=(builder.dim,),
            dtype=np.float32,
        )
        return action_space, obs_space
    return (
        _FallbackDiscrete(n=len(ACTIONS)),
        _FallbackBox(
            low=builder.low,
            high=builder.high,
            shape=(builder.dim,),
            dtype=np.float32 if np is not None else float,
        ),
    )


# ---------------------------------------------------------------------------
# Candidate selection / queue ordering rules (policy-visible features only)
# ---------------------------------------------------------------------------

CandidateSelectorSpec = Union[str, Callable[[RequestView], object], object]

VALID_SELECTOR_NAMES: Tuple[str, ...] = (
    "fcfs",
    "length_aware",
    "shortest_estimated",
    "slo_aware",
    "lowest_slack",
    "earliest_deadline",
)


@dataclass(frozen=True)
class _SingleCandidateSnapshot:
    """Lightweight snapshot wrapper used to adapt a ``CandidateSelector.select()``
    instance to a per-candidate priority key."""

    now_s: float
    candidates: Tuple[CandidateView, ...]
    resources: ResourceView


def _selector_object_to_priority_fn(selector_obj: object) -> Callable[[RequestView], object]:
    """Adapt an object implementing ``priority(view)`` or ``select(snapshot)``."""
    if hasattr(selector_obj, "priority") and callable(getattr(selector_obj, "priority")):
        return getattr(selector_obj, "priority")

    # Map known selector class names directly to their canonical key functions
    cls_name = type(selector_obj).__name__.lower()
    if "fcfs" in cls_name:
        return lambda v: (v.arrival_time_s,)
    if "shortest" in cls_name or "length" in cls_name:
        return lambda v: (v.estimated_service_s, v.arrival_time_s, str(v.request_id))
    if "slack" in cls_name or "slo" in cls_name:
        return lambda v: (v.estimated_slack_s(), v.arrival_time_s, str(v.request_id))
    if "deadline" in cls_name:
        return lambda v: (v.deadline_s, v.arrival_time_s, str(v.request_id))

    # Generic fallback for custom CandidateSelector.select(snapshot) objects
    select_fn = getattr(selector_obj, "select")

    def _key(view: RequestView) -> Tuple[float, str]:
        snap = _SingleCandidateSnapshot(
            now_s=float(view.arrival_time_s + view.time_in_queue),
            candidates=(view,),
            resources=ResourceView(can_admit=True),
        )
        chosen = select_fn(snap)
        return (0.0 if chosen.request_id == view.request_id else 1.0, str(view.request_id))

    return _key


def resolve_priority_fn(
    selector: CandidateSelectorSpec = "fcfs",
) -> Optional[Callable[[RequestView], object]]:
    """Convert a selector name, ``CandidateSelector`` object, or callable into a
    ``priority_fn(RequestView)`` suitable for :class:`~llm_triage.simulation.TriageKernel`.

    All built-in selectors inspect only policy-visible :class:`RequestView`
    fields (never ``output_tokens``).
    """
    if selector is None:
        return None
    if hasattr(selector, "select") and callable(getattr(selector, "select")):
        return _selector_object_to_priority_fn(selector)
    if hasattr(selector, "priority") and callable(getattr(selector, "priority")):
        return getattr(selector, "priority")
    if callable(selector):
        return selector
    if not isinstance(selector, str):
        raise ValueError(
            f"candidate_selector must be a selector name, CandidateSelector, or callable, "
            f"got {selector!r}"
        )
    name = selector.strip().lower()
    if name not in VALID_SELECTOR_NAMES:
        raise ValueError(
            f"Unknown candidate_selector {selector!r}; expected one of {VALID_SELECTOR_NAMES}"
        )
    if name == "fcfs":
        return None
    if name in ("length_aware", "shortest_estimated"):
        return lambda v: (v.estimated_service_s, v.arrival_time_s, str(v.request_id))
    if name in ("slo_aware", "lowest_slack"):
        return lambda v: (v.estimated_slack_s(), v.arrival_time_s, str(v.request_id))
    # earliest_deadline
    return lambda v: (v.deadline_s, v.arrival_time_s, str(v.request_id))


def _unpack_workload(raw: object) -> Tuple[Tuple[Request, ...], Optional[float]]:
    """Extract ``(requests, inferred_horizon_s)`` from either a sequence of
    requests or a ``ReplayWindow`` object (with ``.requests`` and ``.horizon_s``)."""
    if hasattr(raw, "requests"):
        reqs_seq = getattr(raw, "requests")
        inferred_horizon = getattr(raw, "horizon_s", None)
        if inferred_horizon is not None:
            inferred_horizon = float(inferred_horizon)
        return tuple(Request.from_any(r) for r in reqs_seq), inferred_horizon
    return tuple(Request.from_any(r) for r in raw), None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Gymnasium Environment Wrapper
# ---------------------------------------------------------------------------

class TriageEnv(_GymEnvBase):
    """Gymnasium-compatible RL environment wrapping :class:`TriageKernel`.

    Parameters
    ----------
    requests:
        Initial episode workload (a sequence of :class:`Request` / duck-typed
        request objects, a ``ReplayWindow``, or a callable
        ``(rng: random.Random) -> Sequence[Request] | ReplayWindow`` invoked
        on each :meth:`reset`).
    hardware:
        Simulated GPU & KV-cache configuration (:class:`HardwareConfig`).
    profile / service_profile:
        Iteration-level prefill/decode timing model (:class:`ServiceProfile`).
    horizon_s / horizon:
        Nominal measurement window / episode horizon in seconds (used for
        episode progress normalization and fixed-window throughput metrics).
    defer_retry_s / defer_seconds:
        Per-request cooldown in seconds when ``Action.DEFER`` is selected.
    max_defers:
        Maximum allowed defers per request before ``Action.DEFER`` is masked
        out (or expires the request).
    expire_overdue_in_queue:
        Whether overdue requests are automatically expired in queue before
        reaching the policy (defaults to ``False`` so policies must actively
        triage overdue requests).
    estimated_output_tokens:
        Policy-visible output length estimate exposed on :class:`RequestView`.
    service_estimator:
        Optional policy-visible ``(input_tokens, estimated_output_tokens) -> float``
        service-time estimator.
    candidate_selector / queue_order:
        Candidate ordering rule: ``"fcfs"`` (default), ``"length_aware"`` /
        ``"shortest_estimated"``, ``"slo_aware"`` / ``"lowest_slack"``,
        ``"earliest_deadline"``, a ``CandidateSelector`` instance, or a custom
        ``(RequestView) -> key`` callable.
    observation_preset:
        ``"full"`` (default, 15 features) or ``"compact_7"`` (7 features).
    observation_config / observation_builder:
        Optional custom observation configuration or builder.
    mask_config:
        Optional :class:`ActionMaskConfig` controlling ``Action.DEFER`` masking.
    reward_config / reward_model:
        Optional :class:`RewardConfig` or custom :class:`RewardModel`.
    invalid_action_mode:
        How :meth:`step` handles an action whose mask bit is ``False``:
        - ``"raise"`` (default): raise ``ValueError``.
        - ``"fallback"``: substitute a deterministic valid action without penalty.
        - ``"penalize"``: substitute a deterministic valid action and subtract
          ``reward_config.invalid_action_penalty``.
    seed:
        Optional initial random seed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        requests: Union[Sequence[object], Callable[[random.Random], object], object, None] = None,
        hardware: Optional[HardwareConfig] = None,
        profile: Optional[ServiceProfile] = None,
        *,
        service_profile: Optional[ServiceProfile] = None,
        horizon_s: Optional[float] = None,
        horizon: Optional[float] = None,
        defer_retry_s: float = 0.01,
        defer_seconds: Optional[float] = None,
        max_defers: Optional[int] = 64,
        expire_overdue_in_queue: bool = False,
        estimated_output_tokens: Optional[float] = None,
        service_estimator: Optional[Callable[[int, float], float]] = None,
        candidate_selector: Optional[CandidateSelectorSpec] = None,
        queue_order: Optional[CandidateSelectorSpec] = None,
        observation_preset: ObservationPreset = "full",
        observation_config: Optional[ObservationConfig] = None,
        observation_builder: Optional[ObservationBuilder] = None,
        mask_config: Optional[ActionMaskConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        reward_model: Optional[RewardModel] = None,
        invalid_action_mode: InvalidActionMode = "raise",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        explicit_horizon = horizon if horizon is not None else horizon_s
        resolved_horizon = float(explicit_horizon if explicit_horizon is not None else 120.0)
        if not math.isfinite(resolved_horizon) or resolved_horizon <= 0:
            raise ValueError(f"horizon_s must be finite and positive, got {resolved_horizon!r}")
        self._explicit_horizon = explicit_horizon is not None
        self.horizon_s = resolved_horizon

        resolved_defer_s = float(defer_seconds if defer_seconds is not None else defer_retry_s)
        if not math.isfinite(resolved_defer_s) or resolved_defer_s <= 0:
            raise ValueError(f"defer_retry_s must be finite and positive, got {resolved_defer_s!r}")
        self.defer_retry_s = resolved_defer_s

        if max_defers is not None:
            if isinstance(max_defers, bool) or not isinstance(max_defers, int) or max_defers < 1:
                raise ValueError(f"max_defers must be a positive integer or None, got {max_defers!r}")
        self.max_defers = max_defers

        if invalid_action_mode not in VALID_INVALID_ACTION_MODES:
            raise ValueError(
                f"invalid_action_mode must be one of {VALID_INVALID_ACTION_MODES}, "
                f"got {invalid_action_mode!r}"
            )
        self.invalid_action_mode: InvalidActionMode = invalid_action_mode

        selector_spec: CandidateSelectorSpec = (
            candidate_selector
            if candidate_selector is not None
            else (queue_order if queue_order is not None else "fcfs")
        )
        self.candidate_selector = selector_spec
        self._priority_fn = resolve_priority_fn(selector_spec)

        self.hardware = hardware or HardwareConfig()
        resolved_profile = profile if profile is not None else service_profile
        self.profile = resolved_profile or ServiceProfile()
        self.expire_overdue_in_queue = bool(expire_overdue_in_queue)
        self.estimated_output_tokens = estimated_output_tokens
        self.service_estimator = service_estimator

        self.kernel = TriageKernel(
            hardware=self.hardware,
            profile=self.profile,
            defer_retry_s=self.defer_retry_s,
            max_defers=self.max_defers,
            expire_overdue_in_queue=self.expire_overdue_in_queue,
            estimated_output_tokens=self.estimated_output_tokens,
            service_estimator=self.service_estimator,
            priority_fn=self._priority_fn,
            record_iterations=True,
        )

        self._rng = random.Random(seed)
        self._workload_source: Optional[Callable[[random.Random], object]] = None
        self._default_requests: Tuple[Request, ...] = ()
        if callable(requests):
            self._workload_source = requests
        elif requests is not None:
            unpacked_reqs, inferred_horizon = _unpack_workload(requests)
            self._default_requests = unpacked_reqs
            if inferred_horizon is not None and not self._explicit_horizon:
                self.horizon_s = inferred_horizon

        if observation_builder is not None:
            self.observation_builder = observation_builder
        else:
            obs_cfg = observation_config or ObservationConfig(
                preset=observation_preset,
                horizon_s=self.horizon_s,
                max_defers=self.max_defers or 64,
            )
            self.observation_builder = ObservationBuilder(obs_cfg)

        self.mask_config = mask_config or ActionMaskConfig(
            max_defers=self.max_defers,
            defer_retry_s=self.defer_retry_s,
        )

        self.reward_config = reward_config or RewardConfig()
        if reward_model is not None:
            self.reward_model: RewardModel = reward_model
        else:
            self.reward_model = AsymmetricRewardModel(
                self.reward_config,
                defer_retry_s=self.defer_retry_s,
            )

        self.action_space, self.observation_space = _make_spaces(self.observation_builder)

        self.actions_taken: Dict[str, int] = {"admit": 0, "defer": 0, "drop": 0}
        self.invalid_action_count: int = 0
        self.episode_reward: float = 0.0
        self.step_count: int = 0
        self._current_snapshot: Optional[SimState] = None

        if requests is not None:
            self.reset(seed=seed)

    # -- properties --------------------------------------------------------

    @property
    def done(self) -> bool:
        """``True`` when the current episode has reached a terminal state."""
        return self.kernel.is_terminal()

    @property
    def now(self) -> float:
        """Current simulated time in seconds."""
        return self.kernel.now

    def _sync_horizon(self, new_horizon_s: float) -> None:
        if not math.isfinite(new_horizon_s) or new_horizon_s <= 0:
            raise ValueError(f"horizon_s must be finite and positive, got {new_horizon_s!r}")
        self.horizon_s = float(new_horizon_s)
        if hasattr(self.observation_builder, "config") and isinstance(
            self.observation_builder.config, ObservationConfig
        ):
            if self.observation_builder.config.horizon_s != self.horizon_s:
                new_cfg = replace(self.observation_builder.config, horizon_s=self.horizon_s)
                self.observation_builder = ObservationBuilder(new_cfg)

    # -- Gymnasium API -----------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, object]] = None,
    ):
        """Reset the environment for a new episode and return ``(obs, info)``."""
        if gym is not None:
            super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        opts = dict(options) if options is not None else {}
        horizon_opt = opts.get("horizon_s", opts.get("horizon"))
        if horizon_opt is not None:
            self._sync_horizon(float(horizon_opt))  # type: ignore[arg-type]

        if "requests" in opts and opts["requests"] is not None:
            episode_requests, inferred_horizon = _unpack_workload(opts["requests"])
            self._default_requests = episode_requests
            if horizon_opt is None and inferred_horizon is not None:
                self._sync_horizon(inferred_horizon)
        elif self._workload_source is not None:
            sampled = self._workload_source(self._rng)
            episode_requests, inferred_horizon = _unpack_workload(sampled)
            if horizon_opt is None and inferred_horizon is not None and not self._explicit_horizon:
                self._sync_horizon(inferred_horizon)
        else:
            episode_requests = self._default_requests

        raw_snap = self.kernel.reset(episode_requests)
        self._current_snapshot = None if raw_snap is None else sanitize_snapshot(raw_snap)

        if isinstance(self.reward_model, AsymmetricRewardModel):
            self.reward_model.bind(self.kernel, defer_retry_s=self.defer_retry_s)
        elif hasattr(self.reward_model, "reset"):
            self.reward_model.reset()  # type: ignore[attr-defined]

        self.actions_taken = {"admit": 0, "defer": 0, "drop": 0}
        self.invalid_action_count = 0
        self.episode_reward = 0.0
        self.step_count = 0

        obs = self.features()
        info = self._build_info()
        return obs, info

    def step(self, action: object):
        """Apply one triage action and advance to the next decision point.

        Returns
        -------
        tuple
            Standard Gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``.
        """
        if self.done or self._current_snapshot is None:
            raise RuntimeError("Episode is done; call reset() before step()")

        snap = self._current_snapshot
        target_cand = snap.candidate
        raw_action = action

        if hasattr(action, "request_id") and hasattr(action, "action"):
            target_id = getattr(action, "request_id")
            matched = next(
                (
                    v
                    for v in snap.queue_requests
                    if v.request_id == target_id or str(v.request_id) == str(target_id)
                ),
                None,
            )
            if matched is None:
                raise ValueError(
                    f"Decision request_id {target_id!r} is not in eligible candidates"
                )
            target_cand = matched
            raw_action = getattr(action, "action")

        mask = compute_action_mask(snap, self.mask_config, candidate=target_cand)
        requested_action, executed_action, was_invalid = resolve_action(
            raw_action,
            mask,
            snap,
            mode=self.invalid_action_mode,
            candidate=target_cand,
        )

        decision = Decision(request_id=target_cand.request_id, action=executed_action)
        transition = self.kernel.apply(decision)

        self.actions_taken[executed_action.value] += 1
        if was_invalid:
            self.invalid_action_count += 1
        self.step_count += 1

        penalize_invalid = was_invalid and self.invalid_action_mode == "penalize"
        reward = self._score_transition(transition, invalid_action=penalize_invalid)
        self.episode_reward += reward

        if self.kernel.is_terminal():
            self._current_snapshot = None
        else:
            self._current_snapshot = sanitize_snapshot(self.kernel.snapshot())

        obs = self.features()
        terminated = bool(transition.done)
        truncated = False
        info = self._build_info(
            transition=transition,
            requested_action=requested_action,
            executed_action=executed_action,
            was_invalid=was_invalid,
        )
        return obs, float(reward), terminated, truncated, info

    # -- Observation, Mask, and Metrics Helpers ----------------------------

    def snapshot(self) -> Optional[SimState]:
        """Return the current sanitized :class:`SimState` (or ``None`` if terminal)."""
        return self._current_snapshot

    def observation(self) -> Optional[RequestView]:
        """Return the current candidate :class:`RequestView` (or ``None`` if terminal).

        Ground-truth ``output_tokens`` is strictly absent from this view.
        """
        if self._current_snapshot is None:
            return None
        return sanitize_request_view(self._current_snapshot.candidate)

    def features(self):
        """Return the current 1D observation vector."""
        return self.observation_builder.build(self._current_snapshot)

    def action_masks(self):
        """Return the boolean action mask ``(can_admit, can_defer, can_drop)``.

        Named ``action_masks`` for direct compatibility with ``sb3-contrib``'s
        ``MaskablePPO``.
        """
        return compute_action_mask(self._current_snapshot, self.mask_config)

    def mask(self):
        """Alias for :meth:`action_masks` (compatible with ``contributions/hengyi/rl_env.py``)."""
        return self.action_masks()

    def events(self) -> Tuple[LifecycleEvent, ...]:
        """Return all chronological :class:`LifecycleEvent` objects emitted so far."""
        return self.kernel.events()

    def result(self) -> SimResult:
        """Return the current :class:`SimResult` from the underlying kernel."""
        return self.kernel.result()

    def metrics(self) -> Dict[str, Union[float, int, None]]:
        """Compute end-of-episode evaluation metrics from the kernel's lifecycle events."""
        if not self.done:
            raise RuntimeError("Episode metrics are available only after episode termination")

        events = self.kernel.events()
        if not self.kernel.requests:
            return {
                "arrivals": 0,
                "completed": 0,
                "slo_met": 0,
                "dropped": 0,
                "unfinished": 0,
                "overall_slo_attainment": 0.0,
                "slo_attainment": 0.0,
                "drop_rate": 0.0,
                "unfinished_rate": 0.0,
                "window_throughput_rps": 0.0,
                "window_goodput_rps": 0.0,
                "wasted_gpu_time_s": 0.0,
                "gpu_busy_time_s": 0.0,
                "preemptions": 0,
                "episode_reward": float(self.episode_reward),
                "action_admit": self.actions_taken["admit"],
                "action_defer": self.actions_taken["defer"],
                "action_drop": self.actions_taken["drop"],
                "invalid_actions": self.invalid_action_count,
            }

        max_arrival_s = max(r.arrival_time for r in self.kernel.requests)
        effective_window_s = self.horizon_s
        if max_arrival_s >= effective_window_s:
            effective_window_s = math.nextafter(max_arrival_s, math.inf)

        eval_result = evaluate_events(
            events,
            measurement_window_s=effective_window_s,
            window_start_s=0.0,
        )
        sim_res = self.kernel.result()
        summary = dict(eval_result.summary)
        summary["slo_attainment"] = summary.get("overall_slo_attainment", sim_res.slo_attainment)
        summary["wasted_gpu_time_s"] = sim_res.wasted_gpu_time
        summary["gpu_busy_time_s"] = sim_res.gpu_busy_time
        summary["preemptions"] = sim_res.preemptions
        summary["episode_reward"] = float(self.episode_reward)
        summary["action_admit"] = self.actions_taken["admit"]
        summary["action_defer"] = self.actions_taken["defer"]
        summary["action_drop"] = self.actions_taken["drop"]
        summary["invalid_actions"] = self.invalid_action_count
        return summary

    # -- Internal helpers --------------------------------------------------

    def _score_transition(self, transition: Transition, *, invalid_action: bool) -> float:
        if isinstance(self.reward_model, AsymmetricRewardModel):
            return self.reward_model.reward(transition, invalid_action=invalid_action)
        base_reward = float(self.reward_model.reward(transition))
        if invalid_action:
            base_reward -= self.reward_config.invalid_action_penalty
        return base_reward

    def _build_info(
        self,
        *,
        transition: Optional[Transition] = None,
        requested_action: Optional[Action] = None,
        executed_action: Optional[Action] = None,
        was_invalid: bool = False,
    ) -> Dict[str, object]:
        snap = self._current_snapshot
        info: Dict[str, object] = {
            "action_mask": self.action_masks(),
            "now_s": self.kernel.now,
            "horizon_s": self.horizon_s,
            "is_terminal": self.done,
            "queue_length": 0 if snap is None else snap.queue_length,
            "active_batch_size": 0 if snap is None else snap.active_count,
            "kv_utilization": 0.0 if snap is None else snap.kv_utilization,
            "candidate_id": None if snap is None else snap.candidate.request_id,
            "actions_taken": dict(self.actions_taken),
            "invalid_actions": self.invalid_action_count,
        }
        if transition is not None:
            info["decision_request_id"] = transition.decision.request_id
            info["requested_action"] = (
                requested_action.value
                if requested_action is not None
                else transition.decision.action.value
            )
            info["executed_action"] = (
                executed_action.value
                if executed_action is not None
                else transition.decision.action.value
            )
            info["invalid_action"] = was_invalid
            info["previous_time_s"] = transition.previous_time_s
            info["events"] = transition.events
            info["outcomes"] = transition.outcomes
            if transition.done:
                info["episode_metrics"] = self.metrics()
        return info


# ---------------------------------------------------------------------------
# SchedulerCallback / Scheduler / TriagePolicy adapter for RL policies
# ---------------------------------------------------------------------------

class RLPolicyCallback:
    """Adapts a learned RL agent or callable into both a ``SchedulerCallback`` /
    ``Scheduler`` (``decide(snapshot) -> Decision | None``) and a
    ``TriagePolicy`` (``act(candidate, snapshot) -> Action``).

    This allows any trained agent (e.g., Stable-Baselines3 ``PPO`` / ``DQN`` /
    ``MaskablePPO``, ``contributions/hengyi/dqn.py`` ``DQN``, or a Python
    function ``(obs, mask) -> action``) to be evaluated directly with
    :func:`~llm_triage.simulation.simulate`,
    :class:`~llm_triage.simulation.SimulatorExecutor`,
    ``CompositeScheduler``, ``SyntheticExecutor``, and
    :func:`~llm_triage.evaluation.evaluate_policy_matrix`.
    """

    def __init__(
        self,
        policy: object,
        *,
        observation_builder: Optional[ObservationBuilder] = None,
        observation_config: Optional[ObservationConfig] = None,
        mask_config: Optional[ActionMaskConfig] = None,
        candidate_selector: CandidateSelectorSpec = "fcfs",
        invalid_action_mode: InvalidActionMode = "fallback",
        deterministic: bool = True,
    ) -> None:
        self.policy = policy
        self.observation_builder = observation_builder or ObservationBuilder(observation_config)
        self.mask_config = mask_config or ActionMaskConfig()
        self._selector_obj = (
            candidate_selector
            if hasattr(candidate_selector, "select") and callable(getattr(candidate_selector, "select"))
            else None
        )
        self._priority_fn = resolve_priority_fn(candidate_selector)
        self.invalid_action_mode: InvalidActionMode = invalid_action_mode
        self.deterministic = bool(deterministic)

    def priority(self, candidate: RequestView) -> object:
        """Order eligible queued candidates using policy-visible features only."""
        if self._priority_fn is None:
            return (candidate.arrival_time_s,)
        return self._priority_fn(candidate)

    def act(self, candidate: CandidateView, snapshot: SchedulerSnapshot) -> Action:
        """Choose Admit, Defer, or Drop for *candidate* given *snapshot*.

        Satisfies the ``TriagePolicy`` protocol for composition inside
        ``CompositeScheduler(selector=..., policy=RLPolicyCallback(...))``.
        """
        assert_no_ground_truth_leakage(snapshot)
        assert_view_fields_only(candidate)
        obs = self.observation_builder.build(snapshot, candidate=candidate)
        mask = compute_action_mask(snapshot, self.mask_config, candidate=candidate)
        raw_action = self._query_policy(obs, mask, snapshot)
        _, executed, _ = resolve_action(
            raw_action,
            mask,
            snapshot,
            mode=self.invalid_action_mode,
            candidate=candidate,
        )
        return self._coerce_action_for_snapshot(executed, snapshot)

    def decide(self, state: SchedulerSnapshot) -> Optional[Decision]:
        """Select a candidate from *state* and return ``Decision(request_id, action)``
        (or ``None`` if *state* has no candidates)."""
        candidates = getattr(state, "candidates", getattr(state, "queue_requests", ()))
        if not candidates and not hasattr(state, "candidate"):
            return None
        if hasattr(state, "candidates") and not state.candidates:
            return None

        if self._selector_obj is not None:
            target = self._selector_obj.select(state)  # type: ignore[attr-defined]
        elif isinstance(state, SimState):
            target = state.candidate
        else:
            target = min(candidates, key=self.priority)

        executed = self.act(target, state)
        return self._make_decision_for_snapshot(target.request_id, executed, state)

    @staticmethod
    def _coerce_action_for_snapshot(executed: Action, snapshot: object) -> Action:
        # When called from legacy llm_triage.scheduling.base.CompositeScheduler / SyntheticExecutor,
        # return llm_triage.scheduling.base.Action if snapshot is not a SimState.
        if not isinstance(snapshot, SimState):
            try:
                from llm_triage.scheduling.base import Action as BaseAction

                return BaseAction(executed.value)  # type: ignore[return-value]
            except ImportError:
                pass
        return executed

    @staticmethod
    def _make_decision_for_snapshot(
        request_id: Union[int, str],
        executed: Action,
        snapshot: object,
    ) -> Decision:
        if not isinstance(snapshot, SimState):
            try:
                from llm_triage.scheduling.base import Action as BaseAction, Decision as BaseDecision

                return BaseDecision(str(request_id), BaseAction(executed.value))  # type: ignore[return-value]
            except ImportError:
                pass
        return Decision(request_id=request_id, action=Action.coerce(executed))

    def _query_policy(self, obs: object, mask: object, state: SchedulerSnapshot) -> object:
        # 1. Object with .act(obs, mask) (e.g., Hengyi's DQN)
        if hasattr(self.policy, "act") and callable(getattr(self.policy, "act")):
            return self.policy.act(obs, mask)  # type: ignore[attr-defined]
        # 2. Stable-Baselines3 / SB3-Contrib model with .predict(obs, ...)
        if hasattr(self.policy, "predict") and callable(getattr(self.policy, "predict")):
            try:
                action, _ = self.policy.predict(  # type: ignore[attr-defined]
                    obs,
                    action_masks=mask,
                    deterministic=self.deterministic,
                )
            except TypeError:
                action, _ = self.policy.predict(  # type: ignore[attr-defined]
                    obs,
                    deterministic=self.deterministic,
                )
            return action
        # 3. Callable: try (obs, mask) first, then (state,) or (obs,)
        if callable(self.policy):
            try:
                return self.policy(obs, mask)
            except TypeError:
                return self.policy(state)
        raise TypeError(
            f"Unsupported policy object {self.policy!r}; expected callable or object "
            "with .act(obs, mask) or .predict(obs, ...)"
        )


def run_policy(
    env: TriageEnv,
    policy: Callable[..., object],
) -> Dict[str, Union[float, int, None]]:
    """Run one complete episode on *env* using *policy* and return summary metrics.

    *policy* may accept either ``(obs, mask)`` or a single :class:`RequestView`
    / :class:`SimState` argument.
    """
    obs, info = env.reset()
    while not env.done:
        mask = info["action_mask"]
        try:
            action = policy(obs, mask)
        except TypeError:
            action = policy(env.observation())
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return env.metrics()


__all__ = [
    "ACTIONS",
    "ACTION_TO_INDEX",
    "CandidateSelectorSpec",
    "RLPolicyCallback",
    "TriageEnv",
    "VALID_SELECTOR_NAMES",
    "resolve_priority_fn",
    "run_policy",
]
