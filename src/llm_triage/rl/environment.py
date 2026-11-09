"""Gymnasium adapter and observation encoder shared with DQN inference."""

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np

from llm_triage.metrics import evaluate_events
from llm_triage.simulation import Action, EventType, TriageKernel

ACTIONS = (Action.ADMIT, Action.DEFER, Action.DROP)
FEATURES = (
    "remaining_slo", "estimated_service", "input_length", "eligible_queue",
    "estimated_output", "episode_progress", "defer_fraction", "batch_utilization",
    "kv_utilization",
)
LOW = np.array([-5.0] + [0.0] * (len(FEATURES) - 1), dtype=np.float32)
HIGH = np.array([2.0, 10.0, 2.0, 2.0, 10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class DecisionConfig:
    defer_retry_s: float
    max_defers: int

    def __post_init__(self):
        if not math.isfinite(self.defer_retry_s) or self.defer_retry_s <= 0:
            raise ValueError("defer_retry_s must be finite and positive")
        if isinstance(self.max_defers, bool) or not isinstance(self.max_defers, int) or self.max_defers < 1:
            raise ValueError("max_defers must be a positive integer")


def encode_decision(snapshot, horizon_s, config):
    """Use the kernel's current candidate, estimates, and resource state."""
    candidate = snapshot.candidate
    slo = max(candidate.deadline_s - candidate.arrival_time_s, 1e-6)
    values = np.array([
        candidate.slo_slack / slo,
        candidate.estimated_service_s / slo,
        math.log1p(candidate.input_tokens) / 10,
        math.log1p(snapshot.queue_length) / 5,
        candidate.estimated_output_tokens / 1000,
        snapshot.now_s / horizon_s,
        candidate.defer_count / config.max_defers,
        snapshot.batch_utilization,
        snapshot.kv_utilization,
    ], dtype=np.float32)
    can_defer = (
        candidate.defer_count + 1 < config.max_defers
        and candidate.slo_slack > config.defer_retry_s
    )
    return np.clip(values, LOW, HIGH), np.array(
        [snapshot.candidate_fits(), can_defer, True], dtype=bool
    )


@dataclass(frozen=True)
class RewardConfig:
    on_time: float = 1.0
    late: float = -0.4
    drop: float = -0.2
    defer: float = -0.02
    unfinished: float = -0.4
    terminal_attainment: float = 1.0
    wasted_gpu_fraction: float = 0.2

    def __post_init__(self):
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("Reward weights must be finite")
        if self.on_time <= 0 or self.drop >= 0 or self.unfinished >= 0:
            raise ValueError("On-time work needs a bonus and unserved work needs a penalty")
        if self.late > 0 or self.defer > 0 or min(self.terminal_attainment, self.wasted_gpu_fraction) < 0:
            raise ValueError("Late/defer penalties must be nonpositive and terminal weights nonnegative")


class TriageEnv(gym.Env):
    """One replay window with three actions on the kernel's current candidate."""

    def __init__(self, replay, hardware, profile, estimate, decisions, reward):
        super().__init__()
        self.replay = replay
        self.decisions = decisions
        self.reward = reward
        self.deadlines = {str(request.request_id): request.deadline_s for request in replay.requests}
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH, dtype=np.float32)
        self.kernel = TriageKernel(
            hardware=hardware,
            profile=profile,
            estimated_output_tokens=estimate,
            defer_retry_s=decisions.defer_retry_s,
            max_defers=decisions.max_defers,
            record_iterations=False,
        )
        self.snapshot = None

    def _advance_capacity_waits(self):
        # Let the kernel handle blocked admission without spending a DEFER.
        self.snapshot = None
        events = []
        while not self.kernel.is_terminal():
            snapshot = self.kernel.snapshot()
            if snapshot.candidate_fits():
                self.snapshot = snapshot
                break
            events.extend(self.kernel.apply(Action.ADMIT).events)
        return events

    def _observation(self):
        if self.snapshot is None:
            return np.zeros(len(FEATURES), dtype=np.float32), np.zeros(len(ACTIONS), dtype=bool)
        return encode_decision(self.snapshot, self.replay.horizon_s, self.decisions)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.kernel.reset(self.replay.requests)
        self._advance_capacity_waits()
        observation, mask = self._observation()
        return observation, {"action_mask": mask, "terminal_on_reset": self.snapshot is None}

    def step(self, action):
        if self.snapshot is None:
            raise RuntimeError("No decision available. Reset with a runnable workload.")
        _, mask = self._observation()
        if not self.action_space.contains(action) or not mask[action]:
            raise ValueError(f"Invalid action {action!r} for the current mask")
        chosen = ACTIONS[action]
        transition = self.kernel.apply(chosen)
        events = (*transition.events, *self._advance_capacity_waits())
        reward = self.reward.defer if chosen is Action.DEFER else 0.0
        for event in events:
            if event.event_type is EventType.COMPLETION:
                on_time = event.timestamp_s <= self.deadlines[event.request_id]
                reward += self.reward.on_time if on_time else self.reward.late
            elif event.event_type is EventType.DROP:
                reward += self.reward.drop
        observation, mask = self._observation()
        info = {"action_mask": mask}
        done = self.snapshot is None
        if done:
            summary = evaluate_events(
                self.kernel.events(), measurement_window_s=self.replay.horizon_s
            ).summary
            reward += self.reward.unfinished * summary["unfinished"]
            reward += self.reward.terminal_attainment * summary["slo_attainment"]
            backend = self.kernel.result()
            gpu_time = sum(request.gpu_time_s for request in backend.requests)
            if gpu_time > 0:
                reward -= self.reward.wasted_gpu_fraction * backend.wasted_gpu_time / gpu_time
            info["metrics"] = summary
        return observation, float(reward), done, False, info


@dataclass(frozen=True)
class GreedyScheduler:
    model: object
    horizon_s: float
    decisions: DecisionConfig

    def decide(self, snapshot):
        if not snapshot.candidate_fits():
            return Action.ADMIT
        observation, mask = encode_decision(snapshot, self.horizon_s, self.decisions)
        return ACTIONS[self.model.act(observation, mask)]
