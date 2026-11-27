"""Causal FCFS/custom switching using recent versus preceding arrival rates."""

from collections import deque
from dataclasses import dataclass, replace
import math

from llm_triage.simulation import Decision
from .fcfs import FCFSScheduler


@dataclass(frozen=True)
class BurstSwitchConfig:
    recent_window_s: float = 1.0
    reference_window_s: float = 10.0
    enter_ratio: float = 2.0
    exit_ratio: float = 1.2
    min_reference_rps: float = 1.0
    min_recent_arrivals: int = 3
    min_dwell_s: float = 2.0

    def __post_init__(self):
        for name in ("recent_window_s", "reference_window_s", "enter_ratio",
                     "exit_ratio", "min_reference_rps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.exit_ratio < self.enter_ratio:
            raise ValueError("Require 1 <= exit_ratio < enter_ratio")
        if isinstance(self.min_dwell_s, bool) or not math.isfinite(self.min_dwell_s) or self.min_dwell_s < 0:
            raise ValueError("min_dwell_s must be finite and nonnegative")
        if type(self.min_recent_arrivals) is not int or self.min_recent_arrivals < 1:
            raise ValueError("min_recent_arrivals must be a positive integer")


class BurstSwitchScheduler:
    """Use FCFS in calm traffic and a frozen custom callback during bursts.

    Arrival notifications come from the kernel, including arrivals when no
    admission decision is possible. No lengths, future arrivals, or outcomes
    enter the detector. Switches take effect only at scheduling decisions.
    """

    def __init__(self, custom, config=None):
        self.custom = custom
        self.config = config or BurstSwitchConfig()
        self.fcfs = FCFSScheduler()
        self.reset_episode()

    def reset_episode(self):
        self.arrivals = deque()
        self.mode = "fcfs"
        self.last_switch_s = 0.0
        self.last_arrival_s = -math.inf
        self.switches = []
        self.decisions = {"fcfs": 0, "custom": 0}

    def _prune(self, now):
        cutoff = now - self.config.recent_window_s - self.config.reference_window_s
        while self.arrivals and self.arrivals[0] <= cutoff:
            self.arrivals.popleft()

    def observe_arrival(self, timestamp_s):
        if not math.isfinite(timestamp_s) or timestamp_s < 0 or timestamp_s < self.last_arrival_s:
            raise ValueError("Arrival times must be finite, nonnegative and chronological")
        self.last_arrival_s = timestamp_s
        self._prune(timestamp_s)
        self.arrivals.append(timestamp_s)

    def decide(self, state):
        now = state.now_s
        if self.last_arrival_s > now:
            raise ValueError("Detector received a future arrival")
        cfg = self.config
        self._prune(now)
        recent = sum(time > now - cfg.recent_window_s for time in self.arrivals)
        reference = len(self.arrivals) - recent
        recent_rps = recent / cfg.recent_window_s
        reference_rps = reference / cfg.reference_window_s
        ratio = recent_rps / max(cfg.min_reference_rps, reference_rps)
        ready = now >= cfg.recent_window_s + cfg.reference_window_s
        if ready and now - self.last_switch_s >= cfg.min_dwell_s:
            next_mode = self.mode
            if self.mode == "fcfs" and recent >= cfg.min_recent_arrivals and ratio >= cfg.enter_ratio:
                next_mode = "custom"
            elif self.mode == "custom" and ratio <= cfg.exit_ratio:
                next_mode = "fcfs"
            if next_mode != self.mode:
                self.switches.append(dict(time_s=now, previous=self.mode, mode=next_mode,
                                          recent_rps=recent_rps, reference_rps=reference_rps, ratio=ratio))
                self.mode = next_mode
                self.last_switch_s = now
        policy = self.custom if self.mode == "custom" else self.fcfs
        self.decisions[self.mode] += 1
        # Preserve the selected policy's ordering as well as its action.
        priority = getattr(policy, "priority", None)
        candidate = min(state.queue_requests, key=priority) if priority else state.queue_requests[0]
        selected = replace(state, candidate=candidate)
        action = policy.decide(selected)
        return action if hasattr(action, "request_id") else Decision(candidate.request_id, action)
