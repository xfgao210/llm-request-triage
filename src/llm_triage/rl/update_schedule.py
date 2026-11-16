"""Causal online-update gates, independent of arrival/source-time scaling."""

from dataclasses import dataclass
import math


def adaptation_preset(name):
    """Untuned ablation settings; conservative does not imply proven improvement."""
    presets = {
        "legacy": dict(learning_rate=.0001, warmup=256, update_interval=4,
                       min_update_seconds=0.0, min_new_outcomes=0, max_updates=None),
        "conservative": dict(learning_rate=.00001, warmup=2048, update_interval=32,
                             min_update_seconds=1.0, min_new_outcomes=32, max_updates=100),
        "custom": {},
    }
    if name not in presets:
        raise ValueError("Adaptation preset must be conservative, legacy or custom")
    return presets[name].copy()


@dataclass
class OnlineUpdateSchedule:
    interval: int = 4
    min_seconds: float = 0.0
    min_outcomes: int = 0
    max_updates: int | None = None
    last_time_s: float = 0.0
    last_outcomes: int = 0

    def __post_init__(self):
        for name, minimum in (("interval", 1), ("min_outcomes", 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(f"Invalid {name}")
        if (isinstance(self.min_seconds, bool) or not isinstance(self.min_seconds, (int, float))
                or not math.isfinite(self.min_seconds) or self.min_seconds < 0):
            raise ValueError("min_seconds must be finite and nonnegative")
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 1):
            raise ValueError("max_updates must be None or a positive integer")

    def ready(self, *, steps, now_s, outcomes, updates):
        return (steps % self.interval == 0
                and (self.max_updates is None or updates < self.max_updates)
                and now_s-self.last_time_s >= self.min_seconds
                and outcomes-self.last_outcomes >= self.min_outcomes)

    def mark(self, now_s, outcomes):
        self.last_time_s, self.last_outcomes = now_s, outcomes
