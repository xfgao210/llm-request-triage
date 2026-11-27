"""Wall-clock-throttled, observational simulator progress (never policy input)."""

import math
import time


class ReplayProgress:
    def __init__(self, label, replay, interval_s=30):
        if isinstance(interval_s, bool) or not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("progress interval must be positive and finite")
        self.label, self.replay, self.interval_s = label, replay, interval_s
        self.started = time.monotonic()
        self.last = None

    def __call__(self, kernel):
        wall = time.monotonic()
        if self.last is not None and wall - self.last < self.interval_s:
            return
        self.last = wall
        source_time = self.replay.source_origin_s + kernel.now / self.replay.time_scale
        resolved = kernel.completed_count + kernel.violated_count + kernel.dropped_count
        queue = len(kernel.queue) + len(kernel._preempted_queue)
        phase = "tail" if kernel.now >= self.replay.horizon_s else "arrivals"
        print(f"Progress {self.label}: elapsed={wall-self.started:.1f}s "
              f"source_time={source_time:.1f} "
              f"source_elapsed_min={kernel.now/self.replay.time_scale/60:.2f} "
              f"arrived={kernel._next}/{len(kernel.requests)} "
              f"resolved={resolved}/{len(kernel.requests)} queue={queue} phase={phase}", flush=True)
