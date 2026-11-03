"""End-to-end smoke test of the simulator on a synthetic bursty workload.

This is NOT the BurstGPT pipeline (that is the data-ingestion task); it only
exercises the engine at realistic scale: Poisson arrivals with a 3x burst in
the middle and log-normal prompt / output lengths.

    python scripts/smoke_sim.py
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage import (  # noqa: E402
    FCFSCallback,
    KV_CONSTRAINED,
    RTX_5070_QWEN05B,
    Request,
    ServiceProfile,
    simulate,
)


def bursty_workload(
    n: int = 2000,
    base_rate: float = 1.5,
    burst_factor: float = 3.0,
    slo_s: float = 8.0,
    seed: int = 0,
) -> list:
    rng = random.Random(seed)
    reqs, t = [], 0.0
    burst_start, burst_end = n // 3, 2 * n // 3
    for i in range(n):
        rate = base_rate * (burst_factor if burst_start <= i < burst_end else 1.0)
        t += rng.expovariate(rate)
        inp = max(1, min(4096, int(rng.lognormvariate(math.log(400), 0.9))))
        out = max(1, min(1024, int(rng.lognormvariate(math.log(150), 0.8))))
        reqs.append(Request(i, t, inp, out, t + slo_s))
    return reqs


def main() -> None:
    workload = bursty_workload()
    profile = ServiceProfile()
    print(f"profile: {profile.provenance}")
    print(f"workload: {len(workload)} requests over {workload[-1].arrival_time:.1f}s "
          f"(3x burst in the middle third)\n")
    for hw in (RTX_5070_QWEN05B, KV_CONSTRAINED):
        t0 = time.perf_counter()
        res = simulate(workload, FCFSCallback(), hw, profile)
        wall = time.perf_counter() - t0
        print(f"== FCFS on {hw.gpu_name}  (batch={hw.max_batch_size}, "
              f"KV={hw.kv_total_tokens:,} tokens)")
        print(res.summary())
        peak_kv = max(it.kv_blocks_used for it in res.iterations) / hw.kv_total_blocks
        print(f"{'Peak KV utilization':<24} {peak_kv:.1%}")
        print(f"{'Iterations':<24} {len(res.iterations):,}")
        print(f"{'Wall-clock runtime (s)':<24} {wall:.2f}\n")


if __name__ == "__main__":
    main()
