# Simulator Architecture (`src/llm_triage/simulation/`)

**Owner:** Vikas (vc2466) | **Milestone:** Week 1 — Core Simulator Backbone

An iteration-level LLM serving simulator with continuous batching, a paged KV-cache block allocator, recompute preemption, a steppable `TriageKernel` for RL training, and a ground-truth-masked scheduling callback hook. Pure Python with zero runtime dependencies.

## Quickstart

```python
from llm_triage.simulation import (
    FCFSCallback, Request, RTX_5070_QWEN05B, ServiceProfile, simulate,
)

reqs = [
    Request(
        request_id="req-0",
        arrival_time=0.0,
        input_tokens=512,
        output_tokens=128,
        slo_deadline=5.0,
    )
]
result = simulate(reqs, FCFSCallback(), RTX_5070_QWEN05B, ServiceProfile())
print(result.summary())
```

Run unit tests and the synthetic 3x-burst smoke test from the repository root:

```bash
python -m pytest tests/        # 73 unit tests
python scripts/smoke_sim.py    # FCFS on a 2,000-request 3x-burst workload
```

## Module Layout

The simulator has one canonical implementation under `llm_triage.simulation`:

| Module | Contents |
|---|---|
| `simulation/request.py` | `Request`, `RequestState`, and `Action` |
| `simulation/hardware.py` | `HardwareConfig` and `KVCacheManager` |
| `simulation/profile.py` | Prefill/decode timing and service estimation |
| `simulation/callback.py` | Canonical scheduler snapshots, decisions, transitions, and lifecycle events |
| `simulation/engine.py` | `TriageKernel`, `simulate()`, and `SimulatorExecutor` |
| `simulation/baselines.py` | Reference `FCFSCallback` implementation |

## Execution Model

At every iteration boundary `TriageKernel`:
1. **Absorbs arrivals** up to `now` (emitting `ARRIVAL` lifecycle events) and auto-resumes any previously admitted requests that were preempted once KV-cache blocks free up.
2. **Queries the policy** on eligible queued candidates when batch and KV capacity are available:
   - By default (`expire_overdue_in_queue=False`), overdue requests remain in the queue so naive policies like FCFS serve them late and incur wasted GPU compute unless a policy explicitly drops them (`Action.DROP`).
   - `Action.DEFER` places a per-request retry cooldown (`defer_retry_s`) and skips that candidate for the rest of the current scheduling pass so other queued candidates can be considered without head-of-line blocking (bounded by `max_defers`).
3. **Executes one GPU iteration**:
   - **Prefill:** Runs a single batched prefill iteration over newly admitted prompts (emitting `SERVICE_START` and `FIRST_TOKEN`), immediately updates KV block allocations, and interleaves at least one decode step before the next prefill when `running` is non-empty.
   - **Decode:** Advances all running requests by one token in lockstep. If the KV-cache cannot hold the next token across the batch, any sequence exceeding total hardware KV capacity is terminated first; otherwise the most recently admitted request is **preempted** (recompute-style: KV blocks freed, progress reset, queued for automatic re-admission).
   - **Idle:** Advances `now` to the next arrival or defer retry timestamp.
