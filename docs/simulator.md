# Simulator Architecture (`triage/` & `src/llm_triage/simulation/`)

**Owner:** Vikas (vc2466) | **Milestone:** Week 1 — Core Simulator Backbone

An iteration-level LLM serving simulator with continuous batching, a paged KV-cache block allocator, recompute preemption, a steppable `TriageKernel` for RL training, and a ground-truth-masked scheduling callback hook. Pure Python with zero runtime dependencies.

## Quickstart

```python
from triage import simulate, FCFSCallback, Request, RTX_5070_QWEN05B, ServiceProfile

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

All symbols are exported under both `triage` and `llm_triage.simulation` / `llm_triage.service`:

| Module | Contents |
|---|---|
| `triage/request.py` | `Request` (supports `int` or `str` IDs and `llm_triage` field aliases), `RequestState`, and `Action` (`ADMIT`, `DEFER`, `DROP` with string/int/enum coercion) |
| `triage/hardware.py` | `HardwareConfig` (batch limit, KV block size/budget, `from_model()` sizing from GPU VRAM + model shape), `KVCacheManager` |
| `triage/profile.py` | `ServiceProfile`: prefill/decode iteration timing and policy-visible `estimate_service_time()`; defaults match the measured RTX 5070 / Qwen2.5-0.5B `context_quadratic` calibration |
| `triage/callback.py` | `SchedulerCallback` protocol, `SimState` / `SchedulerSnapshot`, `RequestView` / `CandidateView` (strictly omits ground-truth `output_tokens`), `ResourceView`, `Decision`, and `LifecycleEvent` |
| `triage/engine.py` | `TriageKernel` (steppable `reset` / `snapshot` / `apply` / `is_terminal` / `events` state machine), `simulate()` → `SimResult`, and `SimulatorExecutor` (`PolicyExecutor` adapter) |
| `triage/baselines.py` | Reference `FCFSCallback` implementation |

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
