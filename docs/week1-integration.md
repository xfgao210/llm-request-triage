# Week 1 integration

This branch combines the workload/FCFS delivery, Xiaofei's evaluation engine,
and Vikas's iteration-level simulator. Contributor commits are preserved in
merge history. It is proposed for team review before merging to `main`.

## Reproduce

From the repository root, using Python 3.11+:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/prepare_burstgpt.py
python experiments/run_burstgpt.py
```

The preparation command accepts `--source /path/to/BurstGPT_1.csv` for an
existing pinned copy. The ~51 MB raw trace and verbose per-window request logs
are ignored by Git. The same hash-verified data preparation is used as in the
original Week 1 delivery.

The integrated runner defaults to eight validation windows at three arrival
rates and three policies (72 runs). All policies receive identical immutable
requests, completion deadlines, training-only output estimates, hardware, and
service profile. Test windows are prepared but not evaluated in this delivery.

Run the additional artificial KV stress scenario:

```bash
python experiments/run_burstgpt.py --config configs/burstgpt_kv_stress.json --output results/runs/burstgpt-kv-stress --summary-dir results/week1-integrated-kv-stress
```

For a separate reproduction, pass a different `--output` and `--summary-dir`.
`summary.csv`, `data_manifest.json`, and `provenance.json` should reproduce
byte-for-byte with the same code/data/configuration. Real measured scheduler
overhead is stored separately in `scheduler_overhead.csv` and varies between
runs. Verbose per-window tables also include the measured overhead.

## Resolved integration boundaries

- The validated `llm_triage.scheduling.base` types are the canonical policy
  contracts. `llm_triage.types` retains the immutable workload `Request` and
  re-exports those contracts. The earlier duplicate scheduling types are removed.
- `triage.callback` re-exports the metrics layer's `EventType` and
  `LifecycleEvent`. Previously the collector used enum identity checks while
  the simulator produced a different enum class, failing on its first event.
- Source arrival times are emitted during prefill/decode before iteration-end
  events. Arrivals during an iteration retain their true timestamps even though
  admission waits until an iteration boundary. This preserves chronological
  event order, TTFT, and the measurement-window cohort.
- FCFS preserves enqueue order for tied arrival times. Source-row IDs remain
  unpadded strings; lexical sorting must not put source row `10` before `2`.
  Length-aware and SLO-aware selectors retain their existing deterministic ties.
- `SimulatorExecutor.results` retains the backend result per policy for
  attribution/preemption diagnostics and cross-checking the metrics denominator.
- Packaging includes both `triage` and `llm_triage`; Python 3.11+ is the supported
  common version. Both CI workflows run pytest, including simulator function tests
  that unittest discovery alone would not execute.

## Timing and metrics

The default service profile reproduces the earlier RTX 5070 / Qwen2.5-0.5B
batch=1 calibration. Continuous batching uses an assumed contention term; it
has not been calibrated on real batched inference. KV capacity is modeled in
paged blocks. The additional stress configuration deliberately reduces capacity
and batch size; it is not the physical capacity of an RTX 5070.

TTFT is first-token time minus true arrival time. Completion SLO attainment
includes every arrival, including dropped and unfinished requests. Fixed-window
throughput uses the metrics engine's half-open interval `[0, horizon)`;
completions during drain contribute to final accounting only. This boundary
differs from the historical reference runner's inclusive completion boundary.

Conditional latency is always presented alongside drop and unfinished rates.
`total_service_s` / `wasted_service_s` describe request residency intervals from
the metrics layer; overlapping batched requests can overlap in those intervals.
Use the additional `total_gpu_time_s` / `wasted_gpu_time_s` fields for the
simulator's attributed iteration compute, including recomputation. Both are
simulated quantities, not GPU profiling measurements.

## Validation and follow-up

Regression tests cover direct simulator-to-metrics event consumption, true
arrival timestamps during prefill and decode, deadlines expiring within an
iteration, drain accounting, paired policies, stable FCFS ties, actual KV
preemption, drop/defer failures, and an offline trace-to-all-baselines run.
The original contribution tests and GPU evidence verifier remain separate.

The old `results/week1/` artifacts remain historical evidence for commit
`e2cc138`; current integrated code has different hashes. Use the new integrated
results for the current simulator, not the original assumed-rate tables.

Team review should confirm the common Python version, event ownership, FCFS
tie semantics, primary hardware/profile, and merge order of the existing PRs.
The next implementation milestone is connecting an RL environment to this same
kernel and event stream. No RL training result is claimed by this integration.
