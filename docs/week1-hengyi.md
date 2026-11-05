# Week 1 handoff — BurstGPT pipeline and FCFS

Owner: Hengyi Yin. Delivered November 1, 2026, for the Oct 29–Nov 4 milestone.

This is the original reference-simulator handoff. For the subsequent shared
simulator/metrics integration and updated contracts, see
[Week 1 integration](week1-integration.md).

## Delivered scope

- Shared package with canonical environment-truth and policy-visible types.
- Pinned source acquisition, SHA256 verification, schema checks, filtering,
  invalid-row counts, stable timestamp ordering, and chronological partitions.
- Deterministic source-row-disjoint windows, training-only output estimate,
  native timestamp alignment and optional explicit arrival-rate scaling.
- Composable FCFS selector and admit-when-feasible policy.
- A runnable BurstGPT-to-reference-simulator adapter, compact baseline results,
  offline tests, and reproduction instructions.

This completes Hengyi's Week 1 workload/FCFS scope. It does not mark the entire
team's Week 1 simulator, metrics, and advanced-baseline work as complete.
The original `contributions/hengyi/` implementation and retained GPU evidence
are unchanged.

## Run

From the repository root with Python 3.11+:

```bash
python -m pip install -e .
python -m pytest -q
python scripts/prepare_burstgpt.py
python experiments/run_burstgpt.py
```

If the raw CSV is already available, use
`python scripts/prepare_burstgpt.py --source /path/to/BurstGPT_1.csv`.
It must still match the pinned hash. Offline tests do not download anything;
preparation requires the trace and upstream license, downloading missing assets.

To reproduce the integrated baselines into a separate result directory:

```bash
python experiments/run_burstgpt.py \
  --output results/runs/reproduced-baselines \
  --summary-dir results/runs/reproduced-baselines-summary
```

Compare `summary.csv` and `data_manifest.json` byte-for-byte with
`results/week1-integrated/`.
`provenance.json` also records the resolved settings and source-code hashes.
These results have no model-training seed: FCFS and the selected trace windows
are deterministic. Eight different windows are workload samples, not eight
independent random seeds or a confidence interval.

## Simulator integration — Vikas

```python
from llm_triage.workload.burstgpt import load_windows
from llm_triage.workload.transforms import replay_window

windows = load_windows("data/processed/burstgpt_windows.csv")
episode = replay_window(windows["validation", 0], slo_s=5.0, target_rps=1.2)
# Feed episode.requests into the kernel; use episode.horizon_s for measurement.
# target_rps=None keeps native source spacing instead.
```

Each replay request exposes `request_id`, `arrival_time_s`, `input_tokens`,
`true_output_tokens`, and `deadline_s`. Request IDs are strings of the zero-based
source row, excluding the CSV header. The loader forbids duplicate source rows.
Only the environment/backend may read `true_output_tokens`. The simulator makes
a fresh copy before execution, so a replay can be reused across paired policies.

The shared `llm_triage.simulation.TriageKernel` now constructs the canonical
`SchedulerSnapshot`, enforces resource feasibility, and invokes FCFS and the
other policies. Policies never receive ground-truth output length.

## Evaluation integration — Xiaofei

Use the same `episode.requests` and horizon for every paired policy. Completion
SLO is `completion <= deadline`. All arrivals, including drops, are in its
denominator. Fixed-window throughput counts only completions before or at the
horizon. Drain completions contribute to final accounting, not fixed-window
throughput. Conditional latency must be reported with drop rate and goodput.

The temporary single-worker adapter has been removed. The integrated runner
uses the shared kernel, collector, and JSON configurations in
`configs/burstgpt_baselines.json` and `configs/burstgpt_kv_stress.json`.
It defaults to validation, leaving test windows unevaluated unless explicitly
configured otherwise, and verifies processed data against its manifest.

## Archived reference-run validation and limits

The following figures describe the original assumed-rate runner retained in
`results/week1/`, not the shared-simulator command above:

- 15 new offline tests passed; 25 original contribution tests passed.
- Original GPU evidence verification passed (102 timings, 2 live policies,
  896 calibrated simulation evaluations).
- Actual pinned trace: 1,429,737 rows; 322,739 excluded by model/channel;
  15,249 invalid/failed; 1,091,749 retained.
- 56 prepared windows; 24 FCFS validation runs, 6,144 request executions.
- FCFS completes every arrival after draining, with no drops or unresolved work.

Those archived results came from a serial-worker CPU reference integration with
assumed input/output token rates. The shared runner now provides KV-cache
capacity enforcement, continuous batching, first-token events, and TTFT
metrics. It is still a simulation rather than live GPU execution, so neither
result set should be presented as real serving performance.
