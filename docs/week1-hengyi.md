# Week 1 handoff — BurstGPT pipeline and FCFS

Owner: Hengyi Yin. Delivered November 1, 2026, for the Oct 29–Nov 4 milestone.

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
python -m unittest discover -s tests -v
python scripts/prepare_burstgpt.py
python experiments/run_fcfs.py
```

If the raw CSV is already available, use
`python scripts/prepare_burstgpt.py --source /path/to/BurstGPT_1.csv`.
It must still match the pinned hash. Offline tests do not download anything;
preparation requires the trace and upstream license, downloading missing assets.

To reproduce into a separate result directory:

```bash
python experiments/run_fcfs.py --output results/runs/reproduced-fcfs --summary-dir results/runs/reproduced-fcfs-summary
```

Compare `summary.csv` and `data_manifest.json` byte-for-byte with `results/week1/`.
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

`Request` is immutable and contains `request_id`, `arrival_time_s`, `input_tokens`,
`true_output_tokens`, and `deadline_s`. Request IDs are strings of the zero-based
source row, excluding the CSV header. The loader forbids duplicate source rows.
Only the environment/backend may read `true_output_tokens`.

At a scheduling point, construct a `SchedulerSnapshot` with only arrived,
eligible `CandidateView`s in enqueue order. Call `FCFSScheduler().decide(snapshot)`.
The selector chooses the earliest arrival; tied arrivals preserve candidate
order. It does not use prompt/output length, slack, or lexical request-ID order.
Call only on nonempty queues; empty snapshots raise `ValueError`.

`ResourceView.admission_available` is the kernel's feasibility decision for the
selected head request. Set it after checking resource limits. A blocked head
returns `DEFER`; an available head returns `ADMIT`. FCFS never drops, preempts,
or skips a blocked head. The kernel must advance to a resource/event change on
DEFER rather than immediately polling in a zero-time loop.

The extra feasibility field is an additive proposal to section 7's illustrative
types; it keeps KV-cache and batching logic out of FCFS. `kv_cache_available=None`
means unmodeled, not infinite capacity. Coordinate units and the future resource
contract before freezing the simulator API.

## Evaluation integration — Xiaofei

Use the same `episode.requests` and horizon for every paired policy. Completion
SLO is `completion <= deadline`. All arrivals, including drops, are in its
denominator. Fixed-window throughput counts only completions before or at the
horizon. Drain completions contribute to final accounting, not fixed-window
throughput. Conditional latency must be reported with drop rate and goodput.

`experiments/run_fcfs.py` temporarily adapts the shared requests and scheduler to
the existing single-worker `contributions/hengyi/triage_sim.py`. That simulator
already preserves FCFS queue order and offers only its queue head to callbacks.
The independent selector tests cover multiple candidates until the full snapshot
hook is available. Replace this adapter with the team's shared kernel/collector;
do not copy its transitions into a second implementation.

Current configuration is JSON (`configs/fcfs.json`) to keep this path dependency
free. Change `target_rps` to `[null]` for native timing, or set positive rates.
The runner defaults to validation, leaving test windows unevaluated in this
delivery. It verifies the processed CSV against its manifest before running.

## Validation and limits

- 15 new offline tests passed; 25 original contribution tests passed.
- Original GPU evidence verification passed (102 timings, 2 live policies,
  896 calibrated simulation evaluations).
- Actual pinned trace: 1,429,737 rows; 322,739 excluded by model/channel;
  15,249 invalid/failed; 1,091,749 retained.
- 56 prepared windows; 24 FCFS validation runs, 6,144 request executions.
- FCFS completes every arrival after draining, with no drops or unresolved work.

This is a serial-worker CPU reference integration with assumed input/output
token rates. It has no KV-cache capacity enforcement, continuous batching,
explicit first-token event, TTFT measurements, or live GPU execution. These
results cannot be compared directly with the previous measured-GPU experiment.
The team simulator and metrics implementations remain the next integration step.
