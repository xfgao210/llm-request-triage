# Baseline capacity sweep in Colab

Open `notebooks/colab_burstgpt3_capacity_sweep.ipynb` in a fresh CPU runtime.
Use the branch containing both the notebook and
`experiments/baseline_capacity_sweep.py`. This workflow starts from the raw
`BurstGPT_3.csv`; it needs neither a trained checkpoint nor an earlier results ZIP.

1. Run setup. Upload the CSV to Drive and set `CSV_PATH` in Settings.
2. Keep outputs on Drive if you want to resume after a runtime disconnect.
3. Prepare/verify data against `configs/burstgpt3_ranges.json`.
4. Inspect the two selected 15-minute development windows and their timestamps.
5. Run the 24 cases: two windows × four arrival multipliers × three baselines.
6. Review load curves, queue timelines, and deadline diagnostics.
7. Download the self-contained HTML report and team ZIP from the export cell.

The script selects nonoverlapping, complete blocks inside the explicit validation
range, using only traffic: highest 5-minute request rate for the burst block;
nearest median block request count among the remaining nonempty blocks for the
typical block. `MANUAL_WINDOWS` allows predeclared alternatives inside training
or validation. Held-out test groups cannot be used for capacity tuning.

At arrival multiplier `m`, every arrival offset and the measurement horizon are
divided by `m`. Completion SLOs stay at five seconds; hardware and the service
profile are fixed. Every baseline uses the same training-derived output-length
estimate. Each replay starts empty, retains internal idle gaps, has no internal
episode resets, and drains fully after arrivals stop. This is not a continuation
of the previous long replay.

## Reading the output

SLO attainment uses all arrivals, including dropped requests. Window goodput
counts only on-time completions within the compressed arrival interval. Queue
wait and completion latency are observed-request quantiles, so read them alongside
drop rates. Busy/batch/KV diagnostics are duration-weighted iteration snapshots
within that interval; GPU-time waste includes the entire drained replay.

Offline isolated service time uses true output length **after** simulation, never
as a scheduler input. Requests that exceed SLO even alone identify a timing-model
limitation. Other misses may involve queueing, contention, memory or scheduling;
they are not automatically avoidable. Per-request diagnostics and binned queue
timelines are included in the ZIP, but the raw/prepared dataset is not.

Timing defaults have batch=1 calibration plus assumed batch contention. Capacity
curves describe this simulator, not validated real-GPU throughput. Two short,
cold-start development windows do not establish generalization.

## Runtime and safe resume

The notebook prints elapsed time after each case. Increasing arrival rate does
not add requests, but may create a backlog. The 30,000-request/window guard stops
accidentally large selections; shorten windows rather than silently truncating
requests or ending overloaded replays early. GPU runtimes do not speed up this
CPU simulator. Raw-data preparation is a separate, one-time cost per runtime.

Rerun the same settings and output name to reuse completed cases. Inputs,
simulator code, Python version and output checksums are verified. Interrupted
cases restart while completed cases remain cached. Use a **new `RUN_NAME`** for
changed settings/code or corrupted outputs. Keep the exact original checkout
to resume after reconnecting; do not silently update code mid-experiment.

## What follows this notebook

Use development evidence to predeclare underload/near-saturation/overload settings,
if those regimes occur. Report if FCFS remains competitive. Do not tune hardware
or SLO until RL wins. Fix the training/continuous-replay observation mismatch,
then retrain and compare frozen/adapting DQN against all baselines using identical
held-out traffic and protocols. That training workflow is intentionally not run
by this diagnostic notebook.
