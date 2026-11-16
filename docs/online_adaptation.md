# Online adaptation: mixed-load stress notebook

The current `notebooks/colab_burstgpt3_online_adaptation.ipynb` starts from the
raw BurstGPT 3 CSV, trains new checkpoints, and runs a short five-policy stress
comparison. No old results ZIP or checkpoint upload is needed. Use a new run
name; old notebook folders and clock-enabled checkpoints are not compatible.
The default quick mode runs the unchanged 8× peak condition only; full mode
enables all configured conditions. Neither mode samples requests.

## Current notebook workflow

1. Set the CSV path and use a fresh checkout containing
   `experiments/adaptation_stress.py`. Setup adds both the repository and `src`
   to the running kernel's import path; no kernel restart is needed after pip.
2. Prepare the existing explicit chronological ranges. Training and validation
   windows retain all their requests, without traffic-regime filters.
3. Review the traffic-only selected 15-minute evaluation windows **before**
   running policies: median-count nonempty block for mixed traffic, highest
   five-minute request rate for peak-heavy. Explicit manual windows are allowed
   within the respective test groups. Selection is saved and locked by run name.
4. Train/validate with arrival multipliers `[1, 2, 4]`, using fresh models with
   `observation.include_episode_progress=false`. The feature slot stays in the
   15-feature vector but is constantly zero in every phase. Do not disable it
   only at evaluation time on an old checkpoint.
5. Compare FCFS, length-aware, static SLO-aware, frozen DQN and adapting DQN at
   `[1, 4, 8]` load. All five receive identical requests and scaled arrivals.
6. Export unified tables, charts, per-request offline diagnostics, learning
   audits/checkpoints, and a self-contained HTML/ZIP report.

Multipliers divide every arrival offset and measurement duration by the load
factor, while keeping the completion SLO at five seconds and holding hardware,
timing profile, network and reward fixed. **8× is not 8 requests/second.** Configs
use `arrival_multipliers`; a non-native `target_rps` cannot be combined with it.
Training cycles through the load levels so short runs see each one. Checkpoint
selection uses only the configured validation load levels, never 8× outcomes.

For runtime control, defaults select at most 24 nonempty training candidate
windows and 8 validation windows, deterministically spread across each timeline.
This is sampled episode evaluation, not coverage of the complete validation
range. A 24-episode run is an initial budget, not evidence of convergence. Start
with one seed; for stronger exploratory evidence use `[7, 17, 27]` and inspect
individual seed results. Baselines run once per condition, not once per seed.

## Independent conditions and safe resume

### Quick mode without changing the burst

The notebook defaults to `RUN_MODE="quick"`: it first selects the same complete
windows as full mode, then runs only the existing `peak_heavy` window at **8×**
with all five policies and one seed (`QUICK_SEED`, or the first configured seed).
No requests are removed, no window is shortened, and arrivals, SLO, hardware,
queue initialization/draining, and adaptation settings are unchanged. Set
`RUN_MODE="full"` for all configured windows/load factors/seeds.

This reduces evaluation cases from 30 to 5 with the default one seed, not the
number of requests in the burst. It is **not a 1% sample**, and runtime will not
necessarily fall sixfold because the 8× peak condition may be the expensive one.
Data preparation and the default training episode budget are unchanged.

`QUICK_TRAINING=False` preserves the configured training budget. Optionally use
`QUICK_TRAINING=True` with `QUICK_TRAIN_EPISODES=3` for an **undertrained workflow
smoke test**. At least one episode per configured training load is required.
Only this explicit option reduces training episodes/validation interval; it
does not change evaluation traffic. It does change the trained policy, so do
not interpret it as equivalent policy-quality evidence.

Quick runs get a `_quick` folder suffix, or `_quick_short_training` for shorter
training. `execution_plan.json` records the original and selected conditions,
effective seeds/training budget, and mode; it is included in the report/ZIP.
Full-mode folders keep their existing names. Use separate folders to avoid
mixing quick and full results. Quick mode requires an already configured 8×
peak condition rather than silently inventing one.

### State and resume semantics

Each evaluation window/load/seed is an independent condition. All queues start
empty, persist across internal bins, and drain fully after arrivals stop. Frozen
and adapting copies load identical source weights. The adapting copy starts
with fresh optimizer/target/buffer state; no learned state carries between
independent windows or load factors. Thus this measures learning **within** each
short window, not persistent learning across the original multi-day stream.
Longer predeclared windows can be used if short windows supply too little feedback.

Feedback is inserted only after `env.step` returns it; future outcomes are not
exposed. Frozen weights are checked unchanged. Learning computation is timed
separately, not charged to simulated server latency. Online epsilon defaults
to zero; nonzero epsilon introduces exploration costs and changes the experiment.

`PERSIST_RESULTS_TO_DRIVE=True` preserves run outputs across runtime disconnects.
Keep the same `RUN_NAME`, code, source, settings and runtime to resume. Completed
policy cases are checksum-verified and skipped individually. Interrupted cases
restart from the original checkpoint, not from partially recovered learning
state. Incomplete offline training still restarts according to the existing
pipeline behavior. Changed settings/code/corrupted outputs require a new name.

## Metrics and interpretation

- TTFT p50/p95/p99: arrival-to-first-token latency among requests producing a token.
- First-token coverage: requests producing any token divided by all arrivals.
- TTFT-target attainment: first token within `TTFT_TARGET_SECONDS` divided by
  **all arrivals**, including drops/no-token requests as failures. The one-second
  default is an analysis setting, not validated user patience or a reward change.
- Completion-SLO attainment: on-time completions divided by all arrivals.
- Window goodput: on-time completions inside the compressed arrival interval;
  post-interval completions can still contribute to cohort SLO attainment.
- Drop rate, p95 completion latency, queue wait, batch/KV/busy diagnostics,
  GPU-time waste, and offline isolated-service diagnostics.
- Outcome breakdowns by predeclared prompt/output-length buckets. Actual output
  lengths are used only offline, never provided to the policies.
- Arrival-cohort timelines plus update/action counts and paired adapting-minus-
  frozen deltas. Each seed is displayed separately; percentiles are not averaged
  across seeds. Timeline outcomes are retrospective reporting, never learner input.

Latency quantiles exclude requests without observed latency: read them with
drop/coverage rates. Queue/batch/KV diagnostics are approximate iteration
snapshots; GPU waste includes drain. Service timing has batch=1 calibration and
assumed, unvalidated batch contention, not real-GPU throughput validation.

Judge RL against length-aware/static SLO-aware, not just FCFS. A load multiplier
that overloads one development window need not overload another. These mixed/
peak groups were inspected in earlier experiments: label this exploratory and
reserve fresh windows for any confirmatory claim. Never tune on these outcomes
and call the same results an untouched final test.

## Legacy multi-day continuous protocol (not the current notebook default)

The remainder documents the original `experiments.online_adaptation` CLI and
the standalone baseline add-on for previously completed runs. Its persistent
cross-group state and native timing differ from the independent stress protocol.

### Original workflow

The original notebook revision accepted only
`BurstGPT_3.csv` through Google Drive or upload. This is one self-contained
workflow: prepare data, train offline, select checkpoints on validation, then
compare frozen and adapting copies on the same traffic. No previous notebook
run, checkpoint upload, or results ZIP is required. The original frozen-only
notebook remains available as an alternative, not a prerequisite.

In that original revision offline training was unchanged: it used the original training cells
and pipeline, with the same default seeds, episode budget, and validation
selection. `ADAPTATION_SEED` selects one of the trained `SEEDS` for both copies.
Use a blank `RUN_NAME` to rerun the entire workflow in a fresh output folder.
The optional execution device changes where minibatch math runs, not the
Double-DQN objective, masks, reward, or validation protocol.

## Optional GPU minibatch learning

Both notebooks expose `DEVICE="auto"`. In Colab, optionally select a GPU
runtime before running setup. Auto uses CUDA if PyTorch reports a usable CUDA
device, otherwise it uses the original NumPy CPU learner. `DEVICE="cpu"`
forces CPU; `DEVICE="cuda"` fails clearly if CUDA is unavailable. Apple MPS
is not selected. Local users wanting CUDA can install `pip install -e ".[rl,gpu]"`;
PyTorch is optional for CPU-only use. Colab setup includes the GPU extra.

Only sampled neural-network minibatch updates use CUDA. The event simulator,
baselines, and single-request greedy inference remain on CPU. Keeping inference
on CPU avoids synchronizing a tiny prediction with CUDA at every scheduling
decision. Weights are synchronized back after each update, so checkpoints remain
portable `.npz` arrays and the frozen copy need not use CUDA.

**GPU availability does not imply a speedup.** The default 32-unit network is
small, and transfer/kernel overhead and float64 math may make CUDA slower.
Auto chooses by availability, not measured speed. Compare `DEVICE="auto"`
with `DEVICE="cpu"` on the same short development trace before spending a
long GPU session. Offline history and online learning summaries include
`learner_seconds`; run records include the actual learner device/backend.
CUDA uses PyTorch's masked Double-DQN, Huber loss, gradient clipping, and Adam
with reference float64 precision; device/library floating-point differences
can still change learning trajectories. No end-to-end speedup is claimed.

PyTorch documents CUDA availability and execution in its
[CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## Legacy online protocol

The legacy CLI runs two copies from identical selected weights:

- Frozen: greedy decisions with no updates.
- Adapting: fresh optimizer, target network initialized from the checkpoint,
  empty replay buffer, and updates after `TriageEnv.step` delivers feedback.
  Exploration defaults to zero; enabling it changes the comparison to include
  exploration costs. Learning rate defaults to 0.0001, not a tuned optimum.

Both have independent continuous simulations with identical native arrivals,
SLOs, hardware, and observations. Ranges are traversed chronologically once;
weights and replay persist from the earlier group into the later group.
Neither request queues nor learning state reset every five minutes. Outcomes
still in flight are not used prematurely. Updates are timestamped in
`adapting_updates.csv`, after the corresponding transition is observed.
Terminal reward is delivered only when the entire stream terminates.

The original notebook used complete selected test ranges
(`MAX_SECONDS_PER_GROUP=None`). For a manageable initial workflow check, set
`MAX_SECONDS_PER_GROUP=1800` to use a 30-minute prefix of each group.
`RUN_ONLINE_EXPERIMENT=True` runs the comparison after training and validation;
set it to false to stop after validation (skip later comparison/report cells).
A full continuous
run may require a high-RAM runtime because the simulator retains lifecycle
events. Traffic outside selected ranges is omitted and treated as idle;
no-record intervals may actually be collection gaps.

The existing time-progress observation is normalized by the total continuous
replay horizon for both policies, rather than the offline 300-second episode.
Consequently, use this run's frozen copy as the comparison, not the frozen
notebook's episodic results. The learner's wall-clock computation is not
charged as simulated GPU latency. This is simulated online learning, not a
production deployment latency benchmark.

Reports include request-weighted SLO attainment, drop rate, cohort-local
goodput, arrival-cohort timelines, update counts, weight changes, final weights,
and provenance. Timeline metrics use eventual outcomes for reporting only;
they never feed earlier actions. Goodput counts on-time completions of a
group's own arrivals within that interval, excluding completions crossing its
end. Completion-SLO attainment includes eventual completions after the end.

## Add missing baselines without repeating DQN runs

The original notebook's final section added FCFS, length-aware and static SLO-aware on
the same continuous test traffic, then exports a five-policy report, charts,
CSV summary/timelines, and `unified_results.zip`. `RUN_RULE_BASELINES=True`
enables it by default. Each baseline runs in its own independent simulation;
none shares a queue or resources with another policy. Completed baseline
results are cached individually and verified before reuse.

For an already completed Colab run using the older notebook, copy the entire
contents of `notebooks/colab_add_continuous_baselines.py` into one new cell at
the bottom and run only that cell. No repository update is needed for this
standalone patch. Do not rerun settings, training, or DQN comparison cells.
The original `ONLINE_OUTPUT`, `TRAINING_DIR`, and `DATA_DIR` folders must still
exist in the runtime; the downloaded HTML alone is insufficient. Exact ranges,
prefix settings, hardware, SLOs, service profile, and train-only length estimate
come from saved artifacts, not possibly edited notebook settings. Outputs go
to a separate `unified_comparison/` folder and do not modify original results.

This is an online evaluation protocol, not a frozen held-out score. Choose
adaptation settings on a different development stream and report multiple
starting seeds. Do not repeatedly tune against these test outcomes and then
describe them as an untouched benchmark. Completed runs may be reused without
learning again; incomplete runs require a fresh output directory.

With an existing prepared dataset and training folder, the same engine can run
outside Colab:

```bash
python -m experiments.online_adaptation \
  --training-dir results/runs/burstgpt3-initial/training \
  --data-dir data/burstgpt3 \
  --output results/runs/online-check \
  --seed 7 --max-seconds-per-group 1800
```

The CLI defaults to complete ranges when `--max-seconds-per-group` is omitted.
