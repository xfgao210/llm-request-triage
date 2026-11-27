# Online adaptation: mixed-load stress notebook

The current `notebooks/colab_burstgpt3_online_adaptation.ipynb` starts from the
raw BurstGPT 3 CSV, trains new checkpoints, and runs a short five-policy stress
comparison. No old results ZIP or checkpoint upload is needed. Use a new run
name; old notebook folders and clock-enabled checkpoints are not compatible.
The current defaults run five windows per group at 8× in full mode. Quick mode
runs the original 8× peak burst condition only. Neither mode samples requests.

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

### Stability experiment v2: discounting, update budgets and audit logs

Use a fresh run name, default `load_stress_stability_v2`. The notebook now
defaults to `DISCOUNT_MODE="elapsed_time"` and
`ADAPTATION_PRESET="conservative"`. These are initial **development ablation**
settings, not a proven improvement or parameters tuned on an untouched test.
The network, reward weights, serving profile, completion SLO, and default 8×
peak requests/window remain unchanged.

Elapsed-time bootstrapping uses
`gamma ** (elapsed_simulation_seconds / discount_time_unit_s)` in offline
training and online adaptation, on both NumPy and optional CUDA learners.
Defaults are gamma=0.99 per one simulated second. Zero-time scheduling
decisions do not discount future value. This changes the bootstrap multiplier,
not the observed step reward: individual completion events inside one step are
still aggregated by the existing reward model, not separately time-discounted.
The time unit is simulated server time, not original CSV time or learner wall time.

Replay buffers record each transition's elapsed seconds. Time-aware batches
reject missing/negative/nonfinite durations. Checkpoints preserve the discount
mode, gamma and time unit; old checkpoints retain legacy per-decision semantics.
**Retrain when changing discount semantics.** Do not change them only after
loading old weights. Frozen inference does not update under either mode.

| Online preset | LR | Warmup transitions | Decision interval | Min simulated seconds | Min new outcomes | Max updates/condition |
|---|---:|---:|---:|---:|---:|---:|
| conservative | 1e-5 | 2048 | 32 | 1.0 | 32 | 100 |
| legacy | 1e-4 | 256 | 4 | 0 | 0 | unlimited |

All gates must pass after feedback has been delivered. Outcomes include drops
as well as served/violated requests. Gates reset after an update; there is no
catch-up burst. A short run can end with zero updates, which is reported rather
than hidden. `custom` uses the notebook's explicit schedule values. The existing
target network synchronizes every 200 updates; a 100-update budget therefore
keeps its target anchored to the starting checkpoint for that condition.

The two axes are separate: the legacy update preset does not automatically
select decision discounting. For controlled comparisons, vary one axis at a
time: decision+legacy, decision+conservative, elapsed_time+legacy,
elapsed_time+conservative. Retrain consistently for each discount mode; use
separate run names. Changing both axes at once cannot establish which helped.

Set `EVALUATION_SPLIT="validation"` for development ablations. This explicitly
selects a burst window within the configured validation range, not a test
window, and labels it as development rather than held-out evaluation. The
default `"test"` still selects the original mixed/peak conditions. Selecting
validation does change which traffic is evaluated; no such change is hidden
inside the conservative preset. Preserve the same development window/load
when comparing ablations, then evaluate the chosen settings separately.

Each DQN case now exports `*_transitions.csv` and `*_updates.csv`. They record:

- pre-update Q-values and action masks, selected action and request ID;
- simulated elapsed time, source timestamp and actual bootstrap multiplier;
- observed reward components, total/cumulative reward and outcome counts;
- queue lengths, update counts, time/feedback since the previous update,
  update loss and parameter drift.

Plots show decision frequencies, signed reward components, updates against
simulation time and Q(admit)-Q(drop) gaps. Decision shares are not request drop
rates; Q-value gaps are not counterfactual performance. Other-request outcomes
can contribute to a global step reward; the latest action did not necessarily
cause every completion in that step. Logs reflect already delivered feedback,
never future outcomes. Diagnostic logging increases archive size.

**Objective distinction:** primary checkpoint selection and reported ranking
use completion-SLO attainment over all arrivals. The unchanged shaped reward
also penalizes lateness, drops, deferral and wasted GPU time, so higher reward
need not mean higher SLO attainment. Report actual undiscounted reward totals
beside SLO/drop/TTFT metrics; do not silently retune reward weights or call the
two objectives equivalent.

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

## Candidate-order and drop-penalty ablations

The adaptation notebook exposes `RL_CANDIDATE_ORDER` (`fcfs` or `length_aware`)
and `DROP_PENALTY` (default `0.2`). They are recorded as top-level
`rl_candidate_order` and `reward.drop` in the training configuration. Ordering
is applied consistently to training, checkpoint validation, pipeline reload
evaluation, and frozen/adapting replay. Static baselines retain their own rules.
Changing either setting requires fresh training and a unique `RUN_NAME`.

Start with separate validation-data runs: FCFS ordering/drop=0.2, then
length-aware ordering/drop=0.2. With the selected ordering held fixed, compare
drop=0.5 and drop=1.0. These are untuned experiment values, not recommended
optimal penalties. Set `EVALUATION_SPLIT="validation"` for development.
Do not silently compare checkpoints trained under different semantics as a
pure online-adaptation effect. No hard drop-rate cap is implemented.

The notebook now validates every four episodes and displays every checkpoint's
validation attainment/drop rate. Checkpoint selection remains based on mean
per-window validation SLO attainment, not loss. Per-seed
`training_updates_seed*.csv` files record each update's loss, global step,
episode, epsilon and episode-local simulated time; they are included in the
export ZIP. The loss plot uses learner updates and a 50-update moving average.
The validation setting increases evaluation overhead but does not change the
training episode budget. Older run folders lack these logs; use a new run.

## Multi-window exploratory evaluation

The notebook defaults to `RUN_MODE="full"`, `WINDOWS_PER_GROUP=5`,
`MIN_REQUESTS_PER_EVAL_WINDOW=100`, and `EVAL_ARRIVAL_MULTIPLIERS=[8]`.
This means 10 whole source windows, 50 policy cases for one seed, not a full
replay of both multi-day ranges. Windows are 900 source seconds (112.5 simulated
seconds at 8x). Every condition starts empty with the same selected checkpoint
and independent online optimizer/buffer/update budget. No state carries between
windows. The conservative 100-update cap remains per condition.

Selection uses non-overlapping blocks aligned to each explicit range start.
The original group method is selected first (typical for mixed, burst for peak),
then remaining roles from burst, typical, token_heavy, low_count and high_count.
Typical targets the median nonempty count; other roles select the best remaining
block, with earliest timestamp breaking ties. Thus token_heavy/high_count are
not necessarily global maxima when an earlier role already selected that block.
All chosen blocks must meet the explicit minimum request count; requests within
them are not filtered or sampled. Too few eligible blocks raises an error.
`WINDOWS_PER_GROUP=1` and minimum=1 restore the old single-window rules.

The per-window comparison table/CSV and plots preserve separate results, along
with mean input/output tokens, native request/token rates and exact timestamps.
Token volume is a workload descriptor, not calibrated GPU work. Recorded output
lengths are used only for offline selection/descriptors, never supplied to RL.
Selection uses no policy outcomes. These deliberately selected windows are not
a random/representative sample or a new untouched holdout. Reserve new windows
and seeds before final evaluation.

To reuse the previously selected model locally, set `REUSE_TRAINING_RUN` to the
completed run's `training_run` folder, e.g.
`results/runs/colab_burstgpt3_stress/length_aware_drop02_full/training_run`.
Leave all training settings identical, including validation frequency and seed.
The source run must retain its prepared data and traffic-analysis folders.
The notebook reads those shared inputs and copies checked training artifacts
into the new run without modifying the source. Changed configs/checkpoint hashes
are rejected. Reuse provenance is saved in `training_reuse.json`; the exported
ZIP still includes the checkpoint, training diagnostics and prepared manifest,
not the raw/prepared CSV. Leave this setting blank to train fresh as before.

## Continuous first test: runtime pilot and optional full range

Use `notebooks/colab_burstgpt3_continuous_first_test.ipynb` for a runtime pilot
inside the earliest explicit test range. The default is a continuous source hour
from the first nonempty prepared bin: `[21206730, 21210330)`, 471 requests.
The leading 16.58 source hours have no recorded requests. Pilot selection is
traffic-only and retains every request inside its declared interval. It is a
temporal subset, not evaluation of the complete test range. This notebook does
not train. Use a new name, `continuous_mixed_8x_pilot_1h_v1`, not the interrupted
full-range folder. `PILOT_SOURCE_MINUTES` controls duration;
`PILOT_START_TIMESTAMP_S` optionally selects a bin-aligned start. Leaving the
start unset chooses the first nonempty bin. Setting duration to `None` and start
to `None` explicitly selects the full `[21147030, 22155630)` range with 1,153,470
requests; the unoptimized full run proved impractical on this laptop.
`SOURCE_RUN` points to the previous completed run folder containing
`data/processed` and `training_run/training`; default is `length_aware_drop02_full`.
Prepared data and checkpoint fingerprints must match. The previous run is read
only. Local and Colab paths are configurable; Colab requires those original run
files on Drive, not just the compact report ZIP.

Default: 8x, one seed, FCFS/length-aware/static SLO-aware/frozen RL. Queues persist
throughout the selected interval. Optional adapting RL uses one fresh optimizer/buffer
at range start, not per reporting bin. It uses LR=1e-5, warmup=2048, batch=64,
32-decision update eligibility, >=32 new outcomes, >=60 simulated seconds between
updates, and no global cap by default. This is an experimental long-run schedule,
not the prior 100-update short-window protocol. Settings are recorded and a new
run name is required to change them.

There is no drain or reset at reporting boundaries. The existing simulator fully
drains work once after the final arrival; no bounded-tail stop was added. Full
tail latency is measurable, but a backlog may add considerable runtime. Goodput
counts only on-time completions inside each reporting interval, excluding tail.

5/10/15-minute **source-time** reports are calculated directly from request
diagnostics after replay; at 8x these intervals are 37.5/75/112.5 simulated
seconds. Arrival-cohort SLO/drop rates include dropped and unfinished requests.
Empty rates are undefined; percentile metrics are recomputed, not averaged.
The default chart cadence is 15 minutes. Reporting cells can be rerun without
replaying or training. Stability statistics exclude bins with <30 arrivals by
default and expose eligible time/request coverage. The 80% attainment analysis
target is not a change to the five-second request deadline or the reward.

Progress prints roughly every `PROGRESS_EVERY_SECONDS=30` wall seconds between
simulator steps, including decode/drain phases. It reports absolute/source-elapsed
time, observed arrivals, resolved requests, queue size and wall time. It is an
observational hook, not input to the policy. A replay-complete message separates
simulation from report generation. An individual expensive step may delay a
progress message, so this is not a separate heartbeat thread.

Set `CONFIRM_REPLAY=True` only after reviewing the preflight. Complete
policy cases resume via checked caches; interrupted individual cases restart
from their beginning. Large per-decision CSVs are disabled (header only), but
per-request diagnostics and in-memory simulator events/iterations remain large.
No memory/runtime guarantee is made. The default HTML/report ZIP omits the large
request CSVs; retain the run folder for rebinning. It is not a complete resumable
simulation archive. This remains exploratory reused test data, not a new holdout.
