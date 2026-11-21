# SLO-aware LLM Request Triage

COMS E6998-007 project. Shared BurstGPT ingestion, scheduling, iteration-level
simulation, and lifecycle-based evaluation.

## Run

Python 3.11+. Baseline runs use the standard library. Run from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/prepare_burstgpt.py
python experiments/run_burstgpt.py
```

Trace preparation downloads the pinned ~51 MB BurstGPT CSV and license, verifies
its SHA256, and creates chronological train/validation/test windows. Use
`--source /path/to/BurstGPT_1.csv` for a verified local copy. Raw/processed traces
and verbose run logs are ignored by Git.

The integrated baseline command runs FCFS, length-aware, and static SLO-aware
policies on the same validation windows through `SimulatorExecutor`, collecting
TTFT, completion SLO attainment, throughput, goodput, drops, and unfinished work.
It uses the RTX 5070/Qwen service profile and simulated paged KV-cache constraints.
Batch contention is an uncalibrated assumption; these are simulation results.

See [integration instructions](docs/week1-integration.md) and the
[retained results](results/week1-integrated/REPORT.md).
An additional [artificial KV stress scenario](results/week1-integrated-kv-stress/REPORT.md)
exercises capacity pressure and preemption on the same validation cohort.

## Run training and evaluation end to end

For BurstGPT 3, use the [Colab training and scenario notebook](notebooks/colab_burstgpt3_experiments.ipynb).
It prepares explicit time ranges, trains and validates DQN, optionally evaluates
frozen test scenarios, and exports charts and a shareable HTML report.
For a separate past-feedback-only experiment, use the
[online adaptation notebook](notebooks/colab_burstgpt3_online_adaptation.ipynb).
It trains clock-disabled models at mixed arrival loads, then compares all three
baselines with frozen/adapting DQN at 1×/4×/8× load on short paired windows.
It includes TTFT/coverage metrics, per-policy caching and a unified report.
The default quick mode runs only the unchanged 8× peak condition with all five
policies and one seed; full mode enables the entire configured comparison.
Quick mode does not thin requests or shorten windows. Shorter, undertrained
smoke-test training is a separate explicit option.
The stability workflow adds consistent elapsed-time bootstrapping, a conservative
online-update budget, and reward/action/Q-value diagnostics. These are development
ablation controls, not a demonstrated policy improvement; changing discount
semantics requires fresh training. Reward weights and the default traffic stay fixed.
Only the CSV is required; no prior
results ZIP is needed. See the [protocol](docs/online_adaptation.md).
Both notebooks offer `DEVICE="auto"` (CUDA minibatch learning when available,
CPU fallback), or explicit `cpu`/`cuda`. Simulation remains CPU-based; GPU mode
is optional and may not be faster for this small network.

Before retraining, use the [baseline capacity notebook](notebooks/colab_burstgpt3_capacity_sweep.ipynb)
to compare FCFS, length-aware, and static SLO-aware on short development windows
at increasing arrival rates. It diagnoses deadline misses and queueing, caches
completed cases, and exports a team report without training or replaying the
held-out test groups. See the [capacity sweep guide](docs/baseline_capacity_sweep.md).

From a checkout with Python 3.11 or 3.12, install the package and run:

```bash
python -m pip install -e ".[dev,rl]"
python -m experiments.run_pipeline
```

The pipeline prepares BurstGPT, trains three DQN seeds, reloads the checkpoints,
and compares them with the rules on validation. See the
[team run guide](docs/run_pipeline.md) for the quick check, custom training
settings, and the KV pressure experiment. The original Week 2 run is recorded
in [its results folder](results/week2/README.md).

## Reproduce the Week 3 holdout

From this checkout with Python 3.12:

```bash
python -m pip install -e ".[rl]" gymnasium==1.4.0 numpy==2.5.4
python -m experiments.reproduce_week3
```

The second command prepares BurstGPT, rebuilds the three validation-selected
checkpoints, checks their hashes, and runs the paired test comparison. See
[the Week 3 results](results/week3/README.md) for the replay settings and output files.

## Components

- [Small learned FCFS / DQN selector](docs/policy_selector.md): offline
  same-state rollouts, a shallow decision tree, and paired validation against
  fixed policies and burst-based switching.
- [KV-pressure selector comparison](docs/pressure_policy_selector.md): the
  existing 4,096-token scenario and standard-cache control, with common
  length ordering and the original causal 5% drop allowance.
- [Full peak-heavy window at native 1x / 8x traffic](docs/traffic_8x.md):
  all 10,435 requests, frozen policies and explicit multiplier semantics.

- [Burst-aware FCFS / frozen-DQN switching](docs/burst_switching.md): causal
  arrival-rate detection, hysteresis, per-episode switch traces, and paired
  validation against fixed policies. Development thresholds require validation.

- [Workload pipeline and original Week 1 handoff](docs/week1-hengyi.md).
- [Simulator architecture](docs/simulator.md): continuous batching, paged
  KV-cache allocation, recompute preemption, and a steppable kernel.
- [RL environment wrapper](docs/rl_environment.md): Gymnasium-compatible
  `TriageEnv` around `TriageKernel`, normalized observation builder with strict
  output-length masking, action feasibility masking (`ADMIT`, `DEFER`, `DROP`),
  configurable invalid-action handling, and `RLPolicyCallback` evaluation adapter.
- [Metrics](docs/metrics.md) and [simulator integration contract](docs/simulator_integration.md).
- [Technical design](TECH_DESIGN.md).

The original assumed-rate CPU results remain archived in `results/week1/`, but
the temporary reference runner has been retired. Both BurstGPT and synthetic
smoke workloads now run through the shared simulator. Use
`python experiments/run_baselines.py` or the
[Colab notebook](notebooks/colab_baselines.ipynb) for a small local example.

## Reward shaping

The shared asymmetric reward model under `src/llm_triage/rl/` rewards on-time
completion, penalizes late computation and proactive drops at different
magnitudes, charges for deferral, and includes terminal completion and
drop-all guardrails. RL environments consume the stable `RewardModel`
interface, whose canonical implementation is `AsymmetricRewardModel`; reward
weights and formulas remain outside the environment. The formula, initial
untuned weights, and information boundary are documented in
[`docs/reward_design.md`](docs/reward_design.md).

## Contributions

- Vikas Chelur: core simulator architecture, resource management, and Week 2 RL environment wrapper.
- Xiaofei Gao: evaluation engine, advanced baseline schedulers, and visualization workflow.
- Hengyi Yin: BurstGPT pipeline, FCFS, GPU calibration, and integration/reproduction.
  See the [initial contribution report](contributions/hengyi/REPORT.md).
