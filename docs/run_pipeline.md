# Run the project end to end

Use Python 3.11 or 3.12. The pipeline runs on CPU and prepares the public
BurstGPT trace on first use. Set aside about 51 MB for the raw trace.

While [PR #9](https://github.com/xfgao210/llm-request-triage/pull/9) is open, use its branch:

```bash
git clone --branch hengyi/week3-repro https://github.com/xfgao210/llm-request-triage.git
cd llm-request-triage
python -m venv .venv
```

If you already cloned the repo, run `git fetch origin`, switch to
`hengyi/week3-repro`, and run `git pull --ff-only`. After the PR merges,
use the default branch.

Activate the environment with `source .venv/bin/activate` on Linux or macOS.
On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1`, or run each
Python command with `.\.venv\Scripts\python.exe`.

```bash
python -m pip install -e ".[dev,rl]"
python -m experiments.run_pipeline --config configs/pipeline_smoke.json --output results/runs/quick-check
python -m experiments.run_pipeline --output results/runs/full
```

The quick check trains one seed for two episodes. The full run trains seeds
7, 17, and 27 for 24 episodes each with the current reward model. Both commands
train on the train split, pick checkpoints on validation, and reload those
checkpoints to compare against FCFS, length-aware, and static SLO-aware.
The test split stays reserved for the frozen Week 3 benchmark.

Each output folder contains:

- `summary.csv` for mean attainment, drop rate, and goodput by policy.
- `validation.csv` for the paired results by window and arrival rate.
- `training/` for checkpoints, training history, selection scores, and run settings.
- `pipeline.json` for the resolved config, runtime, data and code hashes,
  and the path to the training log.

The next run reuses the prepared trace. Pass `--source /path/to/BurstGPT_1.csv`
to use a local copy of the pinned CSV. For a new experiment, use a fresh output
directory and a copy of `configs/week2_dqn.json`. The JSON controls seeds,
episodes, arrival rates, SLO, hardware, defer behavior, and reward weights.
The optional `agent` block controls `hidden`, `learning_rate`, and `gamma`.

If a run fails, check `pipeline.json`, the training log, and `error.log`.
Rerun the same command with `--resume` and the same environment. It checks
saved checkpoints before reusing them. An incomplete training attempt
starts over, while completed training can go straight to evaluation.

## Try the KV pressure setup

```bash
python -m experiments.run_pipeline --config configs/training_kv_stress.json --output results/runs/kv-stress
```

I trained and compared the policies on this config with four batch slots and
4,096 simulated KV tokens. At 2.4 RPS, the eight paired validation windows gave:

| Policy | SLO attainment | Drop rate | Goodput (req/s) |
|---|---:|---:|---:|
| FCFS | 62.30% | 0.00% | 1.484 |
| Static SLO-aware | 70.31% | 23.24% | 1.671 |
| Length-aware | 75.15% | 0.00% | 1.785 |
| DQN, three seeds | 72.61%–73.54% | 20.85%–21.48% | 1.719–1.744 |

DQN was ahead of FCFS and static SLO-aware on attainment and goodput in
this setup. Length-aware still led with zero drops. Under the standard capacity
config, FCFS averaged 84.33% across both validation rates and DQN reached
83.62%–84.16%. KV pressure is a useful place to test whether the agent can
learn a better trade-off when both request length and queue state matter.
These are validation results with simulated resource limits.

To reproduce the published holdout with its original weights and reward, follow
the separate [Week 3 instructions](../results/week3/README.md).
