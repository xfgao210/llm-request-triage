# SLO-aware LLM Request Triage

COMS E6998-007 project. Shared BurstGPT ingestion, scheduling, iteration-level
simulation, and lifecycle-based evaluation.

## Run

Python 3.11+; runtime uses only the standard library. Run from the repository root:

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

## Components

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
drop-all guardrails. The formula, initial untuned weights, and information
boundary are documented in [`docs/reward_design.md`](docs/reward_design.md).

## Contributions

- Vikas Chelur: core simulator architecture, resource management, and Week 2 RL environment wrapper.
- Xiaofei Gao: evaluation engine, advanced baseline schedulers, and visualization workflow.
- Hengyi Yin: BurstGPT pipeline, FCFS, GPU calibration, and integration/reproduction.
  See the [initial contribution report](contributions/hengyi/REPORT.md).
