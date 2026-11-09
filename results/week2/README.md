# Week 2 DQN

The agent now trains through a Gymnasium wrapper around the same kernel we use
for FCFS, length-aware, and static SLO-aware. I reused the NumPy Double-DQN from
the prototype and trained new weights for the shared simulator.

At each decision, the agent sees the kernel's current candidate, estimated
service time, remaining SLO, queue state, and resource use. The same encoder and
action mask are used in training and checkpoint evaluation. The kernel handles
capacity waits. An intentional defer retries after 0.25 seconds, with up to
four defers per request. The initial reward weights are in
[`configs/week2_dqn.json`](../../configs/week2_dqn.json). Late and unfinished
work carry a larger penalty than a proactive drop. The episode reward also
accounts for overall attainment and wasted simulated GPU time.

## First run

I trained 24 episodes for each of three seeds on the BurstGPT train split.
All three runs updated the weights with finite loss. I selected each seed's
checkpoint by mean attainment across eight validation windows at 1.2 and
2.4 requests per second. Every comparison uses the same requests, deadlines,
hardware and timing settings. The test split is reserved for Week 3.

| Policy | Validation SLO attainment | Drop rate | Goodput (req/s) |
|---|---:|---:|---:|
| FCFS | 84.33% | 0.00% | 1.504 |
| Length-aware | 84.30% | 0.00% | 1.503 |
| Static SLO-aware | 84.30% | 0.05% | 1.503 |
| DQN, seed 7 | 83.50% | 0.49% | 1.486 |
| DQN, seed 17 | 83.47% | 0.05% | 1.488 |
| DQN, seed 27 | 84.03% | 0.00% | 1.498 |

Each row averages 16 validation runs. All three trained checkpoints improved
over their initial policies. FCFS still leads on attainment in this comparison.
The results use the batch=1 calibration with assumed batch contention. This
run establishes that training, checkpoint selection, and evaluation work with
the team simulator.

## Reproduce

```bash
python -m pip install -e ".[dev,rl]"
python -m pytest -q
python scripts/prepare_burstgpt.py
python experiments/train_dqn.py
python experiments/train_dqn.py --checkpoint results/runs/week2-dqn/dqn_seed27.npz --output results/runs/week2-reloaded
```

Use a fresh `--output` directory for each run. The train command writes
one inference checkpoint per seed. Checkpoints are generated under
`results/runs/`, which is ignored by Git. This folder keeps the training
history, checkpoint scores, paired validation rows, and run settings with
the dataset and checkpoint hashes. The run used Python 3.12, NumPy 2.5.4,
and Gymnasium 1.4.0.
