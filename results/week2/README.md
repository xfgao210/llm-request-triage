# Week 2 DQN

The agent trains through the team's Gymnasium environment from PR #7. I kept
the NumPy Double-DQN and switched training and evaluation to the shared
15-feature observation builder and action mask.

At each decision, the agent sees the kernel's current candidate, estimated
service time, remaining SLO, queue state, and resource use. The action mask
blocks admission when the candidate does not fit and blocks the defer that
would exhaust its budget. An intentional defer retries after 0.25 seconds,
with up to four defers per request. This run uses the reward defaults shipped
with PR #7. The resolved weights are recorded in [run.json](run.json).

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
| DQN, seed 7 | 81.03% | 9.67% | 1.441 |
| DQN, seed 17 | 83.98% | 0.44% | 1.497 |
| DQN, seed 27 | 83.69% | 2.54% | 1.492 |

Each row averages 16 validation runs. Seeds 7 and 17 improved over their
initial policies, while seed 27 scored lower after training. FCFS leads on
attainment in this comparison. The earlier nine-feature run is available in
the [original PR #6 results](https://github.com/xfgao210/llm-request-triage/blob/9f5d78002a87579c2690a022dcd35962999e9d75/results/week2/README.md).
The results use the batch=1 calibration with assumed batch contention. This
run establishes that training, checkpoint selection, and evaluation work with
the team simulator.

## Reproduce

```bash
python -m pip install -e ".[dev,rl]"
python -m pytest -q
python scripts/prepare_burstgpt.py
python experiments/train_dqn.py
python experiments/train_dqn.py --checkpoint results/runs/week2-dqn/dqn_seed17.npz --output results/runs/week2-reloaded
```

Use a fresh `--output` directory for each run. The train command writes
one inference checkpoint per seed. Checkpoints are generated under
`results/runs/`, which is ignored by Git. This folder keeps the training
history, checkpoint scores, paired validation rows, and run settings with
the dataset and checkpoint hashes. The run used Python 3.12, NumPy 2.5.4,
and Gymnasium 1.4.0.
