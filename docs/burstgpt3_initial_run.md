# Initial training on the three annotated ranges

Dataset definition: `configs/burstgpt3_ranges.json`. Training settings:
`configs/burstgpt3_initial.json`. The annotation-derived outer boundaries are
snapped estimates; the timestamp config defines the reproducible intervals.

| Split/group | Start (included) | End (excluded) | Valid requests | Nonempty 300-second episodes |
|---|---:|---:|---:|---:|
| Training | 19462830 | 20465430 | 398,240 | 1,297 |
| Validation | 20465730 | 20716230 | 102,983 | 246 |
| Mixed-traffic test | 21147030 | 22155630 | 1,153,470 | 2,198 |
| Peak-heavy test | 23076030 | 23570430 | 733,698 | 1,125 |

The first annotated range is split approximately 80/20 chronologically, with
a five-minute gap between training and validation. No valid requests occur in
that gap. Requests are filtered to valid ChatGPT API logs. No traffic-regime
filter is applied. Empty episodes remain recorded in the prepared manifest.

Prepare once in a fresh data directory:

```bash
python scripts/prepare_burstgpt3.py \
  --source BurstGPT_3.csv --analysis-dir results/traffic_analysis \
  --ranges-config configs/burstgpt3_ranges.json --data-dir data/burstgpt3
```

Start the exploratory training run:

```bash
python -m experiments.run_pipeline \
  --config configs/burstgpt3_initial.json --data-dir data/burstgpt3 \
  --output results/runs/burstgpt3-initial
```

This uses seed 7, 24 sampled training episodes, native arrival timestamps,
a five-second completion SLO, and validation every 12 episodes. Sampling is
with replacement; 24 episodes is not a full pass over the 1,297 training
episodes. This run establishes a starting point, not convergence or final
multi-seed performance. More training will likely be needed.

Checkpoint selection uses validation only, and the pipeline reloads the chosen
checkpoint to compare it with the baselines on validation. The two test groups
are prepared but are not evaluated by this training command. Keep them reserved
until training settings and model selection are complete.

`pipeline.json` records status and the training-log path. On success,
`training/training.csv` records the learning history, `training/checkpoints.csv`
records validation selection scores, and `summary.csv` contains the validation
comparison. Use a fresh output directory for another run, or `--resume` only to
retry this same run with unchanged code, configuration and runtime.

Continuous online adaptation is a separate experiment. The current saved model
contains inference weights and metadata, not the replay buffer, optimizer state
and target-network state required for full training continuation.

The initial run completed and selected episode 12. DQN mean validation SLO
attainment was 94.38%, compared with 94.51% for FCFS. See the local
`results/runs/burstgpt3-initial/REPORT.md` for both episode-averaged and
request-weighted results. The two test groups have not been evaluated.
