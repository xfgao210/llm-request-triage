# Capacity, admission and adaptation pilot

The [study plan](../../docs/scheduling_study.md) defines the rates, budgets and
comparisons. This run uses the pinned BurstGPT 1 trace, Python 3.12, NumPy 2.5.4,
and Gymnasium 1.4.0. The evaluation windows were used by the earlier Week 3
benchmark, so treat the results as exploratory. The online streams stitch those
windows together and retain simulator state across load changes.

## Capacity in development windows

Eight validation windows per rate. Values below are means across windows.

| FCFS configuration | Offered RPS | SLO attainment | Mean queue | Goodput (req/s) |
|---|---:|---:|---:|---:|
| Standard | 2.4 | 83.84% | 0.03 | 1.995 |
| Standard | 4.8 | 80.91% | 4.62 | 3.795 |
| KV stress | 0.6 | 79.15% | 3.07 | 0.475 |
| KV stress | 1.2 | 75.78% | 10.52 | 0.906 |
| KV stress | 2.4 | 62.30% | 25.39 | 1.484 |
| KV stress | 4.8 | 36.82% | 46.82 | 1.758 |

The plan's trigger fired at 2.4 RPS in the KV-stress configuration. It selects
1.2, 2.4 and 4.8 RPS for the next comparison. The trigger marks a useful range
on this grid. It does not estimate a universal saturation point. About 14.01%
of requests exceed the deadline in the isolated service-time diagnostic,
regardless of load. Between 2.4 and 4.8 RPS under KV stress, doubling load adds
about 0.27 on-time completions per second while the mean queue grows by 21.4.
Both the KV budget and batch limit differ between capacity configurations.

## Matching the request order

All policies below use the same shortest-estimated-service ordering. DQN used
that order from training onward. The cutoff rule was selected using validation
goodput within each drop budget.

| Rate | Drop budget | Selected rule | Rule attainment | DQN attainment across seeds |
|---|---:|---|---:|---:|
| 1.2 | 5% | Length-aware | 93.63% | 93.43%–93.60% within budget |
| 2.4 | 5% | Length-aware | 93.07% | 92.60%–93.07% within budget |
| 4.8 | 5% | Length-aware | 91.41% | 90.43%–91.38% within budget |

At 2.4 RPS and a 5% budget, seed 17 tied the rule's attainment with 3.54% drops
versus zero for the rule. Its goodput difference was +0.0006 requests/s.
The 16 paired windows split one ahead, 14 tied and one behind. At a 20% budget,
seed 17 gained 0.17 percentage points over the selected slack rule at that
rate, but the two other seeds fell behind. Across this grid the data show no
consistent incremental benefit from learned admission. There were only 26
long-output requests (>128 tokens) in the paired evaluation cohort, and all
policies recorded zero on-time completions for them at 2.4 RPS.

## Online updates on continuous streams

Four disjoint stitched streams per load pattern and three seeds. Both copies
start from the same checkpoint and take greedy actions. The 2,048-transition
warmup produced zero updates in all 24 adapting runs. With 256 warmup
transitions, all 24 runs updated 15–22 times.

| Short warmup | Phase | Adapting minus frozen on-time completions | Ahead / tied / behind stream-seed pairs |
|---|---|---:|---:|
| Short pressure | Pressure | -1 / 3,072 arrivals | 0 / 11 / 1 |
| Short pressure | Recovery | +1 / 6,144 arrivals | 1 / 11 / 0 |
| Sustained pressure | Pressure | 0 / 6,144 arrivals | 2 / 9 / 1 |
| Sustained pressure | Recovery | 0 / 3,072 arrivals | 1 / 10 / 1 |

Each denominator counts arrivals across seed runs, with the same four streams
shared across seeds. The online learner used a median of about 4 ms of measured
update time per run. The simulation clock excludes this cost. The short warmup
produced updates but no consistent gain at these settings. This comparison
characterizes short streams and conservative updates.

## Reproduce

From the repository root with the RL dependencies installed:

```bash
python scripts/prepare_burstgpt.py
python -m experiments.scheduling_study --output results/runs/study
python -m experiments.scheduling_study --short-warmup-from results/runs/study --output results/runs/study-short
```

`protocol.json` records the selected loads and dataset hashes. The CSVs retain
the window metrics and the paired phase and update counts. Use
`adaptation_short_updates.csv` for the high-resolution learning time.
