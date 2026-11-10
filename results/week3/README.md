# Week 3 holdout comparison

I froze the three validation-selected Week 2 checkpoints before running the
test windows. The protocol uses the merged Week 2 training code at
`fea1ded`, its recorded reward settings, the shared simulator, and the same
five-second completion SLO for each policy. The runner checks the prepared
trace, its manifest, and all three checkpoint hashes before starting the holdout.

## What I ran

The comparison uses 16 BurstGPT windows held out from Week 2 training and
checkpoint selection, with 256 requests each. Each window runs at 1.2 and 2.4
source RPS. I replayed the original spacing and a version with the middle third
compressed to one third of its duration.
The compression triples local arrival density in that section and preserves
request order, token lengths, and each request's deadline budget. It also
shortens the full measurement window to seven ninths of the source horizon.
Goodput is measured over the resulting horizon.

I ran three rule-based baselines and the three DQN checkpoints on each paired
window. That gives 384 policy runs and 98,304 request executions from 4,096
unique test requests.

## Results

| Replay | Source RPS | FCFS attainment | DQN attainment across seeds |
|---|---:|---:|---:|
| Original spacing | 1.2 | 99.37% | 92.90%–96.66% |
| Original spacing | 2.4 | 99.37% | 90.50%–93.63% |
| Middle-third 3× | 1.2 | 99.37% | 92.90%–96.66% |
| Middle-third 3× | 2.4 | 99.37% | 90.50%–93.60% |

Length-aware and static SLO-aware matched FCFS on mean attainment in all four
conditions. Seed 17 had the best DQN attainment in each condition. In the
2.4 RPS burst replay it reached 93.60% and dropped 5.79% of requests. FCFS reached
99.37% with zero drops. Seed 7 had a lower mean window P95 TTFT in that replay
(0.060 s versus 0.208 s for FCFS) and dropped 9.08% of requests. TTFT percentiles
are calculated on requests with a first-token event.

FCFS was already near the attainment ceiling at these loads, and the middle-third
burst left its attainment unchanged. The simulator uses the batch=1 calibration
with assumed contention for larger batches. The paired counts describe these
16 windows. The three seeds vary training initialization and exploration, while
each seed sees the same test requests.

## Run it again

Use the Week 3 checkout and Python 3.12:

```bash
python -m pip install -e ".[rl]" gymnasium==1.4.0 numpy==2.5.4
python -m experiments.reproduce_week3
```

The runner downloads the pinned trace when the prepared cache is absent. Pass
`--source /path/to/BurstGPT_1.csv` to prepare from an existing raw copy. Use a
fresh `--output` directory for repeated runs. If you already have all three
frozen checkpoints, pass their directory with `--checkpoints-dir` to skip training.
The strict checksum gate was verified with Python 3.12 on Windows. A different
numerical build can require reusing the exact checkpoints.

- [test_runs.csv](test_runs.csv) has each policy/window result.
- [summary.csv](summary.csv) has means, paired differences from FCFS, and counts
  of windows above, equal to, and below FCFS.
- [run.json](run.json) records the protocol, dataset hashes, and checkpoint hashes.
