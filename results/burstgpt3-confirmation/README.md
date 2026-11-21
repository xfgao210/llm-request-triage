# BurstGPT 3 scheduling study

We froze the [protocol](../../docs/burstgpt3_confirmation.md) before running
the follow-up to the [BurstGPT 1 pilot](../scheduling-study/README.md).
BurstGPT 3 contributes 11,730 training requests in 24 source windows,
2,105 validation requests in six windows, and 14,953 evaluation requests
in 24 windows. The evaluation windows form four contiguous three-window
blocks in each of the mixed-traffic and peak-heavy ranges. They are
disjoint from the training and validation windows used here.

| Cohort | Requests | Output above 128 tokens |
|---|---:|---:|
| Training | 11,730 | 1,007 |
| Validation | 2,105 | 0 |
| Mixed traffic | 3,097 | 243 |
| Peak heavy | 11,856 | 680 |

The longest validation output was 37 tokens. The validation choices
therefore reflect performance on short outputs, while the evaluation
includes longer requests. True output lengths were used to define these
reporting cohorts after policy selection.

The SLO is completion within five seconds of arrival. Every evaluated
arrival counts toward attainment, including drops. Goodput counts on-time
completions per simulated second inside the replay horizon. The capacity and
admission tables weight windows equally. Online comparisons group requests
by arrival phase and include on-time completions after phase boundaries.
Replay scales timestamps within each source window to the target mean rate.
The capacity numbers come from the simulator's batch-one calibration and
batch contention model.

The fixed-rule runs omit repeated ADMIT callbacks for requests that cannot
fit in the current KV cache, while retaining blocked candidates the rule
would drop. An unoptimized development replay produced identical capacity
and admission CSVs. Regression tests also compare the iteration trace and
request lifecycle events apart from decision logging. The queue-adaptive rule
skips only the leading no-op decisions within each pass and keeps their
effect on the rule's eligible queue count.

## What limits capacity?

At 9.6 offered requests per second, the four-way FCFS check on validation gave
the following results.

| Simulated capacity limits | SLO attainment | Mean waiting requests | Goodput (req/s) |
|---|---:|---:|---:|
| Standard (32 slots) | 100.00% | 1.16 | 9.438 |
| Four slots only | 79.63% | 23.56 | 7.042 |
| 4,096 KV tokens only | 94.67% | 15.15 | 8.254 |
| Both limits | 79.74% | 23.48 | 7.065 |

The four-slot limit accounts for most of the observed SLO loss under the
combined limits. The KV-only case also builds a queue. At 4.8 RPS, the
combined limit still attained 97.96% of SLOs. The first grid point that
satisfied the declared trigger was 9.6 RPS, up from 2.4 RPS in the BurstGPT 1
pilot. We used 2.4, 4.8, and 9.6 RPS for the evaluation. These rates depend
on the sampled trace, simulator settings, and load grid.

Under both limits, FCFS already struggled at 2.4 RPS on the evaluation
ranges. At 9.6 RPS, the no-drop length-aware scheduler raised goodput
from 1.404 to 2.081 req/s in mixed traffic and from 2.660 to 4.254 req/s
in peak-heavy traffic.

| Test range | Offered RPS | FCFS attainment | FCFS mean queue | Length-aware attainment |
|---|---:|---:|---:|---:|
| Mixed traffic | 2.4 | 26.47% | 46.93 | 39.14% |
| Mixed traffic | 4.8 | 26.37% | 73.41 | 34.61% |
| Mixed traffic | 9.6 | 15.39% | 93.61 | 23.66% |
| Peak heavy | 2.4 | 51.29% | 44.16 | 64.33% |
| Peak heavy | 4.8 | 34.35% | 71.17 | 53.80% |
| Peak heavy | 9.6 | 27.75% | 150.32 | 44.67% |

## Does learned admission help under the same drop budget?

The rule and DQN used the same shortest-estimated-service candidate
order. We selected each rule on validation by goodput subject to the
drop budget, then checked both policies' realized drop rates on each
test range. Each DQN seed used its own best validation checkpoint.
All three were selected at episode 8 of 24.

At the 5% budget, **none of the 18 DQN seed/rate/range comparisons was
eligible**. DQN dropped 24.80% to 60.45% of requests across the cells.
The selected rule also exceeded the budget at 2.4 and 4.8 RPS in both
ranges, where validation had selected the −1 s cutoff. At 9.6 RPS,
validation selected the zero-drop length-aware rule, which stayed at
zero drops in both ranges. A validation drop budget alone did not
control test drop rates through this shift.

Across all five budgets, the three eligible comparisons were at
30% drop and 2.4 RPS in the peak-heavy range. The −1 s rule achieved
1.615 on-time req/s at 25.58% drops. The DQNs achieved 1.520–1.546
on-time req/s at 24.80%–25.85% drops. For outputs over 128 tokens, mean
window attainment in this cell was 11.70% for the rule and
9.66%–10.54% for DQN. The
[budget results](admission_budgets.csv) include comparisons where one
or both policies exceeded the budget.

## Do online updates help during a load shift?

Eight continuous replays raised the mean rate from 2.4 to 9.6 RPS and
then returned to 2.4 RPS. Frozen and adapting DQNs started from the same
checkpoint for each seed. Wins, ties, and losses below compare SLO
attainment. Each block averages its three paired seed effects.

| Arrival phase | Block wins / ties / losses | Seed-pair wins / ties / losses | Mean block effect |
|---|---:|---:|---:|
| Before pressure | 0 / 8 / 0 | 0 / 24 / 0 | 0.000 pp |
| Pressure | 2 / 5 / 1 | 3 / 20 / 1 | +0.055 pp |
| Recovery | 0 / 4 / 4 | 1 / 18 / 5 | −0.102 pp |
| Whole replay | 1 / 5 / 2 | 2 / 19 / 3 | −0.022 pp |

Online updates produced a small pressure-phase gain and lost it by
recovery on the block average. The validation-selected queue rule used
the −1 s cutoff below four eligible candidates and admitted whenever
possible at or above four. During pressure it attained 56.83% of SLOs
with 6.02% drops. The adapting DQN attained 56.22% with 35.28% drops.
These percentages pool requests across blocks and average over DQN seeds.
The queue rule is a low-drop reference. It exceeded 5% drops during
pressure as well.

Each adapting replay made 1–100 updates. Fifteen of the 24 replays
updated before pressure. Four made no update during pressure. The median
per-replay mean CPU learner time was 0.253 ms per update, and median
maximum absolute weight movement by the end of the replay was
0.000180. If every pressure-phase update instead blocked serving for
10 ms or 100 ms, it would consume a median 0.11% or 1.12% of the
pressure-phase time budget, respectively. The largest shares were
0.24% and 2.39%. This is a synchronous time-budget calculation.
Reported attainment uses the simulator's original service times.

The [phase counts](online_phases.csv) and [update timing](online_updates.csv)
show each matched replay. The eight source blocks are the units of
replication. Seeds share each block's traffic.

## Reproduce the run

From the repository root, use Python 3.12. The recorded run used Python
3.12.14, NumPy 2.5.4, and Gymnasium 1.4.0.

```sh
python -m pip install -e ".[rl]" numpy==2.5.4 gymnasium==1.4.0
python -c "from pathlib import Path; Path('data/raw').mkdir(parents=True, exist_ok=True)"
curl -fL https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv -o data/raw/BurstGPT_3.csv
python scripts/analyze_burstgpt_traffic.py --source data/raw/BurstGPT_3.csv --output-dir results/traffic_analysis
python scripts/prepare_burstgpt3.py --source data/raw/BurstGPT_3.csv --ranges-config configs/burstgpt3_ranges.json --data-dir data/burstgpt3/confirmation-full
python -m experiments.confirm_scheduling --full-data data/burstgpt3/confirmation-full --output results/runs/burstgpt3-confirmation-repro
```

Use `curl.exe` in Windows PowerShell. Choose a fresh experiment output
directory for each run. The preparation step checks the
[pinned source hash](../../configs/burstgpt3_ranges.json). The run records
the exact window IDs, checkpoint hashes, and validation choices in
[`frozen_selection.json`](frozen_selection.json). Manifest and checkpoint
hashes include provenance from the run, including a digest of a preparation
manifest that stores the source's local path.
