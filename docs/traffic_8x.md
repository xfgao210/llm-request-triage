# Native 8x traffic on the full peak-heavy window

The preceding KV-pressure comparison used **absolute** arrival rates of 2.4,
4.8 and 9.6 RPS. Those are not fixed traffic multipliers. Across its six
validation windows, 9.6 RPS corresponds to roughly 4.95x–13.78x native traffic.

This follow-up uses the documented stress notebook's traffic-only selection:
the complete 15-minute block in the peak-heavy range with the highest five-minute
request rate, breaking ties by total requests and earlier start time. No
requests are sampled or removed.

## Selected traffic

- Source interval: `[23442330, 23443230)` seconds in the pinned BurstGPT 3 trace.
- 10,435 requests over 900 native seconds; average 11.5944 RPS.
- 1x control: original arrival offsets and 900-second measurement horizon.
- 8x stress: divide **every** arrival offset and the measurement horizon by
  eight, giving 112.5 seconds and average 92.7556 RPS.
- Request IDs, order, timestamp ties, input/output lengths and five-second
  completion-SLO budgets remain unchanged. Drain work after the arrival window
  so unfinished or late requests do not vanish from the SLO denominator.

This is eight times the native traffic, not eight RPS or eight times the old
9.6-RPS replay. The selected traffic has mean input length 322.7411 tokens,
mean output length 8.2184, maximum output 63, and no outputs above 128 tokens.
It stresses queueing under high arrival load rather than long-output behavior.

## Frozen comparison

Start with seed 7 and standard cache, following the notebook's one-seed quick
experiment approach. Hardware remains 32 batch slots and 768,944 KV tokens;
timing, SLO, common length order and causal 5% drop allowance stay fixed.

Compare admit-only, the frozen slack reference, fixed DQN, the burst threshold
hybrid and learned switching at 1x and 8x. DQN and the learned tree come from
the earlier pressure experiment; neither is retrained, adapted or refitted at
8x. This probes their behavior under a load shift rather than how well newly
trained 8x models could perform.

The slack cutoff is the original 9.6-RPS validation-selected rule, held fixed
across both multipliers. It was not reselected for the new load; in the standard
configuration it is admit-only. The learned tree is also held fixed, including
entry/exit thresholds and dwell time. The standard-cache seed-7 tree has no
DQN-win leaves, so it cannot start selecting DQN merely because the load rises.

The traffic and 8x definition follow the online-adaptation guide. The serving
and policy controls follow the previous pressure-selector experiment, including
its frozen pressure-trained DQN and drop cap. This is not a reproduction of
the notebook's freshly mixed-load-trained, adapting-DQN comparison.

This peak-heavy test range has been used in prior exploratory work. Its reuse
is explicit; it does not create a new untouched holdout or justify tuning
selector settings on this window. A single window/seed cannot establish a
general improvement.

## Run

```powershell
python -m experiments.evaluate_traffic_multiplier `
  --study-dir results/runs/kv-pressure-standard `
  --selector-dir results/policy-selector-pressure/standard `
  --scope peak `
  --full-data data/burstgpt3/confirmation-full `
  --multipliers 1 8 `
  --seeds 7 `
  --output results/runs/traffic8x-peak-standard
```

Use a fresh output directory. `traffic.json` records the chosen bounds, counts
and effective RPS before evaluation. `runs.partial.csv` saves progress after
each policy completes. Final `runs.csv`, `summary.csv` and `run.json` record
outcomes, mode usage, input hashes and the unchanged model settings. The `load`
column in summary tables is an **arrival multiplier**, not RPS.

The evaluator also supports `--scope validation` for the same six earlier
validation windows, using their original native timing as the multiplier base.

## Result

Completed ten full policy-window replays: five policies at each of 1x and 8x,
with 10,435 requests per replay. All five reached 100% SLO attainment at 1x,
with zero drops and window goodput 11.5944 req/s.

| Policy | 8x SLO attainment | 8x window goodput (req/s) | Drop rate |
|---|---:|---:|---:|
| Admit-only / frozen slack reference | 64.27% | 58.8267 | 0.00% |
| Fixed DQN | 62.25% | 56.5244 | 4.99% |
| Burst switching | 61.45% | 56.2400 | 3.48% |
| Learned switching | 64.27% | 58.8267 | 0.00% |

Burst switching made four switches at 8x, with 3,570 DQN and 9,419 admit-mode
decisions. At 1x it made 96 switches without reducing attainment. Scaling
absolute traffic does not multiply the recent/reference burst ratio by eight;
the window lengths, warmup and dwell remain in simulated seconds.

The frozen learned tree made no switches or DQN decisions at either load.
Matching admit-only therefore does not demonstrate a learned routing gain.
On this window/seed, fixed DQN and burst switching both underperform the simple
admission reference even when the arrival load is high enough to produce
substantial deadline misses. These are exploratory simulation results, not
general evidence that switching cannot help.

Every explicit-drop prefix passed the causal cap check. All frozen model,
selector and configuration hashes stayed unchanged. The final full test run
passed 245 tests and 12 subtests, with 10 optional tests skipped.

See [retained tables and provenance](../results/traffic8x-peak-standard/README.md).
Mode-decision counts in CSV files apply to hybrid policies; zeros in fixed-policy
rows are placeholders rather than counts of how often the fixed policy ran.
