# DQN under isolated KV-cache pressure

The [protocol](../../docs/kv_pressure_study.md) fixes the batch limit at 32
and changes only the KV-cache limit in the simulated hardware. The standard
configuration has 768,944 KV tokens. The reduced configuration has 4,096.
Both replay the same BurstGPT 3 arrival windows at 2.4, 4.8 and 9.6 RPS
with a five-second completion SLO. The 24 test windows contain 14,953
requests across eight contiguous blocks. The earlier study already used
these windows.

Every policy uses shortest estimated service as its candidate priority.
The cache configurations each train three DQN seeds using the same recipe.
Validation selects checkpoints for those seeds and selects a slack rule
for each load. Both test policies get the same causal cap of one explicit
drop for every 20 observed arrivals. The cap is applied during evaluation.
Unfinished requests count against SLO attainment and are reported separately.

Goodput is on-time completions per second in the replay window. The
comparisons average window results equally. Paired block results group
three consecutive windows and use the same block for each seed.

## Results

At 9.6 RPS with 4,096 KV tokens, all three DQN seeds trailed the
validation-selected −1 s slack rule in both traffic ranges.

| Cache | Traffic | Rule goodput (req/s) | DQN goodput (req/s) | Rule drop | DQN drop |
|---|---|---:|---:|---:|---:|
| Standard | Mixed | 3.459 | 2.552–2.771 | 0% | 3.21–3.63% |
| Standard | Peak-heavy | 6.458 | 5.875–5.960 | 0% | 1.95–2.13% |
| 4,096 tokens | Mixed | 3.247 | 2.555–2.764 | 3.76% | 4.68–4.81% |
| 4,096 tokens | Peak-heavy | 5.819 | 5.388–5.589 | 3.61% | 3.74–4.36% |

All 624 policy-window replays obeyed the 5% drop cap. In the constrained
cache, DQN won 1 of 24 block/seed comparisons at 9.6 RPS and lost the
other 23. Across all three tested rates it won 2, tied 4, and lost 66.
The two wins were +0.008 req/s (mixed traffic at 4.8 RPS, seed 7)
and +0.010 req/s (peak-heavy at 9.6 RPS, seed 27). In each of those
blocks every window had the same SLO attainment and drop rate for both
policies. The differences came from which on-time completions fell
inside the replay horizon.

The small cache produced preemptions in both traffic ranges. At 9.6 RPS
the selected rule averaged 45.5 per mixed window and 38.3 per peak-heavy
window. DQN averaged 46.0–50.5 and 33.2–34.8, respectively. The
peak-heavy DQN reduced preemptions while its goodput and attainment
were lower. The average DQN goodput gap to the selected rule shrank
by 0.262 req/s in mixed traffic and 0.224 req/s in peak-heavy traffic
at 9.6 RPS when KV was restricted, and the remaining gaps favored the
rules. Each gap uses the rule selected for that configuration.

This tests the existing DQN training recipe with a cap enforced at
evaluation. The validation windows contain only outputs up to 37 tokens,
and the test windows contain 923 outputs above 128 tokens. These results
describe the simulator and the tested replay conditions.

See the [standard-cache block pairs](standard/paired_blocks.csv),
[restricted-cache block pairs](kv_only/paired_blocks.csv), and
[restricted-cache window results](kv_only/evaluation.csv).

## Reproduce

Follow the [BurstGPT 3 reproduction instructions](../burstgpt3-confirmation/README.md#reproduce-the-run)
through the `prepare_burstgpt3.py` step.
Then run the two cache configurations from the repository root.

```sh
python -m experiments.kv_pressure_study --full-data data/burstgpt3/confirmation-full --output results/runs/kv-pressure-standard --mode standard
python -m experiments.kv_pressure_study --full-data data/burstgpt3/confirmation-full --output results/runs/kv-pressure-kv_only --mode kv_only
```

Use fresh output directories for a repeat run. Each output records
validation results, frozen selections, and test results by window and block.
