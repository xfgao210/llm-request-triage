# Learned selector under the existing KV-pressure protocol

Development comparison using the original frozen BurstGPT 3 checkpoints, common shortest-service ordering, five-second completion SLO, 32 batch slots, and a causal 5% drop cap. The non-DQN mode is length-ordered admit-only, not FCFS.

Selectors use eight training windows at three loads; evaluation uses all six original validation windows (2,105 requests) at 2.4/4.8/9.6 RPS. No test windows were replayed, DQN weights were frozen, and selector settings were unchanged from the initial pilot.

## Training-state labels

| Cache | Seed | Samples | DQN wins | Ties | DQN losses |
|---|---:|---:|---:|---:|---:|
| 4,096 tokens | 7 | 266 | 20 | 80 | 166 |
| 4,096 tokens | 17 | 267 | 15 | 52 | 200 |
| 4,096 tokens | 27 | 267 | 23 | 79 | 165 |
| 768,944 tokens | 7 | 260 | 0 | 205 | 55 |
| 768,944 tokens | 17 | 260 | 1 | 223 | 36 |
| 768,944 tokens | 27 | 260 | 2 | 209 | 49 |

## Primary load: 9.6 RPS

Mean window goodput in requests per second, matching the averaging used by the original study. Each cell is based on six paired validation windows.

| Cache | Seed | Fixed DQN | Burst switch | Learned switch | Selected slack rule | Learned DQN decisions |
|---|---:|---:|---:|---:|---:|---:|
| 4,096 tokens | 7 | 7.9482 | 7.9377 | 8.4950 | 8.5328 | 0 |
| 4,096 tokens | 17 | 7.9222 | 7.8690 | 8.4950 | 8.5328 | 0 |
| 4,096 tokens | 27 | 8.0779 | 7.9930 | 8.4950 | 8.5328 | 0 |
| 768,944 tokens | 7 | 9.3770 | 9.4230 | 9.4785 | 9.4785 | 0 |
| 768,944 tokens | 17 | 9.4532 | 9.4532 | 9.4785 | 9.4785 | 0 |
| 768,944 tokens | 27 | 9.3234 | 9.3346 | 9.4785 | 9.4785 | 0 |

## Checks and interpretation

The learned selectors made zero switches and zero DQN decisions at every
load and seed in both capacities, exactly reproducing common-order admit-only.
They avoid the degradation of the tested DQN policies, but do not establish a
useful learned routing policy. Under restricted KV at 9.6 RPS they remain below
the original selected slack rule.

The two capacities complete 540 policy-window replays across five policies, three seeds, three loads and six validation windows. All explicit-drop prefixes obey the shared cap. Fixed admit-only, selected-rule and DQN aggregates were checked against the original study (270 metric comparisons within 1e-12).

The label objective is additional on-time completions after draining a bounded ten-second arrival cohort, rather than goodput or immediate next-action value. Rare training wins do not establish a stable routing advantage. Leaf fractions are empirical training frequencies, not calibrated confidence estimates.

The validation requests have no outputs above 128 tokens, so long-output attainment is unavailable. These results do not reproduce the old mixed/peak test comparison or establish a new held-out result.

Each capacity retains summary.csv, label_counts.csv, run.json and per-seed selector.json. Full labels, request-window metrics and switch traces remain under the corresponding results/runs/policy-selector-*-pressure-v2 directories.

See [implementation and commands](../../docs/pressure_policy_selector.md) and the [original protocol](../../docs/kv_pressure_study.md).
