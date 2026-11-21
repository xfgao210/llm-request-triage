# Trying the selector in the existing KV-pressure scenarios

This extends the learned-selector pilot using the high-pressure settings from
[`kv_pressure_study.md`](kv_pressure_study.md), as published at Git commit
`76a2933a9fc8b4378aebbbd03cf2b1d959b5c2f9`.

## What stays fixed

- BurstGPT 3, the original selected chronological cohorts and data hashes.
- 32 batch slots, five-second completion SLO and the recorded service profile.
- 2.4, 4.8 and 9.6 RPS, with every request retained in each evaluated window.
- Both the 4,096-token KV configuration and the standard 768,944-token control.
- The three already trained, validation-selected DQN seeds in each configuration.
- Shortest estimated service ordering for every policy.
- The original causal drop cap: explicit drops must stay at or below one per
  twenty arrivals observed so far, for every prefix of the window.

The exact ordering requirement means the non-DQN mode here is **length-ordered
admit-when-possible**, not FCFS. This is a protocol-matched admission selector,
separate from the earlier FCFS/DQN comparison. Both the burst threshold hybrid
and learned hybrid use the same common order and shared drop allowance.

## What is new

One depth-three decision tree per DQN seed is fitted on paired counterfactual
labels from eight evenly spaced training windows at all three loads. Each
fixed-policy behavior contributes up to six temporally spaced states per case.
Both branches inherit the same running/queued work and already earned drop
allowance, receive the same next ten seconds of arrivals, and drain that cohort.
The DQN label requires strictly more on-time completions; ties favor admission.
The budget is enforced while generating these labels as well as at evaluation.

Changing branches can change which blocked requests are actionable. After
binding the new policy, the offline fork advances to its next valid decision
point before asking it to act. The regression tests cover this case, inherited
drop allowances, common ordering, and budget exhaustion without invalid actions.

The tree settings are unchanged from the original pilot: maximum depth three,
at least ten samples per leaf, entry fraction 0.7, exit fraction 0.4 and minimum
two-second dwell. No setting is selected on the test split.

## Comparison

All six original validation windows (2,105 requests) are replayed at every load.
Each seed compares:

1. Common-order admit-only.
2. The original validation-selected slack rule for that capacity/load.
3. Frozen DQN with the shared cap.
4. Burst-threshold admit/DQN switching with the shared cap.
5. Learned admit/DQN switching with the shared cap.

The selected slack cutoffs are loaded from the original frozen selection;
they are not selected again after inspecting selector outcomes. Rules and DQN
are checked against the original validation aggregates. Report window pairs,
attainment, goodput, actual drops, unfinished requests, preemptions and mode use.
The validation cohort has no outputs above 128 tokens, so its long-output
attainment is unavailable. Historical test windows already appeared in earlier
studies; this experiment does not replay them and makes no fresh holdout claim.

The label objective is on-time completions, while the historical study's primary
comparison uses goodput. Those measures need not rank policies identically.
The tree does not include the remaining drop allowance as an input feature;
the action mask still enforces the cap. Treat this as an initial extension,
not a policy optimized for the full constrained control problem.

## Reproduce

After installing `.[dev,rl,selector]` and reproducing the original KV study:

```powershell
python -m experiments.pressure_policy_selector `
  --study-dir results/runs/kv-pressure-kv_only `
  --output results/runs/policy-selector-kv-pressure-v2

python -m experiments.pressure_policy_selector `
  --study-dir results/runs/kv-pressure-standard `
  --output results/runs/policy-selector-standard-pressure-v2
```

Use fresh output directories. The checkpoint configuration, data, frozen rules,
drop cap and source fingerprint are checked; DQN is not retrained or adapted.
Full output includes per-seed labels/JSON trees, window metrics and switch
traces, summaries, label counts and configuration/provenance records.

## Result

The two capacities completed 540 validation policy-window replays. The fixed
admit-only, selected-rule and DQN aggregates reproduce the original study
across all seeds and loads (270 metric comparisons within 1e-12). All drop
prefixes respected the shared allowance, and DQN weights stayed unchanged.

The 4,096-token condition produced 20/266, 15/267 and 23/267 DQN-win labels
for seeds 7, 17 and 27. The standard condition produced only 0/260, 1/260 and
2/260. The fitted trees never reached the 0.7 entry threshold: learned
switching made zero DQN decisions and exactly matched common-order admit-only
at every load and seed.

At 9.6 RPS with 4,096 KV tokens, mean window goodput was 8.4950 req/s for the
learned selector, 7.9222–8.0779 for fixed DQN, 7.8690–7.9930 for burst switching,
and 8.5328 for the original selected slack rule. Avoiding DQN removes degradation
on these windows, but does not establish a useful learned routing policy or
beat the existing rule. No threshold was lowered to manufacture switches.

See the [retained result tables and models](../results/policy-selector-pressure/README.md).
