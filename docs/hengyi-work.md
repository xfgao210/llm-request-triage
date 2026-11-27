# Hengyi's experiments

I compared fixed rules, DQN and switching under matched simulation settings.
The work is split into four dependent PRs, in this review order:

| Part | Work | Results |
|---|---|---|
| [1 · PR #11](https://github.com/xfgao210/llm-request-triage/pull/11) | Capacity, admission and online-update pilots on BurstGPT 1 | [Pilot report](../results/scheduling-study/README.md) |
| [2 · PR #12](https://github.com/xfgao210/llm-request-triage/pull/12) | Repeat the study on BurstGPT 3 and separate batch/KV limits | [BurstGPT 3 report](../results/burstgpt3-confirmation/README.md) |
| [3 · PR #13](https://github.com/xfgao210/llm-request-triage/pull/13) | Isolate KV pressure with common ordering and a causal 5% drop cap | [KV comparison](../results/kv-pressure-study/README.md) |
| [4 · PR #10](https://github.com/xfgao210/llm-request-triage/pull/10) | Try burst-based and learned switching, then replay the full peak window at 8x | [Selector results](../results/policy-selector-pressure/README.md), [8x peak results](../results/traffic8x-peak-standard/README.md) |

DQN did not consistently beat the matched rules in these experiments.
The learned selectors stayed with the simple admission policy, so matching
that baseline does not show a learned switching benefit.

At 8x peak traffic (92.76 RPS), admit-only reached 64.27% SLO attainment,
fixed DQN 62.25%, and burst switching 61.45%. The learned selector matched
admit-only without switching. This run used one seed and a previously used
test window, so it is exploratory.

The initial switch compares FCFS with DQN. The pressure comparisons use
common length ordering, so their non-DQN mode is length-ordered admit-only.
Reports retain the exact settings, data/checkpoint hashes and reproduction
commands. These are simulator results, not measured deployment gains.
