# Frozen policies on the full 8x peak-heavy window

Seed 7, standard cache, all 10,435 requests from the documented 900-second peak-heavy block. The 8x replay divides arrival offsets and the measurement horizon by eight: average native traffic is 11.5944 RPS, versus 92.7556 RPS over 112.5 seconds at 8x. Five-second SLO budgets are unchanged.

DQN, the learned tree and switching settings are frozen from the earlier pressure experiment. Common length ordering and the causal 5% drop cap remain in force. The selected_slack policy is the original standard-cache 9.6-RPS reference held fixed, not a new rule selected on 8x outcomes.

| Traffic | Policy | SLO attainment | Window goodput (req/s) | Drop rate | Switches | Hybrid DQN decisions |
|---|---|---:|---:|---:|---:|---:|
| 1x | admit_only | 100.00% | 11.5944 | 0.00% | 0 | - |
| 1x | selected_slack | 100.00% | 11.5944 | 0.00% | 0 | - |
| 1x | dqn | 100.00% | 11.5944 | 0.00% | 0 | - |
| 1x | burst_switch | 100.00% | 11.5944 | 0.00% | 96 | 6696 |
| 1x | learned_switch | 100.00% | 11.5944 | 0.00% | 0 | 0 |
| 8x | admit_only | 64.27% | 58.8267 | 0.00% | 0 | - |
| 8x | selected_slack | 64.27% | 58.8267 | 0.00% | 0 | - |
| 8x | dqn | 62.25% | 56.5244 | 4.99% | 0 | - |
| 8x | burst_switch | 61.45% | 56.2400 | 3.48% | 4 | 3570 |
| 8x | learned_switch | 64.27% | 58.8267 | 0.00% | 0 | 0 |

This is a reused exploratory test window, not an untouched holdout. One window and seed do not establish a general benefit. The window contains no outputs above 128 tokens; its mean output length is 8.2184 tokens. Treat this as a high-arrival-load stress test, not a long-output stress test.

The original online-adaptation guide supplies the window-selection rule and multiplier definition. These runs retain the frozen pressure-study models, common-order admission controls and drop cap; they do not reproduce fresh mixed-load training or online adaptation from that notebook.

See [settings and reproduction command](../../docs/traffic_8x.md). All model/selector hashes, traffic bounds and source hashes are retained in the JSON records.
