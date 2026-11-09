# Artificial KV-capacity stress test

Hengyi Yin — November 3, 2026. Supplemental integration scenario, not a measured
RTX 5070 capacity/performance result.

Same eight validation windows, 256 requests each, 2.4 RPS, 5-second completion
SLO, training-only length estimate, and service profile as the primary run.
Hardware is deliberately restricted to **4 batch slots and 4,096 KV tokens**
(256 pages of 16 tokens). Three policies yield **24 runs and 6,144 executions**.

| Policy | SLO attainment | Drop rate | Unfinished rate | Mean preemptions/window | Mean window P95 TTFT (s) | Goodput (req/s) |
|---|---:|---:|---:|---:|---:|---:|
| FCFS | 62.30% | 0.00% | 0.00% | 8.625 | 68.481 | 1.484 |
| Length-aware | 75.15% | 0.00% | 2.00% | 3.125 | 68.442 | 1.785 |
| Static SLO-aware | 70.31% | 23.24% | 0.00% | 2.375 | 0.971 | 1.671 |

This exercises real simulator KV allocation/preemption on BurstGPT-derived
requests. Length-aware has the highest completion attainment in this scenario;
static SLO-aware has lower conditional TTFT while rejecting more work.
No single policy is uniformly best across all objectives.

Unfinished requests include those made infeasible by capacity or the bounded
defer mechanism, not only requests still waiting at the end of an arrival
window. Every such request remains an SLO failure. Latency/TTFT percentiles are
conditional on the relevant event occurring and must be read with drop and
unfinished rates. The cohort is paired; eight selected windows are not eight
independent random seeds. These results are descriptive, not statistical proof.

```bash
python experiments/run_burstgpt.py --config configs/burstgpt_kv_stress.json --output results/runs/burstgpt-kv-stress --summary-dir results/week1-integrated-kv-stress
```

The directory retains summaries, measured scheduler overhead, data manifest,
and provenance. Per-request logs are regenerated under the ignored run directory.
See [integration instructions](../../docs/week1-integration.md).
