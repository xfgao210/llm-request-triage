# Integrated BurstGPT baseline results

Hengyi Yin — November 3, 2026. Simulation using the combined team implementation.

## Protocol

Eight chronological validation windows, 256 requests each. Each is replayed at
0.6, 1.2, and 2.4 requests/s under FCFS, length-aware, and static SLO-aware:
**72 policy runs, 18,432 request executions, 2,048 unique source requests**.
All policies receive paired requests and a 5-second completion SLO. The output
estimate is the selected training-window mean (109.607 tokens); current/future
true output lengths are hidden from policies. Test windows are not evaluated.

The simulator uses 32 batch slots, 16-token KV pages, and 48,059 KV blocks from
the RTX 5070/Qwen hardware sizing model. The timing profile matches the earlier
batch=1 measurements; batched contention is assumed. No GPU inference is run.
81.98% of selected validation requests are inside the measured input/output
length bounds; this does not validate every shape or batched behavior.

## Mean results over eight windows

All three policies have 85.01% attainment at 0.6 RPS and 84.81% at 1.2 RPS.
Differences remain small at 2.4 RPS:

| Policy, 2.4 RPS | SLO attainment | Mean window P95 TTFT (s) | Window goodput (req/s) | Drop rate | Unfinished rate |
|---|---:|---:|---:|---:|---:|
| FCFS | 83.84% | 0.199 | 1.995 | 0.00% | 0.00% |
| Length-aware | 83.79% | 0.153 | 1.993 | 0.00% | 0.00% |
| Static SLO-aware | 83.79% | 0.115 | 1.993 | 0.10% | 0.00% |

P95 values are arithmetic means of per-window P95s, not pooled percentiles.
No preemptions occurred in the primary configuration. These loads do not
establish a strong admission-policy advantage; capacity sensitivity is tested
separately in the [artificial KV stress scenario](../week1-integrated-kv-stress/REPORT.md).

## Evidence and reproduction

- `summary.csv`: deterministic per-window metrics for all 72 runs, including
  TTFT, attainment, drop/unfinished accounting, throughput/goodput, preemptions,
  and attributed simulated GPU compute.
- `scheduler_overhead.csv`: actual CPU wall-clock policy overhead, kept separate
  because it varies across repetitions.
- `data_manifest.json`, `provenance.json`: pinned source, partitions, resolved
  hardware/service configuration, and exact source-code hashes.
- Ignored `results/runs/burstgpt-integrated/`: request-level lifecycle records.

```bash
python scripts/prepare_burstgpt.py
python experiments/run_burstgpt.py
```

Counts include every arrival, including rejection and unfinished work.
Fixed-window throughput excludes drain completions. TTFT is simulated first
token minus true arrival; networking and tokenization are not modeled.
See [the handoff](../../docs/week1-integration.md) for integration fixes and limits.
