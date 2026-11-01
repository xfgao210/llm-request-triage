# Week 1 FCFS baseline — validation only

Hengyi Yin, November 1, 2026.

## Setup

Pinned BurstGPT ChatGPT API rows, chronological 60%/20%/20% partitions, and
eight validation windows of 256 requests. Each window is replayed at 0.6, 1.2,
and 2.4 requests/s: **24 runs, 6,144 request executions**. These are 2,048 unique
validation requests replayed at three rates, not 6,144 distinct source requests.
Train/test window preparation is deterministic; test windows were not evaluated.

Backend: existing single-worker Python simulator via the shared FCFS interface.
Service time is **assumed**, `input_tokens/2000 + output_tokens/400` seconds.
This is not the measured GPU profile and no inference runs on a GPU here.
Every request has a 5-second completion deadline relative to its aligned arrival.

## Results

Each value is the arithmetic mean across the eight equal-sized validation windows.
P95 is the mean of per-window P95 values, not a pooled P95.

| Target RPS | Overall SLO attainment | Window throughput (req/s) | Window goodput (req/s) | Mean window P95 latency (s) |
|---:|---:|---:|---:|---:|
| 0.6 | 91.26% | 0.599 | 0.547 | 2.958 |
| 1.2 | 84.28% | 1.176 | 1.010 | 5.902 |
| 2.4 | 68.85% | 2.149 | 1.648 | 21.412 |

All arrivals completed after drain; there were no drops or unresolved requests.
Higher load lowers timely completion and increases latency in this reference
model. This establishes an FCFS integration baseline, not a claim about learned
policies, measured hardware gains, independent random-seed variability, or
production serving performance.

## Evidence and reproduction

- `summary.csv`: 24 per-window/rate summaries, including counts and drain metrics.
- `data_manifest.json`: source revision/hash, filtering counts, partition cuts,
  selected-window bounds and source-row hashes.
- `provenance.json`: resolved configuration, code hashes, data hashes, semantics.
- Ignored `results/runs/week1-fcfs/`: per-request and queue-event records.

```bash
python scripts/prepare_burstgpt.py
python experiments/run_fcfs.py
```

See [the integration handoff](../../docs/week1-hengyi.md) for tests, interfaces,
and the boundaries with the simulator/metrics workstreams.
