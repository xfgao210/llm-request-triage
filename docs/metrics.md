# Evaluation Metrics

The evaluation engine consumes request lifecycle events and produces one final
record per request plus one summary per policy run. It does not depend on a
specific simulator or scheduling policy.

## Required lifecycle

The expected lifecycle is:

```text
arrival -> decision -> service_start -> first_token -> completion
                    \-> drop
                    \-> unfinished at episode_end
```

`arrival` events require `deadline_s`. The engine marks every request that has
not completed or dropped by `episode_end` as unfinished.

## Primary definitions

For all requests arriving within the configured measurement window:

```text
TTFT = first_token_time - arrival_time

slo_met = completed and completion_time <= deadline
slo_attainment = requests meeting SLO / all arrivals

window_throughput_rps =
    completions inside measurement window / measurement window seconds

window_goodput_rps =
    on-time completions inside measurement window / measurement window seconds

drop_rate = dropped requests / all arrivals
unfinished_rate = unfinished requests / all arrivals
```

Dropped and unfinished requests remain in the SLO denominator. TTFT and
completion-latency percentiles are conditional on requests that produced the
corresponding timestamp, so they must be interpreted beside SLO attainment,
goodput, and drop rate.

## Integration API

Backends should emit `LifecycleEvent` values and call:

```python
from llm_triage.metrics import evaluate_events

result = evaluate_events(
    events,
    measurement_window_s=120.0,
)
```

Alternatively, callers with complete `RequestRecord` values can call
`summarize_records` directly.

The optional `queue_length` event detail produces `max_queue_length`.
`decision_overhead_s` on decision events produces scheduler-overhead metrics.

