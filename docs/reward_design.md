# Reward Design

**Status:** Initial specification; weights are not tuned.

The reward model favors on-time completion, treats late computation as more
costly than proactive dropping, assigns a bounded cost to deferral, and uses a
terminal objective to prevent the policy from learning to drop every request.

## Transition reward

For one completed transition:

```text
on-time completion:
    + completion_reward

late completion:
    - late_penalty
    - wasted_service_weight * (service_time / request_SLO_window)

drop:
    - drop_penalty

defer:
    - defer_penalty_per_second * deferred_seconds
```

The initial weights guarantee the intended qualitative ordering for a typical
request:

```text
on-time completion > proactive drop > late completion with wasted compute
```

Service waste is divided by the request's SLO window to keep its scale
comparable when requests have different deadline classes.

## Terminal reward

After an episode:

```text
+ attainment_weight * (SLO-met requests / arrivals)
+ completion_weight * (completed requests / arrivals)
- unfinished_weight * (unfinished requests / arrivals)
- drop_all_penalty, if every arrival was dropped
```

The completion-fraction term is an episode-normalized throughput guard. It
prevents improved conditional latency or attainment from hiding a policy that
serves almost no work. Actual throughput and goodput in requests per second
remain evaluation metrics and are logged separately.

## Initial configuration

| Component | Magnitude |
|---|---:|
| On-time completion reward | 1.0 |
| Late-completion penalty | 1.0 |
| Normalized wasted-service penalty | 0.5 |
| Drop penalty | 0.2 |
| Defer penalty per second | 0.02 |
| Terminal SLO-attainment reward | 1.0 |
| Terminal completion-fraction reward | 0.5 |
| Terminal unfinished-fraction penalty | 0.5 |
| Drop-all penalty | 2.0 |

These values are defaults for implementation testing, not final
hyperparameters. Tuning must use training and validation workloads only. The
held-out evaluation windows must remain untouched until weights are frozen.

## Information boundary

Transition rewards are calculated only after the corresponding outcome is
known. Their inputs exclude:

- true output token count;
- future arrivals;
- future queue state; and
- service outcomes that have not completed.

The environment may use hidden truth to advance simulation, but the policy
observation remains separately sanitized.

## Required logging

Each reward call returns a component-level breakdown plus the total. Training
logs should retain at least:

- completion reward;
- late penalty;
- wasted-service penalty;
- drop penalty;
- defer penalty;
- terminal attainment and completion rewards;
- unfinished and drop-all penalties; and
- total episode return.

Component logging is required to diagnose reward domination and degenerate
policies before interpreting aggregate return.

## Simulator integration

`KernelRewardAdapter` consumes the canonical simulator `Transition` and its
lifecycle events. The Gymnasium environment should initialize it with the
arrival events emitted by `TriageKernel.reset()`, then call exactly one of
`reward()` or `reward_breakdown()` for every transition returned by
`TriageKernel.apply()`.

The adapter reconstructs only outcome facts that have already been emitted:
arrival/deadline, first service start, completion, drop, and elapsed defer time.
It never reads true output length or future workload state. `last_breakdown`
and `cumulative_breakdown` provide the component-level logging required above.
Until lifecycle events expose attributed GPU time, the wasted-service term uses
the observable interval from first service start through late completion.
