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
    - wasted_service_weight * (attributed_GPU_time / request_SLO_window)

drop:
    - drop_penalty

defer:
    - defer_penalty_per_second * configured_retry_interval
```

The initial weights guarantee the intended qualitative ordering for a typical
request:

```text
on-time completion > proactive drop > late completion with wasted compute
```

Attributed GPU time comes from the immutable request outcomes emitted by the
simulator. It is divided by the request's SLO window to keep its scale
comparable when requests have different deadline classes.

The defer cost is charged immediately on the `DEFER` transition. It covers
exactly one configured retry interval; later service, capacity, or queue waits
cannot accumulate against the earlier action. Repeated explicit defer actions
each incur one bounded cost.

## Terminal reward

After an episode:

```text
+ attainment_weight * (SLO-met requests / arrivals)
+ completion_weight * (served requests / arrivals)
- unfinished_weight * (unfinished requests / arrivals)
- drop_all_penalty, if every arrival was dropped
```

The served-fraction term includes both on-time and late completions and is an
episode-normalized throughput guard. It
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

`RewardModel` is the environment-facing protocol. It exposes the stable
`reward(Transition) -> float` contract from `TECH_DESIGN.md`; the Gymnasium
environment does not depend on reward formulas or event-accounting details.

`AsymmetricRewardModel` is the canonical implementation. It consumes the
immutable `RequestOutcome` objects included in canonical simulator transitions,
including terminal state, SLO timestamps, and attributed GPU time. The
environment integration is therefore:

```python
reward_model: RewardModel = AsymmetricRewardModel(
    config,
    defer_retry_s=kernel.defer_retry_s,
)
kernel.reset(requests)

transition = kernel.apply(action)
reward = reward_model.reward(transition)
```

The Gymnasium environment calls exactly one of `reward()` or
`reward_breakdown()` for every transition returned by `TriageKernel.apply()`.
It must not duplicate reward weights or formulas inside `step()`.
`last_breakdown` and `cumulative_breakdown` provide the component-level logging
required above.
