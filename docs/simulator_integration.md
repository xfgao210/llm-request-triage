# Simulator Integration Contract

The evaluation and baseline code is independent of a particular simulator.
The shared implementation provides `llm_triage.simulation.SimulatorExecutor`,
implementing `PolicyExecutor`.
The integrated BurstGPT entry point is `experiments/run_burstgpt.py`; see
[Week 1 integration](week1-integration.md) for the shared event schema and timing fixes.

## Inputs to the simulator

For every decision point, construct a `SchedulerSnapshot` containing:

- the current simulated time;
- every currently eligible request as a `CandidateView`; and
- sanitized resource state as a `ResourceView`.

`CandidateView` intentionally has no true output-length field. The simulator
may use the true length after admission, but schedulers cannot access it.

Call:

```python
decision = scheduler.decide(snapshot)
```

No decision is returned only when the candidate set is empty. Otherwise, apply
the selected request ID and action to the simulator.

## Events emitted by the simulator

The adapter returns a chronological iterable of `LifecycleEvent` values:

```text
ARRIVAL        details: deadline_s, optional input_tokens and queue_length
DECISION       details: action, optional decision_overhead_s
SERVICE_START
FIRST_TOKEN
COMPLETION
DROP
EPISODE_END
```

The same workload, service profile, output-length estimator, SLO construction,
measurement window, and random seed must be restored before every policy run.

## Adapter shape

```python
class ExampleExecutor:
    def __init__(self, requests, simulation_config):
        self.requests = requests
        self.simulation_config = simulation_config

    def run(self, policy_name, scheduler):
        simulator = build_fresh_simulator(
            self.requests,
            self.simulation_config,
        )
        yield from simulator.run(scheduler)
```

Then execute all baselines through one adapter:

```python
from llm_triage.evaluation import EvaluationConfig, evaluate_policy_matrix
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import HardwareConfig, ServiceProfile, SimulatorExecutor

executor = SimulatorExecutor(
    requests,
    hardware=HardwareConfig(),
    profile=ServiceProfile(),
)

results = evaluate_policy_matrix(
    baseline_schedulers(),
    executor,
    EvaluationConfig(measurement_window_s=120.0),
)
```

Use `write_evaluation_artifacts` to generate the request CSV, summary CSV, and
resolved configuration JSON.
