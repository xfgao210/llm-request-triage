from dataclasses import replace
import json

import pytest

from llm_triage.scheduling import BurstSwitchConfig, BurstSwitchScheduler
from llm_triage.simulation import Action, HardwareConfig, Request, ServiceProfile, TriageKernel, simulate


class DropPolicy:
    def decide(self, state):
        return Action.DROP


def snapshot(now):
    kernel = TriageKernel()
    state = kernel.reset([Request("a", 0, 16, 1, 100)])
    return replace(state, now=now)


def test_causal_switch_hysteresis_dwell_and_recovery():
    switch = BurstSwitchScheduler(DropPolicy(), BurstSwitchConfig(min_dwell_s=2))
    for time in range(11):
        switch.observe_arrival(time)
    assert switch.decide(snapshot(10)).action is Action.ADMIT  # warmup
    for time in (11, 11, 11):
        switch.observe_arrival(time)
    assert switch.decide(snapshot(11)).action is Action.DROP
    assert switch.decide(snapshot(12)).action is Action.DROP  # dwell
    assert switch.decide(snapshot(13)).action is Action.ADMIT
    assert [row["mode"] for row in switch.switches] == ["custom", "fcfs"]
    assert switch.switches[0]["recent_rps"] == 3
    assert switch.switches[0]["reference_rps"] == 1


def test_hysteresis_band_and_ordering():
    class Shortest(DropPolicy):
        def priority(self, candidate):
            return candidate.input_tokens

    switch = BurstSwitchScheduler(Shortest(), BurstSwitchConfig(min_dwell_s=0))
    for time in (11, 11, 11):
        switch.observe_arrival(time)
    state = snapshot(11)
    short = replace(state.candidate, request_id="short", input_tokens=1)
    state = replace(state, queue_requests=(state.candidate, short))
    assert switch.decide(state).request_id == "short"
    switch.observe_arrival(12)
    switch.observe_arrival(12)
    switch.decide(replace(state, now=12))
    assert switch.mode == "custom"  # ratio 2 remains above exit


def test_arrival_hook_counts_requests_while_gpu_is_full_and_resets():
    switch = BurstSwitchScheduler(DropPolicy())
    requests = [Request(str(i), i * 0.1, 16, 2, 20) for i in range(5)]
    profile = ServiceProfile(prefill_overhead_s=1, prefill_quadratic_s=0,
                             decode_rate=1, decode_context_s=0)
    hardware = HardwareConfig(max_batch_size=1)
    first = simulate(requests, switch, hardware, profile)
    assert switch.last_arrival_s == 0.4
    assert switch.decisions["fcfs"] == 5
    again = simulate(requests, switch, hardware, profile)
    assert switch.decisions["fcfs"] == 5
    assert first.slo_attainment == again.slo_attainment


def test_reject_future_and_out_of_order_arrivals():
    switch = BurstSwitchScheduler(DropPolicy())
    switch.observe_arrival(2)
    with pytest.raises(ValueError):
        switch.observe_arrival(1)
    with pytest.raises(ValueError):
        switch.decide(snapshot(1))


@pytest.mark.parametrize("options", [{"exit_ratio": 2}, {"recent_window_s": 0},
                                     {"min_dwell_s": -1}, {"min_recent_arrivals": True}])
def test_invalid_config(options):
    with pytest.raises(ValueError):
        BurstSwitchConfig(**options)


def test_evaluation_includes_fixed_and_switching_dqn_with_paired_cohorts():
    from experiments.train_dqn import evaluate
    from llm_triage.rl.dqn import DQN
    from llm_triage.workload.transforms import ReplayWindow

    replay = ReplayWindow(tuple(Request(str(i), time, 16, 2, time + 5)
                                for i, time in enumerate([0, 11, 11, 11, 14])), 20, 0, 1)
    model = DQN()
    before = model.copy_weights()
    rows = evaluate({"dqn": model}, [(0, None, replay)], HardwareConfig(), ServiceProfile(),
                    2, dict(max_defers=5, defer_retry_s=0.25), baselines=True,
                    burst_switch_config=BurstSwitchConfig())
    assert {r["policy"] for r in rows} == {"fcfs", "length_aware", "static_slo_aware", "dqn", "switch_dqn"}
    assert {r["arrivals"] for r in rows} == {5}
    hybrid = next(r for r in rows if r["policy"] == "switch_dqn")
    assert hybrid["switch_count"] >= 1
    assert hybrid["custom_decisions"] > 0
    assert json.loads(hybrid["switch_trace_json"])[0]["mode"] == "custom"
    assert all((before[key] == model.params[key]).all() for key in before)
