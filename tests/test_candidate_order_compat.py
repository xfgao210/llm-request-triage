import pytest

from experiments.train_dqn import configured_candidate_order, evaluate, policy_options
from llm_triage.simulation import HardwareConfig, Request, ServiceProfile
from llm_triage.workload.transforms import ReplayWindow


@pytest.mark.parametrize("config, expected", [
    ({}, "fcfs"),
    ({"candidate_selector": "length_aware"}, "length_aware"),
    ({"rl_candidate_order": "length_aware"}, "length_aware"),
    ({"candidate_selector": "length_aware", "rl_candidate_order": "length_aware"}, "length_aware"),
])
def test_saved_configuration_names_remain_compatible_without_mutation(config, expected):
    original = dict(config)
    assert configured_candidate_order(config) == expected
    assert config == original


@pytest.mark.parametrize("config", [
    {"candidate_selector": "length_aware", "rl_candidate_order": "fcfs"},
    {"candidate_selector": "unknown"},
    {"rl_candidate_order": "unknown"},
])
def test_ambiguous_or_invalid_orders_fail(config):
    with pytest.raises(ValueError):
        configured_candidate_order(config)


@pytest.mark.parametrize("keyword", ["candidate_selector", "candidate_order"])
def test_both_public_arguments_evaluate_with_the_requested_order(keyword):
    class Admit:
        def act(self, observation, mask):
            return 0

    replay = ReplayWindow((Request("long", 0, 1000, 1, 5), Request("short", 0, 1, 1, .1)), 10, 0, 1)
    decisions = dict(max_defers=5, defer_retry_s=.25)
    profile = ServiceProfile(prefill_overhead_s=.01, prefill_rate=1000, prefill_quadratic_s=0)
    options = policy_options(replay, decisions, **{keyword: "length_aware"})
    assert options["candidate_selector"] == "length_aware"
    rows = evaluate({"agent": Admit()}, [(0, None, replay)], HardwareConfig(max_batch_size=1),
                    profile, 1, decisions, **{keyword: "length_aware"})
    assert rows[0]["slo_attainment"] == 1
    fcfs = evaluate({"agent": Admit()}, [(0, None, replay)], HardwareConfig(max_batch_size=1),
                    profile, 1, decisions, **{keyword: "fcfs"})
    assert fcfs[0]["slo_attainment"] == .5
