import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")

from experiments.scheduling_study import RATES, budget_comparison, choose_rates, make_stream
from experiments.train_dqn import policy_options
from llm_triage.rl import RLPolicyCallback, TriageEnv
from llm_triage.simulation import Request
from llm_triage.workload.transforms import ReplayWindow


def test_common_selector_matches_training_and_inference():
    replay = ReplayWindow((Request("long", 0, 512, 1, 5), Request("short", 0, 16, 1, 5)), 1, 0, 1)
    options = policy_options(replay, dict(max_defers=5, defer_retry_s=.25),
                             candidate_selector="length_aware")
    env = TriageEnv(replay, **options)
    _, info = env.reset()
    assert info["candidate_id"] == "short"
    callback = RLPolicyCallback(lambda obs, mask: 0, **options)
    assert min(env.snapshot().candidates, key=callback.priority).request_id == "short"


def test_capacity_rule_uses_both_conditions_and_handles_edge_rates():
    rows = [dict(capacity="kv_stress", policy="fcfs", rate=rate,
                 slo_attainment=attainment, mean_queue_in_window=queue)
            for rate, attainment, queue in zip(RATES, [.98, .96, .90, .89, .70], [0, 0, .1, .2, 5])]
    assert choose_rates(rows) == ([2.4, 4.8, 9.6], 9.6)
    rows[-1]["mean_queue_in_window"] = .5
    assert choose_rates(rows) == ([1.2, 2.4, 4.8], None)


def test_budget_rule_is_selected_on_validation_and_checked_on_evaluation():
    def row(name, goodput, drop):
        return dict(rate=2.4, policy=name, attainment=.6, goodput=goodput, drop=drop)
    validation = [row("length_aware", 1.2, 0), row("slack_0", 1.4, .03)]
    evaluation = [row("length_aware", 1.2, 0), row("slack_0", .9, .08), row("dqn_7", 1.3, .04)]
    result = next(r for r in budget_comparison(validation, evaluation) if r["budget"] == .05)
    assert result["rule"] == "slack_0"
    assert not result["both_within_budget"]


def test_continuous_stream_does_not_reset_deadlines_or_request_order():
    first = ReplayWindow((Request("a", .5, 16, 1, 5.5),), 2, 11, 1)
    second = ReplayWindow((Request("b", 1, 32, 2, 6),), 3, 20, 1)
    replay, phases = make_stream([("pre", first), ("pressure", second)])
    assert [(r.request_id, r.arrival_time_s, r.deadline_s) for r in replay.requests] == [
        ("a", .5, 5.5), ("b", 3, 8)
    ]
    assert replay.horizon_s == 5
    assert phases["pressure"] == {"start": 2, "end": 5, "ids": {"b"}}
