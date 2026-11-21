import pytest

from experiments.kv_pressure_study import BudgetReplay, CappedDQN
from experiments.pressure_policy_selector import CappedSelector, make_replay, rollout_score, verify_drop_prefix
from experiments.scheduling_study import SlackAdmission
from llm_triage.rl.masking import ACTION_TO_INDEX
from llm_triage.scheduling import BurstSwitchConfig, BurstSwitchScheduler
from llm_triage.scheduling.policy_selector import SELECTOR_FEATURES, LearnedSwitchScheduler
from llm_triage.simulation import Action, EventType, HardwareConfig, Request, ServiceProfile


class PreferDrop:
    def act(self, observation, mask):
        for action in (Action.DROP, Action.ADMIT, Action.DEFER):
            if mask[ACTION_TO_INDEX[action]]:
                return ACTION_TO_INDEX[action]
        raise AssertionError("An all-masked candidate should wait")


def learned(probability):
    custom = CappedDQN(PreferDrop(), candidate_selector="length_aware")
    tree = dict(schema_version=1, features=list(SELECTOR_FEATURES), root=dict(probability=probability))
    selector = LearnedSwitchScheduler(custom, tree, config=BurstSwitchConfig(min_dwell_s=0))
    return CappedSelector(custom, selector)


def test_hybrid_obeys_causal_drop_cap_and_uses_dqn():
    policy = learned(1)
    requests = [Request(str(i), i * 0.75, 12, 1, i * 0.75 + 5) for i in range(41)]
    result = make_replay(policy, HardwareConfig(), ServiceProfile(), 16, {}, policy.selector).run(requests)
    verify_drop_prefix(result.events)
    arrivals = 0
    earned = []
    for event in result.events:
        arrivals += event.event_type is EventType.ARRIVAL
        if event.event_type is EventType.DROP:
            earned.append(arrivals)
    assert earned == [20, 40]
    assert policy.selector.decisions["custom"] > 0
    assert policy.selector.last_arrival_s == 30


def test_constant_admit_tree_matches_original_length_admit_including_blocked_candidates():
    requests = [Request("long", 0, 28, 8, 5), Request("short", 0, 12, 1, 5),
                Request("blocked", 0.05, 20, 1, 5)]
    hw = HardwareConfig(kv_total_blocks=3)
    original = BudgetReplay(SlackAdmission(float("-inf")), hw, ServiceProfile(), 16, {}).run(requests)
    policy = learned(0)
    replayed = make_replay(policy, hw, ServiceProfile(), 16, {}, policy.selector).run(requests)
    assert [(r.request_id, r.state, r.completion_time) for r in original.requests] == [
        (r.request_id, r.state, r.completion_time) for r in replayed.requests]
    starts = [e.request_id for e in replayed.events if e.event_type is EventType.SERVICE_START]
    assert starts[0] == "short"  # Common length order, not arrival tie order.


def test_fork_keeps_already_earned_allowance_and_does_not_mutate_parent():
    policy = CappedDQN(PreferDrop(), candidate_selector="length_aware")
    requests = [Request(str(i), 0, 12, 1, 5) for i in range(20)]
    requests += [Request("outside", 50, 12, 1, 55)]
    kernel = make_replay(policy, HardwareConfig(), ServiceProfile(), 16, {})
    kernel.reset(requests)
    assert kernel._next == 20 and kernel.drop_allowed
    before = kernel.events()
    rollout_score(kernel, policy, 10)
    assert kernel.events() == before and kernel.dropped_count == 0
    branch = kernel.fork(arrivals_before_s=10, priority_fn=policy.priority)
    branch.policy = policy
    while not branch.is_terminal():
        branch.apply(branch.choose_action(branch.snapshot()))
    verify_drop_prefix(branch.events())
    assert branch.dropped_count == 1  # Cap remains earned from the original window prefix.
    assert len(branch.requests) == 20


def test_exhausted_budget_waits_for_capacity_without_invalid_actions():
    policy = learned(1)
    hw = HardwareConfig(kv_total_blocks=3)
    requests = [Request("running", 0, 25, 22, 5), Request("blocked", 0.1, 20, 1, 0.108)]
    result = make_replay(policy, hw, ServiceProfile(), 16, {}, policy.selector).run(requests)
    verify_drop_prefix(result.events)
    assert len([e for e in result.events if e.event_type is EventType.COMPLETION]) == 2
    assert not result.dropped


def test_counterfactual_from_blocked_dqn_state_reestablishes_admit_decision_point():
    custom = CappedDQN(PreferDrop(), candidate_selector="length_aware")
    hw = HardwareConfig(kv_total_blocks=3)
    kernel = make_replay(custom, hw, ServiceProfile(), 16, {})
    kernel.reset([Request("running", 0, 25, 22, 5), Request("blocked", 0.1, 20, 1, 5)])
    kernel.apply(Action.ADMIT)
    assert not kernel.snapshot().candidate_fits()
    score = rollout_score(kernel, SlackAdmission(float("-inf")), 10)
    assert isinstance(score, int)
    assert not kernel.is_terminal()
