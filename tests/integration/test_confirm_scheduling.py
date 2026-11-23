import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")

from experiments.confirm_scheduling import QueueSlackReplay, QueueSlackRule, choose_rule, cutoff, select_cohorts
from experiments.scheduling_study import SlackAdmission
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import EventType, HardwareConfig, Request, ServiceProfile, simulate


def test_cohorts_keep_complete_contiguous_blocks_and_ignore_empty_windows():
    windows, empty = [], []
    specs = [("train", "training", 30), ("validation", "validation", 12),
             ("test", "mixed_traffic", 18), ("test", "peak_heavy", 18)]
    for split, group, size in specs:
        for index in range(size):
            key = len(windows) + len(empty)
            item = dict(split=split, group=group, window=key, start_timestamp_s=key * 300,
                        end_timestamp_s=(key + 1) * 300, requests=200)
            if split == "test" and index in (1, 7):
                empty.append(dict(item, requests=0))
            else:
                windows.append(item)
    picked, blocks = select_cohorts(dict(windows=windows, empty_windows=empty))
    assert len(picked) == 24 + 6 + 2 * 4 * 3
    assert len({w["window"] for w in picked}) == len(picked)
    for group, candidates in blocks.items():
        assert len(candidates) == 4
        for block in candidates:
            assert all(w["group"] == group and w["requests"] >= 128 for w in block)
            assert [w["window"] for w in block] == list(range(block[0]["window"], block[0]["window"] + 3))


def test_validation_threshold_selection_respects_budget_and_excludes_learned_policies():
    rows = [
        dict(policy="length_aware", rate=1.2, drop=0, goodput=0.8),
        dict(policy="slack_-1", rate=1.2, drop=.04, goodput=.9),
        dict(policy="slack_1", rate=1.2, drop=.07, goodput=1.0),
        dict(policy="dqn_7", rate=1.2, drop=.01, goodput=1.1),
    ]
    assert choose_rule(rows, 1.2, .05) == "slack_-1"
    assert cutoff("slack_-1") == -1
    assert choose_rule(rows, 1.2, 0) == "length_aware"
    assert cutoff("length_aware") == float("-inf")


@pytest.mark.parametrize("policy", ["fcfs", "length_aware", "static_slo_aware", "slack"])
def test_skipping_blocked_admits_preserves_service_and_drops(policy):
    hardware = HardwareConfig(max_batch_size=4, kv_block_size=16, kv_total_blocks=3)
    requests = [
        Request("running", 0, 25, 80, 5),
        Request("blocked", .02, 20, 3, .55),
        Request("later", .03, 20, 8, 5),
        Request("fits", .4, 8, 3, 5),
    ]
    if policy == "slack":
        scheduler = SlackAdmission(-1)
        drop_blocked = scheduler.should_drop
    else:
        scheduler = baseline_schedulers()[policy]
        drop_blocked = None
        if policy == "static_slo_aware":
            drop_blocked = lambda candidate: candidate.estimated_slack_s() < 0
    before = simulate(requests, scheduler, hardware)
    after = simulate(requests, scheduler, hardware, skip_blocked_admits=True,
                     drop_blocked_candidate=drop_blocked)

    assert after.iterations == before.iterations
    assert after.preemptions == before.preemptions
    assert [e for e in after.events if e.event_type is not EventType.DECISION] == [
        e for e in before.events if e.event_type is not EventType.DECISION
    ]
    expected_drops = ["blocked"] if drop_blocked is not None else []
    assert [e.request_id for e in after.events if e.event_type is EventType.DROP] == expected_drops
    assert len(after.events) < len(before.events)


def test_queue_rule_replay_preserves_the_visible_queue_after_each_blocked_admit():
    hardware = HardwareConfig(max_batch_size=4, kv_block_size=16, kv_total_blocks=3)
    profile = ServiceProfile()
    rule = QueueSlackRule(0., float("-inf"))
    estimate = 16
    requests = [
        Request("running", 0, 25, 80, 5),
        *(Request(f"blocked-{index}", .02, 20 + index, 3, .4) for index in range(5)),
        Request("fits", .4, 8, 3, 5),
    ]
    before = simulate(requests, rule, hardware, profile, estimated_output_tokens=estimate,
                      record_iterations=False)
    after = QueueSlackReplay(rule, hardware, profile, estimate, {}).run(requests)

    assert [e for e in after.events if e.event_type is not EventType.DECISION] == [
        e for e in before.events if e.event_type is not EventType.DECISION
    ]
    assert any(e.event_type is EventType.DROP for e in after.events)
    assert any(e.event_type is EventType.COMPLETION for e in after.events)
    assert len(after.events) < len(before.events)
