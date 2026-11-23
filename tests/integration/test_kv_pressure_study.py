import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")

from experiments.kv_pressure_study import BudgetReplay, CappedDQN, DROP_DENOMINATOR
from experiments.scheduling_study import SlackAdmission
from llm_triage.rl.masking import ACTION_TO_INDEX
from llm_triage.simulation import Action, EventType, HardwareConfig, Request, ServiceProfile


class PreferDrop:
    def act(self, observation, mask):
        for action in (Action.DROP, Action.ADMIT, Action.DEFER):
            index = ACTION_TO_INDEX[action]
            if mask[index]:
                return index
        raise AssertionError("The policy should wait when every action is masked")


@pytest.mark.parametrize("learned", [False, True])
def test_drop_allowance_is_earned_only_from_observed_arrivals(learned):
    # The strongest possible drop preference still has to earn each allowance.
    policy = (CappedDQN(PreferDrop(), candidate_selector="length_aware") if learned
              else SlackAdmission(float("inf")))
    requests = [Request(str(i), i * .75, 12, 1, i * .75 + 5) for i in range(41)]
    kernel = BudgetReplay(policy, HardwareConfig(max_batch_size=32), ServiceProfile(), 16, {})
    result = kernel.run(requests)

    arrivals = drops = 0
    earned_at = []
    for event in result.events:
        if event.event_type is EventType.ARRIVAL:
            arrivals += 1
        if event.event_type is EventType.DROP:
            drops += 1
            earned_at.append(arrivals)
        assert drops <= arrivals // DROP_DENOMINATOR
    assert earned_at == [20, 40]


def test_budget_exhaustion_waits_when_admit_and_defer_are_both_blocked():
    policy = CappedDQN(PreferDrop(), candidate_selector="length_aware")
    hardware = HardwareConfig(max_batch_size=32, kv_block_size=16, kv_total_blocks=3)
    requests = [Request("running", 0, 25, 22, 5), Request("blocked", .1, 20, 1, .108)]
    result = BudgetReplay(policy, hardware, ServiceProfile(), 16, {}).run(requests)
    assert [e.request_id for e in result.events if e.event_type is EventType.COMPLETION] == [
        "running", "blocked"
    ]
    assert all(e.event_type is not EventType.DROP for e in result.events)
