import pytest

from experiments.evaluate_traffic_multiplier import scaled_cases
from llm_triage.workload.burstgpt import TraceRow
from llm_triage.workload.transforms import replay_window


def test_true_eightfold_native_traffic_preserves_requests_and_slo():
    rows = [TraceRow(0, 100, 16, 7), TraceRow(1, 110, 32, 100), TraceRow(2, 110, 48, 300)]
    native = replay_window(rows, slo_s=5, bounds=(100, 1000))
    cases = scaled_cases([(0, None, native)], [1, 8])
    assert [case[2].horizon_s for case in cases] == [900, 112.5]
    for _, multiplier, replay in cases:
        assert len(replay.requests) == len(native.requests)
        for before, after in zip(native.requests, replay.requests):
            assert (after.request_id, after.input_tokens, after.output_tokens) == (
                before.request_id, before.input_tokens, before.output_tokens)
            assert after.arrival_time == before.arrival_time / multiplier
            assert after.slo_deadline - after.arrival_time == pytest.approx(5)
        assert len(replay.requests) / replay.horizon_s == multiplier * len(native.requests) / native.horizon_s


def test_absolute_rps_replay_cannot_be_mistaken_for_native_multiplier():
    rows = [TraceRow(0, 0, 16, 7), TraceRow(1, 10, 16, 7)]
    scaled = replay_window(rows, target_rps=9.6, bounds=(0, 300))
    with pytest.raises(ValueError, match="native timing"):
        scaled_cases([(0, 9.6, scaled)], [8])


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), True])
def test_reject_invalid_multiplier(multiplier):
    native = replay_window([TraceRow(0, 0, 16, 1), TraceRow(1, 10, 16, 1)], bounds=(0, 300))
    with pytest.raises(ValueError):
        scaled_cases([(0, None, native)], [multiplier])
