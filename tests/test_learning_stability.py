"""Time-aware bootstrapping and causal update-budget gates."""

import numpy as np
import pytest

from llm_triage.rl.dqn import DQN, ReplayBuffer
from llm_triage.rl.observation import FULL_FEATURE_NAMES
from llm_triage.rl.environment import ACTIONS
from llm_triage.rl.update_schedule import OnlineUpdateSchedule, adaptation_preset


def test_elapsed_discount_does_not_decay_at_zero_time():
    model = DQN(gamma=.9, discount_mode="elapsed_time", discount_time_unit_s=2)
    np.testing.assert_allclose(model.bootstrap_discounts([0, 1, 2, 4]), [1, .9**.5, .9, .9**2])
    # Splitting an interval preserves cumulative bootstrap decay.
    assert model.bootstrap_discounts(1) * model.bootstrap_discounts(3) == pytest.approx(model.bootstrap_discounts(4))
    legacy = DQN(gamma=.9)
    np.testing.assert_allclose(legacy.bootstrap_discounts([0, 1, 20]), [.9]*3)
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="durations"):
            model.bootstrap_discounts(bad)


def test_duration_buffer_and_checkpoint_roundtrip(tmp_path):
    state = np.zeros(len(FULL_FEATURE_NAMES))
    mask = np.ones(len(ACTIONS), dtype=bool)
    buffer = ReplayBuffer(4, 7, include_elapsed_time=True)
    with pytest.raises(ValueError, match="elapsed_s"):
        buffer.add(state, 0, 1, state, False, mask)
    for duration in (0, .5, 1, 10):
        buffer.add(state, 0, 1, state, False, mask, elapsed_s=duration)
    batch = buffer.sample(4)
    assert len(batch) == 7 and sorted(batch[-1]) == [0, .5, 1, 10]
    model = DQN(discount_mode="elapsed_time", gamma=.95, discount_time_unit_s=2)
    assert np.isfinite(model.train_batch(batch))
    with pytest.raises(ValueError, match="recorded transition durations"):
        model.train_batch(batch[:6])
    path = tmp_path / "time-aware.npz"
    model.save(path, dict(features=list(FULL_FEATURE_NAMES), actions=[a.value for a in ACTIONS],
                         config={"agent": {"gamma": .95, "discount_mode": "elapsed_time", "discount_time_unit_s": 2}}))
    loaded, _ = DQN.load(path)
    assert loaded.discount_mode == "elapsed_time" and loaded.gamma == .95 and loaded.discount_time_unit_s == 2
    for key in model.params:
        np.testing.assert_array_equal(model.params[key], loaded.params[key])


def test_schedule_requires_time_feedback_and_honors_cap():
    schedule = OnlineUpdateSchedule(interval=4, min_seconds=1, min_outcomes=32, max_updates=2)
    assert not schedule.ready(steps=4, now_s=0, outcomes=100, updates=0)
    assert not schedule.ready(steps=4, now_s=1, outcomes=31, updates=0)
    assert not schedule.ready(steps=5, now_s=1, outcomes=32, updates=0)
    assert schedule.ready(steps=8, now_s=1, outcomes=32, updates=0)
    schedule.mark(1, 32)
    assert not schedule.ready(steps=12, now_s=1, outcomes=100, updates=1)
    assert not schedule.ready(steps=12, now_s=2, outcomes=63, updates=1)
    assert schedule.ready(steps=12, now_s=2, outcomes=64, updates=1)
    assert not schedule.ready(steps=16, now_s=10, outcomes=100, updates=2)
    conservative = adaptation_preset("conservative")
    assert conservative["learning_rate"] < adaptation_preset("legacy")["learning_rate"]
    conservative["max_updates"] = 999
    assert adaptation_preset("conservative")["max_updates"] == 100
    with pytest.raises(ValueError, match="preset"):
        adaptation_preset("unknown")


@pytest.mark.parametrize("kwargs", [{"interval": 0}, {"min_seconds": -1}, {"min_outcomes": -1}, {"max_updates": 0}])
def test_invalid_schedule_rejected(kwargs):
    with pytest.raises(ValueError):
        OnlineUpdateSchedule(**kwargs)
