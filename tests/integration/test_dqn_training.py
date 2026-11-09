"""Week 2 regressions across the Gymnasium, kernel, and checkpoint boundaries."""

import csv
import json

import pytest

gymnasium = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from gymnasium.utils.env_checker import check_env

from experiments.train_dqn import run
from llm_triage.rl.environment import DecisionConfig, GreedyScheduler, RewardConfig, TriageEnv
from llm_triage.simulation import EventType, HardwareConfig, Request, ServiceProfile, simulate
from llm_triage.workload.burstgpt import sha256
from llm_triage.workload.transforms import ReplayWindow


def make_env(requests, hardware=None, decisions=DecisionConfig(0.25, 5)):
    return TriageEnv(
        ReplayWindow(tuple(requests), 3.0, 0.0, 1.0), hardware or HardwareConfig(),
        ServiceProfile(), 8, decisions, RewardConfig(),
    )


def test_gymnasium_and_observation_hides_current_and_future_truth():
    env = make_env([Request("a", 0, 16, 1, 2), Request("b", 1, 16, 1, 3)])
    changed = make_env([Request("a", 0, 16, 200, 2), Request("b", 1, 8192, 500, 3)])
    check_env(env, skip_render_check=True)
    first, info = env.reset(seed=3)
    second, other = changed.reset(seed=3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(info["action_mask"], other["action_mask"])
    assert env.observation_space.contains(first)


def test_defer_mask_blocks_terminal_defer_and_drop_all_is_penalized():
    env = make_env([Request("a", 0, 16, 1, 2)], decisions=DecisionConfig(0.25, 2))
    _, info = env.reset()
    assert info["action_mask"].tolist() == [True, True, True]
    _, defer_reward, done, _, info = env.step(1)
    assert not done and env.snapshot.now_s == pytest.approx(0.25)
    assert info["action_mask"].tolist() == [True, False, True]
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(1)
    _, drop_reward, done, truncated, info = env.step(2)
    assert done and not truncated and not info["action_mask"].any()
    assert defer_reward + drop_reward < 0
    assert info["metrics"]["slo_attainment"] == 0
    assert info["metrics"]["drop_rate"] == 1


def test_training_and_greedy_scheduler_handle_capacity_wait_the_same_way():
    requests = [Request("resident", 0, 32, 1, 1), Request("big", 0, 48, 1, 2),
                Request("small", 0, 16, 1, 3)]
    hardware = HardwareConfig(max_batch_size=2, kv_block_size=16, kv_total_blocks=4)
    decisions = DecisionConfig(0.25, 1)
    env = make_env(requests, hardware, decisions)
    env.reset()
    done = False
    while not done:
        _, _, done, _, info = env.step(0)
    assert info["metrics"]["completed"] == 3

    class AlwaysAdmit:
        def act(self, observation, mask):
            assert mask[0]
            return 0

    scheduler = GreedyScheduler(AlwaysAdmit(), env.replay.horizon_s, decisions)
    result = simulate(requests, scheduler, hardware, estimated_output_tokens=8,
                      defer_retry_s=decisions.defer_retry_s, max_defers=decisions.max_defers)

    def decisions_in(events):
        return [(e.request_id, e.timestamp_s, e.details["action"]) for e in events
                if e.event_type is EventType.DECISION]

    assert decisions_in(env.kernel.events()) == decisions_in(result.events)
    assert all(request.defer_count == 0 for request in result.requests)


def test_train_updates_weights_and_checkpoint_replays_same_validation(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    trace = processed / "burstgpt_windows.csv"
    with trace.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "window", "source_row", "timestamp_s", "input_tokens", "true_output_tokens"])
        for index in range(300):
            writer.writerow(["train", 0, index, index, 16, 2])
        for index in range(300, 304):
            writer.writerow(["validation", 0, index, index, 16, 3])
        # A one-row holdout would be invalid if fed to replay_window.
        writer.writerow(["test", 0, 304, 304, 16, 200])
    (processed / "manifest.json").write_text(json.dumps({
        "selected_sha256": sha256(trace), "source_sha256": "fixture",
        "estimated_output_tokens_train_mean": 2,
    }), encoding="utf-8")
    config = {
        "seeds": [7], "episodes": 1, "validate_every": 1, "target_rps": [1.2], "slo_s": 5,
        "hardware": {"max_batch_size": 4}, "decisions": {"defer_retry_s": 0.25, "max_defers": 5},
        "reward": {},
    }
    output = tmp_path / "trained"
    rows = run(config, processed, output)
    with (output / "checkpoints.csv").open(encoding="utf-8") as stream:
        scores = list(csv.DictReader(stream))
    assert int(scores[-1]["updates"]) > 0
    assert float(scores[-1]["max_weight_change"]) > 0
    reloaded = run(config, processed, tmp_path / "reloaded", output / "dqn_seed7.npz")
    assert reloaded == rows
    assert all(row["arrivals"] == 4 for row in rows)
