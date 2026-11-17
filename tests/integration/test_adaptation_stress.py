"""Small, real training/replay checks for the independent-window protocol."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import adaptation_stress as stress
from experiments.train_dqn import load_cohorts, policy_options
from experiments.run_pipeline import run as train
from llm_triage.rl import ObservationBuilder, ObservationConfig, TriageEnv
from llm_triage.rl.dqn import DQN
from llm_triage.simulation import Request
from llm_triage.workload.burstgpt import sha256
from llm_triage.workload.time_ranges import prepare_ranges
from llm_triage.workload.transforms import ReplayWindow
from scripts.analyze_burstgpt_traffic import analyze


@pytest.fixture
def stress_data(tmp_path):
    source = tmp_path / "BurstGPT_3.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Timestamp", "Session ID", "Elapsed time", "Model",
                         "Request tokens", "Response tokens", "Total tokens", "Log Type"])
        for start, count, output in ((0, 300, 4), (300, 300, 4), (900, 30, 6),
                                     (1500, 80, 4), (2100, 120, 6)):
            for i in range(count):
                writer.writerow([start+i/10, "", 1, "ChatGPT", 16, output, 16+output, "API log"])
    analysis = tmp_path / "analysis"
    analyze(source, analysis)
    ranges = dict(schema_version=1, source_sha256=sha256(source), episode_seconds=300,
                  min_split_gap_seconds=300, ranges=[
                      dict(split="train", group="training", start_timestamp_s=0, end_timestamp_s=600),
                      dict(split="validation", group="validation", start_timestamp_s=900, end_timestamp_s=1200),
                      dict(split="test", group="mixed_traffic", start_timestamp_s=1500, end_timestamp_s=1800),
                      dict(split="test", group="peak_heavy", start_timestamp_s=2100, end_timestamp_s=2400),
                  ])
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(ranges))
    data = tmp_path / "data"
    manifest = prepare_ranges(source, analysis, data, path)
    config = dict(dataset="burstgpt3", seeds=[7], episodes=3, validate_every=3,
                  target_rps=[None], arrival_multipliers=[1, 2, 4], slo_s=5,
                  hardware={"max_batch_size": 4}, observation={"include_episode_progress": False},
                  cohort_window_limit={"train": 2, "validation": 1},
                  decisions={"defer_retry_s": .25, "max_defers": 5}, reward={})
    return data, manifest, config, source, path


def test_disabled_clock_keeps_shape_and_is_horizon_invariant():
    replay = ReplayWindow((Request("a", 2, 16, 4, 7),), 30, 0, 1)
    decisions = dict(max_defers=5, defer_retry_s=.25)
    env = TriageEnv(replay, **decisions, **policy_options(replay, decisions, {"include_episode_progress": False}))
    observation, _ = env.reset()
    assert observation.shape == (15,) and observation[-1] == 0
    first = ObservationBuilder(ObservationConfig(horizon_s=30, include_episode_progress=False)).build(env.snapshot())
    second = ObservationBuilder(ObservationConfig(horizon_s=3000, include_episode_progress=False)).build(env.snapshot())
    np.testing.assert_array_equal(first, second)
    env.reset(options={"horizon_s": 3000})
    assert env.observation_builder.config.include_episode_progress is False
    with pytest.raises(ValueError, match="boolean"):
        ObservationConfig(include_episode_progress=0)


@pytest.mark.parametrize("order,expected", [("fcfs", "a-long"), ("length_aware", "b-short")])
def test_candidate_order_matches_env_and_validation_callback(order, expected):
    from llm_triage.rl import RLPolicyCallback

    replay = ReplayWindow((Request("a-long", 0, 1000, 4, 5),
                           Request("b-short", 0, 16, 4, 5)), 30, 0, 1)
    decisions = dict(max_defers=5, defer_retry_s=.25)
    options = policy_options(replay, decisions, candidate_order=order)
    env = TriageEnv(replay, **decisions, **options)
    _, info = env.reset()
    assert info["candidate_id"] == expected
    views = []
    for request in replay.requests:
        single = TriageEnv([request])
        single.reset()
        views.append(single.snapshot().candidate)
    callback = RLPolicyCallback(DQN(), **options)
    assert min(views, key=callback.priority).request_id == expected
    with pytest.raises(ValueError, match="rl_candidate_order"):
        policy_options(replay, decisions, candidate_order="typo")


def test_multiplier_cohorts_and_balanced_training(stress_data, tmp_path):
    data, _, config, _, _ = stress_data
    cohorts, estimate, _ = load_cohorts(data / "processed", config)
    assert estimate == 4
    assert len(cohorts["train"]) == 6 and len(cohorts["validation"]) == 3
    a, b = cohorts["train"][0][2], cohorts["train"][2][2]
    assert b.horizon_s == a.horizon_s / 4
    assert b.requests[1].arrival_time == a.requests[1].arrival_time / 4
    assert all(r.slo_deadline-r.arrival_time == pytest.approx(5) for r in b.requests)
    output = tmp_path / "training"
    train(config, data, output)
    with (output / "training/training.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert [float(r["arrival_multiplier"]) for r in history] == [1, 2, 4]
    assert "target_rps" not in history[0]
    with pytest.raises(ValueError, match="Do not combine"):
        load_cohorts(data / "processed", dict(config, target_rps=[8]))


def test_all_five_paired_causal_policies_and_safe_cache(stress_data, tmp_path, monkeypatch):
    data, manifest, config, _, _ = stress_data
    training = tmp_path / "training"
    train(config, data, training)
    training = training / "training"
    original_hash = sha256(training / "dqn_seed7.npz")
    windows = stress.select_evaluation_windows(manifest, seconds=300)
    checkpoint_record = json.loads((training / "run.json").read_text())
    delivered = []
    original_step, original_update = TriageEnv.step, DQN.train_batch

    def step(self, action):
        result = original_step(self, action)
        delivered.append(result[-1]["now_s"])
        assert self.observation_builder.config.include_episode_progress is False
        return result

    def update(self, batch):
        assert delivered, "Learner updated before a transition was delivered"
        return original_update(self, batch)

    monkeypatch.setattr(TriageEnv, "step", step)
    monkeypatch.setattr(DQN, "train_batch", update)
    output = tmp_path / "stress"
    kwargs = dict(multipliers=[1, 4, 8], warmup=8, batch_size=4, device="cpu")
    rows = stress.run(training, data, output, windows, **kwargs)
    assert len(rows) == 30
    for name in {r["window"] for r in rows}:
        for level in (1, 4, 8):
            paired = [r for r in rows if r["window"] == name and r["multiplier"] == level]
            assert {r["policy"] for r in paired} == set(stress.POLICIES)
            assert len({r["arrivals"] for r in paired}) == 1
            assert all(0 <= r["ttft_target_attainment"] <= r["first_token_coverage"] <= 1 for r in paired)
    assert sha256(training / "dqn_seed7.npz") == original_hash
    with (output / "learning_summary.csv").open() as stream:
        learning = list(csv.DictReader(stream))
    assert all(float(r["updates"]) == 0 for r in learning if r["policy"] == "frozen")
    assert any(float(r["updates"]) > 0 for r in learning if r["policy"] == "adapting")
    for path in output.rglob("frozen_final.npz"):
        original, _ = DQN.load(training / "dqn_seed7.npz")
        frozen, metadata = DQN.load(path)
        assert metadata["config"] == checkpoint_record["config"]
        for key in original.params:
            np.testing.assert_array_equal(original.params[key], frozen.params[key])

    def forbidden(*args, **kwargs):
        pytest.fail("A cached policy was rerun")

    monkeypatch.setattr(stress, "replay_dqn", forbidden)
    monkeypatch.setattr(stress, "simulate", forbidden)
    assert stress.run(training, data, output, windows, **kwargs) == rows
    with pytest.raises(ValueError, match="new RUN_NAME"):
        stress.run(training, data, output, windows, **dict(kwargs, ttft_target_s=2))
    cached = output / "mixed_traffic/load_1/fcfs/summary.json"
    cached.write_text("{}")
    with pytest.raises(ValueError, match="Cached outputs changed"):
        stress.run(training, data, output, windows, **kwargs)


def test_interrupt_preserves_completed_policies(stress_data, tmp_path, monkeypatch):
    data, manifest, config, _, _ = stress_data
    training = tmp_path / "training"
    train(config, data, training)
    windows = stress.select_evaluation_windows(manifest, seconds=300)[:1]
    output = tmp_path / "interrupted"
    original = stress.replay_dqn

    def interrupt(*args, **kwargs):
        if args[4] == "adapting":
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(stress, "replay_dqn", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        stress.run(training / "training", data, output, windows, multipliers=[1])
    cached = output / "mixed_traffic/load_1/frozen_seed7/frozen_final.npz"
    digest = sha256(cached)
    calls = []

    def retry(*args, **kwargs):
        calls.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(stress, "replay_dqn", retry)
    stress.run(training / "training", data, output, windows, multipliers=[1])
    assert calls == ["adapting"]
    assert sha256(cached) == digest


def test_ttft_denominator_includes_no_token_requests():
    from llm_triage.metrics import RequestRecord, RequestStatus
    records = [RequestRecord("a", 0, 5, RequestStatus.COMPLETED,
                             service_start_time_s=0, first_token_time_s=.5, completion_time_s=1),
               RequestRecord("b", 0, 5, RequestStatus.DROPPED)]
    score = stress._ttft_score(records, 1)
    assert score == {"first_token_coverage": .5, "ttft_target_attainment": .5}


def test_multiple_seeds_share_baselines_and_reject_old_clock(stress_data, tmp_path):
    data, manifest, config, _, _ = stress_data
    config["seeds"] = [7, 17]
    training = tmp_path / "two-seeds"
    train(config, data, training)
    windows = stress.select_evaluation_windows(manifest, seconds=300)[:1]
    rows = stress.run(training / "training", data, tmp_path / "paired-seeds", windows,
                      seeds=[7, 17], multipliers=[1], device="cpu")
    assert len(rows) == 7  # Three baselines, not six, plus four seeded copies.
    assert {r["seed"] for r in rows if r["policy"] == "adapting"} == {7, 17}
    record_path = training / "training/run.json"
    record = json.loads(record_path.read_text())
    record["config"]["observation"]["include_episode_progress"] = True
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Retrain"):
        stress.run(training / "training", data, tmp_path / "reject-clock", windows)


@pytest.mark.parametrize("mode,short_training", [("full", False), ("quick", False), ("quick", True)])
def test_full_adaptation_notebook_on_small_trace(stress_data, tmp_path, monkeypatch, mode, short_training):
    pytest.importorskip("pandas")
    pytest.importorskip("IPython")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import zipfile

    data, _, _, source, ranges = stress_data
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    monkeypatch.setenv("PYTHONPATH", str(root / "src"))
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    notebook = json.loads((root / "notebooks/colab_burstgpt3_online_adaptation.ipynb").read_text())
    namespace = dict(__name__="__main__", fixture_ranges=ranges)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        if cell["id"] == "install":
            # Dependencies already present in this test environment. Still run
            # all imports and every analysis/training/replay/export cell.
            text = text[text.index("import base64"):]
        if cell["id"] == "parameters":
            text = text.replace('RESULTS_ROOT = REPO / "results/runs/colab_burstgpt3_stress"',
                                f"RESULTS_ROOT = Path({str(tmp_path / 'notebook-results')!r})")
            text = text.replace('RUN_MODE = "quick"', f'RUN_MODE = "{mode}"')
            text = text.replace('QUICK_TRAINING = False', f'QUICK_TRAINING = {short_training}')
            text = text.replace('EPISODES = 24', 'EPISODES = 6')
            text = text.replace('VALIDATE_EVERY = 4', 'VALIDATE_EVERY = 3')
        if cell["id"] == "data":
            text = text.replace('(REPO / "configs/burstgpt3_ranges.json")', "fixture_ranges")
        exec(compile(text, cell["id"], "exec"), namespace)
        if cell["id"] == "parameters":
            namespace.update(LOCAL_CSV_PATH=str(source),
                             EVAL_WINDOW_SECONDS=300, EVAL_ARRIVAL_MULTIPLIERS=([1, 4] if mode == "full" else [1, 4, 8]),
                             ANALYSIS_DIR=data.parent / "analysis", DEVICE="cpu",
                             BATCH_SIZE=4, WARMUP=8)
    output = namespace["ONLINE_OUTPUT"]
    with (output / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == (20 if mode == "full" else 5)
    assert {row["policy"] for row in rows} == set(stress.POLICIES)
    assert namespace["train_config"]["episodes"] == (3 if short_training else 6)
    plan = json.loads((namespace["RUN_ROOT"] / "execution_plan.json").read_text())
    assert plan["mode"] == mode and plan["short_training"] == short_training
    assert not plan["request_sampling"] and not plan["window_shortening"]
    if mode == "quick":
        original = next(w for w in plan["original_windows"] if w["group"] == "peak_heavy")
        assert plan["selected_windows"] == [original]
        assert {r["multiplier"] for r in rows} == {"8"}
        assert {r["arrivals"] for r in rows} == {"120"}
    assert namespace["REPORT"].stat().st_size > 10000
    diagnostic_figures = len(namespace["evaluation_windows"]) * len(namespace["ACTIVE_EVAL_MULTIPLIERS"]) * 2 * len(namespace["ACTIVE_SEEDS"])
    assert len(namespace["generated_figures"]) == (9 if mode == "full" else 5) + diagnostic_figures
    with zipfile.ZipFile(namespace["ARCHIVE"]) as archive:
        names = archive.namelist()
        assert "online_report.html" in names
        assert "execution_plan.json" in names
        assert "stress_comparison/length_groups.csv" in names
        assert "training_run/training/training_updates_seed7.csv" in names
        assert any(name.endswith("adapting_transitions.csv") for name in names)
        assert not any("burstgpt_windows.csv" in name or "BurstGPT_3.csv" in name for name in names)


@pytest.mark.parametrize("evaluation_split", ["test", "validation"])
@pytest.mark.parametrize("candidate_order", ["fcfs", "length_aware"])
def test_time_discount_and_budget_audit_end_to_end(stress_data, tmp_path, evaluation_split, candidate_order, monkeypatch):
    data, manifest, config, _, _ = stress_data
    config["rl_candidate_order"] = candidate_order
    config["reward"] = {"drop": .5}
    original_step = TriageEnv.step

    def checked_step(self, action):
        assert self.candidate_selector == candidate_order
        assert self.reward_config.drop == .5
        return original_step(self, action)

    monkeypatch.setattr(TriageEnv, "step", checked_step)
    config["agent"] = dict(discount_mode="elapsed_time", discount_time_unit_s=1.0, gamma=.99)
    training = tmp_path / "time-training"
    train(config, data, training)
    with (training / "training/training_updates_seed7.csv").open() as stream:
        training_updates = list(csv.DictReader(stream))
    assert training_updates
    assert [int(row["update"]) for row in training_updates] == list(range(1, len(training_updates) + 1))
    assert all(np.isfinite(float(row["loss"])) for row in training_updates)
    checkpoint_record = json.loads((training / "training/run.json").read_text())
    assert checkpoint_record["config"]["rl_candidate_order"] == candidate_order
    assert checkpoint_record["reward_config"]["drop"] == .5
    source = training / "training/dqn_seed7.npz"
    source_digest = sha256(source)
    windows = stress.select_evaluation_windows(manifest, seconds=300, split=evaluation_split)[:1]
    output = tmp_path / "budgeted"
    stress.run(training / "training", data, output, windows, multipliers=[4], batch_size=4, warmup=8,
               update_interval=4, min_update_seconds=.1, min_new_outcomes=2, max_updates=2, device="cpu",
               evaluation_split=evaluation_split)
    group = "mixed_traffic" if evaluation_split == "test" else "validation"
    folder = output / group / "load_4/adapting_seed7"
    with (folder / "adapting_updates.csv").open() as stream:
        updates = list(csv.DictReader(stream))
    assert 0 < len(updates) <= 2
    assert all(float(row["seconds_since_update"]) >= .1 for row in updates)
    assert all(int(row["outcomes_since_update"]) >= 2 for row in updates)
    with (folder / "adapting_transitions.csv").open() as stream:
        transitions = list(csv.DictReader(stream))
    for row in transitions:
        dt = float(row["elapsed_s"])
        expected = 0.0 if row["terminal"] == "True" else .99**dt
        assert float(row["bootstrap_discount"]) == pytest.approx(expected)
        components = [float(value) for key,value in row.items() if key.startswith("reward_") and key != "reward_total"]
        assert sum(components) == pytest.approx(float(row["reward_total"]))
    stats = json.loads((folder / "learning.json").read_text())
    assert stats["cumulative_reward"] == pytest.approx(sum(float(row["reward_total"]) for row in transitions))
    assert stats["admit_actions"] + stats["defer_actions"] + stats["drop_actions"] == len(transitions)
    assert stats["reward_drop"] == pytest.approx(-.5 * stats["drop_actions"])
    assert sha256(source) == source_digest
    with (output / group / "load_4/frozen_seed7/frozen_transitions.csv").open() as stream:
        frozen = list(csv.DictReader(stream))
    assert all(int(row["updates"]) == 0 and row["updated"] == "False" for row in frozen)


def test_quick_mode_filters_conditions_without_changing_windows(stress_data):
    _, manifest, _, _, _ = stress_data
    windows = stress.select_evaluation_windows(manifest, seconds=300)
    before = json.dumps(windows, sort_keys=True)
    peak = next(w for w in windows if w["group"] == "peak_heavy")
    chosen, loads, seeds = stress.select_run_conditions(windows, [1, 4, 8], [7, 17], mode="quick")
    assert chosen == [peak] and loads == [8] and seeds == [7]
    assert json.dumps(windows, sort_keys=True) == before
    assert stress.select_run_conditions(windows, [1, 4, 8], [7, 17], mode="full") == (windows, [1, 4, 8], [7, 17])
    assert stress.select_run_conditions(windows, [1, 8], [7, 17], mode="quick", quick_seed=17)[2] == [17]
    with pytest.raises(ValueError, match="existing 8x"):
        stress.select_run_conditions(windows, [1, 4], [7], mode="quick")
    with pytest.raises(ValueError, match="QUICK_SEED"):
        stress.select_run_conditions(windows, [8], [7], mode="quick", quick_seed=17)
    with pytest.raises(ValueError, match="RUN_MODE"):
        stress.select_run_conditions(windows, [8], [7], mode="sample")
