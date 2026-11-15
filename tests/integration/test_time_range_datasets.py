"""Explicit range selection retains the traffic mix and isolates held-out groups."""

from copy import deepcopy
import csv
import json

import pytest

from scripts.analyze_burstgpt_traffic import analyze
from scripts.report_burstgpt_traffic import report, summarize_periods
from llm_triage.workload.burstgpt import load_windows
from llm_triage.workload.time_ranges import prepare_ranges, validate_ranges


@pytest.fixture
def range_data(tmp_path):
    source = tmp_path / "BurstGPT_3.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Timestamp", "Session ID", "Elapsed time", "Model",
                         "Request tokens", "Response tokens", "Total tokens", "Log Type"])
        for window, output in ((0, 2), (1, 2), (4, 4), (6, 20), (9, 50)):
            for index in range(300):
                writer.writerow([window * 300 + index / 2, "", 1, "ChatGPT", 16, output, 16 + output, "API log"])
    analysis = tmp_path / "analysis"
    summary = analyze(source, analysis)
    config = dict(schema_version=1, source_sha256=summary["source"]["sha256"],
                  episode_seconds=300, min_split_gap_seconds=300, ranges=[
                      dict(split="train", group="training", start_timestamp_s=0, end_timestamp_s=900),
                      dict(split="validation", group="validation", start_timestamp_s=1200, end_timestamp_s=1500),
                      dict(split="test", group="ordinary", start_timestamp_s=1800, end_timestamp_s=2400),
                      dict(split="test", group="peak_heavy", start_timestamp_s=2700, end_timestamp_s=3000),
                  ])
    path = tmp_path / "ranges.json"
    path.write_text(json.dumps(config))
    return source, analysis, config, path


def test_ranges_extract_every_request_preserve_boundaries_and_record_idle(range_data, tmp_path):
    source, analysis, config, path = range_data
    manifest = prepare_ranges(source, analysis, tmp_path / "data", path)
    windows = load_windows(tmp_path / "data/processed/burstgpt_windows.csv")
    assert sum(map(len, windows.values())) == 1500
    assert sum(len(rows) for (split, _), rows in windows.items() if split == "train") == 600
    assert manifest["estimated_output_tokens_train_mean"] == 2
    assert [item["window"] for item in manifest["empty_windows"]] == [2, 5]
    assert windows["train", 1][0].timestamp_s == 300
    assert all(row.timestamp_s < 300 for row in windows["train", 0])
    assert manifest["range_config"] == config
    assert all("traffic_regime" not in item for item in manifest["windows"])
    # Explicit validation range 1200..1500 overrides the old automatic partition.
    assert windows["validation", 3][0].timestamp_s == 1200
    with pytest.raises(FileExistsError):
        prepare_ranges(source, analysis, tmp_path / "data", path)


@pytest.mark.parametrize("change, message", [
    (lambda config: config["ranges"][1].update(start_timestamp_s=800), "overlap"),
    (lambda config: config["ranges"][0].update(split="test"), "train before"),
    (lambda config: config.update(min_split_gap_seconds=301), "gap"),
    (lambda config: config.update(episode_seconds=0), "episode_seconds"),
    (lambda config: config["ranges"][0].update(end_timestamp_s=float("nan")), "timestamps"),
    (lambda config: config["ranges"][2].update(group="training"), "cross splits"),
])
def test_invalid_range_protocols_reject_leakage(range_data, change, message):
    config = deepcopy(range_data[2])
    change(config)
    # Keep all three split labels for the chronological-order check.
    if message == "train before":
        config["ranges"][2]["split"] = "train"
    with pytest.raises(ValueError, match=message):
        validate_ranges(config)


def test_no_requests_and_wrong_source_do_not_publish(range_data, tmp_path):
    source, analysis, config, path = range_data
    config["ranges"][0].update(start_timestamp_s=600, end_timestamp_s=900)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="no valid requests"):
        prepare_ranges(source, analysis, tmp_path / "empty", path)
    assert not (tmp_path / "empty/processed").exists()
    config["source_sha256"] = "0" * 64
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="SHA256"):
        prepare_ranges(source, analysis, tmp_path / "wrong", path)


def test_report_ranks_continuous_periods_and_records_collection_gaps(range_data, tmp_path):
    _, analysis, _, _ = range_data
    output = tmp_path / "report"
    summary = report(analysis, output, period_seconds=600, plots=False)
    assert summary["periods"] == 5
    assert summary["idle_gaps"] == 3
    with (output / "periods.csv").open(newline="") as stream:
        periods = list(csv.DictReader(stream))
    assert float(periods[0]["mean_rps"]) == 1
    assert float(periods[1]["empty_bin_fraction"]) == 1
    assert float(periods[4]["mean_output_tokens"]) == 50
    assert (output / "candidate_periods.csv").exists()
    with pytest.raises(ValueError, match="integer multiple"):
        summarize_periods([], 350, 300)


def test_frozen_groups_and_idle_duration_are_reported_separately(range_data, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    from experiments.run_pipeline import run
    from experiments.evaluate_scenarios import run as evaluate
    from llm_triage.rl.dqn import DQN
    source, analysis, _, path = range_data
    data = tmp_path / "data"
    prepare_ranges(source, analysis, data, path)
    config = dict(dataset="burstgpt3", seeds=[7], episodes=2, validate_every=1,
                  target_rps=[None], slo_s=5, hardware={"max_batch_size": 4},
                  decisions={"defer_retry_s": 0.25, "max_defers": 5}, reward={})
    training = tmp_path / "training"
    run(config, data, training)

    def unexpected_update(*args):
        pytest.fail("Frozen evaluation attempted training")

    monkeypatch.setattr(DQN, "train_batch", unexpected_update)
    rows = evaluate(training / "training", data, tmp_path / "test", split="test")
    assert {row["group"] for row in rows} == {"ordinary", "peak_heavy"}
    with (tmp_path / "test/summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    assert all(float(row["evaluated_duration_s"]) == 600 for row in summaries if row["group"] == "ordinary")
    assert all(float(row["evaluated_duration_s"]) == 300 for row in summaries if row["group"] == "peak_heavy")
    ordinary = evaluate(training / "training", data, tmp_path / "ordinary",
                        split="test", groups=["ordinary"], target_rps=[2.4])
    assert len(ordinary) == 4
    assert {row["group"] for row in ordinary} == {"ordinary"}
    assert all(row["measurement_window_s"] == 125 for row in ordinary)
    with pytest.raises(ValueError, match="Unknown evaluation group"):
        evaluate(training / "training", data, tmp_path / "bad-group", split="test", groups=["missing"])


def test_online_copy_learns_only_after_step_and_keeps_source_frozen(range_data, tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    from experiments.run_pipeline import run as train
    from experiments.online_adaptation import run
    from llm_triage.rl import TriageEnv
    from llm_triage.rl.dqn import DQN
    from llm_triage.workload.burstgpt import sha256

    source, analysis, _, path = range_data
    data, training, output = tmp_path / "data", tmp_path / "training", tmp_path / "online"
    prepare_ranges(source, analysis, data, path)
    config = dict(dataset="burstgpt3", seeds=[7], episodes=2, validate_every=1,
                  target_rps=[None], slo_s=5, hardware={"max_batch_size": 4},
                  decisions={"defer_retry_s": 0.25, "max_defers": 5}, reward={})
    train(config, data, training)
    checkpoint = training / "training/dqn_seed7.npz"
    digest = sha256(checkpoint)
    delivered, update_times = [], []
    original_step, original_train = TriageEnv.step, DQN.train_batch

    def step(self, action):
        value = original_step(self, action)
        delivered.append(value[-1]["now_s"])
        return value

    def update(self, batch):
        assert delivered, "Training before any outcomes were delivered"
        update_times.append(delivered[-1])
        return original_train(self, batch)

    monkeypatch.setattr(TriageEnv, "step", step)
    monkeypatch.setattr(DQN, "train_batch", update)
    rows = run(training / "training", data, output, warmup=16, batch_size=8)
    assert {r["group"] for r in rows} == {"ordinary", "peak_heavy"}
    assert {r["policy"] for r in rows} == {"frozen", "adapting"}
    assert all(r["arrivals"] == 300 for r in rows)
    assert update_times == sorted(update_times) and update_times
    saved = json.loads((output / "run.json").read_text())
    assert saved["status"] == "completed"
    assert saved["policy_runs"][0]["updates"] == 0
    assert saved["policy_runs"][1]["updates"] > 0
    assert saved["policy_runs"][1]["max_weight_change"] > 0
    assert sha256(checkpoint) == digest
    original, _ = DQN.load(checkpoint)
    frozen, _ = DQN.load(output / "frozen_final.npz")
    assert all(np.array_equal(v, frozen.params[k]) for k, v in original.params.items())
    with (output / "adapting_updates.csv").open() as stream:
        audit = list(csv.DictReader(stream))
    assert all(float(r["source_timestamp_s"]) >= 1800 for r in audit)
    assert [float(r["source_timestamp_s"]) for r in audit] == sorted(float(r["source_timestamp_s"]) for r in audit)
    with (output / "timeline.csv").open() as stream:
        timeline = list(csv.DictReader(stream))
    assert len(timeline) == 6  # Includes the ordinary group's empty bin for both copies.
    with pytest.raises(FileExistsError):
        run(training / "training", data, output)
