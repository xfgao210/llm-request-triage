"""Scenario extraction, native replay, and frozen-policy evaluation."""

import csv
import json

import pytest

from scripts.analyze_burstgpt_traffic import analyze
from scripts.prepare_burstgpt3 import prepare
from llm_triage.workload.burstgpt import TraceRow, load_windows, sha256
from llm_triage.workload.transforms import replay_window


@pytest.fixture
def scenarios(tmp_path):
    source = tmp_path / "BurstGPT_3.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Timestamp", "Session ID", "Elapsed time", "Model",
                         "Request tokens", "Response tokens", "Total tokens", "Log Type"])
        for window in range(10):
            for request in range(300):
                writer.writerow([window * 300 + 10 + request / 2, "", 1, "ChatGPT", 16, 2, 18, "API log"])
    analysis = tmp_path / "analysis"
    analyze(source, analysis)
    data = tmp_path / "data"
    manifest = prepare(source, analysis, data, counts=dict(train=2, validation=1, test=1),
                       regimes={split: ["regular"] for split in ("train", "validation", "test")})
    return source, analysis, data, manifest


def test_extraction_preserves_source_identity_boundaries_and_splits(scenarios):
    source, analysis, data, manifest = scenarios
    windows = load_windows(data / "processed/burstgpt_windows.csv")
    assert len(windows) == 4
    assert sum(map(len, windows.values())) == 1200
    assert manifest["source_sha256"] == sha256(source)
    assert manifest["estimated_output_tokens_train_mean"] == 2
    for item in manifest["windows"]:
        rows = windows[item["split"], item["window"]]
        assert all(item["start_timestamp_s"] <= row.timestamp_s < item["end_timestamp_s"] for row in rows)
        assert len(rows) == item["requests"]
    with pytest.raises(FileExistsError):
        prepare(source, analysis, data)


def test_source_and_catalog_mismatch_fail_before_publishing(scenarios, tmp_path):
    source, analysis, _, _ = scenarios
    source.write_text(source.read_text() + "9999,,1,ChatGPT,1,1,2,API log\n")
    with pytest.raises(ValueError, match="SHA256"):
        prepare(source, analysis, tmp_path / "wrong-source")
    assert not (tmp_path / "wrong-source/processed").exists()


def test_missing_regime_and_catalog_counts_are_not_silently_substituted(scenarios, tmp_path):
    source, analysis, _, _ = scenarios
    counts = dict(train=1, validation=1, test=1)
    with pytest.raises(ValueError, match="eligible train"):
        prepare(source, analysis, tmp_path / "no-peaks", counts=counts,
                regimes={split: ["peak"] for split in counts})
    catalog_path = analysis / "scenario_catalog.csv"
    with catalog_path.open(newline="") as stream:
        catalog = list(csv.DictReader(stream))
    for item in catalog:
        item["request_count"] = "301"
    with catalog_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)
    output = tmp_path / "bad-counts"
    with pytest.raises(ValueError, match="count mismatch"):
        prepare(source, analysis, output, counts=counts,
                regimes={split: ["regular"] for split in counts})
    assert not (output / "processed").exists()


def test_fixed_bounds_preserve_idle_time_and_allow_single_arrival():
    rows = [TraceRow(1, 125, 16, 2)]
    native = replay_window(rows, bounds=(100, 400))
    assert native.horizon_s == 300
    assert native.requests[0].arrival_time_s == 25
    scaled = replay_window(rows, bounds=(100, 400), target_rps=2)
    assert scaled.horizon_s == 0.5
    assert scaled.requests[0].arrival_time_s == pytest.approx(25 / 600)
    for bounds in ((125, 125), (126, 400), (100, 125)):
        with pytest.raises(ValueError, match="bounds"):
            replay_window(rows, bounds=bounds)


def test_pipeline_and_frozen_evaluation_use_native_burstgpt3_scenarios(scenarios, tmp_path, monkeypatch):
    pytest.importorskip("gymnasium")
    pytest.importorskip("numpy")
    from experiments.run_pipeline import run
    from experiments.evaluate_scenarios import run as evaluate
    from experiments.train_dqn import load_cohorts
    _, _, data, manifest = scenarios
    config = dict(dataset="burstgpt3", seeds=[7], episodes=2, validate_every=1,
                  target_rps=[None], slo_s=5, hardware={"max_batch_size": 4},
                  decisions={"defer_retry_s": 0.25, "max_defers": 5}, reward={})
    cohorts, _, _ = load_cohorts(data / "processed", config)
    assert all(case[2].horizon_s == 300 for cases in cohorts.values() for case in cases)
    output = tmp_path / "pipeline"
    run(config, data, output)
    checkpoint = output / "training/dqn_seed7.npz"
    before = sha256(checkpoint)
    from llm_triage.rl.dqn import DQN

    def unexpected_training(*args, **kwargs):
        pytest.fail("Evaluation must not update the model")

    monkeypatch.setattr(DQN, "train_batch", unexpected_training)
    rows = evaluate(output / "training", data, tmp_path / "holdout", split="test", target_rps=[1.2, 4.8])
    assert len(rows) == 8
    assert {row["split"] for row in rows} == {"test"}
    assert {row["target_rps"] for row in rows} == {1.2, 4.8}
    assert sha256(checkpoint) == before
    assert not json.loads((output / "training/run.json").read_text())["test_evaluated"]
    assert (tmp_path / "holdout/summary.csv").exists()
    assert {row["traffic_regime"] for row in rows} == {"regular"}
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum changed"):
        evaluate(output / "training", data, tmp_path / "tampered-checkpoint", split="test")
    with pytest.raises(ValueError, match="dataset=burstgpt3"):
        run(dict(config, dataset="burstgpt1"), data, tmp_path / "wrong-dataset")
