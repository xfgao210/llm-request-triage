"""Pipeline retries, checkpoint integrity, and configuration compatibility."""

import csv
import json

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")

import experiments.run_pipeline as pipeline
from llm_triage.workload.burstgpt import sha256


@pytest.fixture
def fixture_run(tmp_path):
    data = tmp_path / "data"
    processed = data / "processed"
    processed.mkdir(parents=True)
    trace = processed / "burstgpt_windows.csv"
    with trace.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "window", "source_row", "timestamp_s", "input_tokens", "true_output_tokens"])
        for index in range(300):
            writer.writerow(["train", 0, index, index, 16, 2])
        for index in range(300, 304):
            writer.writerow(["validation", 0, index, index, 16, 3])
        writer.writerow(["test", 0, 304, 304, 16, 200])
    (processed / "manifest.json").write_text(json.dumps({
        "selected_sha256": sha256(trace), "source_sha256": "fixture",
        "estimated_output_tokens_train_mean": 2,
    }), encoding="utf-8")
    config = dict(
        seeds=[7], episodes=1, validate_every=1, target_rps=[1.2], slo_s=5,
        hardware={"max_batch_size": 4}, decisions={"defer_retry_s": 0.25, "max_defers": 5},
        reward={}, agent={"hidden": 16, "learning_rate": 0.0005, "gamma": 0.95},
    )
    return config, data, tmp_path / "output"


def test_retry_after_evaluation_failure_reuses_completed_training(fixture_run, monkeypatch):
    config, data, output = fixture_run
    monkeypatch.chdir(output.parent)
    output = output.relative_to(output.parent)
    original = pipeline.evaluate
    with monkeypatch.context() as patch:
        def failed(*args, **kwargs):
            raise RuntimeError("interrupted evaluation")
        patch.setattr(pipeline, "evaluate", failed)
        with pytest.raises(RuntimeError, match="interrupted evaluation"):
            pipeline.run(config, data, output)
    state = json.loads((output / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed" and state["training_complete"]
    assert (output / state["training_log"]).is_file()
    checksum = sha256(output / "training/dqn_seed7.npz")
    with monkeypatch.context() as patch:
        def unexpected(*args, **kwargs):
            pytest.fail("Resume retrained a completed checkpoint")
        patch.setattr(pipeline, "train", unexpected)
        rows = pipeline.run(config, data, output, resume=True)
    assert len(rows) == 4
    assert sha256(output / "training/dqn_seed7.npz") == checksum
    assert (output / "summary.csv").is_file()
    state = json.loads((output / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert pipeline.evaluate is original
    changed = dict(config, episodes=2)
    with pytest.raises(ValueError, match="same configuration"):
        pipeline.run(changed, data, output, resume=True)
    path = output / "training/dqn_seed7.npz"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum changed"):
        pipeline.run(config, data, output, resume=True)


def test_resume_recovers_after_training_promotion(fixture_run, monkeypatch):
    config, data, output = fixture_run
    pipeline.run(config, data, output)
    path = output / "pipeline.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state.update(training_complete=False, stage="train")
    path.write_text(json.dumps(state), encoding="utf-8")
    with monkeypatch.context() as patch:
        def unexpected(*args, **kwargs):
            pytest.fail("Recovered training was repeated")
        patch.setattr(pipeline, "train", unexpected)
        pipeline.run(config, data, output, resume=True)
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "completed"


def test_output_lock_excludes_a_second_run(tmp_path):
    with pipeline.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another pipeline process"):
            with pipeline.run_lock(tmp_path):
                pytest.fail("Second process acquired the run lock")
