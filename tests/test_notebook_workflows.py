"""Retain ingestion while explicitly versioning mixed-load training and replay."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cells(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def test_online_notebook_trains_before_adapting_without_archive_input():
    offline = cells("colab_burstgpt3_experiments.ipynb")
    online = cells("colab_burstgpt3_online_adaptation.ipynb")
    assert list(online).index("train") < list(online).index("run")
    for name in ("source", "data", "learning-charts", "validation-charts"):
        assert online[name] == offline[name]
    assert "TRAINING_ARCHIVE" not in "\n".join(online.values())
    assert "archive_path" not in online["export"]
    assert "EVAL_WINDOW_SECONDS = 900" in online["parameters"]
    assert "EVAL_ARRIVAL_MULTIPLIERS = [1, 4, 8]" in online["parameters"]
    assert 'DEVICE = "auto"' in online["parameters"]
    assert "device=DEVICE" in online["run"]
    assert 'observation={"include_episode_progress": False}' in online["train"]
    assert "arrival_multipliers=TRAIN_ARRIVAL_MULTIPLIERS" in online["train"]
    assert 'train_config.setdefault("agent", {})["device"] = DEVICE' in online["train"]
    for text in ("training.csv", "checkpoints.csv", "validation.csv", "TRAIN_CONFIG_PATH"):
        assert text in online["export"]


def test_combined_notebook_code_cells_compile_without_saved_outputs():
    path = ROOT / "notebooks/colab_burstgpt3_online_adaptation.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["execution_count"] is None
            assert not cell["outputs"]


def test_five_policy_stress_replaces_legacy_baseline_addon():
    online = cells("colab_burstgpt3_online_adaptation.ipynb")
    assert "baseline-addon" not in online
    assert "experiments.adaptation_stress" in online["run"]
    for metric in ("p99_ttft_s", "ttft_target_attainment", "first_token_coverage"):
        assert metric in online["charts"]
    assert "length_groups" in online["export"]
