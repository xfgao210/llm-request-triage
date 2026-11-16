"""Keep the combined Colab workflow self-contained without changing training."""

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
    for name in ("source", "data", "train", "learning-charts", "validation-charts"):
        assert online[name] == offline[name]
    assert "TRAINING_ARCHIVE" not in "\n".join(online.values())
    assert "archive_path" not in online["export"]
    assert "MAX_SECONDS_PER_GROUP = None" in online["parameters"]
    assert "RUN_ONLINE_EXPERIMENT = True" in online["parameters"]
    assert 'DEVICE = "auto"' in online["parameters"]
    assert '"--device", DEVICE' in online["run"]
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


def test_baseline_cell_matches_standalone_paste_in_patch():
    online = cells("colab_burstgpt3_online_adaptation.ipynb")
    patch = (ROOT / "notebooks/colab_add_continuous_baselines.py").read_text()
    assert online["baseline-addon"] == patch
    assert list(online).index("baseline-addon") > list(online).index("export")
    assert "RUN_RULE_BASELINES = True" in online["parameters"]
