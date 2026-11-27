"""Development selection, load scaling, diagnostics and safe resume."""

import csv
import json
from pathlib import Path

import pytest

from experiments import baseline_capacity_sweep as sweep
from llm_triage.simulation import HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import sha256
from llm_triage.workload.time_ranges import prepare_ranges
from scripts.analyze_burstgpt_traffic import analyze


@pytest.fixture
def capacity_data(tmp_path):
    source = tmp_path / "BurstGPT_3.csv"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Timestamp", "Session ID", "Elapsed time", "Model",
                         "Request tokens", "Response tokens", "Total tokens", "Log Type"])
        for start, count, output in ((0, 6, 4), (1200, 12, 4), (2100, 36, 4), (3600, 2, 999)):
            for index in range(count):
                writer.writerow([start + index / 100, "", 1, "ChatGPT", 16, output, 16 + output, "API log"])
    analysis = tmp_path / "analysis"
    analyze(source, analysis)
    ranges = dict(schema_version=1, source_sha256=sha256(source), episode_seconds=300,
                  min_split_gap_seconds=300, ranges=[
                      dict(split="train", start_timestamp_s=0, end_timestamp_s=900),
                      dict(split="validation", start_timestamp_s=1200, end_timestamp_s=3000),
                      dict(split="test", start_timestamp_s=3300, end_timestamp_s=3900),
                  ])
    config_path = tmp_path / "ranges.json"
    config_path.write_text(json.dumps(ranges))
    data = tmp_path / "data"
    manifest = prepare_ranges(source, analysis, data, config_path)
    return data, manifest, source, config_path


def test_development_only_selection_and_no_internal_resets(capacity_data):
    data, manifest, _, _ = capacity_data
    windows = sweep.select_development_windows(manifest)
    assert [w["start_timestamp_s"] for w in windows] == [1200, 2100]
    assert [w["requests"] for w in windows] == [12, 36]
    rows = sweep.load_selected_rows(data / "processed/burstgpt_windows.csv", windows)
    assert sum(map(len, rows.values())) == 48
    assert all(r.true_output_tokens == 4 for group in rows.values() for r in group)
    with pytest.raises(ValueError, match="never test"):
        sweep.select_development_windows(manifest, split="test")
    with pytest.raises(ValueError, match="train/validation"):
        sweep.validate_windows(manifest, [dict(name="heldout", split="test",
                                             start_timestamp_s=3300, end_timestamp_s=3900)])
    with pytest.raises(ValueError, match="overlap"):
        sweep.validate_windows(manifest, [windows[0], dict(windows[0], name="duplicate")])


def test_paired_scaling_metrics_and_cache(capacity_data, tmp_path, monkeypatch):
    data, manifest, _, _ = capacity_data
    windows = sweep.select_development_windows(manifest)
    original = sweep.simulate
    observed = []

    def capture(requests, callback, hardware, profile, **kwargs):
        observed.append([(r.request_id, r.arrival_time, r.slo_deadline) for r in requests])
        assert kwargs["estimated_output_tokens"] == 4  # training, not test mean
        return original(requests, callback, hardware, profile, **kwargs)

    monkeypatch.setattr(sweep, "simulate", capture)
    options = dict(multipliers=[1, 8], slo_s=1, hardware=HardwareConfig(max_batch_size=1),
                   profile=ServiceProfile(decode_rate=10))
    destination = tmp_path / "sweep"
    result = sweep.run_sweep(data, destination, windows, **options)
    assert len(result) == len(observed) == 12
    assert observed[0] == observed[1] == observed[2]
    for native, scaled in zip(observed[0], observed[3]):
        assert native[0] == scaled[0]
        assert scaled[1] == pytest.approx(native[1] / 8)
        assert native[2] - native[1] == pytest.approx(1)
        assert scaled[2] - scaled[1] == pytest.approx(1)
    assert all(r["arrivals"] in (12, 36) for r in result)
    assert all(0 <= r["busy_fraction_in_window"] <= 1.0000001 for r in result)
    assert any(r["slo_attainment"] < 1 and r["missed_but_isolated_within_slo"] > 0 for r in result)
    assert all(r["isolated_exceeds_slo"] == 0 for r in result)
    assert all(r["drop_rate"] == 0 for r in result if r["policy"] == "fcfs")
    assert sweep.run_sweep(data, destination, windows, **options) == result
    assert len(observed) == 12  # all completed cases skipped
    with pytest.raises(ValueError, match="NEW RUN_NAME"):
        sweep.run_sweep(data, destination, windows, **dict(options, slo_s=2))
    cached = destination / "typical_development/load_1/fcfs/summary.json"
    cached.write_text("{}")
    with pytest.raises(ValueError, match="Cached output changed"):
        sweep.run_sweep(data, destination, windows, **options)


def test_interrupted_case_resumes_only_incomplete_case(capacity_data, tmp_path, monkeypatch):
    data, manifest, _, _ = capacity_data
    windows = sweep.select_development_windows(manifest)
    destination = tmp_path / "resume"
    original = sweep.simulate
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(sweep, "simulate", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        sweep.run_sweep(data, destination, windows, multipliers=[1])
    first = destination / "typical_development/load_1/fcfs/summary.json"
    first_digest = sha256(first)
    monkeypatch.setattr(sweep, "simulate", original)
    assert len(sweep.run_sweep(data, destination, windows, multipliers=[1])) == 6
    assert sha256(first) == first_digest


def test_isolated_service_diagnostic_is_not_queueing(capacity_data, tmp_path):
    data, manifest, _, _ = capacity_data
    windows = sweep.select_development_windows(manifest)
    result = sweep.run_sweep(data, tmp_path / "too-short-slo", windows, multipliers=[1],
                             slo_s=.1, profile=ServiceProfile(decode_rate=10))
    assert all(row["isolated_exceeds_slo"] == row["arrivals"] for row in result)
    assert all(row["missed_but_isolated_within_slo"] == 0 for row in result)


def test_capacity_notebook_is_clean_and_compiles():
    path = Path(__file__).resolve().parents[1] / "notebooks/colab_burstgpt3_capacity_sweep.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_setup_imports_src_without_installed_package():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/colab_burstgpt3_capacity_sweep.ipynb").read_text(encoding="utf-8"))
    setup = next("".join(cell["source"]) for cell in notebook["cells"] if cell["id"] == "setup")
    # No PYTHONPATH, user site, or editable-install startup hooks. This models
    # the running kernel before it has processed a newly installed .pth file.
    result = subprocess.run([sys.executable, "-I", "-S", "-c", setup], cwd=root,
                            capture_output=True, text=True, check=True)
    assert str(root / "src/llm_triage/__init__.py") in result.stdout


def test_entire_notebook_with_small_trace(capacity_data, tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    pytest.importorskip("IPython")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import zipfile

    data, _, source, ranges = capacity_data
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    notebook = json.loads((root / "notebooks/colab_burstgpt3_capacity_sweep.ipynb").read_text(encoding="utf-8"))
    namespace = dict(__name__="__main__")
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        if cell["id"] == "setup":
            namespace.update(REPO=root, IN_COLAB=False, Path=Path)
            continue
        exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)
        if cell["id"] == "parameters":
            namespace.update(CSV_PATH=source, OUTPUT_ROOT=tmp_path / "reports",
                             DATA_DIR=tmp_path / "fresh-data", ANALYSIS_DIR=data.parent / "analysis",
                             RANGE_CONFIG=ranges)
    output = namespace["RUN_DIR"]
    with (output / "summary.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 24
    assert (output / "capacity_report.html").stat().st_size > 10000
    assert len(list((output / "figures").glob("*.png"))) == 4
    with zipfile.ZipFile(namespace["archive_path"]) as archive:
        names = archive.namelist()
        assert "capacity_report.html" in names
        assert len([n for n in names if n.endswith("request_diagnostics.csv")]) == 24
        assert not any("burstgpt_windows" in n or "BurstGPT_3.csv" in n for n in names)
