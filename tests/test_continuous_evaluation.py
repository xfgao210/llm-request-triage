import csv
import json

import pytest

from experiments import continuous_evaluation as continuous
from llm_triage.workload.burstgpt import sha256


@pytest.mark.parametrize("finish_time", [76.1, 112.5, 113.4])
def test_reporting_preserves_empty_bins_and_uses_completion_clock(tmp_path, finish_time):
    output, data = tmp_path / "run", tmp_path / "data"
    processed = data / "processed"
    processed.mkdir(parents=True)
    manifest = dict(range_config=dict(ranges=[dict(split="test", group="mixed", start_timestamp_s=1000,
                                                 end_timestamp_s=1900)]), windows=[
        dict(split="test", start_timestamp_s=1000+i*300, end_timestamp_s=1300+i*300,
             requests=n, input_tokens=n*20, output_tokens=n*10)
        for i,n in enumerate((2, 0, 1))])
    manifest_path = processed / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    window = continuous.first_test_window(manifest)
    folder = output / "evaluation/mixed_full/load_8/frozen_seed7"
    folder.mkdir(parents=True)
    (output / "continuous_plan.json").write_text(json.dumps(dict(window=window, multiplier=8,
        policies=["frozen"], seed=7, manifest_sha256=sha256(manifest_path))))
    (output / "evaluation/run.json").write_text(json.dumps(dict(status="completed")))
    rows = [dict(arrival_time_s=37.4, status="completed", slo_met=True, ttft_s=.1, completion_latency_s=.2),
            dict(arrival_time_s=0, status="dropped", slo_met=False, ttft_s="", completion_latency_s=""),
            dict(arrival_time_s=finish_time-1, status="completed", slo_met=True, ttft_s=.9, completion_latency_s=1)]
    path = folder / "request_diagnostics.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    timeline = [dict(source_start_s=1000+i*300, sim_start_s=i*37.5, sim_end_s=(i+1)*37.5,
                     mean_queue=i+1, busy_fraction=.5, queue_max=i+2) for i in range(3)]
    (folder / "timeline.json").write_text(json.dumps(timeline))
    (folder / "case.json").write_text(json.dumps(dict(status="completed", files={
        p:sha256(folder / p) for p in ("timeline.json", "request_diagnostics.csv")})))
    summary = continuous.build_reports(output, data, min_arrivals=1)
    with (output / "timeline_5min.csv").open() as stream:
        five = list(csv.DictReader(stream))
    assert five[0]["slo_attainment"] == "0.5"
    assert five[0]["interval_on_time_completions"] == "0"
    assert five[1]["arrivals"] == "0" and five[1]["slo_attainment"] == ""
    assert five[1]["interval_on_time_completions"] == "1"
    assert five[1]["mean_queue"] == "2.0"  # No reset in an empty arrival bin.
    with (output / "timeline_15min.csv").open() as stream:
        fifteen = list(csv.DictReader(stream))[0]
    assert float(fifteen["slo_attainment"]) == pytest.approx(2/3)
    assert float(fifteen["p95_ttft_s"]) == pytest.approx(.86)  # Not mean of bin percentiles.
    assert float(fifteen["mean_queue"]) == 2
    # An on-time request can finish outside the measurement range. It remains
    # a successful arrival-cohort outcome, but must not count as in-window goodput.
    in_window_completions = 1 if finish_time >= 112.5 else 2
    assert float(fifteen["completion_goodput_rps"]) == pytest.approx(in_window_completions/112.5)
    with (output / "timeline_10min.csv").open() as stream:
        ten = list(csv.DictReader(stream))
    assert ten[-1]["source_end_s"] == "1900"
    assert sum(int(r["arrivals"]) for r in ten) == 3
    assert sum(int(r["interval_on_time_completions"]) for r in ten) == in_window_completions
    assert summary[0]["eligible_source_time_fraction"] == pytest.approx(2/3)
    assert summary[0]["fraction_eligible_time_below_target"] == .5
    with pytest.raises(ValueError, match="Cadences"):
        continuous.build_reports(output, data, cadences=(7,))


def test_continuous_notebook_code_compiles():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "notebooks/colab_burstgpt3_continuous_first_test.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert cell["execution_count"] is None and not cell["outputs"]


def test_time_bins_use_actual_edges_at_floating_point_boundaries():
    import math
    from llm_triage.metrics.time_bins import TimeBins
    from llm_triage.metrics.schemas import RequestRecord, RequestStatus
    from llm_triage.metrics.summary import summarize_records
    width = 60/9
    clock = TimeBins([i*width for i in range(6)])
    assert int(20.0 // width) == 2  # Regression: quotient-floor indexing was wrong.
    assert clock.index(20.0) == 3
    assert clock.index(math.nextafter(20.0, -math.inf)) == 2
    row = RequestRecord(request_id="boundary", arrival_time_s=20, deadline_s=25,
                        status=RequestStatus.DROPPED, decision_time_s=20)
    index = clock.index(row.arrival_time_s)
    assert summarize_records([row], measurement_window_s=width,
                             window_start_s=clock.edges[index])["arrivals"] == 1
    partial = TimeBins([0, 75, 112.5])
    assert partial.index(112.4) == 1
    assert partial.index(112.5) is None and partial.index(113) is None
    assert partial.index(-1) is None
    with pytest.raises(ValueError):
        partial.index(math.nan)
    with pytest.raises(ValueError):
        TimeBins([0, 0, 1])


def test_pilot_skips_leading_empty_bins_without_sampling_or_mutating_manifest():
    manifest = dict(range_config=dict(episode_seconds=300, ranges=[
        dict(split="test", group="mixed", start_timestamp_s=0, end_timestamp_s=7200)]),
        windows=[dict(split="test", start_timestamp_s=i*300, end_timestamp_s=(i+1)*300,
                      requests=0 if i < 2 else i+1) for i in range(24)])
    before = json.dumps(manifest)
    pilot = continuous.first_test_window(manifest, source_minutes=60)
    assert pilot["start_timestamp_s"] == 600
    assert pilot["end_timestamp_s"] == 4200
    assert pilot["requests"] == sum(range(3, 15))
    assert pilot["name"] == "mixed_pilot_60min"
    whole = continuous.first_test_window(manifest)
    assert whole["start_timestamp_s"] == 0 and whole["end_timestamp_s"] == 7200
    assert json.dumps(manifest) == before
    explicit = continuous.first_test_window(manifest, source_minutes=30, source_start_s=1200)
    assert explicit["start_timestamp_s"] == 1200 and explicit["end_timestamp_s"] == 3000
    with pytest.raises(ValueError, match="Pilot start"):
        continuous.first_test_window(manifest, source_minutes=60, source_start_s=1234)
    for minutes in (0, 1, 7, True, -5):
        with pytest.raises(ValueError, match="Pilot minutes"):
            continuous.first_test_window(manifest, source_minutes=minutes)


def test_progress_is_wall_clock_throttled_and_reports_tail(monkeypatch, capsys):
    from types import SimpleNamespace
    from experiments.replay_progress import ReplayProgress
    clock = [10.0]
    monkeypatch.setattr("experiments.replay_progress.time.monotonic", lambda: clock[0])
    replay = SimpleNamespace(source_origin_s=100, time_scale=.125, horizon_s=450)
    kernel = SimpleNamespace(now=0, completed_count=0, violated_count=0, dropped_count=0,
                             queue=[1, 2, 3], _preempted_queue=[4], _next=0, requests=[0]*5)
    progress = ReplayProgress("fcfs", replay, 30)
    progress(kernel)
    assert "queue=4" in capsys.readouterr().out
    progress(kernel)
    assert capsys.readouterr().out == ""
    clock[0] = 41
    kernel.now, kernel._next, kernel.completed_count = 450, 5, 5
    progress(kernel)
    output = capsys.readouterr().out
    assert "resolved=5/5" in output and "phase=tail" in output and "source_time=3700.0" in output
    with pytest.raises(ValueError):
        ReplayProgress("invalid", replay, 0)


def test_progress_observation_does_not_change_simulation_outcomes():
    from llm_triage.simulation import Request, simulate
    from llm_triage.scheduling import baseline_schedulers
    requests = [Request("a", 0, 16, 4, 5), Request("b", .1, 32, 8, 5.1)]
    reference = simulate(requests, baseline_schedulers()["fcfs"])
    observations = []
    observed = simulate(requests, baseline_schedulers()["fcfs"],
                        progress_callback=lambda kernel: observations.append(kernel.now))
    assert observations and observations == sorted(observations)
    assert reference.total_time == observed.total_time
    assert [(r.state, r.completion_time, r.gpu_time_s) for r in reference.requests] == [
        (r.state, r.completion_time, r.gpu_time_s) for r in observed.requests]
