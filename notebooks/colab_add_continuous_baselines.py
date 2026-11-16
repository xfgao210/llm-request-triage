# Paste this entire file into ONE NEW CELL after the existing online comparison
# and export cells. It needs the original Colab runtime/output folders, not just
# the downloaded HTML. No training or frozen/adapting DQN replay is performed.
from pathlib import Path
import base64
import html
import json
import math
import time
import zipfile

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display, Image, FileLink

from llm_triage.metrics import evaluate_events
from llm_triage.metrics.summary import summarize_records
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import HardwareConfig, ServiceProfile, simulate
from llm_triage.workload.burstgpt import load_windows, sha256
from llm_triage.workload.transforms import replay_window


def add_continuous_baselines(online_dir, training_dir, data_dir, destination):
    """Add missing policies using the completed run's exact continuous replay."""
    completed = json.loads((online_dir / "run.json").read_text())
    assert completed["status"] == "completed", "Finish frozen/adapting replay first."
    assert completed["protocol"] == "online_prequential_continuous_v1"
    training_path = training_dir / "run.json"
    training = json.loads(training_path.read_text())
    assert sha256(training_path) == completed["source_training_record_sha256"]
    manifest_path = data_dir / "processed/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    trace_path = data_dir / "processed/burstgpt_windows.csv"
    assert sha256(manifest_path) == completed["manifest_sha256"]
    assert sha256(trace_path) == completed["trace_sha256"] == manifest["selected_sha256"]
    assert manifest["source_sha256"] == completed["source_sha256"]
    checkpoint = training_dir / f"dqn_seed{completed['settings']['seed']}.npz"
    assert sha256(checkpoint) == completed["source_checkpoint_sha256"]
    old_summary = pd.read_csv(online_dir / "summary.csv")
    old_timeline = pd.read_csv(online_dir / "timeline.csv")
    assert set(old_summary["policy"]) == {"frozen", "adapting"}

    # Verify relevant installed code against the original exported provenance,
    # when available. Do not pull/switch simulator versions between comparisons.
    repo = Path.cwd()
    source_paths = [p for folder in ("simulation", "scheduling", "metrics", "workload")
                    for p in (repo / "src/llm_triage" / folder).rglob("*.py")]
    assert source_paths, "Run from the repository root, as in the setup cell."
    code = {p.relative_to(repo).as_posix(): sha256(p) for p in sorted(source_paths)}
    original_provenance = online_dir.parent / "online_provenance.json"
    if original_provenance.exists():
        original_code = json.loads(original_provenance.read_text())["code_sha256"]
        assert all(original_code.get(k) == v for k, v in code.items()), "Simulator code changed; restore the original checkout."
    identity = dict(protocol="continuous_baseline_addon_v1", online_run_sha256=sha256(online_dir / "run.json"),
                    summary_sha256=sha256(online_dir / "summary.csv"),
                    timeline_sha256=sha256(online_dir / "timeline.csv"), code_sha256=code)
    destination.mkdir(parents=True, exist_ok=True)
    ranges = completed["selected_ranges"]  # Includes any prefix/group selection actually used.
    origin, end = ranges[0]["start_timestamp_s"], ranges[-1]["end_timestamp_s"]
    bin_seconds = manifest["range_config"]["episode_seconds"]
    replay, labels = None, None
    summary_frames, timeline_frames = [old_summary], [old_timeline]

    class Progress:
        def __init__(self, name, scheduler):
            self.name, self.scheduler, self.decisions = name, scheduler, 0

        def __getattr__(self, name):
            return getattr(self.scheduler, name)  # Preserve each baseline's priority rule.

        def decide(self, state):
            self.decisions += 1
            if self.decisions % 100000 == 0:
                print(f"{self.name}: decisions={self.decisions}, source_time={origin + state.now_s:.1f}", flush=True)
            return self.scheduler.decide(state)

    for name, scheduler in baseline_schedulers().items():
        folder = destination / name
        record_path = folder / "run.json"
        saved = json.loads(record_path.read_text()) if record_path.exists() else None
        if saved is not None:
            assert saved["identity"] == identity, "Cached inputs differ; choose a new destination folder."
        if saved is not None and saved["status"] == "completed":
            for filename, digest in saved["files"].items():
                assert sha256(folder / filename) == digest, "Cached baseline output changed."
            print("Reusing completed baseline:", name)
        else:
            # Load requests only if a baseline still needs to run. Preserve the
            # exact sorting, source IDs, bounds, idle gaps and native time scale.
            if replay is None:
                windows = load_windows(trace_path)
                rows = sorted((r for (split, _), values in windows.items() if split == "test"
                               for r in values if any(g["start_timestamp_s"] <= r.timestamp_s < g["end_timestamp_s"]
                                                      for g in ranges)), key=lambda r: (r.timestamp_s, r.source_row))
                replay = replay_window(rows, slo_s=training["config"]["slo_s"], bounds=(origin, end))
                labels = {}
                for row in rows:
                    group = next(g for g in ranges if g["start_timestamp_s"] <= row.timestamp_s < g["end_timestamp_s"])
                    labels[str(row.source_row)] = (group["group"], int((row.timestamp_s-group["start_timestamp_s"]) // bin_seconds))
                del windows, rows
                for group in ranges:
                    expected = sum(label[0] == group["group"] for label in labels.values())
                    paired = old_summary[old_summary["group"] == group["group"]]
                    assert len(paired) == 2 and (paired["arrivals"] == expected).all(), "DQN cohort differs."
            folder.mkdir(exist_ok=True)
            record_path.write_text(json.dumps(dict(identity=identity, status="running"), indent=2))
            print("Running continuous baseline:", name, flush=True)
            started = time.monotonic()
            callback = Progress(name, scheduler)
            simulation = simulate(replay.requests, callback,
                                  HardwareConfig(**training["config"]["hardware"]),
                                  ServiceProfile(**training["service_profile"]),
                                  estimated_output_tokens=training["estimated_output_tokens"],
                                  record_iterations=False, **training["config"]["decisions"])
            result = evaluate_events(simulation.events, measurement_window_s=replay.horizon_s)
            assert {r.request_id for r in result.request_records} == set(labels), "Unpaired baseline cohort."
            by_group, by_bin = {}, {}
            for row in result.request_records:
                group, index = labels[row.request_id]
                by_group.setdefault(group, []).append(row)
                by_bin.setdefault((group, index), []).append(row)
            scores, timeline = [], []
            for group in ranges:
                label = group["group"]
                start, stop = group["start_timestamp_s"], group["end_timestamp_s"]
                score = summarize_records(by_group.get(label, []), measurement_window_s=stop-start, window_start_s=start-origin)
                scores.append(dict(policy=name, group=label, **score))
                for index in range(math.ceil((stop-start) / bin_seconds)):
                    a, b = start + index * bin_seconds, min(stop, start + (index+1) * bin_seconds)
                    score = summarize_records(by_bin.get((label, index), []), measurement_window_s=b-a, window_start_s=a-origin)
                    timeline.append(dict(policy=name, group=label, source_start_s=a, source_end_s=b, **score))
            pd.DataFrame(scores).to_csv(folder / "summary.csv", index=False)
            pd.DataFrame(timeline).to_csv(folder / "timeline.csv", index=False)
            record_path.write_text(json.dumps(dict(identity=identity, status="completed", policy=name,
                decisions=callback.decisions, wall_seconds=time.monotonic()-started,
                selected_ranges=ranges, training_config=training["config"],
                service_profile=training["service_profile"],
                files={f: sha256(folder / f) for f in ("summary.csv", "timeline.csv")}), indent=2))
            print(name, "completed", flush=True)
            del simulation, result, by_group, by_bin
        summary_frames.append(pd.read_csv(folder / "summary.csv"))
        timeline_frames.append(pd.read_csv(folder / "timeline.csv"))

    unified_summary = pd.concat(summary_frames, ignore_index=True)
    unified_timeline = pd.concat(timeline_frames, ignore_index=True)
    order = ["fcfs", "length_aware", "static_slo_aware", "frozen", "adapting"]
    expected = {(g["group"], p) for g in ranges for p in order}
    assert set(zip(unified_summary["group"], unified_summary["policy"])) == expected
    assert not unified_summary.duplicated(["group", "policy"]).any()
    unified_summary = unified_summary.sort_values(["group", "policy"]).reset_index(drop=True)
    unified_summary.to_csv(destination / "unified_summary.csv", index=False)
    unified_timeline.to_csv(destination / "unified_timeline.csv", index=False)
    (destination / "provenance.json").write_text(json.dumps(dict(identity=identity,
        original_online_run=completed, original_training_record=training,
        simulator_code_verified_against_export=original_provenance.exists(),
        notes="Independent continuous replays; same native traffic, bounds, gaps, capacity, SLO and train-only length estimate. No DQN replay or training repeated."), indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, metric, title in zip(axes, ["slo_attainment", "drop_rate", "window_goodput_rps"],
                                ["SLO attainment", "Drop rate", "Goodput (req/s)"]):
        unified_summary.pivot(index="group", columns="policy", values=metric).reindex(columns=order).plot.bar(ax=ax, rot=0)
        ax.set_title(title)
        if metric != "window_goodput_rps":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(0, 1.05 if metric == "slo_attainment" else max(0.05, float(unified_summary[metric].max())*1.1))
    fig.tight_layout()
    fig.savefig(destination / "unified_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(len(ranges), 1, figsize=(14, 4*len(ranges)), squeeze=False)
    for ax, group in zip(axes[:, 0], ranges):
        for policy in order:
            rows = unified_timeline[(unified_timeline["group"] == group["group"]) & (unified_timeline["policy"] == policy)].sort_values("source_start_s")
            ax.plot(rows["source_start_s"], rows["slo_attainment"].where(rows["arrivals"] > 0), label=policy)
        ax.set(title=group["group"], xlabel="Original CSV timestamp (seconds)", ylabel="Arrival-cohort SLO attainment", ylim=(0, 1.05))
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "unified_timeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    columns = ["group", "policy", "arrivals", "slo_met", "slo_attainment", "drop_rate", "window_goodput_rps"]
    display(unified_summary[columns])
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Unified policy comparison</title>",
             "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:16px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}pre{white-space:pre-wrap}</style></head><body>",
             "<h1>Unified continuous-replay policy comparison</h1><p>Same traffic, hardware, completion SLO, and intervals for all five policies. Existing DQN results reused; only the three rule-based policies were added. Online adaptation is not frozen held-out evaluation.</p>",
             unified_summary[columns].to_html(index=False),
             "<p>SLO denominator includes all arrivals, including drops. Timeline scores use eventual arrival-cohort outcomes for reporting only. Goodput counts on-time completions of each group's arrivals within its interval; gaps inside intervals contribute duration. Queues persist across bins/groups; omitted gaps are treated as idle.</p>"]
    for filename in ("unified_comparison.png", "unified_timeline.png"):
        parts.append("<img alt='" + filename + "' src='data:image/png;base64," + base64.b64encode((destination / filename).read_bytes()).decode() + "'>")
    parts += ["<h2>Actual replay settings</h2><pre>", html.escape(json.dumps(completed, indent=2)), "</pre></body></html>"]
    (destination / "unified_report.html").write_text("\n".join(parts), encoding="utf-8")
    members = [destination / f for f in ("unified_summary.csv", "unified_timeline.csv", "provenance.json",
               "unified_report.html", "unified_comparison.png", "unified_timeline.png")]
    members += [destination / p / f for p in order[:3] for f in ("summary.csv", "timeline.csv", "run.json")]
    archive = destination / "unified_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in members:
            bundle.write(path, path.relative_to(destination).as_posix())
    display(Image(filename=str(destination / "unified_comparison.png")))
    display(FileLink(str(destination / "unified_report.html")))
    display(FileLink(str(archive)))
    return unified_summary, unified_timeline, archive


# These paths were created by the existing notebook. Do not rerun its settings
# or training cells: that could point to a new, empty RUN_ROOT instead.
if globals().get("RUN_RULE_BASELINES", True):
    UNIFIED_DIR = Path(ONLINE_OUTPUT).parent / "unified_comparison"
    unified_summary, unified_timeline, UNIFIED_ARCHIVE = add_continuous_baselines(
        Path(ONLINE_OUTPUT), Path(TRAINING_DIR), Path(DATA_DIR), UNIFIED_DIR)
    if globals().get("IS_COLAB", False):
        from google.colab import files
        files.download(str(UNIFIED_ARCHIVE))
else:
    print("Rule-based comparison skipped; existing DQN report remains available.")
