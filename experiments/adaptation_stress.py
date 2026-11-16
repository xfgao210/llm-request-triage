"""Resumable five-policy load-shift evaluation on short, predeclared windows."""

import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import gymnasium
import numpy as np

from experiments.baseline_capacity_sweep import (
    diagnose_replay, load_selected_rows, write_csv, write_json,
)
from experiments.online_adaptation import replay_dqn
from experiments.run_pipeline import run_lock
from llm_triage.metrics import evaluate_events, summarize_records
from llm_triage.rl.devices import resolve_device
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import HardwareConfig, ServiceProfile, simulate
from llm_triage.workload.burstgpt import sha256
from llm_triage.workload.transforms import replay_window


POLICIES = ("fcfs", "length_aware", "static_slo_aware", "frozen", "adapting")


def select_run_conditions(windows, multipliers, seeds, *, mode="full", quick_seed=None):
    """Reduce independent cases, never requests or the selected window bounds.

    Selection occurs AFTER the usual full-window selection, so quick's peak
    condition is exactly the same one that full mode would run.
    """
    if mode not in ("quick", "full"):
        raise ValueError("RUN_MODE must be quick or full")
    if not windows or not multipliers or not seeds:
        raise ValueError("Select windows, arrival multipliers and seeds first")
    if mode == "full":
        return [dict(w) for w in windows], list(multipliers), list(seeds)
    peaks = [w for w in windows if w["group"] == "peak_heavy"]
    if len(peaks) != 1 or 8 not in multipliers:
        raise ValueError("Quick mode requires exactly one peak_heavy window and an existing 8x condition")
    seed = seeds[0] if quick_seed is None else quick_seed
    if seed not in seeds:
        raise ValueError("QUICK_SEED must be in SEEDS")
    return [dict(peaks[0])], [8], [seed]


def select_evaluation_windows(manifest, seconds=900, methods=None):
    """Traffic-only selection, with rules declared before policy evaluation.

    These are reused exploratory evaluation groups, not newly untouched data.
    Typical = median-count nonempty block; burst = highest sub-bin request rate.
    """
    methods = methods or {"mixed_traffic": "typical", "peak_heavy": "burst"}
    bin_s = manifest["range_config"]["episode_seconds"]
    if not math.isfinite(seconds) or seconds < bin_s or seconds % bin_s:
        raise ValueError("Evaluation duration must be a multiple of episode_seconds")
    bins = manifest["windows"] + manifest.get("empty_windows", [])
    chosen = []
    for group, method in methods.items():
        if method not in ("typical", "burst"):
            raise ValueError("Selection method must be typical or burst")
        periods = [r for r in manifest["range_config"]["ranges"] if r["split"] == "test" and r["group"] == group]
        if len(periods) != 1:
            raise ValueError(f"Require one explicit test range for {group}")
        period = periods[0]
        candidates = []
        origin, end = period["start_timestamp_s"], period["end_timestamp_s"]
        for index in range(int((end-origin) // seconds)):
            start = origin + index * seconds
            selected = [b for b in bins if b["split"] == "test" and start <= b["start_timestamp_s"] < start + seconds]
            if sum(b["end_timestamp_s"]-b["start_timestamp_s"] for b in selected) != seconds:
                raise ValueError("Manifest does not cover complete evaluation blocks")
            count = sum(b["requests"] for b in selected)
            if count:
                candidates.append(dict(name=group, group=group, split="test", start_timestamp_s=start,
                                       end_timestamp_s=start+seconds, requests=count, selection=method,
                                       peak_subbin_rps=max(b["requests"] / (b["end_timestamp_s"]-b["start_timestamp_s"])
                                                           for b in selected)))
        if not candidates:
            raise ValueError(f"No nonempty complete blocks in {group}; shorten evaluation windows")
        median = statistics.median(c["requests"] for c in candidates)
        chosen.append(max(candidates, key=lambda c: (c["peak_subbin_rps"], c["requests"], -c["start_timestamp_s"]))
                      if method == "burst" else min(candidates, key=lambda c: (abs(c["requests"]-median), c["start_timestamp_s"])))
    return sorted(chosen, key=lambda w: w["start_timestamp_s"])


def validate_evaluation_windows(manifest, windows):
    import re
    if not windows:
        raise ValueError("Select evaluation windows first")
    names = set()
    for window in windows:
        name = window["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError("Use unique safe window names")
        names.add(name)
        start, end = window["start_timestamp_s"], window["end_timestamp_s"]
        if not all(math.isfinite(t) for t in (start, end)) or not start < end:
            raise ValueError("Invalid evaluation timestamps")
        if window["split"] != "test" or not any(
            r["split"] == "test" and r["group"] == window["group"]
            and r["start_timestamp_s"] <= start < end <= r["end_timestamp_s"]
            for r in manifest["range_config"]["ranges"]
        ):
            raise ValueError("Evaluation windows must lie within their explicit test group")
    ordered = sorted(windows, key=lambda w: w["start_timestamp_s"])
    if any(a["end_timestamp_s"] > b["start_timestamp_s"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Evaluation windows overlap")


def _ttft_score(records, target):
    return dict(first_token_coverage=sum(r.ttft_s is not None for r in records) / len(records) if records else 0.0,
                ttft_target_attainment=sum(r.ttft_s is not None and r.ttft_s <= target for r in records) / len(records) if records else 0.0)


def run(training_dir, data_dir, output, windows, *, multipliers=(1, 4, 8), seeds=(7,),
        ttft_target_s=1.0, timeline_source_bin_s=60.0, max_requests=30000,
        epsilon=0.0, learning_rate=.0001, batch_size=64, warmup=256,
        update_interval=4, capacity=50000, device="auto"):
    """Every window/load/seed is independent; each policy cache is resumable.

    Baselines are seed-independent and run once per window/load. Each DQN copy
    starts from its corresponding source checkpoint, with fresh optimizer/buffer.
    No adaptation carries between independent windows or load conditions.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        return _run(Path(training_dir), Path(data_dir), output, windows, multipliers=list(multipliers),
                    seeds=list(seeds), ttft_target_s=ttft_target_s, timeline_source_bin_s=timeline_source_bin_s,
                    max_requests=max_requests, epsilon=epsilon, learning_rate=learning_rate,
                    batch_size=batch_size, warmup=warmup, update_interval=update_interval,
                    capacity=capacity, device=device)


def _run(training_dir, data_dir, output, windows, **settings):
    for key in ("batch_size", "warmup", "update_interval", "capacity", "max_requests"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not settings["batch_size"] <= settings["warmup"] <= settings["capacity"]:
        raise ValueError("Require batch_size <= warmup <= capacity")
    for key in ("ttft_target_s", "timeline_source_bin_s", "learning_rate"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not math.isfinite(settings["epsilon"]) or not 0 <= settings["epsilon"] <= 1:
        raise ValueError("Invalid epsilon")
    levels, seeds = settings["multipliers"], settings["seeds"]
    if (not levels or any(isinstance(m, bool) or not math.isfinite(m) or m <= 0 for m in levels)
            or len({f"{m:g}" for m in levels}) != len(levels)):
        raise ValueError("Use distinct positive multipliers")
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Use distinct nonnegative seeds")
    resolved_device = resolve_device(settings["device"])
    runtime = dict(python=platform.python_version(), numpy=np.__version__, gymnasium=gymnasium.__version__,
                   resolved_device=resolved_device)
    if resolved_device == "cuda":
        import torch
        runtime.update(torch=str(torch.__version__), cuda=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0))
    record_path = training_dir / "run.json"
    record = json.loads(record_path.read_text())
    if record["selection_split"] != "validation" or record["test_evaluated"]:
        raise ValueError("Require a validation-selected checkpoint")
    if record["config"].get("observation", {}).get("include_episode_progress") is not False:
        raise ValueError("Retrain with include_episode_progress=false before running this protocol")
    manifest_path = data_dir / "processed/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    trace = data_dir / "processed/burstgpt_windows.csv"
    if (sha256(trace) != manifest["selected_sha256"] or
            record["trace_sha256"] != manifest["selected_sha256"] or
            record["manifest_sha256"] != sha256(manifest_path) or
            record["source_sha256"] != manifest["source_sha256"]):
        raise ValueError("Evaluation data differs from checkpoint training data")
    validate_evaluation_windows(manifest, windows)
    rows_by_window = load_selected_rows(trace, windows, allowed_splits=("test",))
    if any(len(rows) > settings["max_requests"] for rows in rows_by_window.values()):
        raise ValueError("Selected window exceeds max_requests; shorten windows")
    checkpoints = {}
    for seed in seeds:
        path = training_dir / f"dqn_seed{seed}.npz"
        expected = next((c["sha256"] for c in record["checkpoints"] if c["seed"] == seed), None)
        if expected is None or sha256(path) != expected:
            raise ValueError(f"Missing/changed selected checkpoint for seed {seed}")
        checkpoints[str(seed)] = expected
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root / "src/llm_triage").rglob("*.py")),
             *(root / "experiments" / name for name in ("adaptation_stress.py", "online_adaptation.py",
                                                        "baseline_capacity_sweep.py", "train_dqn.py", "run_pipeline.py"))]
    identity = dict(protocol="independent_load_stress_v1", settings=settings, windows=windows,
                    training_record_sha256=sha256(record_path), checkpoints=checkpoints,
                    runtime=runtime,
                    code_sha256={p.relative_to(root).as_posix(): sha256(p) for p in paths})
    identity = json.loads(json.dumps(identity, allow_nan=False))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    state_path = output / "run.json"
    if state_path.exists() and json.loads(state_path.read_text())["identity"] != identity:
        raise ValueError("Inputs/code/runtime changed; use a new RUN_NAME")
    write_json(state_path, dict(status="running", identity=identity,
                               interpretation="Exploratory reused evaluation windows, not an untouched final holdout"))
    config = record["config"]
    hardware, profile = HardwareConfig(**config["hardware"]), ServiceProfile(**record["service_profile"])
    summaries, timelines, lengths, learning = [], [], [], []
    for window in windows:
        rows = rows_by_window[window["name"]]
        span = window["end_timestamp_s"] - window["start_timestamp_s"]
        for multiplier in levels:
            replay = replay_window(rows, slo_s=config["slo_s"], target_rps=len(rows)/span*multiplier,
                                   bounds=(window["start_timestamp_s"], window["end_timestamp_s"]))
            for policy in POLICIES:
                for seed in (seeds if policy in ("frozen", "adapting") else [None]):
                    name = policy if seed is None else f"{policy}_seed{seed}"
                    label = dict(group=window["group"], window=window["name"], multiplier=multiplier,
                                 policy=policy, seed=seed)
                    folder = output / window["name"] / f"load_{multiplier:g}" / name
                    folder.mkdir(parents=True, exist_ok=True)
                    marker = folder / "case.json"
                    saved = json.loads(marker.read_text()) if marker.exists() else {}
                    if saved and saved.get("identity_sha256") != digest:
                        raise ValueError("Cached case inputs differ; use a new RUN_NAME")
                    if saved.get("status") == "completed":
                        if any(not (folder / f).is_file() or sha256(folder / f) != h for f, h in saved["files"].items()):
                            raise ValueError("Cached outputs changed; use a new RUN_NAME")
                        print("Cached:", window["name"], multiplier, name, flush=True)
                    else:
                        print("Running:", window["name"], multiplier, name, len(rows), "requests", flush=True)
                        write_json(marker, dict(status="running", identity_sha256=digest))
                        began = time.monotonic()
                        if seed is None:
                            simulation = simulate(replay.requests, baseline_schedulers()[policy], hardware, profile,
                                                  estimated_output_tokens=record["estimated_output_tokens"],
                                                  **config["decisions"])
                            stats = dict(decisions=None, updates=0, max_weight_change=0.0)
                        else:
                            simulation, stats = replay_dqn(training_dir / f"dqn_seed{seed}.npz", record, replay,
                                                           folder, policy, device=resolved_device,
                                                           **{k: settings[k] for k in ("epsilon", "learning_rate", "batch_size",
                                                                                      "warmup", "update_interval", "capacity")}, seed=seed)
                        metrics, diagnostics, timeline = diagnose_replay(
                            simulation, replay, config["slo_s"], settings["timeline_source_bin_s"] / multiplier,
                            settings["ttft_target_s"])
                        records = evaluate_events(simulation.events, measurement_window_s=replay.horizon_s).request_records
                        if {r.request_id for r in records} != {str(row.source_row) for row in rows}:
                            raise RuntimeError("Unpaired cohort")
                        for entry in timeline:
                            a, b = entry["sim_start_s"], entry["sim_end_s"]
                            cohort = [r for r in records if a <= r.arrival_time_s < b]
                            entry.update(summarize_records(cohort, measurement_window_s=b-a, window_start_s=a))
                            entry.update(_ttft_score(cohort, settings["ttft_target_s"]))
                        by_id = {r.request_id: r for r in records}
                        grouped = []
                        for dimension in ("input_length_group", "output_length_group"):
                            for group in ("short", "medium", "long"):
                                cohort = [by_id[d["request_id"]] for d in diagnostics if d[dimension] == group]
                                if cohort:
                                    grouped.append(dict(dimension=dimension, length_group=group,
                                                        **summarize_records(cohort, measurement_window_s=replay.horizon_s),
                                                        **_ttft_score(cohort, settings["ttft_target_s"])))
                        write_json(folder / "summary.json", dict(label, offered_rps=len(rows)/replay.horizon_s, **metrics))
                        write_json(folder / "learning.json", dict(label, **{k: v for k, v in stats.items() if k != "policy"},
                                                                   case_wall_seconds=time.monotonic()-began))
                        for filename, table in (("timeline.json", timeline), ("length_groups.json", grouped)):
                            write_json(folder / filename, [dict(label, **row) for row in table])
                        write_csv(folder / "request_diagnostics.csv", diagnostics)
                        files = [p for p in folder.iterdir() if p.is_file() and p.name != "case.json" and not p.name.endswith(".tmp")]
                        write_json(marker, dict(status="completed", identity_sha256=digest,
                                                files={p.name: sha256(p) for p in files}))
                    summaries.append(json.loads((folder / "summary.json").read_text()))
                    learning.append(json.loads((folder / "learning.json").read_text()))
                    timelines.extend(json.loads((folder / "timeline.json").read_text()))
                    lengths.extend(json.loads((folder / "length_groups.json").read_text()))
    for filename, table in (("summary.csv", summaries), ("timeline.csv", timelines),
                            ("length_groups.csv", lengths), ("learning_summary.csv", learning)):
        # Baseline and learner audit fields differ; publish a rectangular table.
        fields = list(dict.fromkeys(k for row in table for k in row))
        write_csv(output / filename, [{k: row.get(k) for k in fields} for row in table])
    if any(sha256(training_dir / f"dqn_seed{s}.npz") != h for s, h in checkpoints.items()):
        raise RuntimeError("Source checkpoint changed")
    write_json(state_path, dict(status="completed", identity=identity,
                               files={f: sha256(output / f) for f in ("summary.csv", "timeline.csv", "length_groups.csv", "learning_summary.csv")}))
    return summaries
