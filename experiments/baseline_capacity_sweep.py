"""Short, development-only capacity sweeps; no training or held-out replay.

The Colab notebook calls these functions so simulation and metric definitions
are shared with the rest of the project, rather than copied into a notebook.
"""

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import time

from llm_triage.metrics import evaluate_events
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import HardwareConfig, ServiceProfile, simulate
from llm_triage.workload.burstgpt import TraceRow, sha256
from llm_triage.workload.transforms import replay_window


def write_json(path, value):
    """Publish complete metadata, never a partially written completion marker."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Refusing to publish an empty results table")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_development_windows(manifest, *, split="validation", seconds=900):
    """Choose disjoint typical/bursty blocks by traffic only, before simulation.

    Blocks align to the explicit range start, not the analyzer's older automatic
    split labels. Idle sub-bins are retained. The burst score is max sub-bin RPS;
    typical is nearest median positive block count among the remaining blocks.
    """
    if split not in ("train", "validation"):
        raise ValueError("Capacity tuning must use train or validation, never test")
    bin_s = manifest["range_config"]["episode_seconds"]
    if not math.isfinite(seconds) or seconds < bin_s or seconds % bin_s:
        raise ValueError("Window seconds must be a positive multiple of episode_seconds")
    bins = manifest["windows"] + manifest.get("empty_windows", [])
    candidates = []
    for period in manifest["range_config"]["ranges"]:
        if period["split"] != split:
            continue
        origin, end = period["start_timestamp_s"], period["end_timestamp_s"]
        for index in range(int((end - origin) // seconds)):
            start = origin + index * seconds
            selected = [b for b in bins if b["split"] == split
                        and start <= b["start_timestamp_s"] < start + seconds]
            if sum(b["end_timestamp_s"] - b["start_timestamp_s"] for b in selected) != seconds:
                raise ValueError("Manifest does not cover complete candidate blocks")
            count = sum(b["requests"] for b in selected)
            if count:
                candidates.append(dict(
                    split=split, start_timestamp_s=start, end_timestamp_s=start + seconds,
                    requests=count, native_mean_rps=count / seconds,
                    peak_subbin_rps=max(b["requests"] / (b["end_timestamp_s"] - b["start_timestamp_s"])
                                        for b in selected),
                    input_tokens=sum(b["input_tokens"] for b in selected),
                    output_tokens=sum(b["output_tokens"] for b in selected),
                ))
    if len(candidates) < 2:
        raise ValueError("Need two nonempty development blocks; shorten WINDOW_SECONDS")
    burst = max(candidates, key=lambda b: (b["peak_subbin_rps"], b["requests"], -b["start_timestamp_s"]))
    remaining = [c for c in candidates if c is not burst]
    median = statistics.median(c["requests"] for c in remaining)
    typical = min(remaining, key=lambda b: (abs(b["requests"] - median), b["start_timestamp_s"]))
    return [dict(typical, name="typical_development"), dict(burst, name="burst_development")]


def validate_windows(manifest, windows):
    if not windows:
        raise ValueError("Select at least one development window")
    names = set()
    for window in windows:
        name = window["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError("Window names must be unique safe folder names")
        names.add(name)
        start, end = window["start_timestamp_s"], window["end_timestamp_s"]
        if not all(math.isfinite(t) for t in (start, end)) or end <= start:
            raise ValueError("Invalid window timestamps")
        if not any(r["split"] in ("train", "validation")
                   and r["split"] == window["split"]
                   and r["start_timestamp_s"] <= start < end <= r["end_timestamp_s"]
                   for r in manifest["range_config"]["ranges"]):
            raise ValueError("Windows must lie entirely within one train/validation range")
    ordered = sorted(windows, key=lambda w: w["start_timestamp_s"])
    if any(a["end_timestamp_s"] > b["start_timestamp_s"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Development windows must not overlap")


def load_selected_rows(trace_path, windows, *, allowed_splits=("train", "validation")):
    """Stream prepared data; do not load the full test trace into memory."""
    selected = {w["name"]: [] for w in windows}
    with Path(trace_path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in allowed_splits:
                continue
            timestamp = float(row["timestamp_s"])
            for w in windows:
                if row["split"] == w["split"] and w["start_timestamp_s"] <= timestamp < w["end_timestamp_s"]:
                    selected[w["name"]].append(TraceRow(int(row["source_row"]), timestamp,
                                                       int(row["input_tokens"]), int(row["true_output_tokens"])))
    if any(not rows for rows in selected.values()):
        raise ValueError("A selected window has no requests")
    return selected


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def diagnose_replay(result, replay, slo_s, timeline_bin_s, ttft_target_s=None):
    evaluation = evaluate_events(result.events, measurement_window_s=replay.horizon_s)
    records = evaluation.request_records
    requests = result.by_id()
    diagnostics, waits = [], []
    for record in records:
        request = requests[record.request_id]
        # Ground-truth output length is used ONLY AFTER simulation, offline.
        # Closed form equals batch=1 isolated_service_time for integer outputs.
        alone = result.profile.estimate_service_time(request.input_tokens, request.output_tokens)
        wait = None if record.service_start_time_s is None else record.service_start_time_s - record.arrival_time_s
        if wait is not None:
            waits.append(wait)
        diagnostics.append(dict(
            request_id=str(record.request_id), arrival_time_s=record.arrival_time_s,
            slo_met=bool(record.slo_met), status=record.status.value,
            ttft_s=record.ttft_s, input_tokens=request.input_tokens,
            # Offline grouping only; never supplied to a policy.
            output_length_group=("short" if request.output_tokens <= 32 else
                                 "medium" if request.output_tokens <= 128 else "long"),
            input_length_group=("short" if request.input_tokens <= 512 else
                                "medium" if request.input_tokens <= 2048 else "long"),
            queue_wait_s=wait, completion_latency_s=record.completion_latency_s,
            isolated_service_s=alone, isolated_exceeds_slo=alone > slo_s,
        ))
    horizon = replay.horizon_s
    timeline = [dict(sim_start_s=i * timeline_bin_s, sim_end_s=min((i + 1) * timeline_bin_s, horizon),
                     busy_s=0.0, queue_area=0.0, batch_area=0.0, kv_area=0.0, queue_max=0)
                for i in range(math.ceil(horizon / timeline_bin_s))]
    # Queue/batch are iteration snapshots; KV is post-iteration occupancy.
    # Duration-weighting is a diagnostic approximation, not a cycle-accurate counter.
    for iteration in result.iterations:
        start, end = max(0.0, iteration.start), min(horizon, iteration.end)
        if end <= start:
            continue
        first, last = int(start // timeline_bin_s), min(len(timeline) - 1, int(end // timeline_bin_s))
        for index in range(first, last + 1):
            row = timeline[index]
            duration = max(0.0, min(end, row["sim_end_s"]) - max(start, row["sim_start_s"]))
            row["busy_s"] += duration * (iteration.kind != "idle")
            row["queue_area"] += duration * iteration.queue_length
            row["batch_area"] += duration * iteration.batch_size
            row["kv_area"] += duration * iteration.kv_blocks_used
            if duration:
                row["queue_max"] = max(row["queue_max"], iteration.queue_length)
    queue_area = sum(r["queue_area"] for r in timeline)
    batch_area = sum(r["batch_area"] for r in timeline)
    kv_area = sum(r["kv_area"] for r in timeline)
    busy = sum(r["busy_s"] for r in timeline)
    for row in timeline:
        duration = row["sim_end_s"] - row["sim_start_s"]
        row["source_start_s"] = replay.source_origin_s + row["sim_start_s"] / replay.time_scale
        row["source_end_s"] = replay.source_origin_s + row["sim_end_s"] / replay.time_scale
        row["busy_fraction"] = row.pop("busy_s") / duration
        row["mean_queue"] = row.pop("queue_area") / duration
        row["batch_slot_utilization"] = row.pop("batch_area") / duration / result.hardware.max_batch_size
        row["kv_utilization"] = row.pop("kv_area") / duration / result.hardware.kv_total_blocks
    n = len(diagnostics)
    extra = dict(
        first_token_coverage=sum(d["ttft_s"] is not None for d in diagnostics) / n,
        p95_queue_wait_s=_percentile(waits, .95), queue_wait_observed=len(waits),
        isolated_exceeds_slo=sum(d["isolated_exceeds_slo"] for d in diagnostics),
        isolated_exceeds_slo_fraction=sum(d["isolated_exceeds_slo"] for d in diagnostics) / n,
        missed_but_isolated_within_slo=sum(not d["slo_met"] and not d["isolated_exceeds_slo"] for d in diagnostics),
        mean_queue_in_window=queue_area / horizon,
        max_queue_in_window=max(r["queue_max"] for r in timeline),
        busy_fraction_in_window=busy / horizon,
        batch_slot_utilization_in_window=batch_area / horizon / result.hardware.max_batch_size,
        kv_utilization_in_window=kv_area / horizon / result.hardware.kv_total_blocks,
        drain_s=max(0.0, result.total_time - horizon), preemptions=result.preemptions,
        full_replay_busy_gpu_s=result.gpu_busy_time,
        full_replay_wasted_gpu_s=result.wasted_gpu_time,
        full_replay_wasted_gpu_fraction=(result.wasted_gpu_time / result.gpu_busy_time if result.gpu_busy_time else 0.0),
    )
    if ttft_target_s is not None:
        if not math.isfinite(ttft_target_s) or ttft_target_s <= 0:
            raise ValueError("TTFT target must be positive and finite")
        extra.update(ttft_target_s=ttft_target_s,
                     ttft_target_met=sum(d["ttft_s"] is not None and d["ttft_s"] <= ttft_target_s for d in diagnostics),
                     ttft_target_attainment=sum(d["ttft_s"] is not None and d["ttft_s"] <= ttft_target_s for d in diagnostics) / n)
    return dict(evaluation.summary, **extra), diagnostics, timeline


def run_sweep(data_dir, destination, windows, *, multipliers=(1, 2, 4, 8), slo_s=5.0,
              hardware=None, profile=None, timeline_bin_s=10.0, max_requests=30000):
    """Drain each short replay fully; resume only input/code-identical cases."""
    data_dir, destination = Path(data_dir), Path(destination)
    manifest_path = data_dir / "processed/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    trace_path = data_dir / "processed/burstgpt_windows.csv"
    if sha256(trace_path) != manifest["selected_sha256"]:
        raise ValueError("Prepared trace no longer matches its manifest")
    validate_windows(manifest, windows)
    multipliers = list(multipliers)
    if not multipliers or any(isinstance(m, bool) or not math.isfinite(m) or m <= 0 for m in multipliers):
        raise ValueError("Load multipliers must be positive finite numbers")
    if len(set(multipliers)) != len(multipliers):
        raise ValueError("Duplicate multipliers")
    if len({f"{m:g}" for m in multipliers}) != len(multipliers):
        raise ValueError("Load multiplier folder names collide; use more distinct values")
    if not math.isfinite(slo_s) or slo_s <= 0 or not math.isfinite(timeline_bin_s) or timeline_bin_s <= 0:
        raise ValueError("SLO and timeline bin must be positive finite numbers")
    hardware, profile = hardware or HardwareConfig(), profile or ServiceProfile()
    selected = load_selected_rows(trace_path, windows)
    if any(len(rows) > max_requests for rows in selected.values()):
        raise ValueError("Window exceeds max_requests safety limit; shorten it before running")
    estimate = manifest["estimated_output_tokens_train_mean"]
    root = Path(__file__).resolve().parents[1]
    code_paths = [Path(__file__)] + [p for directory in ("simulation", "scheduling", "metrics", "workload")
                                  for p in (root / "src/llm_triage" / directory).rglob("*.py")]
    identity = dict(protocol="development_baseline_capacity_v1", source_sha256=manifest["source_sha256"],
                    manifest_sha256=sha256(manifest_path), trace_sha256=manifest["selected_sha256"],
                    windows=windows, multipliers=multipliers, slo_s=slo_s, hardware=asdict(hardware),
                    profile=asdict(profile), timeline_bin_s=timeline_bin_s,
                    estimated_output_tokens_train_mean=estimate, python=platform.python_version(),
                    code_sha256={p.relative_to(root).as_posix(): sha256(p) for p in sorted(code_paths)})
    # Normalize caller tuples and integer/float JSON representations for resume.
    identity = json.loads(json.dumps(identity, allow_nan=False))
    identity_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    record_path = destination / "sweep.json"
    if record_path.exists() and json.loads(record_path.read_text())["identity"] != identity:
        raise ValueError("Inputs/code changed; choose a NEW RUN_NAME/output folder")
    write_json(record_path, dict(status="running", identity=identity))
    summaries = []
    for window in windows:
        rows = selected[window["name"]]
        span = window["end_timestamp_s"] - window["start_timestamp_s"]
        for multiplier in multipliers:
            replay = replay_window(rows, slo_s=slo_s, target_rps=len(rows) / span * multiplier,
                                   bounds=(window["start_timestamp_s"], window["end_timestamp_s"]))
            for policy, scheduler in baseline_schedulers().items():
                label = f"{window['name']} / {multiplier:g}x / {policy}"
                folder = destination / window["name"] / f"load_{multiplier:g}" / policy
                folder.mkdir(parents=True, exist_ok=True)
                case_path = folder / "case.json"
                previous = json.loads(case_path.read_text()) if case_path.exists() else {}
                if previous and previous.get("identity_sha256") != identity_digest:
                    raise ValueError("Cached case belongs to different inputs/code; use a fresh output folder")
                if previous.get("status") == "completed":
                    if any(sha256(folder / name) != digest for name, digest in previous["files"].items()):
                        raise ValueError("Cached output changed; use a fresh output folder")
                    summary = json.loads((folder / "summary.json").read_text())
                    print("Cached:", label, flush=True)
                else:
                    write_json(case_path, dict(status="running", identity_sha256=identity_digest))
                    print("Running:", label, f"({len(rows):,} requests)", flush=True)
                    began = time.perf_counter()
                    result = simulate(replay.requests, scheduler, hardware, profile,
                                      estimated_output_tokens=estimate, record_iterations=True,
                                      expire_overdue_in_queue=False)
                    metrics, diagnostics, timeline = diagnose_replay(result, replay, slo_s, timeline_bin_s)
                    summary = dict(window=window["name"], split=window["split"], multiplier=multiplier,
                                   policy=policy, native_mean_rps=len(rows) / span,
                                   offered_rps=len(rows) / replay.horizon_s,
                                   wall_seconds=time.perf_counter() - began, **metrics)
                    write_json(folder / "summary.json", summary)
                    write_csv(folder / "request_diagnostics.csv", diagnostics)
                    write_csv(folder / "timeline.csv", timeline)
                    write_json(case_path, dict(status="completed", identity_sha256=identity_digest, files={
                        name: sha256(folder / name) for name in ("summary.json", "request_diagnostics.csv", "timeline.csv")}))
                    print(f"  SLO={metrics['slo_attainment']:.2%}; drops={metrics['drop_rate']:.2%}; "
                          f"elapsed={summary['wall_seconds']:.1f}s", flush=True)
                summaries.append(summary)
    write_csv(destination / "summary.csv", summaries)
    write_json(record_path, dict(status="completed", identity=identity,
                                summary_sha256=sha256(destination / "summary.csv")))
    return summaries
