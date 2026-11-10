"""Reproduce the frozen BurstGPT holdout comparison."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import fmean

import gymnasium
import numpy as np

from experiments.train_dqn import evaluate, run as train, write_csv, write_json
from llm_triage.rl.dqn import DQN
from llm_triage.simulation import HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import load_windows, prepare, sha256
from llm_triage.workload.transforms import replay_window


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/week3_protocol.json"
TRAINING_RECORD = ROOT / "results/week2/run.json"


def compress_middle_third(replay, multiplier):
    """Compress time in the middle third and translate the final third."""
    if multiplier == 1:
        return replay
    start = replay.horizon_s / 3
    end = 2 * start

    def shift(time_s):
        if time_s <= start:
            return time_s
        if time_s <= end:
            return start + (time_s - start) / multiplier
        return start + (end - start) / multiplier + (time_s - end)

    requests = []
    for request in replay.requests:
        arrival = shift(request.arrival_time_s)
        deadline = arrival + (request.deadline_s - request.arrival_time_s)
        requests.append(replace(request, arrival_time=arrival, slo_deadline=deadline))
    return replace(replay, requests=tuple(requests), horizon_s=shift(replay.horizon_s))


def verify_checkpoints(directory, checkpoints):
    paths = {}
    for checkpoint in checkpoints:
        seed = checkpoint["seed"]
        path = directory / f"dqn_seed{seed}.npz"
        if not path.is_file() or sha256(path) != checkpoint["sha256"]:
            raise ValueError(
                f"Checkpoint for seed {seed} differs from Week 2. "
                "Use the recorded runtime or pass a directory with the frozen checkpoints."
            )
        paths[f"dqn_seed{seed}"] = path
    return paths


def summarize(rows):
    groups = {}
    fcfs = {}
    for row in rows:
        key = row["scenario"], row["source_rps"]
        groups.setdefault((*key, row["policy"]), []).append(row)
        if row["policy"] == "fcfs":
            fcfs[(*key, row["window"])] = row["slo_attainment"]
    summaries = []
    for (scenario, rate, policy), group in groups.items():
        deltas = [
            row["slo_attainment"] - fcfs[(scenario, rate, row["window"])]
            for row in group
        ]
        ttfts = [row["p95_ttft_s"] for row in group if row["p95_ttft_s"] is not None]
        summaries.append(dict(
            scenario=scenario, source_rps=rate, policy=policy, windows=len(group),
            requests=sum(row["arrivals"] for row in group),
            mean_slo_attainment=fmean(row["slo_attainment"] for row in group),
            mean_drop_rate=fmean(row["drop_rate"] for row in group),
            mean_unfinished_rate=fmean(row["unfinished_rate"] for row in group),
            mean_goodput_rps=fmean(row["window_goodput_rps"] for row in group),
            mean_window_p95_ttft_s=fmean(ttfts) if ttfts else None,
            windows_with_ttft=len(ttfts),
            delta_vs_fcfs_pp=100 * fmean(deltas),
            windows_above_fcfs=sum(delta > 0 for delta in deltas),
            windows_tied_fcfs=sum(delta == 0 for delta in deltas),
            windows_below_fcfs=sum(delta < 0 for delta in deltas),
        ))
    return summaries


def run(output, data_dir, source=None, checkpoints_dir=None):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if sha256(TRAINING_RECORD) != protocol["training_record_sha256"]:
        raise ValueError("Week 2 training record differs from the frozen protocol")
    training = json.loads(TRAINING_RECORD.read_text(encoding="utf-8"))
    if training["selection_split"] != "validation" or training["test_evaluated"]:
        raise ValueError("Frozen checkpoints must come from validation-only selection")
    if (gymnasium.__version__, np.__version__) != (training["gymnasium"], training["numpy"]):
        raise ValueError(
            f"Install gymnasium=={training['gymnasium']} and numpy=={training['numpy']}"
        )

    config = training["config"]
    profile = ServiceProfile(**training["service_profile"])
    hardware = HardwareConfig(**config["hardware"])
    processed = data_dir / "processed"
    trace_path = processed / "burstgpt_windows.csv"
    manifest_path = processed / "manifest.json"
    if source is not None or not (trace_path.exists() or manifest_path.exists()):
        prepare(data_dir=data_dir, source=source)
    if (not trace_path.is_file() or not manifest_path.is_file()
            or sha256(trace_path) != training["trace_sha256"]
            or sha256(manifest_path) != training["manifest_sha256"]):
        raise ValueError("Processed data differs from the frozen BurstGPT trace")

    output.mkdir(parents=True, exist_ok=False)
    if checkpoints_dir is None:
        checkpoints_dir = output / "training"
        train(config, processed, checkpoints_dir)
    paths = verify_checkpoints(checkpoints_dir, training["checkpoints"])
    models = {name: DQN.load(path)[0] for name, path in paths.items()}

    # Build holdout replays only after every checkpoint has been verified.
    windows = [(window, rows) for (split, window), rows in sorted(load_windows(trace_path).items())
               if split == protocol["split"]]
    if (len(windows) != protocol["windows"]
            or any(len(rows) != protocol["requests_per_window"] for _, rows in windows)):
        raise ValueError("Unexpected frozen test window layout")
    results = []
    for scenario, multiplier in protocol["scenarios"].items():
        for rate in protocol["source_rates_rps"]:
            cases = [
                (window, rate, compress_middle_third(
                    replay_window(rows, slo_s=config["slo_s"], target_rps=rate), multiplier
                ))
                for window, rows in windows
            ]
            runs = evaluate(
                models, cases, hardware, profile, training["estimated_output_tokens"],
                config["decisions"], baselines=True,
            )
            for row in runs:
                row.pop("target_rps")
                results.append(dict(scenario=scenario, source_rps=rate, **row))
            print(f"{scenario} at {rate:g} source RPS: {len(runs)} policy runs", flush=True)

    write_csv(output / "test_runs.csv", results)
    write_csv(output / "summary.csv", summarize(results))
    write_json(output / "run.json", dict(
        protocol=protocol, protocol_sha256=sha256(PROTOCOL),
        trace_sha256=training["trace_sha256"], manifest_sha256=training["manifest_sha256"],
        checkpoints=training["checkpoints"], training_config=config,
        gymnasium=training["gymnasium"], numpy=training["numpy"],
        policy_runs=len(results), unique_test_requests=sum(len(rows) for _, rows in windows),
        request_executions=sum(row["arrivals"] for row in results),
    ))
    print(f"Holdout results: {output / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", type=Path, help="Optional local copy of the pinned raw CSV")
    parser.add_argument("--checkpoints-dir", type=Path, help="Reuse all three exact Week 2 checkpoints")
    parser.add_argument("--output", type=Path, default=Path("results/runs/week3"))
    args = parser.parse_args()
    run(args.output, args.data_dir, args.source, args.checkpoints_dir)


if __name__ == "__main__":
    main()
