"""Run paired BurstGPT baselines through the shared simulator and metrics."""
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from llm_triage.evaluation import EvaluationConfig, evaluate_policy_matrix, write_evaluation_artifacts
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import HardwareConfig, ServiceProfile, SimulatorExecutor
from llm_triage.workload.burstgpt import load_windows, sha256
from llm_triage.workload.transforms import replay_window


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8", newline="\n")


def _write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(config, processed_dir, output_dir, summary_dir):
    """Use validation by default; evaluation of another split must be explicit.

    Scheduling overhead is actual wall-clock measurement, so it is kept in a
    separate table. The primary simulation summary is deterministic.
    """
    processed_dir, output_dir, summary_dir = map(Path, (processed_dir, output_dir, summary_dir))
    if config["split"] not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    rates = config["target_rps"]
    if not rates or len(set(rates)) != len(rates) or any(
        rate is not None and (not math.isfinite(rate) or rate <= 0) for rate in rates
    ):
        raise ValueError("Need unique positive finite rates, or null for native timing")
    if not math.isfinite(config["slo_s"]) or config["slo_s"] <= 0:
        raise ValueError("SLO must be positive and finite")
    manifest_path = processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    window_path = processed_dir / "burstgpt_windows.csv"
    if sha256(window_path) != manifest["selected_sha256"]:
        raise ValueError("Prepared windows differ from manifest")
    estimate = manifest["estimated_output_tokens_train_mean"]
    windows = [(key, rows) for key, rows in sorted(load_windows(window_path).items()) if key[0] == config["split"]]
    if not windows:
        raise ValueError("No windows in selected split")
    hardware = HardwareConfig(**config["hardware"])
    profile = ServiceProfile()
    summaries, overheads = [], []
    all_requests = [r for _, rows in windows for r in rows]
    for (split, index), rows in windows:
        for rate in rates:
            replay = replay_window(rows, slo_s=config["slo_s"], target_rps=rate)
            executor = SimulatorExecutor(
                replay.requests, hardware, profile,
                estimated_output_tokens=estimate,
                defer_retry_s=config["defer_retry_s"], max_defers=config["max_defers"],
                expire_overdue_in_queue=config["expire_overdue_in_queue"],
            )
            results = evaluate_policy_matrix(
                baseline_schedulers(), executor,
                EvaluationConfig(measurement_window_s=replay.horizon_s),
            )
            label = f"{split}_{index:02d}_" + ("native" if rate is None else f"rps{rate:g}")
            cohort_requests = [
                {
                    "request_id": str(request.request_id),
                    "arrival_time_s": request.arrival_time_s,
                    "input_tokens": request.input_tokens,
                    "true_output_tokens": request.true_output_tokens,
                    "deadline_s": request.deadline_s,
                }
                for request in replay.requests
            ]
            cohort = hashlib.sha256(
                json.dumps(cohort_requests, sort_keys=True).encode()
            ).hexdigest()
            write_evaluation_artifacts(results, output_dir / label,
                resolved_config=dict(**config, window=index, horizon_s=replay.horizon_s,
                                     time_scale=replay.time_scale, cohort_sha256=cohort,
                                     service_profile=asdict(profile), estimated_output_tokens=estimate))
            for policy, result in results.items():
                summary = dict(result.summary)
                backend = executor.results[policy]
                if summary["arrivals"] != len(rows) or summary["arrivals"] != sum(
                    summary[field] for field in ("completed", "dropped", "unfinished")
                ):
                    raise RuntimeError("Request accounting mismatch")
                if not math.isclose(summary["slo_attainment"], backend.slo_attainment, abs_tol=1e-12):
                    raise RuntimeError("Backend and metrics disagree on completion SLO attainment")
                diagnostics = {key: summary.pop(key) for key in list(summary) if "overhead" in key}
                identity = dict(policy=policy, split=split, window=index, target_rps=rate)
                overheads.append(dict(**identity, **diagnostics))
                summaries.append(dict(**identity, source_origin_s=replay.source_origin_s,
                                      time_scale=replay.time_scale, cohort_sha256=cohort,
                                      **summary, preemptions=backend.preemptions,
                                      total_gpu_time_s=sum(r.gpu_time_s for r in backend.requests),
                                      wasted_gpu_time_s=backend.wasted_gpu_time))
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "summary.csv", summaries)
    _write_csv(summary_dir / "scheduler_overhead.csv", overheads)
    (summary_dir / "data_manifest.json").write_bytes(manifest_path.read_bytes())
    paths = [ROOT / "experiments/run_burstgpt.py",
             *sorted((ROOT / "src/llm_triage").rglob("*.py"))]
    write_json(summary_dir / "provenance.json", dict(
        config=config, service_profile=asdict(profile), hardware=asdict(hardware),
        estimated_output_tokens_train_mean=estimate,
        manifest_sha256=sha256(manifest_path), window_sha256=sha256(window_path),
        code_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in paths},
        policy_runs=len(summaries), unique_source_requests=len(all_requests),
        request_executions=sum(row["arrivals"] for row in summaries),
        measured_domain_fraction=sum(32 <= r.input_tokens <= 2048 and 1 <= r.true_output_tokens <= 128
                                     for r in all_requests) / len(all_requests),
        limitations=["Simulation, not real GPU inference",
                     "Batch=1 timing calibration; batch contention is assumed",
                     "Selected validation windows, not independent random seeds",
                     "Wall-clock scheduler overhead varies between repetitions",
                     "Service residency time in metrics is not attributed GPU compute; see GPU diagnostics"],
    ))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/burstgpt_baselines.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/runs/burstgpt-integrated"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/week1-integrated"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = run_suite(config, args.processed_dir, args.output, args.summary_dir)
    print(json.dumps(dict(policy_runs=len(rows), request_executions=sum(r["arrivals"] for r in rows),
                          summary=str(args.summary_dir / "summary.csv")), indent=2))


if __name__ == "__main__":
    main()
