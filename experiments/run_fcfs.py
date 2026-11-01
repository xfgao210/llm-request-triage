"""Week 1 FCFS integration against the existing single-worker reference simulator.

This adapter is a temporary validation consumer, not Vikas's shared kernel or
Xiaofei's evaluation engine. It reuses existing metrics; it does not claim TTFT,
KV-cache modeling, continuous batching, or measured GPU execution.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from contributions.hengyi import triage_sim as reference
from llm_triage.scheduling.fcfs import FCFSScheduler
from llm_triage.types import CandidateView, ResourceView, SchedulerSnapshot
from llm_triage.workload.burstgpt import load_windows, sha256
from llm_triage.workload.transforms import replay_window


def run_reference(replay, *, prefill_rate=2000.0, decode_rate=400.0, estimated_output_tokens=144.0):
    """Only the environment adapter reads true_output_tokens; FCFS never does."""
    requests = replay.requests
    backend_requests = [reference.Request(index, r.arrival_time_s, r.input_tokens,
                        r.true_output_tokens, r.deadline_s) for index, r in enumerate(requests)]
    scheduler = FCFSScheduler()

    def callback(observation):
        request = requests[observation.request_id]
        candidate = CandidateView(request.request_id, request.arrival_time_s, observation.input_tokens,
                                  observation.estimated_output_tokens, observation.estimated_service_s,
                                  observation.now + observation.slack)
        snapshot = SchedulerSnapshot(observation.now, (candidate,),
                                     ResourceView(active_batch_size=int(observation.server_busy),
                                                  admission_available=not observation.server_busy))
        return reference.Action(scheduler.decide(snapshot).action.value)

    records, trace, summary = reference.simulate(
        backend_requests, callback, horizon=replay.horizon_s, prefill_rate=prefill_rate,
        decode_rate=decode_rate, estimated_output_tokens=estimated_output_tokens)
    for row in records:
        row["request_id"] = requests[row["request_id"]].request_id
    return records, trace, summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_suite(config, processed_dir, output_dir, summary_dir):
    processed_dir, output_dir, summary_dir = map(Path, (processed_dir, output_dir, summary_dir))
    if config["split"] not in {"train", "validation", "test"}:
        raise ValueError("Unknown workload split")
    rates = config["target_rps"]
    if not rates or any(rate is not None and (not math.isfinite(rate) or rate <= 0) for rate in rates):
        raise ValueError("Need finite positive rates, or null for native timing")
    for field in ("slo_s", "prefill_tokens_per_s", "decode_tokens_per_s"):
        if not math.isfinite(config[field]) or config[field] <= 0:
            raise ValueError(f"{field} must be positive and finite")
    manifest_path = processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    window_path = processed_dir / "burstgpt_windows.csv"
    if sha256(window_path) != manifest["selected_sha256"]:
        raise ValueError("Prepared-window digest differs from manifest; regenerate the data")
    estimate = manifest["estimated_output_tokens_train_mean"]
    windows = load_windows(window_path)
    selected = [(key, rows) for key, rows in sorted(windows.items()) if key[0] == config["split"]]
    if not selected:
        raise ValueError("Selected split has no windows")
    summaries = []
    for (split, index), rows in selected:
        for rate in rates:
            replay = replay_window(rows, slo_s=config["slo_s"], target_rps=rate)
            records, trace, metrics = run_reference(
                replay, prefill_rate=config["prefill_tokens_per_s"],
                decode_rate=config["decode_tokens_per_s"], estimated_output_tokens=estimate)
            if metrics["completed"] != len(rows) or metrics["dropped"] or metrics["deferred"]:
                raise RuntimeError("FCFS accounting failed: all arrivals must complete after drain")
            rate_label = "native" if rate is None else f"rps{rate:g}"
            label = f"{split}_{index:02d}_{rate_label}"
            write_csv(output_dir / label / "requests.csv", records)
            write_csv(output_dir / label / "queue.csv", trace)
            summaries.append(dict(policy="fcfs", split=split, window=index, target_rps=rate,
                                  source_origin_s=replay.source_origin_s, time_scale=replay.time_scale,
                                  **metrics))
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "summary.csv", summaries)
    code_paths = [ROOT / "experiments/run_fcfs.py", ROOT / "contributions/hengyi/triage_sim.py",
                  *sorted((ROOT / "src/llm_triage").rglob("*.py"))]
    provenance = dict(
        resolved_config=config, backend="contributions/hengyi/triage_sim.py (single serial worker)",
        timing="assumed linear token rates; CPU simulation, not GPU execution",
        metric_semantics=dict(slo="completion <= deadline; all arrivals in denominator",
                              throughput="completions <= horizon / horizon; drain reported separately",
                              latency="completed-request latency including queue wait",
                              ttft="not emitted by the reference simulator"),
        data_source={key: manifest[key] for key in ("source_url", "revision", "source_sha256", "selected_sha256")},
        manifest_sha256=sha256(manifest_path), code_sha256={
            path.relative_to(ROOT).as_posix(): sha256(path) for path in code_paths},
        windows_evaluated=len(selected), policy_runs=len(summaries),
        estimated_output_tokens_train_mean=estimate)
    (summary_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    (summary_dir / "data_manifest.json").write_bytes(manifest_path.read_bytes())
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fcfs.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/runs/week1-fcfs"))
    parser.add_argument("--summary-dir", type=Path, default=Path("results/week1"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summaries = run_suite(config, args.processed_dir, args.output, args.summary_dir)
    print(json.dumps(dict(policy_runs=len(summaries), requests=sum(r["arrivals"] for r in summaries),
                          summary=str(args.summary_dir / "summary.csv")), indent=2))


if __name__ == "__main__":
    main()
