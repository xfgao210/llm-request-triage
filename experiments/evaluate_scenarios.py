"""Evaluate a frozen pipeline checkpoint on prepared validation or test scenarios."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from experiments.train_dqn import evaluate, load_cohorts, write_csv, write_json
from llm_triage.rl.dqn import DQN
from llm_triage.simulation import HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import sha256


def run(training_dir, data_dir, output, *, split="validation", target_rps=None, groups=None):
    if split not in {"validation", "test"}:
        raise ValueError("Evaluate validation or test scenarios only")
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((training_dir / "run.json").read_text(encoding="utf-8"))
    if record["selection_split"] != "validation" or record["test_evaluated"]:
        raise ValueError("Require checkpoints selected on validation only")
    config = dict(record["config"])
    if target_rps is not None:
        if (not target_rps or any(rate is not None and (
            isinstance(rate, bool) or not isinstance(rate, (int, float))
            or rate <= 0 or not math.isfinite(rate)
        ) for rate in target_rps) or len(set(target_rps)) != len(target_rps)):
            raise ValueError("Evaluation rates must be unique, positive and finite, or null for native timing")
        config["target_rps"] = target_rps
    cohorts, estimate, provenance = load_cohorts(data_dir / "processed", config, splits=(split,))
    if any(record.get(key) != value for key, value in provenance.items()):
        raise ValueError("Prepared scenarios differ from the training dataset")
    manifest = json.loads((data_dir / "processed/manifest.json").read_text(encoding="utf-8"))
    layout = {item["window"]: item for item in manifest.get("windows", []) if item["split"] == split}
    available_groups = {item.get("group", split) for item in layout.values()} or {split}
    if groups is not None:
        if not groups or not set(groups) <= available_groups:
            raise ValueError(f"Unknown evaluation group; available: {sorted(available_groups)}")
        cohorts[split] = [case for case in cohorts[split]
                          if layout.get(case[0], {}).get("group", split) in groups]
    horizons = {(window, rate): replay.horizon_s for window, rate, replay in cohorts[split]}
    models = {}
    before = {}
    for checkpoint in record["checkpoints"]:
        path = training_dir / f"dqn_seed{checkpoint['seed']}.npz"
        before[path.name] = sha256(path)
        if before[path.name] != checkpoint["sha256"]:
            raise ValueError("Checkpoint checksum changed")
        model, metadata = DQN.load(path)
        if (metadata["config"] != record["config"]
                or any(metadata.get(key) != value for key, value in provenance.items())):
            raise ValueError("Checkpoint metadata differs from the training record")
        models[f"dqn_seed{checkpoint['seed']}"] = model
    if not models:
        raise ValueError("No frozen checkpoints found")
    rows = evaluate(models, cohorts[split], HardwareConfig(**config["hardware"]),
                    ServiceProfile(**record["service_profile"]), estimate, config["decisions"], baselines=True)
    for row in rows:
        item = layout.get(row["window"], {})
        row.update(split=split, scenario_id=item.get("scenario_id", str(row["window"])),
                   traffic_regime=item.get("traffic_regime", "unfiltered"), group=item.get("group", split),
                   start_timestamp_s=item.get("start_timestamp_s"), end_timestamp_s=item.get("end_timestamp_s"),
                   measurement_window_s=horizons[row["window"], row["target_rps"]])
    summaries = []
    for label, regime, policy, rate in dict.fromkeys(
        (row["group"], row["traffic_regime"], row["policy"], row["target_rps"]) for row in rows
    ):
        group = [row for row in rows if (row["group"], row["traffic_regime"], row["policy"], row["target_rps"])
                 == (label, regime, policy, rate)]
        empty = [item for item in manifest.get("empty_windows", [])
                 if item["split"] == split and item.get("group", split) == label]
        idle_duration = sum(item["end_timestamp_s"] - item["start_timestamp_s"] for item in empty)
        duration = sum(row["measurement_window_s"] for row in group) + (idle_duration if rate is None else 0)
        requests = sum(row["arrivals"] for row in group)
        summaries.append(dict(group=label, traffic_regime=regime, policy=policy, target_rps=rate, windows=len(group),
                              empty_episodes=len(empty), native_empty_duration_s=idle_duration,
                              requests=requests, evaluated_duration_s=duration,
                              overall_slo_attainment=sum(row["slo_met"] for row in group) / requests,
                              overall_drop_rate=sum(row["drop_rate"] * row["arrivals"] for row in group) / requests,
                              overall_goodput_rps=sum(row["window_goodput_rps"] * row["measurement_window_s"]
                                                      for row in group) / duration,
                              mean_slo_attainment=fmean(row["slo_attainment"] for row in group),
                              mean_drop_rate=fmean(row["drop_rate"] for row in group),
                              mean_goodput_rps=fmean(row["window_goodput_rps"] for row in group)))
    if any(sha256(training_dir / name) != digest for name, digest in before.items()):
        raise RuntimeError("Frozen checkpoint changed during evaluation")
    output.mkdir(parents=True)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summaries)
    write_json(output / "run.json", dict(split=split, training_dir=str(training_dir),
               config=config, groups=groups, checkpoints=before,
               empty_episode_handling="Skipped execution; included in overall_goodput_rps duration for native timing only",
               **provenance))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/burstgpt3"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    rates = parser.add_mutually_exclusive_group()
    rates.add_argument("--target-rps", type=float, nargs="+", help="Override evaluation load only")
    rates.add_argument("--native", action="store_true", help="Use source timing regardless of training rates")
    parser.add_argument("--groups", nargs="+", help="Evaluate selected named groups from the requested split")
    args = parser.parse_args()
    rows = run(args.training_dir, args.data_dir, args.output, split=args.split,
               target_rps=[None] if args.native else args.target_rps, groups=args.groups)
    print(f"Completed {len(rows)} frozen policy runs. Results: {args.output / 'summary.csv'}")


if __name__ == "__main__":
    main()
