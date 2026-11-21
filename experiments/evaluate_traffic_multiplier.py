"""Replay frozen pressure selectors at native 1x and true 8x traffic."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.adaptation_stress import select_evaluation_windows
from experiments.baseline_capacity_sweep import load_selected_rows
from experiments.kv_pressure_study import CappedDQN
from experiments.pressure_policy_selector import evaluate_cases, summarize
from experiments.train_dqn import load_cohorts, policy_options, write_csv, write_json
from llm_triage.rl.dqn import DQN
from llm_triage.scheduling import BurstSwitchConfig
from llm_triage.simulation import HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import sha256
from llm_triage.workload.transforms import replay_window


def scaled_cases(native_cases, multipliers):
    """Preserve every request and SLO budget; divide offsets and horizon by M."""
    from dataclasses import replace

    cases = []
    for window, _, replay in native_cases:
        if replay.time_scale != 1.0:
            raise ValueError("Multiplier scaling requires unscaled native timing")
        for multiplier in multipliers:
            if isinstance(multiplier, bool) or not np.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("Multipliers must be finite and positive")
            requests = tuple(replace(r, arrival_time=r.arrival_time / multiplier,
                                     slo_deadline=r.arrival_time / multiplier + (r.slo_deadline - r.arrival_time))
                             for r in replay.requests)
            cases.append((window, multiplier, replace(replay, requests=requests,
                          horizon_s=replay.horizon_s / multiplier, time_scale=1 / multiplier)))
    return cases


def native_cases(study_dir, config, scope, full_data):
    if scope == "validation":
        native_config = dict(config, target_rps=[None])
        native_config.pop("arrival_multipliers", None)
        cohorts, estimate, provenance = load_cohorts(study_dir / "processed", native_config, ("validation",))
        return cohorts["validation"], estimate, provenance, None
    manifest_path = full_data / "processed/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    study_manifest = json.loads((study_dir / "processed/manifest.json").read_text(encoding="utf-8"))
    if (manifest["source_sha256"] != study_manifest["source_sha256"]
            or manifest["range_config_sha256"] != study_manifest["range_config_sha256"]):
        raise ValueError("Peak data differs from the study's source or chronological ranges")
    trace = full_data / "processed/burstgpt_windows.csv"
    if sha256(trace) != manifest["selected_sha256"]:
        raise ValueError("Full prepared trace differs from manifest")
    windows = select_evaluation_windows(manifest, seconds=900, methods={"peak_heavy": "burst"}, split="test")
    selected = load_selected_rows(trace, windows, allowed_splits=("test",))
    cases = []
    for index, window in enumerate(windows):
        replay = replay_window(selected[window["name"]], slo_s=config["slo_s"],
                               bounds=(window["start_timestamp_s"], window["end_timestamp_s"]))
        if len(replay.requests) != window["requests"]:
            raise RuntimeError("Peak window did not retain every request")
        cases.append((index, None, replay))
    return cases, study_manifest["estimated_output_tokens_train_mean"], dict(
        source_sha256=manifest["source_sha256"], trace_sha256=sha256(trace),
        manifest_sha256=sha256(manifest_path)), windows


def run(study_dir, selector_dir, output, *, scope="validation", full_data=None,
        multipliers=(1.0, 8.0), seeds=None):
    if scope not in ("validation", "peak"):
        raise ValueError("Select validation or peak scope")
    if not multipliers or len(set(multipliers)) != len(multipliers):
        raise ValueError("Use unique positive multipliers")
    if output.exists():
        raise FileExistsError(output)
    record_path = study_dir / "training/run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    selected_path = study_dir / "frozen_selection.json"
    frozen = json.loads(selected_path.read_text(encoding="utf-8"))
    selector_record_path = selector_dir / "run.json"
    selector_record = json.loads(selector_record_path.read_text(encoding="utf-8"))
    config = record["config"]
    if (selector_record["config"] != config or frozen["config"] != config
            or selector_record["test_evaluated"] or selector_record["selector_training_split"] != "train"
            or selector_record["evaluation_split"] != "validation"):
        raise ValueError("Require the existing train-fitted, validation-evaluated pressure selector")
    native, estimate, provenance, windows = native_cases(study_dir, config, scope, full_data)
    if scope == "validation" and any(record.get(k) != v for k, v in provenance.items()):
        raise ValueError("Validation data differs from checkpoint training provenance")
    cases = scaled_cases(native, multipliers)
    hardware, profile = HardwareConfig(**config["hardware"]), ServiceProfile(**record["service_profile"])
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "experiments/pressure_policy_selector.py",
             root / "experiments/kv_pressure_study.py", root / "experiments/confirm_scheduling.py",
             root / "experiments/scheduling_study.py", root / "experiments/train_dqn.py",
             root / "experiments/train_policy_selector.py", root / "experiments/adaptation_stress.py",
             root / "experiments/baseline_capacity_sweep.py", *sorted((root / "src/llm_triage").rglob("*.py"))]
    fingerprint = {p.relative_to(root).as_posix(): sha256(p) for p in paths}
    checkpoints = [c for c in record["checkpoints"] if seeds is None or c["seed"] in seeds]
    if not checkpoints or (seeds is not None and {c["seed"] for c in checkpoints} != set(seeds)):
        raise ValueError("Requested seeds are unavailable")
    # Keep the original 9.6-RPS slack cutoff fixed for both native and 8x.
    # It is a frozen reference, not a rule newly selected for 8x traffic.
    reference_rule = frozen["rules"]["9.6"]
    rules = {str(m): reference_rule for m in multipliers}
    native_rates = {w: len(r.requests) / r.horizon_s for w, _, r in native}
    inputs = {str(p): sha256(p) for p in (record_path, selected_path, selector_record_path)}
    models = {}
    for checkpoint in checkpoints:
        seed = checkpoint["seed"]
        path = study_dir / "training" / f"dqn_seed{seed}.npz"
        tree_path = selector_dir / f"seed{seed}/selector.json"
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        inputs[str(path)] = sha256(path)
        inputs[str(tree_path)] = sha256(tree_path)
        if (inputs[str(path)] != checkpoint["sha256"] or tree["checkpoint_sha256"] != checkpoint["sha256"]
                or selector_record["checkpoints"][path.name] != checkpoint["sha256"]):
            raise ValueError("Frozen selector/checkpoint binding changed")
        if (tree["enter_probability"], tree["exit_probability"], tree["drop_denominator"]) != (0.7, 0.4, 20):
            raise ValueError("This replay expects the existing frozen pressure selector settings")
        model, metadata = DQN.load(path)
        if metadata["config"] != config or any(metadata.get(k) != record[k]
                                                for k in ("trace_sha256", "manifest_sha256", "source_sha256")):
            raise ValueError("Checkpoint metadata differs from frozen training")
        models[seed] = (model, tree)
    output.mkdir(parents=True)
    write_json(output / "traffic.json", dict(scope=scope, windows=windows, cases=[dict(
        window=w, arrival_multiplier=m, requests=len(r.requests), horizon_s=r.horizon_s,
        native_rps=native_rates[w], effective_rps=native_rates[w] * m) for w, m, r in cases]))
    rows = []
    for seed, (model, tree) in models.items():
        weights = model.copy_weights()
        def factory(replay):
            return CappedDQN(model, **policy_options(replay, config["decisions"], config["observation"], "length_aware"))
        count_before = len(rows)
        for case in cases:
            for policy_name in ("admit_only", "selected_slack", "dqn", "burst_switch", "learned_switch"):
                evaluated = evaluate_cases([case], factory, tree, hardware, profile, estimate, config["decisions"],
                                           BurstSwitchConfig(**tree["burst_config"]), rules,
                                           load_unit="arrival_multiplier", policy_names=[policy_name])
                for row in evaluated:
                    multiplier = row["load"]
                    rows.append(dict(seed=seed, **row, arrival_multiplier=multiplier,
                                     native_rps=native_rates[row["window"]], effective_rps=native_rates[row["window"]] * multiplier))
                write_csv(output / "runs.partial.csv", rows)
        if any(not np.array_equal(weights[k], model.params[k]) for k in weights):
            raise RuntimeError("Frozen DQN weights changed")
        print(f"seed={seed}: completed {len(rows) - count_before} frozen-policy replays", flush=True)
    if any(sha256(Path(p)) != value for p, value in inputs.items()):
        raise RuntimeError("Frozen experiment inputs changed")
    if {p.relative_to(root).as_posix(): sha256(p) for p in paths} != fingerprint:
        raise RuntimeError("Source changed during replay")
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    write_json(output / "run.json", dict(scope=scope, load_semantics="multiplier of original native traffic",
               arrival_multipliers=list(multipliers), study_dir=str(study_dir), selector_dir=str(selector_dir),
               frozen_inputs_sha256=inputs, config=config, reference_rule=reference_rule,
               rule_selection="original 9.6-RPS cutoff held fixed; not selected at 8x",
               evaluation_split="validation" if scope == "validation" else "previously_used_test_peak_heavy",
               test_evaluated=scope == "peak", models_retrained=False, selectors_refitted=False,
               source_sha256=provenance, code_sha256=fingerprint, total_policy_runs=len(rows)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=("validation", "peak"), default="validation")
    parser.add_argument("--full-data", type=Path, default=Path("data/burstgpt3/confirmation-full"))
    parser.add_argument("--multipliers", type=float, nargs="+", default=[1.0, 8.0])
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    rows = run(**vars(args))
    print(f"Completed {len(rows)} policy replays. Results: {args.output}")


if __name__ == "__main__":
    main()
