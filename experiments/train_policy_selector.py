"""Train a shallow FCFS/DQN selector with offline paired counterfactuals."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform

import numpy as np

from experiments.train_dqn import load_cohorts, policy_options, write_csv, write_json
from llm_triage.metrics import evaluate_events
from llm_triage.rl import RLPolicyCallback
from llm_triage.rl.dqn import DQN
from llm_triage.scheduling import BurstSwitchConfig, BurstSwitchScheduler, FCFSScheduler
from llm_triage.scheduling.policy_selector import (
    SELECTOR_FEATURES, LearnedSwitchScheduler, fit_selector, selector_features,
)
from llm_triage.simulation import HardwareConfig, ServiceProfile, TriageKernel, simulate
from llm_triage.workload.burstgpt import sha256


def choose_cases(cases, limit):
    """Select evenly spaced windows before inspecting policy outcomes."""
    windows = sorted({window for window, _, _ in cases})
    if len(windows) > limit:
        indices = np.linspace(0, len(windows) - 1, limit, dtype=int)
        windows = [windows[index] for index in indices]
    return [case for case in cases if case[0] in windows]


def policy_decision(policy, state):
    priority = getattr(policy, "priority", None)
    if priority is not None:
        candidate = min(state.queue_requests, key=priority)
        state = replace(state, candidate=candidate)
    return policy.decide(state)


def counterfactual(kernel, policy, lookahead_s):
    """Score an identical bounded arrival cohort, draining past the cutoff.

    Include all work still active/queued at the fork and all arrivals in
    [fork_time, fork_time + lookahead). Previously terminal outcomes cancel.
    Future arrivals after that cutoff cannot interfere with the cohort.
    Both branches run the cohort to termination so no pending work disappears
    from scoring merely because the wall-clock lookahead ended.
    """
    branch = kernel.fork(priority_fn=getattr(policy, "priority", None),
                         arrivals_before_s=kernel.now + lookahead_s)
    initial = branch.completed_count
    while not branch.is_terminal():
        branch.apply(policy.decide(branch.snapshot()))
    return branch.completed_count - initial


def collect_training_rows(cases, model, hardware, profile, estimate, decisions,
                          observation, candidate_selector, switch_config, *, samples_per_case, lookahead_s):
    rows = []
    for window, rate, replay in cases:
        custom = RLPolicyCallback(model, **policy_options(replay, decisions, observation, candidate_selector))
        # Cover states visited by each fixed policy, rather than FCFS alone.
        for behavior_name, behavior in (("fcfs", FCFSScheduler()), ("dqn", custom)):
            tracker = BurstSwitchScheduler(custom, switch_config)
            kernel = TriageKernel(hardware, profile, estimated_output_tokens=estimate,
                                  arrival_observer=tracker.observe_arrival, record_iterations=False, **decisions)
            kernel.reset(replay.requests)
            next_sample = 0.0
            interval = replay.horizon_s / samples_per_case
            samples = 0
            while not kernel.is_terminal():
                state = kernel.snapshot()
                if state.now_s >= next_sample and samples < samples_per_case:
                    features = selector_features(tracker, state)
                    fcfs = counterfactual(kernel, FCFSScheduler(), lookahead_s)
                    dqn = counterfactual(kernel, custom, lookahead_s)
                    rows.append(dict(window=window, load=rate, behavior=behavior_name,
                                     time_s=state.now_s, fcfs_on_time=fcfs, dqn_on_time=dqn,
                                     gain_on_time=dqn - fcfs, label_custom=int(dqn > fcfs),
                                     **dict(zip(SELECTOR_FEATURES, features))))
                    samples += 1
                    # Skip missed sampling intervals; never duplicate one state.
                    next_sample = (int(state.now_s / interval) + 1) * interval
                kernel.apply(policy_decision(behavior, state))
        print(f"labels window={window} load={rate} total={len(rows)}", flush=True)
    return rows


def evaluate_selector(cases, custom_factory, tree, hardware, profile, estimate, decisions,
                      switch_config, *, enter_probability, exit_probability):
    rows = []
    for window, rate, replay in cases:
        policies = {
            "fcfs": FCFSScheduler(),
            "dqn": custom_factory(replay),
            "burst_switch": BurstSwitchScheduler(custom_factory(replay), switch_config),
            "learned_switch": LearnedSwitchScheduler(custom_factory(replay), tree, config=switch_config,
                                                     enter_probability=enter_probability,
                                                     exit_probability=exit_probability),
        }
        expected = {str(r.request_id) for r in replay.requests}
        for name, policy in policies.items():
            result = simulate(replay.requests, policy, hardware, profile,
                              estimated_output_tokens=estimate, record_iterations=False, **decisions)
            evaluation = evaluate_events(result.events, measurement_window_s=replay.horizon_s)
            if {r.request_id for r in evaluation.request_records} != expected:
                raise RuntimeError("Policies did not evaluate the same complete request cohort")
            counts = getattr(policy, "decisions", {})
            rows.append(dict(window=window, load=rate, policy=name,
                             arrivals=evaluation.summary["arrivals"],
                             slo_met=evaluation.summary["slo_met"],
                             slo_attainment=evaluation.summary["slo_attainment"],
                             drop_rate=evaluation.summary["drop_rate"],
                             goodput_rps=evaluation.summary["window_goodput_rps"],
                             first_token_count=evaluation.summary["first_token_count"],
                             p95_ttft_s=evaluation.summary["p95_ttft_s"],
                             horizon_s=replay.horizon_s,
                             switches=len(getattr(policy, "switches", [])),
                             fcfs_decisions=counts.get("fcfs", 0), custom_decisions=counts.get("custom", 0),
                             switch_trace_json=json.dumps(getattr(policy, "switches", []), sort_keys=True)))
    return rows


def summarize(rows):
    summaries = []
    fcfs = {(r["seed"], r["window"], r["load"]): r["slo_attainment"] for r in rows if r["policy"] == "fcfs"}
    for seed, load, policy in dict.fromkeys((r["seed"], r["load"], r["policy"]) for r in rows):
        group = [r for r in rows if (r["seed"], r["load"], r["policy"]) == (seed, load, policy)]
        total = sum(r["arrivals"] for r in group)
        summaries.append(dict(seed=seed, load=load, policy=policy, windows=len(group), arrivals=total,
                              slo_attainment=sum(r["slo_met"] for r in group) / total,
                              drop_rate=sum(r["drop_rate"] * r["arrivals"] for r in group) / total,
                              goodput_rps=sum(r["goodput_rps"] * r["horizon_s"] for r in group)
                              / sum(r["horizon_s"] for r in group),
                              mean_paired_delta_vs_fcfs_pp=100 * sum(
                                  r["slo_attainment"] - fcfs[seed, r["window"], load] for r in group) / len(group),
                              switches=sum(r["switches"] for r in group),
                              custom_decisions=sum(r["custom_decisions"] for r in group),
                              fcfs_decisions=sum(r["fcfs_decisions"] for r in group)))
    return summaries


def run(training_dir, data_dir, output, *, train_windows=8, validation_windows=8,
        samples_per_case=6, lookahead_s=10.0, max_depth=3, min_samples_leaf=10,
        seeds=None, enter_probability=0.7, exit_probability=0.4):
    for name, value in dict(train_windows=train_windows, validation_windows=validation_windows,
                            samples_per_case=samples_per_case, max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf).items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(lookahead_s) or lookahead_s <= 0:
        raise ValueError("lookahead_s must be finite and positive")
    if not 0 <= exit_probability < enter_probability <= 1:
        raise ValueError("Require 0 <= exit_probability < enter_probability <= 1")
    if output.exists():
        raise FileExistsError(output)
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *sorted((root / "src/llm_triage").rglob("*.py")),
             Path(__file__).resolve().with_name("train_dqn.py")]
    fingerprint = {p.relative_to(root).as_posix(): sha256(p) for p in paths}
    record = json.loads((training_dir / "run.json").read_text(encoding="utf-8"))
    if record["selection_split"] != "validation" or record["test_evaluated"]:
        raise ValueError("Require validation-selected frozen DQN checkpoints")
    config = record["config"]
    cohorts, estimate, provenance = load_cohorts(data_dir / "processed", config, splits=("train", "validation"))
    if any(record.get(k) != v for k, v in provenance.items()):
        raise ValueError("Prepared data differs from DQN training data")
    training = choose_cases(cohorts["train"], train_windows)
    validation = choose_cases(cohorts["validation"], validation_windows)
    checkpoints = [c for c in record["checkpoints"] if seeds is None or c["seed"] in seeds]
    if not checkpoints or (seeds is not None and {c["seed"] for c in checkpoints} != set(seeds)):
        raise ValueError("Requested seeds are not available")
    models, before = {}, {}
    for checkpoint in checkpoints:
        path = training_dir / f"dqn_seed{checkpoint['seed']}.npz"
        before[path.name] = sha256(path)
        if before[path.name] != checkpoint["sha256"]:
            raise ValueError("Checkpoint checksum changed")
        model, metadata = DQN.load(path)
        if metadata["config"] != config or any(metadata.get(k) != v for k, v in provenance.items()):
            raise ValueError("Checkpoint metadata differs from training record")
        models[checkpoint["seed"]] = model
    hardware, profile = HardwareConfig(**config["hardware"]), ServiceProfile(**record["service_profile"])
    switch_config = BurstSwitchConfig()
    output.mkdir(parents=True)
    all_rows, label_counts = [], []
    for seed, model in models.items():
        seed_dir = output / f"seed{seed}"
        seed_dir.mkdir()
        weights = model.copy_weights()
        labels = collect_training_rows(training, model, hardware, profile, estimate, config["decisions"],
                                       config.get("observation"), config.get("candidate_selector", "fcfs"),
                                       switch_config, samples_per_case=samples_per_case, lookahead_s=lookahead_s)
        write_csv(seed_dir / "training_labels.csv", labels)
        tree = fit_selector(labels, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        write_json(seed_dir / "selector.json", dict(tree, burst_config=vars(switch_config),
                   enter_probability=enter_probability, exit_probability=exit_probability,
                   checkpoint_sha256=before[f"dqn_seed{seed}.npz"]))
        def custom_factory(replay):
            return RLPolicyCallback(model, **policy_options(
                replay, config["decisions"], config.get("observation"), config.get("candidate_selector", "fcfs")))
        rows = evaluate_selector(validation, custom_factory, tree, hardware, profile, estimate,
                                 config["decisions"], switch_config, enter_probability=enter_probability,
                                 exit_probability=exit_probability)
        all_rows.extend(dict(seed=seed, **row) for row in rows)
        label_counts.append(dict(seed=seed, samples=len(labels), custom_wins=sum(r["label_custom"] for r in labels),
                                 ties=sum(r["gain_on_time"] == 0 for r in labels),
                                 custom_losses=sum(r["gain_on_time"] < 0 for r in labels)))
        if any(not np.array_equal(weights[k], model.params[k]) for k in weights):
            raise RuntimeError("Frozen DQN weights changed")
        print(f"seed={seed} labels={label_counts[-1]} validation_runs={len(rows)}", flush=True)
    if any(sha256(training_dir / name) != digest for name, digest in before.items()):
        raise RuntimeError("Frozen checkpoint file changed")
    if {p.relative_to(root).as_posix(): sha256(p) for p in paths} != fingerprint:
        raise RuntimeError("Experiment source code changed during the run")
    write_csv(output / "validation.csv", all_rows)
    write_csv(output / "summary.csv", summarize(all_rows))
    write_csv(output / "label_counts.csv", label_counts)
    import sklearn
    write_json(output / "run.json", dict(
        training_dir=str(training_dir), data_dir=str(data_dir), config=config,
        checkpoints=before, **provenance, selector_training_split="train", evaluation_split="validation",
        test_evaluated=False, train_cases=[dict(window=w, load=r) for w, r, _ in training],
        validation_cases=[dict(window=w, load=r) for w, r, _ in validation],
        label_counts=label_counts, samples_per_case=samples_per_case, lookahead_s=lookahead_s,
        label_objective="additional on-time completions after draining identical bounded arrival cohort; ties choose FCFS",
        behaviors=["fcfs", "dqn"], max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        enter_probability=enter_probability, exit_probability=exit_probability,
        burst_config=vars(switch_config), runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                                     sklearn=sklearn.__version__),
        code_sha256=fingerprint,
    ))
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-windows", type=int, default=8)
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--samples-per-case", type=int, default=6)
    parser.add_argument("--lookahead-s", type=float, default=10.0)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--enter-probability", type=float, default=0.7)
    parser.add_argument("--exit-probability", type=float, default=0.4)
    args = parser.parse_args()
    rows = run(**vars(args))
    print(f"Completed {len(rows)} validation policy runs. Results: {args.output}")


if __name__ == "__main__":
    main()
