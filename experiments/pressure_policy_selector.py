"""Try the learned selector under the documented KV-pressure protocol."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from experiments.kv_pressure_study import BudgetReplay, CappedDQN, DROP_DENOMINATOR, RATES
from experiments.scheduling_study import SlackAdmission
from experiments.train_dqn import load_cohorts, policy_options, write_csv, write_json
from experiments.train_policy_selector import choose_cases
from llm_triage.metrics import evaluate_events
from llm_triage.rl.dqn import DQN
from llm_triage.scheduling import BurstSwitchConfig, BurstSwitchScheduler, LengthAwareScheduler
from llm_triage.scheduling.policy_selector import (
    SELECTOR_FEATURES, LearnedSwitchScheduler, fit_selector, selector_features,
)
from llm_triage.simulation import EventType, HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import sha256


class CappedSelector(CappedDQN):
    """Expose a hybrid through the original budget replay's DQN contract.

    All modes see identical shortest-service ordering. The admit-only mode
    never drops; the DQN mode ranks actions with the original causal drop mask.
    Blocked candidates use the original replay's defer/drop feasibility test.
    """

    def __init__(self, custom, selector):
        self.custom = custom
        self.selector = selector
        self.mask_config = custom.mask_config
        self.drop_allowed = True

    def priority(self, candidate):
        return LengthAwareScheduler().priority(candidate)

    def decide(self, state):
        self.custom.drop_allowed = self.drop_allowed
        ranked = tuple(sorted(state.queue_requests, key=self.priority))
        return self.selector.decide(replace(state, queue_requests=ranked, candidate=ranked[0]))


def make_replay(policy, hardware, profile, estimate, decisions, tracker=None):
    kernel = BudgetReplay(policy, hardware, profile, estimate, decisions)
    kernel._arrival_observer = tracker.observe_arrival if tracker is not None else None
    return kernel


def rollout_score(kernel, policy, lookahead_s):
    branch = kernel.fork(arrivals_before_s=kernel.now + lookahead_s, priority_fn=policy.priority)
    branch.policy = policy
    before = branch.completed_count
    # A policy change can make the old blocked candidate no longer actionable.
    # Re-establish the decision point with the new budget/feasibility contract.
    branch._advance_until_decision_or_done()
    while not branch.is_terminal():
        branch.apply(branch.choose_action(branch.snapshot()))
    return branch.completed_count - before


def verify_drop_prefix(events):
    arrivals = drops = 0
    for event in events:
        if event.event_type is EventType.ARRIVAL:
            arrivals += 1
        elif event.event_type is EventType.DROP:
            drops += 1
        if drops > arrivals // DROP_DENOMINATOR:
            raise RuntimeError("Policy exceeded the causal 5% drop allowance")


def training_labels(cases, custom_factory, hardware, profile, estimate, decisions,
                    switch_config, samples_per_case, lookahead_s):
    rows = []
    for window, rate, replay in cases:
        custom = custom_factory(replay)
        admit = SlackAdmission(float("-inf"))
        for behavior_name, behavior in (("admit_only", admit), ("dqn", custom)):
            tracker = BurstSwitchScheduler(custom, switch_config)
            kernel = make_replay(behavior, hardware, profile, estimate, decisions, tracker)
            kernel.reset(replay.requests)
            samples, next_sample = 0, 0.0
            interval = replay.horizon_s / samples_per_case
            while not kernel.is_terminal():
                state = kernel.snapshot()
                if state.now_s >= next_sample and samples < samples_per_case:
                    features = selector_features(tracker, state)
                    admit_score = rollout_score(kernel, admit, lookahead_s)
                    dqn_score = rollout_score(kernel, custom, lookahead_s)
                    rows.append(dict(window=window, load=rate, behavior=behavior_name, time_s=state.now_s,
                                     admit_on_time=admit_score, dqn_on_time=dqn_score,
                                     gain_on_time=dqn_score - admit_score, label_custom=int(dqn_score > admit_score),
                                     **dict(zip(SELECTOR_FEATURES, features))))
                    samples += 1
                    next_sample = (int(state.now_s / interval) + 1) * interval
                kernel.apply(kernel.choose_action(state))
            verify_drop_prefix(kernel.events())
        print(f"labels window={window} rps={rate} total={len(rows)}", flush=True)
    return rows


def evaluate_cases(cases, custom_factory, tree, hardware, profile, estimate, decisions,
                   switch_config, rules):
    rows = []
    for window, rate, replay in cases:
        burst_custom, learned_custom = custom_factory(replay), custom_factory(replay)
        burst = BurstSwitchScheduler(burst_custom, switch_config)
        learned = LearnedSwitchScheduler(learned_custom, tree, config=switch_config)
        rule = rules[str(rate)]
        cutoff = float("-inf") if rule == "length_aware" else float(rule.removeprefix("slack_"))
        policies = {
            "admit_only": SlackAdmission(float("-inf")),
            "selected_slack": SlackAdmission(cutoff),
            "dqn": custom_factory(replay),
            "burst_switch": CappedSelector(burst_custom, burst),
            "learned_switch": CappedSelector(learned_custom, learned),
        }
        long_ids = {str(r.request_id) for r in replay.requests if r.output_tokens > 128}
        for name, policy in policies.items():
            tracker = policy.selector if isinstance(policy, CappedSelector) else None
            kernel = make_replay(policy, hardware, profile, estimate, decisions, tracker)
            result = kernel.run(replay.requests)
            verify_drop_prefix(result.events)
            evaluation = evaluate_events(result.events, measurement_window_s=replay.horizon_s)
            if {r.request_id for r in evaluation.request_records} != {str(r.request_id) for r in replay.requests}:
                raise RuntimeError("Evaluation changed the paired arrival cohort")
            summary = evaluation.summary
            counts = tracker.decisions if tracker is not None else {}
            rows.append(dict(window=window, load=rate, policy=name, selected_rule=rule,
                             arrivals=summary["arrivals"], slo_met=summary["slo_met"], drops=summary["dropped"],
                             attainment=summary["slo_attainment"], goodput=summary["window_goodput_rps"],
                             drop_rate=summary["drop_rate"], unfinished_rate=summary["unfinished_rate"],
                             preemptions=result.preemptions, horizon_s=replay.horizon_s,
                             long_arrivals=len(long_ids),
                             long_met=sum(r.slo_met for r in evaluation.request_records if r.request_id in long_ids),
                             switches=len(tracker.switches) if tracker is not None else 0,
                             admit_decisions=counts.get("fcfs", 0), dqn_decisions=counts.get("custom", 0),
                             switch_trace_json=json.dumps(tracker.switches if tracker is not None else [], sort_keys=True)))
        print(f"validation window={window} rps={rate}", flush=True)
    return rows


def summarize(rows):
    reference = {(r["seed"], r["window"], r["load"]): r for r in rows if r["policy"] == "selected_slack"}
    output = []
    for seed, load, policy in dict.fromkeys((r["seed"], r["load"], r["policy"]) for r in rows):
        group = [r for r in rows if (r["seed"], r["load"], r["policy"]) == (seed, load, policy)]
        n = sum(r["arrivals"] for r in group)
        longs = sum(r["long_arrivals"] for r in group)
        deltas = [r["goodput"] - reference[seed, r["window"], load]["goodput"] for r in group]
        output.append(dict(seed=seed, load=load, policy=policy, windows=len(group), arrivals=n,
                           attainment=sum(r["slo_met"] for r in group) / n,
                           mean_window_attainment=fmean(r["attainment"] for r in group),
                           drop_rate=sum(r["drops"] for r in group) / n,
                           goodput=sum(r["goodput"] * r["horizon_s"] for r in group) / sum(r["horizon_s"] for r in group),
                           mean_window_goodput=fmean(r["goodput"] for r in group),
                           mean_paired_goodput_delta_vs_rule=fmean(deltas),
                           windows_above_rule=sum(d > 0 for d in deltas),
                           windows_equal_rule=sum(d == 0 for d in deltas),
                           windows_below_rule=sum(d < 0 for d in deltas),
                           unfinished_rate=sum(r["unfinished_rate"] * r["arrivals"] for r in group) / n,
                           preemptions=sum(r["preemptions"] for r in group),
                           long_attainment=sum(r["long_met"] for r in group) / longs if longs else None,
                           switches=sum(r["switches"] for r in group),
                           admit_decisions=sum(r["admit_decisions"] for r in group),
                           dqn_decisions=sum(r["dqn_decisions"] for r in group)))
    return output


def run(study_dir, output, *, train_windows=8, samples_per_case=6, lookahead_s=10.0, seeds=None):
    if type(train_windows) is not int or train_windows < 1 or type(samples_per_case) is not int or samples_per_case < 1:
        raise ValueError("Window/sample limits must be positive integers")
    if not np.isfinite(lookahead_s) or lookahead_s <= 0:
        raise ValueError("Lookahead must be finite and positive")
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((study_dir / "training/run.json").read_text(encoding="utf-8"))
    frozen = json.loads((study_dir / "frozen_selection.json").read_text(encoding="utf-8"))
    config = record["config"]
    if (config != frozen["config"] or record["checkpoints"] != frozen["checkpoints"]
            or config["target_rps"] != RATES or config["candidate_selector"] != "length_aware"
            or config["hardware"]["max_batch_size"] != 32
            or record["selection_split"] != "validation" or record["test_evaluated"]
            or config["slo_s"] != 5 or frozen["drop_denominator"] != DROP_DENOMINATOR):
        raise ValueError("Study does not match the documented frozen KV-pressure protocol")
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "experiments/train_policy_selector.py",
             root / "experiments/train_dqn.py", root / "experiments/kv_pressure_study.py",
             root / "experiments/confirm_scheduling.py", root / "experiments/scheduling_study.py",
             *sorted((root / "src/llm_triage").rglob("*.py"))]
    fingerprint = {p.relative_to(root).as_posix(): sha256(p) for p in paths}
    cohorts, estimate, provenance = load_cohorts(study_dir / "processed", config, ("train", "validation"))
    if any(record.get(k) != v or frozen["data"].get(k) != v for k, v in provenance.items()):
        raise ValueError("Study data differs from frozen training/selection")
    hardware, profile = HardwareConfig(**config["hardware"]), ServiceProfile(**record["service_profile"])
    training = choose_cases(cohorts["train"], train_windows)
    checkpoints = [c for c in record["checkpoints"] if seeds is None or c["seed"] in seeds]
    if not checkpoints or (seeds is not None and {c["seed"] for c in checkpoints} != set(seeds)):
        raise ValueError("Requested seeds are unavailable")
    before = {}
    for c in checkpoints:
        path = study_dir / "training" / f"dqn_seed{c['seed']}.npz"
        before[path.name] = sha256(path)
        if before[path.name] != c["sha256"]:
            raise ValueError("Checkpoint checksum changed")
    output.mkdir(parents=True)
    switch_config = BurstSwitchConfig()
    rows, counts = [], []
    for c in checkpoints:
        seed = c["seed"]
        model, metadata = DQN.load(study_dir / "training" / f"dqn_seed{seed}.npz")
        if metadata["config"] != config or any(metadata.get(k) != v for k, v in provenance.items()):
            raise ValueError("Checkpoint metadata changed")
        weights = model.copy_weights()
        def factory(replay):
            return CappedDQN(model, **policy_options(replay, config["decisions"], config["observation"], "length_aware"))
        labels = training_labels(training, factory, hardware, profile, estimate, config["decisions"],
                                 switch_config, samples_per_case, lookahead_s)
        tree = fit_selector(labels)
        seed_dir = output / f"seed{seed}"
        seed_dir.mkdir()
        write_csv(seed_dir / "training_labels.csv", labels)
        write_json(seed_dir / "selector.json", dict(tree, checkpoint_sha256=c["sha256"],
                   burst_config=vars(switch_config), enter_probability=0.7, exit_probability=0.4,
                   baseline="length_order_admit_only", drop_denominator=DROP_DENOMINATOR))
        counts.append(dict(seed=seed, samples=len(labels), dqn_wins=sum(r["label_custom"] for r in labels),
                           ties=sum(r["gain_on_time"] == 0 for r in labels),
                           dqn_losses=sum(r["gain_on_time"] < 0 for r in labels)))
        case_rows = evaluate_cases(cohorts["validation"], factory, tree, hardware, profile, estimate,
                                   config["decisions"], switch_config, frozen["rules"])
        rows.extend(dict(seed=seed, **r) for r in case_rows)
        if any(not np.array_equal(weights[k], model.params[k]) for k in weights):
            raise RuntimeError("DQN weights changed")
        print(f"seed={seed}: {counts[-1]}; validation runs={len(case_rows)}", flush=True)
    if any(sha256(study_dir / "training" / name) != value for name, value in before.items()):
        raise RuntimeError("Frozen checkpoint changed")
    if {p.relative_to(root).as_posix(): sha256(p) for p in paths} != fingerprint:
        raise RuntimeError("Source changed during the experiment")
    write_csv(output / "validation.csv", rows)
    write_csv(output / "summary.csv", summarize(rows))
    write_csv(output / "label_counts.csv", counts)
    import sklearn
    write_json(output / "run.json", dict(study_dir=str(study_dir), config=config, **provenance,
               checkpoints=before, rules=frozen["rules"], protocol_sha256=sha256(root / "docs/kv_pressure_study.md"),
               protocol_git_ref="76a2933a9fc8b4378aebbbd03cf2b1d959b5c2f9",
               selector_training_split="train", evaluation_split="validation", test_evaluated=False,
               train_cases=[dict(window=w, load=r) for w, r, _ in training],
               validation_cases=[dict(window=w, load=r) for w, r, _ in cohorts["validation"]],
               label_counts=counts, samples_per_case=samples_per_case, lookahead_s=lookahead_s,
               max_depth=3, min_samples_leaf=10, enter_probability=0.7, exit_probability=0.4,
               drop_denominator=DROP_DENOMINATOR, candidate_order="length_aware",
               label_objective="additional on-time completions after draining identical bounded cohort; ties use admit-only",
               code_sha256=fingerprint, numpy=np.__version__, sklearn=sklearn.__version__))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-windows", type=int, default=8)
    parser.add_argument("--samples-per-case", type=int, default=6)
    parser.add_argument("--lookahead-s", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    rows = run(**vars(args))
    print(f"Completed {len(rows)} validation policy runs. Results: {args.output}")


if __name__ == "__main__":
    main()
