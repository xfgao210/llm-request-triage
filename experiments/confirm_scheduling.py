"""Evaluate the preregistered BurstGPT 3 follow-up."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile

from experiments.online_adaptation import replay_dqn
from experiments.scheduling_study import (
    BUDGETS, ONLINE, RATES, ROOT, SlackAdmission, admission, budget_comparison,
    capacity_scan, choose_rates, make_stream, mean_rows,
)
from experiments.train_dqn import load_cohorts, run as train, write_csv, write_json
from llm_triage.metrics import evaluate_events
from llm_triage.rl.dqn import DQN
from llm_triage.simulation import Action, HardwareConfig, ServiceProfile, TriageKernel
from llm_triage.workload.burstgpt import sha256

METRICS = ["attainment", "drop", "goodput", "long_attainment"]
GROUPS = ("mixed_traffic", "peak_heavy")
UPDATE_SETTINGS = dict(ONLINE, warmup=256)


def spaced(items, count):
    if len(items) < count:
        raise ValueError(f"Need {count} complete eligible blocks, found {len(items)}")
    return [items[index * (len(items) - 1) // (count - 1)] for index in range(count)]


def select_cohorts(manifest):
    all_windows = sorted(manifest["windows"] + manifest["empty_windows"],
                         key=lambda item: item["start_timestamp_s"])

    def eligible(item):
        return (item["end_timestamp_s"] - item["start_timestamp_s"] == 300
                and item["requests"] >= 128)

    train_windows = spaced([w for w in all_windows if w["split"] == "train" and eligible(w)], 24)
    validation = spaced([w for w in all_windows if w["split"] == "validation" and eligible(w)], 6)
    blocks = {}
    for group in GROUPS:
        windows = [w for w in all_windows if w["split"] == "test" and w["group"] == group]
        triplets = [windows[i:i + 3] for i in range(0, len(windows) - 2, 3)]
        candidates = [
            block for block in triplets if all(eligible(w) for w in block)
            and all(a["end_timestamp_s"] == b["start_timestamp_s"] for a, b in zip(block, block[1:]))
        ]
        blocks[group] = spaced(candidates, 4)
    picked = [*train_windows, *validation, *(w for group in blocks.values() for block in group for w in block)]
    keys = {(w["split"], w["window"]) for w in picked}
    if len(keys) != len(picked):
        raise ValueError("Window was selected more than once")
    return picked, blocks


def subset(full_processed, output):
    manifest_path = full_processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    range_config = ROOT / "configs/burstgpt3_ranges.json"
    expected_source = json.loads(range_config.read_text(encoding="utf-8"))["source_sha256"]
    if (manifest["dataset"] != "burstgpt3" or manifest["source_sha256"] != expected_source
            or manifest["range_config_sha256"] != sha256(range_config)):
        raise ValueError("Use the pinned BurstGPT 3 source and ranges to prepare data")
    trace = full_processed / "burstgpt_windows.csv"
    if sha256(trace) != manifest["selected_sha256"]:
        raise ValueError("Full trace differs from manifest")
    picked, blocks = select_cohorts(manifest)
    wanted = {(item["split"], item["window"]): item["requests"] for item in picked}
    counts = defaultdict(int)
    with tempfile.TemporaryDirectory(dir=output, prefix="cohorts-") as temporary:
        staging = Path(temporary) / "processed"
        staging.mkdir()
        destination = staging / trace.name
        output_sum = 0
        with trace.open(encoding="utf-8", newline="") as source, destination.open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(stream, reader.fieldnames)
            writer.writeheader()
            for row in reader:
                key = row["split"], int(row["window"])
                if key in wanted:
                    writer.writerow(row)
                    counts[key] += 1
                    if key[0] == "train":
                        output_sum += int(row["true_output_tokens"])
        if counts != wanted:
            raise ValueError("Selected requests differ from the full manifest")
        n_train = sum(amount for (split, _), amount in counts.items() if split == "train")
        write_json(staging / "manifest.json", dict(
            dataset="burstgpt3", windowing="fixed_duration", windows=picked,
            source_sha256=manifest["source_sha256"], selected_sha256=sha256(destination),
            range_config_sha256=manifest["range_config_sha256"],
            full_manifest_sha256=sha256(manifest_path),
            estimated_output_tokens_train_mean=output_sum / n_train,
        ))
        staging.rename(output / "processed")
    return output / "processed", picked, blocks


def choose_rule(rows, rate, budget):
    candidates = [r for r in rows if r["rate"] == rate and not r["policy"].startswith("dqn_")
                  and r["drop"] <= budget]
    return max(candidates, key=lambda row: (row["goodput"], -row["drop"]))["policy"]


def cutoff(policy):
    return float("-inf") if policy == "length_aware" else int(policy.removeprefix("slack_"))


@dataclass(frozen=True)
class QueueSlackRule(SlackAdmission):
    busy_cutoff: float

    def decide(self, snapshot):
        drop = self.should_drop_at_queue_length(snapshot.candidate, snapshot.queue_length)
        return Action.DROP if drop else Action.ADMIT

    def should_drop_at_queue_length(self, candidate, eligible_count):
        threshold = self.busy_cutoff if eligible_count >= 4 else self.cutoff
        return candidate.estimated_slack_s() < threshold


class ActionableReplay(TriageKernel):
    """Skip leading no-op admissions while keeping their effect on the current pass."""

    def __init__(self, policy, hardware, profile, estimate, decisions):
        super().__init__(hardware, profile, estimated_output_tokens=estimate,
                         priority_fn=policy.priority, record_iterations=False, **decisions)
        self.policy = policy

    def _eligible_candidates(self):
        eligible = super()._eligible_candidates()
        free_tokens = self.kv.blocks_free * self.hw.kv_block_size
        if all(request.input_tokens <= free_tokens for request in eligible):
            return eligible
        ranked = sorted(((self._view(request), request) for request in eligible),
                        key=lambda pair: self.policy.priority(pair[0]))
        for index, (candidate, request) in enumerate(ranked):
            if (request.input_tokens <= free_tokens
                    or self.can_act_on_blocked(candidate, len(ranked) - index)):
                return [item for item in eligible if item.request_id not in self._deferred_in_pass]
            # The original failed ADMIT would hide this request for this pass.
            self._deferred_in_pass.add(request.request_id)
        return []

    def run(self, requests):
        self.reset(requests)
        while not self.is_terminal():
            self.apply(self.choose_action(self.snapshot()))
        return self.result()

    def choose_action(self, snapshot):
        return self.policy.decide(snapshot)

    def can_act_on_blocked(self, candidate, eligible_count):
        raise NotImplementedError


class QueueSlackReplay(ActionableReplay):
    def can_act_on_blocked(self, candidate, eligible_count):
        return self.policy.should_drop_at_queue_length(candidate, eligible_count)


def phase_records(result, replay, phases):
    evaluated = evaluate_events(result.events, measurement_window_s=replay.horizon_s)
    records = {r.request_id: r for r in evaluated.request_records}
    expected = {str(r.request_id) for r in replay.requests}
    if set(records) != expected:
        raise RuntimeError("Paired stream does not contain the same arrivals")
    for phase, spec in phases.items():
        selected = [records[request_id] for request_id in spec["ids"]]
        yield dict(phase=phase, arrivals=len(selected), on_time=sum(r.slo_met for r in selected),
                   dropped=sum(r.status.value == "dropped" for r in selected))


def compare_online(cases, blocks, rates, training, record, rules, output):
    hardware = HardwareConfig(**record["config"]["hardware"])
    profile = ServiceProfile(**record["service_profile"])
    decisions = record["config"]["decisions"]
    estimate = record["estimated_output_tokens"]
    ordered = {(window, rate): replay for window, rate, replay in cases}
    phase_rows, audits = [], []
    for group, group_blocks in blocks.items():
        for index, block in enumerate(group_blocks):
            parts = [(phase, ordered[(window["window"], rate)]) for phase, rate, window in
                     zip(("pre", "pressure", "recovery"), (rates[0], rates[-1], rates[0]), block)]
            replay, phases = make_stream(parts)
            switch = QueueSlackRule(cutoff(rules[rates[0]]), cutoff(rules[rates[-1]]))
            rule_result = QueueSlackReplay(switch, hardware, profile, estimate, decisions).run(replay.requests)
            phase_rows.extend(dict(group=group, block=index, seed=None, policy="queue_slack", **row)
                              for row in phase_records(rule_result, replay, phases))
            for seed in record["config"]["seeds"]:
                directory = output / "online" / f"{group}-{index}-{seed}"
                directory.mkdir(parents=True)
                for policy in ("frozen", "adapting"):
                    result, audit = replay_dqn(training / f"dqn_seed{seed}.npz", record, replay, directory,
                                               policy, seed=seed, **UPDATE_SETTINGS)
                    phase_rows.extend(dict(group=group, block=index, seed=seed, policy=policy, **row)
                                      for row in phase_records(result, replay, phases))
                    with (directory / f"{policy}_updates.csv").open(encoding="utf-8", newline="") as stream:
                        times = [float(row["sim_time_s"]) for row in csv.DictReader(stream)]
                    start, end = phases["pressure"]["start"], phases["pressure"]["end"]
                    audits.append(dict(group=group, block=index, seed=seed, policy=policy,
                                       pressure_start_s=start, pressure_end_s=end,
                                       first_update_s=times[0] if times else None,
                                       updates_before_pressure=sum(t < start for t in times),
                                       updates_during_pressure=sum(start <= t < end for t in times),
                                       updates=audit["updates"], learner_seconds=audit["learner_seconds"]))
            print(f"Continuous replay complete: {group} block {index + 1}", flush=True)
    return phase_rows, audits


def run(full_data, output):
    output.mkdir(parents=True, exist_ok=False)
    processed, selected, blocks = subset(full_data / "processed", output)
    config = json.loads((ROOT / "configs/training_kv_stress.json").read_text(encoding="utf-8"))
    normal = json.loads((ROOT / "configs/week2_dqn.json").read_text(encoding="utf-8"))
    standard = HardwareConfig(**normal["hardware"])
    reduced = HardwareConfig(**config["hardware"])
    profile = ServiceProfile()
    dev, estimate, provenance = load_cohorts(processed, dict(config, target_rps=RATES), ("validation",))
    cases, capacity = capacity_scan(dev["validation"], {
        "standard": standard,
        "batch_only": replace(standard, max_batch_size=reduced.max_batch_size),
        "kv_only": replace(standard, kv_total_blocks=reduced.kv_total_blocks),
        "kv_stress": reduced,
    }, profile, estimate, config["decisions"], config["slo_s"],
        policy_names={"fcfs"}, skip_blocked_admits=True)
    rates, knee = choose_rates(capacity)
    write_csv(output / "capacity_development_runs.csv", cases)
    write_csv(output / "capacity_development_summary.csv", capacity)
    config.update(target_rps=rates, candidate_selector="length_aware",
                  observation={"include_episode_progress": False})
    config["agent"]["discount_mode"] = "elapsed_time"
    training = output / "training"
    print(f"BurstGPT 3 rates: {rates}, knee={knee}", flush=True)
    train(config, processed, training)
    record = json.loads((training / "run.json").read_text(encoding="utf-8"))
    models = {}
    for entry in record["checkpoints"]:
        seed = entry["seed"]
        path = training / f"dqn_seed{seed}.npz"
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Checkpoint changed for seed {seed}")
        models[f"dqn_{seed}"] = DQN.load(path)[0]
    dev, _, _ = load_cohorts(processed, config, ("validation",))
    validation = mean_rows(admission(dev["validation"], models, config, estimate, reduced, profile,
                                    skip_blocked_admits=True), ["rate", "policy"], METRICS)
    write_csv(output / "validation_admission_summary.csv", validation)
    rules = {rate: choose_rule(validation, rate, .05) for rate in rates}
    write_json(output / "frozen_selection.json", dict(
        protocol_sha256=sha256(ROOT / "docs/burstgpt3_confirmation.md"), rates=rates, knee=knee,
        config=config, data=provenance, validation_rules={str(rate): {
            str(budget): choose_rule(validation, rate, budget) for budget in BUDGETS} for rate in rates},
        online_settings=UPDATE_SETTINGS, checkpoints=record["checkpoints"],
        windows=selected, online_blocks=blocks,
    ))
    print("Validation choices frozen. Evaluating the two BurstGPT 3 test ranges.", flush=True)

    test, _, _ = load_cohorts(processed, config, ("test",))
    group_by_window = {w["window"]: w["group"] for w in selected if w["split"] == "test"}
    capacity_rows, admission_rows, budgets = [], [], []
    for group in GROUPS:
        cohort = [case for case in test["test"] if group_by_window[case[0]] == group]
        print(f"Evaluating {group} capacity ({len(cohort)} window/rate cases).", flush=True)
        raw, _ = capacity_scan(cohort, {"kv_stress": reduced}, profile, estimate,
                               config["decisions"], config["slo_s"], skip_blocked_admits=True)
        capacity_rows.extend(dict(group=group, **row) for row in raw)
        print(f"Evaluating {group} admission.", flush=True)
        rows = admission(cohort, models, config, estimate, reduced, profile,
                         skip_blocked_admits=True)
        admission_rows.extend(dict(group=group, **row) for row in rows)
        summary = mean_rows(rows, ["rate", "policy"], METRICS)
        budgets.extend(dict(group=group, **row) for row in budget_comparison(validation, summary))
        print(f"Capacity and admission evaluation complete: {group}", flush=True)
    write_csv(output / "capacity_evaluation_runs.csv", capacity_rows)
    write_csv(output / "admission_evaluation_runs.csv", admission_rows)
    write_csv(output / "admission_evaluation_summary.csv",
              mean_rows(admission_rows, ["group", "rate", "policy"], METRICS))
    write_csv(output / "admission_budgets.csv", budgets)
    phases, audits = compare_online(test["test"], blocks, rates, training, record, rules, output)
    write_csv(output / "online_phases.csv", phases)
    write_csv(output / "online_updates.csv", audits)
    print(f"BurstGPT 3 study complete: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.full_data, args.output)
