"""Run the three scheduling pilots on the prepared BurstGPT 1 windows."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from statistics import fmean

from experiments.baseline_capacity_sweep import diagnose_replay
from experiments.online_adaptation import replay_dqn
from experiments.train_dqn import load_cohorts, policy_options, run as train, write_csv, write_json
from llm_triage.metrics import evaluate_events
from llm_triage.rl import RLPolicyCallback
from llm_triage.rl.dqn import DQN
from llm_triage.scheduling import baseline_schedulers
from llm_triage.scheduling.length_aware import LengthAwareScheduler
from llm_triage.simulation import Action, HardwareConfig, ServiceProfile, simulate
from llm_triage.workload.burstgpt import sha256
from llm_triage.workload.transforms import ReplayWindow

RATES = [0.6, 1.2, 2.4, 4.8, 9.6]
THRESHOLDS = [-2, -1, 0, 1, 2]
BUDGETS = [0, .05, .10, .20, .30]
ONLINE = dict(epsilon=0.0, learning_rate=1e-5, warmup=2048, update_interval=32,
              min_update_seconds=1.0, min_new_outcomes=32, max_updates=100)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SlackAdmission(LengthAwareScheduler):
    cutoff: float

    def decide(self, snapshot):
        return Action.DROP if self.should_drop(snapshot.candidate) else Action.ADMIT

    def should_drop(self, candidate):
        return candidate.estimated_slack_s() < self.cutoff


def mean_rows(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [dict(zip(keys, group), windows=len(values),
                 **{key: fmean(row[key] for row in values if row[key] is not None)
                    if any(row[key] is not None for row in values) else None for key in metrics})
            for group, values in groups.items()]


def capacity_scan(cases, hardware_configs, profile, estimate, decisions, slo, *,
                  policy_names=None, skip_blocked_admits=False):
    rows = []
    fields = ("slo_attainment", "window_goodput_rps", "drop_rate", "p95_queue_wait_s",
              "mean_queue_in_window", "busy_fraction_in_window", "kv_utilization_in_window",
              "batch_slot_utilization_in_window", "preemptions", "isolated_exceeds_slo_fraction")
    for mode, hardware in hardware_configs.items():
        for window, rate, replay in cases:
            for policy, scheduler in baseline_schedulers().items():
                if policy_names is not None and policy not in policy_names:
                    continue
                drop_blocked = None
                if policy == "static_slo_aware":
                    drop_blocked = lambda candidate: candidate.estimated_slack_s() < 0
                result = simulate(replay.requests, scheduler, hardware, profile,
                                  skip_blocked_admits=skip_blocked_admits,
                                  drop_blocked_candidate=drop_blocked,
                                  estimated_output_tokens=estimate, **decisions)
                summary, _, _ = diagnose_replay(result, replay, slo, 10.0)
                rows.append(dict(capacity=mode, window=window, rate=rate, policy=policy,
                                 **{key: summary[key] for key in fields}))
    return rows, mean_rows(rows, ["capacity", "rate", "policy"], fields)


def choose_rates(summary):
    fcfs = {row["rate"]: row for row in summary
            if row["capacity"] == "kv_stress" and row["policy"] == "fcfs"}
    reference = fcfs[RATES[0]]["slo_attainment"]
    knee = next((rate for rate in RATES[1:] if reference - fcfs[rate]["slo_attainment"] >= .05
                 and fcfs[rate]["mean_queue_in_window"] >= 1), None)
    if knee is None:
        return RATES[1:4], None
    index = min(len(RATES) - 2, RATES.index(knee))
    return RATES[index - 1:index + 2], knee


def admission(cases, models, config, estimate, hardware, profile, *, skip_blocked_admits=False):
    rows = []
    for window, rate, replay in cases:
        policies = {"length_aware": LengthAwareScheduler()}
        policies.update({f"slack_{cutoff}": SlackAdmission(cutoff) for cutoff in THRESHOLDS})
        policies.update({
            name: RLPolicyCallback(model, **policy_options(
                replay, config["decisions"], config["observation"], config["candidate_selector"]
            )) for name, model in models.items()
        })
        long_ids = {str(r.request_id) for r in replay.requests if r.output_tokens > 128}
        for name, scheduler in policies.items():
            skip = skip_blocked_admits and name not in models
            drop_blocked = scheduler.should_drop if isinstance(scheduler, SlackAdmission) else None
            result = simulate(replay.requests, scheduler, hardware, profile, record_iterations=False,
                              skip_blocked_admits=skip, drop_blocked_candidate=drop_blocked,
                              estimated_output_tokens=estimate, **config["decisions"])
            evaluation = evaluate_events(result.events, measurement_window_s=replay.horizon_s)
            summary = evaluation.summary
            assert summary["arrivals"] == len(replay.requests)
            long_met = sum(r.slo_met for r in evaluation.request_records if r.request_id in long_ids)
            rows.append(dict(window=window, rate=rate, policy=name, arrivals=summary["arrivals"],
                             attainment=summary["slo_attainment"], drop=summary["drop_rate"],
                             goodput=summary["window_goodput_rps"], long_arrivals=len(long_ids),
                             long_met=long_met,
                             long_attainment=long_met / len(long_ids) if long_ids else None))
    return rows


def budget_comparison(validation, evaluation):
    comparisons = []
    for rate in sorted({r["rate"] for r in validation}):
        candidates = [r for r in validation if r["rate"] == rate and not r["policy"].startswith("dqn_")]
        observed = {r["policy"]: r for r in evaluation if r["rate"] == rate}
        for budget in BUDGETS:
            rule = max((r for r in candidates if r["drop"] <= budget), key=lambda r: (r["goodput"], -r["drop"]))
            baseline = observed[rule["policy"]]
            for policy, model in observed.items():
                if not policy.startswith("dqn_"):
                    continue
                comparisons.append(dict(
                    rate=rate, budget=budget, rule=rule["policy"], dqn=policy,
                    rule_validation_drop=rule["drop"], rule_drop=baseline["drop"], dqn_drop=model["drop"],
                    both_within_budget=baseline["drop"] <= budget and model["drop"] <= budget,
                    goodput_delta=model["goodput"] - baseline["goodput"],
                    attainment_delta=model["attainment"] - baseline["attainment"],
                ))
    return comparisons


def make_stream(parts):
    requests, phases = [], {}
    offset = 0.0
    for phase, replay in parts:
        if phase not in phases:
            phases[phase] = dict(start=offset, end=offset, ids=set())
        for request in replay.requests:
            requests.append(replace(request, arrival_time=offset + request.arrival_time_s,
                                    slo_deadline=offset + request.deadline_s))
            phases[phase]["ids"].add(str(request.request_id))
        offset += replay.horizon_s
        phases[phase]["end"] = offset
    return ReplayWindow(tuple(requests), offset, 0.0, 1.0), phases


def online_comparison(cases, rates, training_dir, record, output, *, settings):
    replay_by_rate = {(window, rate): replay for window, rate, replay in cases}
    windows = sorted({window for window, _, _ in cases})
    if len(windows) != 16:
        raise ValueError("The pilot requires the 16 pinned BurstGPT 1 test windows")
    phase_rows, audit_rows = [], []
    for stream_id, index in enumerate(range(0, len(windows), 4)):
        block = windows[index:index + 4]
        for condition, pattern in (
            ("short", ("pre", "pressure", "recovery", "recovery")),
            ("sustained", ("pre", "pressure", "pressure", "recovery")),
        ):
            parts = [(phase, replay_by_rate[(window, rates[-1] if phase == "pressure" else rates[0])])
                     for window, phase in zip(block, pattern)]
            replay, phases = make_stream(parts)
            pressure_start = phases["pressure"]["start"]
            pressure_end = phases["pressure"]["end"]
            for seed in record["config"]["seeds"]:
                directory = output / "online" / f"{stream_id}-{condition}-{seed}"
                directory.mkdir(parents=True)
                checkpoint = training_dir / f"dqn_seed{seed}.npz"
                for policy in ("frozen", "adapting"):
                    result, audit = replay_dqn(checkpoint, record, replay, directory, policy, seed=seed, **settings)
                    with (directory / f"{policy}_updates.csv").open(newline="", encoding="utf-8") as stream:
                        update_times = [float(row["sim_time_s"]) for row in csv.DictReader(stream)]
                    audit_rows.append(dict(
                        stream=stream_id, condition=condition, seed=seed, policy=policy,
                        pressure_start_s=pressure_start, pressure_end_s=pressure_end,
                        first_update_s=update_times[0] if update_times else None,
                        first_pressure_update_delay_s=next(
                            (t - pressure_start for t in update_times if pressure_start <= t < pressure_end), None
                        ),
                        updates_before_pressure=sum(t < pressure_start for t in update_times),
                        updates_during_pressure=sum(pressure_start <= t < pressure_end for t in update_times),
                        updates=audit["updates"], learner_seconds=audit["learner_seconds"],
                        max_weight_change=audit["max_weight_change"],
                    ))
                    metrics = evaluate_events(result.events, measurement_window_s=replay.horizon_s)
                    for phase, spec in phases.items():
                        records = [r for r in metrics.request_records if r.request_id in spec["ids"]]
                        assert len(records) == len(spec["ids"])
                        phase_rows.append(dict(
                            stream=stream_id, condition=condition, seed=seed, policy=policy,
                            phase=phase, arrivals=len(records), on_time=sum(r.slo_met for r in records),
                            dropped=sum(r.status.value == "dropped" for r in records),
                        ))
    return phase_rows, audit_rows


def run(data, output):
    config = json.loads((ROOT / "configs/training_kv_stress.json").read_text())
    normal = json.loads((ROOT / "configs/week2_dqn.json").read_text())
    processed = data / "processed"
    output.mkdir(parents=True, exist_ok=False)
    profile = ServiceProfile()
    stress = HardwareConfig(**config["hardware"])
    cases, estimate, provenance = load_cohorts(processed, dict(config, target_rps=RATES), ("validation",))
    raw, summary = capacity_scan(cases["validation"], {
        "standard": HardwareConfig(**normal["hardware"]), "kv_stress": stress,
    }, profile, estimate, config["decisions"], config["slo_s"])
    rates, knee = choose_rates(summary)
    write_csv(output / "capacity_runs.csv", raw)
    write_csv(output / "capacity_summary.csv", summary)
    config.update(target_rps=rates, candidate_selector="length_aware",
                  observation={"include_episode_progress": False})
    config["agent"]["discount_mode"] = "elapsed_time"
    write_json(output / "protocol.json", dict(rates=rates, knee=knee, config=config,
               online=ONLINE, data=provenance))
    print(f"Capacity scan selected {rates}, knee={knee}", flush=True)

    training = output / "training"
    train(config, processed, training)
    record = json.loads((training / "run.json").read_text())
    models = {}
    for checkpoint in record["checkpoints"]:
        seed = checkpoint["seed"]
        path = training / f"dqn_seed{seed}.npz"
        if sha256(path) != checkpoint["sha256"]:
            raise ValueError("Training checkpoint changed")
        models[f"dqn_{seed}"] = DQN.load(path)[0]
    metrics = ["attainment", "drop", "goodput", "long_attainment"]
    validation, _, _ = load_cohorts(processed, config, ("validation",))
    rows = admission(validation["validation"], models, config, estimate, stress, profile)
    selection = mean_rows(rows, ["rate", "policy"], metrics)
    write_csv(output / "admission_validation.csv", selection)
    test, _, _ = load_cohorts(processed, config, ("test",))
    rows = admission(test["test"], models, config, estimate, stress, profile)
    summary = mean_rows(rows, ["rate", "policy"], metrics)
    write_csv(output / "admission_runs.csv", rows)
    write_csv(output / "admission_summary.csv", summary)
    write_csv(output / "admission_budgets.csv", budget_comparison(selection, summary))
    print("Matched-order admission comparison complete", flush=True)

    phases, updates = online_comparison(test["test"], rates, training, record, output, settings=ONLINE)
    write_csv(output / "adaptation_phases.csv", phases)
    write_csv(output / "adaptation_updates.csv", updates)
    print(f"Scheduling study complete: {output}", flush=True)


def run_short_warmup(data, prior, output):
    record = json.loads((prior / "training/run.json").read_text())
    config = record["config"]
    cases, _, provenance = load_cohorts(data / "processed", config, ("test",))
    if any(record[key] != value for key, value in provenance.items()):
        raise ValueError("Pilot data differs from the training data")
    for checkpoint in record["checkpoints"]:
        path = prior / "training" / f"dqn_seed{checkpoint['seed']}.npz"
        if sha256(path) != checkpoint["sha256"]:
            raise ValueError("Pilot checkpoint differs from training")
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(ONLINE, warmup=256)
    phases, updates = online_comparison(cases["test"], config["target_rps"], prior / "training",
                                        record, output, settings=settings)
    write_csv(output / "adaptation_phases.csv", phases)
    write_csv(output / "adaptation_updates.csv", updates)
    write_json(output / "online_protocol.json", settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-warmup-from", type=Path, help="Reuse checkpoints from a completed pilot")
    args = parser.parse_args()
    if args.short_warmup_from:
        run_short_warmup(args.data_dir, args.short_warmup_from, args.output)
    else:
        run(args.data_dir, args.output)
