"""Compare learned admission and slack rules with a causal 5% drop allowance."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from statistics import fmean

from experiments.confirm_scheduling import ActionableReplay, choose_rule, subset
from experiments.scheduling_study import ROOT, THRESHOLDS, SlackAdmission, mean_rows
from experiments.train_dqn import load_cohorts, policy_options, run as train, write_csv, write_json
from llm_triage.metrics import evaluate_events
from llm_triage.rl import RLPolicyCallback
from llm_triage.rl.dqn import DQN
from llm_triage.rl.masking import ACTION_TO_INDEX
from llm_triage.simulation import Action, HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import sha256

RATES = [2.4, 4.8, 9.6]
DROP_DENOMINATOR = 20
METRICS = ("goodput", "attainment", "drop", "unfinished", "preemptions", "long_attainment")


class CappedDQN(RLPolicyCallback):
    """Apply the shared drop allowance before the DQN ranks its actions."""

    drop_allowed = True

    def _query_policy(self, obs, mask, state):
        mask[ACTION_TO_INDEX[Action.DROP]] &= self.drop_allowed
        return super()._query_policy(obs, mask, state)


class BudgetReplay(ActionableReplay):
    """Track the drop allowance from the kernel's observed arrivals."""

    @property
    def drop_allowed(self):
        return self.dropped_count < self._next // DROP_DENOMINATOR

    def can_act_on_blocked(self, candidate, eligible_count):
        if isinstance(self.policy, CappedDQN):
            cfg = self.policy.mask_config
            defers_left = cfg.max_defers is None or candidate.defer_count < cfg.max_defers
            has_slack = not cfg.require_positive_slack_for_defer or candidate.slo_slack > cfg.defer_retry_s
            return self.drop_allowed or (defers_left and has_slack)
        return self.drop_allowed and self.policy.should_drop(candidate)

    def choose_action(self, snapshot):
        if isinstance(self.policy, CappedDQN):
            self.policy.drop_allowed = self.drop_allowed
            return self.policy.decide(snapshot)
        return self.policy.decide(snapshot) if self.drop_allowed else Action.ADMIT


def evaluate(cases, models, config, hardware, estimate, window_groups, selected_rules=None):
    rows = []
    profile = ServiceProfile()
    for window, rate, replay in cases:
        rules = {"length_aware": SlackAdmission(float("-inf"))}
        rules.update({f"slack_{value}": SlackAdmission(value) for value in THRESHOLDS})
        if selected_rules is not None:
            rules = {name: rules[name] for name in dict.fromkeys(("length_aware", selected_rules[rate]))}
        policies = dict(rules)
        for name, model in models.items():
            policies[name] = CappedDQN(model, **policy_options(
                replay, config["decisions"], config["observation"], config["candidate_selector"]
            ))
        long_ids = {str(r.request_id) for r in replay.requests if r.output_tokens > 128}
        for name, policy in policies.items():
            kernel = BudgetReplay(policy, hardware, profile, estimate, config["decisions"])
            result = kernel.run(replay.requests)
            metrics = evaluate_events(result.events, measurement_window_s=replay.horizon_s)
            summary = metrics.summary
            arrivals = len(replay.requests)
            if summary["arrivals"] != arrivals or summary["dropped"] > arrivals // DROP_DENOMINATOR:
                raise RuntimeError("Replay did not preserve arrivals or exceeded its drop allowance")
            long_met = sum(r.slo_met for r in metrics.request_records if r.request_id in long_ids)
            group, block = window_groups[window]
            rows.append(dict(group=group, block=block, window=window, rate=rate, policy=name,
                             arrivals=summary["arrivals"], drops=summary["dropped"],
                             goodput=summary["window_goodput_rps"], attainment=summary["slo_attainment"],
                             drop=summary["drop_rate"], unfinished=summary["unfinished_rate"],
                             preemptions=result.preemptions, long_arrivals=len(long_ids),
                             long_attainment=long_met / len(long_ids) if long_ids else None))
    return rows


def paired_rows(evaluation, selected_rules):
    baseline = {(r["window"], r["rate"]): r for r in evaluation if r["policy"] == selected_rules[r["rate"]]}
    for r in evaluation:
        if r["policy"].startswith("dqn_"):
            rule = baseline[r["window"], r["rate"]]
            yield dict(group=r["group"], block=r["block"], window=r["window"], rate=r["rate"],
                       policy=r["policy"], rule=rule["policy"],
                       **{key: r[key] - rule[key] if r[key] is not None and rule[key] is not None else None
                          for key in METRICS})


def run(full_data, output, mode):
    output.mkdir(parents=True, exist_ok=False)
    processed, windows, blocks = subset(full_data / "processed", output)
    config = json.loads((ROOT / "configs/training_kv_stress.json").read_text(encoding="utf-8"))
    normal = json.loads((ROOT / "configs/week2_dqn.json").read_text(encoding="utf-8"))
    hardware = HardwareConfig(**normal["hardware"])
    if mode == "kv_only":
        hardware = replace(hardware, kv_total_blocks=config["hardware"]["kv_total_blocks"])
    config.update(hardware=vars(hardware), target_rps=RATES, candidate_selector="length_aware",
                  observation={"include_episode_progress": False})
    config["agent"]["discount_mode"] = "elapsed_time"
    training = output / "training"
    print(f"{mode}: training at {hardware.max_batch_size} slots and "
          f"{hardware.kv_total_blocks * hardware.kv_block_size} KV tokens", flush=True)
    train(config, processed, training)
    record = json.loads((training / "run.json").read_text(encoding="utf-8"))
    models = {}
    for checkpoint in record["checkpoints"]:
        seed = checkpoint["seed"]
        path = training / f"dqn_seed{seed}.npz"
        if checkpoint["sha256"] != sha256(path):
            raise ValueError("Checkpoint hash mismatch")
        models[f"dqn_{seed}"] = DQN.load(path)[0]
    dev, estimate, provenance = load_cohorts(processed, config, ("validation",))
    groups = {w["window"]: ("validation", None) for w in windows if w["split"] == "validation"}
    raw = evaluate(dev["validation"], models, config, hardware, estimate, groups)
    validation = mean_rows(raw, ["rate", "policy"], METRICS)
    rules = {rate: choose_rule(validation, rate, 1 / DROP_DENOMINATOR) for rate in RATES}
    write_csv(output / "validation.csv", validation)
    write_json(output / "frozen_selection.json", dict(
        mode=mode, protocol_sha256=sha256(ROOT / "docs/kv_pressure_study.md"),
        config=config, data=provenance, checkpoints=record["checkpoints"], rules=rules,
        drop_denominator=DROP_DENOMINATOR, test_blocks=blocks,
    ))
    print(f"{mode}: validation rules locked {rules}", flush=True)
    test, _, _ = load_cohorts(processed, config, ("test",))
    groups = {w["window"]: (group, index) for group, group_blocks in blocks.items()
              for index, block in enumerate(group_blocks) for w in block}
    evaluation = []
    for group in blocks:
        cases = [case for case in test["test"] if groups[case[0]][0] == group]
        evaluation.extend(evaluate(cases, models, config, hardware, estimate, groups, rules))
        print(f"{mode}: evaluated {group}", flush=True)
    pairs = list(paired_rows(evaluation, rules))
    write_csv(output / "evaluation.csv", evaluation)
    write_csv(output / "evaluation_summary.csv", mean_rows(evaluation, ["group", "rate", "policy"], METRICS))
    write_csv(output / "paired_windows.csv", pairs)
    write_csv(output / "paired_blocks.csv", mean_rows(pairs, ["group", "block", "rate", "policy", "rule"], METRICS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("standard", "kv_only"), required=True)
    args = parser.parse_args()
    run(args.full_data, args.output, args.mode)
