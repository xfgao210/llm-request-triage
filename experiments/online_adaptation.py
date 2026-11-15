"""Paired, chronological frozen-versus-adapting DQN replay.

The simulator knows the trace; agents receive only sanitized observations and
rewards returned by completed env.step calls. No resampling of traffic episodes,
validation selection, or replay-buffer preloading is performed.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.train_dqn import policy_options, write_csv, write_json
from llm_triage.metrics import evaluate_events
from llm_triage.metrics.summary import summarize_records
from llm_triage.rl import RewardConfig, TriageEnv
from llm_triage.rl.dqn import DQN, ReplayBuffer
from llm_triage.simulation import HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import load_windows, sha256
from llm_triage.workload.transforms import replay_window


def run(training_dir, data_dir, output, *, seed=7, groups=None,
        epsilon=0.0, learning_rate=0.0001, batch_size=64, warmup=256,
        update_interval=4, capacity=50000, max_seconds_per_group=None, device="auto"):
    """Replay each policy once; persist adapting weights across chronological groups.

    Both policies use ONE continuous environment covering selected ranges and
    intervening idle gaps. Rewards learned after a step can affect only subsequent
    decisions at or after that step's simulated time. Optimizer, target network
    and replay buffer start fresh; this is adaptation, not exact training resume.
    """
    if output.exists():
        raise FileExistsError(output)
    from llm_triage.rl.devices import resolve_device
    resolved_device = resolve_device(device)
    for name, value in dict(seed=seed, batch_size=batch_size, warmup=warmup,
                            update_interval=update_interval, capacity=capacity).items():
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid {name}")
    if not batch_size <= warmup <= capacity:
        raise ValueError("Require batch_size <= warmup <= capacity")
    if (not math.isfinite(epsilon) or not 0 <= epsilon <= 1
            or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid epsilon or learning rate")
    if max_seconds_per_group is not None and (
        not math.isfinite(max_seconds_per_group) or max_seconds_per_group <= 0
    ):
        raise ValueError("max_seconds_per_group must be positive and finite")
    record = json.loads((training_dir / "run.json").read_text())
    if record["selection_split"] != "validation" or record["test_evaluated"]:
        raise ValueError("Start with a validation-selected checkpoint")
    checkpoint = training_dir / f"dqn_seed{seed}.npz"
    selected = next((c for c in record["checkpoints"] if c["seed"] == seed), None)
    if selected is None or sha256(checkpoint) != selected["sha256"]:
        raise ValueError("Missing or changed selected checkpoint")
    original_hash = sha256(checkpoint)
    manifest_path = data_dir / "processed/manifest.json"
    trace = data_dir / "processed/burstgpt_windows.csv"
    manifest = json.loads(manifest_path.read_text())
    bin_seconds = manifest["range_config"]["episode_seconds"]
    hashes = dict(source_sha256=manifest["source_sha256"], trace_sha256=sha256(trace),
                  manifest_sha256=sha256(manifest_path))
    if (hashes["trace_sha256"] != manifest["selected_sha256"]
            or any(record.get(k) != v for k, v in hashes.items())):
        raise ValueError("Prepared dataset does not match checkpoint training data")
    ranges = sorted((dict(r) for r in manifest["range_config"]["ranges"] if r["split"] == "test"),
                    key=lambda r: r["start_timestamp_s"])
    available = {r["group"] for r in ranges}
    if groups is not None and (not groups or len(set(groups)) != len(groups) or not set(groups) <= available):
        raise ValueError("Unknown or duplicate test groups")
    ranges = [r for r in ranges if groups is None or r["group"] in groups]
    if not ranges or len({r["group"] for r in ranges}) != len(ranges):
        raise ValueError("Require one explicit range per test group")
    for r in ranges:
        if max_seconds_per_group is not None:
            r["end_timestamp_s"] = min(r["end_timestamp_s"], r["start_timestamp_s"] + max_seconds_per_group)
    if any(a["end_timestamp_s"] > b["start_timestamp_s"] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("Test ranges overlap")
    windows = load_windows(trace)
    source_rows = sorted((row for (split, _), rows in windows.items() if split == "test"
                          for row in rows if any(r["start_timestamp_s"] <= row.timestamp_s < r["end_timestamp_s"]
                                                for r in ranges)), key=lambda row: (row.timestamp_s, row.source_row))
    if not source_rows:
        raise ValueError("Selected traffic is empty")
    config = record["config"]
    origin, end = ranges[0]["start_timestamp_s"], ranges[-1]["end_timestamp_s"]
    replay = replay_window(source_rows, slo_s=config["slo_s"], bounds=(origin, end))
    del windows
    # Cohort labels never enter the policy or reward function.
    labels = {}
    for row in source_rows:
        r = next(r for r in ranges if r["start_timestamp_s"] <= row.timestamp_s < r["end_timestamp_s"])
        bin_index = int((row.timestamp_s - r["start_timestamp_s"]) // bin_seconds)
        labels[str(row.source_row)] = (r["group"], bin_index)
    del source_rows
    output.mkdir(parents=True)
    settings = dict(seed=seed, groups=groups, epsilon=epsilon, learning_rate=learning_rate,
                    batch_size=batch_size, warmup=warmup, update_interval=update_interval,
                    capacity=capacity, max_seconds_per_group=max_seconds_per_group, device=device)
    provenance = dict(protocol="online_prequential_continuous_v1", settings=settings,
                      selected_ranges=ranges, source_checkpoint_sha256=original_hash, **hashes,
                      source_training_record_sha256=sha256(training_dir / "run.json"),
                      interpretation="Online experiment; not frozen held-out evaluation. Idle gaps have no updates.",
                      optimizer="Fresh Adam state, fresh replay, target initialized to checkpoint weights",
                      traffic="Native timestamps; queues persist across bins and groups; omitted gaps treated as idle",
                      observation="Existing full features normalized by total replay horizon for BOTH copies",
                      learning_clock="Simulated time only; training computation is not charged as GPU service",
                      status="running")
    write_json(output / "run.json", provenance)
    summaries, bins, final = [], [], []
    for policy in ("frozen", "adapting"):
        started = time.monotonic()
        model, metadata = DQN.load(checkpoint, device=resolved_device if policy == "adapting" else "cpu")
        print(f"{policy} learner={model.learner_runtime}", flush=True)
        if any(metadata.get(k) != record.get(k) for k in (
            "config", "features", "actions", "service_profile", "reward_config", *hashes
        )):
            raise ValueError("Checkpoint metadata differs from training record")
        initial = model.copy_weights()
        model.rng = np.random.default_rng(seed)
        model.learning_rate = learning_rate
        model.gamma = config.get("agent", {}).get("gamma", 0.99)
        buffer = ReplayBuffer(capacity, seed + 1000) if policy == "adapting" else None
        env = TriageEnv(replay, HardwareConfig(**config["hardware"]),
                        ServiceProfile(**record["service_profile"]),
                        estimated_output_tokens=record["estimated_output_tokens"],
                        reward_config=RewardConfig(**record["reward_config"]),
                        **config["decisions"], **policy_options(replay, config["decisions"]))
        state, info = env.reset(seed=seed)
        steps, observed_outcomes, learner_seconds = 0, 0, 0.0
        # Stream audit to disk rather than retain a per-decision history in RAM.
        with (output / f"{policy}_updates.csv").open("w", newline="") as stream:
            audit = csv.DictWriter(stream, fieldnames=["step", "source_timestamp_s", "updates", "loss", "buffer_size", "observed_outcomes"])
            audit.writeheader()
            while not info["is_terminal"]:
                before = info["now_s"]
                action = model.act(state, info["action_mask"], epsilon if buffer is not None else 0)
                next_state, reward, done, truncated, next_info = env.step(action)
                if truncated or next_info["now_s"] < before:
                    raise RuntimeError("Noncausal or truncated replay")
                if any(event.timestamp_s > next_info["now_s"] + 1e-8 for event in next_info["events"]):
                    raise RuntimeError("Future event exposed by environment")
                if any(outcome.completion_time_s > next_info["now_s"] + 1e-8 for outcome in next_info["outcomes"]):
                    raise RuntimeError("Future outcome exposed by environment")
                steps += 1
                observed_outcomes += len(next_info["outcomes"])
                if buffer is not None:
                    # Insert ONLY AFTER step has delivered the reward and next state.
                    buffer.add(state, action, reward, next_state, done, next_info["action_mask"])
                    if buffer.size >= warmup and steps % update_interval == 0:
                        update_started = time.monotonic()
                        loss = model.train_batch(buffer.sample(batch_size))
                        learner_seconds += time.monotonic() - update_started
                        audit.writerow(dict(step=steps, source_timestamp_s=origin + next_info["now_s"],
                                            updates=model.updates, loss=loss, buffer_size=buffer.size,
                                            observed_outcomes=observed_outcomes))
                state, info = next_state, next_info
                if steps % 100000 == 0:
                    print(f"{policy}: decisions={steps}, updates={model.updates}, source_time={origin + info['now_s']:.1f}", flush=True)
        result = evaluate_events(env.events(), measurement_window_s=replay.horizon_s)
        if {r.request_id for r in result.request_records} != set(labels):
            raise RuntimeError("Unpaired request cohort")
        by_group, by_bin = {}, {}
        for row in result.request_records:
            group, bin_index = labels[row.request_id]
            by_group.setdefault(group, []).append(row)
            by_bin.setdefault((group, bin_index), []).append(row)
        for r in ranges:
            start, stop = r["start_timestamp_s"], r["end_timestamp_s"]
            score = summarize_records(by_group.get(r["group"], []), measurement_window_s=stop-start, window_start_s=start-origin)
            summaries.append(dict(policy=policy, group=r["group"], **score))
            for index in range(math.ceil((stop-start) / bin_seconds)):
                a = start + index * bin_seconds
                b = min(stop, a + bin_seconds)
                score = summarize_records(by_bin.get((r["group"], index), []), measurement_window_s=b-a, window_start_s=a-origin)
                bins.append(dict(policy=policy, group=r["group"], source_start_s=a, source_end_s=b, **score))
        change = max(float(np.max(np.abs(model.params[k] - v))) for k, v in initial.items())
        if policy == "frozen" and (model.updates or change):
            raise RuntimeError("Frozen copy was modified")
        model.save(output / f"{policy}_final.npz", dict(metadata, online_adaptation=policy == "adapting",
                   online_settings=settings, learner_runtime=model.learner_runtime))
        final.append(dict(policy=policy, decisions=steps, updates=model.updates,
                          learner_device=model.device, learner_seconds=learner_seconds,
                          observed_outcomes=observed_outcomes, max_weight_change=change,
                          wall_seconds=time.monotonic()-started, final_source_timestamp_s=origin+info["now_s"]))
        print(f"{policy} completed: {steps} decisions, {model.updates} updates", flush=True)
        del env, buffer, result, by_group, by_bin
    if sha256(checkpoint) != original_hash:
        raise RuntimeError("Source checkpoint modified")
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "timeline.csv", bins)
    write_csv(output / "learning_summary.csv", final)
    write_json(output / "run.json", dict(provenance, status="completed", policy_runs=final,
               final_checkpoints={p: sha256(output / f"{p}_final.npz") for p in ("frozen", "adapting")}))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=50000)
    parser.add_argument("--max-seconds-per-group", type=float)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
