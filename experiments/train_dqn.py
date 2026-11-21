"""Train Double-DQN on BurstGPT train windows and compare on validation only."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import gymnasium
    GYMNASIUM_VERSION = gymnasium.__version__
except ImportError:  # pragma: no cover
    gymnasium = None
    GYMNASIUM_VERSION = "unavailable"
import numpy as np

from llm_triage.evaluation import EvaluationConfig, evaluate_policy_matrix
from llm_triage.rl import (
    ACTIONS,
    FULL_FEATURE_NAMES,
    RANKED_FEATURE_NAMES,
    ActionMaskConfig,
    ObservationConfig,
)
from llm_triage.rl import RewardConfig, RLPolicyCallback, TriageEnv
from llm_triage.rl.dqn import DQN, ReplayBuffer
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import HardwareConfig, ServiceProfile, SimulatorExecutor
from llm_triage.workload.burstgpt import load_windows, sha256
from llm_triage.workload.transforms import replay_window


BATCH_SIZE = 64
WARMUP = 256
REPLAY_CAPACITY = 50_000
UPDATE_INTERVAL = 4
METRICS = (
    "arrivals", "completed", "slo_met", "slo_attainment", "drop_rate", "unfinished_rate",
    "window_throughput_rps", "window_goodput_rps", "first_token_count", "p95_ttft_s",
)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_cohorts(processed_dir, config):
    csv_path = processed_dir / "burstgpt_windows.csv"
    manifest_path = processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(csv_path) != manifest["selected_sha256"]:
        raise ValueError("Prepared windows differ from manifest")
    windows = load_windows(csv_path)
    cohorts = {}
    for split in ("train", "validation"):
        max_windows = config.get(f"max_{split}_windows")
        split_windows = [
            (window, rows)
            for (partition, window), rows in sorted(windows.items())
            if partition == split
        ]
        if max_windows is not None:
            if type(max_windows) is not int or max_windows < 1:
                raise ValueError(f"max_{split}_windows must be a positive integer")
            split_windows = split_windows[:max_windows]
        cohorts[split] = [
            (window, rate, replay_window(rows, slo_s=config["slo_s"], target_rps=rate))
            for window, rows in split_windows
            for rate in config["target_rps"]
        ]
        if not cohorts[split]:
            raise ValueError(f"Missing {split} windows")
    train_rows = [row for (split, _), rows in windows.items() if split == "train" for row in rows]
    estimate = fmean(row.true_output_tokens for row in train_rows)
    if not math.isclose(estimate, manifest["estimated_output_tokens_train_mean"], rel_tol=1e-12):
        raise ValueError("Manifest output estimate does not match train windows")
    return cohorts, estimate, {
        "source_sha256": manifest["source_sha256"],
        "trace_sha256": manifest["selected_sha256"],
        "manifest_sha256": sha256(manifest_path),
    }


def policy_options(replay, decisions, *, ranked=False):
    if ranked:
        return dict(
            observation_config=ObservationConfig(
                preset="full_ranked",
                horizon_s=replay.horizon_s,
                max_defers=decisions["max_defers"],
                include_output_length_rank=True,
            ),
            # Reaching the kernel limit expires the request, so mask that final defer.
            mask_config=ActionMaskConfig(
                max_defers=decisions["max_defers"] - 1, defer_retry_s=decisions["defer_retry_s"],
            ),
            candidate_selector="output_length_rank",
            rank_by_output_length=True,
            invalid_action_mode="raise",
        )
    return dict(
        observation_config=ObservationConfig(
            preset="full", horizon_s=replay.horizon_s, max_defers=decisions["max_defers"],
        ),
        # Reaching the kernel limit expires the request, so mask that final defer.
        mask_config=ActionMaskConfig(
            max_defers=decisions["max_defers"] - 1, defer_retry_s=decisions["defer_retry_s"],
        ),
        invalid_action_mode="raise",
    )


def _is_ranked_model(name, model):
    features = getattr(model, "features", None)
    if features is not None:
        return tuple(features) == tuple(RANKED_FEATURE_NAMES)
    return "ranked" in str(name)


def evaluate(models, cases, hardware, profile, estimate, decisions, *, baselines=False):
    rows = []
    for window, rate, replay in cases:
        policies = baseline_schedulers() if baselines else {}
        policies.update({
            name: RLPolicyCallback(
                model,
                **policy_options(replay, decisions, ranked=_is_ranked_model(name, model)),
            )
            for name, model in models.items()
        })
        executor = SimulatorExecutor(
            replay.requests, hardware, profile, estimated_output_tokens=estimate,
            **decisions,
        )
        results = evaluate_policy_matrix(
            policies, executor, EvaluationConfig(measurement_window_s=replay.horizon_s)
        )
        expected = {str(request.request_id) for request in replay.requests}
        for name, result in results.items():
            if {record.request_id for record in result.request_records} != expected:
                raise RuntimeError("Policy did not evaluate the complete paired cohort")
            if result.summary["slo_attainment"] != executor.results[name].slo_attainment:
                raise RuntimeError("Simulator and metrics disagree on attainment")
            rows.append(dict(policy=name, window=window, target_rps=rate,
                             **{key: result.summary[key] for key in METRICS}))
    return rows


def train_seed(
    seed, config, cohorts, hardware, profile, estimate, decisions, reward, output, metadata,
    *, ranked=False,
):
    features = RANKED_FEATURE_NAMES if ranked else FULL_FEATURE_NAMES
    prefix = "dqn_ranked" if ranked else "dqn"
    model_name = f"{prefix}_seed{seed}"
    include_variant = bool(ranked or config.get("include_ranked", False))
    rng = random.Random(seed)
    model = DQN(seed=seed, features=features, **config.get("agent", {}))
    initial = model.copy_weights()
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY, seed + 1000, state_dim=len(features))
    history, selection = [], []
    total_steps = 0
    best_score, best_weights, best_episode = -math.inf, None, None

    def validate(episode):
        rows = evaluate({prefix: model}, cohorts["validation"], hardware, profile, estimate, decisions)
        score = fmean(row["slo_attainment"] for row in rows)
        weight_delta = max(float(np.max(np.abs(model.params[key] - value))) for key, value in initial.items())
        sel_row = dict(
            seed=seed, episode=episode, updates=model.updates, max_weight_change=weight_delta,
            validation_attainment=score, validation_drop_rate=fmean(row["drop_rate"] for row in rows),
        )
        if include_variant:
            sel_row = dict(variant=prefix, **sel_row)
        selection.append(sel_row)
        print(f"variant={prefix} seed={seed} episode={episode} updates={model.updates} validation={score:.4f}", flush=True)
        return score

    validate(0)
    for episode in range(1, config["episodes"] + 1):
        window, rate, replay = rng.choice(cohorts["train"])
        env = TriageEnv(
            replay, hardware, profile, estimated_output_tokens=estimate, reward_config=reward,
            **decisions, **policy_options(replay, decisions, ranked=ranked),
        )
        state, info = env.reset(seed=seed)
        if info["is_terminal"]:
            raise ValueError(f"Training window {window} has no runnable decision")
        epsilon = max(0.05, 1 - (episode - 1) / max(1, 0.8 * config["episodes"]))
        losses = []
        episode_reward = 0.0
        done = False
        while not done:
            action = model.act(state, info["action_mask"], epsilon)
            next_state, value, done, truncated, next_info = env.step(action)
            if truncated:
                raise RuntimeError("The finite trace environment truncated unexpectedly")
            replay_buffer.add(state, action, value, next_state, done, next_info["action_mask"])
            state, info = next_state, next_info
            total_steps += 1
            episode_reward += value
            if replay_buffer.size >= WARMUP and total_steps % UPDATE_INTERVAL == 0:
                losses.append(model.train_batch(replay_buffer.sample(BATCH_SIZE)))
        hist_row = dict(
            seed=seed, episode=episode, window=window, target_rps=rate, steps=total_steps,
            updates=model.updates, reward=episode_reward, loss=fmean(losses) if losses else None,
            train_attainment=info["episode_metrics"]["slo_attainment"],
            train_drop_rate=info["episode_metrics"]["drop_rate"],
        )
        if include_variant:
            hist_row = dict(variant=prefix, **hist_row)
        history.append(hist_row)
        if episode % config["validate_every"] == 0 or episode == config["episodes"]:
            score = validate(episode)
            if model.updates and selection[-1]["max_weight_change"] > 0 and score > best_score:
                best_score, best_weights, best_episode = score, model.copy_weights(), episode
    if best_weights is None:
        raise RuntimeError("Training did not change the model weights")
    model.params = best_weights
    path = output / f"{model_name}.npz"
    saved_meta = dict(
        metadata,
        features=list(features),
        seed=seed,
        episode=best_episode,
        validation_attainment=best_score,
    )
    if include_variant:
        saved_meta["variant"] = prefix
    model.save(path, saved_meta)
    restored, _ = DQN.load(path)
    if any(not np.array_equal(value, restored.params[key]) for key, value in best_weights.items()):
        raise RuntimeError("Checkpoint round trip changed the model weights")
    checkpoint_info = dict(
        seed=seed, episode=best_episode, validation_attainment=best_score, sha256=sha256(path),
    )
    if include_variant:
        checkpoint_info = dict(name=model_name, variant=prefix, **checkpoint_info)
    return restored, history, selection, checkpoint_info


def run(config, processed_dir, output, checkpoint=None, *, include_ranked=False):
    for key in ("episodes", "validate_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    seeds = config["seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Use unique nonnegative integer seeds")
    rates = config["target_rps"]
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates) or len(set(rates)) != len(rates):
        raise ValueError("Use unique positive arrival rates")
    use_ranked = bool(include_ranked or config.get("include_ranked", False))
    cohorts, estimate, provenance = load_cohorts(processed_dir, config)
    hardware = HardwareConfig(**config["hardware"])
    profile = ServiceProfile()
    decisions = config["decisions"]
    reward = RewardConfig(**config["reward"])
    # Validate optional agent parameters before creating a run directory.
    DQN(**config.get("agent", {}))
    metadata = dict(config=config, **provenance, features=list(FULL_FEATURE_NAMES),
                    actions=[a.value for a in ACTIONS], estimated_output_tokens=estimate,
                    service_profile=vars(profile), reward_config=vars(reward),
                    reward_model="llm_triage.rl.rewards.AsymmetricRewardModel")
    models, history, selection, checkpoints = {}, [], [], []
    if checkpoint is not None:
        model, saved = DQN.load(checkpoint)
        expected_meta = dict(metadata, features=list(model.features))
        if any(saved.get(key) != value for key, value in expected_meta.items()):
            raise ValueError("Checkpoint does not match the configuration, profile, or prepared trace")
        variant = saved.get("variant", "dqn_ranked" if tuple(model.features) == tuple(RANKED_FEATURE_NAMES) else "dqn")
        name = f"{variant}_seed{saved['seed']}"
        models[name] = model
        ckpt_entry = dict(seed=saved["seed"], episode=saved["episode"],
                          validation_attainment=saved["validation_attainment"], sha256=sha256(checkpoint))
        if use_ranked or variant != "dqn":
            ckpt_entry = dict(name=name, variant=variant, **ckpt_entry)
        checkpoints.append(ckpt_entry)
    output.mkdir(parents=True, exist_ok=False)
    if checkpoint is None:
        variants = (False, True) if use_ranked else (False,)
        for ranked in variants:
            prefix = "dqn_ranked" if ranked else "dqn"
            for seed in seeds:
                model, episodes, scores, selected = train_seed(
                    seed, config, cohorts, hardware, profile, estimate, decisions, reward, output, metadata,
                    ranked=ranked,
                )
                models[f"{prefix}_seed{seed}"] = model
                history.extend(episodes)
                selection.extend(scores)
                checkpoints.append(selected)
        write_csv(output / "training.csv", history)
        write_csv(output / "checkpoints.csv", selection)
    rows = evaluate(models, cohorts["validation"], hardware, profile, estimate, decisions, baselines=True)
    write_csv(output / "validation.csv", rows)
    write_json(output / "run.json", dict(
        **metadata, checkpoints=checkpoints, gymnasium=GYMNASIUM_VERSION, numpy=np.__version__,
        batch_size=BATCH_SIZE, warmup=WARMUP, replay_capacity=REPLAY_CAPACITY, update_interval=UPDATE_INTERVAL,
        train_windows=len({window for window, _, _ in cohorts["train"]}),
        validation_windows=len({window for window, _, _ in cohorts["validation"]}),
        validation_runs=len(rows), selection_split="validation", test_evaluated=False,
    ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/week2_dqn.json"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/runs/week2-dqn"))
    parser.add_argument("--checkpoint", type=Path, help="Reload one checkpoint and evaluate only")
    parser.add_argument("--include-ranked", action="store_true", help="Also train and evaluate output-length-ranked DQN")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.include_ranked:
        config["include_ranked"] = True
    rows = run(config, args.processed_dir, args.output, args.checkpoint, include_ranked=args.include_ranked)
    for policy in dict.fromkeys(row["policy"] for row in rows):
        scores = [row["slo_attainment"] for row in rows if row["policy"] == policy]
        print(f"{policy}: {fmean(scores):.2%} validation attainment ({len(scores)} runs)")
    print(f"Results: {args.output / 'validation.csv'}")


if __name__ == "__main__":
    main()
