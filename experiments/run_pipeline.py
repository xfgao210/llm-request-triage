"""Prepare BurstGPT, train DQN, reload checkpoints, and compare validation policies."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
from statistics import fmean

import gymnasium
import numpy as np

from experiments.train_dqn import (
    evaluate, load_cohorts, run as train, write_csv, write_json,
)
from llm_triage.rl.dqn import DQN
from llm_triage.simulation import HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import prepare, sha256


ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def run_lock(output):
    """Release the OS lock even when a process exits during a training attempt."""
    with (output / ".pipeline.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another pipeline process is using this output directory") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def training_fingerprint():
    paths = [Path(__file__), ROOT / "experiments/train_dqn.py",
             *sorted((ROOT / "src/llm_triage").rglob("*.py"))]
    return {
        path.resolve().relative_to(ROOT).as_posix(): sha256(path) for path in paths
    }


def atomic_state(path, state):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, state)
    temporary.replace(path)


def checkpoint_models(training_dir, config, provenance):
    record = json.loads((training_dir / "run.json").read_text(encoding="utf-8"))
    if record["config"] != config:
        raise ValueError("Saved training configuration differs from this run")
    for key, expected in provenance.items():
        if record.get(key) != expected:
            raise ValueError(f"Saved training data differs for {key}")
    models = {}
    for checkpoint in record["checkpoints"]:
        seed = checkpoint["seed"]
        path = training_dir / f"dqn_seed{seed}.npz"
        if sha256(path) != checkpoint["sha256"]:
            raise ValueError(f"Checkpoint checksum changed for seed {seed}")
        model, metadata = DQN.load(path)
        if metadata["config"] != config or metadata["trace_sha256"] != provenance["trace_sha256"]:
            raise ValueError(f"Checkpoint metadata changed for seed {seed}")
        models[f"dqn_seed{seed}"] = model
    if set(models) != {f"dqn_seed{seed}" for seed in config["seeds"]}:
        raise ValueError("Saved checkpoints differ from the requested seed set")
    return models


def run(config, data_dir, output, *, source=None, resume=False):
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        return run_locked(config, data_dir, output, source=source, resume=resume)


def run_locked(config, data_dir, output, *, source=None, resume=False):
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    fingerprint = training_fingerprint()
    runtime = dict(python=platform.python_version(), numpy=np.__version__, gymnasium=gymnasium.__version__)
    state_path = output / "pipeline.json"
    if resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (state["config_sha256"] != config_id or state["code_sha256"] != fingerprint
                or state["runtime"] != runtime):
            raise ValueError("Resume requires the same configuration, source code, and runtime")
    else:
        state = dict(config_sha256=config_id, config=config, code_sha256=fingerprint,
                     runtime=runtime, status="running", stage="prepare", training_complete=False)
        atomic_state(state_path, state)

    try:
        state.update(status="running", stage="prepare")
        atomic_state(state_path, state)
        processed = data_dir / "processed"
        if source is not None or not (processed / "manifest.json").exists():
            raw = data_dir / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            with run_lock(raw):
                if source is not None or not (processed / "manifest.json").exists():
                    prepare(data_dir=data_dir, source=source)
        cohorts, estimate, provenance = load_cohorts(processed, config)
        if "data" in state and state["data"] != provenance:
            raise ValueError("Prepared data changed since the previous attempt")
        state["data"] = provenance
        state["stage"] = "train"
        atomic_state(state_path, state)
        training_dir = output / "training"
        if not state["training_complete"] and training_dir.is_dir():
            # Recover if the previous process stopped between promotion and state update.
            checkpoint_models(training_dir, config, provenance)
            state["training_complete"] = True
            atomic_state(state_path, state)
        if not state["training_complete"]:
            # A failed attempt stays available for inspection and a retry starts fresh.
            attempt = Path(tempfile.mkdtemp(prefix="training-attempt-", dir=output))
            state["training_log"] = f"{attempt.name}/training.log"
            atomic_state(state_path, state)
            print(f"Training log: {output / state['training_log']}", flush=True)
            with (attempt / "training.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    train(config, processed, attempt / "training")
                checkpoint_models(attempt / "training", config, provenance)
            (attempt / "training").rename(training_dir)
            state["training_complete"] = True
            atomic_state(state_path, state)

        state["stage"] = "evaluate"
        atomic_state(state_path, state)
        models = checkpoint_models(training_dir, config, provenance)
        hardware = HardwareConfig(**config["hardware"])
        rows = evaluate(models, cohorts["validation"], hardware, ServiceProfile(),
                        estimate, config["decisions"], baselines=True)
        write_csv(output / "validation.csv", rows)
        # Every row should match the evaluation recorded by the training entry point.
        if (output / "validation.csv").read_bytes() != (training_dir / "validation.csv").read_bytes():
            raise RuntimeError("Reloaded checkpoints did not reproduce the training evaluation")
        summaries = []
        for policy in dict.fromkeys(row["policy"] for row in rows):
            values = [row for row in rows if row["policy"] == policy]
            summaries.append(dict(
                policy=policy, paired_cases=len(values),
                mean_slo_attainment=fmean(row["slo_attainment"] for row in values),
                mean_drop_rate=fmean(row["drop_rate"] for row in values),
                mean_goodput_rps=fmean(row["window_goodput_rps"] for row in values),
            ))
        write_csv(output / "summary.csv", summaries)
        state.update(stage="done", status="completed", validation_runs=len(rows),
                     checkpoints=json.loads((training_dir / "run.json").read_text(encoding="utf-8"))["checkpoints"])
        state.pop("error", None)
        atomic_state(state_path, state)
        print(f"Completed {len(rows)} paired validation runs. Results are in {output}")
        return rows
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_state(state_path, state)
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/week2_dqn.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", type=Path, help="Optional local copy of the pinned raw CSV")
    parser.add_argument("--output", type=Path, default=Path("results/runs/end-to-end"))
    parser.add_argument("--resume", action="store_true", help="Retry this run or recheck its saved checkpoints")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        print(f"Running {args.config}. Training logs will be saved under {args.output}.", flush=True)
        run(config, args.data_dir, args.output, source=args.source, resume=args.resume)
    except Exception as error:
        print(f"Pipeline failed: {error}. Check {args.output / 'pipeline.json'} and logs.", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
