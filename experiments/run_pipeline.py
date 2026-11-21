"""Prepare BurstGPT, train DQN, reload checkpoints, and compare validation policies."""

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import traceback
from statistics import fmean

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

from experiments.train_dqn import (
    evaluate, load_cohorts, run as train, write_csv, write_json,
)
from llm_triage.rl.dqn import DQN
from llm_triage.simulation import HardwareConfig, ServiceProfile
from llm_triage.workload.burstgpt import prepare, sha256


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
        name = checkpoint.get("name", f"dqn_seed{seed}")
        path = training_dir / f"{name}.npz"
        if sha256(path) != checkpoint["sha256"]:
            raise ValueError(f"Checkpoint checksum changed for seed {seed}")
        model, metadata = DQN.load(path)
        if metadata["config"] != config or metadata["trace_sha256"] != provenance["trace_sha256"]:
            raise ValueError(f"Checkpoint metadata changed for seed {seed}")
        models[name] = model
    expected_names = {f"dqn_seed{seed}" for seed in config["seeds"]}
    if config.get("include_ranked"):
        expected_names |= {f"dqn_ranked_seed{seed}" for seed in config["seeds"]}
    if set(models) != expected_names:
        raise ValueError("Saved checkpoints differ from the requested seed set")
    return models


def build_comparison_summary(rows):
    """Build a 4-way comparison of FCFS, BurstGPT length-aware, RL policy, and RL policy + ranking."""
    fcfs_rows = [r for r in rows if r["policy"] == "fcfs"]
    fcfs_attainment = fmean(float(r["slo_attainment"]) for r in fcfs_rows) if fcfs_rows else 0.0

    groups = [
        ("fcfs", "FCFS", lambda p: p == "fcfs"),
        ("length_aware", "BurstGPT Length-Aware", lambda p: p == "length_aware"),
        ("rl_policy", "RL Policy (Repo DQN)", lambda p: p.startswith("dqn_seed")),
        ("rl_policy_ranked", "RL Policy + Output-Length Ranking", lambda p: p.startswith("dqn_ranked_seed")),
    ]
    comparison = []
    for group_key, label, predicate in groups:
        matched = [r for r in rows if predicate(str(r["policy"]))]
        if not matched:
            continue
        constituent_policies = list(dict.fromkeys(str(r["policy"]) for r in matched))
        per_policy_attainment = [
            fmean(float(r["slo_attainment"]) for r in matched if str(r["policy"]) == p)
            for p in constituent_policies
        ]
        mean_att = fmean(float(r["slo_attainment"]) for r in matched)
        ttfts = [
            float(r["p95_ttft_s"])
            for r in matched
            if r.get("p95_ttft_s") is not None and str(r.get("p95_ttft_s")) != ""
        ]
        comparison.append(
            dict(
                policy_group=group_key,
                policy_label=label,
                policies=",".join(constituent_policies),
                paired_cases=len(matched),
                mean_slo_attainment=mean_att,
                best_seed_slo_attainment=max(per_policy_attainment),
                delta_vs_fcfs_pp=100.0 * (mean_att - fcfs_attainment),
                mean_drop_rate=fmean(float(r["drop_rate"]) for r in matched),
                mean_unfinished_rate=fmean(float(r["unfinished_rate"]) for r in matched),
                mean_goodput_rps=fmean(float(r["window_goodput_rps"]) for r in matched),
                mean_throughput_rps=fmean(float(r["window_throughput_rps"]) for r in matched),
                mean_p95_ttft_s=fmean(ttfts) if ttfts else None,
            )
        )
    return comparison


def make_cpu_test_config(config, *, include_ranked=None):
    """Return a lightweight copy of *config* suitable for a fast local CPU smoke test."""
    smoke = json.loads(json.dumps(config))
    seeds = smoke.get("seeds", [7])
    smoke["seeds"] = [seeds[0] if seeds else 7]
    smoke["episodes"] = 2
    smoke["validate_every"] = 1
    rates = smoke.get("target_rps", [1.2])
    smoke["target_rps"] = [rates[-1] if rates else 1.2]
    smoke["max_train_windows"] = 4
    smoke["max_validation_windows"] = 2
    if include_ranked is not None:
        smoke["include_ranked"] = bool(include_ranked)
    return smoke


def run(config, data_dir, output, *, source=None, resume=False):
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        return run_locked(config, data_dir, output, source=source, resume=resume)


def run_locked(config, data_dir, output, *, source=None, resume=False):
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    fingerprint = training_fingerprint()
    runtime = dict(python=platform.python_version(), numpy=np.__version__, gymnasium=GYMNASIUM_VERSION)
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
        comparison = build_comparison_summary(rows)
        if comparison:
            write_csv(output / "comparison.csv", comparison)
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


def run_training_loop(
    *,
    cpu_test_mode=False,
    include_ranked=True,
    config=ROOT / "configs/week2_dqn.json",
    data_dir=ROOT / "data",
    output=None,
    source=None,
    resume=False,
    overwrite=True,
):
    """Notebook- and script-callable entry point for the full training and comparison loop.

    Parameters
    ----------
    cpu_test_mode:
        When ``True``, runs a fast subset of training (1 seed, 2 episodes,
        4 train windows, 2 validation windows, 1 arrival rate) for local CPU
        smoke testing. When ``False``, runs the full training configuration.
    include_ranked:
        When ``True`` (default), trains both the baseline repo RL policy
        (``dqn_seed*``) and the output-length-ranked RL policy
        (``dqn_ranked_seed*``) and compares them against ``fcfs`` and
        ``length_aware``.
    config:
        Path to a JSON config file or a configuration dictionary.
    data_dir:
        Directory for raw/processed BurstGPT trace data.
    output:
        Output directory for checkpoints and CSV summaries. Defaults to
        ``results/runs/cpu-smoke`` when ``cpu_test_mode=True`` or
        ``results/runs/full-comparison`` when ``cpu_test_mode=False``.
    source:
        Optional local path to ``BurstGPT_1.csv``.
    resume:
        Resume an interrupted run in *output*.
    overwrite:
        When ``True`` (default) and ``resume=False``, clears *output* if it
        already exists so notebook cells can be re-executed cleanly.
    """
    if isinstance(config, (str, Path)):
        resolved_config = json.loads(Path(config).read_text(encoding="utf-8"))
    else:
        resolved_config = json.loads(json.dumps(config))

    if include_ranked:
        resolved_config["include_ranked"] = True
    if cpu_test_mode:
        resolved_config = make_cpu_test_config(resolved_config, include_ranked=include_ranked)

    data_path = Path(data_dir)
    if output is None:
        default_name = "cpu-smoke" if cpu_test_mode else "full-comparison"
        output_path = ROOT / "results/runs" / default_name
    else:
        output_path = Path(output)

    if overwrite and not resume and output_path.exists():
        shutil.rmtree(output_path)

    source_path = None if source is None else Path(source)
    validation_rows = run(
        resolved_config,
        data_path,
        output_path,
        source=source_path,
        resume=resume,
    )

    def _read_csv_rows(csv_file):
        if not csv_file.is_file():
            return []
        with csv_file.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    summary_rows = _read_csv_rows(output_path / "summary.csv")
    comparison_rows = build_comparison_summary(validation_rows)
    training_rows = _read_csv_rows(output_path / "training" / "training.csv")
    checkpoint_rows = _read_csv_rows(output_path / "training" / "checkpoints.csv")

    return dict(
        output_dir=output_path,
        config=resolved_config,
        cpu_test_mode=bool(cpu_test_mode),
        validation_rows=validation_rows,
        summary_rows=summary_rows,
        comparison_rows=comparison_rows,
        training_rows=training_rows,
        checkpoint_rows=checkpoint_rows,
    )


run_comparison_pipeline = run_training_loop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/week2_dqn.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--source", type=Path, help="Optional local copy of the pinned raw CSV")
    parser.add_argument("--output", type=Path, default=Path("results/runs/end-to-end"))
    parser.add_argument("--resume", action="store_true", help="Retry this run or recheck its saved checkpoints")
    parser.add_argument("--cpu-test", action="store_true", help="Run a small subset of training for local CPU smoke testing")
    parser.add_argument("--include-ranked", action="store_true", help="Also train and compare the output-length-ranked RL policy")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.include_ranked:
        config["include_ranked"] = True
    if args.cpu_test:
        config = make_cpu_test_config(config, include_ranked=args.include_ranked or config.get("include_ranked"))
    try:
        print(f"Running {args.config}. Training logs will be saved under {args.output}.", flush=True)
        run(config, args.data_dir, args.output, source=args.source, resume=args.resume)
    except Exception as error:
        print(f"Pipeline failed: {error}. Check {args.output / 'pipeline.json'} and logs.", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
