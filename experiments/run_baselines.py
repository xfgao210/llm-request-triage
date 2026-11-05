"""Run all deterministic baselines on a synthetic workload."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Sequence

# Permit a source-checkout invocation before an editable install.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from llm_triage.evaluation import (  # noqa: E402
    EvaluationConfig,
    evaluate_policy_matrix,
    write_evaluation_artifacts,
)
from llm_triage.scheduling import baseline_schedulers  # noqa: E402
from llm_triage.simulation import ServiceProfile, SimulatorExecutor  # noqa: E402
from llm_triage.workload import SyntheticConfig, generate_synthetic_workload  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run FCFS, length-aware, and static SLO-aware policies through "
            "the shared simulator with a deterministic synthetic workload."
        )
    )
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--slo", type=float, default=3.0)
    parser.add_argument("--burst-multiplier", type=float, default=3.0)
    parser.add_argument("--prefill-rate", type=float, default=2000.0)
    parser.add_argument("--decode-rate", type=float, default=400.0)
    parser.add_argument("--estimated-output-tokens", type=float, default=144.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    synthetic_config = SyntheticConfig(
        duration_s=args.duration,
        request_count=args.requests,
        seed=args.seed,
        slo_s=args.slo,
        burst_multiplier=args.burst_multiplier,
        prefill_tokens_per_s=args.prefill_rate,
        decode_tokens_per_s=args.decode_rate,
        estimated_output_tokens=args.estimated_output_tokens,
    )
    requests = generate_synthetic_workload(synthetic_config)
    policies = baseline_schedulers()
    results = evaluate_policy_matrix(
        policies,
        SimulatorExecutor(
            requests,
            profile=ServiceProfile(
                prefill_overhead_s=0.0,
                prefill_rate=synthetic_config.prefill_tokens_per_s,
                prefill_quadratic_s=0.0,
                decode_rate=synthetic_config.decode_tokens_per_s,
                decode_context_s=0.0,
                contention_factor=0.0,
            ),
            defer_retry_s=synthetic_config.defer_seconds,
            max_defers=synthetic_config.max_defer_decisions,
            estimated_output_tokens=synthetic_config.estimated_output_tokens,
        ),
        EvaluationConfig(measurement_window_s=synthetic_config.duration_s),
    )
    resolved_config = {
        "executor": "shared_simulator",
        "warning": "Synthetic workload; not evidence for final project claims.",
        "synthetic": asdict(synthetic_config),
        "policies": list(policies),
    }
    paths = write_evaluation_artifacts(
        results,
        args.output_dir,
        resolved_config=resolved_config,
    )

    print(
        f"{'policy':<20} {'SLO':>8} {'throughput':>12} "
        f"{'goodput':>10} {'drop':>8} {'p95 TTFT':>10}"
    )
    for policy_name, result in results.items():
        summary = result.summary
        print(
            f"{policy_name:<20} "
            f"{summary['slo_attainment']:>8.3f} "
            f"{summary['window_throughput_rps']:>12.3f} "
            f"{summary['window_goodput_rps']:>10.3f} "
            f"{summary['drop_rate']:>8.3f} "
            f"{_format_optional(summary['p95_ttft_s']):>10}"
        )
    print(f"\nRequest records: {paths.requests_csv}")
    print(f"Summary:         {paths.summary_csv}")
    print(f"Configuration:   {paths.config_json}")
    return 0


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
