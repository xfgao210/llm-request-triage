from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from experiments.run_baselines import main
from llm_triage.evaluation import EvaluationConfig, evaluate_policy_matrix
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import ServiceProfile, SimulatorExecutor
from llm_triage.workload import SyntheticConfig, generate_synthetic_workload


def _executor(requests, config):
    return SimulatorExecutor(
        requests,
        profile=ServiceProfile(
            prefill_overhead_s=0.0,
            prefill_rate=config.prefill_tokens_per_s,
            prefill_quadratic_s=0.0,
            decode_rate=config.decode_tokens_per_s,
            decode_context_s=0.0,
            contention_factor=0.0,
        ),
        defer_retry_s=config.defer_seconds,
        max_defers=config.max_defer_decisions,
        estimated_output_tokens=config.estimated_output_tokens,
    )


class SyntheticBaselineTests(unittest.TestCase):
    def test_workload_and_executor_are_deterministic(self) -> None:
        config = SyntheticConfig(duration_s=10.0, request_count=20, seed=7)
        requests_a = generate_synthetic_workload(config)
        requests_b = generate_synthetic_workload(config)
        self.assertEqual(requests_a, requests_b)

        executor = _executor(requests_a, config)
        policies = baseline_schedulers()
        events_a = list(executor.run("fcfs", policies["fcfs"]))
        events_b = list(executor.run("fcfs", policies["fcfs"]))
        stable = lambda event: (
            event.event_type,
            event.timestamp_s,
            event.request_id,
            {key: value for key, value in event.details.items() if key != "decision_overhead_s"},
        )
        self.assertEqual([stable(event) for event in events_a], [stable(event) for event in events_b])

    def test_all_baselines_run_on_the_same_request_cohort(self) -> None:
        config = SyntheticConfig(duration_s=10.0, request_count=20, seed=11)
        requests = generate_synthetic_workload(config)
        results = evaluate_policy_matrix(
            baseline_schedulers(),
            _executor(requests, config),
            EvaluationConfig(measurement_window_s=config.duration_s),
        )

        self.assertEqual(list(results), ["fcfs", "length_aware", "static_slo_aware"])
        self.assertTrue(
            all(result.summary["arrivals"] == 20 for result in results.values())
        )
        cohorts = [
            [record.request_id for record in result.request_records]
            for result in results.values()
        ]
        self.assertTrue(all(cohort == cohorts[0] for cohort in cohorts[1:]))

    def test_cli_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(io.StringIO()):
                exit_code = main(
                    [
                        "--duration",
                        "5",
                        "--requests",
                        "10",
                        "--output-dir",
                        directory,
                    ]
                )

            self.assertEqual(exit_code, 0)
            expected = {
                "baseline_requests.csv",
                "baseline_summary.csv",
                "baseline_config.json",
            }
            self.assertEqual({path.name for path in Path(directory).iterdir()}, expected)
            with (Path(directory) / "baseline_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
