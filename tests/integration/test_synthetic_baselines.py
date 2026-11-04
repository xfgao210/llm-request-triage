from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from experiments.run_baselines import main
from llm_triage.evaluation import (
    EvaluationConfig,
    SyntheticConfig,
    SyntheticExecutor,
    evaluate_policy_matrix,
    generate_synthetic_workload,
)
from llm_triage.scheduling import baseline_schedulers


class SyntheticBaselineTests(unittest.TestCase):
    def test_workload_and_executor_are_deterministic(self) -> None:
        config = SyntheticConfig(duration_s=10.0, request_count=20, seed=7)
        requests_a = generate_synthetic_workload(config)
        requests_b = generate_synthetic_workload(config)
        self.assertEqual(requests_a, requests_b)

        executor = SyntheticExecutor(requests_a, config)
        policies = baseline_schedulers()
        events_a = list(executor.run("fcfs", policies["fcfs"]))
        events_b = list(executor.run("fcfs", policies["fcfs"]))
        self.assertEqual(events_a, events_b)

    def test_all_baselines_run_on_the_same_request_cohort(self) -> None:
        config = SyntheticConfig(duration_s=10.0, request_count=20, seed=11)
        requests = generate_synthetic_workload(config)
        results = evaluate_policy_matrix(
            baseline_schedulers(),
            SyntheticExecutor(requests, config),
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
