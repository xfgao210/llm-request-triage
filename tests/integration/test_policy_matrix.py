from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from llm_triage.evaluation import (
    EvaluationConfig,
    evaluate_policy_matrix,
    write_evaluation_artifacts,
)
from llm_triage.scheduling import baseline_schedulers
from llm_triage.simulation import HardwareConfig, Request, ServiceProfile, SimulatorExecutor


class PolicyMatrixTests(unittest.TestCase):
    def test_runs_all_baselines_and_writes_stable_artifacts(self) -> None:
        executor = SimulatorExecutor(
            [Request("request-1", 0.0, 100, 5, 5.0)],
            hardware=HardwareConfig(max_batch_size=1),
            profile=ServiceProfile(
                prefill_overhead_s=0.1,
                prefill_rate=None,
                prefill_quadratic_s=0.0,
                decode_rate=10,
                decode_context_s=0.0,
                contention_factor=0.0,
            ),
            estimated_output_tokens=5,
        )
        results = evaluate_policy_matrix(
            baseline_schedulers(),
            executor,
            EvaluationConfig(measurement_window_s=5.0),
        )

        self.assertEqual(
            list(executor.results),
            ["fcfs", "length_aware", "static_slo_aware"],
        )
        self.assertEqual(list(results), list(executor.results))
        self.assertTrue(
            all(result.summary["slo_attainment"] == 1.0 for result in results.values())
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = write_evaluation_artifacts(
                results,
                directory,
                resolved_config={"seed": 42, "measurement_window_s": 5.0},
            )

            self.assertEqual(paths.requests_csv.name, "baseline_requests.csv")
            self.assertEqual(paths.summary_csv.name, "baseline_summary.csv")
            self.assertEqual(paths.config_json.name, "baseline_config.json")
            self.assertTrue(all(path.exists() for path in paths.__dict__.values()))

            with paths.summary_csv.open(newline="", encoding="utf-8") as stream:
                summary_rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["policy"] for row in summary_rows], list(executor.results)
            )
            self.assertEqual(summary_rows[0]["slo_attainment"], "1.0")

            with paths.requests_csv.open(newline="", encoding="utf-8") as stream:
                request_rows = list(csv.DictReader(stream))
            self.assertEqual(len(request_rows), 3)
            self.assertEqual(request_rows[0]["status"], "completed")
            self.assertEqual(request_rows[0]["ttft_s"], "0.1")

            config = json.loads(Path(paths.config_json).read_text(encoding="utf-8"))
            self.assertEqual(config["seed"], 42)


if __name__ == "__main__":
    unittest.main()
