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
from llm_triage.metrics import EventType, LifecycleEvent
from llm_triage.scheduling import (
    CandidateView,
    ResourceView,
    Scheduler,
    SchedulerSnapshot,
    baseline_schedulers,
)


class _FakeExecutor:
    def __init__(self) -> None:
        self.policy_names: list[str] = []

    def run(self, policy_name: str, scheduler: Scheduler):
        self.policy_names.append(policy_name)
        candidate = CandidateView(
            request_id="request-1",
            arrival_time_s=0.0,
            input_tokens=100,
            estimated_output_tokens=50.0,
            deadline_s=5.0,
            estimated_service_s=2.0,
        )
        snapshot = SchedulerSnapshot(
            now_s=0.0,
            candidates=(candidate,),
            resources=ResourceView(can_admit=True),
        )
        decision = scheduler.decide(snapshot)
        if decision is None:
            raise AssertionError("Baseline did not select the available request")
        return [
            LifecycleEvent(
                EventType.ARRIVAL,
                0.0,
                "request-1",
                {"deadline_s": 5.0, "input_tokens": 100, "queue_length": 1},
            ),
            LifecycleEvent(
                EventType.DECISION,
                0.0,
                "request-1",
                {"action": decision.action.value, "decision_overhead_s": 0.001},
            ),
            LifecycleEvent(EventType.SERVICE_START, 0.0, "request-1"),
            LifecycleEvent(EventType.FIRST_TOKEN, 0.5, "request-1"),
            LifecycleEvent(EventType.COMPLETION, 2.0, "request-1"),
            LifecycleEvent(EventType.EPISODE_END, 5.0),
        ]


class PolicyMatrixTests(unittest.TestCase):
    def test_runs_all_baselines_and_writes_stable_artifacts(self) -> None:
        executor = _FakeExecutor()
        results = evaluate_policy_matrix(
            baseline_schedulers(),
            executor,
            EvaluationConfig(measurement_window_s=5.0),
        )

        self.assertEqual(
            executor.policy_names,
            ["fcfs", "length_aware", "static_slo_aware"],
        )
        self.assertEqual(list(results), executor.policy_names)
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
                [row["policy"] for row in summary_rows], executor.policy_names
            )
            self.assertEqual(summary_rows[0]["slo_attainment"], "1.0")

            with paths.requests_csv.open(newline="", encoding="utf-8") as stream:
                request_rows = list(csv.DictReader(stream))
            self.assertEqual(len(request_rows), 3)
            self.assertEqual(request_rows[0]["status"], "completed")
            self.assertEqual(request_rows[0]["ttft_s"], "0.5")

            config = json.loads(Path(paths.config_json).read_text(encoding="utf-8"))
            self.assertEqual(config["seed"], 42)


if __name__ == "__main__":
    unittest.main()

