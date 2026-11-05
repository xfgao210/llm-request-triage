"""Regressions across workload, scheduler, simulator, and metrics boundaries."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_burstgpt import run_suite
from llm_triage.evaluation import EvaluationConfig, evaluate_policy_matrix
from llm_triage.metrics import EventType, LifecycleEvent, evaluate_events
from llm_triage.scheduling import Action, FCFSScheduler, baseline_schedulers
from llm_triage.workload.burstgpt import TraceRow, prepare
from llm_triage.workload.transforms import replay_window
from llm_triage.simulation import HardwareConfig, Request, ServiceProfile, SimulatorExecutor, simulate
from llm_triage.simulation.callback import EventType as SimulatorEventType, LifecycleEvent as SimulatorEvent


EXACT = ServiceProfile(prefill_overhead_s=.1, prefill_rate=None, prefill_quadratic_s=0,
                       decode_rate=10, decode_context_s=0, contention_factor=0)


class IntegratedSimulatorTests(unittest.TestCase):
    def test_lifecycle_schema_is_shared(self):
        self.assertIs(SimulatorEventType, EventType)
        self.assertIs(SimulatorEvent, LifecycleEvent)

    def test_arrivals_during_prefill_and_decode_keep_true_timestamps(self):
        requests = [Request("0", 0, 16, 3, 2), Request("1", .05, 16, 1, 2.05),
                    Request("2", .15, 16, 1, 2.15)]
        executor = SimulatorExecutor(requests, HardwareConfig(max_batch_size=1), EXACT)
        result = evaluate_policy_matrix({"fcfs": FCFSScheduler()}, executor,
                                       EvaluationConfig(measurement_window_s=.2))["fcfs"]
        records = {r.request_id: r for r in result.request_records}
        backend = executor.results["fcfs"]
        self.assertEqual([records[r.request_id].arrival_time_s for r in requests], [0, .05, .15])
        for request in backend.requests:
            self.assertAlmostEqual(records[request.request_id].ttft_s,
                                   request.first_token_time - request.arrival_time)
        times = [event.timestamp_s for event in backend.events]
        self.assertEqual(times, sorted(times))
        self.assertEqual(result.summary["arrivals"], 3)
        self.assertEqual(result.summary["completed"], 3)
        self.assertEqual(result.summary["window_completed"], 0)

    def test_deadline_can_expire_during_nonpreemptible_iteration(self):
        executor = SimulatorExecutor([Request("0", 0, 16, 2, 1), Request("1", .05, 16, 1, .06)],
                                     HardwareConfig(max_batch_size=1), EXACT)
        result = evaluate_policy_matrix({"fcfs": FCFSScheduler()}, executor,
                                       EvaluationConfig(measurement_window_s=.1))["fcfs"]
        self.assertEqual(result.summary["arrivals"], 2)
        self.assertEqual(result.summary["slo_met"], 1)

    def test_three_baselines_have_paired_requests_and_do_not_mutate_inputs(self):
        replay = replay_window([TraceRow(2, 0, 16, 3), TraceRow(10, .05, 16, 2),
                                TraceRow(11, .15, 16, 1)], slo_s=.3)
        before = tuple(replay.requests)
        results = evaluate_policy_matrix(baseline_schedulers(), SimulatorExecutor(
            replay.requests, HardwareConfig(max_batch_size=1), EXACT, estimated_output_tokens=2),
            EvaluationConfig(measurement_window_s=replay.horizon_s))
        self.assertEqual(replay.requests, before)
        for result in results.values():
            self.assertEqual({r.request_id for r in result.request_records}, {"2", "10", "11"})
            summary = result.summary
            self.assertEqual(summary["arrivals"], sum(summary[k] for k in ("completed", "dropped", "unfinished")))

    def test_fcfs_keeps_enqueue_order_for_tied_numeric_source_ids(self):
        requests = [Request("2", 0, 16, 1, 1), Request("10", 0, 16, 1, 1)]
        result = simulate(requests, FCFSScheduler(), HardwareConfig(max_batch_size=1), EXACT)
        starts = [e.request_id for e in result.events if e.event_type is EventType.SERVICE_START]
        self.assertEqual(starts, ["2", "10"])

    def test_kv_preemption_still_yields_valid_lifecycle_metrics(self):
        hardware = HardwareConfig(max_batch_size=2, kv_block_size=16, kv_total_blocks=3)
        backend = simulate([Request("0", 0, 16, 17, 10), Request("1", 0, 16, 17, 10)],
                           FCFSScheduler(), hardware, EXACT)
        self.assertGreater(backend.preemptions, 0)
        self.assertTrue(all(it.kv_blocks_used <= hardware.kv_total_blocks for it in backend.iterations))
        result = evaluate_events(backend.events, measurement_window_s=1)
        self.assertEqual(result.summary["completed"], 2)
        self.assertEqual(result.summary["first_token_count"], 2)
        self.assertEqual(result.summary["slo_attainment"], backend.slo_attainment)

    def test_drop_all_and_defer_limit_are_counted_as_failures(self):
        class Fixed:
            def __init__(self, action):
                self.action = action

            def decide(self, snapshot):
                return self.action

        requests = [Request("0", 0, 16, 1, 1)]
        for action, status in ((Action.DROP, "dropped"), (Action.DEFER, "unfinished")):
            events = SimulatorExecutor(requests, max_defers=2).run("fixed", Fixed(action))
            summary = evaluate_events(events, measurement_window_s=1).summary
            self.assertEqual(summary[status], 1)
            self.assertEqual(summary["slo_attainment"], 0)

    def test_offline_trace_to_all_baselines_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trace.csv"
            source.write_text("Timestamp,Model,Request tokens,Response tokens,Log Type\n" +
                              "\n".join(f"{i},ChatGPT,32,2,API log" for i in range(100)), encoding="utf-8")
            data = root / "data"
            (data / "raw").mkdir(parents=True)
            (data / "raw/LICENSE-BurstGPT.txt").write_text("test license fixture", encoding="utf-8")
            with patch("llm_triage.workload.burstgpt.acquire", return_value=source):
                prepare(data, requests_per_window=4, counts=dict(train=1, validation=1, test=1))
            config = dict(split="validation", target_rps=[1], slo_s=2,
                          hardware=dict(max_batch_size=2, kv_block_size=16, kv_total_blocks=32),
                          defer_retry_s=.01, max_defers=64, expire_overdue_in_queue=False)
            first = run_suite(config, data / "processed", root / "runs", root / "summary")
            stable = (root / "summary/summary.csv").read_bytes()
            second = run_suite(config, data / "processed", root / "runs", root / "summary")
            self.assertEqual(first, second)
            self.assertEqual(stable, (root / "summary/summary.csv").read_bytes())
            self.assertEqual(len(first), 3)
            self.assertEqual(len({r["cohort_sha256"] for r in first}), 1)
            self.assertTrue(all(r["first_token_count"] == 4 for r in first))
            provenance = json.loads((root / "summary/provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["request_executions"], 12)
            with (root / "summary/scheduler_overhead.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)


if __name__ == "__main__":
    unittest.main()
