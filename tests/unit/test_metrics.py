from __future__ import annotations

import math
import unittest

from llm_triage.metrics import RequestRecord, RequestStatus, summarize_records


class SummaryMetricsTests(unittest.TestCase):
    def test_summary_matches_hand_calculated_metrics(self) -> None:
        records = [
            RequestRecord(
                request_id="on-time",
                arrival_time_s=0.0,
                deadline_s=5.0,
                status=RequestStatus.COMPLETED,
                decision_time_s=0.0,
                service_start_time_s=0.0,
                first_token_time_s=0.5,
                completion_time_s=4.0,
            ),
            RequestRecord(
                request_id="late",
                arrival_time_s=1.0,
                deadline_s=5.0,
                status=RequestStatus.COMPLETED,
                decision_time_s=4.0,
                service_start_time_s=4.0,
                first_token_time_s=4.5,
                completion_time_s=6.0,
            ),
            RequestRecord(
                request_id="dropped",
                arrival_time_s=2.0,
                deadline_s=5.0,
                status=RequestStatus.DROPPED,
                decision_time_s=2.1,
            ),
            RequestRecord(
                request_id="unfinished",
                arrival_time_s=3.0,
                deadline_s=8.0,
                status=RequestStatus.UNFINISHED,
            ),
        ]

        result = summarize_records(
            records,
            measurement_window_s=10.0,
            max_queue_length=3,
            decision_overheads_s=(0.001, 0.002, 0.003),
            episode_end_s=10.0,
        )

        self.assertEqual(result["arrivals"], 4)
        self.assertEqual(result["completed"], 2)
        self.assertEqual(result["dropped"], 1)
        self.assertEqual(result["unfinished"], 1)
        self.assertEqual(result["slo_met"], 1)
        self.assertAlmostEqual(result["slo_attainment"], 0.25)
        self.assertAlmostEqual(result["drop_rate"], 0.25)
        self.assertAlmostEqual(result["unfinished_rate"], 0.25)
        self.assertEqual(result["first_token_count"], 2)
        self.assertAlmostEqual(result["mean_ttft_s"], 2.0)
        self.assertAlmostEqual(result["p95_ttft_s"], 3.35)
        self.assertAlmostEqual(result["mean_completion_latency_s"], 4.5)
        self.assertEqual(result["window_completed"], 2)
        self.assertAlmostEqual(result["window_throughput_rps"], 0.2)
        self.assertAlmostEqual(result["window_goodput_rps"], 0.1)
        self.assertAlmostEqual(result["total_service_s"], 6.0)
        self.assertAlmostEqual(result["wasted_service_s"], 2.0)
        self.assertEqual(result["max_queue_length"], 3)
        self.assertEqual(result["scheduler_decisions"], 3)
        self.assertAlmostEqual(result["total_scheduler_overhead_s"], 0.006)
        self.assertAlmostEqual(result["mean_scheduler_overhead_ms"], 2.0)
        self.assertAlmostEqual(result["p95_scheduler_overhead_ms"], 2.9)

    def test_dropped_requests_count_against_slo_attainment(self) -> None:
        records = [
            RequestRecord(
                request_id="success",
                arrival_time_s=0.0,
                deadline_s=3.0,
                status=RequestStatus.COMPLETED,
                service_start_time_s=0.0,
                completion_time_s=2.0,
            ),
            RequestRecord(
                request_id="drop",
                arrival_time_s=1.0,
                deadline_s=3.0,
                status=RequestStatus.DROPPED,
            ),
        ]

        result = summarize_records(records, measurement_window_s=5.0)

        self.assertAlmostEqual(result["slo_attainment"], 0.5)

    def test_empty_run_has_zero_rates_and_undefined_latency(self) -> None:
        result = summarize_records([], measurement_window_s=5.0)

        self.assertEqual(result["arrivals"], 0)
        self.assertEqual(result["slo_attainment"], 0.0)
        self.assertEqual(result["drop_rate"], 0.0)
        self.assertEqual(result["window_throughput_rps"], 0.0)
        self.assertIsNone(result["mean_ttft_s"])
        self.assertIsNone(result["p95_completion_latency_s"])

    def test_measurement_window_must_be_finite_and_positive(self) -> None:
        for duration in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                summarize_records([], measurement_window_s=duration)

    def test_rejects_inconsistent_request_timestamps(self) -> None:
        record = RequestRecord(
            request_id="bad",
            arrival_time_s=2.0,
            deadline_s=5.0,
            status=RequestStatus.COMPLETED,
            service_start_time_s=1.0,
            completion_time_s=4.0,
        )

        with self.assertRaisesRegex(ValueError, "precedes arrival"):
            summarize_records([record], measurement_window_s=10.0)

    def test_rejects_duplicate_request_ids(self) -> None:
        record = RequestRecord(
            request_id="duplicate",
            arrival_time_s=0.0,
            deadline_s=5.0,
            status=RequestStatus.DROPPED,
        )

        with self.assertRaisesRegex(ValueError, "Duplicate request_id"):
            summarize_records([record, record], measurement_window_s=10.0)


if __name__ == "__main__":
    unittest.main()
