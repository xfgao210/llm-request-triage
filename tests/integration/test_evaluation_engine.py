from __future__ import annotations

import unittest

from llm_triage.metrics import EventType, LifecycleEvent, evaluate_events


def _event(
    event_type: EventType,
    timestamp_s: float,
    request_id: str | None = None,
    **details: object,
) -> LifecycleEvent:
    return LifecycleEvent(event_type, timestamp_s, request_id, details)


class EvaluationEngineIntegrationTests(unittest.TestCase):
    def test_event_stream_produces_records_and_summary(self) -> None:
        events = [
            _event(
                EventType.ARRIVAL,
                0.0,
                "r1",
                deadline_s=5.0,
                input_tokens=100,
                queue_length=1,
            ),
            _event(
                EventType.DECISION,
                0.0,
                "r1",
                action="admit",
                decision_overhead_s=0.001,
            ),
            _event(EventType.SERVICE_START, 0.0, "r1"),
            _event(EventType.ARRIVAL, 1.0, "r2", deadline_s=4.0, queue_length=1),
            _event(EventType.FIRST_TOKEN, 1.5, "r1"),
            _event(EventType.COMPLETION, 3.0, "r1", queue_length=1),
            _event(EventType.DECISION, 3.0, "r2", action="drop"),
            _event(EventType.DROP, 3.0, "r2", queue_length=0),
            _event(EventType.ARRIVAL, 4.0, "r3", deadline_s=9.0, queue_length=1),
            _event(EventType.EPISODE_END, 10.0),
        ]

        result = evaluate_events(events, measurement_window_s=10.0)

        self.assertEqual(
            [record.request_id for record in result.request_records],
            ["r1", "r2", "r3"],
        )
        self.assertEqual(
            [record.status.value for record in result.request_records],
            ["completed", "dropped", "unfinished"],
        )
        self.assertEqual(result.summary["arrivals"], 3)
        self.assertAlmostEqual(result.summary["slo_attainment"], 1 / 3)
        self.assertAlmostEqual(result.summary["mean_ttft_s"], 1.5)
        self.assertEqual(result.summary["max_queue_length"], 1)
        self.assertAlmostEqual(result.summary["mean_scheduler_overhead_ms"], 1.0)

    def test_collector_rejects_nonchronological_events(self) -> None:
        events = [
            _event(EventType.ARRIVAL, 2.0, "r1", deadline_s=5.0),
            _event(EventType.DECISION, 1.0, "r1", action="drop"),
        ]

        with self.assertRaisesRegex(ValueError, "chronological"):
            evaluate_events(events, measurement_window_s=10.0, episode_end_s=10.0)

    def test_collector_rejects_events_for_unknown_requests(self) -> None:
        events = [_event(EventType.DROP, 1.0, "missing")]

        with self.assertRaisesRegex(ValueError, "unknown request"):
            evaluate_events(events, measurement_window_s=10.0, episode_end_s=10.0)


if __name__ == "__main__":
    unittest.main()
