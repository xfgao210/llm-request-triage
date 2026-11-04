from __future__ import annotations

from dataclasses import fields
import unittest

from llm_triage.scheduling import (
    Action,
    CandidateView,
    FCFSScheduler,
    LengthAwareScheduler,
    ResourceView,
    SchedulerSnapshot,
    StaticSLOAwareScheduler,
)


def _candidate(
    request_id: str,
    *,
    arrival: float,
    service: float,
    deadline: float,
) -> CandidateView:
    return CandidateView(
        request_id=request_id,
        arrival_time_s=arrival,
        input_tokens=100,
        estimated_output_tokens=50.0,
        deadline_s=deadline,
        estimated_service_s=service,
    )


def _snapshot(*candidates: CandidateView, can_admit: bool = True) -> SchedulerSnapshot:
    return SchedulerSnapshot(
        now_s=3.0,
        candidates=tuple(candidates),
        resources=ResourceView(can_admit=can_admit),
    )


class SchedulingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _candidate("first", arrival=0.0, service=5.0, deadline=12.0)
        self.short = _candidate("short", arrival=1.0, service=1.0, deadline=12.0)
        self.urgent = _candidate("urgent", arrival=2.0, service=3.0, deadline=5.0)

    def test_fcfs_selects_earliest_arrival(self) -> None:
        decision = FCFSScheduler().decide(
            _snapshot(self.short, self.urgent, self.first)
        )

        self.assertEqual(decision.request_id, "first")
        self.assertIs(decision.action, Action.ADMIT)

    def test_length_aware_selects_shortest_estimated_service(self) -> None:
        decision = LengthAwareScheduler().decide(
            _snapshot(self.first, self.short, self.urgent)
        )

        self.assertEqual(decision.request_id, "short")
        self.assertIs(decision.action, Action.ADMIT)

    def test_length_aware_defers_when_resources_are_unavailable(self) -> None:
        decision = LengthAwareScheduler().decide(
            _snapshot(self.first, self.short, can_admit=False)
        )

        self.assertEqual(decision.request_id, "short")
        self.assertIs(decision.action, Action.DEFER)

    def test_slo_aware_drops_lowest_negative_slack(self) -> None:
        decision = StaticSLOAwareScheduler().decide(
            _snapshot(self.first, self.short, self.urgent)
        )

        self.assertEqual(decision.request_id, "urgent")
        self.assertIs(decision.action, Action.DROP)

    def test_slo_aware_admits_feasible_lowest_slack(self) -> None:
        less_urgent = _candidate(
            "less-urgent", arrival=2.0, service=1.0, deadline=6.0
        )
        decision = StaticSLOAwareScheduler().decide(
            _snapshot(self.first, self.short, less_urgent)
        )

        self.assertEqual(decision.request_id, "less-urgent")
        self.assertIs(decision.action, Action.ADMIT)

    def test_slo_aware_defers_feasible_request_without_capacity(self) -> None:
        decision = StaticSLOAwareScheduler().decide(
            _snapshot(self.first, self.short, can_admit=False)
        )

        self.assertEqual(decision.request_id, "first")
        self.assertIs(decision.action, Action.DEFER)

    def test_ties_are_broken_by_arrival_then_request_id(self) -> None:
        later = _candidate("c", arrival=2.0, service=1.0, deadline=10.0)
        earlier_b = _candidate("b", arrival=1.0, service=1.0, deadline=10.0)
        earlier_a = _candidate("a", arrival=1.0, service=1.0, deadline=10.0)

        decision = LengthAwareScheduler().decide(
            _snapshot(later, earlier_b, earlier_a)
        )

        self.assertEqual(decision.request_id, "a")
        self.assertEqual(earlier_a.arrival_time_s, 1.0)

    def test_empty_candidate_set_returns_no_decision(self) -> None:
        snapshot = SchedulerSnapshot(
            now_s=3.0,
            candidates=(),
            resources=ResourceView(can_admit=True),
        )

        self.assertIsNone(FCFSScheduler().decide(snapshot))
        self.assertIsNone(LengthAwareScheduler().decide(snapshot))
        self.assertIsNone(StaticSLOAwareScheduler().decide(snapshot))

    def test_candidate_view_cannot_carry_true_output_length(self) -> None:
        visible_fields = {field.name for field in fields(CandidateView)}

        self.assertNotIn("true_output_tokens", visible_fields)
        self.assertNotIn("output_tokens", visible_fields)


if __name__ == "__main__":
    unittest.main()
