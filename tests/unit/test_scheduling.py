from __future__ import annotations

from dataclasses import fields
import unittest

from llm_triage import scheduling
from llm_triage.scheduling import (
    FCFSScheduler,
    LengthAwareScheduler,
    StaticSLOAwareScheduler,
    baseline_schedulers,
)
from llm_triage.simulation import (
    Action,
    CandidateView,
    HardwareConfig,
    Request,
    RequestState,
    SchedulerSnapshot,
    ServiceProfile,
    simulate,
)


PROFILE = ServiceProfile(
    prefill_overhead_s=0.001,
    prefill_rate=None,
    prefill_quadratic_s=0.000001,
    decode_rate=1000,
    decode_context_s=0,
    contention_factor=0,
    default_estimated_output_tokens=1,
)
HARDWARE = HardwareConfig(max_batch_size=1, kv_block_size=16, kv_total_blocks=1024)


def _request(request_id: str, input_tokens: int, deadline: float = 10.0) -> Request:
    return Request(request_id, 0.0, input_tokens, 1, deadline)


class SchedulingTests(unittest.TestCase):
    def test_scheduling_reexports_the_simulator_contract(self) -> None:
        self.assertIs(scheduling.Action, Action)
        self.assertIs(scheduling.CandidateView, CandidateView)
        self.assertIs(scheduling.SchedulerSnapshot, SchedulerSnapshot)

    def test_fcfs_preserves_enqueue_order(self) -> None:
        result = simulate(
            [_request("first", 100), _request("second", 10)],
            FCFSScheduler(),
            HARDWARE,
            PROFILE,
            estimated_output_tokens=1,
        )
        order = sorted(result.requests, key=lambda request: request.admit_time)
        self.assertEqual([request.request_id for request in order], ["first", "second"])

    def test_length_aware_prioritizes_shortest_estimated_service(self) -> None:
        result = simulate(
            [_request("long", 100), _request("short", 10), _request("middle", 50)],
            LengthAwareScheduler(),
            HARDWARE,
            PROFILE,
            estimated_output_tokens=1,
        )
        order = sorted(result.requests, key=lambda request: request.admit_time)
        self.assertEqual(
            [request.request_id for request in order],
            ["short", "middle", "long"],
        )

    def test_slo_aware_drops_estimated_infeasible_request(self) -> None:
        result = simulate(
            [_request("urgent", 1000, deadline=0.01), _request("feasible", 10)],
            StaticSLOAwareScheduler(),
            HARDWARE,
            PROFILE,
            estimated_output_tokens=1,
        )
        by_id = result.by_id()
        self.assertIs(by_id["urgent"].state, RequestState.DROPPED)
        self.assertIs(by_id["feasible"].state, RequestState.COMPLETED)

    def test_capacity_wait_does_not_spend_slo_aware_defer_budget(self) -> None:
        result = simulate(
            [_request("resident", 32, 1), _request("big", 48, 2), _request("small", 16, 3)],
            StaticSLOAwareScheduler(),
            HardwareConfig(max_batch_size=2, kv_block_size=16, kv_total_blocks=4),
            PROFILE,
            estimated_output_tokens=1,
            max_defers=1,
        )
        self.assertEqual(len(result.completed), 3)
        self.assertEqual(result.by_id()["big"].defer_count, 0)
        self.assertGreater(result.by_id()["big"].admit_time, 0)

    def test_baseline_registry_uses_canonical_callbacks(self) -> None:
        self.assertEqual(
            list(baseline_schedulers()),
            ["fcfs", "length_aware", "static_slo_aware"],
        )
        self.assertTrue(all(hasattr(policy, "decide") for policy in baseline_schedulers().values()))

    def test_candidate_view_hides_ground_truth_output_length(self) -> None:
        visible_fields = {field.name for field in fields(CandidateView)}
        self.assertNotIn("true_output_tokens", visible_fields)
        self.assertNotIn("output_tokens", visible_fields)


if __name__ == "__main__":
    unittest.main()
