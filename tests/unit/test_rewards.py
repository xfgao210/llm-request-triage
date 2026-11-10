from __future__ import annotations

from dataclasses import fields
import math
import unittest

from llm_triage.rl import (
    AsymmetricRewardModel,
    RewardBreakdown,
    RewardConfig,
    RewardModel,
)
from llm_triage.simulation import (
    Action,
    HardwareConfig,
    Request,
    ServiceProfile,
    TriageKernel,
)


PROFILE = ServiceProfile(
    prefill_overhead_s=0.001,
    prefill_rate=None,
    prefill_quadratic_s=0.0,
    decode_rate=1000.0,
    decode_context_s=0.0,
    contention_factor=0.0,
)
HARDWARE = HardwareConfig(max_batch_size=4, kv_block_size=16, kv_total_blocks=64)


def request(request_id: str, *, output_tokens: int = 1, slo_s: float = 1.0) -> Request:
    return Request(request_id, 0.0, 16, output_tokens, slo_s)


class RewardModelTests(unittest.TestCase):
    def _episode(
        self,
        requests: list[Request],
        *,
        defer_retry_s: float = 0.01,
        max_defers: int = 64,
        profile: ServiceProfile = PROFILE,
        config: RewardConfig | None = None,
    ) -> tuple[TriageKernel, AsymmetricRewardModel]:
        kernel = TriageKernel(
            hardware=HARDWARE,
            profile=profile,
            defer_retry_s=defer_retry_s,
            max_defers=max_defers,
        )
        kernel.reset(requests)
        model = AsymmetricRewardModel(config, defer_retry_s=defer_retry_s)
        model.bind(kernel, defer_retry_s=defer_retry_s)
        return kernel, model

    def test_model_implements_environment_protocol(self) -> None:
        self.assertIsInstance(AsymmetricRewardModel(), RewardModel)

    def test_on_time_completion_includes_terminal_objectives(self) -> None:
        kernel, model = self._episode([request("on-time")])

        reward = model.reward_breakdown(kernel.apply(Action.ADMIT))

        self.assertEqual(reward.success, 1.0)
        self.assertEqual(reward.terminal_attainment, 1.0)
        self.assertEqual(reward.terminal_completion, 0.5)
        self.assertEqual(reward.total, 2.5)

    def test_late_completion_uses_attributed_gpu_time(self) -> None:
        kernel, model = self._episode(
            [request("late", output_tokens=4, slo_s=0.002)]
        )

        transition = kernel.apply(Action.ADMIT)
        reward = model.reward_breakdown(transition)

        self.assertEqual(reward.late, -1.0)
        self.assertAlmostEqual(transition.outcomes[0].gpu_time_s, 0.004)
        self.assertAlmostEqual(reward.wasted_service, -1.0)
        self.assertEqual(reward.terminal_completion, 0.5)
        self.assertEqual(reward.terminal_attainment, 0.0)

    def test_drop_all_guardrail(self) -> None:
        kernel, model = self._episode([request("drop")])

        reward = model.reward_breakdown(kernel.apply(Action.DROP))

        self.assertEqual(reward.drop, -0.2)
        self.assertEqual(reward.terminal_drop_all, -2.0)
        self.assertEqual(reward.total, -2.2)

    def test_unfinished_work_reduces_terminal_reward(self) -> None:
        kernel, model = self._episode(
            [request("defer-limit")], defer_retry_s=0.1, max_defers=1
        )

        reward = model.reward_breakdown(kernel.apply(Action.DEFER))

        self.assertAlmostEqual(reward.defer, -0.002)
        self.assertEqual(reward.late, -1.0)
        self.assertEqual(reward.terminal_unfinished, -0.5)
        self.assertEqual(reward.terminal_completion, 0.0)

    def test_defer_is_charged_immediately_for_one_retry_interval(self) -> None:
        slow_profile = ServiceProfile(
            prefill_overhead_s=1.0,
            prefill_rate=None,
            prefill_quadratic_s=0.0,
            decode_rate=1.0,
            decode_context_s=0.0,
            contention_factor=0.0,
        )
        kernel, model = self._episode(
            [request("deferred", slo_s=10.0), request("served", slo_s=10.0)],
            defer_retry_s=0.1,
            profile=slow_profile,
        )

        defer_transition = kernel.apply(Action.DEFER)
        defer_reward = model.reward_breakdown(defer_transition)
        other_transition = kernel.apply(Action.ADMIT)
        other_reward = model.reward_breakdown(other_transition)

        self.assertEqual(defer_transition.now_s, 0.0)
        self.assertAlmostEqual(defer_reward.defer, -0.002)
        self.assertEqual(other_transition.now_s, 1.0)
        self.assertEqual(other_reward.defer, 0.0)
        self.assertEqual(kernel.snapshot().candidate.request_id, "deferred")

    def test_repeated_defers_pay_one_bounded_cost_each(self) -> None:
        kernel, model = self._episode(
            [request("repeated", slo_s=10.0)], defer_retry_s=0.25
        )

        first = model.reward_breakdown(kernel.apply(Action.DEFER))
        second = model.reward_breakdown(kernel.apply(Action.DEFER))

        self.assertAlmostEqual(first.defer, -0.005)
        self.assertAlmostEqual(second.defer, -0.005)
        self.assertAlmostEqual(model.cumulative_breakdown.defer, -0.01)

    def test_invalid_action_is_a_separate_component(self) -> None:
        kernel, model = self._episode([request("invalid")])

        reward = model.reward_breakdown(
            kernel.apply(Action.ADMIT), invalid_action=True
        )

        self.assertEqual(reward.invalid_action, -0.1)
        self.assertEqual(reward.total, 2.4)

    def test_breakdown_addition_and_serialization(self) -> None:
        reward = RewardBreakdown(success=1.0) + RewardBreakdown(drop=-0.2)

        self.assertEqual(reward.total, 0.8)
        self.assertEqual(reward.as_dict()["total"], 0.8)

    def test_config_rejects_invalid_weights(self) -> None:
        for invalid in (-1.0, math.inf, math.nan, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RewardConfig(drop=invalid)

    def test_defer_retry_must_be_positive_and_finite(self) -> None:
        for invalid in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                AsymmetricRewardModel(defer_retry_s=invalid)

    def test_outcome_schema_excludes_true_output_length(self) -> None:
        kernel, _ = self._episode([request("safe")])
        transition = kernel.apply(Action.ADMIT)
        visible_fields = {field.name for field in fields(transition.outcomes[0])}

        self.assertNotIn("output_tokens", visible_fields)
        self.assertNotIn("true_output_tokens", visible_fields)

    def test_describe_returns_resolved_weights(self) -> None:
        weights = AsymmetricRewardModel().describe()

        self.assertEqual(weights["success"], 1.0)
        self.assertEqual(weights["terminal_drop_all"], 2.0)


if __name__ == "__main__":
    unittest.main()
