from __future__ import annotations

from dataclasses import fields
import math
import unittest

from llm_triage.rl import (
    AsymmetricRewardModel,
    EpisodeOutcome,
    RewardConfig,
    RewardEvent,
    RewardTransition,
)


class RewardModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = AsymmetricRewardModel()

    def test_intended_transition_ordering(self) -> None:
        on_time = self.model.transition_reward(
            RewardTransition(RewardEvent.ON_TIME_COMPLETION)
        )
        dropped = self.model.transition_reward(
            RewardTransition(RewardEvent.DROP)
        )
        late = self.model.transition_reward(
            RewardTransition(
                RewardEvent.LATE_COMPLETION,
                service_time_s=2.0,
                request_slo_s=5.0,
            )
        )

        self.assertGreater(on_time.total, dropped.total)
        self.assertGreater(dropped.total, late.total)

    def test_more_wasted_service_receives_larger_penalty(self) -> None:
        short_waste = self.model.transition_reward(
            RewardTransition(
                RewardEvent.LATE_COMPLETION,
                service_time_s=1.0,
                request_slo_s=5.0,
            )
        )
        long_waste = self.model.transition_reward(
            RewardTransition(
                RewardEvent.LATE_COMPLETION,
                service_time_s=4.0,
                request_slo_s=5.0,
            )
        )

        self.assertLess(long_waste.wasted_service, short_waste.wasted_service)
        self.assertLess(long_waste.total, short_waste.total)

    def test_defer_cost_accumulates_by_duration(self) -> None:
        first = self.model.transition_reward(
            RewardTransition(RewardEvent.DEFER, defer_duration_s=0.25)
        )
        second = self.model.transition_reward(
            RewardTransition(RewardEvent.DEFER, defer_duration_s=0.75)
        )
        combined = first + second
        one_second = self.model.transition_reward(
            RewardTransition(RewardEvent.DEFER, defer_duration_s=1.0)
        )

        self.assertAlmostEqual(combined.defer, one_second.defer)
        self.assertAlmostEqual(combined.total, -0.02)

    def test_drop_all_episode_is_worse_than_serving_requests(self) -> None:
        drop_all = self.model.episode_reward(
            EpisodeOutcome(
                arrivals=10,
                completed=0,
                slo_met=0,
                dropped=10,
                unfinished=0,
                duration_s=10.0,
            )
        )
        useful = self.model.episode_reward(
            EpisodeOutcome(
                arrivals=10,
                completed=8,
                slo_met=7,
                dropped=2,
                unfinished=0,
                duration_s=10.0,
            )
        )

        self.assertEqual(drop_all.terminal_drop_all, -2.0)
        self.assertLess(drop_all.total, useful.total)
        self.assertGreater(useful.terminal_completion_fraction, 0.0)

    def test_unfinished_work_reduces_terminal_reward(self) -> None:
        complete = self.model.episode_reward(
            EpisodeOutcome(10, 10, 5, 0, 0, 10.0)
        )
        unfinished = self.model.episode_reward(
            EpisodeOutcome(10, 5, 5, 0, 5, 10.0)
        )

        self.assertLess(unfinished.total, complete.total)

    def test_empty_episode_has_zero_terminal_reward(self) -> None:
        reward = self.model.episode_reward(
            EpisodeOutcome(0, 0, 0, 0, 0, 10.0)
        )

        self.assertEqual(reward.total, 0.0)

    def test_breakdown_reports_components_and_total(self) -> None:
        reward = self.model.transition_reward(
            RewardTransition(
                RewardEvent.LATE_COMPLETION,
                service_time_s=2.0,
                request_slo_s=4.0,
            )
        )
        components = reward.as_dict()

        self.assertEqual(components["late_completion"], -1.0)
        self.assertEqual(components["wasted_service"], -0.25)
        self.assertEqual(components["total"], -1.25)

    def test_transition_schema_excludes_future_and_hidden_truth(self) -> None:
        visible_fields = {field.name for field in fields(RewardTransition)}

        self.assertNotIn("true_output_tokens", visible_fields)
        self.assertNotIn("future_arrivals", visible_fields)
        self.assertNotIn("future_queue", visible_fields)

    def test_transition_rejects_irrelevant_or_missing_fields(self) -> None:
        with self.assertRaises(ValueError):
            RewardTransition(RewardEvent.LATE_COMPLETION, service_time_s=1.0)
        with self.assertRaises(ValueError):
            RewardTransition(RewardEvent.DROP, service_time_s=1.0)
        with self.assertRaises(ValueError):
            RewardTransition(RewardEvent.DEFER, defer_duration_s=0.0)

    def test_episode_rejects_inconsistent_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to arrivals"):
            EpisodeOutcome(10, 5, 1, 2, 1, 10.0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            EpisodeOutcome(10, 5, 6, 5, 0, 10.0)

    def test_reward_config_rejects_invalid_weights(self) -> None:
        for invalid in (-1.0, math.inf, math.nan, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RewardConfig(drop_penalty=invalid)

    def test_describe_returns_resolved_weights(self) -> None:
        weights = self.model.describe()

        self.assertEqual(weights["on_time_completion_reward"], 1.0)
        self.assertEqual(weights["terminal_drop_all_penalty"], 2.0)


if __name__ == "__main__":
    unittest.main()

