"""Unit and integration tests for llm_triage.rl (Week 2 RL Environment API)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm_triage.evaluation import EvaluationConfig, evaluate_policy_matrix
from llm_triage.rl import (
    ACTIONS,
    ACTION_TO_INDEX,
    ActionMaskConfig,
    AsymmetricRewardModel,
    COMPACT_7_FEATURE_NAMES,
    FULL_FEATURE_NAMES,
    ObservationBuilder,
    ObservationConfig,
    RLPolicyCallback,
    RewardConfig,
    TriageEnv,
    assert_no_ground_truth_leakage,
    compute_action_mask,
    resolve_action,
    run_policy,
    sanitize_request_view,
    sanitize_snapshot,
)
from llm_triage.simulation import (
    Action,
    Decision,
    FCFSCallback,
    HardwareConfig,
    Request,
    RequestState,
    RequestView,
    SimState,
    SimulatorExecutor,
    Transition,
    simulate,
)

try:
    from llm_triage.simulation import ServiceProfile
except ImportError:
    from llm_triage.service import ServiceProfile


PROFILE = ServiceProfile(
    prefill_overhead_s=0.001,
    prefill_rate=None,
    prefill_quadratic_s=0.0,
    decode_rate=1000.0,
    decode_context_s=0.0,
    contention_factor=0.0,
)
HW = HardwareConfig(
    gpu_name="test-gpu",
    max_batch_size=4,
    kv_block_size=16,
    kv_total_blocks=64,
)


def make_req(
    rid: int | str,
    arrival: float = 0.0,
    inp: int = 16,
    out: int = 4,
    slo: float = 1.0,
) -> Request:
    return Request(
        request_id=rid,
        arrival_time=arrival,
        input_tokens=inp,
        output_tokens=out,
        slo_deadline=arrival + slo,
    )


def _as_list(vec: object) -> list[float]:
    return [float(x) for x in vec]  # type: ignore[attr-defined]


def _as_bool_list(vec: object) -> list[bool]:
    return [bool(x) for x in vec]  # type: ignore[attr-defined]


class TestOutputLengthAndFutureMasking(unittest.TestCase):
    """Verify that true output lengths and future arrivals never leak to the agent."""

    def test_observation_and_mask_ignore_ground_truth_output_and_future_arrivals(self) -> None:
        env_short = TriageEnv(
            [
                make_req("r0", arrival=0.0, inp=64, out=1, slo=2.0),
                make_req("r1", arrival=5.0, inp=64, out=1, slo=2.0),
            ],
            hardware=HW,
            profile=PROFILE,
        )
        env_long = TriageEnv(
            [
                make_req("r0", arrival=0.0, inp=64, out=9999, slo=2.0),
                make_req("r1", arrival=50.0, inp=2048, out=5000, slo=10.0),
            ],
            hardware=HW,
            profile=PROFILE,
        )

        self.assertEqual(_as_list(env_short.features()), _as_list(env_long.features()))
        self.assertEqual(_as_bool_list(env_short.mask()), _as_bool_list(env_long.mask()))
        self.assertEqual(env_short.observation(), env_long.observation())

        obs_view = env_long.observation()
        self.assertIsNotNone(obs_view)
        self.assertFalse(hasattr(obs_view, "output_tokens"))
        self.assertFalse(hasattr(obs_view, "true_output_tokens"))

    def test_leakage_guard_rejects_output_tokens_and_future_candidates(self) -> None:
        env = TriageEnv([make_req("r0", arrival=0.0, inp=32, out=8, slo=2.0)], hardware=HW, profile=PROFILE)
        snap = env.snapshot()
        self.assertIsNotNone(snap)
        assert snap is not None

        # Injecting a leaked attribute onto a candidate view subclass must fail validation
        class LeakedView(RequestView):
            pass

        leaked = LeakedView(
            request_id="r0",
            arrival_time=0.0,
            input_tokens=32,
            slo_deadline=2.0,
            slo_slack=2.0,
            time_in_queue=0.0,
            state=RequestState.QUEUED,
            tokens_generated=0,
            defer_count=0,
            preemption_count=0,
        )
        object.__setattr__(leaked, "output_tokens", 123)
        with self.assertRaisesRegex(ValueError, "forbidden ground-truth attribute"):
            sanitize_request_view(leaked)

        # Future arrival in snapshot must fail validation
        future_cand = RequestView(
            request_id="future",
            arrival_time=10.0,
            input_tokens=32,
            slo_deadline=12.0,
            slo_slack=12.0,
            time_in_queue=-10.0,
            state=RequestState.QUEUED,
            tokens_generated=0,
            defer_count=0,
            preemption_count=0,
        )
        bad_snap = SimState(
            now=0.0,
            candidate=future_cand,
            queue_length=1,
            queue_requests=(future_cand,),
            active_count=0,
            active_requests=(),
            pending_prefill_count=0,
            pending_prefill_tokens=0,
            max_batch_size=4,
            batch_utilization=0.0,
            kv_block_size=16,
            kv_blocks_used=0,
            kv_blocks_total=64,
            kv_utilization=0.0,
            completed_count=0,
            violated_count=0,
            dropped_count=0,
            preempted_count=0,
        )
        with self.assertRaisesRegex(ValueError, "future arrival_time_s"):
            assert_no_ground_truth_leakage(bad_snap)

    def test_length_aware_selector_uses_visible_prompt_not_true_output(self) -> None:
        # r1 has larger prompt (512) but tiny output (1); r2 has tiny prompt (16) but huge output (500).
        # Length-aware candidate selector must pick r2 before r1 because true output is masked.
        reqs = [
            make_req("r1", arrival=0.0, inp=512, out=1, slo=5.0),
            make_req("r2", arrival=0.0, inp=16, out=500, slo=5.0),
        ]
        env = TriageEnv(
            reqs,
            hardware=HardwareConfig(max_batch_size=1, kv_block_size=16, kv_total_blocks=128),
            profile=ServiceProfile(),
            candidate_selector="length_aware",
        )
        first_cand = env.observation()
        self.assertIsNotNone(first_cand)
        assert first_cand is not None
        self.assertEqual(first_cand.request_id, "r2")

    def test_observation_presets_and_space_bounds(self) -> None:
        reqs = [
            make_req("r0", arrival=0.0, inp=64, out=10, slo=1.0),
            make_req("r1", arrival=0.0, inp=128, out=10, slo=2.0),
        ]
        full_env = TriageEnv(reqs, hardware=HW, profile=PROFILE, observation_preset="full")
        obs_full, info_full = full_env.reset()
        self.assertEqual(len(_as_list(obs_full)), len(FULL_FEATURE_NAMES))
        self.assertTrue(full_env.observation_space.contains(obs_full))
        self.assertEqual(len(_as_bool_list(info_full["action_mask"])), 3)

        compact_env = TriageEnv(reqs, hardware=HW, profile=PROFILE, observation_preset="compact_7")
        obs_compact, _ = compact_env.reset()
        self.assertEqual(len(_as_list(obs_compact)), len(COMPACT_7_FEATURE_NAMES))
        self.assertTrue(compact_env.observation_space.contains(obs_compact))


class TestActionMaskingAndInvalidActionModes(unittest.TestCase):
    """Verify feasibility masking and configurable invalid_action_mode behavior."""

    def test_admit_masked_when_candidate_exceeds_currently_free_kv_blocks(self) -> None:
        # Total KV blocks = 4 (64 tokens).
        # r0 uses 3 blocks (48 prompt tokens), leaving 1 free block (16 tokens).
        # At t=0.0, r1 (32 prompt tokens -> needs 2 blocks) and r2 (16 prompt tokens -> needs 1 block) wait.
        hw_small = HardwareConfig(max_batch_size=4, kv_block_size=16, kv_total_blocks=4)
        reqs = [
            make_req("r0", arrival=0.0, inp=48, out=5, slo=1.0),
            make_req("r1", arrival=0.0, inp=32, out=5, slo=1.0),
            make_req("r2", arrival=0.0, inp=16, out=5, slo=1.0),
        ]
        env = TriageEnv(reqs, hardware=hw_small, profile=PROFILE)
        # Step 1: admit r0 (fits in 3 blocks)
        self.assertEqual(_as_bool_list(env.mask()), [True, True, True])
        env.step(Action.ADMIT)

        # Next candidate in FCFS order is r1 (needs 2 blocks, only 1 block free!)
        cand = env.observation()
        assert cand is not None
        self.assertEqual(cand.request_id, "r1")
        mask = _as_bool_list(env.mask())
        self.assertEqual(mask, [False, True, True])  # ADMIT is masked out!

    def test_defer_masked_after_max_defers_or_expired_slack(self) -> None:
        env = TriageEnv(
            [make_req("r0", arrival=0.0, inp=16, out=2, slo=1.0)],
            hardware=HW,
            profile=PROFILE,
            defer_retry_s=0.1,
            max_defers=3,
        )
        # Defer twice: defer_count becomes 2 (< max_defers=3), so DEFER still allowed if slack > 0.1
        env.step(Action.DEFER)
        self.assertAlmostEqual(env.now, 0.1)
        self.assertTrue(_as_bool_list(env.mask())[ACTION_TO_INDEX[Action.DEFER]])

        # When slo_slack <= defer_retry_s, DEFER is masked out
        env_tight = TriageEnv(
            [make_req("r0", arrival=0.0, inp=16, out=2, slo=0.05)],
            hardware=HW,
            profile=PROFILE,
            defer_retry_s=0.1,
            max_defers=5,
        )
        self.assertEqual(_as_bool_list(env_tight.mask()), [True, False, True])

    def test_invalid_action_mode_raise(self) -> None:
        env = TriageEnv(
            [make_req("r0", arrival=0.0, inp=16, out=2, slo=0.05)],
            hardware=HW,
            profile=PROFILE,
            defer_retry_s=0.1,
            invalid_action_mode="raise",
        )
        # DEFER is masked because slo (0.05) <= defer_retry_s (0.1)
        self.assertFalse(_as_bool_list(env.mask())[1])
        with self.assertRaisesRegex(ValueError, "Invalid or masked action"):
            env.step(Action.DEFER)

    def test_invalid_action_mode_fallback(self) -> None:
        # Masked DEFER with positive estimated slack falls back to ADMIT without raising
        env = TriageEnv(
            [make_req("r0", arrival=0.0, inp=16, out=2, slo=0.05)],
            hardware=HW,
            profile=PROFILE,
            defer_retry_s=0.1,
            invalid_action_mode="fallback",
        )
        _, reward, terminated, _, info = env.step(Action.DEFER)
        self.assertTrue(terminated)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["requested_action"], "defer")
        self.assertEqual(info["executed_action"], "admit")
        self.assertEqual(env.invalid_action_count, 1)
        # Completed on time (0.002s <= 0.05s) with no invalid_action_penalty in "fallback" mode
        self.assertAlmostEqual(
            reward,
            env.reward_config.success
            + env.reward_config.terminal_attainment
            + env.reward_config.terminal_completion,
        )

    def test_invalid_action_mode_penalize(self) -> None:
        reward_cfg = RewardConfig(success=1.0, terminal_attainment=1.0, invalid_action_penalty=0.25)
        env = TriageEnv(
            [make_req("r0", arrival=0.0, inp=16, out=2, slo=0.05)],
            hardware=HW,
            profile=PROFILE,
            defer_retry_s=0.1,
            reward_config=reward_cfg,
            invalid_action_mode="penalize",
        )
        _, reward, terminated, _, info = env.step(Action.DEFER)
        self.assertTrue(terminated)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["requested_action"], "defer")
        self.assertEqual(info["executed_action"], "admit")
        self.assertAlmostEqual(
            reward,
            reward_cfg.success
            + reward_cfg.terminal_attainment
            + reward_cfg.terminal_completion
            - reward_cfg.invalid_action_penalty,
        )


class TestKernelEquivalenceAndCallbacks(unittest.TestCase):
    """Verify TriageEnv shares exact transitions with simulate() and SimulatorExecutor."""

    def test_env_stepping_matches_simulate_outcomes_and_events(self) -> None:
        reqs = [
            make_req(f"req-{i}", arrival=i * 0.001, inp=16 + (i % 4) * 16, out=3 + (i % 5), slo=0.025)
            for i in range(15)
        ]
        hw_cfg = HardwareConfig(max_batch_size=2, kv_block_size=16, kv_total_blocks=16)

        sim_res = simulate(reqs, FCFSCallback(), hw_cfg, PROFILE)
        env = TriageEnv(reqs, hardware=hw_cfg, profile=PROFILE, invalid_action_mode="fallback")
        obs, info = env.reset()
        while not env.done:
            mask = _as_bool_list(info["action_mask"])
            action = Action.ADMIT if mask[0] else Action.DEFER
            obs, _, terminated, _, info = env.step(action)
            if terminated:
                break

        env_res = env.result()
        self.assertEqual(
            [(r.request_id, r.state, r.first_token_time, r.completion_time) for r in sim_res.requests],
            [(r.request_id, r.state, r.first_token_time, r.completion_time) for r in env_res.requests],
        )
        self.assertEqual(len(sim_res.events), len(env.events()))

    def test_rl_policy_callback_integrates_with_simulate_and_evaluation_matrix(self) -> None:
        reqs = [
            make_req("r0", arrival=0.0, inp=16, out=50, slo=0.2),
            make_req("r1", arrival=0.001, inp=16, out=4, slo=0.01),
            make_req("r2", arrival=0.002, inp=16, out=4, slo=0.054),
        ]
        hw_single = HardwareConfig(max_batch_size=1, kv_block_size=16, kv_total_blocks=16)

        # Policy that drops requests whose slack ratio (index 0) is negative, else admits
        def slack_triage_fn(obs: object, mask: object) -> int:
            vals = _as_list(obs)
            m = _as_bool_list(mask)
            if vals[0] < 0.0:
                return ACTION_TO_INDEX[Action.DROP]
            return ACTION_TO_INDEX[Action.ADMIT] if m[0] else ACTION_TO_INDEX[Action.DEFER]

        callback = RLPolicyCallback(slack_triage_fn, candidate_selector="fcfs")
        sim_res = simulate(reqs, callback, hw_single, PROFILE)
        by_id = sim_res.by_id()
        # r1 waited behind r0 (which took 0.05s) so at t=0.05s r1's deadline (0.011s) had passed -> dropped!
        # Dropping r1 lets r2 finish at t=0.054s <= deadline 0.056s (whereas FCFS serves r1 and causes r2 to violate!).
        self.assertEqual(by_id["r0"].state, RequestState.COMPLETED)
        self.assertEqual(by_id["r1"].state, RequestState.DROPPED)
        self.assertEqual(by_id["r2"].state, RequestState.COMPLETED)

        # Also works through evaluate_policy_matrix + SimulatorExecutor
        executor = SimulatorExecutor(reqs, hardware=hw_single, profile=PROFILE)
        matrix_results = evaluate_policy_matrix(
            {"fcfs": FCFSCallback(), "rl_triage": callback},
            executor,
            EvaluationConfig(measurement_window_s=1.0),
        )
        self.assertEqual(matrix_results["rl_triage"].summary["dropped"], 1)
        self.assertEqual(matrix_results["rl_triage"].summary["slo_met"], 2)
        self.assertEqual(matrix_results["fcfs"].summary["slo_met"], 1)

    def test_rl_policy_callback_supports_act_and_sb3_predict_interfaces(self) -> None:
        class MockDQNAgent:
            def act(self, obs: object, mask: object) -> int:
                m = _as_bool_list(mask)
                return 0 if m[0] else 2

        class MockSB3Agent:
            def predict(self, obs: object, action_masks: object = None, deterministic: bool = True):
                m = _as_bool_list(action_masks) if action_masks is not None else [True, True, True]
                return (0 if m[0] else 2), None

        reqs = [make_req("r0", arrival=0.0, inp=16, out=3, slo=1.0)]
        res_dqn = simulate(reqs, RLPolicyCallback(MockDQNAgent()), HW, PROFILE)
        res_sb3 = simulate(reqs, RLPolicyCallback(MockSB3Agent()), HW, PROFILE)
        self.assertEqual(res_dqn.completed[0].request_id, "r0")
        self.assertEqual(res_sb3.completed[0].request_id, "r0")


class TestRewardsAndEpisodeLifecycle(unittest.TestCase):
    """Verify reward structure, custom RewardModel, reset options, and empty workloads."""

    def test_drop_all_has_negative_return_and_serving_beats_dropping(self) -> None:
        reqs = [
            make_req("r0", arrival=0.0, inp=16, out=4, slo=1.0),
            make_req("r1", arrival=0.01, inp=16, out=4, slo=1.0),
        ]
        drop_metrics = run_policy(
            TriageEnv(reqs, hardware=HW, profile=PROFILE),
            lambda obs, mask: Action.DROP,
        )
        serve_metrics = run_policy(
            TriageEnv(reqs, hardware=HW, profile=PROFILE),
            lambda obs, mask: Action.ADMIT,
        )
        self.assertLess(float(drop_metrics["episode_reward"] or 0.0), 0.0)
        self.assertGreater(
            float(serve_metrics["episode_reward"] or 0.0),
            float(drop_metrics["episode_reward"] or 0.0),
        )
        self.assertEqual(drop_metrics["slo_attainment"], 0.0)
        self.assertEqual(serve_metrics["slo_attainment"], 1.0)

    def test_custom_reward_model_protocol(self) -> None:
        class ConstantReward:
            def reward(self, transition: Transition) -> float:
                return 0.5 if transition.decision.action is Action.ADMIT else -0.25

        env = TriageEnv(
            [make_req("r0", arrival=0.0, inp=16, out=2, slo=1.0)],
            hardware=HW,
            profile=PROFILE,
            reward_model=ConstantReward(),
        )
        _, reward, terminated, _, _ = env.step(Action.ADMIT)
        self.assertTrue(terminated)
        self.assertAlmostEqual(reward, 0.5)

    def test_empty_workload_and_reset_options(self) -> None:
        empty_env = TriageEnv([], hardware=HW, profile=PROFILE)
        self.assertTrue(empty_env.done)
        self.assertEqual(_as_bool_list(empty_env.mask()), [False, False, False])
        self.assertEqual(empty_env.metrics()["arrivals"], 0)
        with self.assertRaises(RuntimeError):
            empty_env.step(Action.ADMIT)

        # Supplying new requests via reset(options={"requests": ...}) starts a live episode
        obs, info = empty_env.reset(options={"requests": [make_req("r0", inp=16, out=2, slo=1.0)]})
        self.assertFalse(empty_env.done)
        self.assertEqual(_as_bool_list(info["action_mask"]), [True, True, True])
        _, _, terminated, _, end_info = empty_env.step(0)
        self.assertTrue(terminated)
        self.assertIn("episode_metrics", end_info)

    def test_late_arrivals_beyond_default_horizon_and_replay_window_support(self) -> None:
        # Issue 1: requests arriving at or after default horizon_s (120.0s) must not crash terminal step() or metrics()
        late_reqs = [
            make_req("r0", arrival=0.0, inp=16, out=2, slo=1.0),
            make_req("r1", arrival=250.0, inp=16, out=2, slo=1.0),
        ]
        env = TriageEnv(late_reqs, hardware=HW, profile=PROFILE)
        env.step(Action.ADMIT)
        _, _, terminated, _, info = env.step(Action.ADMIT)
        self.assertTrue(terminated)
        self.assertEqual(info["episode_metrics"]["arrivals"], 2)
        self.assertEqual(env.metrics()["completed"], 2)

        # ReplayWindow object is accepted directly and sets horizon_s automatically
        from llm_triage.workload.burstgpt import TraceRow
        from llm_triage.workload.transforms import replay_window

        rows = [
            TraceRow(source_row=0, timestamp_s=10.0, input_tokens=32, true_output_tokens=8),
            TraceRow(source_row=1, timestamp_s=20.0, input_tokens=64, true_output_tokens=16),
            TraceRow(source_row=2, timestamp_s=30.0, input_tokens=48, true_output_tokens=12),
        ]
        replay = replay_window(rows, slo_s=2.0, target_rps=0.5)
        replay_env = TriageEnv(replay, hardware=HW, profile=PROFILE)
        self.assertAlmostEqual(replay_env.horizon_s, replay.horizon_s)
        self.assertAlmostEqual(replay_env.observation_builder.config.horizon_s, replay.horizon_s)

    def test_reset_horizon_option_updates_observation_builder(self) -> None:
        # Issue 2: updating horizon_s in reset(options=...) must update observation_builder.config.horizon_s
        reqs = [
            make_req("r0", arrival=0.0, inp=16, out=2, slo=2.0),
            make_req("r1", arrival=10.0, inp=16, out=2, slo=2.0),
        ]
        env = TriageEnv(reqs, hardware=HW, profile=PROFILE, horizon_s=100.0)
        env.step(Action.ADMIT)
        obs_100 = _as_list(env.features())
        # At t=10.0 with horizon=100.0, episode_progress (last feature) is 10 / 100 = 0.10
        self.assertAlmostEqual(obs_100[-1], 0.10, places=5)

        env.reset(options={"horizon_s": 20.0})
        self.assertAlmostEqual(env.horizon_s, 20.0)
        self.assertAlmostEqual(env.observation_builder.config.horizon_s, 20.0)
        env.step(Action.ADMIT)
        obs_20 = _as_list(env.features())
        # At t=10.0 with horizon=20.0, episode_progress (last feature) is 10 / 20 = 0.50
        self.assertAlmostEqual(obs_20[-1], 0.50, places=5)

    def test_standalone_reward_model_distinguishes_late_violations_without_bind(self) -> None:
        # Issue 3: standalone AsymmetricRewardModel.reward(transition) without bind(kernel)
        # must penalize late completions using transition.outcomes and award terminal attainment.
        from llm_triage.simulation import TriageKernel

        kernel = TriageKernel(hardware=HW, profile=PROFILE)
        # Request takes 0.004s (1ms prefill + 3ms decode) > slo 0.002s -> VIOLATED!
        kernel.reset([make_req("r_late", arrival=0.0, inp=16, out=4, slo=0.002)])
        tr = kernel.apply(Action.ADMIT)
        self.assertTrue(tr.done)
        self.assertEqual(len(tr.outcomes), 1)
        self.assertFalse(tr.outcomes[0].slo_met)
        self.assertGreater(tr.outcomes[0].gpu_time_s, 0.0)

        standalone_rm = AsymmetricRewardModel(
            RewardConfig(
                success=1.0,
                late=0.1,
                wasted_service=0.2,
                terminal_attainment=1.0,
                terminal_completion=0.0,
            )
        )
        r_val = standalone_rm.reward(tr)
        # Expected: -0.1 - 0.2 * (0.004 / 0.002) = -0.5 (and terminal attainment is 0.0)
        self.assertAlmostEqual(r_val, -0.5, places=6)

    def test_hengyi_keyword_aliases_and_candidate_selector_protocol_compat(self) -> None:
        # Issues 4 & 5: service_profile, horizon, defer_seconds, queue_order aliases,
        # plus CandidateSelector, CompositeScheduler, and SyntheticExecutor compatibility.
        reqs = [
            make_req("r0", arrival=0.0, inp=512, out=4, slo=2.0),
            make_req("r1", arrival=0.0, inp=16, out=4, slo=2.0),
        ]
        env = TriageEnv(
            reqs,
            hardware=HW,
            service_profile=ServiceProfile(),
            horizon=60.0,
            defer_seconds=0.05,
            queue_order="shortest_estimated",
        )
        self.assertAlmostEqual(env.horizon_s, 60.0)
        self.assertAlmostEqual(env.defer_retry_s, 0.05)
        first_cand = env.observation()
        assert first_cand is not None
        self.assertEqual(first_cand.request_id, "r1")

        # If Week 1 scheduling selectors/CompositeScheduler/SyntheticExecutor exist on this branch, verify compatibility
        try:
            from llm_triage.evaluation.synthetic import (
                SyntheticConfig,
                SyntheticExecutor,
                generate_synthetic_workload,
            )
            from llm_triage.scheduling.base import CompositeScheduler
            from llm_triage.scheduling.selectors import ShortestEstimatedServiceSelector
        except ImportError:
            return

        env_with_selector = TriageEnv(
            reqs,
            hardware=HW,
            profile=ServiceProfile(),
            candidate_selector=ShortestEstimatedServiceSelector(),
        )
        cand = env_with_selector.observation()
        assert cand is not None
        self.assertEqual(cand.request_id, "r1")

        rl_cb = RLPolicyCallback(
            lambda obs, mask: 0 if _as_bool_list(mask)[0] else 2,
            candidate_selector=ShortestEstimatedServiceSelector(),
        )
        composite = CompositeScheduler(selector=ShortestEstimatedServiceSelector(), policy=rl_cb)
        syn_cfg = SyntheticConfig(duration_s=5.0, request_count=8, seed=7)
        syn_reqs = generate_synthetic_workload(syn_cfg)
        syn_exec = SyntheticExecutor(syn_reqs, syn_cfg)
        syn_results = evaluate_policy_matrix(
            {"rl_direct": rl_cb, "rl_composite": composite},
            syn_exec,
            EvaluationConfig(measurement_window_s=syn_cfg.duration_s),
        )
        self.assertEqual(
            syn_results["rl_direct"].summary["completed"],
            syn_results["rl_composite"].summary["completed"],
        )


if __name__ == "__main__":
    unittest.main()
