import tempfile
from pathlib import Path
import unittest
import numpy as np
from dqn import DQN, ReplayBuffer
from rl_env import TriageEnv, run_policy
from triage_sim import Request, Action, fcfs, static_slo, expired_only, generate_workload, simulate
from workloads import replay_window, load_windows


class EnvironmentTests(unittest.TestCase):
    def test_shortest_estimated_uses_visible_prompt_only(self):
        requests = [Request(0, 0, 1, 2, 10), Request(1, 1, 8, 1, 20),
                    Request(2, 1, 1, 50, 20)]
        env = TriageEnv(requests, prefill_rate=1, decode_rate=1, queue_order='shortest_estimated')
        env.step(Action.ADMIT)
        self.assertEqual(env.observation().request_id, 2)
        env.step(Action.ADMIT)
        self.assertEqual(env.records[2]['start'], 3)
        # Huge actual output does not prevent choosing the shorter visible prompt.
        self.assertEqual(env.records[1]['status'], 'pending')

    def test_callback_and_interactive_accounting_match(self):
        requests = generate_workload(seed=12)
        for policy in [fcfs, expired_only, static_slo]:
            rows, _, expected = simulate(requests, policy)
            env = TriageEnv(requests)
            actual = run_policy(env, policy)
            for key in ['arrivals', 'completed', 'slo_met', 'dropped', 'slo_attainment',
                        'window_goodput_rps', 'wasted_service_s', 'p95_latency_s']:
                self.assertAlmostEqual(actual[key], expected[key], msg=key)
            for row in rows:
                other = env.records[row['request_id']]
                self.assertEqual(row['status'], other['status'])
                if row['completion'] is None:
                    self.assertIsNone(other['completion'])
                else:
                    self.assertAlmostEqual(row['completion'], other['completion'])

    def test_no_output_or_future_information_in_features(self):
        first = TriageEnv([Request(0, 0, 100, 1, 5), Request(1, 10, 100, 1, 15)])
        changed = TriageEnv([Request(0, 0, 100, 999, 5), Request(1, 50, 2000, 100, 55)])
        np.testing.assert_array_equal(first.features(), changed.features())
        self.assertEqual(first.observation(), changed.observation())

    def test_defer_advances_time_and_is_bounded(self):
        env = TriageEnv([Request(0, 0, 1, 1, 5)], max_defers=2)
        env.step(Action.DEFER)
        env.step(Action.DEFER)
        self.assertEqual(env.now, .5)
        self.assertFalse(env.mask()[1])
        with self.assertRaises(ValueError):
            env.step(Action.DEFER)
        env.step(Action.ADMIT)
        self.assertTrue(env.done)

    def test_all_drop_has_negative_return(self):
        requests = [Request(0, 0, 1, 1, 5), Request(1, 1, 1, 1, 6)]
        dropping = run_policy(TriageEnv(requests), lambda obs: Action.DROP)
        serving = run_policy(TriageEnv(requests), fcfs)
        self.assertLess(dropping['episode_reward'], 0)
        self.assertGreater(serving['episode_reward'], dropping['episode_reward'])
        self.assertEqual(dropping['slo_attainment'], 0)

    def test_terminal_and_reset(self):
        empty = TriageEnv([])
        self.assertTrue(empty.done)
        self.assertEqual(empty.metrics()['arrivals'], 0)
        with self.assertRaises(RuntimeError):
            empty.step(0)
        env = TriageEnv([Request(0, 0, 10, 10, 5)])
        before = env.features().copy()
        env.step(0)
        env.reset()
        np.testing.assert_array_equal(before, env.features())

    def test_replay_time_scaling_and_split_disjointness(self):
        requests, horizon, scale = replay_window([(10, 1, 2), (20, 3, 4)], target_rate=2)
        self.assertEqual(horizon, 1)
        self.assertEqual(requests[-1].arrival, .5)
        self.assertEqual(scale, .05)
        self.assertEqual(requests[1].output_tokens, 4)
        path = Path('data/processed/burstgpt_windows.csv')
        if path.exists():
            windows = load_windows(path)
            train = [r[0] for (s, _), rows in windows.items() if s == 'train' for r in rows]
            valid = [r[0] for (s, _), rows in windows.items() if s == 'validation' for r in rows]
            test = [r[0] for (s, _), rows in windows.items() if s == 'test' for r in rows]
            self.assertLess(max(train), min(valid))
            self.assertLess(max(valid), min(test))


class DQNTests(unittest.TestCase):
    def test_mask_and_checkpoint_roundtrip(self):
        model = DQN(2)
        state = np.arange(7)/7
        self.assertEqual(model.act(state, np.array([False, False, True])), 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.npz'
            model.save(path)
            loaded = DQN.load(path)
            np.testing.assert_array_equal(model.forward(state), loaded.forward(state))

    def test_terminal_reward_fit(self):
        # Terminal targets must not bootstrap, even with all-false next masks.
        model = DQN(seed=4, learning_rate=.01)
        states = np.tile(np.array([1, .2, .3, .2, .1, 0, 0]), (64, 1))
        actions = np.array([0, 2]*32)
        rewards = np.where(actions == 0, 1., -.05)
        batch = (states, actions, rewards, np.zeros_like(states), np.ones(64), np.zeros((64, 3), bool))
        initial = model.train_batch(batch)
        for _ in range(199):
            loss = model.train_batch(batch)
        self.assertLess(loss, initial/100)
        self.assertEqual(model.act(states[0], np.array([True, False, True])), 0)

    def test_replay_buffer_wraps(self):
        buffer = ReplayBuffer(capacity=3)
        for i in range(5):
            buffer.add(np.ones(7)*i, 0, i, np.zeros(7), True, np.zeros(3, bool))
        self.assertEqual(buffer.size, 3)
        self.assertEqual(set(buffer.sample(3)[2]), {2, 3, 4})


if __name__ == '__main__':
    unittest.main()
