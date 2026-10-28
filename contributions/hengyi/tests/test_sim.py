import unittest
from dataclasses import fields
from triage_sim import Request, Observation, Action, generate_workload, simulate, static_slo, expired_only


class SimulationTests(unittest.TestCase):
    def test_nonfinite_inputs_rejected_before_event_loop(self):
        for bad in [float('nan'), float('inf'), -1, 0]:
            with self.assertRaises(ValueError):
                generate_workload(duration=bad)
        with self.assertRaises(ValueError):
            simulate([Request(0, float('nan'), 1, 1, 5)])
        with self.assertRaises(ValueError):
            simulate([Request(0, 0, 1, float('inf'), 5)])

    def test_static_drops_predicted_miss(self):
        # True output is short, but fixed estimate predicts a miss: no oracle.
        rows, _, metrics = simulate([Request(0, 0, 1, 1, 1.5)], static_slo,
                                    prefill_rate=1, decode_rate=1, estimated_output_tokens=2)
        self.assertEqual(rows[0]['status'], 'dropped')
        self.assertEqual(metrics['total_service_s'], 0)
        # The expired-only policy still serves this unexpired request.
        self.assertEqual(simulate([Request(0, 0, 1, 1, 1.5)], expired_only,
                         prefill_rate=1, decode_rate=1)[0][0]['status'], 'completed')

    def test_bad_estimate_can_still_miss(self):
        rows, _, metrics = simulate([Request(0, 0, 1, 10, 3)], static_slo,
                                    prefill_rate=1, decode_rate=1, estimated_output_tokens=1)
        self.assertEqual(rows[0]['status'], 'completed')
        self.assertFalse(rows[0]['slo_met'])
        self.assertEqual(metrics['wasted_service_s'], 11)

    def test_expired_queue_heads_are_removed(self):
        requests = [Request(0, 0, 1, 1, 10), Request(1, 1, 1, 1, 1.5),
                    Request(2, 1, 1, 1, 10)]
        rows, _, metrics = simulate(requests, expired_only, prefill_rate=1, decode_rate=1)
        self.assertEqual(rows[1]['status'], 'dropped')
        self.assertEqual(rows[2]['start'], 2)
        self.assertEqual(metrics['slo_attainment'], 2/3)

    def test_hand_calculated_queue(self):
        # Each request takes 2s; second waits 1s and misses its 2s SLO.
        rows, _, metrics = simulate([Request(0, 0, 1, 1, 2),
                                    Request(1, 1, 1, 1, 3)],
                                   horizon=3, prefill_rate=1, decode_rate=1)
        self.assertEqual([r['start'] for r in rows], [0, 2])
        self.assertEqual([r['completion'] for r in rows], [2, 4])
        self.assertEqual(rows[1]['wait'], 1)
        self.assertEqual(metrics['slo_attainment'], .5)
        self.assertEqual(metrics['window_completed'], 1)
        self.assertEqual(metrics['drain_end_s'], 4)

    def test_no_true_output_in_observation(self):
        self.assertNotIn('output_tokens', {f.name for f in fields(Observation)})
        observations = []
        def spy(obs):
            observations.append(obs)
            return Action.ADMIT
        simulate([Request(0, 0, 100, 999, 20)], spy)
        self.assertEqual(observations[0].estimated_output_tokens, 144)

    def test_drop_counts_as_failure(self):
        _, _, metrics = simulate([Request(0, 0, 1, 1, 10)], lambda obs: Action.DROP)
        self.assertEqual(metrics['arrivals'], 1)
        self.assertEqual(metrics['dropped'], 1)
        self.assertEqual(metrics['slo_attainment'], 0)

    def test_seed_reproducibility_and_phase_boundaries(self):
        first = generate_workload()
        self.assertEqual(first, generate_workload())
        self.assertNotEqual(first, generate_workload(seed=43))
        self.assertTrue(all(0 <= r.arrival < 120 for r in first))
        counts = [sum(lo <= r.arrival < hi for r in first)
                  for lo, hi in [(0, 40), (40, 80), (80, 120)]]
        self.assertGreater(counts[1], 2*max(counts[0], counts[2]))

    def test_same_time_completion_and_arrival(self):
        rows, _, _ = simulate([Request(0, 0, 1, 1, 10), Request(1, 2, 1, 1, 10)],
                              prefill_rate=1, decode_rate=1)
        self.assertEqual(rows[1]['start'], 2)

    def test_empty_and_defer(self):
        self.assertEqual(simulate([])[2]['arrivals'], 0)
        rows, _, metrics = simulate([Request(0, 0, 1, 1, 10)], lambda obs: Action.DEFER)
        self.assertEqual(rows[0]['status'], 'deferred')
        self.assertEqual(metrics['slo_attainment'], 0)


if __name__ == '__main__':
    unittest.main()
