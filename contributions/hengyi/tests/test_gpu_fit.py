import unittest
import numpy as np
from calibrate_gpu import nonnegative_lstsq, fit
from service_profile import ServiceProfile


class GPUFitTests(unittest.TestCase):
    def test_nonnegative_fit_rejects_negative_slope(self):
        x = np.array([[1., 1.], [1., 2.], [1., 3.]])
        fitted = nonnegative_lstsq(x, np.array([3., 2., 1.]))
        self.assertAlmostEqual(fitted[0], 2.)
        self.assertEqual(fitted[1], 0.)

    def test_context_profile_recovery_from_synthetic_fixture(self):
        # Synthetic numerical fixture only, not hardware data.
        truth = ServiceProfile(prefill_rate=30000, decode_rate=50, prefill_overhead_s=.02,
                               prefill_quadratic_s=1e-8, decode_context_s=1e-8, decode_token_offset=1)
        rows = [dict(input_tokens=p, output_tokens=o, prefill_s=truth.prefill_s(p), decode_s=truth.decode_s(p, o))
                for p in [32, 64, 256, 512, 2048] for o in [1, 8, 32, 128]]
        predicted = fit(rows, 'context_quadratic')
        for p, o in [(128, 16), (1024, 64)]:
            self.assertAlmostEqual(predicted.service_s(p, o), truth.service_s(p, o), places=10)


if __name__ == '__main__':
    unittest.main()
