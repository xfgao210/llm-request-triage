import json
from pathlib import Path
import tempfile
import unittest
from service_profile import ServiceProfile


class ProfileTests(unittest.TestCase):
    def test_default_preserves_original_formula(self):
        self.assertAlmostEqual(ServiceProfile().service_s(1000, 200), 1.)

    def test_measured_offset_and_nonlinear_cost(self):
        profile = ServiceProfile(prefill_rate=None, decode_rate=10, prefill_overhead_s=.02,
                                 prefill_quadratic_s=.000001, decode_context_s=.00001, decode_token_offset=1)
        self.assertAlmostEqual(profile.prefill_s(100), .03)
        self.assertEqual(profile.decode_s(100, 1), 0.)
        self.assertAlmostEqual(profile.decode_s(100, 3), .2+.00001*203)

    def test_invalid_rate_and_failed_fit_rejected(self):
        with self.assertRaises(ValueError):
            ServiceProfile(decode_rate=float('nan'))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'profile.json'
            path.write_text(json.dumps(dict(service_profile=ServiceProfile().as_dict(),
                validation_status='poor_fit_do_not_use_without_review')), encoding='utf-8')
            with self.assertRaises(ValueError):
                ServiceProfile.load(path)
