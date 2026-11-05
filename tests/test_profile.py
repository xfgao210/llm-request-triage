"""Tests for triage.profile: timing model and calibration loading."""

import json
from pathlib import Path

import pytest

from llm_triage.simulation import FCFSCallback, HardwareConfig, Request, ServiceProfile, simulate

HENGYI_PROFILE = (
    Path(__file__).resolve().parents[1] / "contributions/hengyi/profiles/rtx5070_qwen05b.json"
)


def hengyi_service_s(sp: dict, inp: int, out: int) -> float:
    """Reference: Hengyi's batch=1 context_quadratic model (service_profile.py)."""
    linear_p = inp / sp["prefill_rate"] if sp["prefill_rate"] else 0.0
    prefill = sp["prefill_overhead_s"] + linear_p + sp["prefill_quadratic_s"] * inp**2
    steps = max(0, out - sp["decode_token_offset"])
    if steps == 0:
        return prefill
    ctx = steps * inp + steps * (steps + 1) / 2
    decode = sp["decode_overhead_s"] + steps / sp["decode_rate"] + sp["decode_context_s"] * ctx
    return prefill + decode


class TestCalibration:
    @pytest.mark.skipif(not HENGYI_PROFILE.exists(), reason="calibration file not present")
    def test_loads_hengyi_profile(self):
        p = ServiceProfile.from_json(HENGYI_PROFILE)
        assert p.decode_rate == pytest.approx(35.849389110100596)
        assert p.prefill_rate is None

    @pytest.mark.skipif(not HENGYI_PROFILE.exists(), reason="calibration file not present")
    @pytest.mark.parametrize("inp,out", [(32, 1), (128, 16), (1024, 64), (2048, 128)])
    def test_batch1_matches_calibration(self, inp, out):
        sp = json.loads(HENGYI_PROFILE.read_text())["service_profile"]
        p = ServiceProfile.from_json(HENGYI_PROFILE)
        assert p.isolated_service_time(inp, out) == pytest.approx(hengyi_service_s(sp, inp, out))

    @pytest.mark.parametrize("inp,out", [(64, 1), (256, 20)])
    def test_engine_single_request_matches_isolated_time(self, inp, out):
        p = ServiceProfile()
        r = simulate([Request(1, 0.0, inp, out, 100.0)], FCFSCallback(), HardwareConfig(), p)
        assert r.requests[0].completion_time == pytest.approx(p.isolated_service_time(inp, out))

    def test_rejects_poor_fit(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"validation_status": "poor_fit_do_not_use_without_review",
                                 "service_profile": {}}))
        with pytest.raises(ValueError):
            ServiceProfile.from_json(f)

    def test_flat_roundtrip(self, tmp_path):
        p = ServiceProfile(contention_factor=0.3, prefill_rate=5000.0)
        f = tmp_path / "p.json"
        f.write_text(json.dumps(p.to_dict()))
        assert ServiceProfile.from_json(f) == p


class TestTiming:
    def test_contention_increases_step_time(self):
        p = ServiceProfile(decode_context_s=0.0, contention_factor=0.5, contention_ref_batch=32)
        t1, t16, t32 = (p.decode_step_time(b) for b in (1, 16, 32))
        assert t1 < t16 < t32
        assert t32 == pytest.approx(1.5 * t1)

    def test_no_contention(self):
        p = ServiceProfile(contention_factor=0.0)
        assert p.decode_step_time(1) == p.decode_step_time(32)

    def test_decode_context_cost(self):
        p = ServiceProfile(decode_context_s=1e-6, contention_factor=0.0)
        assert p.decode_iteration_time([1000, 1000]) - p.decode_iteration_time([10, 10]) == \
            pytest.approx(1e-6 * 1980)

    def test_batched_prefill_cheaper_than_sequential(self):
        p = ServiceProfile()
        assert p.prefill_iteration_time([100, 100]) < 2 * p.prefill_time(100)

    def test_empty_iterations(self):
        p = ServiceProfile()
        assert p.prefill_iteration_time([]) == 0.0 and p.decode_iteration_time([]) == 0.0

    def test_validation(self):
        with pytest.raises(ValueError):
            ServiceProfile(decode_rate=0.0)
        with pytest.raises(ValueError):
            ServiceProfile(prefill_overhead_s=-1.0)
        with pytest.raises(ValueError):
            ServiceProfile(decode_rate=float("nan"))
        with pytest.raises(ValueError):
            ServiceProfile(
                prefill_overhead_s=0.0,
                prefill_rate=None,
                prefill_quadratic_s=0.0,
            )
