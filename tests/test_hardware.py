"""Tests for triage.hardware: HardwareConfig and KVCacheManager."""

import pytest

from triage.hardware import RTX_5070_QWEN05B, HardwareConfig, KVCacheManager
from triage.request import Request


def _req(rid=1, inp=64, out=32):
    return Request(rid, 0.0, inp, out, 10.0)


def _kv(blocks=10, block=16):
    return KVCacheManager(HardwareConfig(kv_block_size=block, kv_total_blocks=blocks))


class TestHardwareConfig:
    def test_kv_total_tokens(self):
        assert HardwareConfig(kv_block_size=16, kv_total_blocks=100).kv_total_tokens == 1600

    def test_frozen(self):
        with pytest.raises(AttributeError):
            HardwareConfig().max_batch_size = 64  # type: ignore[misc]

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError):
            HardwareConfig(kv_total_blocks=0)
        with pytest.raises(ValueError):
            HardwareConfig(max_batch_size=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            HardwareConfig(gpu_name="")

    def test_from_model_sizing(self):
        # 1 layer, 1 head, dim 64, fp16 → 2*1*1*64*2 = 256 bytes/token.
        # (1 GiB * 1.0 − 0.5 GiB) / 256 B = 2^21 tokens → / 16 = 131072 blocks.
        hw = HardwareConfig.from_model("x", 1.0, 0.5, 1, 1, 64, gpu_memory_utilization=1.0)
        assert hw.kv_total_blocks == 131072

    def test_from_model_rejects_no_memory(self):
        with pytest.raises(ValueError):
            HardwareConfig.from_model("x", 1.0, 2.0, 1, 1, 64)

    def test_rtx_preset_is_sane(self):
        # 12 KiB/token for Qwen2.5-0.5B; several GB of cache → hundreds of thousands of tokens
        assert 100_000 < RTX_5070_QWEN05B.kv_total_tokens < 2_000_000


class TestKVCacheManager:
    def test_blocks_needed_rounds_up(self):
        kv = _kv()
        assert [kv.blocks_needed(n) for n in (1, 16, 17, 32, 33)] == [1, 1, 2, 2, 3]

    def test_allocate_and_release(self):
        kv, r = _kv(10), _req(inp=48)
        assert kv.allocate(r, 48) == 3
        assert (kv.blocks_used, kv.blocks_free, r.kv_blocks_allocated) == (3, 7, 3)
        kv.release(r)
        assert (kv.blocks_used, kv.blocks_free, r.kv_blocks_allocated) == (0, 10, 0)

    def test_release_is_idempotent(self):
        kv, r = _kv(), _req()
        kv.release(r)
        kv.allocate(r, 16)
        kv.release(r)
        kv.release(r)
        assert kv.blocks_used == 0

    def test_can_allocate_and_fits_ever(self):
        kv = _kv(2)
        assert kv.can_allocate(32) and not kv.can_allocate(33)
        kv.allocate(_req(1), 16)
        assert not kv.can_allocate(32) and kv.fits_ever(32)
        assert not kv.fits_ever(33)

    def test_allocate_fails_when_full(self):
        with pytest.raises(RuntimeError, match="Cannot allocate"):
            _kv(1).allocate(_req(inp=32), 32)

    def test_double_allocate_rejected(self):
        kv, r = _kv(), _req()
        kv.allocate(r, 16)
        with pytest.raises(RuntimeError):
            kv.allocate(r, 16)

    def test_update_grows_only_at_block_boundaries(self):
        kv, r = _kv(10), _req(inp=16)
        kv.allocate(r, 16)
        kv.update(r, 16)
        assert kv.blocks_used == 1
        kv.update(r, 17)
        assert kv.blocks_used == 2
        kv.update(r, 32)
        assert kv.blocks_used == 2
        kv.update(r, 33)
        assert kv.blocks_used == 3 and r.kv_blocks_allocated == 3

    def test_update_fails_when_full(self):
        kv, r = _kv(1), _req(inp=16)
        kv.allocate(r, 16)
        assert kv.extra_blocks_needed(r, 17) == 1
        with pytest.raises(RuntimeError, match="Cannot grow"):
            kv.update(r, 17)

    def test_utilization_multiple_requests(self):
        kv = _kv(4)
        r1, r2 = _req(1), _req(2)
        kv.allocate(r1, 16)
        kv.allocate(r2, 32)
        assert kv.utilization == pytest.approx(0.75)
        kv.release(r1)
        assert kv.blocks_used == 2
