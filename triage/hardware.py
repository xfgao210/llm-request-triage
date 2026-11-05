"""Hardware configuration and paged KV-cache block management."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Union

from triage.request import Request, _check_finite, _check_int


# ---------------------------------------------------------------------------
# Hardware configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareConfig:
    """Physical serving constraints of a single GPU + model deployment.

    KV-cache uses **paged allocation**: the token budget is divided into
    fixed-size blocks (as in vLLM / SGLang).  Admission is gated on both
    ``max_batch_size`` (resident sequences) and free KV-cache blocks.
    """

    gpu_name: str = "generic"
    max_batch_size: int = 32
    kv_block_size: int = 16      # tokens per KV-cache block
    kv_total_blocks: int = 2048  # total blocks available

    def __post_init__(self) -> None:
        if not isinstance(self.gpu_name, str) or not self.gpu_name.strip():
            raise ValueError("gpu_name must be a nonempty string")
        _check_int(self.max_batch_size, "max_batch_size", min_val=1)
        _check_int(self.kv_block_size, "kv_block_size", min_val=1)
        _check_int(self.kv_total_blocks, "kv_total_blocks", min_val=1)

    @property
    def kv_total_tokens(self) -> int:
        return self.kv_total_blocks * self.kv_block_size

    @classmethod
    def from_model(
        cls,
        gpu_name: str,
        gpu_memory_gb: float,
        model_memory_gb: float,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,
        gpu_memory_utilization: float = 0.9,
        activation_reserve_gb: float = 0.0,
        max_batch_size: int = 32,
        kv_block_size: int = 16,
    ) -> "HardwareConfig":
        """Derive the KV-cache block budget from GPU memory and model shape.

        Mirrors vLLM's sizing: usable memory = ``gpu_memory_gb *
        gpu_memory_utilization`` minus weights and activation reserve;
        the remainder is KV-cache.  Per-token KV bytes =
        ``2 (K and V) * layers * kv_heads * head_dim * dtype_bytes``.
        """
        _check_finite(gpu_memory_gb, "gpu_memory_gb", strictly_positive=True)
        _check_finite(model_memory_gb, "model_memory_gb", min_val=0.0)
        _check_finite(gpu_memory_utilization, "gpu_memory_utilization", strictly_positive=True)
        if gpu_memory_utilization > 1.0:
            raise ValueError("gpu_memory_utilization must be <= 1.0")
        _check_finite(activation_reserve_gb, "activation_reserve_gb", min_val=0.0)
        _check_int(num_layers, "num_layers", min_val=1)
        _check_int(num_kv_heads, "num_kv_heads", min_val=1)
        _check_int(head_dim, "head_dim", min_val=1)
        _check_int(dtype_bytes, "dtype_bytes", min_val=1)
        _check_int(kv_block_size, "kv_block_size", min_val=1)

        kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        kv_bytes = (
            gpu_memory_gb * gpu_memory_utilization - model_memory_gb - activation_reserve_gb
        ) * 1024**3
        if kv_bytes <= 0:
            raise ValueError("No memory left for KV-cache after weights/activations")
        total_tokens = int(kv_bytes // kv_bytes_per_token)
        if total_tokens < kv_block_size:
            raise ValueError("Insufficient memory for even a single KV block")
        return cls(
            gpu_name=gpu_name,
            max_batch_size=max_batch_size,
            kv_block_size=kv_block_size,
            kv_total_blocks=total_tokens // kv_block_size,
        )


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

# Qwen2.5-0.5B: 24 layers, 2 KV heads (GQA), head_dim 64, FP16 → 12 KiB/token.
# ~1 GB of FP16 weights; 1 GB reserved for activations / CUDA context.
# Matches the GPU used for Hengyi's timing calibration (profiles/rtx5070_qwen05b.json).
RTX_5070_QWEN05B = HardwareConfig.from_model(
    gpu_name="RTX-5070-12GB/Qwen2.5-0.5B",
    gpu_memory_gb=12.0,
    model_memory_gb=1.0,
    num_layers=24,
    num_kv_heads=2,
    head_dim=64,
    activation_reserve_gb=1.0,
    max_batch_size=32,
)

# A deliberately KV-constrained config (32k tokens) for stress-testing
# memory-based admission and preemption.
KV_CONSTRAINED = HardwareConfig(
    gpu_name="kv-constrained-32k",
    max_batch_size=32,
    kv_block_size=16,
    kv_total_blocks=2048,
)


# ---------------------------------------------------------------------------
# KV-cache block allocator
# ---------------------------------------------------------------------------

class KVCacheManager:
    """Block-level KV-cache allocator, keyed by ``request_id``.

    Blocks are allocated on admission, grown immediately whenever prefill or
    decode produces a token that crosses a block boundary, and released when a
    request finishes or is preempted.
    """

    def __init__(self, config: HardwareConfig) -> None:
        self._block_size = config.kv_block_size
        self._total_blocks = config.kv_total_blocks
        self._allocations: Dict[Union[int, str], int] = {}
        self._used = 0

    # -- queries -----------------------------------------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    def blocks_needed(self, total_tokens: int) -> int:
        """Blocks required to hold *total_tokens* of KV-cache."""
        _check_int(total_tokens, "total_tokens", min_val=0)
        return math.ceil(total_tokens / self._block_size)

    def blocks_held(self, request: Request) -> int:
        return self._allocations.get(request.request_id, 0)

    def extra_blocks_needed(self, request: Request, total_tokens: int) -> int:
        """Additional blocks *request* needs to hold *total_tokens*."""
        return max(0, self.blocks_needed(total_tokens) - self.blocks_held(request))

    @property
    def blocks_used(self) -> int:
        return self._used

    @property
    def blocks_free(self) -> int:
        return self._total_blocks - self._used

    @property
    def tokens_free(self) -> int:
        return self.blocks_free * self._block_size

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def utilization(self) -> float:
        return self._used / self._total_blocks

    def can_allocate(self, total_tokens: int) -> bool:
        """``True`` if a *new* allocation of *total_tokens* fits right now."""
        return self.blocks_needed(total_tokens) <= self.blocks_free

    def fits_ever(self, total_tokens: int) -> bool:
        """``True`` if *total_tokens* could fit in an otherwise empty cache."""
        return self.blocks_needed(total_tokens) <= self._total_blocks

    # -- mutations ---------------------------------------------------------

    def allocate(self, request: Request, total_tokens: int) -> int:
        """Reserve blocks for a request not currently holding any."""
        if request.request_id in self._allocations:
            raise RuntimeError(f"request {request.request_id} already holds KV blocks")
        needed = self.blocks_needed(total_tokens)
        if needed > self.blocks_free:
            raise RuntimeError(
                f"Cannot allocate {needed} blocks for request "
                f"{request.request_id}: only {self.blocks_free} free"
            )
        self._allocations[request.request_id] = needed
        self._used += needed
        request.kv_blocks_allocated = needed
        return needed

    def update(self, request: Request, new_total_tokens: int) -> None:
        """Grow *request*'s allocation if it crossed a block boundary."""
        extra = self.extra_blocks_needed(request, new_total_tokens)
        if extra == 0:
            return
        if extra > self.blocks_free:
            raise RuntimeError(
                f"Cannot grow KV for request {request.request_id}: "
                f"need {extra} more blocks but only {self.blocks_free} free"
            )
        self._allocations[request.request_id] = self.blocks_held(request) + extra
        self._used += extra
        request.kv_blocks_allocated = self._allocations[request.request_id]

    def release(self, request: Request) -> None:
        """Free all blocks held by *request*."""
        self._used -= self._allocations.pop(request.request_id, 0)
        request.kv_blocks_allocated = 0
