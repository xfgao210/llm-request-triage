"""Iteration-level service timing model for prefill and decode.

The engine executes the GPU one *iteration* at a time (as vLLM does): an
iteration is either a batched **prefill** of newly admitted requests or one
batched **decode** step that appends one token to every running request.

Defaults are the measured RTX 5070 / Qwen2.5-0.5B calibration from
``contributions/hengyi/profiles/rtx5070_qwen05b.json``.  At batch size 1 the
model reduces exactly to that calibration's ``context_quadratic`` fit; the
batch contention term is an *assumption* (the calibration was batch=1 only).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence

from triage.request import _check_finite, _check_int


@dataclass(frozen=True)
class ServiceProfile:
    """Latency model.

    Prefill iteration over sequences with prompt lengths ``n_i``::

        prefill_overhead_s + sum(n_i) / prefill_rate + prefill_quadratic_s * sum(n_i^2)

    (the linear term is omitted when ``prefill_rate`` is ``None``).  The
    prefill produces each sequence's first output token.

    Decode iteration over ``B`` sequences with resident contexts ``c_i``::

        (1 / decode_rate) * contention(B) + decode_context_s * sum(c_i)
        contention(B) = 1 + contention_factor * (B - 1) / (contention_ref_batch - 1)
    """

    # Prefill
    prefill_overhead_s: float = 0.025490567869413613
    prefill_rate: float | None = None  # tokens/s linear term; None = no linear term
    prefill_quadratic_s: float = 3.183762086121867e-08

    # Decode
    decode_rate: float = 35.849389110100596  # tokens/s per sequence at batch=1
    decode_context_s: float = 4.057988306430102e-08  # per resident context token per step

    # Policy-visible default output length estimate (never derived from ground truth)
    default_estimated_output_tokens: float = 144.0

    # Batch contention (assumed, uncalibrated)
    contention_factor: float = 0.5
    contention_ref_batch: int = 32

    provenance: str = "measured RTX 5070 / Qwen2.5-0.5B batch=1 + assumed batch contention"

    def __post_init__(self) -> None:
        _check_finite(self.prefill_overhead_s, "prefill_overhead_s", min_val=0.0)
        _check_finite(self.prefill_quadratic_s, "prefill_quadratic_s", min_val=0.0)
        _check_finite(self.decode_context_s, "decode_context_s", min_val=0.0)
        _check_finite(self.contention_factor, "contention_factor", min_val=0.0)
        _check_finite(self.decode_rate, "decode_rate", strictly_positive=True)
        _check_finite(
            self.default_estimated_output_tokens,
            "default_estimated_output_tokens",
            strictly_positive=True,
        )
        if self.prefill_rate is not None:
            _check_finite(self.prefill_rate, "prefill_rate", strictly_positive=True)
        if (
            self.prefill_overhead_s == 0.0
            and self.prefill_rate is None
            and self.prefill_quadratic_s == 0.0
        ):
            raise ValueError(
                "ServiceProfile must have at least one positive prefill timing term"
            )
        _check_int(self.contention_ref_batch, "contention_ref_batch", min_val=1)

    # ---- iteration costs (used by the engine) ----------------------------

    def prefill_iteration_time(self, prompt_lengths: Sequence[int]) -> float:
        """Duration of one batched prefill over the given prompt lengths."""
        if not prompt_lengths:
            return 0.0
        linear = sum(prompt_lengths) / self.prefill_rate if self.prefill_rate else 0.0
        quadratic = self.prefill_quadratic_s * sum(n * n for n in prompt_lengths)
        return self.prefill_overhead_s + linear + quadratic

    def contention(self, batch_size: int) -> float:
        if self.contention_ref_batch <= 1 or batch_size <= 1:
            return 1.0
        return 1.0 + self.contention_factor * (batch_size - 1) / (self.contention_ref_batch - 1)

    def decode_iteration_time(self, context_lengths: Sequence[int]) -> float:
        """Duration of one decode step over sequences with these resident contexts."""
        if not context_lengths:
            return 0.0
        base = self.contention(len(context_lengths)) / self.decode_rate
        return base + self.decode_context_s * sum(context_lengths)

    # ---- convenience & policy-visible estimation -------------------------

    def prefill_time(self, input_tokens: int) -> float:
        """Prefill time for a single request."""
        return self.prefill_iteration_time([input_tokens])

    def decode_step_time(self, batch_size: int, total_context_tokens: int = 0) -> float:
        """Decode step time for ``batch_size`` sequences (context cost optional)."""
        if batch_size <= 0:
            return 0.0
        return self.contention(batch_size) / self.decode_rate + self.decode_context_s * total_context_tokens

    def estimate_service_time(
        self,
        input_tokens: int,
        estimated_output_tokens: float | None = None,
    ) -> float:
        """Policy-visible service time estimate for ``input_tokens`` and an
        estimated output token count (never ground truth).

        Uses closed-form context sum so ``estimated_output_tokens`` may be a
        float (e.g. a training-set mean).
        """
        est_out = (
            self.default_estimated_output_tokens
            if estimated_output_tokens is None
            else float(estimated_output_tokens)
        )
        prefill = self.prefill_time(input_tokens)
        steps = max(0.0, est_out - 1.0)
        if steps <= 0.0:
            return prefill
        context_sum = steps * input_tokens + steps * (steps + 1.0) / 2.0
        decode = steps / self.decode_rate + self.decode_context_s * context_sum
        return prefill + decode

    def isolated_service_time(self, input_tokens: int, output_tokens: int) -> float:
        """Time to serve one request alone (batch=1).  Engine-internal / analysis only."""
        total = self.prefill_time(input_tokens)
        for k in range(1, output_tokens):  # prefill produced token 1
            total += self.decode_iteration_time([input_tokens + k])
        return total

    # ---- serialization ---------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "ServiceProfile":
        """Load a profile.

        Accepts either a flat dict of this class's fields or Hengyi's
        calibration format (``{"service_profile": {...}, "validation_status": ...}``).
        Unknown keys are ignored.  Profiles flagged as poor fits are rejected.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("validation_status") == "poor_fit_do_not_use_without_review":
            raise ValueError(f"{path}: profile failed holdout fit threshold")
        params = dict(data.get("service_profile", data))
        if params.pop("decode_overhead_s", 0.0):
            raise ValueError(f"{path}: non-zero decode_overhead_s is not supported")
        if params.pop("decode_token_offset", 1) != 1:
            raise ValueError(f"{path}: decode_token_offset must be 1")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)
