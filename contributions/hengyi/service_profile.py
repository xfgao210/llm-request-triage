"""Explicit assumed or measured service profiles; no inferred hardware calibration."""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ServiceProfile:
    prefill_rate: float | None = 2000.
    decode_rate: float | None = 400.
    provenance: str = 'assumed single-slot rates; not hardware calibrated'
    prefill_overhead_s: float = 0.
    decode_overhead_s: float = 0.
    prefill_quadratic_s: float = 0.
    decode_context_s: float = 0.
    decode_token_offset: int = 0

    def __post_init__(self):
        if any(value is not None and (not math.isfinite(value) or value <= 0) for value in [self.prefill_rate, self.decode_rate]):
            raise ValueError('Service rates must be finite and positive')
        if any(not math.isfinite(value) or value < 0 for value in [self.prefill_overhead_s,
               self.decode_overhead_s, self.prefill_quadratic_s, self.decode_context_s]):
            raise ValueError('Service time coefficients must be finite and nonnegative')
        if self.decode_token_offset not in [0, 1]:
            raise ValueError('Decode token offset must be 0 or 1')

    def prefill_s(self, input_tokens):
        linear = input_tokens/self.prefill_rate if self.prefill_rate is not None else 0.
        return self.prefill_overhead_s+linear+self.prefill_quadratic_s*input_tokens**2

    def decode_s(self, input_tokens, output_tokens):
        steps = max(0., output_tokens-self.decode_token_offset)
        if steps == 0:
            return 0.
        context_sum = steps*input_tokens+steps*(steps+1)/2
        linear = steps/self.decode_rate if self.decode_rate is not None else 0.
        return self.decode_overhead_s+linear+self.decode_context_s*context_sum

    def service_s(self, input_tokens, output_tokens):
        return self.prefill_s(input_tokens)+self.decode_s(input_tokens, output_tokens)

    def as_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('validation_status') == 'poor_fit_do_not_use_without_review':
            raise ValueError('Profile failed holdout fit threshold; improve measurements/model first')
        return cls(**data['service_profile'])
