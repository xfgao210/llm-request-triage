"""Calibrate measured CUDA backend times, selecting model by training-only CV.

Hold out all prompt lengths 128 and 1024, including all their repeats, so final
error is measured on prompt lengths absent from fitting/model selection.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
from dataclasses import replace
import numpy as np
from service_profile import ServiceProfile


def nonnegative_lstsq(x, y):
    best = (float('inf'), np.zeros(x.shape[1]))
    # Exact active-set enumeration is tiny (at most 3 coefficients).
    for mask in itertools.product([False, True], repeat=x.shape[1]):
        selected = np.flatnonzero(mask)
        coef = np.zeros(x.shape[1])
        if len(selected):
            columns = x[:, selected]
            scales = np.maximum(np.linalg.norm(columns, axis=0), 1e-12)
            fitted = np.linalg.lstsq(columns/scales, y, rcond=None)[0]/scales
            if (fitted < -1e-12).any():
                continue
            coef[selected] = np.maximum(fitted, 0)
        loss = float(np.mean((x@coef-y)**2))
        if loss < best[0]:
            best = (loss, coef)
    return best[1]


def matrices(rows, kind):
    prompt = np.array([r['input_tokens'] for r in rows], dtype=float)
    steps = np.array([r['output_tokens']-1 for r in rows], dtype=float)
    if kind == 'through_origin':
        return prompt[:, None], steps[:, None]
    if kind == 'affine':
        return np.column_stack([np.ones_like(prompt), prompt]), np.column_stack([(steps > 0), steps])
    context = steps*prompt+steps*(steps+1)/2
    return np.column_stack([np.ones_like(prompt), prompt, prompt**2]), np.column_stack([(steps > 0), steps, context])


def fit(rows, kind):
    prefill, decode = matrices(rows, kind)
    a = nonnegative_lstsq(prefill, np.array([r['prefill_s'] for r in rows]))
    b = nonnegative_lstsq(decode, np.array([r['decode_s'] for r in rows]))
    if kind == 'through_origin':
        a, b = np.array([0., a[0], 0.]), np.array([0., b[0], 0.])
    elif kind == 'affine':
        a, b = np.append(a, 0.), np.append(b, 0.)
    return ServiceProfile(prefill_rate=1/a[1] if a[1] > 1e-15 else None,
                          decode_rate=1/b[1] if b[1] > 1e-15 else None,
                          prefill_overhead_s=float(a[0]), prefill_quadratic_s=float(a[2]),
                          decode_overhead_s=float(b[0]), decode_context_s=float(b[2]),
                          decode_token_offset=1, provenance='service-time fit; runtime provenance assigned by calibration CLI')


def errors(profile, rows):
    result = {}
    for phase in ['prefill', 'decode', 'e2e']:
        relative, absolute = [], []
        for r in rows:
            observed = r['prefill_s'] if phase == 'prefill' else r['decode_s'] if phase == 'decode' else r['prefill_s']+r['decode_s']
            predicted = profile.prefill_s(r['input_tokens']) if phase == 'prefill' else profile.decode_s(r['input_tokens'], r['output_tokens']) if phase == 'decode' else profile.service_s(r['input_tokens'], r['output_tokens'])
            if observed > 1e-9:
                relative.append(abs(predicted-observed)/observed)
                absolute.append((predicted-observed)**2)
        result[phase] = dict(mape=float(np.mean(relative)), p95_ape=float(np.percentile(relative, 95)),
                             rmse_s=float(np.sqrt(np.mean(absolute))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', nargs='+', type=Path,
                        default=[Path('results/gpu/measurements.csv'), Path('results/gpu/short/measurements.csv')])
    parser.add_argument('--output', type=Path, default=Path('profiles/rtx5070_qwen05b.json'))
    args = parser.parse_args()
    raw = []
    environments = []
    for path in args.measurements:
        environments.append(json.loads((path.parent/'environment.json').read_text(encoding='utf-8')))
        with path.open(encoding='utf-8', newline='') as stream:
            raw.extend(csv.DictReader(stream))
    environment = environments[0]
    for other in environments:
        for key in ['gpu', 'torch', 'cuda_runtime', 'transformers', 'precision', 'attention', 'batch_size', 'model']:
            if other[key] != environment[key]:
                raise ValueError(f'Mixed benchmark environments: {key}')
        if not other['all_parameters_on_cuda']:
            raise ValueError('Calibration requires verified all-CUDA model placement')
    groups = {}
    for row in raw:
        key = (int(row['input_tokens']), int(row['output_tokens']))
        groups.setdefault(key, []).append(row)
    rows = [dict(input_tokens=p, output_tokens=o,
                 prefill_s=statistics.median(float(r['prefill_s']) for r in group),
                 decode_s=statistics.median(float(r['decode_s']) for r in group))
            for (p, o), group in sorted(groups.items())]
    train = [r for r in rows if r['input_tokens'] not in [128, 1024]]
    holdout = [r for r in rows if r['input_tokens'] in [128, 1024]]
    candidates = []
    for kind in ['through_origin', 'affine', 'context_quadratic']:
        scores = []
        for length in sorted({r['input_tokens'] for r in train}):
            fitting = [r for r in train if r['input_tokens'] != length]
            validation = [r for r in train if r['input_tokens'] == length]
            e = errors(fit(fitting, kind), validation)
            scores.append(statistics.mean(e[phase]['mape'] for phase in ['prefill', 'decode']))
        candidates.append(dict(kind=kind, training_cv_mean_mape=statistics.mean(scores)))
    selected = min(candidates, key=lambda r: r['training_cv_mean_mape'])['kind']
    profile = fit(train, selected)
    provenance = (f'measured {environment["gpu"].removeprefix("NVIDIA GeForce ")} / '
                  f'{environment["model"]["model_id"].split("/")[-1]} / torch {environment["torch"]} / '
                  f'transformers {environment["transformers"]} / FP16 {environment["attention"].upper()} batch={environment["batch_size"]}')
    profile = replace(profile, provenance=provenance)
    diagnostic = errors(profile, holdout)
    status = 'acceptable_under_20pct_MAPE' if max(r['mape'] for r in diagnostic.values()) <= .2 else 'poor_fit_do_not_use_without_review'
    payload = dict(service_profile=profile.as_dict(), validation_status=status, selected_model=selected,
                   candidates=candidates, holdout_diagnostics=diagnostic, measurements=len(raw),
                   fit_shapes=len(train), holdout_shapes=len(holdout), holdout_prompt_lengths=[128, 1024],
                   calibration_domain=dict(input_tokens=[min(r['input_tokens'] for r in rows), max(r['input_tokens'] for r in rows)],
                                           output_tokens=[min(r['output_tokens'] for r in rows), max(r['output_tokens'] for r in rows)]),
                   measurement_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.measurements},
                   semantics='Medians of repeated shapes; CUDA-synchronized backend wall time. Prefill selects first output; decode models max(output-1,0) forwards.',
                   limitations='Single small model, one desktop GPU/backend, batch=1, short contexts. Outside measured length range is extrapolation. No vLLM or concurrent batching validation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
