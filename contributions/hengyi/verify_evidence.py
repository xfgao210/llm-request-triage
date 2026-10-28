"""Verify retained evidence without GPU/model/dataset downloads."""
import csv
import hashlib
import json
from pathlib import Path
from service_profile import ServiceProfile


def read(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    metadata = json.loads(Path('profiles/rtx5070_qwen05b.json').read_text(encoding='utf-8'))
    profile = ServiceProfile.load('profiles/rtx5070_qwen05b.json')
    timings = []
    for name, expected in metadata['measurement_sha256'].items():
        path = Path(name.replace('\\', '/'))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Measurement hash mismatch: {path}')
        timings.extend(read(path))
    if len(timings) != metadata['measurements']:
        raise ValueError('Measurement count mismatch')
    if not all(int(r['decode_forward_steps']) == int(r['output_tokens'])-1 for r in timings):
        raise ValueError('First-token convention mismatch')
    gpu = json.loads(Path('results/gpu/environment.json').read_text(encoding='utf-8'))
    if not gpu['all_parameters_on_cuda'] or gpu['cpu_offload']:
        raise ValueError('Recorded model was not all-CUDA')
    for row in read('results/gpu/live_queue_checked/metrics.csv'):
        records = read(f'results/gpu/live_queue_checked/{row["policy"]}_requests.csv')
        counts = (len(records), sum(r['status'] == 'completed' for r in records),
                  sum(r['status'] == 'dropped' for r in records), sum(r['slo_met'] == 'True' for r in records))
        if counts != tuple(int(row[k]) for k in ['arrivals', 'completed', 'dropped', 'slo_met']):
            raise ValueError('Live request accounting mismatch')
        if abs(float(row['slo_attainment'])-counts[3]/counts[0]) > 1e-12:
            raise ValueError('SLO denominator mismatch')
        for record in records:
            if record['status'] == 'completed' and abs(float(record['latency'])-(float(record['completion'])-float(record['arrival']))) > 1e-8:
                raise ValueError('Latency arithmetic mismatch')
    configuration = json.loads(Path('results/gpu_evaluation/evaluation_config.json').read_text(encoding='utf-8'))
    if ServiceProfile(**configuration['service_profile']).as_dict() != profile.as_dict():
        raise ValueError('Simulation/profile mismatch')
    if not all(0 <= float(r['slo_attainment_mean']) <= 1 for r in read('results/gpu_evaluation/aggregate.csv')):
        raise ValueError('Invalid aggregate attainment')
    print(json.dumps(dict(status='passed', timing_runs=len(timings), live_queue_policies=2,
                          calibrated_simulation_runs=configuration['policy_runs']), indent=2))


if __name__ == '__main__':
    main()
