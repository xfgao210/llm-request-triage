"""Small wall-clock single-worker CUDA queue replay: FCFS vs static SLO triage.

Controlled token-length requests; EOS ignored. Policies never receive the true
forced output count. No HTTP, tokenization, continuous batching, or client network.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fetch_gpu_model import MODEL_DIR
from gpu_benchmark import measure
from service_profile import ServiceProfile
from triage_sim import Request, Observation, Action, fcfs, static_slo, summarize_records


def save_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def requests_for_trial(seed, duration, slo):
    rng = random.Random(seed)
    requests = []
    for start, end, rate in [(0, duration/3, 2.), (duration/3, 2*duration/3, 6.),
                             (2*duration/3, duration, 2.)]:
        now = start
        while True:
            now += rng.expovariate(rate)
            if now >= end:
                break
            requests.append(Request(len(requests), now, rng.choice([64, 128, 256]),
                                    rng.choice([8, 16, 32]), now+slo))
    return requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=6.)
    parser.add_argument('--slo', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=9000)
    parser.add_argument('--output', type=Path, default=Path('results/gpu/live_queue'))
    args = parser.parse_args()
    if args.duration <= 0 or args.slo <= 0:
        parser.error('duration and SLO must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(4)
    profile = ServiceProfile.load('profiles/rtx5070_qwen05b.json')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
             trust_remote_code=False, dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    assert all(p.device.type == 'cuda' for p in model.parameters())
    sentence = tokenizer.encode('This is a controlled real GPU request scheduling experiment. ', add_special_tokens=False)
    tensors = {n: torch.tensor((sentence*((n+len(sentence)-1)//len(sentence)))[:n], device='cuda').unsqueeze(0)
               for n in [64, 128, 256]}
    requests = requests_for_trial(args.seed, args.duration, args.slo)
    if not requests:
        raise RuntimeError('No requests generated')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, policy in [('fcfs', fcfs), ('static_slo', static_slo)]:
        for _ in range(3):
            measure(model, tensors[128], 16)
        torch.cuda.synchronize()
        records = {r.request_id: dict(request_id=r.request_id, arrival=r.arrival,
                   input_tokens=r.input_tokens, output_tokens=r.output_tokens, deadline=r.deadline,
                   start=None, completion=None, wait=None, latency=None,
                   decision_time=None, status='pending', slo_met=False) for r in requests}
        cursor, queue, trace = 0, [], [dict(time=0., queue_length=0, server_busy=0)]
        started = time.perf_counter()
        while cursor < len(requests) or queue:
            now = time.perf_counter()-started
            while cursor < len(requests) and requests[cursor].arrival <= now:
                queue.append(requests[cursor])
                cursor += 1
            if not queue:
                if cursor < len(requests):
                    time.sleep(min(.005, max(0., requests[cursor].arrival-now)))
                continue
            r = queue.pop(0)
            # Known workload distribution mean; not computed from trial outcomes.
            estimate = (8+16+32)/3
            obs = Observation(now, r.request_id, r.input_tokens, estimate, r.deadline-now,
                              len(queue)+1, False, profile.service_s(r.input_tokens, estimate))
            action = policy(obs)
            row = records[r.request_id]
            row['decision_time'] = time.perf_counter()-started
            if action == Action.DROP:
                row['status'] = 'dropped'
            else:
                actual_start = time.perf_counter()-started
                row.update(start=actual_start, wait=actual_start-r.arrival)
                # True count is passed only to model execution, after the policy decision.
                timing = measure(model, tensors[r.input_tokens], r.output_tokens)
                completion = time.perf_counter()-started
                row.update(completion=completion, latency=completion-r.arrival,
                           status='completed', slo_met=completion <= r.deadline)
            trace.append(dict(time=time.perf_counter()-started, queue_length=len(queue), server_busy=0))
        final_time = time.perf_counter()-started
        # Reconstruct waiting queue from wall-clock events, including arrivals
        # that occurred while the synchronous GPU worker was executing.
        events = []
        for row in records.values():
            events.append((row['arrival'], 0, 1))
            removal = row['start'] if row['status'] == 'completed' else row['decision_time']
            events.append((removal, 1, -1))
        waiting, trace = 0, [dict(time=0., queue_length=0, server_busy=0)]
        for event_time, _, delta in sorted(events):
            waiting += delta
            assert waiting >= 0
            trace.append(dict(time=event_time, queue_length=waiting, server_busy=0))
        metrics = summarize_records(list(records.values()), trace, args.duration, final_time)
        results.append(dict(policy=name, **metrics))
        save_csv(output/f'{name}_requests.csv', list(records.values()))
        print(json.dumps(dict(policy=name, **metrics)), flush=True)
    save_csv(output/'metrics.csv', results)
    (output/'protocol.json').write_text(json.dumps(dict(seed=args.seed, arrival_window_s=args.duration,
         slo_s=args.slo, arrival_rates=[2, 6, 2], gpu=torch.cuda.get_device_name(0),
         precision='FP16', model='Qwen2.5-0.5B-Instruct', batch=1,
         output_distribution=[8, 16, 32], estimated_output_tokens=(8+16+32)/3,
         timing='real wall clock; GPU synchronize; no HTTP/tokenization/host transfer in request latency',
         limitations='One paired trial, sequential policy order; display apps share GPU; not statistical proof. Forced output length with EOS ignored.'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
