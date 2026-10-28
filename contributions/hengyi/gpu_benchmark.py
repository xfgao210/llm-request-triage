"""Real single-request CUDA prefill/decode benchmark. Run in .venv-gpu.

Full precision mode is FP16, SDPA, no CPU offload, batch=1, KV cache enabled.
Prefill includes selection of first output token; decode covers remaining N-1.
"""
import argparse
import csv
import json
import platform
from pathlib import Path
import random
import subprocess
import time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from fetch_gpu_model import MODEL_DIR


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def measure(model, input_ids, output_tokens):
    prompt_tokens = input_ids.shape[1]
    mask = torch.ones((1, prompt_tokens+output_tokens), dtype=torch.long, device='cuda')
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    start.record()
    result = model(input_ids=input_ids, attention_mask=mask[:, :prompt_tokens],
                   use_cache=True, return_dict=True, logits_to_keep=1)
    token = result.logits[:, -1:, :].argmax(dim=-1)
    past = result.past_key_values
    end.record()
    end.synchronize()
    prefill_wall = time.perf_counter()-started
    prefill_cuda = start.elapsed_time(end)/1000
    start.record()
    started = time.perf_counter()
    for index in range(output_tokens-1):
        result = model(input_ids=token, attention_mask=mask[:, :prompt_tokens+index+1],
                       past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1)
        token = result.logits[:, -1:, :].argmax(dim=-1)
        past = result.past_key_values
    end.record()
    end.synchronize()
    decode_wall = time.perf_counter()-started if output_tokens > 1 else 0.
    decode_cuda = start.elapsed_time(end)/1000 if output_tokens > 1 else 0.
    return dict(input_tokens=prompt_tokens, output_tokens=output_tokens,
                decode_forward_steps=output_tokens-1, prefill_s=prefill_wall, decode_s=decode_wall,
                prefill_cuda_s=prefill_cuda, decode_cuda_s=decode_cuda,
                ttft_s=prefill_wall, e2e_s=prefill_wall+decode_wall,
                decode_tokens_per_second=(output_tokens-1)/decode_wall if output_tokens > 1 else None,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prompts', type=int, nargs='+', default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument('--outputs', type=int, nargs='+', default=[16, 64, 128])
    parser.add_argument('--output', type=Path, default=Path('results/gpu'))
    args = parser.parse_args()
    if args.repeats < 1 or min(args.prompts) < 1 or min(args.outputs) < 1:
        parser.error('Positive repeats/prompts/outputs required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; refusing a CPU fallback benchmark')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    print(f'Loading model on {torch.cuda.get_device_name(0)}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
             trust_remote_code=False, dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    assert all(p.device.type == 'cuda' for p in model.parameters())
    source = json.loads((MODEL_DIR/'source.json').read_text(encoding='utf-8'))
    info = dict(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                total_memory_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
                torch=torch.__version__, cuda_runtime=torch.version.cuda, transformers=transformers.__version__,
                python=platform.python_version(), platform=platform.platform(), precision='float16',
                attention='sdpa', batch_size=1, kv_cache=True, cpu_offload=False,
                all_parameters_on_cuda=True, model=source, repeats=args.repeats,
                prompt_lengths=args.prompts, output_lengths=args.outputs,
                output_semantics='exact N output tokens; EOS ignored for shape-controlled timing; prefill includes first token selection; decode N-1 forwards',
                timing='Wall clock with CUDA synchronization; additional CUDA-event spans, including enqueue gaps, recorded separately',
                limitations='Windows WDDM desktop GPU shared with display/apps, no power/clock locking. No concurrent batch serving or vLLM.')
    info['driver'] = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    (args.output/'environment.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    # Natural-length smoke test, separate from forced-length timing.
    chat = tokenizer.apply_chat_template([{'role': 'user', 'content': 'What is 2 + 2? Answer briefly.'}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    smoke = tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    (args.output/'smoke_test.txt').write_text(smoke, encoding='utf-8')
    print(f'Real model response: {smoke}', flush=True)
    sentence = tokenizer.encode('This is a reproducible inference timing experiment. Read the context and continue the explanation. ', add_special_tokens=False)
    prompt_ids = {n: torch.tensor((sentence*((n+len(sentence)-1)//len(sentence)))[:n],
                                  dtype=torch.long, device='cuda').unsqueeze(0) for n in args.prompts}
    # Global warmup and one additional warmup per shape, excluded from measurements.
    for _ in range(3):
        measure(model, prompt_ids[args.prompts[0]], args.outputs[0])
    shapes = [(p, o) for p in args.prompts for o in args.outputs]
    random.Random(42).shuffle(shapes)
    rows = []
    for prompt, output in shapes:
        measure(model, prompt_ids[prompt], output)
        for repeat in range(args.repeats):
            torch.cuda.reset_peak_memory_stats()
            row = dict(repeat=repeat, **measure(model, prompt_ids[prompt], output))
            rows.append(row)
            write_rows(args.output/'measurements.csv', rows)
        print(f'prompt={prompt} output={output} completed {args.repeats} repeats', flush=True)
    print(f'Completed {len(rows)} real GPU measurements', flush=True)


if __name__ == '__main__':
    main()
