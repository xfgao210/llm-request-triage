"""Summarize measured GPU timing, profile validation, and real wall-clock queue replay."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from service_profile import ServiceProfile


def read(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    folder = Path('results/gpu')
    environment = json.loads((folder/'environment.json').read_text(encoding='utf-8'))
    calibrated = json.loads(Path('profiles/rtx5070_qwen05b.json').read_text(encoding='utf-8'))
    profile = ServiceProfile.load('profiles/rtx5070_qwen05b.json')
    first = read(folder/'measurements.csv')
    rows = first+read(folder/'short/measurements.csv')
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row['input_tokens']), int(row['output_tokens'])), []).append(row)
    shapes = [dict(input=p, output=o, prefill=statistics.median(float(r['prefill_s']) for r in group),
                   decode=statistics.median(float(r['decode_s']) for r in group))
              for (p, o), group in sorted(grouped.items())]
    live = read(folder/'live_queue_checked/metrics.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    ax = axes[0, 0]
    for p in sorted({r['input'] for r in shapes}):
        data = [r['prefill']*1000 for r in shapes if r['input'] == p]
        ax.scatter([p]*len(data), data, color='#2563eb', s=18)
    lengths = np.linspace(32, 2048, 100)
    ax.plot(lengths, [profile.prefill_s(p)*1000 for p in lengths], color='#059669', label='calibrated model')
    ax.set(title='Real CUDA prefill (includes first token)', xlabel='input tokens', ylabel='milliseconds')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for o, color in [(16, '#64748b'), (64, '#2563eb'), (128, '#059669')]:
        group = [r for r in shapes if r['output'] == o]
        ax.plot([r['input'] for r in group], [(o-1)/r['decode'] for r in group], 'o-', label=f'{o} output tokens', color=color)
    ax.set(title='Measured decode speed: this backend', xlabel='input tokens', ylabel='cached decode tokens/second')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for held, label, color in [(False, 'fit lengths', '#2563eb'), (True, 'held-out lengths', '#d97706')]:
        group = [r for r in shapes if (r['input'] in [128, 1024]) == held]
        ax.scatter([r['prefill']+r['decode'] for r in group], [profile.service_s(r['input'], r['output']) for r in group], label=label, color=color)
    maximum = max(r['prefill']+r['decode'] for r in shapes)*1.05
    ax.plot([0, maximum], [0, maximum], '--', color='#64748b')
    ax.set(title=f'Shape-median E2E: held-out MAPE {calibrated["holdout_diagnostics"]["e2e"]["mape"]:.1%}', xlabel='measured seconds', ylabel='predicted seconds')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    bars = ax.bar([r['policy'] for r in live], [100*float(r['slo_attainment']) for r in live], color=['#64748b', '#059669'])
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set(title='Real GPU queue: one paired 21-request example', ylabel='overall SLO attainment (%)', ylim=(0, 105))
    fig.suptitle('RTX 5070 + Qwen2.5-0.5B-Instruct | FP16 SDPA | batch=1', fontsize=14)
    fig.supxlabel('102 timing runs; forced output lengths. Desktop WDDM backend, not maximum GPU throughput or production validation.', fontsize=9)
    fig.savefig(folder/'gpu_figure.png', dpi=180)
    fig.savefig(folder/'gpu_figure.svg')
    print((folder/'gpu_figure.png').resolve())


if __name__ == '__main__':
    main()
