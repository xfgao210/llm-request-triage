"""Real-token replay helpers. Time scaling is explicit; output length is environment-only."""
import csv
from pathlib import Path
from triage_sim import Request


def load_windows(path=Path('data/processed/burstgpt_windows.csv')):
    windows = {}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['split'], int(row['window']))
            windows.setdefault(key, []).append((float(row['timestamp']), int(row['input_tokens']),
                                                int(row['output_tokens'])))
    return windows


def replay_window(rows, target_rate=1.2, slo=5.0):
    if target_rate <= 0 or slo <= 0 or len(rows) < 2:
        raise ValueError('Need positive target rate/SLO and at least two rows')
    if any(rows[i][0] > rows[i+1][0] for i in range(len(rows)-1)):
        raise ValueError('Replay rows must be chronological')
    span = rows[-1][0]-rows[0][0]
    if span <= 0:
        raise ValueError('Replay window must have positive time span')
    # Count/window duration equals target_rate. Last arrival stays strictly inside horizon.
    horizon = len(rows)/target_rate
    scale = ((len(rows)-1)/target_rate)/span
    requests = [Request(i, (t-rows[0][0])*scale, prompt, output,
                        (t-rows[0][0])*scale+slo) for i, (t, prompt, output) in enumerate(rows)]
    return requests, horizon, scale
