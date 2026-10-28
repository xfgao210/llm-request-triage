"""Fetch public BurstGPT and prepare small, chronologically disjoint replay windows."""
import csv
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'https://api.github.com/repos/HPMLL/BurstGPT'
PINNED_REVISION = 'd895a53bb7b8ec137d0d2fe203b335835a78c10a'
PINNED_SHA256 = '46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a'


def get_json(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'SLO-triage-research'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def prepare():
    raw_dir = Path('data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir/'BurstGPT_1.csv'
    source_manifest = raw_dir/'source.json'
    if not path.exists():
        revision = PINNED_REVISION
        url = f'https://raw.githubusercontent.com/HPMLL/BurstGPT/{revision}/data/BurstGPT_1.csv'
        request = urllib.request.Request(url, headers={'User-Agent': 'SLO-triage-research'})
        temporary = path.with_suffix('.part')
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as out:
            while block := response.read(1024*1024):
                out.write(block)
        temporary.replace(path)
        source_manifest.write_text(json.dumps(dict(url=url, revision=revision), indent=2), encoding='utf-8')
    if not source_manifest.exists():
        raise ValueError('Missing source provenance; restore data/raw/source.json or re-fetch raw file')
    source = json.loads(source_manifest.read_text(encoding='utf-8'))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PINNED_SHA256 or source['revision'] != PINNED_REVISION:
        raise ValueError('Raw dataset does not match pinned source/hash; do not mix dataset versions')
    valid = []
    counts = dict(total_rows=0, excluded_model_or_channel=0, invalid_or_failed=0)
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'Timestamp', 'Model', 'Request tokens', 'Response tokens', 'Log Type'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Unexpected BurstGPT CSV schema')
        for index, row in enumerate(reader):
            counts['total_rows'] += 1
            if row['Model'] != 'ChatGPT' or row['Log Type'] != 'API log':
                counts['excluded_model_or_channel'] += 1
                continue
            timestamp = float(row['Timestamp'])
            prompt, output = int(row['Request tokens']), int(row['Response tokens'])
            if timestamp < 0 or prompt <= 0 or output <= 0:
                counts['invalid_or_failed'] += 1
                continue
            valid.append((timestamp, index, prompt, output))
    valid.sort()
    if len(valid) < 10000:
        raise ValueError('Too few valid API requests')
    # Split using timestamp boundaries, so tied source timestamps cannot straddle partitions.
    cut1, cut2 = valid[int(.6*len(valid))][0], valid[int(.8*len(valid))][0]
    partitions = {'train': [r for r in valid if r[0] < cut1],
                  'validation': [r for r in valid if cut1 <= r[0] < cut2],
                  'test': [r for r in valid if r[0] >= cut2]}
    windows, metadata = [], []
    n = 256
    for split, count in [('train', 32), ('validation', 8), ('test', 16)]:
        rows = partitions[split]
        if len(rows) < count*n:
            raise ValueError(f'Not enough rows for disjoint windows in {split}')
        for window in range(count):
            start = int(window*(len(rows)-n)/(count-1))
            chunk = rows[start:start+n]
            if chunk[-1][0] <= chunk[0][0]:
                raise ValueError('A selected window has no positive time span')
            metadata.append(dict(split=split, window=window, source_start_s=chunk[0][0],
                                 source_end_s=chunk[-1][0], requests=n))
            for timestamp, source_row, prompt, output in chunk:
                windows.append(dict(split=split, window=window, source_row=source_row,
                                    timestamp=timestamp, input_tokens=prompt, output_tokens=output))
    # Reserve a second untouched test suite between the first test windows.
    # It is never used by train_dqn.py or evaluate_dqn.py checkpoint selection.
    first_test = [r for r in metadata if r['split'] == 'test']
    for index, (left, right) in enumerate(zip(first_test, first_test[1:])):
        gap = [r for r in partitions['test'] if left['source_end_s'] < r[0] < right['source_start_s']]
        if len(gap) < n:
            raise ValueError('Insufficient timestamp-disjoint audit gap')
        middle = (len(gap)-n)//2
        chunk = gap[middle:middle+n]
        metadata.append(dict(split='audit', window=index, source_start_s=chunk[0][0],
                             source_end_s=chunk[-1][0], requests=n))
        for timestamp, source_row, prompt, output in chunk:
            windows.append(dict(split='audit', window=index, source_row=source_row,
                                timestamp=timestamp, input_tokens=prompt, output_tokens=output))
    audit_metadata = [r for r in metadata if r['split'] == 'audit']
    for index, (left, right) in enumerate(zip(first_test, audit_metadata)):
        gap = [r for r in partitions['test'] if left['source_end_s'] < r[0] < right['source_start_s']]
        if len(gap) < n:
            raise ValueError('Insufficient timestamp-disjoint robust test gap')
        middle = (len(gap)-n)//2
        chunk = gap[middle:middle+n]
        metadata.append(dict(split='robust_test', window=index, source_start_s=chunk[0][0],
                             source_end_s=chunk[-1][0], requests=n))
        for timestamp, source_row, prompt, output in chunk:
            windows.append(dict(split='robust_test', window=index, source_row=source_row,
                                timestamp=timestamp, input_tokens=prompt, output_tokens=output))
    out_dir = Path('data/processed')
    out_dir.mkdir(parents=True, exist_ok=True)
    from repro_utils import write_csv
    write_csv(out_dir/'burstgpt_windows.csv', windows)
    training = [r for r in windows if r['split'] == 'train']
    mean_output = sum(r['output_tokens'] for r in training)/len(training)
    manifest = dict(source=source, sha256=digest, source_bytes=path.stat().st_size,
                    license='CC-BY-4.0', license_url='https://github.com/HPMLL/BurstGPT/blob/main/LICENSE',
                    paper='https://doi.org/10.1145/3711896.3737413', counts=counts,
                    valid_requests=len(valid), filter='ChatGPT, API log, positive input and output tokens',
                    chronological_cuts_s=[cut1, cut2], windows=metadata,
                    estimated_output_tokens_train_mean=mean_output,
                    limitations='Count-based evenly spaced windows, not the full trace. Arrival rates may be rescaled. No measured GPU service times in this source.')
    (out_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    license_request = urllib.request.Request(
        f'https://raw.githubusercontent.com/HPMLL/BurstGPT/{source["revision"]}/LICENSE',
        headers={'User-Agent': 'SLO-triage-research'})
    if not (out_dir/'LICENSE-BurstGPT.txt').exists():
        with urllib.request.urlopen(license_request, timeout=30) as response:
            (out_dir/'LICENSE-BurstGPT.txt').write_bytes(response.read())
    print(json.dumps(dict(counts=counts, valid_requests=len(valid), selected=len(windows),
                          estimated_output_tokens_train_mean=mean_output, sha256=digest), indent=2))


if __name__ == '__main__':
    prepare()
