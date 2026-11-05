"""Pinned BurstGPT acquisition and strict, deterministic trace preparation.

Source timestamps are seconds from the source trace origin. Source row identifiers
are zero-based data-row indices, excluding the CSV header. No request text is used.
"""
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import urllib.request

REVISION = "d895a53bb7b8ec137d0d2fe203b335835a78c10a"
SHA256 = "46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a"
BASE_URL = f"https://raw.githubusercontent.com/HPMLL/BurstGPT/{REVISION}"
SOURCE_URL = f"{BASE_URL}/data/BurstGPT_1.csv"
REQUIRED = {"Timestamp", "Model", "Request tokens", "Response tokens", "Log Type"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "llm-request-triage"})
    with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def acquire(raw_dir, source=None):
    """Reuse verified cache, copy a pinned local source, or download atomically.

    A provided local file must match the same pinned digest; fixtures should use
    parse_trace directly. Corrupt cache is rejected, never silently overwritten.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / "BurstGPT_1.csv"
    if source is not None and sha256(source) != SHA256:
        raise ValueError("Local source does not match the pinned BurstGPT SHA256")
    if not target.exists():
        temporary = target.with_suffix(".part")
        try:
            if source is None:
                _download(SOURCE_URL, temporary)
            else:
                shutil.copyfile(source, temporary)
            if sha256(temporary) != SHA256:
                raise ValueError("Downloaded trace does not match the pinned BurstGPT SHA256")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(target) != SHA256:
        raise ValueError("Cached trace does not match the pinned BurstGPT SHA256")
    license_path = raw_dir / "LICENSE-BurstGPT.txt"
    if not license_path.exists():
        temporary = license_path.with_suffix(".part")
        try:
            _download(f"{BASE_URL}/LICENSE", temporary)
            temporary.replace(license_path)
        finally:
            temporary.unlink(missing_ok=True)
    return target


@dataclass(frozen=True)
class TraceRow:
    source_row: int
    timestamp_s: float
    input_tokens: int
    true_output_tokens: int


def parse_trace(path):
    """Filter ChatGPT API rows; count invalid records instead of aborting a run."""
    rows = []
    counts = dict(total_rows=0, excluded_model_or_channel=0, invalid_or_failed=0,
                  valid_requests=0)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing BurstGPT columns: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        for index, row in enumerate(reader):
            counts["total_rows"] += 1
            if row["Model"] != "ChatGPT" or row["Log Type"] != "API log":
                counts["excluded_model_or_channel"] += 1
                continue
            try:
                timestamp = float(row["Timestamp"])
                prompt, output = int(row["Request tokens"]), int(row["Response tokens"])
                if not math.isfinite(timestamp) or timestamp < 0 or prompt <= 0 or output <= 0:
                    raise ValueError("Invalid timestamp or token length")
            except (TypeError, ValueError, OverflowError):
                counts["invalid_or_failed"] += 1
                continue
            rows.append(TraceRow(index, timestamp, prompt, output))
    rows.sort(key=lambda row: (row.timestamp_s, row.source_row))
    counts["valid_requests"] = len(rows)
    if not rows:
        raise ValueError("No valid ChatGPT API requests")
    return rows, counts


def chronological_split(rows, train_fraction=0.6, validation_fraction=0.2):
    """Timestamp-boundary split: equal timestamps never cross partitions."""
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and
            train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than one")
    if len(rows) < 3 or any(a.timestamp_s > b.timestamp_s for a, b in zip(rows, rows[1:])):
        raise ValueError("Need at least three chronological rows")
    if len({r.source_row for r in rows}) != len(rows):
        raise ValueError("Duplicate source rows")
    cut1 = rows[int(len(rows) * train_fraction)].timestamp_s
    cut2 = rows[int(len(rows) * (train_fraction + validation_fraction))].timestamp_s
    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        key = "train" if row.timestamp_s < cut1 else "validation" if row.timestamp_s < cut2 else "test"
        splits[key].append(row)
    if not all(splits.values()):
        raise ValueError("Timestamp ties leave an empty partition; use a longer trace")
    return splits, [cut1, cut2]


def select_windows(splits, requests_per_window=256, counts=None):
    """Evenly spaced, source-row-disjoint windows within each partition.

    Equal timestamps may occur in different windows *within* one partition;
    windows are episodes, not claims of disjoint wall-clock intervals.
    """
    counts = {"train": 32, "validation": 8, "test": 16} if counts is None else counts
    if requests_per_window < 2 or set(counts) != set(splits):
        raise ValueError("Need at least two requests per window and counts for all splits")
    windows = {}
    for split, rows in splits.items():
        count = counts[split]
        if count < 1 or len(rows) < requests_per_window * count:
            raise ValueError(f"Not enough rows for {count} disjoint {split} windows")
        for index in range(count):
            start = index * (len(rows) - requests_per_window) // (count - 1) if count > 1 else 0
            chunk = rows[start:start + requests_per_window]
            if chunk[-1].timestamp_s <= chunk[0].timestamp_s:
                raise ValueError(f"{split}/{index} has zero time span")
            windows[split, index] = chunk
    return windows


def prepare(data_dir="data", source=None, requests_per_window=256, counts=None):
    data_dir = Path(data_dir)
    raw = acquire(data_dir / "raw", source)
    rows, filtering = parse_trace(raw)
    splits, cuts = chronological_split(rows)
    windows = select_windows(splits, requests_per_window, counts)
    processed = data_dir / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    csv_path = processed / "burstgpt_windows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "window", *TraceRow.__dataclass_fields__])
        writer.writeheader()
        for (split, window), chunk in windows.items():
            for row in chunk:
                writer.writerow(dict(split=split, window=window, **asdict(row)))
    training = [r for (split, _), chunk in windows.items() if split == "train" for r in chunk]
    manifest = dict(schema_version=1, source_url=SOURCE_URL, revision=REVISION,
                    source_sha256=sha256(raw), source_bytes=raw.stat().st_size,
                    license="CC-BY-4.0", license_url=f"{BASE_URL}/LICENSE",
                    license_sha256=sha256(data_dir / "raw" / "LICENSE-BurstGPT.txt"),
                    timestamp_unit="seconds", source_row_index="zero-based, excluding header",
                    counts=filtering, chronological_cuts_s=cuts,
                    partition_counts={key: len(value) for key, value in splits.items()},
                    selected_sha256=sha256(csv_path),
                    estimated_output_tokens_train_mean=sum(r.true_output_tokens for r in training) / len(training),
                    windows=[dict(split=split, window=index, requests=len(chunk),
                                  source_start_s=chunk[0].timestamp_s, source_end_s=chunk[-1].timestamp_s,
                                  source_rows_sha256=hashlib.sha256(
                                      ",".join(str(r.source_row) for r in chunk).encode()).hexdigest())
                             for (split, index), chunk in windows.items()],
                    limitations="Selected count-based windows, not full-trace replay. No request text or GPU timings.")
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    shutil.copyfile(data_dir / "raw" / "LICENSE-BurstGPT.txt", processed / "LICENSE-BurstGPT.txt")
    return manifest


def load_windows(path):
    windows = {}
    seen = set()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "window", *TraceRow.__dataclass_fields__}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected prepared-window schema")
        for row in reader:
            trace = TraceRow(int(row["source_row"]), float(row["timestamp_s"]),
                             int(row["input_tokens"]), int(row["true_output_tokens"]))
            if trace.source_row in seen:
                raise ValueError("Prepared windows reuse a source row")
            if (trace.source_row < 0 or not math.isfinite(trace.timestamp_s) or trace.timestamp_s < 0
                    or trace.input_tokens <= 0 or trace.true_output_tokens <= 0):
                raise ValueError("Invalid prepared request")
            seen.add(trace.source_row)
            windows.setdefault((row["split"], int(row["window"])), []).append(trace)
    if not windows:
        raise ValueError("No prepared windows")
    return windows
