"""Prepare consecutive episodes from explicit, disjoint BurstGPT time ranges."""

from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .burstgpt import sha256


SPLITS = ("train", "validation", "test")


def validate_ranges(config):
    """Return chronological ranges and reject overlap or temporal leakage."""
    if config.get("schema_version") != 1:
        raise ValueError("Range config schema_version must be 1")
    digest = config.get("source_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("source_sha256 must identify the analyzed CSV")
    duration = config.get("episode_seconds")
    gap = config.get("min_split_gap_seconds", 0)
    for name, value, positive in (("episode_seconds", duration, True), ("min_split_gap_seconds", gap, False)):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or value < 0 or (positive and value == 0)):
            raise ValueError(f"Invalid {name}")
    ranges = config.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("Specify explicit ranges for train, validation, and test")
    normalized = []
    for index, item in enumerate(ranges):
        if not isinstance(item, dict) or item.get("split") not in SPLITS:
            raise ValueError("Range split must be train, validation, or test")
        start, end = item.get("start_timestamp_s"), item.get("end_timestamp_s")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (start, end)) or start < 0 or end <= start:
            raise ValueError("Range timestamps must be finite, nonnegative, and start < end")
        group = item.get("group", item["split"])
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Every range needs a nonempty group label")
        normalized.append(dict(split=item["split"], group=group,
                               range_id=index, start_timestamp_s=start, end_timestamp_s=end))
    normalized.sort(key=lambda item: item["start_timestamp_s"])
    if {item["split"] for item in normalized} != set(SPLITS):
        raise ValueError("Require all three chronological splits")
    for left, right in zip(normalized, normalized[1:]):
        if right["start_timestamp_s"] < left["end_timestamp_s"]:
            raise ValueError("Time ranges must not overlap")
        if SPLITS.index(left["split"]) > SPLITS.index(right["split"]):
            raise ValueError("Require train before validation before test")
        if left["split"] != right["split"] and right["start_timestamp_s"] - left["end_timestamp_s"] < gap:
            raise ValueError("Time ranges violate min_split_gap_seconds")
    groups = {}
    for item in normalized:
        if item["group"] in groups and groups[item["group"]] != item["split"]:
            raise ValueError("Group labels must not cross splits")
        groups[item["group"]] = item["split"]
    return normalized


def prepare_ranges(source, analysis_dir, data_dir, config_path, *, license_path=None):
    """Stream requests into all consecutive episodes, with no regime filtering."""
    source, analysis_dir, data_dir, config_path = map(Path, (source, analysis_dir, data_dir, config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ranges = validate_ranges(config)
    processed = data_dir / "processed"
    if processed.exists():
        raise FileExistsError(f"Prepared output already exists: {processed}")
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if config["source_sha256"] != summary["source"]["sha256"] or sha256(source) != config["source_sha256"]:
        raise ValueError("Source CSV, analysis, and range config must have matching SHA256")
    lower = summary["trace"]["origin_timestamp_s"]
    upper = lower + summary["windowing"]["bin_seconds"] * summary["windowing"]["total_time_bins"]
    if any(item["start_timestamp_s"] < lower or item["end_timestamp_s"] > upper for item in ranges):
        raise ValueError("Range lies outside the analyzed timeline")
    episodes = []
    for item in ranges:
        start = item["start_timestamp_s"]
        count = math.ceil((item["end_timestamp_s"] - start) / config["episode_seconds"])
        for index in range(count):
            window = len(episodes)
            episodes.append(dict(
                    item, window=window, scenario_id=f"{item['split']}_{window:06d}",
                start_timestamp_s=start + index * config["episode_seconds"],
                end_timestamp_s=min(start + (index + 1) * config["episode_seconds"], item["end_timestamp_s"]),
                requests=0, input_tokens=0, output_tokens=0,
            ))
    starts = [item["start_timestamp_s"] for item in episodes]
    # Read the optional license before publishing any prepared artifacts.
    license_bytes = Path(license_path).read_bytes() if license_path is not None else None
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-ranges-", dir=data_dir) as temporary:
        staging = Path(temporary) / "processed"
        staging.mkdir()
        trace_path = staging / "burstgpt_windows.csv"
        previous = None
        with source.open(encoding="utf-8-sig", newline="") as stream, trace_path.open(
            "w", encoding="utf-8", newline=""
        ) as destination:
            reader = csv.DictReader(stream)
            required = {"Timestamp", "Model", "Request tokens", "Response tokens", "Log Type"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Missing BurstGPT columns")
            writer = csv.writer(destination)
            writer.writerow(["split", "window", "source_row", "timestamp_s", "input_tokens", "true_output_tokens"])
            for source_row, row in enumerate(reader):
                if row["Model"] != "ChatGPT" or row["Log Type"] != "API log":
                    continue
                try:
                    timestamp = float(row["Timestamp"])
                    prompt, output = int(row["Request tokens"]), int(row["Response tokens"])
                    if not math.isfinite(timestamp) or timestamp < 0 or prompt <= 0 or output <= 0:
                        continue
                except (TypeError, ValueError, OverflowError):
                    continue
                if previous is not None and timestamp < previous:
                    raise ValueError("Source requests must be chronological")
                previous = timestamp
                index = bisect_right(starts, timestamp) - 1
                if index < 0 or timestamp >= episodes[index]["end_timestamp_s"]:
                    continue
                item = episodes[index]
                writer.writerow([item["split"], item["window"], source_row, timestamp, prompt, output])
                item["requests"] += 1
                item["input_tokens"] += prompt
                item["output_tokens"] += output
        for group in {item["group"] for item in episodes}:
            if not sum(item["requests"] for item in episodes if item["group"] == group):
                raise ValueError(f"Selected group has no valid requests: {group}")
        training = [item for item in episodes if item["split"] == "train"]
        manifest = dict(
            schema_version=3, dataset="burstgpt3", windowing="fixed_duration",
            source_path=str(source.resolve()), source_sha256=config["source_sha256"],
            source_bytes=source.stat().st_size, analysis_summary_sha256=sha256(summary_path),
            range_config=config, range_config_sha256=sha256(config_path),
            selection=dict(method="explicit chronological ranges; no regime filter", ranges=ranges),
            timestamp_unit="seconds", source_row_index="zero-based, excluding header",
            windows=[item for item in episodes if item["requests"]],
            empty_windows=[item for item in episodes if not item["requests"]],
            selected_sha256=sha256(trace_path),
            estimated_output_tokens_train_mean=sum(item["output_tokens"] for item in training)
            / sum(item["requests"] for item in training),
            limitations=["Episode queues reset at boundaries; no queue state crosses episodes.",
                         "Empty episodes are recorded but skipped by training and policy execution.",
                         "No-record periods can be inactivity or missing collection.",
                         "Elapsed time is not used as GPU service time."],
        )
        if license_bytes is not None:
            manifest.update(license="CC-BY-4.0", license_sha256=hashlib.sha256(license_bytes).hexdigest())
            (staging / "LICENSE-BurstGPT.txt").write_bytes(license_bytes)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(processed)
    return manifest
