"""Extract fixed-duration BurstGPT 3 scenarios from a verified analysis catalog."""

import argparse
from bisect import bisect_right
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_triage.workload.burstgpt import TraceRow, sha256


REGIMES = {"low", "regular", "elevated", "peak", "extreme_peak"}
SPLITS = ("train", "validation", "test")


def prepare(source, analysis_dir, data_dir, *, counts=None, regimes=None, license_path=None):
    """Select evenly spaced eligible catalog windows and stream their source rows.

    Source identity is checked against the analysis summary. The catalog's time
    partitions and train-derived thresholds are retained without recomputing them.
    The prepared output is fresh, so existing scenarios cannot be overwritten.
    """
    source, analysis_dir, data_dir = map(Path, (source, analysis_dir, data_dir))
    counts = dict(train=8, validation=4, test=4) if counts is None else counts
    regimes = {split: ["regular"] if split == "train" else ["regular", "peak"]
               for split in SPLITS} if regimes is None else regimes
    if set(counts) != set(SPLITS) or set(regimes) != set(SPLITS):
        raise ValueError("Specify counts and regimes for all three splits")
    for split in SPLITS:
        if type(counts[split]) is not int or counts[split] < 1:
            raise ValueError("Scenario counts must be positive integers")
        if not regimes[split] or not set(regimes[split]) <= REGIMES:
            raise ValueError(f"Unknown or empty traffic regimes for {split}")
    processed = data_dir / "processed"
    if processed.exists():
        raise FileExistsError(f"Prepared output already exists: {processed}")
    summary_path = analysis_dir / "summary.json"
    catalog_path = analysis_dir / "scenario_catalog.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_digest = sha256(source)
    if source_digest != summary["source"]["sha256"]:
        raise ValueError("Source CSV does not match the traffic analysis SHA256")
    selected = []
    with catalog_path.open(encoding="utf-8", newline="") as stream:
        catalog = list(csv.DictReader(stream))
    for split in SPLITS:
        candidates = sorted(
            (item for item in catalog if item["partition"] == split
             and item["traffic_regime"] in regimes[split] and int(item["request_count"]) > 0),
            key=lambda item: float(item["start_timestamp_s"]),
        )
        count = counts[split]
        if len(candidates) < count:
            raise ValueError(f"Need {count} eligible {split} scenarios; found {len(candidates)}")
        for index in range(count):
            position = index * (len(candidates) - 1) // (count - 1) if count > 1 else len(candidates) // 2
            item = candidates[position]
            selected.append(dict(
                split=split, window=int(item["bin_index"]), scenario_id=item["scenario_id"],
                traffic_regime=item["traffic_regime"], requests=int(item["request_count"]),
                start_timestamp_s=float(item["start_timestamp_s"]),
                end_timestamp_s=float(item["end_timestamp_s"]),
                mean_rps=float(item["mean_rps"]),
            ))
    selected.sort(key=lambda item: item["start_timestamp_s"])
    if (any(not math.isfinite(item["start_timestamp_s"]) or not math.isfinite(item["end_timestamp_s"])
            or item["start_timestamp_s"] < 0 or item["end_timestamp_s"] <= item["start_timestamp_s"]
            for item in selected)
            or any(a["end_timestamp_s"] > b["start_timestamp_s"] for a, b in zip(selected, selected[1:]))
            or len({item["window"] for item in selected}) != len(selected)):
        raise ValueError("Catalog contains invalid, overlapping, or duplicate windows")
    starts = [item["start_timestamp_s"] for item in selected]
    windows = {item["window"]: [] for item in selected}
    previous = None
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"Timestamp", "Model", "Request tokens", "Response tokens", "Log Type"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing BurstGPT columns")
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
            if index >= 0 and timestamp < selected[index]["end_timestamp_s"]:
                windows[selected[index]["window"]].append(TraceRow(source_row, timestamp, prompt, output))
    for item in selected:
        if len(windows[item["window"]]) != item["requests"]:
            raise ValueError(f"Catalog request count mismatch for {item['scenario_id']}")
    training = [row for item in selected if item["split"] == "train" for row in windows[item["window"]]]
    manifest = dict(
        schema_version=2, dataset="burstgpt3", windowing="fixed_duration",
        source_path=str(source.resolve()), source_sha256=source_digest, source_bytes=source.stat().st_size,
        analysis_summary_sha256=sha256(summary_path), scenario_catalog_sha256=sha256(catalog_path),
        timestamp_unit="seconds", source_row_index="zero-based, excluding header",
        selection=dict(counts=counts, regimes=regimes, method="evenly spaced eligible windows by time"),
        training_regime_thresholds_rps=summary["training_regime_thresholds_rps"],
        windows=selected,
        estimated_output_tokens_train_mean=sum(row.true_output_tokens for row in training) / len(training),
        limitations="Token metadata only; elapsed time is not used as GPU service time. Test scenarios reserved.",
    )
    license_bytes = None
    if license_path is not None:
        license_path = Path(license_path)
        license_bytes = license_path.read_bytes()
        manifest.update(license="CC-BY-4.0", license_sha256=sha256(license_path))
    processed.mkdir(parents=True)
    csv_path = processed / "burstgpt_windows.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "window", *TraceRow.__dataclass_fields__])
        writer.writeheader()
        for item in selected:
            for row in windows[item["window"]]:
                writer.writerow(dict(split=item["split"], window=item["window"], **asdict(row)))
    manifest["selected_sha256"] = sha256(csv_path)
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if license_bytes is not None:
        (processed / "LICENSE-BurstGPT.txt").write_bytes(license_bytes)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("BurstGPT_3.csv"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/traffic_analysis"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/burstgpt3"))
    parser.add_argument("--license", type=Path, help="Optional upstream CC-BY-4.0 license to copy")
    parser.add_argument("--ranges-config", type=Path,
                        help="Explicit time ranges; replaces automatic catalog/regime selection")
    for split, count in zip(SPLITS, (8, 4, 4)):
        parser.add_argument(f"--{split}-windows", type=int, default=count)
        parser.add_argument(f"--{split}-regimes", nargs="+", choices=sorted(REGIMES),
                            default=["regular"] if split == "train" else ["regular", "peak"])
    args = parser.parse_args()
    if args.ranges_config is not None:
        explicit_flags = {flag.split("=")[0] for flag in sys.argv[1:]}
        selection_flags = {f"--{split}-{suffix}" for split in SPLITS for suffix in ("windows", "regimes")}
        if explicit_flags & selection_flags:
            parser.error("--ranges-config replaces --*-windows and --*-regimes; do not combine them")
        from llm_triage.workload.time_ranges import prepare_ranges
        manifest = prepare_ranges(args.source, args.analysis_dir, args.data_dir, args.ranges_config,
                                  license_path=args.license)
    else:
        manifest = prepare(args.source, args.analysis_dir, args.data_dir,
                           counts={split: getattr(args, f"{split}_windows") for split in SPLITS},
                           regimes={split: getattr(args, f"{split}_regimes") for split in SPLITS},
                           license_path=args.license)
    print(json.dumps(dict(windows=len(manifest["windows"]), data_dir=str(args.data_dir)), indent=2))


if __name__ == "__main__":
    main()
