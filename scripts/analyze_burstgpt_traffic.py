"""Analyze BurstGPT 3 traffic and build chronological RL scenario candidates.

The analyzer streams the source CSV, retains the same ChatGPT API records used
by the workload pipeline, and aggregates them into fixed-duration bins. Traffic
regime thresholds are learned from the chronological training partition only;
the same thresholds are then applied to validation and held-out test bins.

Example:

    python scripts/analyze_burstgpt_traffic.py \
        --source BurstGPT_3.csv \
        --output-dir results/traffic_analysis
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable


EXPECTED_COLUMNS = (
    "Timestamp",
    "Session ID",
    "Elapsed time",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
)

COUNT_FIELDS = (
    "total_rows",
    "excluded_model_or_channel",
    "invalid_or_failed",
    "valid_requests",
    "session_id_present",
    "session_id_missing",
    "invalid_elapsed_time",
    "invalid_total_tokens",
    "total_token_mismatch",
)

TIME_BIN_FIELDS = (
    "bin_index",
    "scenario_id",
    "partition",
    "traffic_regime",
    "start_timestamp_s",
    "end_timestamp_s",
    "source_row_start",
    "source_row_end",
    "request_count",
    "mean_rps",
    "peak_burst_rps",
    "burst_ratio",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_tokens_per_s",
    "output_tokens_per_s",
    "mean_input_tokens",
    "p50_input_tokens",
    "p95_input_tokens",
    "p99_input_tokens",
    "mean_output_tokens",
    "p50_output_tokens",
    "p95_output_tokens",
    "p99_output_tokens",
    "mean_elapsed_time",
    "p50_elapsed_time",
    "p95_elapsed_time",
    "p99_elapsed_time",
    "mean_interarrival_s",
    "p95_interarrival_s",
    "interarrival_cv",
)

SCENARIO_FIELDS = (
    "scenario_id",
    "partition",
    "traffic_regime",
    "bin_index",
    "start_timestamp_s",
    "end_timestamp_s",
    "source_row_start",
    "source_row_end",
    "request_count",
    "mean_rps",
    "peak_burst_rps",
    "burst_ratio",
    "input_tokens_per_s",
    "output_tokens_per_s",
    "mean_input_tokens",
    "p95_input_tokens",
    "mean_output_tokens",
    "p95_output_tokens",
    "mean_elapsed_time",
    "p95_elapsed_time",
    "interarrival_cv",
)


def percentile(values: Iterable[float], probability: float) -> float | None:
    """Return a deterministic linearly interpolated percentile."""
    if not 0 <= probability <= 1:
        raise ValueError("Percentile probability must be between zero and one")
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _safe_mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _coefficient_of_variation(values: list[float]) -> float | None:
    if not values:
        return None
    mean = fmean(values)
    if mean <= 0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class TrafficBin:
    index: int
    start_timestamp_s: float
    bin_seconds: float
    burst_seconds: float
    source_row_start: int | None = None
    source_row_end: int | None = None
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    input_lengths: list[float] = field(default_factory=list)
    output_lengths: list[float] = field(default_factory=list)
    elapsed_times: list[float] = field(default_factory=list)
    interarrivals: list[float] = field(default_factory=list)
    previous_timestamp_s: float | None = None
    current_burst_index: int | None = None
    current_burst_count: int = 0
    peak_burst_count: int = 0

    @property
    def end_timestamp_s(self) -> float:
        return self.start_timestamp_s + self.bin_seconds

    def add(
        self,
        source_row: int,
        timestamp_s: float,
        input_tokens: int,
        output_tokens: int,
        elapsed_time: float | None,
    ) -> None:
        if not self.start_timestamp_s <= timestamp_s < self.end_timestamp_s:
            raise ValueError("Request timestamp lies outside its traffic bin")
        if self.source_row_start is None:
            self.source_row_start = source_row
        self.source_row_end = source_row
        self.request_count += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.input_lengths.append(float(input_tokens))
        self.output_lengths.append(float(output_tokens))
        if elapsed_time is not None:
            self.elapsed_times.append(elapsed_time)
        if self.previous_timestamp_s is not None:
            self.interarrivals.append(timestamp_s - self.previous_timestamp_s)
        self.previous_timestamp_s = timestamp_s

        burst_index = int((timestamp_s - self.start_timestamp_s) // self.burst_seconds)
        if burst_index == self.current_burst_index:
            self.current_burst_count += 1
        else:
            self.peak_burst_count = max(self.peak_burst_count, self.current_burst_count)
            self.current_burst_index = burst_index
            self.current_burst_count = 1

    def record(self) -> dict:
        peak_count = max(self.peak_burst_count, self.current_burst_count)
        mean_rps = self.request_count / self.bin_seconds
        peak_burst_rps = peak_count / self.burst_seconds
        return {
            "bin_index": self.index,
            "scenario_id": None,
            "partition": None,
            "traffic_regime": None,
            "start_timestamp_s": self.start_timestamp_s,
            "end_timestamp_s": self.end_timestamp_s,
            "source_row_start": self.source_row_start,
            "source_row_end": self.source_row_end,
            "request_count": self.request_count,
            "mean_rps": mean_rps,
            "peak_burst_rps": peak_burst_rps,
            "burst_ratio": peak_burst_rps / mean_rps if mean_rps else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_tokens_per_s": self.input_tokens / self.bin_seconds,
            "output_tokens_per_s": self.output_tokens / self.bin_seconds,
            "mean_input_tokens": _safe_mean(self.input_lengths),
            "p50_input_tokens": percentile(self.input_lengths, 0.50),
            "p95_input_tokens": percentile(self.input_lengths, 0.95),
            "p99_input_tokens": percentile(self.input_lengths, 0.99),
            "mean_output_tokens": _safe_mean(self.output_lengths),
            "p50_output_tokens": percentile(self.output_lengths, 0.50),
            "p95_output_tokens": percentile(self.output_lengths, 0.95),
            "p99_output_tokens": percentile(self.output_lengths, 0.99),
            "mean_elapsed_time": _safe_mean(self.elapsed_times),
            "p50_elapsed_time": percentile(self.elapsed_times, 0.50),
            "p95_elapsed_time": percentile(self.elapsed_times, 0.95),
            "p99_elapsed_time": percentile(self.elapsed_times, 0.99),
            "mean_interarrival_s": _safe_mean(self.interarrivals),
            "p95_interarrival_s": percentile(self.interarrivals, 0.95),
            "interarrival_cv": _coefficient_of_variation(self.interarrivals),
        }


def _validate_configuration(
    bin_seconds: float,
    burst_seconds: float,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    if not math.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError("bin_seconds must be positive and finite")
    if not math.isfinite(burst_seconds) or burst_seconds <= 0 or burst_seconds > bin_seconds:
        raise ValueError("burst_seconds must be positive and no larger than bin_seconds")
    ratio = bin_seconds / burst_seconds
    if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
        raise ValueError("bin_seconds must be an integer multiple of burst_seconds")
    if not (
        0 < train_fraction < 1
        and 0 < validation_fraction < 1
        and train_fraction + validation_fraction < 1
    ):
        raise ValueError("Train and validation fractions must be positive and sum to less than one")


def aggregate_trace(source: Path, bin_seconds: float, burst_seconds: float) -> tuple[list[dict], dict]:
    """Stream a timestamp-ordered BurstGPT 3 CSV into fixed-duration bins."""
    source = Path(source)
    counts = Counter()
    models = Counter()
    log_types = Counter()
    records: list[dict] = []
    origin: float | None = None
    previous_valid_timestamp: float | None = None
    current: TrafficBin | None = None

    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        missing = sorted(set(EXPECTED_COLUMNS) - set(columns))
        if missing:
            raise ValueError(f"Missing BurstGPT 3 columns: {missing}")

        for source_row, row in enumerate(reader):
            counts["total_rows"] += 1
            models[row["Model"]] += 1
            log_types[row["Log Type"]] += 1
            if row["Session ID"]:
                counts["session_id_present"] += 1
            else:
                counts["session_id_missing"] += 1

            if row["Model"] != "ChatGPT" or row["Log Type"] != "API log":
                counts["excluded_model_or_channel"] += 1
                continue
            try:
                timestamp_s = float(row["Timestamp"])
                input_tokens = int(row["Request tokens"])
                output_tokens = int(row["Response tokens"])
                if (
                    not math.isfinite(timestamp_s)
                    or timestamp_s < 0
                    or input_tokens <= 0
                    or output_tokens <= 0
                ):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                counts["invalid_or_failed"] += 1
                continue

            elapsed_time = None
            try:
                elapsed_time = float(row["Elapsed time"])
                if not math.isfinite(elapsed_time) or elapsed_time < 0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                elapsed_time = None
                counts["invalid_elapsed_time"] += 1
            try:
                if int(row["Total tokens"]) != input_tokens + output_tokens:
                    counts["total_token_mismatch"] += 1
            except (TypeError, ValueError, OverflowError):
                counts["invalid_total_tokens"] += 1

            if previous_valid_timestamp is not None and timestamp_s < previous_valid_timestamp:
                raise ValueError(
                    f"BurstGPT rows must be timestamp ordered; source row {source_row} moves backward"
                )
            previous_valid_timestamp = timestamp_s
            counts["valid_requests"] += 1

            if origin is None:
                origin = timestamp_s
                current = TrafficBin(0, origin, bin_seconds, burst_seconds)
            assert current is not None and origin is not None
            target_index = int((timestamp_s - origin) // bin_seconds)
            while current.index < target_index:
                records.append(current.record())
                next_index = current.index + 1
                current = TrafficBin(
                    next_index,
                    origin + next_index * bin_seconds,
                    bin_seconds,
                    burst_seconds,
                )
            current.add(source_row, timestamp_s, input_tokens, output_tokens, elapsed_time)

    if current is None or origin is None:
        raise ValueError("No valid ChatGPT API requests found")
    records.append(current.record())
    for name in COUNT_FIELDS:
        counts.setdefault(name, 0)
    filtering = {
        "columns": columns,
        "counts": dict(counts),
        "models": dict(models),
        "log_types": dict(log_types),
        "origin_timestamp_s": origin,
        "last_valid_timestamp_s": previous_valid_timestamp,
    }
    return records, filtering


def partition_and_label(
    records: list[dict],
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, float]:
    """Chronologically partition bins and learn RPS regimes from train only."""
    train_end = int(len(records) * train_fraction)
    validation_end = int(len(records) * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= len(records):
        raise ValueError("Trace needs enough time bins for nonempty train, validation, and test partitions")

    for index, record in enumerate(records):
        if index < train_end:
            partition = "train"
        elif index < validation_end:
            partition = "validation"
        else:
            partition = "test"
        record["partition"] = partition
        record["scenario_id"] = f"{partition}_{record['bin_index']:06d}"

    training_rates = [
        record["mean_rps"]
        for record in records[:train_end]
        if record["request_count"] > 0
    ]
    if not training_rates:
        raise ValueError("Training partition contains no requests")
    thresholds = {
        "regular_lower_p25": percentile(training_rates, 0.25),
        "regular_upper_p75": percentile(training_rates, 0.75),
        "peak_lower_p95": percentile(training_rates, 0.95),
        "extreme_peak_lower_p99": percentile(training_rates, 0.99),
    }
    assert all(value is not None for value in thresholds.values())

    for record in records:
        rate = record["mean_rps"]
        if record["request_count"] == 0:
            regime = "idle"
        elif rate < thresholds["regular_lower_p25"]:
            regime = "low"
        elif rate <= thresholds["regular_upper_p75"]:
            regime = "regular"
        elif rate < thresholds["peak_lower_p95"]:
            regime = "elevated"
        elif rate < thresholds["extreme_peak_lower_p99"]:
            regime = "peak"
        else:
            regime = "extreme_peak"
        record["traffic_regime"] = regime
    return thresholds


def _distribution(records: list[dict], field: str) -> dict[str, float | None]:
    values = [float(record[field]) for record in records if record[field] is not None]
    return {
        "mean": _safe_mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def build_summary(
    source: Path,
    records: list[dict],
    filtering: dict,
    thresholds: dict[str, float],
    bin_seconds: float,
    burst_seconds: float,
    train_fraction: float,
    validation_fraction: float,
) -> dict:
    nonempty = [record for record in records if record["request_count"] > 0]
    partition_summary = {}
    for partition in ("train", "validation", "test"):
        selected = [record for record in records if record["partition"] == partition]
        partition_summary[partition] = {
            "time_bins": len(selected),
            "nonempty_bins": sum(record["request_count"] > 0 for record in selected),
            "requests": sum(record["request_count"] for record in selected),
            "traffic_regimes": dict(Counter(record["traffic_regime"] for record in selected)),
        }
    rates = [record["mean_rps"] for record in nonempty]
    input_means = [record["mean_input_tokens"] for record in nonempty]
    output_means = [record["mean_output_tokens"] for record in nonempty]
    return {
        "schema_version": 1,
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "columns": filtering["columns"],
        },
        "filtering": {
            **filtering["counts"],
            "models": filtering["models"],
            "log_types": filtering["log_types"],
            "rule": "Model == ChatGPT, Log Type == API log, finite nonnegative timestamp, positive token lengths",
        },
        "trace": {
            "origin_timestamp_s": filtering["origin_timestamp_s"],
            "last_valid_timestamp_s": filtering["last_valid_timestamp_s"],
            "observed_span_s": filtering["last_valid_timestamp_s"] - filtering["origin_timestamp_s"],
        },
        "windowing": {
            "bin_seconds": bin_seconds,
            "burst_seconds": burst_seconds,
            "total_time_bins": len(records),
            "nonempty_time_bins": len(nonempty),
            "train_fraction": train_fraction,
            "validation_fraction": validation_fraction,
            "test_fraction": 1 - train_fraction - validation_fraction,
            "partition_method": "chronological fixed-duration bins",
        },
        "training_regime_thresholds_rps": thresholds,
        "partitions": partition_summary,
        "distributions": {
            "nonempty_bin_mean_rps": _distribution(nonempty, "mean_rps"),
            "peak_burst_rps": _distribution(nonempty, "peak_burst_rps"),
            "input_tokens_per_s": _distribution(nonempty, "input_tokens_per_s"),
            "output_tokens_per_s": _distribution(nonempty, "output_tokens_per_s"),
        },
        "correlations": {
            "mean_rps_vs_mean_input_tokens": _pearson(rates, input_means),
            "mean_rps_vs_mean_output_tokens": _pearson(rates, output_means),
            "mean_rps_vs_peak_burst_rps": _pearson(
                rates, [record["peak_burst_rps"] for record in nonempty]
            ),
        },
        "limitations": [
            "Traffic regimes use request rate only; token-rate columns support separate workload-mix analysis.",
            "Elapsed time is reported descriptively and is not assumed to be GPU service time.",
            "Session IDs are generally unavailable for API-log records.",
            "Scenario candidates are fixed-duration bins, not yet simulator-ready request traces.",
        ],
    }


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    source: Path,
    output_dir: Path,
    *,
    bin_seconds: float = 300.0,
    burst_seconds: float = 1.0,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict:
    """Analyze a trace and write a summary, all time bins, and scenario catalog."""
    source, output_dir = Path(source), Path(output_dir)
    _validate_configuration(bin_seconds, burst_seconds, train_fraction, validation_fraction)
    if not source.is_file():
        raise FileNotFoundError(source)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    records, filtering = aggregate_trace(source, bin_seconds, burst_seconds)
    thresholds = partition_and_label(records, train_fraction, validation_fraction)
    summary = build_summary(
        source,
        records,
        filtering,
        thresholds,
        bin_seconds,
        burst_seconds,
        train_fraction,
        validation_fraction,
    )

    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "time_bins.csv", records, TIME_BIN_FIELDS)
    scenarios = [record for record in records if record["request_count"] > 0]
    _write_csv(output_dir / "scenario_catalog.csv", scenarios, SCENARIO_FIELDS)
    with (output_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("BurstGPT_3.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/traffic_analysis"))
    parser.add_argument("--bin-seconds", type=float, default=300.0)
    parser.add_argument("--burst-seconds", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    summary = analyze(
        args.source,
        args.output_dir,
        bin_seconds=args.bin_seconds,
        burst_seconds=args.burst_seconds,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(
        json.dumps(
            {
                "valid_requests": summary["filtering"]["valid_requests"],
                "time_bins": summary["windowing"]["total_time_bins"],
                "thresholds_rps": summary["training_regime_thresholds_rps"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
