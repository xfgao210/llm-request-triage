"""Plot an existing traffic analysis and summarize continuous candidate periods.

No scheduler runs or performance scores are used to identify candidate periods.
Timestamps retain the source CSV's relative seconds; they are not calendar dates.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def read_bins(analysis_dir):
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    with (analysis_dir / "time_bins.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Traffic analysis has no time bins")
    numeric = ("start_timestamp_s", "end_timestamp_s", "mean_rps", "peak_burst_rps",
               "input_tokens", "output_tokens", "request_count")
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
        if any(not math.isfinite(row[key]) for key in numeric):
            raise ValueError("Traffic bins contain nonfinite values")
    if any(a["end_timestamp_s"] != b["start_timestamp_s"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Traffic bins must form a continuous timeline")
    if (len(rows) != summary["windowing"]["total_time_bins"]
            or sum(row["request_count"] for row in rows) != summary["filtering"]["valid_requests"]
            or rows[0]["start_timestamp_s"] != summary["trace"]["origin_timestamp_s"]):
        raise ValueError("Traffic bins do not match the analysis summary")
    return rows, summary


def summarize_periods(rows, period_seconds, bin_seconds):
    if (not math.isfinite(period_seconds) or period_seconds < bin_seconds
            or not math.isclose(period_seconds / bin_seconds, round(period_seconds / bin_seconds))):
        raise ValueError("period_seconds must be an integer multiple of the analysis bin duration")
    per_period = round(period_seconds / bin_seconds)
    periods = []
    for offset in range(0, len(rows), per_period):
        chunk = rows[offset:offset + per_period]
        start, end = chunk[0]["start_timestamp_s"], chunk[-1]["end_timestamp_s"]
        requests = int(sum(row["request_count"] for row in chunk))
        inputs = int(sum(row["input_tokens"] for row in chunk))
        outputs = int(sum(row["output_tokens"] for row in chunk))
        rates = sorted(row["mean_rps"] for row in chunk)
        peak = max(row["peak_burst_rps"] for row in chunk)
        periods.append(dict(
            period_id=f"period_{len(periods):06d}", start_timestamp_s=start, end_timestamp_s=end,
            duration_s=end - start, complete=len(chunk) == per_period,
            requests=requests, mean_rps=requests / (end - start),
            peak_burst_rps=peak, p95_bin_rps=rates[math.ceil(0.95 * len(rates)) - 1],
            input_tokens_per_s=inputs / (end - start), output_tokens_per_s=outputs / (end - start),
            mean_input_tokens=inputs / requests if requests else None,
            mean_output_tokens=outputs / requests if requests else None,
            empty_bin_fraction=sum(row["request_count"] == 0 for row in chunk) / len(chunk),
            burst_ratio=peak / (requests / (end - start)) if requests else None,
        ))
    return periods


def idle_gaps(rows):
    gaps = []
    start = end = None
    for row in rows:
        if row["request_count"] == 0:
            start = row["start_timestamp_s"] if start is None else start
            end = row["end_timestamp_s"]
        elif start is not None:
            gaps.append(dict(start_timestamp_s=start, end_timestamp_s=end, duration_s=end - start))
            start = end = None
    if start is not None:
        gaps.append(dict(start_timestamp_s=start, end_timestamp_s=end, duration_s=end - start))
    return gaps


def write_csv(path, rows, fields=None):
    fields = list(rows[0]) if rows else fields
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_timelines(rows, periods, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError('Install plotting dependencies: python -m pip install -e ".[analysis]"') from error
    origin = rows[0]["start_timestamp_s"]

    def timestamp_axis(axis):
        from matplotlib.ticker import MaxNLocator, StrMethodFormatter
        secondary = axis.secondary_xaxis(
            "top", functions=(lambda days: origin + days * 86400,
                              lambda timestamp: (timestamp - origin) / 86400),
        )
        secondary.set_xlabel("Original CSV Timestamp (seconds; use these values in the range config)")
        secondary.xaxis.set_major_locator(MaxNLocator(nbins=6))
        secondary.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    days = [(row["start_timestamp_s"] - origin) / 86400 for row in rows]
    durations = [row["end_timestamp_s"] - row["start_timestamp_s"] for row in rows]
    fig, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [2, 1.3, 2, 0.5]})
    axes[0].plot(days, [row["mean_rps"] for row in rows], linewidth=0.65, color="#2166ac")
    axes[0].set_ylabel("Mean requests/s")
    axes[1].plot(days, [row["peak_burst_rps"] for row in rows], linewidth=0.65, color="#b35806")
    axes[1].set_ylabel("Peak short-bin RPS")
    axes[2].plot(days, [row["input_tokens"] / duration for row, duration in zip(rows, durations)],
                 label="Input", linewidth=0.65)
    axes[2].plot(days, [row["output_tokens"] / duration for row, duration in zip(rows, durations)],
                 label="Output", linewidth=0.65)
    axes[2].set_ylabel("Tokens/s")
    axes[2].legend(loc="upper right")
    axes[3].step(days, [int(row["request_count"] > 0) for row in rows], where="post", linewidth=0.7)
    axes[3].set_yticks([0, 1], ["No records", "Records"])
    axes[3].set_xlabel(f"Days from source timestamp {origin:.15g} seconds")
    axes[0].set_xlim(0, days[-1] + durations[-1] / 86400)
    timestamp_axis(axes[0])
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("BurstGPT 3: complete traffic timeline (valid ChatGPT API requests)")
    fig.savefig(output / "traffic_timeline.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    active = [period for period in periods if period["requests"]]
    for axis, field, label in zip(axes, ("mean_input_tokens", "mean_output_tokens"),
                                  ("Mean input tokens/request", "Mean output tokens/request")):
        points = axis.scatter([period["mean_rps"] for period in active],
                              [period[field] for period in active],
                              c=[period["empty_bin_fraction"] for period in active], s=12,
                              vmin=0, vmax=1, cmap="viridis", alpha=0.65)
        axis.set_xlabel("Period mean requests/s")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    fig.colorbar(points, ax=axes, label="Fraction of bins with no records")
    fig.savefig(output / "request_mix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
    period_days = [(period["start_timestamp_s"] - origin) / 86400 for period in periods]
    for axis, field, label in zip(axes, ("mean_input_tokens", "mean_output_tokens"),
                                  ("Mean input tokens/request", "Mean output tokens/request")):
        axis.plot(period_days, [period[field] if period[field] is not None else float("nan")
                                for period in periods], linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel(f"Days from source timestamp {origin:.15g} seconds")
    axes[0].set_xlim(0, (periods[-1]["end_timestamp_s"] - origin) / 86400)
    timestamp_axis(axes[0])
    fig.suptitle("Request size over time (request-weighted period means; gaps have no records)")
    fig.savefig(output / "request_size_timeline.png", dpi=160)
    plt.close(fig)


def report(analysis_dir, output, *, period_seconds=3600, plots=True):
    analysis_dir, output = Path(analysis_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, summary = read_bins(analysis_dir)
    periods = summarize_periods(rows, period_seconds, summary["windowing"]["bin_seconds"])
    gaps = idle_gaps(rows)
    candidates = []
    eligible = [period for period in periods if period["requests"] and period["complete"]]
    for metric in ("mean_rps", "input_tokens_per_s", "output_tokens_per_s", "peak_burst_rps"):
        for rank, period in enumerate(sorted(eligible, key=lambda item: (-item[metric], item["start_timestamp_s"]))[:20], 1):
            candidates.append(dict(ranked_by=metric, rank=rank, **period))
    output.mkdir(parents=True)
    write_csv(output / "periods.csv", periods)
    write_csv(output / "candidate_periods.csv", candidates, ["ranked_by", "rank", *periods[0]])
    write_csv(output / "idle_gaps.csv", gaps, ["start_timestamp_s", "end_timestamp_s", "duration_s"])
    if plots:
        plot_timelines(rows, periods, output)
    notes = [
        "# BurstGPT 3 time-range selection", "",
        f"Source SHA256: `{summary['source']['sha256']}`.", "",
        f"Analysis bins: {summary['windowing']['bin_seconds']:g} seconds. "
        f"Continuous candidate periods: {period_seconds:g} seconds.", "",
        "Timestamps are source-relative seconds, not calendar dates. Candidate rankings use traffic metadata only.",
        "The top plot axis shows original CSV timestamps; the bottom axis shows days from the first valid request.",
        f"Conversion: CSV Timestamp = {rows[0]['start_timestamp_s']:.15g} + days × 86400.",
        "Copy exact start_timestamp_s/end_timestamp_s values from periods.csv or candidate_periods.csv into the config.",
        "The final short period, if present, is excluded from candidate rankings.", "",
        "Review traffic_timeline.png, request_size_timeline.png and request_mix.png,",
        "then choose an earlier representative training range,",
        "a subsequent validation range, and disjoint later ordinary and peak-heavy test ranges.",
        "The old 60/20/20 automatic partition is not a constraint on an explicit range config.", "",
        "No-record intervals may be inactivity or missing collection; do not assume that all gaps are real idle time.",
        "High request rate is not necessarily high GPU demand: inspect input/output token rates too.", "",
        "## Highest mean-RPS candidate periods", "",
        "| Source start | Source end (exclusive) | Mean RPS | Peak short-bin RPS | No-record fraction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in [item for item in candidates if item["ranked_by"] == "mean_rps"][:10]:
        notes.append(f"| {item['start_timestamp_s']:.15g} | {item['end_timestamp_s']:.15g} | "
                     f"{item['mean_rps']:.3f} | {item['peak_burst_rps']:.1f} | {item['empty_bin_fraction']:.1%} |")
    if not plots:
        notes.append("\nPlots were disabled for this report.")
    (output / "REPORT.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return dict(periods=len(periods), idle_gaps=len(gaps), candidates=len(candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/traffic_analysis"))
    parser.add_argument("--output", type=Path, default=Path("results/traffic_analysis/report"))
    parser.add_argument("--period-seconds", type=float, default=3600)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    print(json.dumps(report(args.analysis_dir, args.output, period_seconds=args.period_seconds,
                            plots=not args.no_plots), indent=2))


if __name__ == "__main__":
    main()
