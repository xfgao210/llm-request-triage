"""Whole-range replay and retrospective reporting; reporting never resets state."""

import csv
import json
import math
from pathlib import Path
import statistics

from experiments import adaptation_stress as stress
from experiments.baseline_capacity_sweep import _percentile, write_csv, write_json
from llm_triage.workload.burstgpt import sha256
from llm_triage.metrics.time_bins import TimeBins


def first_test_window(manifest, *, source_minutes=None, source_start_s=None):
    """Full first range, or a declared pilot from its first recorded-data bin.

    Pilot boundaries use traffic only. No requests within the selected interval
    are sampled. Full-range mode retains the original boundaries and idle gaps.
    """
    if source_minutes is not None and (type(source_minutes) is not int or source_minutes < 5
                                      or source_minutes * 60 % manifest["range_config"]["episode_seconds"]):
        raise ValueError("Pilot minutes must align with prepared bins; use None for the full range")
    ranges = sorted((r for r in manifest["range_config"]["ranges"] if r["split"] == "test"),
                    key=lambda r: r["start_timestamp_s"])
    if not ranges:
        raise ValueError("No explicit test range")
    period = dict(ranges[0])
    if source_start_s is not None and (
            source_minutes is None or type(source_start_s) is not int
            or not period["start_timestamp_s"] <= source_start_s < period["end_timestamp_s"]
            or (source_start_s-period["start_timestamp_s"]) % manifest["range_config"]["episode_seconds"]):
        raise ValueError("Pilot start must be bin-aligned inside the first test range, with a pilot duration")
    bins = [b for b in manifest["windows"] + manifest.get("empty_windows", [])
            if b["split"] == "test" and period["start_timestamp_s"] <= b["start_timestamp_s"]
            < period["end_timestamp_s"]]
    span = period["end_timestamp_s"] - period["start_timestamp_s"]
    if sum(b["end_timestamp_s"] - b["start_timestamp_s"] for b in bins) != span:
        raise ValueError("Manifest must cover the complete first test range")
    if source_minutes is not None:
        nonempty = [b for b in bins if b["requests"]]
        if not nonempty:
            raise ValueError("First test range is empty")
        period["start_timestamp_s"] = (min(b["start_timestamp_s"] for b in nonempty)
                                       if source_start_s is None else source_start_s)
        period["end_timestamp_s"] = min(period["end_timestamp_s"], period["start_timestamp_s"] + source_minutes*60)
        bins = [b for b in bins if period["start_timestamp_s"] <= b["start_timestamp_s"] < period["end_timestamp_s"]]
    count = sum(b["requests"] for b in bins)
    if not count:
        raise ValueError("First test range is empty")
    return dict(period, name=period["group"] + ("_full" if source_minutes is None else f"_pilot_{source_minutes}min"),
                requests=count, pilot_source_minutes=source_minutes,
                selection=("entire first explicit test range; no sampling" if source_minutes is None else
                           "continuous pilot with every request retained; default start is first nonempty bin"))


def run(training_dir, data_dir, output, *, multiplier=8, seed=7, include_adapting=False,
        online_min_seconds=60.0, online_max_updates=None, device="cpu", pilot_source_minutes=None,
        progress_interval_s=30, pilot_start_timestamp_s=None):
    """Four policies by default. Optional adaptation uses an explicit time gate.

    The shared runner receives ONE full-range window, hence exactly one
    environment per policy. Only the end of the entire range is fully drained.
    Completed policy cases are resumable; interrupted cases restart from zero.
    """
    training_dir, data_dir, output = map(Path, (training_dir, data_dir, output))
    if type(include_adapting) is not bool:
        raise ValueError("include_adapting must be boolean")
    manifest_path = data_dir / "processed/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    window = first_test_window(manifest, source_minutes=pilot_source_minutes, source_start_s=pilot_start_timestamp_s)
    policies = ["fcfs", "length_aware", "static_slo_aware", "frozen"]
    if include_adapting:
        policies.append("adapting")
    plan = dict(protocol="continuous_first_test_v2", window=window, multiplier=multiplier,
                seed=seed, policies=policies, source_training_record_sha256=sha256(training_dir / "run.json"),
                manifest_sha256=sha256(manifest_path), base_report_source_seconds=300,
                reporting_resets_state=False, request_sampling=False, tail="full drain only after final arrival",
                log_transitions=False, include_adapting=include_adapting,
                pilot_source_minutes=pilot_source_minutes, progress_interval_s=progress_interval_s,
                pilot_start_timestamp_s=pilot_start_timestamp_s,
                online_min_simulated_seconds=online_min_seconds, online_max_updates=online_max_updates,
                device=device, interpretation="Exploratory reused test range, not an untouched holdout")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "continuous_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Continuous replay settings changed; use a new RUN_NAME")
    write_json(path, plan)
    return stress.run(training_dir, data_dir, output / "evaluation", [window],
                      multipliers=[multiplier], seeds=[seed], policies=policies,
                      max_requests=window["requests"], timeline_source_bin_s=300,
                      epsilon=0.0, learning_rate=1e-5, batch_size=64, warmup=2048,
                      update_interval=32, capacity=50000, min_update_seconds=online_min_seconds,
                      min_new_outcomes=32, max_updates=online_max_updates,
                      log_transitions=False, device=device, progress_interval_s=progress_interval_s)


def build_reports(output, data_dir, *, cadences=(5, 10, 15), min_arrivals=30, attainment_target=.8):
    """Calculate each cadence directly from request outcomes, not averaged rates.

    SLO/drop metrics follow arrival cohorts. Goodput follows completion time,
    including completions of requests arriving in an earlier reporting bin.
    Empty-bin rates are undefined. All timing inputs here are simulated seconds
    except explicitly source-labelled timestamps and reporting cadences.
    """
    output, data_dir = Path(output), Path(data_dir)
    if (not cadences or len(set(cadences)) != len(cadences)
            or any(type(c) is not int or c < 5 or c % 5 for c in cadences)):
        raise ValueError("Cadences must be distinct positive multiples of five minutes")
    if type(min_arrivals) is not int or min_arrivals < 1 or not 0 <= attainment_target <= 1:
        raise ValueError("Invalid stability analysis settings")
    plan = json.loads((output / "continuous_plan.json").read_text())
    manifest_path = data_dir / "processed/manifest.json"
    if sha256(manifest_path) != plan["manifest_sha256"]:
        raise ValueError("Prepared manifest changed")
    manifest = json.loads(manifest_path.read_text())
    evaluation = output / "evaluation"
    state = json.loads((evaluation / "run.json").read_text())
    if state["status"] != "completed":
        raise ValueError("Finish all selected policies before reporting")
    window, multiplier = plan["window"], plan["multiplier"]
    origin, end = window["start_timestamp_s"], window["end_timestamp_s"]
    summaries, stability = [], []
    for policy in plan["policies"]:
        name = f"{policy}_seed{plan['seed']}" if policy in ("frozen", "adapting") else policy
        folder = evaluation / window["name"] / f"load_{multiplier:g}" / name
        marker = json.loads((folder / "case.json").read_text())
        if marker["status"] != "completed" or any(sha256(folder / p) != h for p,h in marker["files"].items()):
            raise ValueError("Policy output checksum differs")
        base = json.loads((folder / "timeline.json").read_text())
        for cadence in cadences:
            source_width = cadence * 60
            bins = [dict(policy=policy, cadence_minutes=cadence, source_start_s=origin+i*source_width,
                         source_end_s=min(end, origin+(i+1)*source_width), arrivals=0, completed=0,
                         slo_met=0, dropped=0, late=0, unfinished=0, first_token_count=0,
                         ttft_target_met=0, interval_on_time_completions=0, queue_area=0.0,
                         queue_max=0, busy_area=0.0, updates_in_bin=0, input_tokens=0,
                         output_tokens=0, _ttfts=[])
                    for i in range(math.ceil((end-origin)/source_width))]
            source_edges = [b["source_start_s"] for b in bins] + [end]
            source_clock = TimeBins(source_edges)
            sim_clock = TimeBins([(t-origin)/multiplier for t in source_edges])
            with (folder / "request_diagnostics.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    arrival = float(row["arrival_time_s"])
                    index = sim_clock.index(arrival)
                    if index is None:
                        raise ValueError("Request arrival outside the selected reporting range")
                    b = bins[index]
                    b["arrivals"] += 1
                    status, met = row["status"], row["slo_met"] == "True"
                    if status not in ("completed", "dropped", "unfinished"):
                        raise ValueError("Unknown request status")
                    b[status] += 1
                    b["slo_met"] += met
                    b["late"] += status == "completed" and not met
                    if row["ttft_s"]:
                        ttft = float(row["ttft_s"])
                        b["first_token_count"] += 1
                        b["ttft_target_met"] += ttft <= 1.0
                        b["_ttfts"].append(ttft)
                    if met:
                        finish_index = sim_clock.index(arrival + float(row["completion_latency_s"]))
                        if finish_index is not None:
                            bins[finish_index]["interval_on_time_completions"] += 1
            if sum(b["arrivals"] for b in bins) != window["requests"]:
                raise ValueError("Reports must include every selected request")
            # The checked base timeline has contiguous five-source-minute bins.
            # Group them by ordinal, avoiding roundoff in inverse source clocks.
            if len(base) != math.ceil((end-origin)/300):
                raise ValueError("Expected a complete five-minute base timeline")
            for base_index, row in enumerate(base):
                index = base_index // (cadence // 5)
                duration = row["sim_end_s"] - row["sim_start_s"]
                bins[index]["queue_area"] += row["mean_queue"] * duration
                bins[index]["busy_area"] += row["busy_fraction"] * duration
                bins[index]["queue_max"] = max(bins[index]["queue_max"], row["queue_max"])
            for row in manifest["windows"]:
                if row["split"] == "test" and origin <= row["start_timestamp_s"] < end:
                    b = bins[source_clock.index(row["start_timestamp_s"])]
                    b["input_tokens"] += row["input_tokens"]
                    b["output_tokens"] += row["output_tokens"]
            updates_path = folder / f"{policy}_updates.csv"
            if updates_path.exists():
                with updates_path.open(newline="") as stream:
                    for row in csv.DictReader(stream):
                        t = float(row["source_timestamp_s"])
                        if origin <= t < end:
                            bins[source_clock.index(t)]["updates_in_bin"] += 1
            cumulative = 0
            for b in bins:
                n = b["arrivals"]
                seconds = (b["source_end_s"]-b["source_start_s"]) / multiplier
                for numerator, metric in (("slo_met", "slo_attainment"), ("dropped", "drop_rate"),
                                          ("late", "late_rate"), ("unfinished", "unfinished_rate"),
                                          ("first_token_count", "first_token_coverage"),
                                          ("ttft_target_met", "ttft_target_attainment")):
                    b[metric] = b[numerator]/n if n else None
                b["p95_ttft_s"] = _percentile(b.pop("_ttfts"), .95)
                b["completion_goodput_rps"] = b["interval_on_time_completions"] / seconds
                b["offered_rps"] = n / seconds
                b["mean_input_tokens"] = b["input_tokens"]/n if n else None
                b["mean_output_tokens"] = b["output_tokens"]/n if n else None
                b["mean_queue"] = b.pop("queue_area") / seconds
                b["busy_fraction"] = b.pop("busy_area") / seconds
                cumulative += b["updates_in_bin"]
                b["cumulative_updates_in_arrival_range"] = cumulative
            # Append policies to an in-memory report, then publish combined files below.
            summaries.extend(bins)
            eligible = [b for b in bins if b["arrivals"] >= min_arrivals]
            rates = [b["slo_attainment"] for b in eligible]
            duration = sum(b["source_end_s"]-b["source_start_s"] for b in eligible)
            stability.append(dict(policy=policy, cadence_minutes=cadence, min_arrivals=min_arrivals,
                                  eligible_bins=len(eligible), total_bins=len(bins),
                                  eligible_source_time_fraction=duration/(end-origin),
                                  eligible_arrivals=sum(b["arrivals"] for b in eligible),
                                  attainment_target=attainment_target,
                                  mean_bin_attainment=statistics.fmean(rates) if rates else None,
                                  std_bin_attainment=statistics.pstdev(rates) if rates else None,
                                  p10_bin_attainment=_percentile(rates, .1),
                                  fraction_eligible_time_below_target=(sum(b["source_end_s"]-b["source_start_s"]
                                      for b in eligible if b["slo_attainment"] < attainment_target)/duration
                                      if duration else None)))
    for cadence in cadences:
        write_csv(output / f"timeline_{cadence}min.csv", [b for b in summaries if b["cadence_minutes"] == cadence])
    write_csv(output / "stability.csv", stability)
    write_json(output / "report_settings.json", dict(cadences=list(cadences), min_arrivals=min_arrivals,
               attainment_target=attainment_target, rates="eventual arrival-cohort outcomes; empty rates undefined",
               goodput="on-time completions occurring in interval, regardless of arrival bin",
               tail="full drain at final range end only; tail completions excluded from interval goodput"))
    return stability
