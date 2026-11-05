"""Write stable CSV and JSON artifacts for policy comparisons."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from llm_triage.metrics import EvaluationResult, RequestRecord


@dataclass(frozen=True)
class ArtifactPaths:
    requests_csv: Path
    summary_csv: Path
    config_json: Path


def write_evaluation_artifacts(
    results: Mapping[str, EvaluationResult],
    output_dir: str | Path,
    *,
    resolved_config: Mapping[str, object],
    prefix: str = "baseline",
) -> ArtifactPaths:
    """Write flattened request rows, summary rows, and resolved configuration."""

    if not results:
        raise ValueError("At least one evaluation result is required")
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("prefix must be a nonempty string")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = ArtifactPaths(
        requests_csv=destination / f"{prefix}_requests.csv",
        summary_csv=destination / f"{prefix}_summary.csv",
        config_json=destination / f"{prefix}_config.json",
    )

    request_rows = [
        _request_row(policy_name, record)
        for policy_name, result in results.items()
        for record in result.request_records
    ]
    request_fields = [
        "policy",
        "request_id",
        "arrival_time_s",
        "deadline_s",
        "input_tokens",
        "decision_time_s",
        "service_start_time_s",
        "first_token_time_s",
        "completion_time_s",
        "status",
        "slo_met",
        "ttft_s",
        "completion_latency_s",
        "service_time_s",
    ]
    _write_csv(paths.requests_csv, request_fields, request_rows)

    summary_rows = [
        {"policy": policy_name, **result.summary}
        for policy_name, result in results.items()
    ]
    summary_fields = _ordered_union(summary_rows)
    _write_csv(paths.summary_csv, summary_fields, summary_rows)

    try:
        with paths.config_json.open("w", encoding="utf-8") as stream:
            json.dump(resolved_config, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except TypeError as error:
        raise ValueError("resolved_config must be JSON serializable") from error

    return paths


def _request_row(policy_name: str, record: RequestRecord) -> dict[str, object]:
    return {
        "policy": policy_name,
        "request_id": record.request_id,
        "arrival_time_s": record.arrival_time_s,
        "deadline_s": record.deadline_s,
        "input_tokens": record.input_tokens,
        "decision_time_s": record.decision_time_s,
        "service_start_time_s": record.service_start_time_s,
        "first_token_time_s": record.first_token_time_s,
        "completion_time_s": record.completion_time_s,
        "status": record.status.value,
        "slo_met": record.slo_met,
        "ttft_s": record.ttft_s,
        "completion_latency_s": record.completion_latency_s,
        "service_time_s": record.service_time_s,
    }


def _ordered_union(rows: list[dict[str, object]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def _write_csv(
    path: Path, fields: list[str], rows: list[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

