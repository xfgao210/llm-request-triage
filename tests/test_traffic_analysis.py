"""Tests for BurstGPT 3 traffic analysis and scenario construction."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_burstgpt_traffic import analyze, percentile


FIELDS = [
    "Timestamp",
    "Session ID",
    "Elapsed time",
    "Model",
    "Request tokens",
    "Response tokens",
    "Total tokens",
    "Log Type",
]


class TrafficAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def write_trace(self) -> Path:
        path = self.folder / "BurstGPT_3.csv"
        rows = [
            [0, "", 1, "ChatGPT", 0, 1, 1, "API log"],
            [1, "", 1, "GPT-4", 1, 1, 2, "API log"],
            [2, "session", 1, "ChatGPT", 1, 1, 2, "Conversation log"],
        ]
        origin = 100.0
        for bin_index, count in enumerate(range(1, 11)):
            for offset in range(count):
                timestamp = origin + bin_index * 10 + offset * 9 / count
                rows.append([
                    timestamp,
                    "",
                    offset + 1,
                    "ChatGPT",
                    10 + bin_index,
                    2 + offset,
                    12 + bin_index + offset,
                    "API log",
                ])
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        return path

    def test_analysis_builds_chronological_scenarios_and_train_thresholds(self):
        source = self.write_trace()
        output = self.folder / "analysis"
        summary = analyze(source, output, bin_seconds=10, burst_seconds=1)

        self.assertEqual(summary["filtering"]["valid_requests"], 55)
        self.assertEqual(summary["filtering"]["invalid_or_failed"], 1)
        self.assertEqual(summary["filtering"]["excluded_model_or_channel"], 2)
        self.assertEqual(summary["filtering"]["total_token_mismatch"], 0)
        self.assertEqual(summary["filtering"]["invalid_elapsed_time"], 0)
        self.assertEqual(summary["windowing"]["total_time_bins"], 10)
        self.assertEqual(summary["partitions"]["train"]["time_bins"], 6)
        self.assertEqual(summary["partitions"]["validation"]["time_bins"], 2)
        self.assertEqual(summary["partitions"]["test"]["time_bins"], 2)
        self.assertEqual(summary["partitions"]["train"]["requests"], 21)
        self.assertAlmostEqual(
            summary["training_regime_thresholds_rps"]["regular_lower_p25"], 0.225
        )
        self.assertAlmostEqual(
            summary["training_regime_thresholds_rps"]["extreme_peak_lower_p99"], 0.595
        )

        with (output / "time_bins.csv").open(encoding="utf-8", newline="") as stream:
            bins = list(csv.DictReader(stream))
        with (output / "scenario_catalog.csv").open(encoding="utf-8", newline="") as stream:
            scenarios = list(csv.DictReader(stream))
        persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(len(bins), 10)
        self.assertEqual(len(scenarios), 10)
        self.assertEqual([row["partition"] for row in bins], ["train"] * 6 + ["validation"] * 2 + ["test"] * 2)
        self.assertEqual(bins[0]["traffic_regime"], "low")
        self.assertEqual(bins[2]["traffic_regime"], "regular")
        self.assertEqual(bins[-1]["traffic_regime"], "extreme_peak")
        self.assertEqual(persisted["source"]["sha256"], summary["source"]["sha256"])

    def test_empty_bins_are_preserved_but_not_scenarios(self):
        source = self.folder / "sparse.csv"
        with source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            for timestamp in (0, 10, 30, 40):
                writer.writerow([timestamp, "", 1, "ChatGPT", 1, 1, 2, "API log"])
        output = self.folder / "sparse-analysis"
        summary = analyze(source, output, bin_seconds=10, burst_seconds=1, train_fraction=0.4, validation_fraction=0.2)
        with (output / "time_bins.csv").open(encoding="utf-8", newline="") as stream:
            bins = list(csv.DictReader(stream))
        with (output / "scenario_catalog.csv").open(encoding="utf-8", newline="") as stream:
            scenarios = list(csv.DictReader(stream))
        self.assertEqual(len(bins), 5)
        self.assertEqual(bins[2]["traffic_regime"], "idle")
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(summary["windowing"]["nonempty_time_bins"], 4)

    def test_schema_ordering_configuration_and_output_are_validated(self):
        missing = self.folder / "missing.csv"
        missing.write_text("Timestamp,Model\n1,ChatGPT\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing BurstGPT 3 columns"):
            analyze(missing, self.folder / "missing-out")

        unordered = self.folder / "unordered.csv"
        with unordered.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerow([2, "", 1, "ChatGPT", 1, 1, 2, "API log"])
            writer.writerow([1, "", 1, "ChatGPT", 1, 1, 2, "API log"])
        with self.assertRaisesRegex(ValueError, "timestamp ordered"):
            analyze(unordered, self.folder / "unordered-out")

        with self.assertRaisesRegex(ValueError, "integer multiple"):
            analyze(self.write_trace(), self.folder / "bad-config", bin_seconds=10, burst_seconds=3)

        output = self.folder / "already-there"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            analyze(self.write_trace(), output)

    def test_percentile(self):
        self.assertIsNone(percentile([], 0.5))
        self.assertEqual(percentile([1], 0.5), 1)
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        with self.assertRaises(ValueError):
            percentile([1], 1.1)


if __name__ == "__main__":
    unittest.main()
