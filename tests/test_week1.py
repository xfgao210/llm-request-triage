"""Offline tests: parsing, alignment, split isolation, FCFS, and integration."""
import csv
from dataclasses import fields, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_triage.types import Action, CandidateView, ResourceView, SchedulerSnapshot
from llm_triage.scheduling.fcfs import FCFSScheduler
from llm_triage.workload.burstgpt import (TraceRow, acquire, chronological_split, load_windows,
                                        parse_trace, prepare, select_windows, sha256)
from llm_triage.workload.transforms import replay_window
from experiments.run_fcfs import run_reference, run_suite


class Week1Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def make_trace(self):
        path = self.folder / "trace.csv"
        path.write_text("Timestamp,Model,Request tokens,Response tokens,Log Type\n" +
                        "\n".join(f"{i},ChatGPT,1,1,API log" for i in range(100)) + "\n", encoding="utf-8")
        return path

    def test_parser_filters_sorts_and_counts_invalid_rows(self):
        path = self.folder / "trace.csv"
        path.write_text("Timestamp,Model,Request tokens,Response tokens,Log Type\n"
                        "10,ChatGPT,5,6,API log\n2,ChatGPT,3,4,API log\n"
                        "2,ChatGPT,7,8,API log\n1,GPT-4,1,1,API log\n"
                        "1,ChatGPT,1,1,Conversation log\nNaN,ChatGPT,1,1,API log\n"
                        "inf,ChatGPT,1,1,API log\n-1,ChatGPT,1,1,API log\n"
                        "1,ChatGPT,0,1,API log\n1,ChatGPT,1,0,API log\n"
                        "1,ChatGPT,1.5,1,API log\n1,ChatGPT,nope,1,API log\n"
                        "1,ChatGPT,,1,API log\n", encoding="utf-8")
        rows, counts = parse_trace(path)
        self.assertEqual([r.source_row for r in rows], [1, 2, 0])
        self.assertEqual(counts, dict(total_rows=13, excluded_model_or_channel=2,
                                     invalid_or_failed=8, valid_requests=3))

    def test_schema_and_empty_trace_rejected(self):
        path = self.folder / "bad.csv"
        for content in ("hello\n", "Timestamp,Model,Request tokens,Response tokens,Log Type\n"):
            path.write_text(content)
            with self.assertRaises(ValueError):
                parse_trace(path)

    def test_timestamp_ties_never_cross_splits(self):
        rows = [TraceRow(i, i // 3, 1, 1) for i in range(30)]
        splits, _ = chronological_split(rows)
        self.assertLess(max(r.timestamp_s for r in splits["train"]), min(r.timestamp_s for r in splits["validation"]))
        self.assertLess(max(r.timestamp_s for r in splits["validation"]), min(r.timestamp_s for r in splits["test"]))
        self.assertEqual(sum(map(len, splits.values())), 30)

    def test_invalid_splits_rejected(self):
        rows = [TraceRow(i, 0, 1, 1) for i in range(10)]
        with self.assertRaises(ValueError):
            chronological_split(rows)
        with self.assertRaises(ValueError):
            chronological_split(rows, train_fraction=.9, validation_fraction=.2)
        with self.assertRaises(ValueError):
            chronological_split([TraceRow(0, 2, 1, 1), TraceRow(1, 1, 1, 1), TraceRow(2, 3, 1, 1)])

    def test_disjoint_deterministic_windows(self):
        rows, _ = parse_trace(self.make_trace())
        splits, _ = chronological_split(rows)
        windows = select_windows(splits, 4, dict(train=3, validation=2, test=2))
        ids = [r.source_row for chunk in windows.values() for r in chunk]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(windows, select_windows(splits, 4, dict(train=3, validation=2, test=2)))
        with self.assertRaises(ValueError):
            select_windows(splits, 40)

    def test_native_alignment_and_deadlines(self):
        rows = [TraceRow(9, 100, 1, 2), TraceRow(10, 100, 1, 2), TraceRow(11, 104, 1, 2)]
        replay = replay_window(rows, slo_s=2)
        self.assertEqual([r.arrival_time_s for r in replay.requests], [0, 0, 4])
        self.assertEqual([r.deadline_s for r in replay.requests], [2, 2, 6])
        self.assertEqual(replay.horizon_s, 6)
        self.assertEqual(replay.time_scale, 1)
        self.assertEqual(rows[-1].timestamp_s, 104)

    def test_rate_scaling_preserves_ties(self):
        rows = [TraceRow(0, 10, 1, 1), TraceRow(1, 10, 1, 1), TraceRow(2, 14, 1, 1)]
        replay = replay_window(rows, target_rps=2)
        self.assertEqual([r.arrival_time_s for r in replay.requests], [0, 0, 1])
        self.assertEqual(replay.horizon_s, 1.5)
        self.assertEqual(len(rows) / replay.horizon_s, 2)

    def test_invalid_replay_inputs(self):
        rows = [TraceRow(0, 1, 1, 1), TraceRow(1, 2, 1, 1)]
        for bad in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replay_window(rows, target_rps=bad)
            with self.assertRaises(ValueError):
                replay_window(rows, slo_s=bad)
        for invalid in (rows[::-1], rows[:1], [rows[0], rows[0]],
                        [rows[0], replace(rows[1], timestamp_s=1)],
                        [rows[0], replace(rows[1], true_output_tokens=0)]):
            with self.assertRaises(ValueError):
                replay_window(invalid)

    def candidate(self, identity, arrival, estimate=1):
        return CandidateView(identity, arrival, 1, estimate, estimate, 10)

    def test_fcfs_earliest_arrival_stable_ties_and_feasibility(self):
        scheduler = FCFSScheduler()
        snapshot = SchedulerSnapshot(3, (self.candidate("later", 2), self.candidate("10", 1, 100),
                                        self.candidate("2", 1, .1)), ResourceView())
        self.assertEqual(scheduler.decide(snapshot).request_id, "10")
        self.assertEqual(scheduler.decide(snapshot).action, Action.ADMIT)
        blocked = replace(snapshot, resources=ResourceView(active_batch_size=1, admission_available=False))
        self.assertEqual(scheduler.decide(blocked).action, Action.DEFER)
        self.assertEqual(scheduler.decide(blocked).request_id, "10")
        # Batching feasibility is supplied by the future kernel, not guessed by FCFS.
        batching = replace(snapshot, resources=ResourceView(active_batch_size=1, admission_available=True))
        self.assertEqual(scheduler.decide(batching).action, Action.ADMIT)
        with self.assertRaises(ValueError):
            scheduler.decide(replace(snapshot, candidates=()))

    def test_hidden_output_not_in_snapshot_contract(self):
        for cls in (CandidateView, ResourceView, SchedulerSnapshot):
            self.assertNotIn("true_output_tokens", {f.name for f in fields(cls)})
            self.assertNotIn("output_tokens", {f.name for f in fields(cls)})

    def test_reference_hand_calculated_queue_and_drain(self):
        replay = replay_window([TraceRow(10, 0, 1, 1), TraceRow(11, 1, 1, 1)], slo_s=2)
        records, _, summary = run_reference(replay, prefill_rate=1, decode_rate=1)
        self.assertEqual([r["request_id"] for r in records], ["10", "11"])
        self.assertEqual([r["start"] for r in records], [0, 2])
        self.assertEqual([r["completion"] for r in records], [2, 4])
        self.assertEqual(summary["slo_attainment"], .5)
        self.assertEqual(summary["window_completed"], 1)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["dropped"], 0)

    def test_tied_arrivals_keep_source_order(self):
        replay = replay_window([TraceRow(2, 0, 1, 1), TraceRow(10, 0, 1, 1),
                                TraceRow(11, 1, 1, 1)], slo_s=10)
        records, _, _ = run_reference(replay, prefill_rate=1, decode_rate=1)
        self.assertEqual([r["start"] for r in records], [0, 2, 4])

    def test_acquisition_rejects_bad_cache_or_local_source(self):
        raw = self.folder / "raw"
        raw.mkdir()
        with self.assertRaisesRegex(ValueError, "Local source"):
            acquire(raw, self.make_trace())
        (raw / "BurstGPT_1.csv").write_text("corrupt")
        with self.assertRaisesRegex(ValueError, "Cached trace"):
            acquire(raw)

    def test_download_failure_never_publishes_unverified_data(self):
        raw = self.folder / "raw"
        def corrupt_download(url, destination):
            destination.write_text("corrupt")
        with patch("llm_triage.workload.burstgpt._download", side_effect=corrupt_download):
            with self.assertRaisesRegex(ValueError, "Downloaded trace"):
                acquire(raw)
        self.assertFalse((raw / "BurstGPT_1.csv").exists())
        self.assertFalse((raw / "BurstGPT_1.part").exists())

    def test_end_to_end_preparation_and_repeatable_baseline(self):
        source = self.make_trace()
        data = self.folder / "data"
        (data / "raw").mkdir(parents=True)
        (data / "raw/LICENSE-BurstGPT.txt").write_text("offline test license fixture")
        # Fixture bypasses only the production download digest, never parsing/splits.
        with patch("llm_triage.workload.burstgpt.acquire", return_value=source):
            manifest = prepare(data, requests_per_window=4, counts=dict(train=2, validation=2, test=2))
        windows = load_windows(data / "processed/burstgpt_windows.csv")
        training = [r for (split, _), chunk in windows.items() if split == "train" for r in chunk]
        self.assertEqual(manifest["estimated_output_tokens_train_mean"],
                         sum(r.true_output_tokens for r in training) / len(training))
        config = dict(split="validation", target_rps=[1.0, None], slo_s=2,
                      prefill_tokens_per_s=1, decode_tokens_per_s=1)
        first = run_suite(config, data / "processed", self.folder / "runs", self.folder / "summary")
        summary_bytes = (self.folder / "summary/summary.csv").read_bytes()
        second = run_suite(config, data / "processed", self.folder / "runs", self.folder / "summary")
        self.assertEqual(first, second)
        self.assertEqual(summary_bytes, (self.folder / "summary/summary.csv").read_bytes())
        self.assertEqual(len(first), 4)
        self.assertTrue(all(r["arrivals"] == r["completed"] == 4 for r in first))
        provenance = json.loads((self.folder / "summary/provenance.json").read_text())
        self.assertEqual(provenance["policy_runs"], 4)
        # Reused source IDs and tampered artifacts fail loudly.
        csv_path = data / "processed/burstgpt_windows.csv"
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with csv_path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=list(rows[0])).writerow(rows[0])
        with self.assertRaisesRegex(ValueError, "reuse"):
            load_windows(csv_path)
        with self.assertRaisesRegex(ValueError, "digest"):
            run_suite(config, data / "processed", self.folder / "runs", self.folder / "summary")


if __name__ == "__main__":
    unittest.main()
