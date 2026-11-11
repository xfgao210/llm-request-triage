"""Freeze checks and trace timing for the final holdout runner."""

import hashlib

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")

from experiments.reproduce_week3 import compress_middle_third, summarize, verify_checkpoints
from llm_triage.simulation import Request
from llm_triage.workload.transforms import ReplayWindow


def test_middle_third_burst_preserves_request_content_and_deadline_budgets():
    requests = tuple(Request(str(i), time, 100 + i, 10 + i, time + 5)
                     for i, time in enumerate([0, 3, 4.5, 6, 8]))
    original = ReplayWindow(requests, 9, 100, 1)
    shifted = compress_middle_third(original, 3)
    assert [request.arrival_time_s for request in shifted.requests] == [0, 3, 3.5, 4, 6]
    assert shifted.horizon_s == 7
    assert [(r.request_id, r.input_tokens, r.true_output_tokens) for r in shifted.requests] == [
        (r.request_id, r.input_tokens, r.true_output_tokens) for r in original.requests
    ]
    assert all(r.deadline_s - r.arrival_time_s == 5 for r in shifted.requests)
    assert [request.arrival_time_s for request in original.requests] == [0, 3, 4.5, 6, 8]


def test_modified_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "dqn_seed7.npz"
    expected = [{"seed": 7, "sha256": hashlib.sha256(b"frozen").hexdigest()}]
    path.write_bytes(b"frozen")
    assert verify_checkpoints(tmp_path, expected) == {"dqn_seed7": path}
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="seed 7 differs"):
        verify_checkpoints(tmp_path, expected)


def test_summary_pairs_by_window_and_keeps_missing_ttft_missing():
    def row(policy, window, attainment):
        return dict(
            scenario="trace", source_rps=1.2, policy=policy, window=window,
            arrivals=10, slo_attainment=attainment, drop_rate=0.0, unfinished_rate=0.0,
            window_goodput_rps=0.0, p95_ttft_s=None,
        )

    result = summarize([row("fcfs", 1, .5), row("fcfs", 2, .9),
                        row("dqn_seed7", 2, .8), row("dqn_seed7", 1, .6)])[1]
    assert result["delta_vs_fcfs_pp"] == pytest.approx(0)
    assert result["windows_above_fcfs"] == result["windows_below_fcfs"] == 1
    assert result["mean_window_p95_ttft_s"] is None
