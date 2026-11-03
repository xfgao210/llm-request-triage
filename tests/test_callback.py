"""Tests for triage.callback: SimState, RequestView, and Action coercion."""

from dataclasses import fields

import pytest

from triage.callback import CandidateView, Decision, RequestView, SimState
from triage.request import Action, Request, RequestState


def view(rid=1, inp=64, now=1.0):
    return RequestView(rid, 0.0, inp, 10.0, 10.0 - now, now, RequestState.QUEUED, 0, 0, 0)


def state(active=0, kv_used=0, inp=64):
    cand = view(inp=inp)
    return SimState(
        now=1.0, candidate=cand, queue_length=1, queue_requests=(cand,),
        active_count=active, active_requests=(), pending_prefill_count=0,
        pending_prefill_tokens=0, max_batch_size=4, batch_utilization=active / 4,
        kv_block_size=16, kv_blocks_used=kv_used, kv_blocks_total=10,
        kv_utilization=kv_used / 10, completed_count=0, violated_count=0,
        dropped_count=0, preempted_count=0,
    )


def test_request_view_has_no_ground_truth():
    field_names = {f.name for f in fields(RequestView)}
    assert "output_tokens" not in field_names
    assert "true_output_tokens" not in field_names
    assert CandidateView is RequestView


def test_views_and_state_are_frozen():
    with pytest.raises(AttributeError):
        view().input_tokens = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        state().now = 2.0  # type: ignore[misc]


def test_candidate_fits_and_scheduler_snapshot_aliases():
    s = state()
    assert s.candidate_fits()                  # 4 blocks needed, 10 free
    assert s.resources.can_admit is True
    assert s.resources.admission_available is True
    assert s.now_s == 1.0
    assert s.candidates == (s.candidate,)
    assert not state(active=4).candidate_fits()      # batch full
    assert not state(kv_used=7).candidate_fits()     # 3 free < 4 needed
    assert state(kv_used=6).candidate_fits()
    assert state(kv_used=7).kv_blocks_free == 3


def test_action_coercion():
    assert Action.coerce("admit") is Action.ADMIT
    assert Action.coerce("DEFER") is Action.DEFER
    assert Action.coerce(2) is Action.DROP
    assert Decision("r1", "admit").action is Action.ADMIT
    with pytest.raises(ValueError):
        Action.coerce(True)
    with pytest.raises(ValueError):
        Action.coerce("invalid")


def test_request_validation_rejects_nan_inf_bool():
    with pytest.raises(ValueError):
        Request(1, float("nan"), 16, 4, 1.0)
    with pytest.raises(ValueError):
        Request(1, 0.0, True, 4, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Request("", 0.0, 16, 4, 1.0)
