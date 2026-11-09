"""Tests for triage.engine: iteration-level simulator with continuous batching."""

from dataclasses import fields

import pytest

from llm_triage.simulation import (
    Action,
    Decision,
    EventType,
    FCFSCallback,
    HardwareConfig,
    Request,
    RequestState,
    ServiceProfile,
    SimState,
    SimulatorExecutor,
    TERMINAL_STATES,
    TriageKernel,
    simulate,
)

# Simple, exact timing: prefill = 1 ms per iteration, decode = 1 ms per step.
P = ServiceProfile(
    prefill_overhead_s=0.001,
    prefill_rate=None,
    prefill_quadratic_s=0.0,
    decode_rate=1000.0,
    decode_context_s=0.0,
    contention_factor=0.0,
)
HW = HardwareConfig(gpu_name="test", max_batch_size=4, kv_block_size=16, kv_total_blocks=64)


def req(rid, arrival=0.0, inp=16, out=4, slo=10.0):
    return Request(rid, arrival, inp, out, arrival + slo)


def hw(**kw):
    base = dict(gpu_name="test", max_batch_size=4, kv_block_size=16, kv_total_blocks=64)
    base.update(kw)
    return HardwareConfig(**base)


class Spy:
    """Admits everything and records every SimState it sees."""

    def __init__(self):
        self.states = []

    def decide(self, state: SimState) -> Action:
        self.states.append(state)
        return Action.ADMIT


# ---------------------------------------------------------------------------
# Lifecycle and timing
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_single_request_timing(self):
        r = simulate([req(1, inp=16, out=4)], FCFSCallback(), HW, P).requests[0]
        assert r.state == RequestState.COMPLETED
        assert r.admit_time == pytest.approx(0.0)
        assert r.first_token_time == pytest.approx(0.001)   # prefill emits token 1
        assert r.completion_time == pytest.approx(0.004)    # + 3 decode steps
        assert r.tokens_generated == 4
        assert r.kv_blocks_allocated == 0                   # released

    def test_single_token_output_finishes_at_prefill(self):
        r = simulate([req(1, out=1)], FCFSCallback(), HW, P).requests[0]
        assert r.state == RequestState.COMPLETED
        assert r.completion_time == pytest.approx(0.001)

    def test_late_completion_is_violation(self):
        res = simulate([req(1, out=4, slo=0.002)], FCFSCallback(), HW, P)
        assert res.requests[0].state == RequestState.VIOLATED
        assert res.slo_attainment == 0.0

    def test_drop(self):
        class DropAll:
            def decide(self, s):
                return Action.DROP

        res = simulate([req(i) for i in range(3)], DropAll(), HW, P)
        assert all(r.state == RequestState.DROPPED for r in res.requests)
        assert res.drop_rate == 1.0
        assert res.slo_attainment == 0.0  # drops count against attainment
        assert res.gpu_busy_time == 0.0

    def test_all_requests_reach_terminal_state(self):
        reqs = [req(i, arrival=i * 0.0005, out=3 + i % 7, slo=0.02) for i in range(50)]
        res = simulate(reqs, FCFSCallback(), hw(max_batch_size=3), P)
        assert all(r.state in TERMINAL_STATES for r in res.requests)

    def test_input_workload_not_mutated(self):
        reqs = [req(1), req(2)]
        simulate(reqs, FCFSCallback(), HW, P)
        assert all(r.state == RequestState.QUEUED and r.tokens_generated == 0 for r in reqs)

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValueError):
            simulate([req(1), req(1)], FCFSCallback(), HW, P)

        with pytest.raises(ValueError):
            simulate([req(1), req("1")], FCFSCallback(), HW, P)


# ---------------------------------------------------------------------------
# Defer & head-of-line behavior (Issue 1.3)
# ---------------------------------------------------------------------------

class TestDefer:
    def test_defer_then_admit_when_idle(self):
        calls = []

        class DeferOnce:
            def decide(self, s):
                calls.append(s.now)
                return Action.DEFER if len(calls) == 1 else Action.ADMIT

        r = simulate([req(1)], DeferOnce(), HW, P, defer_retry_s=0.01).requests[0]
        assert r.state == RequestState.COMPLETED
        assert r.defer_count == 1
        assert r.admit_time == pytest.approx(0.01)

    def test_defer_forever_hits_max_defers_bound(self):
        class DeferAll:
            def decide(self, s):
                return Action.DEFER

        r = simulate(
            [req(1, slo=100.0)], DeferAll(), HW, P, defer_retry_s=0.01, max_defers=4
        ).requests[0]
        assert r.state == RequestState.VIOLATED
        assert r.defer_count == 4
        assert r.admit_time is None
        assert r.completion_time == pytest.approx(0.03)

    def test_defer_does_not_head_of_line_block_other_candidates(self):
        """Deferring request 1 allows request 2 in the queue to be admitted
        in the very same scheduling pass."""

        class DeferFirstAdmitSecond:
            def decide(self, s: SimState) -> Action:
                return Action.DEFER if s.candidate.request_id == 1 else Action.ADMIT

        res = simulate(
            [req(1, arrival=0.0, out=2), req(2, arrival=0.0, out=2)],
            DeferFirstAdmitSecond(),
            HW,
            P,
            defer_retry_s=0.05,
            max_defers=2,
        )
        by = res.by_id()
        # Request 2 is admitted immediately at t=0.0 despite request 1 deferring!
        assert by[2].admit_time == pytest.approx(0.0)
        assert by[2].state == RequestState.COMPLETED

    def test_idle_retry_does_not_skip_arrivals(self):
        """While request 1 is on defer cooldown on an idle GPU, request 2's
        arrival at t=0.003 triggers an immediate decision for request 2."""
        seen = []

        class DeferAll:
            def decide(self, s):
                seen.append((s.now, s.candidate.request_id))
                return Action.DEFER

        simulate(
            [req(1, slo=0.5), req(2, arrival=0.003, slo=0.5)],
            DeferAll(),
            HW,
            P,
            defer_retry_s=0.1,
            max_defers=2,
        )
        assert (pytest.approx(0.003), 2) in seen


# ---------------------------------------------------------------------------
# Batch and KV-cache constraints (Issues 1.2, 1.4, 1.5)
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_batch_capacity_respected(self):
        spy = Spy()
        res = simulate([req(i, out=10) for i in range(6)], spy, hw(max_batch_size=2), P)
        assert all(it.batch_size <= 2 for it in res.iterations)
        assert all(s.active_count <= 2 for s in spy.states)
        assert len(res.completed) == 6

    def test_kv_capacity_gates_admission(self):
        # 2 blocks of 16 tokens; each prompt of 15 tokens + 1 prefill token = 16 tokens (1 block).
        spy = Spy()
        res = simulate([req(i, inp=15, out=2) for i in range(3)], spy,
                       hw(max_batch_size=100, kv_total_blocks=2), P)
        assert all(s.kv_blocks_used <= 2 for s in spy.states)
        assert all(it.kv_blocks_used <= 2 for it in res.iterations)
        by = res.by_id()
        assert by[2].admit_time >= min(by[0].completion_time, by[1].completion_time)
        assert len(res.completed) == 3

    def test_kv_blocks_synchronized_immediately_after_prefill(self):
        """Issue 1.2: when a 16-token prompt emits its 1st token at prefill,
        context_tokens becomes 17 (2 KV blocks).  Before the next scheduling
        pass runs, kv_blocks_used must already report 2 so the scheduler does
        not falsely see a free block."""
        spy = Spy()
        # Total blocks = 2. r0 has inp=16, out=5 -> after prefill it needs 2 blocks (17 tokens).
        # r1 arrives after prefill (t=0.0015). Since r0 already holds all 2 blocks,
        # the scheduler should NOT be called to admit r1 until r0 finishes!
        r0 = req(0, arrival=0.0, inp=16, out=5)
        r1 = req(1, arrival=0.0015, inp=16, out=2)
        res = simulate([r0, r1], spy, hw(kv_total_blocks=2), P)
        assert res.preemptions == 0
        assert res.by_id()[1].admit_time == pytest.approx(res.by_id()[0].completion_time)

    def test_kv_blocks_grow_during_decode(self):
        res = simulate([req(1, inp=16, out=20)], FCFSCallback(), HW, P)
        peak = max(it.kv_blocks_used for it in res.iterations)
        assert peak == 3  # 16 + 19 resident tokens = 35 → 3 blocks

    def test_preemption_when_kv_runs_out(self):
        # Both prompts of 12 tokens fit in 1 block each (even after prefill: 13 tokens).
        # Growing past 16 tokens needs a 2nd block each; with 3 total blocks, the newest is preempted.
        res = simulate([req(0, inp=12, out=20), req(1, inp=12, out=20)], FCFSCallback(),
                       hw(kv_total_blocks=3), P)
        by = res.by_id()
        assert res.preemptions >= 1
        assert by[1].preemption_count >= 1
        assert by[0].preemption_count == 0
        assert all(r.state == RequestState.COMPLETED for r in res.requests)
        assert all(it.kv_blocks_used <= 3 for it in res.iterations)

    def test_preempted_request_auto_resumes_without_drop(self):
        """Issue 1.5: a preempted request has already started service and must
        never be re-offered to callback.decide() where it could be DROPPED."""
        seen_ids = []

        class DropOnSecondSight:
            def decide(self, s: SimState) -> Action:
                rid = s.candidate.request_id
                if rid in seen_ids:
                    return Action.DROP
                seen_ids.append(rid)
                return Action.ADMIT

        res = simulate(
            [req(0, inp=12, out=20), req(1, inp=12, out=20)],
            DropOnSecondSight(),
            hw(kv_total_blocks=3),
            P,
        )
        by = res.by_id()
        assert by[1].preemption_count >= 1
        assert by[1].state == RequestState.COMPLETED
        assert seen_ids == [0, 1]

    def test_single_sequence_outgrowing_cache_does_not_preempt_innocent_batch(self):
        """Issue 1.4: if r0 outgrows the entire hardware KV-cache while r1 fits,
        r0 is terminated directly as VIOLATED without preempting r1."""
        # Add a 3rd arrival at t=0.002 so the kernel pauses at a decision point
        # after prefill (t=0.001) + 1 decode step (t=0.002) while r0 and r1 are in kernel.running.
        kernel = TriageKernel(hardware=hw(kv_total_blocks=5), profile=P)
        kernel.reset([req(0, inp=15, out=30), req(1, inp=15, out=5), req(2, arrival=0.002, inp=15, out=2)])
        kernel.apply(Action.ADMIT)
        kernel.apply(Action.ADMIT)
        assert len(kernel.running) == 2
        # Set r0's context_tokens (15 + 70 = 85) above total capacity (80 tokens = 5 blocks),
        # while r1's context_tokens (17) fits in 2 blocks.
        kernel.running[0].tokens_generated = 70
        kernel._reserve_running_kv()
        assert kernel.requests[0].state == RequestState.VIOLATED
        assert kernel.requests[1] in kernel.running
        assert kernel.requests[1].preemption_count == 0

    def test_lone_request_outgrowing_cache_is_violated(self):
        res = simulate([req(1, inp=16, out=40)], FCFSCallback(), hw(kv_total_blocks=2), P)
        assert res.requests[0].state == RequestState.VIOLATED

    def test_prompt_larger_than_cache_is_rejected_without_callback(self):
        spy = Spy()
        res = simulate([req(1, inp=100)], spy, hw(kv_total_blocks=2), P)
        assert res.requests[0].state == RequestState.VIOLATED
        assert spy.states == []


# ---------------------------------------------------------------------------
# Continuous batching, prefill/decode interleaving, and queue expiration
# (Issues 1.1, 1.6)
# ---------------------------------------------------------------------------

class TestContinuousBatching:
    def test_arrival_joins_running_batch(self):
        res = simulate([req(1, out=10), req(2, arrival=0.003, out=3)], FCFSCallback(), HW, P)
        r1, r2 = res.by_id()[1], res.by_id()[2]
        assert r2.admit_time < r1.completion_time
        # r2's prefill pauses r1's decode for one iteration (1 ms)
        assert r1.completion_time == pytest.approx(0.001 + 9 * 0.001 + 0.001)

    def test_freed_slot_refilled_next_iteration(self):
        res = simulate([req(1, out=2), req(2, out=2)], FCFSCallback(), hw(max_batch_size=1), P)
        assert res.by_id()[2].admit_time == pytest.approx(res.by_id()[1].completion_time)

    def test_simultaneous_arrivals_share_one_prefill(self):
        res = simulate([req(i) for i in range(3)], FCFSCallback(), HW, P)
        first = res.iterations[0]
        assert first.kind == "prefill" and first.batch_size == 3
        assert all(r.first_token_time == pytest.approx(0.001) for r in res.requests)

    def test_no_back_to_back_prefills_when_decode_running(self):
        """Issue 1.6: when a decode batch is running, new arrivals during a
        prefill iteration must wait for at least one decode step before
        triggering another prefill iteration."""
        # r0 arrives at 0.0 (prefill [0.0, 0.001]).
        # r1 arrives at 0.001 (prefill [0.001, 0.002] would be back-to-back if not interleaved!).
        # r2 arrives at 0.002.
        reqs = [
            req(0, arrival=0.0, out=5),
            req(1, arrival=0.001, out=5),
            req(2, arrival=0.002, out=5),
        ]
        res = simulate(reqs, FCFSCallback(), HW, P)
        kinds = [it.kind for it in res.iterations if it.kind != "idle"]
        for a, b in zip(kinds, kinds[1:]):
            assert not (a == "prefill" and b == "prefill")

    def test_iterations_are_non_overlapping(self):
        reqs = [req(i, arrival=i * 0.0007, out=5) for i in range(20)]
        its = simulate(reqs, FCFSCallback(), HW, P).iterations
        for a, b in zip(its, its[1:]):
            assert b.start >= a.end - 1e-12

    def test_fcfs_serves_overdue_requests_by_default(self):
        """Issue 1.1: by default (expire_overdue_in_queue=False), FCFS does NOT
        get free queue shedding—overdue requests remain in the queue, are
        seen with negative slo_slack, and consume wasted GPU time."""
        spy = Spy()
        res = simulate(
            [req(1, out=100), req(2, out=2, slo=0.01)],
            spy,
            hw(max_batch_size=1),
            P,
        )
        r2 = res.by_id()[2]
        assert r2.state == RequestState.VIOLATED
        assert r2.admit_time == pytest.approx(0.1)
        assert r2.gpu_time_s > 0.0
        r2_state = next(s for s in spy.states if s.candidate.request_id == 2)
        assert r2_state.candidate.slo_slack < 0.0

    def test_opt_in_expire_overdue_in_queue(self):
        res = simulate(
            [req(1, out=100), req(2, out=2, slo=0.01)],
            FCFSCallback(),
            hw(max_batch_size=1),
            P,
            expire_overdue_in_queue=True,
        )
        r2 = res.by_id()[2]
        assert r2.state == RequestState.VIOLATED
        assert r2.admit_time is None and r2.gpu_time_s == 0.0


# ---------------------------------------------------------------------------
# Callback hook, capacity gating, and cross-branch contracts
# (Issues 1.7, 2.2, 2.3, 2.4)
# ---------------------------------------------------------------------------

class TestCallbackHook:
    def test_ground_truth_output_never_exposed(self):
        spy = Spy()
        simulate([req(i, out=10 + i) for i in range(5)], spy, hw(max_batch_size=2), P)
        assert spy.states
        assert "output_tokens" not in {f.name for f in fields(SimState)}
        for s in spy.states:
            for v in (s.candidate, *s.queue_requests, *s.active_requests):
                assert not hasattr(v, "output_tokens")
                assert not hasattr(v, "true_output_tokens")

    def test_state_fields_consistent(self):
        spy = Spy()
        simulate([req(i, arrival=i * 0.001, out=6) for i in range(8)], spy,
                 hw(max_batch_size=3), P)
        for s in spy.states:
            assert s.candidate.request_id in {v.request_id for v in s.queue_requests}
            assert s.queue_length == len(s.queue_requests)
            assert s.active_count == len(s.active_requests)
            assert s.batch_utilization == pytest.approx(s.active_count / 3)
            assert s.candidate.slo_slack == pytest.approx(s.candidate.slo_deadline - s.now)
            assert s.candidate.estimated_output_tokens > 0
            assert s.candidate.estimated_service_s > 0

    def test_priority_hook_orders_candidates(self):
        class ShortestPromptFirst:
            def priority(self, view):
                return view.input_tokens

            def decide(self, s):
                return Action.ADMIT

        reqs = [req(0, inp=300), req(1, inp=100), req(2, inp=200)]
        res = simulate(reqs, ShortestPromptFirst(), hw(max_batch_size=1), P)
        order = sorted(res.requests, key=lambda r: r.admit_time)
        assert [r.request_id for r in order] == [1, 2, 0]

    def test_decide_not_called_while_batch_full(self):
        """Issue 1.7: while the batch is full, the engine does not spam
        callback.decide() on every decode step."""
        spy = Spy()
        simulate([req(1, out=5), req(2, out=3)], spy, hw(max_batch_size=1), P)
        r2_views = [s for s in spy.states if s.candidate.request_id == 2]
        assert len(r2_views) == 1
        assert r2_views[0].now == pytest.approx(0.005)
        assert r2_views[0].resources.can_admit is True

    def test_supports_string_request_ids_and_decision_objects(self):
        """Issue 2.4: Scheduler returning Decision(request_id, Action('admit'))
        and string request_ids works natively."""

        class SnapshotScheduler:
            def decide(self, snapshot: SimState) -> Decision:
                # Pick shortest input_tokens from snapshot.candidates
                chosen = min(snapshot.candidates, key=lambda c: c.input_tokens)
                return Decision(request_id=chosen.request_id, action=Action("admit"))

        reqs = [
            Request("req-a", 0.0, 64, 3, 1.0),
            Request("req-b", 0.0, 16, 3, 1.0),
        ]
        res = simulate(reqs, SnapshotScheduler(), hw(max_batch_size=1), P)
        by = res.by_id()
        assert by["req-b"].admit_time < by["req-a"].admit_time

    def test_steppable_triage_kernel_matches_simulate(self):
        """Issue 2.2: stepping TriageKernel manually produces the exact same
        outcomes and timestamps as simulate()."""
        reqs = [req(i, arrival=i * 0.001, inp=16 + i, out=3 + i % 4) for i in range(12)]
        sim_res = simulate(reqs, FCFSCallback(), hw(max_batch_size=2), P)

        kernel = TriageKernel(hardware=hw(max_batch_size=2), profile=P)
        snap = kernel.reset(reqs)
        while not kernel.is_terminal():
            assert snap is not None
            tr = kernel.apply(Action.ADMIT)
            snap = None if tr.done else kernel.snapshot()
        kern_res = kernel.result()

        key = lambda res: [(r.request_id, r.state, r.completion_time) for r in res.requests]
        assert key(sim_res) == key(kern_res)

    def test_lifecycle_event_stream_invariants(self):
        """Issue 2.3: LifecycleEvent stream is chronological, emits SERVICE_START
        and FIRST_TOKEN at most once even under preemption, and ends with EPISODE_END."""
        executor = SimulatorExecutor(
            [req(0, inp=12, out=20), req(1, inp=12, out=20)],
            hardware=hw(kv_total_blocks=3),
            profile=P,
        )
        events = list(executor.run("fcfs", FCFSCallback()))
        assert events[-1].event_type is EventType.EPISODE_END
        for a, b in zip(events, events[1:]):
            assert b.timestamp_s >= a.timestamp_s

        service_starts = [e.request_id for e in events if e.event_type is EventType.SERVICE_START]
        first_tokens = [e.request_id for e in events if e.event_type is EventType.FIRST_TOKEN]
        assert len(service_starts) == len(set(service_starts)) == 2
        assert len(first_tokens) == len(set(first_tokens)) == 2


# ---------------------------------------------------------------------------
# Accounting and determinism
# ---------------------------------------------------------------------------

class TestAccounting:
    def test_gpu_time_attribution_sums_to_busy_time(self):
        reqs = [req(i, arrival=i * 0.0003, inp=16 + i, out=3 + i % 5) for i in range(30)]
        res = simulate(reqs, FCFSCallback(), hw(max_batch_size=4, kv_total_blocks=8), P)
        assert sum(r.gpu_time_s for r in res.requests) == pytest.approx(res.gpu_busy_time)

    def test_wasted_gpu_time_counts_violations(self):
        res = simulate([req(1, out=4, slo=0.002)], FCFSCallback(), HW, P)
        assert res.wasted_gpu_time == pytest.approx(0.004)

    def test_deterministic(self):
        reqs = [req(i, arrival=i * 0.001, out=5, slo=0.02) for i in range(30)]
        a = simulate(reqs, FCFSCallback(), hw(max_batch_size=2), P)
        b = simulate(reqs, FCFSCallback(), hw(max_batch_size=2), P)
        key = lambda res: [(r.request_id, r.state, r.completion_time) for r in res.requests]
        assert key(a) == key(b)

    def test_summary(self):
        text = simulate([req(1)], FCFSCallback(), HW, P).summary()
        assert "SLO attainment" in text and "Wasted GPU time" in text


# ---------------------------------------------------------------------------
# Public package exports
# ---------------------------------------------------------------------------

class TestLLMTriagePackageExports:
    def test_imports_from_llm_triage_simulation_and_service(self):
        from llm_triage.simulation import ServiceProfile as SP
        from llm_triage.simulation import (
            FCFSCallback as FCFS,
            SimulatorExecutor as Exec,
            TriageKernel as Kernel,
            simulate as sim,
        )

        assert SP is ServiceProfile
        assert FCFS is FCFSCallback
        assert Exec is SimulatorExecutor
        assert Kernel is TriageKernel
        assert sim is simulate
