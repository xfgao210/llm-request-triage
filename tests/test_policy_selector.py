from dataclasses import replace
import json

import pytest

from llm_triage.scheduling import BurstSwitchScheduler, FCFSScheduler
from llm_triage.scheduling.policy_selector import (
    SELECTOR_FEATURES, LearnedSwitchScheduler, custom_probability, fit_selector, selector_features,
)
from llm_triage.simulation import Action, HardwareConfig, Request, ServiceProfile, TriageKernel


def test_fork_preserves_aliases_and_isolates_source_and_other_branch():
    observer = []
    kernel = TriageKernel(arrival_observer=observer.append)
    kernel.reset([Request("a", 0, 16, 3, 10), Request("b", 1, 16, 3, 10)])
    left, right = kernel.fork(), kernel.fork()
    assert left.queue[0] is left.requests[0]
    assert left._pending[0] is left.requests[0]
    left.apply(Action.ADMIT)
    assert kernel.requests[0].admit_time is None
    assert right.requests[0].admit_time is None
    assert kernel.now == 0
    assert observer == [0]
    assert left._arrival_observer is None
    assert left.kv is not kernel.kv


def test_counterfactual_drains_identical_cohort_and_excludes_later_arrivals():
    from experiments.train_policy_selector import counterfactual

    class Custom:
        def decide(self, state):
            return Action.DROP if state.candidate.input_tokens >= 100 else Action.ADMIT

    profile = ServiceProfile(prefill_overhead_s=0.1, prefill_quadratic_s=0,
                             decode_rate=1, decode_context_s=0)
    kernel = TriageKernel(HardwareConfig(max_batch_size=1), profile)
    kernel.reset([Request("long", 0, 100, 100, 1), Request("short", 0.2, 16, 1, 5),
                  Request("outside", 10, 16, 1, 100)])
    assert counterfactual(kernel, FCFSScheduler(), 10) == 0
    assert counterfactual(kernel, Custom(), 10) == 1
    assert kernel.now == 0 and len(kernel.requests) == 3
    assert kernel.completed_count == 0


def test_fork_preserves_running_progress_and_kv_allocations():
    kernel = TriageKernel(HardwareConfig(max_batch_size=2))
    kernel.reset([Request("running", 0, 16, 100, 10), Request("next", 0.1, 16, 2, 10)])
    kernel.apply(Action.ADMIT)
    assert kernel.running and kernel.kv.blocks_used > 0
    branch = kernel.fork(arrivals_before_s=kernel.now + 10)
    assert branch.running[0] is branch.requests[0]
    assert branch.running[0].tokens_generated == kernel.running[0].tokens_generated
    assert branch.kv.blocks_used == kernel.kv.blocks_used
    generated = kernel.running[0].tokens_generated
    branch.apply(Action.DROP)
    assert kernel.running[0].tokens_generated == generated
    assert kernel.kv.blocks_used > 0


def test_features_do_not_change_with_future_arrivals_or_output_lengths():
    requests = [Request("a", 0, 16, 3, 10), Request("b", 2, 16, 5, 10)]
    tracker = BurstSwitchScheduler(FCFSScheduler())
    first = TriageKernel(arrival_observer=tracker.observe_arrival)
    state = first.reset(requests)
    features = selector_features(tracker, state)
    tracker2 = BurstSwitchScheduler(FCFSScheduler())
    second = TriageKernel(arrival_observer=tracker2.observe_arrival)
    changed = [replace(requests[0], output_tokens=300),
               replace(requests[1], arrival_time=200, slo_deadline=300, output_tokens=500)]
    assert selector_features(tracker2, second.reset(changed)) == features


def test_constant_tree_and_probability_hysteresis():
    tree = dict(schema_version=1, features=list(SELECTOR_FEATURES), root=dict(probability=1.0))
    class Drop:
        def decide(self, state):
            return Action.DROP

    policy = LearnedSwitchScheduler(Drop(), tree)
    kernel = TriageKernel()
    state = kernel.reset([Request("a", 0, 16, 1, 10)])
    assert policy.decide(state).action is Action.ADMIT
    assert policy.decide(replace(state, now=2)).action is Action.DROP
    tree["root"]["probability"] = 0.5
    assert policy.decide(replace(state, now=4)).action is Action.DROP
    tree["root"]["probability"] = 0.0
    assert policy.decide(replace(state, now=5)).action is Action.ADMIT
    assert len(policy.switches) == 2
    policy.reset_episode()
    assert policy.mode == "fcfs" and policy.switches == []


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_training_is_valid_and_json_round_trip(label):
    pytest.importorskip("sklearn")
    rows = [dict.fromkeys(SELECTOR_FEATURES, 1.0) | dict(label_custom=label)] * 20
    tree = json.loads(json.dumps(fit_selector(rows)))
    assert custom_probability(tree, [1.0] * len(SELECTOR_FEATURES)) == label


def test_shallow_tree_learns_a_known_boundary():
    pytest.importorskip("sklearn")
    rows = [dict.fromkeys(SELECTOR_FEATURES, 0.0) | dict(recent_rps=i, label_custom=int(i >= 20))
            for i in range(40)]
    tree = fit_selector(rows, max_depth=2, min_samples_leaf=5)
    assert custom_probability(tree, [2] + [0] * 7) == 0
    assert custom_probability(tree, [30] + [0] * 7) == 1


def test_export_matches_sklearn_at_float32_split_boundary():
    pytest.importorskip("sklearn")
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier

    values = np.array([[0.1] + [0] * 7, [0.2] + [0] * 7])
    rows = [dict(zip(SELECTOR_FEATURES, v)) | dict(label_custom=i) for i, v in enumerate(values)]
    tree = fit_selector(rows, max_depth=1, min_samples_leaf=1)
    reference = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1, random_state=0).fit(values, [0, 1])
    threshold = tree["root"]["threshold"]
    for value in (0.1, 0.2, threshold, np.nextafter(threshold, 0), np.nextafter(threshold, 1)):
        features = [value] + [0] * 7
        assert custom_probability(tree, features) == reference.predict_proba([features])[0, 1]


def test_selector_training_and_validation_pipeline(tmp_path):
    pytest.importorskip("sklearn")
    import csv
    from experiments.train_policy_selector import run
    from llm_triage.rl import ACTIONS, FULL_FEATURE_NAMES
    from llm_triage.rl.dqn import DQN
    from llm_triage.workload.burstgpt import sha256

    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    trace = processed / "burstgpt_windows.csv"
    with trace.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "window", "source_row", "timestamp_s", "input_tokens", "true_output_tokens"])
        for index in range(24):
            split = "train" if index < 16 else "validation"
            writer.writerow([split, 0, index, index, 16, 2])
        writer.writerow(["test", 0, 24, 1000, 16, 1000])
    manifest = processed / "manifest.json"
    manifest.write_text(json.dumps(dict(selected_sha256=sha256(trace), source_sha256="fixture",
                                       estimated_output_tokens_train_mean=2)))
    config = dict(target_rps=[1.0], slo_s=5, decisions=dict(max_defers=5, defer_retry_s=0.25),
                  hardware=dict(max_batch_size=2))
    provenance = dict(trace_sha256=sha256(trace), manifest_sha256=sha256(manifest), source_sha256="fixture")
    training = tmp_path / "training"
    training.mkdir()
    model = DQN()
    for value in model.params.values():
        value.fill(0)
    model.params["b2"][0] = 1
    path = training / "dqn_seed7.npz"
    model.save(path, dict(config=config, **provenance, features=list(FULL_FEATURE_NAMES),
                         actions=[a.value for a in ACTIONS]))
    record = dict(config=config, **provenance, selection_split="validation", test_evaluated=False,
                  service_profile=vars(ServiceProfile()), checkpoints=[dict(seed=7, sha256=sha256(path))])
    (training / "run.json").write_text(json.dumps(record))
    before = path.read_bytes()
    output = tmp_path / "selector"
    rows = run(training, tmp_path / "data", output, train_windows=1, validation_windows=1,
               samples_per_case=2, min_samples_leaf=1)
    assert {r["policy"] for r in rows} == {"fcfs", "dqn", "burst_switch", "learned_switch"}
    assert {r["arrivals"] for r in rows} == {8}
    assert path.read_bytes() == before
    saved = json.loads((output / "run.json").read_text())
    assert saved["selector_training_split"] == "train" and not saved["test_evaluated"]
    tree = json.loads((output / "seed7/selector.json").read_text())
    assert custom_probability(tree, [0] * len(SELECTOR_FEATURES)) == 0
