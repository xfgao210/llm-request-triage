"""Small, JSON-exportable tree choosing between FCFS and a frozen callback."""

from dataclasses import replace
import math

from llm_triage.simulation import Decision
from .burst_switch import BurstSwitchScheduler


SELECTOR_FEATURES = (
    "recent_rps", "reference_rps", "burst_ratio", "eligible_queue_length",
    "oldest_eligible_wait_s", "candidate_slack_s", "batch_utilization", "kv_utilization",
)


def selector_features(tracker, state):
    """Only current policy-visible state and observed arrivals enter features."""
    tracker._prune(state.now_s)
    if tracker.last_arrival_s > state.now_s:
        raise ValueError("Detector received a future arrival")
    cfg = tracker.config
    recent = sum(t > state.now_s - cfg.recent_window_s for t in tracker.arrivals)
    recent_rate = recent / cfg.recent_window_s
    reference_rate = (len(tracker.arrivals) - recent) / cfg.reference_window_s
    values = (recent_rate, reference_rate,
              recent_rate / max(reference_rate, cfg.min_reference_rps),
              len(state.queue_requests), max(r.time_in_queue for r in state.queue_requests),
              state.candidate.slo_slack, state.batch_utilization, state.kv_utilization)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Selector features must be finite")
    return list(values)


def custom_probability(tree, features):
    """Inference needs no sklearn installation or pickle deserialization."""
    import numpy as np

    if tree.get("features") != list(SELECTOR_FEATURES) or tree.get("schema_version") != 1:
        raise ValueError("Selector feature schema does not match")
    # sklearn trees cast their inputs to float32 before comparing thresholds.
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (len(SELECTOR_FEATURES),) or not np.isfinite(features).all():
        raise ValueError("Invalid selector feature vector")
    node = tree["root"]
    while "probability" not in node:
        node = node["left"] if float(features[node["feature"]]) <= node["threshold"] else node["right"]
    return node["probability"]


def fit_selector(rows, *, max_depth=3, min_samples_leaf=10):
    """Fit on training counterfactuals only; do not oversample rare wins."""
    from sklearn.tree import DecisionTreeClassifier

    if not rows:
        raise ValueError("Need training counterfactual rows")
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
    model.fit([[row[name] for name in SELECTOR_FEATURES] for row in rows],
              [row["label_custom"] for row in rows])
    custom_class = list(model.classes_).index(1) if 1 in model.classes_ else None

    def export(index):
        if model.tree_.children_left[index] == -1:
            counts = model.tree_.value[index][0]
            probability = float(counts[custom_class] / counts.sum()) if custom_class is not None else 0.0
            return dict(probability=probability, samples=int(model.tree_.n_node_samples[index]))
        return dict(feature=int(model.tree_.feature[index]), threshold=float(model.tree_.threshold[index]),
                    left=export(model.tree_.children_left[index]), right=export(model.tree_.children_right[index]))

    return dict(schema_version=1, features=list(SELECTOR_FEATURES), root=export(0))


class LearnedSwitchScheduler(BurstSwitchScheduler):
    def __init__(self, custom, tree, *, config=None, enter_probability=0.7, exit_probability=0.4):
        if not 0 <= exit_probability < enter_probability <= 1:
            raise ValueError("Require 0 <= exit_probability < enter_probability <= 1")
        self.tree = tree
        custom_probability(tree, [0] * len(SELECTOR_FEATURES))
        self.enter_probability = enter_probability
        self.exit_probability = exit_probability
        super().__init__(custom, config)

    def decide(self, state):
        probability = custom_probability(self.tree, selector_features(self, state))
        if state.now_s - self.last_switch_s >= self.config.min_dwell_s:
            mode = self.mode
            if self.mode == "fcfs" and probability >= self.enter_probability:
                mode = "custom"
            elif self.mode == "custom" and probability <= self.exit_probability:
                mode = "fcfs"
            if mode != self.mode:
                self.switches.append(dict(time_s=state.now_s, previous=self.mode, mode=mode,
                                          custom_probability=probability))
                self.mode = mode
                self.last_switch_s = state.now_s
        self.decisions[self.mode] += 1
        policy = self.custom if self.mode == "custom" else self.fcfs
        priority = getattr(policy, "priority", None)
        candidate = min(state.queue_requests, key=priority) if priority else state.queue_requests[0]
        decision = policy.decide(replace(state, candidate=candidate))
        return decision if hasattr(decision, "request_id") else Decision(candidate.request_id, decision)
