"""Non-preemptive FCFS: earliest arrival, stable enqueue ties, never drop.

The kernel supplies admission_available after checking its resources. FCFS does
not infer feasibility using hidden output lengths or invent KV-cache capacity.
An empty queue is not a decision point; calling select/decide raises ValueError.
"""
from llm_triage.types import Action, Decision


class FCFSSelector:
    def select(self, snapshot):
        if not snapshot.candidates:
            raise ValueError("FCFS requires a nonempty candidate queue")
        return min(snapshot.candidates, key=lambda candidate: candidate.arrival_time_s).request_id


class AdmitWhenFeasible:
    def act(self, candidate, snapshot):
        return Action.ADMIT if snapshot.resources.admission_available else Action.DEFER


class FCFSScheduler:
    def __init__(self):
        self.selector = FCFSSelector()
        self.triage = AdmitWhenFeasible()

    def decide(self, snapshot):
        request_id = self.selector.select(snapshot)
        candidate = next(c for c in snapshot.candidates if c.request_id == request_id)
        return Decision(request_id, self.triage.act(candidate, snapshot))
