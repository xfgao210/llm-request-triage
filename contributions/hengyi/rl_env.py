"""Finite episodic dispatch environment with explicit Defer time advancement.

Single non-preemptive slot. State has only current/past information; true output
length is used after an Admit action to generate the environment transition.
"""
from collections import deque
from dataclasses import dataclass
import math
import numpy as np
from triage_sim import Action, Observation, summarize_records
from service_profile import ServiceProfile

ACTIONS = (Action.ADMIT, Action.DEFER, Action.DROP)


@dataclass(frozen=True)
class RewardConfig:
    success: float = 1.0
    late: float = .1
    wasted_service: float = .2  # Penalize service seconds / request SLO.
    drop: float = .05
    defer_per_second: float = .02
    terminal_attainment: float = 1.0


class TriageEnv:
    def __init__(self, requests, horizon=120., prefill_rate=2000., decode_rate=400.,
                 estimated_output_tokens=144., defer_seconds=.25, max_defers=4,
                 reward_config=None, queue_order='fcfs', service_profile=None):
        self.profile = service_profile or ServiceProfile(prefill_rate=prefill_rate, decode_rate=decode_rate)
        # These aliases remain numerical for old callers; timing always uses profile methods.
        prefill_rate, decode_rate = self.profile.prefill_rate or 1e30, self.profile.decode_rate or 1e30
        if not all(math.isfinite(x) and x > 0 for x in
                   [horizon, prefill_rate, decode_rate, estimated_output_tokens, defer_seconds]):
            raise ValueError('Rates, horizon, estimate and defer duration must be finite/positive')
        if not isinstance(max_defers, int) or max_defers < 0:
            raise ValueError('max_defers must be a nonnegative integer')
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError('Duplicate request IDs')
        if any(any(not math.isfinite(v) for v in [r.arrival, r.deadline, r.input_tokens, r.output_tokens]) or
               not 0 <= r.arrival < horizon or r.deadline <= r.arrival or
               r.input_tokens <= 0 or r.output_tokens <= 0 for r in requests):
            raise ValueError('Invalid request')
        self.requests = sorted(requests, key=lambda r: (r.arrival, r.request_id))
        self.horizon, self.prefill_rate, self.decode_rate = horizon, prefill_rate, decode_rate
        self.estimate, self.defer_seconds, self.max_defers = estimated_output_tokens, defer_seconds, max_defers
        self.reward_config = reward_config or RewardConfig()
        if queue_order not in ['fcfs', 'shortest_estimated', 'earliest_deadline']:
            raise ValueError('Unknown queue order')
        self.queue_order = queue_order
        self.reset()

    def reset(self):
        self.now, self.cursor, self.defers = 0., 0, 0
        self.queue = deque()
        self.trace = [dict(time=0., queue_length=0, server_busy=0)]
        self.records = {r.request_id: dict(request_id=r.request_id, arrival=r.arrival,
                        input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                        deadline=r.deadline, start=None, completion=None, latency=None,
                        wait=None, decision_time=None, status='pending', slo_met=False)
                        for r in self.requests}
        self.actions_taken = dict(admit=0, defer=0, drop=0)
        self.done = False
        self._ready()
        return self.features(), self.mask()

    def _record_trace(self, busy=0):
        self.trace.append(dict(time=self.now, queue_length=len(self.queue), server_busy=busy))

    def _arrivals_until(self, end, busy=0):
        while self.cursor < len(self.requests) and self.requests[self.cursor].arrival <= end:
            self.now = self.requests[self.cursor].arrival
            while self.cursor < len(self.requests) and self.requests[self.cursor].arrival == self.now:
                self.queue.append(self.requests[self.cursor])
                self.cursor += 1
            self._record_trace(busy)
        self.now = end

    def _ready(self):
        if not self.queue and self.cursor < len(self.requests):
            self._arrivals_until(self.requests[self.cursor].arrival)
        self.done = not self.queue and self.cursor == len(self.requests)
        if self.queue and self.queue_order != 'fcfs':
            # Only currently arrived requests; no true output length in ordering.
            if self.queue_order == 'shortest_estimated':
                key = lambda r: (self.profile.service_s(r.input_tokens, self.estimate),
                                 r.arrival, r.request_id)
            else:
                key = lambda r: (r.deadline, r.arrival, r.request_id)
            self.queue = deque(sorted(self.queue, key=key))

    def observation(self):
        if self.done:
            return None
        r = self.queue[0]
        return Observation(self.now, r.request_id, r.input_tokens, self.estimate,
                           r.deadline-self.now, len(self.queue), False,
                           self.profile.service_s(r.input_tokens, self.estimate))

    def mask(self):
        if self.done:
            return np.zeros(3, dtype=bool)
        can_defer = self.defers < self.max_defers and self.queue[0].deadline-self.now > self.defer_seconds
        return np.array([True, can_defer, True], dtype=bool)

    def features(self):
        if self.done:
            return np.zeros(7, dtype=np.float64)
        obs = self.observation()
        slo = self.queue[0].deadline-self.queue[0].arrival
        return np.array([np.clip(obs.slack/slo, -5, 2),
                         np.clip(obs.estimated_service_s/slo, 0, 10),
                         math.log1p(obs.input_tokens)/10,
                         math.log1p(obs.queue_length)/5,
                         min(self.estimate/1000, 10),
                         min(self.now/self.horizon, 5),
                         self.defers/max(1, self.max_defers)], dtype=np.float64)

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode is done; call reset')
        if isinstance(action, (str, Action)):
            action = ACTIONS.index(Action(action))
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 3 or not self.mask()[action]:
            raise ValueError('Invalid or masked action')
        selected = ACTIONS[action]
        r = self.queue[0]
        rec = self.records[r.request_id]
        cfg = self.reward_config
        self.actions_taken[selected.value] += 1
        if selected == Action.DEFER:
            self.defers += 1
            self._arrivals_until(self.now+self.defer_seconds)
            self._record_trace()
            reward = -cfg.defer_per_second*self.defer_seconds
        else:
            self.queue.popleft()
            self.defers = 0
            rec['decision_time'] = self.now
            if selected == Action.DROP:
                rec['status'] = 'dropped'
                reward = -cfg.drop
                self._record_trace()
            else:
                rec.update(start=self.now, wait=self.now-r.arrival)
                service = self.profile.service_s(r.input_tokens, r.output_tokens)
                end = self.now+service
                self._record_trace(busy=1)
                self._arrivals_until(end, busy=1)
                success = end <= r.deadline
                rec.update(completion=end, latency=end-r.arrival, status='completed', slo_met=success)
                reward = cfg.success if success else -cfg.late-cfg.wasted_service*service/(r.deadline-r.arrival)
                self._record_trace()
            self._ready()
        if self.done and self.requests:
            reward += cfg.terminal_attainment*sum(row['slo_met'] for row in self.records.values())/len(self.requests)
        return self.features(), float(reward), self.done, self.mask()

    def metrics(self):
        if not self.done:
            raise RuntimeError('Only final episode metrics are supported')
        return summarize_records(list(self.records.values()), self.trace, self.horizon, self.now)


def run_policy(env, policy):
    """Evaluate a callback rule on the same interactive transition implementation."""
    env.reset()
    total_reward = 0.
    while not env.done:
        _, reward, _, _ = env.step(policy(env.observation()))
        total_reward += reward
    return dict(**env.metrics(), episode_reward=total_reward, **{f'action_{k}': v for k, v in env.actions_taken.items()})
