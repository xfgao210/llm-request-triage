"""Small NumPy Double-DQN: replay, target network, Huber loss and Adam.

No framework dependency. Seven visible state features, three masked actions.
"""
import json
from pathlib import Path
import numpy as np


class DQN:
    def __init__(self, seed=0, hidden=32, learning_rate=.001, gamma=.99):
        self.rng = np.random.default_rng(seed)
        self.learning_rate, self.gamma = learning_rate, gamma
        self.params = {'w1': self.rng.normal(0, np.sqrt(2/7), (7, hidden)),
                       'b1': np.zeros(hidden), 'w2': self.rng.normal(0, np.sqrt(1/hidden), (hidden, 3)),
                       'b2': np.zeros(3)}
        self.target = {k: v.copy() for k, v in self.params.items()}
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.updates = 0

    def forward(self, states, params=None):
        p = self.params if params is None else params
        hidden = np.maximum(0, np.asarray(states)@p['w1']+p['b1'])
        return hidden@p['w2']+p['b2']

    def act(self, state, mask, epsilon=0.):
        allowed = np.flatnonzero(mask)
        if not len(allowed):
            raise ValueError('No valid actions')
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(allowed))
        q = self.forward(state)
        return int(np.argmax(np.where(mask, q, -np.inf)))

    def train_batch(self, batch):
        states, actions, rewards, next_states, done, next_masks = batch
        # Double-DQN: online selection, target evaluation; terminal targets do not bootstrap.
        online_next = self.forward(next_states)
        next_actions = np.argmax(np.where(next_masks, online_next, -1e12), axis=1)
        next_values = self.forward(next_states, self.target)[np.arange(len(actions)), next_actions]
        targets = rewards+self.gamma*(1-done)*next_values
        hidden = np.maximum(0, states@self.params['w1']+self.params['b1'])
        q = hidden@self.params['w2']+self.params['b2']
        error = q[np.arange(len(actions)), actions]-targets
        absolute = np.abs(error)
        loss = np.mean(np.where(absolute <= 1, .5*error**2, absolute-.5))
        grad_q = np.zeros_like(q)
        grad_q[np.arange(len(actions)), actions] = np.clip(error, -1, 1)/len(actions)
        grad_hidden = (grad_q@self.params['w2'].T)*(hidden > 0)
        grads = dict(w2=hidden.T@grad_q, b2=grad_q.sum(axis=0),
                     w1=states.T@grad_hidden, b1=grad_hidden.sum(axis=0))
        norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        self.updates += 1
        for key, grad in grads.items():
            grad *= min(1, 10/max(norm, 1e-12))
            self.m[key] = .9*self.m[key]+.1*grad
            self.v[key] = .999*self.v[key]+.001*grad*grad
            mh = self.m[key]/(1-.9**self.updates)
            vh = self.v[key]/(1-.999**self.updates)
            self.params[key] -= self.learning_rate*mh/(np.sqrt(vh)+1e-8)
        if self.updates % 200 == 0:
            self.target = {k: v.copy() for k, v in self.params.items()}
        return float(loss)

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.params)
        path.with_suffix('.json').write_text(json.dumps(metadata or {}, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(hidden=data['b1'].size)
            model.params = {k: data[k].copy() for k in model.params}
            model.target = {k: v.copy() for k, v in model.params.items()}
        return model


class ReplayBuffer:
    def __init__(self, capacity=50000, seed=0):
        self.capacity, self.cursor, self.size = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, 7))
        self.next_states = np.zeros((capacity, 7))
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity)
        self.done = np.zeros(capacity)
        self.masks = np.zeros((capacity, 3), dtype=bool)

    def add(self, state, action, reward, next_state, done, mask):
        i = self.cursor
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.done[i], self.masks[i] = next_state, done, mask
        self.cursor = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, size=64):
        ids = self.rng.choice(self.size, size=size, replace=False)
        return (self.states[ids], self.actions[ids], self.rewards[ids],
                self.next_states[ids], self.done[ids], self.masks[ids])
