"""NumPy Double-DQN adapted from the original Hengyi prototype."""

import json

import numpy as np

from .environment import ACTIONS, FEATURES


class DQN:
    def __init__(self, seed=0, hidden=32, learning_rate=0.001, gamma=0.99):
        self.rng = np.random.default_rng(seed)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.params = {
            "w1": self.rng.normal(0, np.sqrt(2 / len(FEATURES)), (len(FEATURES), hidden)),
            "b1": np.zeros(hidden),
            "w2": self.rng.normal(0, np.sqrt(1 / hidden), (hidden, len(ACTIONS))),
            "b2": np.zeros(len(ACTIONS)),
        }
        self.target = self.copy_weights()
        self.m = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.updates = 0

    def copy_weights(self):
        return {key: value.copy() for key, value in self.params.items()}

    def forward(self, states, params=None):
        params = self.params if params is None else params
        hidden = np.maximum(0, states @ params["w1"] + params["b1"])
        return hidden @ params["w2"] + params["b2"]

    def act(self, state, mask, epsilon=0.0):
        allowed = np.flatnonzero(mask)
        if not len(allowed):
            raise ValueError("No valid actions")
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(allowed))
        return int(allowed[np.argmax(self.forward(state)[allowed])])

    def train_batch(self, batch):
        states, actions, rewards, next_states, done, next_masks = batch
        # Only nonterminal transitions bootstrap. Their masks constrain selection.
        live = ~done
        targets = rewards.copy()
        online = self.forward(next_states[live])
        next_actions = np.argmax(np.where(next_masks[live], online, -np.inf), axis=1)
        targets[live] += self.gamma * self.forward(next_states[live], self.target)[
            np.arange(len(next_actions)), next_actions
        ]
        hidden = np.maximum(0, states @ self.params["w1"] + self.params["b1"])
        q = hidden @ self.params["w2"] + self.params["b2"]
        index = np.arange(len(actions))
        error = q[index, actions] - targets
        absolute = np.abs(error)
        loss = np.mean(np.where(absolute <= 1, 0.5 * error**2, absolute - 0.5))
        grad_q = np.zeros_like(q)
        grad_q[index, actions] = np.clip(error, -1, 1) / len(actions)
        grad_hidden = (grad_q @ self.params["w2"].T) * (hidden > 0)
        grads = {
            "w2": hidden.T @ grad_q, "b2": grad_q.sum(axis=0),
            "w1": states.T @ grad_hidden, "b1": grad_hidden.sum(axis=0),
        }
        norm = np.sqrt(sum(np.sum(grad**2) for grad in grads.values()))
        if not np.isfinite(loss) or not np.isfinite(norm):
            raise FloatingPointError("Nonfinite DQN loss or gradient")
        self.updates += 1
        for key, grad in grads.items():
            grad *= min(1, 10 / max(norm, 1e-12))
            self.m[key] = 0.9 * self.m[key] + 0.1 * grad
            self.v[key] = 0.999 * self.v[key] + 0.001 * grad**2
            mean = self.m[key] / (1 - 0.9**self.updates)
            variance = self.v[key] / (1 - 0.999**self.updates)
            self.params[key] -= self.learning_rate * mean / (np.sqrt(variance) + 1e-8)
        if self.updates % 200 == 0:
            self.target = self.copy_weights()
        return float(loss)

    def save(self, path, metadata):
        np.savez(path, **self.params, metadata=json.dumps(metadata, sort_keys=True))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(data["metadata"].item())
            if metadata["features"] != list(FEATURES) or metadata["actions"] != [a.value for a in ACTIONS]:
                raise ValueError("Checkpoint observation or action schema does not match")
            model = cls(hidden=data["b1"].size)
            for key, expected in model.params.items():
                value = data[key]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid checkpoint weights for {key}")
                model.params[key] = value.copy()
        model.target = model.copy_weights()
        return model, metadata


class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.cursor = self.size = 0
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((capacity, len(FEATURES)), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity)
        self.done = np.zeros(capacity, dtype=bool)
        self.masks = np.zeros((capacity, len(ACTIONS)), dtype=bool)

    def add(self, state, action, reward, next_state, done, next_mask):
        index = self.cursor
        self.states[index], self.actions[index], self.rewards[index] = state, action, reward
        self.next_states[index], self.done[index], self.masks[index] = next_state, done, next_mask
        self.cursor = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, size):
        ids = self.rng.choice(self.size, size=size, replace=False)
        return (self.states[ids], self.actions[ids], self.rewards[ids],
                self.next_states[ids], self.done[ids], self.masks[ids])
