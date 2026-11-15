"""Optional CUDA learner; CPU-only users do not need PyTorch.

Single-request inference stays in NumPy to avoid a GPU synchronization for
every simulator decision. Only sampled minibatch updates use CUDA. Checkpoints
remain device-independent NumPy arrays. No simulator work is moved to the GPU.
"""


def resolve_device(requested):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "auto":
            return "cpu"
        raise RuntimeError("CUDA learning requires PyTorch; install the [gpu] extra") from None
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("CUDA requested but unavailable; choose auto or cpu")
    return "cpu"


class TensorLearner:
    """Same masked Double-DQN/Huber/Adam update as the NumPy reference.

    Float64 preserves the reference precision. CUDA kernels and optimizer math
    need not be bit-identical to NumPy. The CPU device is supported internally
    for parity tests; public CPU learning retains the original NumPy path.
    """

    def __init__(self, model, device):
        import torch
        self.torch = torch
        self.device = device
        self.params = {k: torch.nn.Parameter(torch.as_tensor(v.copy(), device=device))
                       for k, v in model.params.items()}
        self.optimizer = torch.optim.Adam(list(self.params.values()), lr=model.learning_rate)
        self.target_ref = None
        self.params_ref = model.params

    @staticmethod
    def forward(states, params):
        return (states @ params["w1"] + params["b1"]).relu() @ params["w2"] + params["b2"]

    def train_batch(self, model, batch):
        torch = self.torch
        if self.params_ref is not model.params:
            raise RuntimeError("Weights replaced during accelerated learning; start a new learner")
        if self.target_ref is not model.target:
            self.target = {k: torch.as_tensor(v.copy(), device=self.device) for k, v in model.target.items()}
            self.target_ref = model.target
        states, actions, rewards, next_states, done, masks = batch
        states = torch.as_tensor(states, dtype=torch.float64, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float64, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float64, device=self.device)
        done = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        masks = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            live = ~done
            targets = rewards.clone()
            # Terminal transitions may have an all-false mask; never bootstrap them.
            if live.any():
                online = self.forward(next_states[live], self.params).masked_fill(~masks[live], -float("inf"))
                next_actions = online.argmax(dim=1, keepdim=True)
                targets[live] += model.gamma * self.forward(next_states[live], self.target).gather(1, next_actions).squeeze(1)
        self.optimizer.param_groups[0]["lr"] = model.learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        q = self.forward(states, self.params).gather(1, actions[:, None]).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(q, targets)
        loss.backward()
        norm = torch.stack([p.grad.square().sum() for p in self.params.values()]).sum().sqrt()
        if not bool(torch.isfinite(loss) & torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite DQN loss or gradient")
        scale = min(1.0, 10 / max(float(norm), 1e-12))
        for p in self.params.values():
            p.grad.mul_(scale)
        self.optimizer.step()
        for k, p in self.params.items():
            model.params[k] = p.detach().cpu().numpy().copy()
        return float(loss.detach())
