"""Device fallback, portable checkpoints, and reference update parity."""

import builtins

import pytest

np = pytest.importorskip("numpy")
from llm_triage.rl.devices import resolve_device
from llm_triage.rl.dqn import DQN
from llm_triage.rl.environment import ACTIONS
from llm_triage.rl.observation import FULL_FEATURE_NAMES


def test_device_fallback_without_torch(monkeypatch):
    original = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="requires PyTorch"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="device"):
        resolve_device("mps")


def test_auto_and_required_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cuda") == "cuda"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("discount_mode", ["decision", "elapsed_time"])
def test_tensor_update_matches_numpy_and_checkpoints_are_portable(device, discount_mode, tmp_path):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA runtime on this machine")
    from llm_triage.rl.devices import TensorLearner
    reference, accelerated = DQN(seed=7, discount_mode=discount_mode), DQN(seed=7, discount_mode=discount_mode)
    learner = TensorLearner(accelerated, device)
    # Exercise the public accelerated dispatch/target-sync path even on a CPU
    # test machine. CUDA tensors themselves are tested only when available.
    accelerated.device = "cuda"
    accelerated._accelerator = learner
    rng = np.random.default_rng(13)
    for step in range(205):  # Cross the target-network synchronization boundary.
        states = rng.normal(size=(16, len(FULL_FEATURE_NAMES))).astype(np.float32)
        masks = np.ones((16, len(ACTIONS)), dtype=bool)
        masks[::3, 0] = False
        done = np.arange(16) % 2 == 0
        masks[done] = False
        if step == 0:
            done[:] = True  # Empty nonterminal batch must not bootstrap masked terminals.
        batch = (states, rng.integers(len(ACTIONS), size=16), rng.normal(size=16),
                 states * 0.5, done, masks)
        if discount_mode == "elapsed_time":
            durations = rng.uniform(0, 4, size=16)
            durations[::3] = 0
            batch += (durations,)
        loss = reference.train_batch(batch)
        other = accelerated.train_batch(batch)
        assert other == pytest.approx(loss, rel=1e-9, abs=1e-10)
        for key in reference.params:
            np.testing.assert_allclose(accelerated.params[key], reference.params[key], rtol=1e-8, atol=1e-10)
    path = tmp_path / "portable.npz"
    accelerated.save(path, dict(features=list(FULL_FEATURE_NAMES), actions=[a.value for a in ACTIONS]))
    restored, _ = DQN.load(path, device="cpu")
    assert restored.device == "cpu"
    assert restored.discount_mode == discount_mode
    for key in reference.params:
        np.testing.assert_array_equal(accelerated.params[key], restored.params[key])
