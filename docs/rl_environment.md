# RL Environment Wrapper (`src/llm_triage/rl/`)

**Owner:** Vikas (vc2466) | **Milestone:** Week 2 — RL Environment API

A Gymnasium-compatible reinforcement learning environment (`TriageEnv`) wrapping the shared `TriageKernel` simulator, with strict ground-truth output-length masking, feasibility action masking, configurable invalid-action handling, and a pluggable reward interface.

## Quickstart

```python
from llm_triage.rl import (
    ActionMaskConfig,
    ObservationConfig,
    RLPolicyCallback,
    RewardConfig,
    TriageEnv,
)
from llm_triage.simulation import Action, RTX_5070_QWEN05B, Request, ServiceProfile

requests = [
    Request("req-0", arrival_time=0.0, input_tokens=256, output_tokens=128, slo_deadline=3.0),
    Request("req-1", arrival_time=0.1, input_tokens=512, output_tokens=64, slo_deadline=3.1),
]

env = TriageEnv(
    requests,
    hardware=RTX_5070_QWEN05B,
    profile=ServiceProfile(),
    horizon_s=120.0,
    candidate_selector="fcfs",       # "fcfs" | "length_aware" | "slo_aware" | "earliest_deadline" | callable
    observation_preset="full",       # "full" (15 features) | "compact_7" (7 features)
    invalid_action_mode="raise",     # "raise" | "fallback" | "penalize"
)

obs, info = env.reset()
while not env.done:
    mask = info["action_mask"]       # boolean (can_admit, can_defer, can_drop)
    action = Action.ADMIT if mask[0] else Action.DROP
    obs, reward, terminated, truncated, info = env.step(action)
```

## Module Layout

| Module | Contents |
|---|---|
| `src/llm_triage/rl/observation.py` | `ObservationConfig`, `ObservationBuilder`, `FULL_FEATURE_NAMES` (15 features), `COMPACT_7_FEATURE_NAMES` (7 features), and ground-truth/future-arrival leakage guards (`assert_no_ground_truth_leakage`, `sanitize_request_view`, `sanitize_snapshot`) |
| `src/llm_triage/rl/masking.py` | `ACTIONS` (`ADMIT=0`, `DEFER=1`, `DROP=2`), `ActionMaskConfig`, `compute_action_mask()`, and `resolve_action()` with configurable `invalid_action_mode` (`"raise"`, `"fallback"`, `"penalize"`) |
| `src/llm_triage/rl/rewards.py` | `RewardModel` protocol (`reward(transition) -> float`), `RewardConfig`, and `AsymmetricRewardModel` |
| `src/llm_triage/rl/environment.py` | `TriageEnv` (`gymnasium.Env` wrapper around `TriageKernel`), `RLPolicyCallback` (`SchedulerCallback` adapter for `simulate()` and `SimulatorExecutor`), and `run_policy()` |

## Integration Hooks for Week 2 Teammates

- **Agent Training (Hengyi):**
  - Standard Gymnasium / Stable-Baselines3 (`PPO`, `DQN`, `MaskablePPO`): use `env.reset()`, `env.step(action)`, and `env.action_masks()`. For unmasked algorithms (`PPO`, `DQN`), pass `invalid_action_mode="penalize"` or `"fallback"`.
  - Custom DQN (`contributions/hengyi/dqn.py`): `env.features()` and `env.mask()` are available directly; set `observation_preset="compact_7"` for 7-feature compatibility or `"full"` for the 15-feature hardware-aware state vector.
  - Evaluation via shared runner: wrap any trained model or `(obs, mask) -> action` function with `RLPolicyCallback(model)` and pass it to `simulate()` or `evaluate_policy_matrix()`.
- **Reward Shaping (Xiaofei):**
  - Pass a tuned `RewardConfig(...)` or a custom `RewardModel` implementing `reward(transition: Transition) -> float` into `TriageEnv(..., reward_config=..., reward_model=...)`.
