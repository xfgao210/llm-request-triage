"""Train on train windows/seeds, choose checkpoint using validation only."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import numpy as np
from dqn import DQN, ReplayBuffer
from rl_env import TriageEnv
from repro_utils import write_csv
from triage_sim import generate_workload, Action
from workloads import load_windows, replay_window
from service_profile import ServiceProfile


def evaluate(model, windows, estimate, profile=None):
    profile = profile or ServiceProfile()
    scores = []
    for key in sorted(windows):
        if key[0] != 'validation':
            continue
        for rate in [.6, 1.2]:
            requests, horizon, _ = replay_window(windows[key], rate)
            env = TriageEnv(requests, horizon=horizon, estimated_output_tokens=estimate,
                            service_profile=profile)
            while not env.done:
                env.step(model.act(env.features(), env.mask()))
            scores.append(env.metrics()['slo_attainment'])
    return statistics.mean(scores)


def train(args):
    profile = ServiceProfile.load(args.profile)
    windows = load_windows()
    manifest = json.loads(Path('data/processed/manifest.json').read_text(encoding='utf-8'))
    estimate = manifest['estimated_output_tokens_train_mean']
    train_keys = sorted(key for key in windows if key[0] == 'train')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = []
    for seed in args.training_seeds:
        started = time.perf_counter()
        model = DQN(seed=seed)
        buffer = ReplayBuffer(seed=seed+1000)
        rng = random.Random(seed)
        history = []
        best = -float('inf')
        steps = 0
        for episode in range(args.episodes):
            # Independent synthetic seeds and real training windows. No test/validation outputs fitted.
            if episode % 3 == 0:
                rate = rng.choice([.6, 1.2, 1.8])
                multiplier = rng.choice([1., 2., 3.])
                requests = generate_workload(seed=10000+seed*10000+episode,
                                             base_rate=rate, burst_multiplier=multiplier)
                horizon = 120.
            else:
                key = rng.choice(train_keys)
                requests, horizon, _ = replay_window(windows[key], rng.choice([.6, 1.2, 1.8]))
            env = TriageEnv(requests, horizon=horizon, estimated_output_tokens=estimate,
                            service_profile=profile)
            state, mask = env.reset()
            total_reward, losses = 0., []
            epsilon = max(.05, 1.-episode/max(1, args.episodes*.8))
            while not env.done:
                action = model.act(state, mask, epsilon)
                next_state, reward, done, next_mask = env.step(action)
                buffer.add(state, action, reward, next_state, done, next_mask)
                state, mask = next_state, next_mask
                total_reward += reward
                steps += 1
                if buffer.size >= 256 and steps % 4 == 0:
                    losses.append(model.train_batch(buffer.sample()))
            history.append(dict(episode=episode+1, epsilon=epsilon, steps=steps,
                                episode_reward=total_reward, loss=statistics.mean(losses) if losses else None,
                                slo_attainment=env.metrics()['slo_attainment'],
                                drop_rate=env.metrics()['drop_rate']))
            if (episode+1) % 20 == 0 or episode == args.episodes-1:
                score = evaluate(model, windows, estimate, profile)
                checkpoint_rows.append(dict(training_seed=seed, episode=episode+1,
                                            validation_slo_attainment=score))
                print(f'seed={seed} episode={episode+1} validation={score:.4f} steps={steps}', flush=True)
                if score > best:
                    best = score
                    model.save(args.output/f'dqn_seed{seed}.npz',
                               dict(training_seed=seed, selected_episode=episode+1,
                                    validation_slo_attainment=score, estimated_output_tokens=estimate,
                                    learning_rate=model.learning_rate, gamma=model.gamma,
                                    service_profile=profile.as_dict(),
                                    reward='success=1, late=-0.1-0.2*service/SLO, drop=-0.05, defer=-0.02*seconds, terminal=attainment',
                                    checkpoint_type='inference weights only; optimizer/replay not saved',
                                    source_sha256=manifest['sha256']))
        write_csv(args.output/f'training_seed{seed}.csv', history)
        print(f'seed={seed} finished in {time.perf_counter()-started:.1f}s best_validation={best:.4f}', flush=True)
    write_csv(args.output/'validation_checkpoints.csv', checkpoint_rows)
    (args.output/'training_config.json').write_text(json.dumps(dict(
        episodes=args.episodes, training_seeds=args.training_seeds, train_windows=len(train_keys),
        service_profile=profile.as_dict(),
        validation_windows=8, synthetic_seed_formula='10000+training_seed*10000+episode',
        rates=[.6, 1.2, 1.8], synthetic_burst_multipliers=[1, 2, 3],
        model='Double DQN, 7-32-3 ReLU MLP, Huber, Adam, replay 50000, batch 64, update every 4 steps, target copy every 200 updates',
        feature_schema=['slack/SLO', 'estimated_service/SLO', 'log1p(input)/10',
                        'log1p(queue)/5', 'estimated_output/1000', 'now/horizon', 'defers/max_defers'],
        numpy_version=np.__version__,
        processed_trace_sha256=hashlib.sha256(Path('data/processed/burstgpt_windows.csv').read_bytes()).hexdigest(),
        test_used_for_selection=False), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=120)
    parser.add_argument('--training-seeds', type=int, nargs='+', default=[7, 17, 27])
    parser.add_argument('--output', type=Path, default=Path('results/dqn'))
    parser.add_argument('--profile', type=Path, help='Measured or assumed service profile JSON; retrain for new hardware')
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('episodes must be positive')
    train(args)
