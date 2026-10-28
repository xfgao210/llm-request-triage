"""Freeze validation choices, then evaluate DQN and rules on held-out workloads."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from repro_utils import paired_interval
from dqn import DQN
from rl_env import TriageEnv, run_policy
from repro_utils import write_csv
from triage_sim import fcfs, expired_only, static_slo, generate_workload, Action
from workloads import load_windows, replay_window
from service_profile import ServiceProfile


def safety_policy(factor):
    def policy(obs):
        return Action.DROP if obs.slack < factor*obs.estimated_service_s else Action.ADMIT
    return policy


def evaluate_model(model, env):
    total = 0.
    while not env.done:
        _, reward, _, _ = env.step(model.act(env.features(), env.mask()))
        total += reward
    return dict(**env.metrics(), episode_reward=total,
                **{f'action_{k}': v for k, v in env.actions_taken.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=Path('results/dqn'))
    parser.add_argument('--output', type=Path, default=Path('results/heldout'))
    parser.add_argument('--profile', type=Path, help='Must match the model training service profile')
    args = parser.parse_args()
    profile = ServiceProfile.load(args.profile)
    windows = load_windows()
    manifest = json.loads(Path('data/processed/manifest.json').read_text(encoding='utf-8'))
    estimate = manifest['estimated_output_tokens_train_mean']
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for factor in [.5, 1., 1.5, 2., 3.]:
        scores = []
        for key in sorted(k for k in windows if k[0] == 'validation'):
            for rate in [.6, 1.2]:
                requests, horizon, _ = replay_window(windows[key], rate)
                env = TriageEnv(requests, horizon=horizon, estimated_output_tokens=estimate,
                                service_profile=profile)
                scores.append(run_policy(env, safety_policy(factor))['slo_attainment'])
        candidates.append(dict(factor=factor, validation_slo_attainment=statistics.mean(scores)))
    factor = max(candidates, key=lambda row: row['validation_slo_attainment'])['factor']
    models = {path.stem: DQN.load(path) for path in sorted(args.models.glob('dqn_seed*.npz'))}
    if not models:
        raise ValueError('No trained checkpoints found; run train_dqn.py first')
    for name in models:
        metadata = json.loads((args.models/f'{name}.json').read_text(encoding='utf-8'))
        if ServiceProfile(**metadata.get('service_profile', ServiceProfile().as_dict())).as_dict() != profile.as_dict():
            raise ValueError('Evaluation profile differs from checkpoint training profile')
    rules = dict(fcfs=fcfs, expired_only=expired_only, static_slo=static_slo,
                 tuned_static=safety_policy(factor))
    # Evaluation sets are declared before inspecting results; OOD cases are diagnostics, not selection.
    cases = []
    for key in sorted(k for k in windows if k[0] == 'test'):
        for rate in [.6, 1.2, 2.4]:
            requests, horizon, scale = replay_window(windows[key], rate)
            cases.append((f'trace_rps{rate:g}', key[1], requests, horizon, scale))
    for seed in range(2000, 2020):
        for label, rate, multiplier, output_factor in [
            ('synthetic_normal', .8, 1., 1), ('synthetic_burst3', 1.2, 3., 1),
            ('ood_burst5', 1.2, 5., 1), ('ood_long_output', 1.2, 3., 2)]:
            requests = generate_workload(seed, base_rate=rate, burst_multiplier=multiplier)
            if output_factor != 1:
                requests = [replace(r, output_tokens=r.output_tokens*output_factor) for r in requests]
            cases.append((label, seed, requests, 120., 1.))
    results = []
    for scenario, case_id, requests, horizon, scale in cases:
        for name in [*rules, *models]:
            env = TriageEnv(requests, horizon=horizon, estimated_output_tokens=estimate,
                            service_profile=profile)
            metrics = run_policy(env, rules[name]) if name in rules else evaluate_model(models[name], env)
            results.append(dict(scenario=scenario, case_id=case_id, policy=name,
                                time_scale=scale, **metrics))
            if case_id in [0, 2000]:
                folder = args.output/scenario/name
                folder.mkdir(parents=True, exist_ok=True)
                write_csv(folder/'requests.csv', list(env.records.values()))
    write_csv(args.output/'runs.csv', results)
    summary = []
    paired = []
    metric_names = ['slo_attainment', 'window_goodput_rps', 'drop_rate', 'p95_latency_s',
                    'wasted_service_s', 'action_defer']
    for scenario in sorted({case[0] for case in cases}):
        group = [r for r in results if r['scenario'] == scenario]
        for name in [*rules, *models]:
            selected = [r for r in group if r['policy'] == name]
            record = dict(scenario=scenario, policy=name, cases=len(selected))
            for metric in metric_names:
                values = [r[metric] for r in selected if r[metric] is not None]
                record[metric+'_mean'] = statistics.mean(values) if values else None
                record[metric+'_sd'] = statistics.stdev(values) if len(values) > 1 else None
            summary.append(record)
        # Average model initializations within each paired workload first. Avoid pseudoreplication.
        for reference in ['fcfs', 'tuned_static']:
            deltas = []
            for case_id in sorted({r['case_id'] for r in group}):
                current = {r['policy']: r for r in group if r['case_id'] == case_id}
                dqn_mean = statistics.mean(current[name]['slo_attainment'] for name in models)
                deltas.append(100*(dqn_mean-current[reference]['slo_attainment']))
            low, high = paired_interval(deltas)
            paired.append(dict(scenario=scenario, reference=reference,
                               dqn_delta_pp=statistics.mean(deltas), ci_low=low, ci_high=high))
    write_csv(args.output/'aggregate.csv', summary)
    write_csv(args.output/'paired_deltas.csv', paired)
    selection = dict(static_candidates=candidates, selected_factor=factor, service_profile=profile.as_dict(),
                     estimated_output_tokens=estimate, models={name: hashlib.sha256(
                         (args.models/f'{name}.npz').read_bytes()).hexdigest() for name in models},
                     model_metadata={name: json.loads((args.models/f'{name}.json').read_text(encoding='utf-8')) for name in models},
                     source_sha256=manifest['sha256'],
                     selection='Only validation windows at rates 0.6 and 1.2; test never used for checkpoint/factor selection',
                     evaluation_workloads=len(cases), policy_runs=len(results))
    (args.output/'evaluation_config.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    lines = ['# DQN and rule baselines: measured-profile simulation', '',
             'These are simulator results, not real GPU queue measurements. True output lengths are hidden from policies. Checkpoints and rule factors use validation data only.', '',
             f'Service profile: {profile.provenance}. Safety factor selected on validation: {factor:g}.', '',
             '| Scenario | Policy | Overall SLO attainment | Window goodput /s | Drop rate |',
             '|---|---|---:|---:|---:|']
    for row in summary:
        lines.append(f'| {row["scenario"]} | {row["policy"]} | {row["slo_attainment_mean"]:.2%} | {row["window_goodput_rps_mean"]:.3f} | {row["drop_rate_mean"]:.2%} |')
    lines.extend(['', '## Paired DQN differences', '',
                  'Average model initializations within each workload, then bootstrap paired workload differences. Intervals do not include hardware-model error.', '',
                  '| Scenario | Reference | Difference /pp | 95% bootstrap interval |', '|---|---|---:|---|'])
    for row in paired:
        lines.append(f'| {row["scenario"]} | {row["reference"]} | {row["dqn_delta_pp"]:+.2f} | [{row["ci_low"]:+.2f}, {row["ci_high"]:+.2f}] |')
    lines.extend(['', '## Interpretation limits', '',
                  '- All arrivals are in the SLO denominator; dropped requests are failures.',
                  '- This is a single-slot, non-preemptive model with a 5-second end-to-end SLO.',
                  '- Requests outside the measured length bounds are extrapolations. Synthetic outputs can exceed those bounds.',
                  '- Latency is conditional on completion. Read it alongside goodput and rejection rate.',
                  '- DQN beating FCFS does not establish superiority over static or length-aware rules.',
                  '- Defer advances 0.25 seconds, at most four times for each head request; its live-backend implementation remains open.',
                  '- The real GPU queue replay currently compares only FCFS with static triage.'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(selected_factor=factor, policy_runs=len(results), paired_differences=paired), indent=2))


if __name__ == '__main__':
    main()
