# Burst-aware FCFS / DQN switching (Hengyi)

The experiment tests whether FCFS can preserve calm-traffic coverage while a
frozen DQN handles bursts. It does not assume the DQN is better during bursts.
Compare the hybrid with both fixed policies on identical request cohorts.

At each scheduling decision, compute:

```
recent_rate = arrivals in (now - 1s, now] / 1s
reference_rate = arrivals in (now - 11s, now - 1s] / 10s
burst_ratio = recent_rate / max(reference_rate, 1 request/s)
```

Start with FCFS. After 11 seconds of episode warmup, switch to DQN when
`burst_ratio >= 2` and at least three arrivals occurred in the recent second.
Return to FCFS when `burst_ratio <= 1.2`. Stay in each selected mode for at
least two seconds. Parameters live in `configs/burst_switch.json`; these are
initial development values, not validated optimal thresholds.

The kernel supplies each actual arrival timestamp, including arrivals during
GPU iterations, when the batch is full, and requests rejected for capacity.
The detector sees no output lengths, future workload, or completion outcomes.
Switching acts at the next scheduling decision and affects queued requests;
it does not cancel or preempt already running requests. Each mode preserves
its own candidate ordering. Deferred requests retain their existing cooldown.
The scheduler resets its detector and diagnostics for every episode.

The reference window adapts to sustained high load, so this detects relative
surges rather than absolute overload. The initial warmup can miss an early
burst. No scheduling callback runs while the kernel cannot offer a candidate,
so a switch can be delayed until capacity is available. These constraints
should be considered when choosing window lengths on validation traffic.

## Run the paired comparison

From the repository root, with the package's `dev,rl` dependencies installed:

```powershell
python -m experiments.evaluate_scenarios `
  --training-dir results/runs/current-reward-e2e/training `
  --data-dir data `
  --output results/runs/burst-switch-validation `
  --split validation `
  --burst-switch-config configs/burst_switch.json
```

Use the appropriate training and data directories for another prepared dataset.
The runner verifies data provenance and checkpoint checksums, evaluates fixed
FCFS, length-aware, SLO-aware, DQN per seed, and `switch_dqn_seed*`, then checks
that frozen checkpoint files remain unchanged. It does not train or adapt DQN.

- `runs.csv`: paired window metrics plus switch counts, FCFS/custom decision
  counts for hybrid policies, and a JSON switch trace with timestamps/rates.
- `summary.csv`: attainment, drops and goodput by policy/load/traffic group.
- `run.json`: training configuration, exact switch parameters, data hashes and
  checkpoint hashes. Use a fresh output directory for each configuration.

Compare SLO attainment and goodput first; interpret TTFT alongside drop rate
and first-token counts. Report seeds separately and compare paired windows.
High burst ratio alone is not evidence that DQN improves outcomes. Tune the
detector on validation only, then freeze its JSON and checkpoint selection
before explicitly running `--split test`. Do not use test results to retune.

## Next experiments

1. Inspect switch traces on calm / burst / recovery traffic and confirm that
   a useful number of decisions actually reaches each mode.
2. On validation, compare a small predefined grid of recent/reference windows
   and enter/exit ratios. Include the fixed FCFS and fixed DQN controls in each
   comparison; report mode usage and switch frequency.
3. Freeze one configuration and run paired held-out normal, burst and recovery
   scenarios. If the hybrid does not beat FCFS, report that outcome rather than
   selecting a favorable seed or interpreting lower conditional TTFT as a win.

## Initial local validation result

Ran the command above with the existing `current-reward-e2e` checkpoints
(seeds 7, 17, 27), eight validation windows and two arrival rates: 144 policy
runs across nine policies. No checkpoints were trained or updated, and no
held-out test windows were evaluated. This pilot uses the current local
`hengyi/scheduling-study` branch and its current-reward checkpoints; it is a
different experiment from the archived Week 3 main-branch holdout report.

| Target RPS | Policy | SLO attainment | Drop rate |
|---|---|---:|---:|
| 1.2 | FCFS | 84.8145% | 0% |
| 1.2 | DQN seed 7 / switch | 84.5215% / 84.5215% | 0% / 0% |
| 1.2 | DQN seed 17 / switch | 84.7656% / 84.8145% | 0% / 0% |
| 1.2 | DQN seed 27 / switch | 84.4238% / 84.4238% | 0.0488% / 0% |
| 2.4 | FCFS | 83.8379% | 0% |
| 2.4 | DQN seed 7 / switch | 83.2520% / 83.4473% | 1.7578% / 1.1719% |
| 2.4 | DQN seed 17 / switch | 83.5449% / 83.5449% | 0% / 0% |
| 2.4 | DQN seed 27 / switch | 82.8125% / 83.1055% | 3.9063% / 2.7344% |

Across both rates, the hybrid made 232, 206 and 247 switches for seeds 7, 17
and 27 respectively, with both modes receiving decisions in every seed's
aggregate. Switching improves or matches fixed DQN attainment in this pilot,
but none exceeds FCFS. These are descriptive validation results, not evidence
of a general improvement or a statistically established benefit. Before any
test evaluation, inspect whether fewer switches and a more selective burst
trigger improve paired validation outcomes.

The full local artifacts are under `results/runs/burst-switch-validation/`
(ignored by Git), including exact checkpoint hashes and switch traces.
