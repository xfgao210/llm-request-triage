# A small FCFS / DQN policy selector

This is the first learned alternative to the burst threshold rule. A shallow
decision tree chooses which scheduler handles a queued request; the existing
DQN stays frozen. The goal is to discover when DQN helps, rather than assume
that every traffic burst needs DQN.

## Inputs and switching

The tree uses eight current/past signals: recent arrival RPS, preceding
reference RPS, their burst ratio, eligible queue size, oldest eligible waiting
time, candidate slack, batch utilization and KV utilization. Arrival history
uses the same kernel notifications as the burst switch, so it includes arrivals
while capacity is full. Eligible queue features exclude requests still in a
defer cooldown; they are not total-backlog measurements.

The depth limit is three and each training leaf needs at least ten samples.
Start in FCFS, enter DQN when the leaf's observed DQN-win fraction is at least
0.7, and return to FCFS when it falls to 0.4 or below. Wait at least two seconds
between switches. These are initial development settings, not calibrated
confidence levels. The learned switch does not inherit the threshold rule's
11-second warmup: its training features use the same partial arrival history
available at episode start. Rates use the fixed configured window denominators.

Only queued decisions change; active requests continue. Per-episode history,
mode and counters reset on each run. Exported trees are JSON, with the feature
schema and bound DQN checkpoint hash. Inference uses NumPy and does not require
sklearn or loading a pickle.

## Offline labels from the same simulator state

1. Choose evenly spaced **training** windows before looking at outcomes.
2. Visit them under fixed FCFS and fixed DQN. Sample decision states at spaced
   simulated times, up to six states per behavior/case. Large idle gaps can
   produce fewer samples; repeated decisions at one time are not oversampled.
3. Fork each state twice. Mutable requests, KV allocation and queue references
   remain internally consistent within each copy and independent of the source.
4. Retain existing unfinished work and the same future arrivals during the next
   ten seconds. Exclude arrivals at/after that cutoff in both branches.
5. Run one branch with FCFS and one with DQN until the retained cohort drains.
   Completions after the cutoff still count if they meet their own deadline.
6. Label DQN only if it produces strictly more additional on-time completions;
   ties use FCFS. Fit one tree per frozen DQN seed using these training labels.

The fork is a privileged offline API containing the future trace and true
lengths. It is never exposed to the deployed selector. Its features come only
from the original policy-visible snapshot and already observed arrivals.
Earlier terminal requests cancel from the comparison.

These labels estimate a fixed-policy continuation on a truncated future
arrival cohort. They are not optimal next-action labels and do not prove that
switching to DQN for only two seconds will help. States from FCFS/DQN behavior
can also differ from states visited by the learned hybrid. Overlapping sampled
rollouts are correlated; their counts are not independent statistical trials.
The binary target ignores win/loss magnitude. Gain regression or collecting
additional states under a frozen learned selector are possible later studies.

## Run

```powershell
python -m pip install -e ".[dev,rl,selector]"
python -m experiments.train_policy_selector `
  --training-dir results/runs/current-reward-e2e/training `
  --data-dir data `
  --output results/runs/policy-selector-v1
```

Defaults use eight training windows, all eight available validation windows,
the training record's load levels, and all its frozen seeds. Use `--seeds 7`,
`--train-windows`, `--validation-windows`, `--samples-per-case`, `--lookahead-s`,
`--max-depth`, `--min-samples-leaf`, `--enter-probability` and
`--exit-probability` for explicit development changes. Use a fresh output path.

Artifacts include per-seed training-label CSVs and `selector.json`, paired
`validation.csv`, `summary.csv`, `label_counts.csv`, and a `run.json` recording
the selected cohorts, source/data hashes, runtime and checkpoint hashes.
The runner rejects changed source during a run and checks both checkpoint
files and in-memory DQN weights remain unchanged.

Validation compares fixed FCFS, fixed DQN, the burst threshold rule, and the
learned selector. The test split is not simulated or labeled. DQN itself was
previously validation-selected; these validation results are development
evidence, not a fresh held-out result. Freeze all choices before a later test.

## Initial result

The local pilot used seeds 7, 17 and 27 from `current-reward-e2e`, eight selected
training windows and eight validation windows at 1.2/2.4 RPS. It generated 552
paired-state labels and completed 192 validation policy runs.

| DQN seed | Training labels | DQN wins | Ties | DQN losses |
|---|---:|---:|---:|---:|
| 7 | 184 | 4 | 154 | 26 |
| 17 | 184 | 5 | 165 | 14 |
| 27 | 184 | 20 | 131 | 33 |

With the initial 0.7 entry threshold, none of the fitted trees switched to DQN.
All three selectors therefore reproduced FCFS: 84.8145% SLO attainment at
1.2 RPS, 83.8379% at 2.4 RPS, and zero drops. This avoids the degradation of
fixed DQN and the threshold hybrid on these windows, but it is **not an
improvement over FCFS or evidence of successful learned routing**.

Compact results and fitted trees are retained in
[`results/policy-selector-v1/`](../results/policy-selector-v1/). Full labels and
per-window traces remain locally in `results/runs/policy-selector-v1/`.

The next useful experiment is to expand training-state coverage in constrained
or overloaded conditions and check whether DQN wins recur across windows.
Do not lower the threshold merely to force switches; compare any change on
validation, include fixed FCFS, and report mode usage along with attainment,
goodput, drops and conditional TTFT.

The [protocol-matched KV-pressure extension](pressure_policy_selector.md) uses
the existing 4,096-token scenario and standard-cache control at 2.4/4.8/9.6 RPS,
including common candidate ordering and the original causal 5% drop cap.
