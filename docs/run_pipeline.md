# Run the project end to end

Use Python 3.11 or 3.12. The pipeline runs on CPU and prepares the public
BurstGPT trace on first use. Set aside about 51 MB for the raw trace.

While [PR #9](https://github.com/xfgao210/llm-request-triage/pull/9) is open, use its branch:

```bash
git clone --branch hengyi/week3-repro https://github.com/xfgao210/llm-request-triage.git
cd llm-request-triage
python -m venv .venv
```

If you already cloned the repo, run `git fetch origin`, switch to
`hengyi/week3-repro`, and run `git pull --ff-only`. After the PR merges,
use the default branch.

Activate the environment with `source .venv/bin/activate` on Linux or macOS.
On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1`, or run each
Python command with `.\.venv\Scripts\python.exe`.

```bash
python -m pip install -e ".[dev,rl]"
python -m experiments.run_pipeline --config configs/pipeline_smoke.json --output results/runs/quick-check
python -m experiments.run_pipeline --output results/runs/full
```

The quick check trains one seed for two episodes. The full run trains seeds
7, 17, and 27 for 24 episodes each with the current reward model. Both commands
train on the train split, pick checkpoints on validation, and reload those
checkpoints to compare against FCFS, length-aware, and static SLO-aware.
The test split stays reserved for the frozen Week 3 benchmark.

Each output folder contains:

- `summary.csv` for mean attainment, drop rate, and goodput by policy.
- `validation.csv` for the paired results by window and arrival rate.
- `training/` for checkpoints, training history, selection scores, and run settings.
- `pipeline.json` for the resolved config, runtime, data and code hashes,
  and the path to the training log.

The next run reuses the prepared trace. Pass `--source /path/to/BurstGPT_1.csv`
to use a local copy of the pinned CSV. For a new experiment, use a fresh output
directory and a copy of `configs/week2_dqn.json`. The JSON controls seeds,
episodes, arrival rates, SLO, hardware, defer behavior, and reward weights.
The optional `agent` block controls `hidden`, `learning_rate`, and `gamma`.

If a run fails, check `pipeline.json`, the training log, and `error.log`.
Rerun the same command with `--resume` and the same environment. It checks
saved checkpoints before reusing them. An incomplete training attempt
starts over, while completed training can go straight to evaluation.

## Try the KV pressure setup

```bash
python -m experiments.run_pipeline --config configs/training_kv_stress.json --output results/runs/kv-stress
```

I trained and compared the policies on this config with four batch slots and
4,096 simulated KV tokens. At 2.4 RPS, the eight paired validation windows gave:

| Policy | SLO attainment | Drop rate | Goodput (req/s) |
|---|---:|---:|---:|
| FCFS | 62.30% | 0.00% | 1.484 |
| Static SLO-aware | 70.31% | 23.24% | 1.671 |
| Length-aware | 75.15% | 0.00% | 1.785 |
| DQN, three seeds | 72.61%–73.54% | 20.85%–21.48% | 1.719–1.744 |

DQN was ahead of FCFS and static SLO-aware on attainment and goodput in
this setup. Length-aware still led with zero drops. Under the standard capacity
config, FCFS averaged 84.33% across both validation rates and DQN reached
83.62%–84.16%. KV pressure is a useful place to test whether the agent can
learn a better trade-off when both request length and queue state matter.
These are validation results with simulated resource limits.

To reproduce the published holdout with its original weights and reward, follow
the separate [Week 3 instructions](../results/week3/README.md).

## BurstGPT 3: continuous periods and named test groups

The [Colab notebook](../notebooks/colab_burstgpt3_experiments.ipynb) runs this
workflow and produces comparison charts and a team-results ZIP. Open it in
Colab from the `burstgpt_3` branch, place `BurstGPT_3.csv` in Google Drive
(or choose upload), and run the setup, preparation, training, and validation
sections. The simulation runs on CPU; a GPU runtime is not required.
Set `RUN_FINAL_TESTS=True` only after settling model settings on validation.
The notebook evaluates both named test groups with frozen checkpoints and
can add controlled arrival-rate stress scenarios through `EXTRA_TEST_RATES`.
Download the exported ZIP and save a notebook copy with outputs for the team.

Use the local `BurstGPT_3.csv` to build fixed-duration scenarios. The original
BurstGPT 1 cache and frozen Week 3 reproduction keep their existing defaults.
BurstGPT 3 preparation verifies the CSV against the SHA256 recorded by the
analyzer; it does not download a dataset or infer a version from the filename.

First analyze the traffic (skip this command if this analysis already exists):

```bash
python scripts/analyze_burstgpt_traffic.py \
  --source BurstGPT_3.csv \
  --output-dir results/traffic_analysis \
  --bin-seconds 300
```

### 1. Inspect traffic before selecting ranges

Generate a report from the existing analysis; the source CSV is not read again:

```bash
python -m pip install -e ".[analysis]"
python scripts/report_burstgpt_traffic.py \
  --analysis-dir results/traffic_analysis \
  --period-seconds 3600 \
  --output results/traffic_analysis/period_report
```

Inspect `traffic_timeline.png`, `request_size_timeline.png`, `request_mix.png`, `periods.csv`,
`candidate_periods.csv` and `idle_gaps.csv`. Candidate rankings use arrival/token
metadata, never scheduler results. Change `--period-seconds` to compare larger
continuous periods (it must be a multiple of the analysis bin duration).
Use `--no-plots` to generate only tables without the plotting dependency.
The last short period is recorded but excluded from candidate rankings.

No-record intervals may be inactivity or missing collection. Peak RPS alone does
not determine simulated GPU demand; examine token rates and request sizes too.
Source timestamps are relative seconds, not calendar dates.
The plots show original CSV timestamps on the top axis and elapsed days on the
bottom axis. Copy precise `start_timestamp_s` and `end_timestamp_s` values from
`periods.csv` or `candidate_periods.csv` into the range config; plot ticks give
approximate locations, while the CSV tables give exact interval boundaries.

### 2. Define the dataset with explicit timestamps

Copy `configs/burstgpt3_ranges.example.json` to a new config and replace its
placeholder checksum and all illustrative timestamps using the traffic report.
That template deliberately fails validation until it has been filled in.

```json
{
  "schema_version": 1,
  "source_sha256": "<SHA256 from analysis summary.json>",
  "episode_seconds": 300,
  "min_split_gap_seconds": 300,
  "ranges": [
    {"split": "train", "group": "training", "start_timestamp_s": 0, "end_timestamp_s": 86400},
    {"split": "validation", "group": "validation", "start_timestamp_s": 86700, "end_timestamp_s": 108300},
    {"split": "test", "group": "ordinary", "start_timestamp_s": 108600, "end_timestamp_s": 130200},
    {"split": "test", "group": "peak_heavy", "start_timestamp_s": 130500, "end_timestamp_s": 152100}
  ]
}
```

The numbers above illustrate the format and are not selections for the real
CSV. Use the absolute values from the source `Timestamp` column. Intervals are
half-open: `start_timestamp_s <= Timestamp < end_timestamp_s`.
Ranges must be disjoint, inside the analyzed timeline, and ordered train before
validation before test. The optional gap applies between different splits.
Multiple ranges may share a group within one split.

This config replaces the analyzer's automatic 60/20/20 partitions. It does not
use traffic-regime labels. Include the complete natural traffic mix in training
and validation periods and designate later ordinary/peak-heavy groups for testing.
Select and record these ranges before viewing policy performance.

### 3. Extract consecutive episodes and train

After saving your filled-in config, for example as `tmp/burstgpt3_ranges.json`:

```bash
python scripts/prepare_burstgpt3.py \
  --source BurstGPT_3.csv --analysis-dir results/traffic_analysis \
  --ranges-config tmp/burstgpt3_ranges.json --data-dir data/burstgpt3

python -m experiments.run_pipeline \
  --config configs/burstgpt3_smoke.json --data-dir data/burstgpt3 \
  --output results/runs/burstgpt3-smoke
```

All valid ChatGPT API requests within each range are kept. Each range is divided
into consecutive `episode_seconds` intervals. A shorter final episode is kept
with its actual duration. Empty episodes are listed separately in the manifest;
they have no training actions. Group totals must contain at least one request.
The output-token estimate uses only selected training requests.

The manifest records the exact range config, hashes, interval bounds, groups,
request counts and empty intervals. Extraction streams the source into temporary
output and publishes it only after validation. Existing prepared outputs are
not overwritten. Use a fresh data directory for another experiment.
Optionally pass `--license data/raw/LICENSE-BurstGPT.txt` to copy the license.

The training config remains separate: `configs/burstgpt3_smoke.json` trains one
seed for two episodes; `configs/burstgpt3.json` uses three seeds and 24 episodes.
Episode count means sampled training episodes, not full passes over the data.
Increase episodes for a larger dataset and ensure replay-buffer warmup is reached.

`"target_rps": [null]` preserves native arrival times and full episode bounds.
Numeric rates scale the complete interval to `request_count / target_rps` seconds.
Episode queues reset at boundaries: consecutive source intervals do not carry
simulator queue state across episodes. Increase `episode_seconds` to include
longer bursts and recovery in the same episode.

### 4. Evaluate a frozen model

For an exploratory load comparison with frozen parameters, evaluate validation
scenarios at several rates without retraining:

```bash
python -m experiments.evaluate_scenarios \
  --training-dir results/runs/burstgpt3-smoke/training \
  --data-dir data/burstgpt3 \
  --split validation --target-rps 0.6 1.2 2.4 4.8 \
  --output results/runs/burstgpt3-validation-loads
```

When model selection and scenario design are complete, evaluate both test groups:

```bash
python -m experiments.evaluate_scenarios \
  --training-dir results/runs/burstgpt3-smoke/training \
  --data-dir data/burstgpt3 --split test \
  --output results/runs/burstgpt3-test
```

Pass `--groups ordinary` or `--groups peak_heavy` to run one group. Use the
`burstgpt3-full/training` directory instead if you trained the full config.
Evaluation does no learning or exploration and pairs policies on identical
requests. `runs.csv` identifies each group, window, interval and arrival rate;
`summary.csv` groups by group, rate, descriptive regime (if available) and policy.
`overall_slo_attainment` weights requests rather than averaging episode scores.
For native timing, `overall_goodput_rps` includes the duration of empty episodes.
For numeric rate overrides, empty episodes have no defined scaling and are
excluded from that duration; their native duration is recorded separately.
The older `mean_*` columns average executed nonempty episode scores.

Evaluation checks the dataset and checkpoint hashes. It accepts a separate load
but not a replacement prepared dataset: define all held-out groups before training.
Test results must not guide checkpoint selection. Traffic-based test selection is
a deliberately constructed benchmark; describe it as such in the report.

Elapsed time and session IDs remain descriptive analysis fields. They are not
used as GPU service times or exposed to the scheduling policy. The output-token
estimate is computed only from the selected training scenarios.

The earlier automatic catalog/regime selection remains available by omitting
`--ranges-config` and specifying `--train-regimes`, `--validation-regimes`,
`--test-regimes` and counts. Do not combine those selection flags with a range config.
