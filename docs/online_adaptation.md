# Online adaptation comparison

Offline training is unchanged. Run `colab_burstgpt3_experiments.ipynb` and
download its team-results ZIP; a validation-only export is sufficient. Open
`notebooks/colab_burstgpt3_online_adaptation.ipynb`, provide that ZIP and the
original `BurstGPT_3.csv`, and select a checkpoint seed contained in the ZIP.
The notebook verifies hashes and reconstructs the same prepared traffic.

The second notebook runs two copies from identical selected weights:

- Frozen: greedy decisions with no updates.
- Adapting: fresh optimizer, target network initialized from the checkpoint,
  empty replay buffer, and updates after `TriageEnv.step` delivers feedback.
  Exploration defaults to zero; enabling it changes the comparison to include
  exploration costs. Learning rate defaults to 0.0001, not a tuned optimum.

Both have independent continuous simulations with identical native arrivals,
SLOs, hardware, and observations. Ranges are traversed chronologically once;
weights and replay persist from the earlier group into the later group.
Neither request queues nor learning state reset every five minutes. Outcomes
still in flight are not used prematurely. Updates are timestamped in
`adapting_updates.csv`, after the corresponding transition is observed.
Terminal reward is delivered only when the entire stream terminates.

The default notebook uses the first 1,800 seconds of each test group for a
manageable workflow check. Set `MAX_SECONDS_PER_GROUP=None` to use complete
selected ranges. Set `RUN_ONLINE_EXPERIMENT=True` to start. A full continuous
run may require a high-RAM runtime because the simulator retains lifecycle
events. Traffic outside selected ranges is omitted and treated as idle;
no-record intervals may actually be collection gaps.

The existing time-progress observation is normalized by the total continuous
replay horizon for both policies, rather than the offline 300-second episode.
Consequently, use this run's frozen copy as the comparison, not the frozen
notebook's episodic results. The learner's wall-clock computation is not
charged as simulated GPU latency. This is simulated online learning, not a
production deployment latency benchmark.

Reports include request-weighted SLO attainment, drop rate, cohort-local
goodput, arrival-cohort timelines, update counts, weight changes, final weights,
and provenance. Timeline metrics use eventual outcomes for reporting only;
they never feed earlier actions. Goodput counts on-time completions of a
group's own arrivals within that interval, excluding completions crossing its
end. Completion-SLO attainment includes eventual completions after the end.

This is an online evaluation protocol, not a frozen held-out score. Choose
adaptation settings on a different development stream and report multiple
starting seeds. Do not repeatedly tune against these test outcomes and then
describe them as an untouched benchmark. Completed runs may be reused without
learning again; incomplete runs require a fresh output directory.

With an existing prepared dataset and training folder, the same engine can run
outside Colab:

```bash
python -m experiments.online_adaptation \
  --training-dir results/runs/burstgpt3-initial/training \
  --data-dir data/burstgpt3 \
  --output results/runs/online-check \
  --seed 7 --max-seconds-per-group 1800
```

The CLI defaults to complete ranges when `--max-seconds-per-group` is omitted.
