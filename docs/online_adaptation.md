# Online adaptation comparison

Open `notebooks/colab_burstgpt3_online_adaptation.ipynb` and provide only
`BurstGPT_3.csv` through Google Drive or upload. This is one self-contained
workflow: prepare data, train offline, select checkpoints on validation, then
compare frozen and adapting copies on the same traffic. No previous notebook
run, checkpoint upload, or results ZIP is required. The original frozen-only
notebook remains available as an alternative, not a prerequisite.

Offline training is unchanged: the notebook uses the original training cells
and pipeline, with the same default seeds, episode budget, and validation
selection. `ADAPTATION_SEED` selects one of the trained `SEEDS` for both copies.
Use a blank `RUN_NAME` to rerun the entire workflow in a fresh output folder.
The optional execution device changes where minibatch math runs, not the
Double-DQN objective, masks, reward, or validation protocol.

## Optional GPU minibatch learning

Both notebooks expose `DEVICE="auto"`. In Colab, optionally select a GPU
runtime before running setup. Auto uses CUDA if PyTorch reports a usable CUDA
device, otherwise it uses the original NumPy CPU learner. `DEVICE="cpu"`
forces CPU; `DEVICE="cuda"` fails clearly if CUDA is unavailable. Apple MPS
is not selected. Local users wanting CUDA can install `pip install -e ".[rl,gpu]"`;
PyTorch is optional for CPU-only use. Colab setup includes the GPU extra.

Only sampled neural-network minibatch updates use CUDA. The event simulator,
baselines, and single-request greedy inference remain on CPU. Keeping inference
on CPU avoids synchronizing a tiny prediction with CUDA at every scheduling
decision. Weights are synchronized back after each update, so checkpoints remain
portable `.npz` arrays and the frozen copy need not use CUDA.

**GPU availability does not imply a speedup.** The default 32-unit network is
small, and transfer/kernel overhead and float64 math may make CUDA slower.
Auto chooses by availability, not measured speed. Compare `DEVICE="auto"`
with `DEVICE="cpu"` on the same short development trace before spending a
long GPU session. Offline history and online learning summaries include
`learner_seconds`; run records include the actual learner device/backend.
CUDA uses PyTorch's masked Double-DQN, Huber loss, gradient clipping, and Adam
with reference float64 precision; device/library floating-point differences
can still change learning trajectories. No end-to-end speedup is claimed.

PyTorch documents CUDA availability and execution in its
[CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html).

## Online protocol

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

The default notebook uses complete selected test ranges
(`MAX_SECONDS_PER_GROUP=None`). For a manageable initial workflow check, set
`MAX_SECONDS_PER_GROUP=1800` to use a 30-minute prefix of each group.
`RUN_ONLINE_EXPERIMENT=True` runs the comparison after training and validation;
set it to false to stop after validation (skip later comparison/report cells).
A full continuous
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
