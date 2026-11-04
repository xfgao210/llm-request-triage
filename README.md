# llm_request_triage

COMS E6998-007 project: SLO-aware LLM request triage.

## Week 1: shared data pipeline and FCFS (Hengyi)

The shared workload and FCFS modules now live under `src/llm_triage/`, following
[TECH_DESIGN.md](TECH_DESIGN.md). Start with the
[Week 1 handoff](docs/week1-hengyi.md) for contracts, limitations, and integration.
Python 3.11+; this Week 1 path has no runtime dependencies outside the standard library.

Run from the repository root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/prepare_burstgpt.py
python experiments/run_fcfs.py
```

Preparation downloads and verifies the pinned ~51 MB BurstGPT source, or reuses
the verified cache. `--source /path/to/BurstGPT_1.csv` accepts an existing copy
after the same hash check. The license is fetched from the pinned revision.

The FCFS command evaluates **validation** windows only and saves per-request logs
under `results/runs/week1-fcfs/` and compact summaries under `results/week1/`.
The retained [baseline report](results/week1/REPORT.md) covers 24 CPU-simulation
runs using explicitly **assumed** service rates. It is not a real-GPU result.
TTFT, KV-cache constraints, and the shared simulator/metrics engine remain with
their respective owners; this command uses the existing simulator as a reference adapter.

Raw/processed traces and verbose run logs are ignored by Git. Compact results,
the data manifest, and code/data hashes are retained for reproducibility.

## Evaluation engine

The shared evaluation engine under `src/llm_triage/metrics/` converts typed
request lifecycle events into validated per-request records and policy-run
summaries. Metric definitions and integration examples are documented in
[`docs/metrics.md`](docs/metrics.md).

The deterministic baseline set under `src/llm_triage/scheduling/` includes
FCFS, shortest-estimated-service, and static SLO-aware scheduling. The adapter
Vikas's simulator needs to run these policies is documented in
[`docs/simulator_integration.md`](docs/simulator_integration.md).

Run its tests without installing additional dependencies:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run an end-to-end comparison through the deterministic development executor:

```bash
python experiments/run_baselines.py --output-dir results/tables
```

This produces `baseline_requests.csv`, `baseline_summary.csv`, and
`baseline_config.json`. The synthetic executor validates the integration path;
it is not the high-fidelity simulator used for final project claims.

For a hosted workflow, open
[`notebooks/colab_baselines.ipynb`](notebooks/colab_baselines.ipynb) in Colab.

## Contributions

- **[Hengyi Yin — GPU calibration and request-triage prototype](contributions/hengyi/)**: reproducible simulator/RL core, actual RTX 5070 measurements, calibrated timings, and a small real-GPU queue replay. Start with the [contribution report](contributions/hengyi/REPORT.md).
