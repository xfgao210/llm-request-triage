# llm_request_triage

COMS E6998-007 project: SLO-aware LLM request triage.

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

- **[Hengyi Yin — GPU calibration and request-triage prototype](contributions/hengyi/)**: reproducible simulator/RL core, actual RTX 5070 measurements, calibrated timings, and a small real-GPU queue replay. Start with the [English contribution report](contributions/hengyi/REPORT.md).
