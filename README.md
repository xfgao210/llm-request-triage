# llm_request_triage

COMS E6998-007 project: SLO-aware LLM request triage.

## Contributions

- **[Vikas Chelur — Core simulator architecture](docs/simulator.md)**: iteration-level LLM serving simulator (`triage/`) with continuous batching, paged KV-cache block allocation, recompute preemption, and the scheduling callback hook (`scripts/smoke_sim.py`).
- **[Hengyi Yin — GPU calibration and request-triage prototype](contributions/hengyi/)**: reproducible simulator/RL core, actual RTX 5070 measurements, calibrated timings, and a small real-GPU queue replay. Start with the [contribution report](contributions/hengyi/REPORT.md).
