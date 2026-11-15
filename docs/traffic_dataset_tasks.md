# BurstGPT 3 dataset experiment tasks

Goal: train on a representative continuous traffic period, select checkpoints on
a subsequent period, and compare the same frozen policy on ordinary and
peak-heavy held-out periods. Preserve natural fluctuations; no regime filter is
required. Choose time ranges from traffic analysis before seeing policy results.

1. **Implemented — traffic inspection:** generate timeline plots, continuous-period statistics,
   idle-gap diagnostics and candidate stress periods from the existing analysis.
2. **Implemented — dataset definition:** specify source identity, episode duration and explicit
   half-open timestamp ranges in a JSON file. Label ordinary/peak test groups
   explicitly. Validate ordering, disjointness and optional separation gaps.
3. **Implemented — extraction:** stream all valid ChatGPT API requests in those ranges into
   consecutive episodes, preserving timestamps, source row IDs and interval
   bounds. Record empty episodes and all provenance. Fit output-length estimates
   using training requests only.
4. **Implemented — training/evaluation:** use the existing DQN pipeline, then report frozen results
   separately for each named test group. Verify the path with synthetic fixtures
   and a real-data preparation/training/validation smoke check.
5. **Next — experiment selection:** inspect the report, choose and record the actual ranges,
   train the final models and evaluate held-out test groups after model selection.
   This is a separate research decision; example ranges are not a final protocol.

Episodes reset simulator queues; consecutive episodes preserve traffic within
each episode, not queue state across their boundaries. Increase episode duration
when bursts span those boundaries. No-record intervals can represent inactivity
or missing collection. Traffic metadata alone cannot distinguish those cases.

Verification: 174 tests pass, including split ordering, overlap/gap rejection,
half-open episode boundaries, training-only length estimates and separate frozen
test groups. A local real-CSV smoke check extracted 48 consecutive episodes and
completed training plus 48 frozen validation policy runs. Its intervals in
`tmp/burstgpt3-ranges-smoke.json` are for implementation checking, not final study
selections. No real-data test scores were used to choose or tune the implementation.
