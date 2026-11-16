# Scheduling study plan

The three experiments ask where capacity runs out, whether learned admission adds
value with a common request order, and whether online updates help after a shift.
Run them in that order. Keep the five-second completion SLO and the simulator
service profile fixed across policies.

| Stage | Development inputs | Evaluation and output |
|---|---|---|
| Capacity | Eight validation windows at 0.6, 1.2, 2.4, 4.8, and 9.6 RPS. Use standard capacity and the existing artificial KV-stress capacity. | Report attainment, goodput, queue wait, batch/KV utilization, preemptions and isolated service misses. Choose the first KV-stress rate where FCFS loses at least five attainment points relative to 0.6 RPS and accumulates at least one mean queued request. Evaluate that rate and the nearest lower and higher grid values. If the criterion never fires, report that and use the preplanned middle three rates. |
| Admission | Train seeds 7, 17, and 27 from scratch with shortest estimated service as the common ordering. Use the three selected rates and validate checkpoints by attainment. Test fixed per-request slack cutoffs of -2, -1, 0, 1, and 2 seconds. | Compare admit-when-possible, the full rule threshold curve, and the three DQNs on the same evaluation cohorts, using all arrivals in the denominator. In validation choose the highest-goodput rule within each 0%, 5%, 10%, 20%, and 30% mean drop budget, including the admit-all rule. Show the actual drop rate on evaluation. Report long-output attainment separately. |
| Adaptation | Same three weights and ordering. Use greedy frozen and greedy adapting copies to isolate parameter updates. Freeze the conservative update settings before running. | Join consecutive disjoint evaluation windows into four continuous streams. Use the low rate before and after pressure, and the high rate during either one or two middle periods. Keep queue and learner state across those boundaries. Report wins and losses by phase, time to first update, weight drift, and measured learner time. |

The first pass uses the available pinned BurstGPT 1 data. Its test split was
already used by the Week 3 benchmark, so this is an exploratory replication on
reused test traffic. Workload joins are synthetic and retain the request lengths,
order and deadline budgets of each source window. Paired windows and streams are
the units of comparison. Different training seeds share the same evaluation traffic.

For a confirmatory run, obtain the BurstGPT 3 trace, prepare the existing
chronological ranges, and use multiple nonoverlapping mixed and peak-heavy blocks.
Use training and validation only to choose loads, thresholds, reward and update
budgets. Freeze them before running those blocks. Add an adaptive rule that adjusts
its admission threshold using recent queue observations. Run online learner cost
both as a measured diagnostic and as an explicit serving-budget sensitivity study.

Claims require the outcome to follow the comparison. A win over FCFS locates an
opportunity for scheduling. A win over the validation-tuned rule within the same
budget supports an incremental admission benefit. A win over the frozen model
supports an online update benefit for that workload and start state. A stable
rule win or an online regression is equally informative. The simulator uses
batch-one timing calibration and assumed batch contention.
