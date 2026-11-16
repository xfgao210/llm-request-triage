# BurstGPT 3 follow-up protocol

This run compares the three scheduling questions from the
[pilot](../results/scheduling-study/README.md) on a different trace.
It uses the pinned BurstGPT 3 SHA256 and the group's existing chronological ranges.

## Cohorts and selections

- Use complete, nonempty five-minute training and validation windows with at
  least 128 valid requests. Select 24 training and six validation windows at
  evenly spaced positions in chronological order.
- Partition each test range into consecutive, nonoverlapping groups of three
  complete windows from its original start. Retain groups with at least 128
  requests in every window, then select four groups per range at evenly spaced
  positions. Select them using timestamp and count only.
- Train with the selected training requests. Compute the shared output-length
  estimate from those requests. Evaluation includes every request in each chosen
  window. Record all selected IDs and timestamps.

## Experiments

1. **Capacity.** Repeat the pilot's 0.6–9.6 RPS grid on validation. Compare
   standard capacity, batch limit alone, KV limit alone, and both limits.
   Use FCFS for the four-way bottleneck diagnostic. Apply the pilot's declared
   knee rule to the combined limit. Compare FCFS and the two existing rules
   at the resulting three rates on each evaluation range.
2. **Admission.** Keep the pilot's five-second SLO, four slots, 4,096 KV tokens,
   length-aware order, shared output estimate, five slack thresholds, and
   0%, 5%, 10%, 20%, and 30% drop budgets. Retrain three DQN seeds for 24
   episodes each. Choose checkpoints by validation attainment. At each rate
   and budget, choose the highest-goodput validation rule within that budget
   before inspecting evaluation policy outcomes. Compare overall and
   long-output attainment, goodput, and achieved drop rates.
3. **Adaptation.** For each selected contiguous three-window block, apply the
   low selected rate before and after the middle window, and the high selected
   rate during the middle window. Keep state across phases and keep every
   source request. Each frozen/adapting pair starts from the same checkpoint
   and uses greedy actions. Use the pilot's 256-transition warmup and conservative
   schedule. Include the common-order slack rule using the validation-selected
   5% cutoff for low load and switching to the high-load cutoff at four
   visible waiting requests. If validation chooses the same cutoff twice,
   the switch leaves the rule unchanged.

The scaled phase streams are controlled load shifts derived from contiguous
source time blocks. They retain source ordering and request lengths. Evaluate
by arrival cohort so completions after phase boundaries still count. Treat
each three-window block as the unit of replication, with seeds sharing the
same trace. Report wins, ties and losses, and measure real learner compute
separately from simulation time. Reward, checkpoint and capacity selection
remain fixed during evaluation.
