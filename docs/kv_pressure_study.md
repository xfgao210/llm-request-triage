# Isolating KV-cache pressure

Use the same BurstGPT 3 cohorts and arrival scaling as the [follow-up study](burstgpt3_confirmation.md).
The test windows have been used before. Treat this as a targeted comparison
on those workloads. Fix the batch limit at 32 and compare the standard KV
capacity with 4,096 KV tokens. Keep the five-second completion SLO and service
profile fixed. Replay at 2.4, 4.8, and 9.6 requests per second.

Train one DQN per seed (7, 17, 27) in each capacity configuration with the
same training recipe and validation checkpoint selection as the follow-up.
At evaluation, use shortest estimated service time to order candidates for
all policies. Compare admit-when-possible, fixed estimated-slack thresholds
from −2 to +2 seconds, and the three DQNs. Choose the best rule by validation
goodput at each load and capacity before evaluating the test windows.

Give every policy a fresh drop allowance per window. Permit a DROP only
when the number of explicit drops after that action would be at most one
per 20 arrivals observed so far. This applies the 5% limit to every
arrival prefix. Unused allowance carries through the window. Rules fall
back to ADMIT when the allowance runs out. The DQN ranks its remaining
feasible actions. When neither ADMIT nor DEFER is feasible and the
allowance is exhausted, that candidate waits for capacity. Train with the
original reward and action mask. The allowance applies at evaluation, so
the learned comparison tests how well those checkpoints work under a
shared deployment constraint.

Compare paired goodput, attainment, actual drops, unfinished requests,
preemptions, and long-output attainment. Count all arrivals in SLO
attainment. Report complete windows and contiguous blocks as paired
comparisons, with seeds sharing the same traffic. The primary comparison is
the 4,096-token run at 9.6 RPS against its validation-selected rule. Use the
standard-cache run at the same rate to see whether the DQN-rule gap grows
under KV pressure.
