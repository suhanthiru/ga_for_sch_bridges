# Plan changes

Dated entries only. A change to SEARCH_PLAN.md is not valid without an entry here that
names the rule being changed, the evidence, the change, and which evaluations (if any)
the change would have affected. The entry is dated before the affected run starts.

---

## 2026-09-13 — section 0.6, what the "neither" verdict means for the data slot

Rule as committed (d4feb8b): under "neither" the data slot is frozen to the best of
{DEMO, MPPI-rollouts, GC-diff-rollouts} by F6/F7.

Evidence: none from data. The text was narrower than the program it implements, which
says that under "neither" the data slot drops bridges and the program continues on the
remaining slots. DART and PD-noise are non-bridge sources in the same design and were
left out of the candidate set by oversight.

Change: under "neither" the data slot keeps every non-bridge source in the gate as an
option (DEMO, NOISED, PD-noise, PD-iso, DART, MPPI-rollouts, GC-diff-rollouts) and
drops BRIDGE-* and MPC-relabel; the best of them by the cell table is the default.
Under "noise-model-specific" the same set stays available alongside PD-noise as the
default. The three-way decision rule itself is unchanged.

Affected evaluations: none; no cell has been run in this repo. Written before the
prior g0/g1 tables were used for anything other than the 0.4 dynamic-range check.

## 2026-09-13 — section 0.5 fired: MPC-relabel leaves the decision rule

Rule: SEARCH_PLAN 0.5, the MPC-relabel procedure.

Evidence: `python -m sb.cli diag relabel` on 64 L1 demo trajectories, seed 0
(results/gate/diag_relabel.json, diag_relabel_track.json). Greedy cost: speed ratio
2.83 (threshold 0.7-1.4), label consistency across planner seeds 0.19 / 1.35 / 1.38 on
vx / vy / omega (threshold < 0.3), collision 1.00 (threshold < 0.5). Tracking cost:
speed ratio 1.91, consistency 0.24 / 1.06 / 1.27, collision 1.00. Both fail every
threshold except vx consistency. The relabelled commands are faster than the
demonstrator's and their lateral and heading components are mostly planner sampling
noise (the demo's true vy and omega are near zero on L1, so the consistency ratio on
those axes measures noise against noise; that part of the threshold was poorly chosen
and is noted, but the speed and collision thresholds fail on their own).

Change, as the rule prescribes: MPC-relabel is dropped from the decision rule. The
action-label isolation is the 2x2 {bridge, PD} states x {bridge, PD} labels:
  F4a  PD-relabel - BRIDGE-slip     (bridge states; PD labels vs bridge labels)
  F4b  BRIDGE-on-PD - PD-noise      (PD states; bridge labels vs PD labels)
In 0.6 the guard "AND NOT (F4 delta >= +0.10 ...)" now reads F4a in place of F4. The
two new sources are added to the gate's source list for seeds 0-4 (both layouts) and
5-9. MPC-relabel stays in the source list and in the tables as a reported source, with
the greedy cost (the prior g1's), and is excluded from the decision.

Affected evaluations: none; no gate cell has run in this repo. The smoke run started
before this entry does not count as a gate cell.

## 2026-09-13 — section 0.4 pilot: two targets missed, pruning applied

Evidence: bench/results.json (quiet GPU, 15:02-15:19; the other session's chain log did
not move during the pass), rendered in findings/pilot.md; the earlier contended pass is
kept in bench/results_shared.json and agrees in every direction.

| target | registered | measured | ratio |
|---|---|---|---|
| environment steps per second | 50 000 | 51.7 M (CUDA-graph replay, 262 144 robots; 5.8 M at 4 096) | 1034x |
| batched bridge solves per second, grid | 200 | 9 867 (fp32, P = 4 096) | 49x |
| batched bridge solves per second, neural DSBM | 200 | 2.3 (bf16, P = 256, 1 200 steps) | 0.01x |
| population PPO genome-updates per second | 128 | 48.9 (bf16, P = 128, minibatch 1 024) | 0.38x |

Also recorded: the covariance-steering gate passes at fp32 (6.5e-7) and fails at bf16
(3.5e-2), so no solver runs in bf16; two-process bit-equality of a graph-replayed
rollout holds on the GPU; diffusion training takes 35 s per 8 000 steps.

Rule applied (0.4): a quantity below 50 % of its target prunes the component that
depends on it, never extends the grammar.

1. Neural per-mutant bridge solving is pruned from rungs 0 and 1. On those rungs the
   bridge-tagged controller, planner and data components use the per-cell cached nets
   trained once at BRIDGE_CFG (amortised over every mutant in the cell), and the grid
   Sinkhorn bridge is the bridge whose parameters a mutant can vary. The neural solver
   flags (epsilon, IPF iterations, coupling, sampler, steps) take effect at rung 2 only.
2. The RL step budget on rung 1 drops from 500k to 250k; rung 0 stays at 100k and rung 2
   at 2M. RL placements are not removed.

The environment-step target was mis-scaled by three orders of magnitude; it stays as
registered and the internal watchdog is 20 M steps/s at 65k robots. The PPO shortfall
is an implementation matter (the rollout runs the eager step; a graph-captured rollout
would roughly halve the update time) and may be revisited without touching the grammar.

Affected evaluations: none; no search cell has run.
