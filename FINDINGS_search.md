# Search findings

Status: **pilot tranche 1 complete**. 968 rung-0 evaluations, 2 660 archive rows, on the
eight fixed cells of SEARCH_PLAN 2.8, under one grammar (`c4891ed7e54cf4fd`) and one
component-source record (`70b6c4564dbd76b3`). No row is invalid and none is quarantined.
Earlier starts were stopped with near-empty archives and are kept as
`results/search/pilot_t1_aborted_*`; the reasons are in PLAN_CHANGES (2026-09-13,
2026-09-14) and ERRORS. Every number below comes from `sb.stats` or the census functions
in `sb/search/report.py`; the prose only reads them.

The sections follow the order the program registered.

## 1. The map

The map is flat. All eight cells are filled; mean validated elite fitness 0.769. The
ablation delta of every validated bridge-holding elite, over ten seeds each with a tuned
counterfactual, is:

    -0.402, 0.000, 0.000, 0.000, 0.000, 0.000, +0.0001, +0.0002, +0.0004

The largest positive contribution a bridge makes in any cell is **+0.0004**, against the
pre-registered minimum effect of **+0.10** (SEARCH_PLAN 0.3). There is no cell where the
bridge is load-bearing, so there is no heat to plot: a heatmap over any two descriptor
axes is uniformly zero apart from one negative cell. `map.parquet` and `map.png` carry it
for the record.

## 2. Slot census

Over the eight validated elites:

| slot | bridge | RL | other | empty |
|---|---|---|---|---|
| manifold | 0 | 0 | 1.000 | 0 |
| seam | 0 | 0 | 0.625 | 0.375 |
| controller | **0** | **0** | 1.000 | 0 |
| trigger | 0.375 | 0 | 0.125 | 0.500 |
| noise | 0 | 0 | 0.625 | 0.375 |
| safety | 0 | 0 | 0.250 | 0.750 |

Every elite's controller is `controller.pd`. The only slot in which a bridge component
survives to an elite is `trigger`, in three of eight cells, and section 3 shows what that
is worth. No RL component appears in any elite in any slot.

## 3. Bridge contribution by slot

Only one slot ever holds a bridge in an elite, so the table has one row that matters:

| slot holding the bridge | elites | mean ablation delta |
|---|---|---|
| trigger (`bridge_disagreement`) | 3 | +0.000 |
| controller (`grid_bridge`, validated but never an elite) | 1 | −0.402 |

The trigger result is not a small effect, it is a **non-effect with a mechanism**. The
disagreement trigger restarts a robot's skill clock when a bridge's forward and backward
drifts disagree. Over a PD controller there is no drift to disagree, so it cannot act:
instrumented across a full evaluation it had a disagreement signal available on **0 of
2 400** decisions and fired **0** times. Across the whole tranche, of 1 508 rows carrying
any trigger, **1 088 never fired once**; when a trigger did fire the median rate was
0.0008 per robot-step. The bridge content of this map is a passenger the search kept
because it costs nothing, not a component that does work.

## 4. RL contribution by placement

No RL component reaches a validated elite, so there is no contribution to attribute. At
rung 0, `controller.ppo` averages −0.061 over 35 evaluations and `controller.rl_residual`
+0.062 over 21, against 0.726 for the PD. At the pilot's rung-0 budget of 100 k steps
both placements are net-negative or marginal; SEARCH_PLAN 2.4 forbids concluding against
a placement for lack of steps alone, so this is reported as budget-limited, not as a
verdict on RL.

## 5. Full vs no-bridge vs no-RL maps

**Not yet available.** The control searches are implemented (`--control no_bridge|no_rl`,
SEARCH_PLAN 5.8) and the no-bridge control is the next run. Section 1 already bounds what
it can show: since no cell's elite draws any measurable benefit from a bridge, the
no-bridge map is expected to match the full map within noise, which is the condition the
plan's own stop rule calls "the bridge is decoration" (SEARCH_PLAN 9). The claim is not
made until the control has run.

## 6. Novel architectures

None. Every row in the tranche sits at novelty level 0 (850 rows, a structure already in
the seed set) or level 1 (1 660 rows, a new structure over known edges). **No row reached
level 2**, which is the program's threshold for claiming a novel architecture. The search
recombined the seed grammar; it did not invent a stack shape outside it.

## 7. Reconciliation with prior findings

| prior verdict | this search |
|---|---|
| bridge as controller is dead | **agrees.** PD 0.726 mean at rung 0; grid bridge 0.331, neural drift bridge 0.215. The one bridge controller ever validated loses to a tuned PD by 0.402. |
| IPF harmful | **no evidence either way.** The solver flags act at rung 2 only, and only one bridge controller was validated, so `ipf` was never contrasted. |
| seam not load-bearing | **agrees.** `seam.marginal_cloud` survives into two elites and `seam.waypoint` into three, and their ablation contribution is 0.000. |
| D trigger inert | **agrees, emphatically**, with the mechanism measured (section 3). |
| multi-marginal worse | not exercised in the pilot grammar. |
| bridge data good | **contradicted by the gate**, which returned "neither" on seeds 0-4 and on the replication. The surviving refinement is that bridge *states* with tracker *labels* are the best in-family source on L1 — the states carry the value, the drift labels destroy it. |
| slip reference beats unicycle | **agrees** (gate F2), inside the bridge; it stops the bridge being worse, it does not produce a lead. |
| SE(2) wins on curves | **weak support.** `manifold.se2` in six of eight elites, `manifold.flat` in two; no ablation isolates the manifold. |

## 8. Negative space

Enabled components that appear in **zero** validated elites, after 968 evaluations:

    controller.bridge_drift   controller.grid_bridge   controller.diffusion
    controller.ppo            controller.rl_residual   noise.rl
    data.demos                data.dart                data.pd_rollouts
    data.pd_relabel           data.mppi_rollouts       data.gc_diffusion
    data.oracle_rollouts      augment.noised           augment.pd_rollouts
    planner.mppi              planner.spline           reference.geodesic
    reference.nearest_demo    reference.spline         seam.fixed_clock
    value.distance            estimator.ekf            adapt.none
    time_split.equal

Two readings, both worth stating. The first is the program's: a component the search never
used has no niche in this family. The second is a limit of this pilot: the learned
controllers (diffusion, PPO, residual) and every `data` component are reachable only
through a learned stack, and the PD's fitness ceiling in these eight cells is high enough
(up to 0.991) that no learned stack was ever competitive at rung-0 budget. The negative
space here is evidence about *these cells*, not about the components in general.

## 9. Algorithm census

| algorithm | evaluations | mean fitness | cells first filled |
|---|---|---|---|
| CMA-MAE (`cma_mae`) | 1 819 | 0.743 | 0 |
| random control | 432 | 0.458 | 8 |
| seeds | 259 | 0.804 | 0 |

CMA-MAE beats the random control on mean fitness by 0.285, so on this family it earns its
complexity by the criterion of SEARCH_PLAN 5.10. Random filled all eight cells first
simply because the seed population is evaluated before any emitter runs. No algorithm
found a bridge-containing elite, because none exists to find.

## 10. POET edge map

Not run. POET is a main-search member (SEARCH_PLAN 5.2-5.7); the pilot is a single
process on eight fixed cells.

## 11. Variant census

No bridge variant appears in a validated elite as anything but an inert trigger, so the
variant census is empty by construction. Of the four endpoint couplings implemented for
the neural solver (independent, demo-paired, OT, minibatch-OT), all four were proposed and
evaluated at rung 0 inside `controller.bridge_drift` genomes; none reached rung 2. The
`sampler` gene is recorded as inert for a drift-executed controller (ERRORS 2026-09-13):
the environment integrates the SDE, so a sampler choice has nothing to act on, and it
belongs to bridge samples used as data, which the gate froze out of the pilot.

## 12. Transfer matrix

Not run. The tranche exercised E1 cells only (with E2 observability at the partial-view
corners); the eight-family transfer matrix is a main-search item.

## 13. The paragraph

On the evidence of this family, the Schrödinger bridge does not belong in this robotics
stack. A thousand evaluations of a quality-diversity search over a grammar in which half
the candidates carried a bridge component returned eight cells out of eight held by a
plain PD tracker, and the largest causal contribution any bridge made to any cell was
+0.0004 success against a registered floor of +0.10. The generator gate, settled earlier
and replicated, says the same thing about the bridge as a data source. The one place a
bridge still pays is narrow and specific: the *states* a bridge visits, relabelled by a
tracker, are the best in-family data source on L1 — the exploration is worth something and
the drift labels are not. What this family cannot speak to is everything outside it:
these are eight cells of one SE(2) terrain environment with a strong feedback baseline, a
regime where a well-tuned PD reaches 0.99 success, and a bridge is asked to beat feedback
at a task feedback already solves. A family where the target is genuinely multi-modal,
where the reference carries terrain the controller cannot see, and where feedback from the
demonstration manifold cannot reach the target — the three clauses of hypothesis H — is
the test this program has not yet run, and is where the next, much smaller experiment
should go.
