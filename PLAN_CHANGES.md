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

## 2026-09-13 — pilot throughput gate: 24 rung-0 evaluations per hour, below the 40 registered

Evidence: results/search_probe (16 rung-0 evaluations of the frozen pilot grammar on the
eight fixed cells, quiet machine, log and archive kept): 16 rung-0 + 28 rung-1 + 14 rung-2
rows in 39 minutes, i.e. 24 rung-0 evaluations per hour and 89 archive rows per hour.
Rung-0 evaluations average 23 s (a PD stack 8-10 s, a PPO or residual stack 37-121 s,
diffusion 35 s of training); the rung-1 follow-up, which nearly every early mutant earns
because the cells are empty, costs another two evaluations each. The first-time CPU
bridge training per (cell, reference) is amortised and was already cached here.

Rule: the pilot's throughput gate reads "at least 40 rung-0 evaluations per hour or a
PLAN_CHANGES entry". This is the entry.

Decision: the 10 % pilot proceeds as registered in every other respect, in tranches
sized to the measured rate rather than to the plan's estimate: the first tranche is
1 000 rung-0 evaluations (about two days of wall-clock with follow-ups), checkpointed
every 100 and resumable. The engineering items that would restore the registered rate
are implementation, not grammar: a graph-captured rollout inside the PPO update, several
genomes' diffusion trainings batched with the stacked-parameter fitter, and bridge
training on a CPU pool in parallel with GPU work. They are listed here so a later
tranche can run faster without any change to what is being searched.

One more note for the record: the ablation deltas the probe produced for the two
bridge-controller elites are strongly negative (-0.24, -0.35); the PD in the same stack
beats the bridge drift, exactly as the prior findings said. They stand as the first two
rung-2 ablations of the program.

Affected evaluations: none invalidated; the probe rows carry the pilot manifest hash.

## 2026-09-13 — tranche 1 stopped before any row; seam and trigger wired; pilot root slots restricted

Evidence: the pilot manifest ef0314e95a24074f carried seam, trigger and estimator
components whose build returned a description and nothing consumed it (the probe's
elites show them as inert genes). Tranche 1 was stopped ten minutes in with an empty
archive; no evaluation is affected.

Changes, all before any pilot row:
- seam: `marginal_cloud` width and `waypoint` (width 0.25) scale the handoff marginals
  rho_1, rho_2 the bridge is trained between; the neural cache trains per width level
  (powers of two, so a continuous width is not a per-mutant solve) and the grid bridge
  uses the width as given. `fixed_clock` is width 1.
- trigger: `distance` restarts a robot's skill clock when its body-frame distance to the
  reference geodesic exceeds thr, re-anchoring the geodesic at the pose of restart;
  `bridge_disagreement` does the same on the forward/backward drift disagreement D,
  which only a neural bridge controller exposes; over any other controller it never
  fires and is counted as such. The cached search nets therefore carry the backward
  drift (SEARCH_BRIDGE_CFG; the gate's BRIDGE_CFG and its cache are untouched).
- pilot root slots are the six a compiled stack consumes: manifold, seam, controller,
  trigger, noise, safety. reference, planner, value, augment, time_split, estimator and
  adapt remain slot types for the main manifest and return as root slots once wired
  (stage C); data is a controller sub-slot already.
- the manifest carries a hash of the component sources, so a code change in a
  component's behaviour invalidates the freeze even when its declared spec is unchanged.

The pilot is re-frozen (new hash recorded in SEARCH_PLAN 2.10) and tranche 1 restarts
from an empty archive. This is a change to what is being searched, permitted because
no search row exists under the old manifest; the probe rows keep the old hash.

Affected evaluations: none.

## 2026-09-13 — the neural bridge's solver flags were inert at every rung; honoured at rung 2 now; tranche 1 restarted

Evidence: `controller.bridge_drift` carries ipf, eps, coupling, sampler and steps as
genes (SEARCH_PLAN 2.3: "the solver flags on every variant"); the pilot's pruning rule
made them inert on rungs 0-1 by design, but the evaluator never read them at rung 2
either, so the validation rung and its ablation delta would have scored a genome's
flags as noise. Found by reading the component while the first hours of tranche 1 ran
(13 log lines, no rung-2 row).

Change, before any rung-2 row:
- rung 2 trains the neural solver with the genome's `ipf` IPF iterations, its `eps` as
  the reference diffusion and its endpoint `coupling` (independent, demo_paired, ot,
  minibatch_ot are implemented in `sb/core/solver.py::sample_pairs`); every such config
  is its own cache entry (the cache key hashes the whole config).
- `steps` is the number of drift evaluations per skill at every rung (the command is
  held between them), which is what a step count means for a drift executed by the
  environment.
- `sampler` cannot act on a drift-executed controller (the environment integrates the
  SDE; the sampler would apply to bridge *samples* as data, a slot the gate froze). It
  stays a gene of the frozen grammar, recorded as inert; the variant census reports it
  with that label. ERRORS.md 2026-09-13 has the entry.
- `load_frozen` restricts the registry to the manifest's components and refuses a
  component-source drift only for a search run (reports warn), so later stage-C
  components do not disturb reading a frozen manifest.

The pilot manifest is re-frozen (hash in SEARCH_PLAN 2.10) and tranche 1 restarts from
an empty archive; the stopped start is kept as `results/search/pilot_t1_aborted_089779d1`
(rung 0-1 rows only, whose behaviour is unchanged except the `steps` hold).

Affected evaluations: none analysed; no rung-2 row existed.

## 2026-09-14 - the ablation delta gets a tuned counterfactual; the grammar hash is structural

Evidence: ERRORS.md of the same date. The registered definition (SEARCH_PLAN 2.7) says
the ablation delta is "fitness minus the fitness of the same genome with every bridge slot
replaced by its nearest non-bridge neighbour" and does not say where the neighbour's own
free parameters go. Tranche 1's first six validated cells showed that this gap decides the
sign and the size of the pilot's headline number.

Change, effective before any validated row is analysed:

- The substitution is deterministic and neutral: a parameter transfers only between
  identical parameter spaces, every other free parameter of the substitute takes the
  midpoint of its range.
- At rung 2 the substitute is *tuned*: a coordinate sweep from that neutral point over the
  substituted nodes' parameters (at most 8 rung-0 evaluations, at a seed the validation
  never uses, cached per (cell, structure) in `results/search/models/ablation_tuning.json`).
  The delta therefore compares a bridge against the best non-bridge counterpart the same
  budget can find. This can only shrink a bridge's apparent contribution, never inflate it,
  and the tuned counterfactual's DSL is written into every rung-2 row.
- Every evaluation records `trigger_fire_rate` and `trigger_d_seen`. A bridge component
  that never acts (a disagreement trigger over a controller with no forward/backward drift)
  is reported as inert in the census, and its cell is not evidence that "the bridge is
  load-bearing" whatever its delta.
- The grammar's identity hash is now structural (slots, components, parameter spaces,
  innovation table, filter). The component *source* hash stays in the manifest, is still
  refused by `load_frozen` when a search starts from drifted code, and is now written into
  every archive row and checkpoint, so a measurement fix can be made by re-freezing and
  resuming rather than by discarding a run - with the change recorded per row instead of
  hidden. `search.py --resume` logs the transition when it crosses one.

Affected evaluations: tranche 1's 40 rung-2 rows and their 30 ablation rows are tagged
`exclude:ablation_substitution` and kept in the archive; its 100 rung-0 and 84 rung-1 rows
are unaffected in behaviour but were produced under the previous manifest hash, so the
tranche restarts from an empty archive under the re-frozen manifest (the last restart
forced by a hashed-source change; from here a fix of this kind resumes).

## 2026-09-14 - "compute relative to PD" is computed from the genome, not the clock

Evidence: ERRORS.md of the same date, with the per-controller table of what the wall-clock
term was charging. The rule in 2.7 ("success - 0.02 log(compute relative to PD) - 0.05
collision") is unchanged; what changes is how compute is measured, because measuring it
with a stopwatch made the objective depend on cache state, machine load, evaluation order
and rung, and made fitness irreproducible against the bit-check registered in 2.8.

Change: compute is the genome's own training cost in seconds, summed over its components
(including sub-slots, so a residual pays for its base) from per-unit constants measured by
`bench/` on this machine and recorded in `sb/search/cost.py`:

    neural bridge drift   N_SKILL x 2 directions x 1200 solver steps x 8.3 ms
                          (+ IPF iterations x N_SKILL x 2 x 300 steps at rung 2)
    grid bridge           0.04 s x iterations / 50
    diffusion             steps x 0.69 ms
    PPO / residual / RL-noise   the rung's env-step budget x 38 us

Everything else costs nothing, which is right for components that only read demonstrations
or wrap another. Measured wall-clock stays in every row as `train_s`, reported, never
scored. The constants are a property of this machine and are re-measured whenever `bench/`
is re-run; they are part of the plan, so changing them is a PLAN_CHANGES entry.

The source hash now covers the evaluator, the cost model and the ablation tuner as well as
the components, so a row always records the code that gave it its meaning.

Affected evaluations: every fitness value. Tranche 1's 200 evaluations were produced under
the stopwatch objective and are kept as `results/search/pilot_t1_aborted_walltime_fitness`;
the tranche restarts. A run may not be stitched across a change of its own objective.
