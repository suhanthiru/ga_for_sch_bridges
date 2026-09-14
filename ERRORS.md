# Errors

Append-only. One entry per failure, in the order they happened. Never edit an old entry;
if it turns out to be wrong, add a new one that says so.

Each entry: date, class (1 env bug / 2 component bug / 3 search infra / 4 resource /
5 statistical anomaly / 6 plan conflict / 7 unexpected), component, what happened,
what was done, what it invalidated.

---

## 2026-09-13 — class 6 (plan conflict), prior generator run overlaps the gate

A separate session ran the generator suite (g0-g6) in skill_chains today while this
repo was being set up. g0 and g1 cover most of the gate's sources at seeds 0-4. Per the
program rule that nothing re-runs a prior experiment, those cells are read from the
prior shards and only the missing cells are run here (SEARCH_PLAN 0.2). Invalidates
nothing; changes the run order.

## 2026-09-13 — class 2 (component), bridge-net budget confound in the prior g1

`gen_phases.get_bridges` at c1e0cbc loads Phase 2's bridge nets for BRIDGE-slip on
L1/L2 at seeds 0-4 (n_pair 8000, steps0 1500, from a K=5 run) but trains
BRIDGE-brownian/unicycle fresh at BRIDGE_CFG (n_pair 6000, steps0 1200, K=0). The
prior F2 (slip minus unicycle, +0.34 to +0.35) therefore mixes reference covariance
with training budget. The vendored `sb.gen.bridges` has no such short-circuit and keys
its cache on the config hash; the replication set trains every arm at BRIDGE_CFG.
Prior F2 is reported with this caveat.

## 2026-09-13 — class 2 (component), MPC-relabel as run in the prior g1

The prior g1 relabelled every step of a bridge trajectory with the MPC command and
trained the chunked policy on the result; undisturbed success was 0.002. Its findings
attribute this to sequential inconsistency of relabelled 8-step chunks (the next state
came from the bridge, not from the relabelled action). The design pass here proposed a
second mechanism (the greedy per-step goal cost yields full-speed labels on
demonstrator-speed states). SEARCH_PLAN 0.5 registers a diagnostic that separates the
two before any relabel cell is run here.

## 2026-09-13 — class 2 (component), not vendored: g5 seed handling

`gen_phases.phase5` ignores its seed argument and hard-codes seeds (0, 1, 2), and its
`n_eval` argument is ignored by `train_eval`. Not part of the gate; g5 is not vendored.
Recorded so nobody copies it later.

## 2026-09-13 — every neural-bridge genome invalid in the first minutes of tranche 1 (component bug, class 2)

Seen: evaluation 0 of the relaunched pilot (a `controller.bridge_drift` stack) came back
invalid in 0 s. Reproduced on CPU: `get_bridges` passed `with_bwd=False` by keyword and
the new `SEARCH_BRIDGE_CFG` carried `with_bwd=True`, so `train_skill` raised on the
duplicate keyword before any training. The component test for `controller.bridge_drift`
checked the spec, not a build, because a build trains for minutes; the freeze therefore
passed.

Handled: the run was stopped after three evaluations (kept as
`results/search/pilot_t1_aborted_bwd_kw`, no shard written), the keyword collision fixed,
a test added that builds the search cache at a two-step budget and checks the backward
net and D exist (`tests/test_evaluate.py`), and the tranche relaunched from the fixed
commit. No archive row of the pilot was produced under the bug.

Lesson recorded: a component whose full build is too slow for its test still needs a
build at a toy budget in the test; "the spec looks right" is not a component test.

## 2026-09-13 — registered solver flags that nothing read (component bug, class 2)

Seen: `controller.bridge_drift` declared ipf, eps, coupling, sampler and steps and its
build used only `reference`. The docstring said the flags were "honoured at rung 2
only"; no code did. The component test checked the declared spec, which is exactly how
this stays invisible.

Handled: rung 2 now trains with ipf / eps / coupling (four couplings implemented and
tested), `steps` is a command hold at every rung, and `sampler` is documented as inert
for a drift-executed controller (it belongs to bridge samples as data, which the gate
froze out of the pilot). The pilot was re-frozen and tranche 1 restarted before any
rung-2 row (PLAN_CHANGES 2026-09-13).

Lesson: a parameter in a component's spec must be read by a test that changes it and
sees a different build; tests/test_evaluate.py now does that for the drift controller.

## 2026-09-14 - the ablation delta measured the substitute's luck (search infrastructure, class 3)

Seen: the first validated cells of pilot tranche 1 reported ablation deltas of +0.571,
+0.371, +0.313, +0.287, +0.000 and -0.339 - large in both directions, against a prior
that says the bridge is dead as a controller. Per the program's rule for a statistical
anomaly the substitution was checked before the number was believed, and it was the
substitution.

Three defects, all in how the nearest non-bridge neighbour was built:

1. A mapped parameter was copied whenever it happened to be numerically valid in the
   substitute's range. `trigger.bridge_disagreement` (thr in drift-disagreement units,
   0.05-1.0) mapped to `trigger.distance` (thr in metres, 0.02-0.5). Cell 2's elite had
   thr 0.0506, which is a legal distance too, so the counterfactual became a stack that
   restarts its skill clock every ~16 steps: tau never passed 0.25 and fitness fell from
   0.85 to 0.29. Cell 0's elite had thr 0.719, outside the distance range, so it was
   *re-sampled* instead - the same table producing two different behaviours.
2. Unmapped parameters were drawn at random (`p.sample(rng)`), though the docstring
   claimed the midpoint. Cell 3's grid bridge was ablated to a PD at kp 8.7, a gain
   nobody chose.
3. Because the bridge component in cells 0, 1 and 2 was a disagreement trigger sitting
   over a PD, it had no forward/backward drift to read and never fired: instrumented,
   D was available on 0 of 2400 decisions and the fire rate was 0.0000. The "bridge" arm
   was a plain PD, so the delta measured only the damage done to the other arm.

Handled: `ablate_bridges` is deterministic - a parameter transfers only between identical
parameter spaces (`ParamSpec.same_space`), everything else takes `ParamSpec.midpoint()`;
at the validation rung the counterfactual is then tuned by a short coordinate sweep at
rung-0 cost, cached per (cell, structure) (`sb/search/ablation.py`, SEARCH_PLAN 2.7 and
PLAN_CHANGES 2026-09-14); and every evaluation now records `trigger_fire_rate` and
`trigger_d_seen`, so a bridge component that never acted is visible in the archive rather
than inferred. Re-scored under the fix, cell 2 goes from +0.571 to +0.000 (the tuner
moves the distance threshold to 0.5, where it never fires, so the counterfactual is the
same PD) and cell 3 from -0.339 to -0.410 (the PD tunes to kp 30, and the grid bridge
loses to it by more). Both corrections run in the strict direction.

The 70 affected rows of tranche 1 are tagged `exclude:ablation_substitution` and kept.

Lesson: an ablation is an experiment, not a rewrite rule. Its counterfactual needs an
operating point chosen on purpose, and a component that cannot act in a stack must be
recorded as inert rather than credited with the difference.
