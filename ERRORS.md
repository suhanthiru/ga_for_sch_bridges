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
