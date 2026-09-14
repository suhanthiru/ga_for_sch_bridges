# Ideas for a second run

Anything that would extend the grammar or the environment family mid-search goes here
instead of into the code. Nothing in this file is acted on until the current search is
reported.

---

- **Bridge states, tracker labels.** PD-relabel (the bridge's rollouts with the PD
  tracker's commands as labels) was the best in-family training source on L1 in the
  gate, above both parents. A generator that owns the state distribution and a labeller
  that owns the corrections is a different object from either; worth a slot of its own
  in the grammar (`data`: {states: bridge | PD | MPPI} x {labels: bridge | PD | oracle}).

- **A finer per-episode outcome for tails.** `progress` in {0, 1/3, 2/3, 1} floors the
  CVaR_0.1 at zero for every source in the gate. Time-to-first-collision, or the
  Mahalanobis distance to the next marginal at the moment of failure, would give the
  tail criterion something to measure.

## 2026-09-13 — from wiring the pilot's slots

- The drift controller's `sampler` gene (SDE-EM / ODE-Heun / ODE-RK4) has nothing to act
  on: the environment integrates the SDE. It belongs to bridge *samples* used as data
  (the `data` slot the gate froze out of the pilot) or to a bridge-as-planner that
  integrates its own drift to a waypoint. A second run that re-opens the data slot
  should attach sampler and step count there.
- `estimator.ekf` (running slip-covariance estimate) is built but nothing reads it. The
  natural consumer is the observation of learned controllers on E2 (append the 3-d
  estimate to obs_partial) and the reference of a bridge re-solved online (the `adapt`
  slot). Both need E2 cells at rung 0, which the pilot does not have.
- The graph-captured PPO covers the plain-action placement only; a residual over a
  tensor-only base (grid bridge, neural drift) could be captured too if the base exposed
  a stateless tensor step — the PD tracker (demo lookup by step index) and the held
  command (Python state) are the ones that block it.
- A seam width quantised to powers of two for the neural cache is a coarse gene; the
  grid bridge uses the continuous width. If the pilot shows the width matters, the main
  search should cache neural nets on a finer ladder or fit the marginal scaling into the
  drift net's conditioning (width as an input feature) so one net serves every width.

## 2026-09-14 — a third of the pilot's bridge genomes carry a bridge that cannot act

Measured on the frozen pilot seeds (`grammar_pilot/seeds.jsonl`, hash c4891ed7): 143 of
200 seed genomes carry a bridge component, and 43 of those 143 (30 %) carry it *only* as
`trigger.bridge_disagreement` over a controller that is not a bridge. Such a trigger reads
the controller's forward/backward drift disagreement, which only a neural bridge drift
reports, so it never fires: instrumented on cell 2's elite, D was available on 0 of 2400
decisions. The first two validated cells of tranche 1 are exactly this shape and their
ablation deltas are 2e-08 and 2e-06.

The grammar is frozen and is not being changed mid-search (program section 9). The pilot's
machinery now reports these honestly — every row carries `trigger_fire_rate` and
`trigger_d_seen`, and `contribution_by_slot` counts inert bridges apart from active ones —
so the map will not claim a bridge is load-bearing where nothing fired. For the next
freeze, three options in increasing strength:

- a validity rule: a bridge trigger requires a controller that exposes a disagreement
  (the cheapest, and it removes 30 % of dead genomes from the seed population);
- give every controller a defined disagreement observable (a PD could report the gap
  between its command and the reference's tangent), so the trigger means something
  everywhere and the axis is actually searched;
- keep the inert combination deliberately as a negative control: it is the one genome
  shape whose ablation delta *must* be zero, and it caught the ablation bug today.

The third is worth keeping whichever of the first two is chosen: a component that provably
does nothing is a free test that the ablation machinery is honest. Related: the sampler
gene has the same character (registered, inert for a drift-executed controller).

## 2026-09-14 — the counterfactual tuner should prefer the neutral point on a tie

Measured at tranche 1's second checkpoint: six validated cells hold an inert bridge trigger
(fire rate 0.0, D available on no decision), and four of them return an ablation delta of
exactly 0 while two return +3.6e-4 and +1.6e-4. The difference is the tuner: where it
picked a threshold at which the distance trigger never fires, the counterfactual is the
elite and the delta is exactly zero; where it picked one that fires occasionally — better
at its tuning seed (30 episodes, seed 0), marginally worse across the ten validation seeds
at 200 episodes — a small residue appears.

The residue is negligible against the 0.10 minimum effect, but it is selection noise in
the counterfactual and it is avoidable: require an improvement of at least some epsilon
before the sweep moves off the neutral midpoint, so a tie keeps the neutral stack. That
makes an inert bridge score exactly zero by construction, and it reduces the variance of
every ablation delta at no cost. Worth doing at the next freeze, together with tuning at
more than one seed if the budget allows.
