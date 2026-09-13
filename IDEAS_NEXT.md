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
