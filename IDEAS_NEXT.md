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
