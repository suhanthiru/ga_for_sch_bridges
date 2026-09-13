# Search findings

Status: pilot tranche 1 running (restarted 2026-09-13 under grammar_pilot
e80941e5402d6fe3 after the seam and trigger slots were wired and the drift controller's
solver flags made real, PLAN_CHANGES 2026-09-13; the earlier starts were stopped with
(near-)empty archives and are kept as `results/search/pilot_t1_aborted_*`). 1 000
rung-0 evaluations on the eight fixed cells. Interim reports go to
`findings/search_interim_<n>.md` from `python -m sb.cli report search --pilot --root
results/search/pilot_t1`; this file is filled in the order below when a tranche ends.
Every number comes from `sb.stats` or the archive census functions in
`sb/search/report.py`; the prose only reads them.

1. The map — heatmap of the validated elite's ablation delta over the two descriptor
   axes with the most ablation variance; cells whose elite has no bridge are grey.
2. Slot census — over validated elites, how often each slot holds a bridge, an RL
   component, neither.
3. Bridge contribution by slot — mean ablation delta by the slot holding the bridge.
4. RL contribution by placement.
5. Full vs no-bridge vs no-RL maps — where the full map's elite beats the no-bridge
   map's by >= 0.05 (validated), the bridge is load-bearing; those cells in plain terms.
6. Novel architectures — validated elites at novelty level >= 2 that beat every seed in
   their cell by >= 0.05, with the ablation that shows which component matters; "none"
   if none.
7. Reconciliation with prior findings — for each prior verdict (bridge as controller
   dead, IPF harmful, seam not load-bearing, D trigger inert, multi-marginal worse,
   bridge-data good, slip reference 5x, SE(2) wins on curves): agrees, disagrees, or the
   exception, with the cell.
8. Negative space — enabled components that appear in zero validated elites.
9. Algorithm census — per algorithm: cells filled, mean validated elite fitness, first
   finds, novel architectures, score against the random control.
10. POET edge map.
11. Variant census — bridge variants and solver flags in validated elites, with deltas.
12. Transfer matrix — every validated elite on every family.
13. One paragraph, no hedging: where the bridge belongs in a robotics stack for noisy
    environments on this family's evidence, and what the family cannot speak to.

## Interim

(none yet)
