# Search plan

Pre-registration for the whole program. Sections are added as stages open. A section is
frozen at the commit that introduces it; later changes go through PLAN_CHANGES.md with a
dated entry written before the affected run starts.

Prior work: `D:\s_bridges\skill_chains` (four finished experiments and a generator suite).
Prior results are priors, not constraints. Nothing here re-runs a prior experiment.

---

## 0. Gate: is the bridge a better data generator, and why?

### 0.0 Provenance

- Prior repo pinned at `skill_chains@c1e0cbc` ("generator g4 results and findings").
  The core (se2, sde, bridge_fast, solver, terrain, task), the diffusion policy and the
  generator suite are vendored from that commit into `sb/`. `skill_chains` is not modified.
- At registration a chain of generator phases (g0-g6) is still running in `skill_chains`
  on this machine, started by a separate session. g0 and g1 finished for seeds 0-4 before
  this section was written; their shards are `results_generator/phaseg{0,1}_seed{0..4}.parquet`.
  The tables in `findings/gen_0.md` and `findings/gen_1.md` were NOT read before this
  section was committed. What was seen beforehand: the seed-99 quick run (300 training
  steps, in `results_generator_quick_log.txt`) and four g6b map-flip numbers in the tail
  of `chain_log.txt`. None of them enter the decision rule.
- Machine A: RTX 3080 Ti 12 GB (driver 610.74), Windows 11, Python 3.11.9,
  torch 2.11.0+cu128, numpy 1.26.4, scipy 1.17.1, pandas 3.0.3, pyarrow 25.0.1.
  Machine B: none available at registration. Replication is same-machine (0.7).
- Demos: regenerated on CPU with `make_demos(layout, n=500, seed=4242)`; sha256 of
  `data/demos_L1.npz` and `data/demos_L2.npz` recorded here before the first new cell
  runs (step 8 of the run order).

### 0.1 Hypotheses

- H1 (bridge-specific): the bridge's slip-reference covariance and its drift together
  produce training states and labels that a PD tracker under the same noise does not.
- H2 (noise-model-specific): the benefit comes from the state-dependent slip covariance
  alone; any controller rolled out under that noise matches the bridge.
- H0: neither; every difference is inside the minimum effect.

### 0.2 Design

Environment: SE(2) terrain task, layouts L1 and L2, evaluation disturbances
none / slip / rain / push. 8 cells, 6 of them disturbed.

Training-data sources for the diffusion policy (all but MPC-oracle are rollouts on the
reference kinematics under one shared noise stream, seed 90000+seed, so only the
controller and, for BRIDGE-*, the covariance differ):

| source | controller | noise covariance | labels |
|---|---|---|---|
| DEMO | 20 nominal-PD demos, no generated data | - | demonstrator |
| NOISED | demo states + Gaussian jitter (0.03, 0.03, 0.1) | - | demo actions kept |
| BRIDGE-slip | iteration-0 bridge drift, slip reference | terrain Sigma(x) | clipped drift |
| BRIDGE-unicycle | iteration-0 bridge drift, unicycle reference | isotropic 0.05^2 | clipped drift |
| PD-noise | PD tracker kp=10 + feed-forward on nearest demo | terrain Sigma(x) | PD command |
| PD-iso | same PD tracker | isotropic, variance matched to mean tr Sigma/3 on bridge states | PD command |
| DART | kp=6 demonstrator with action noise Sigma^1/2 z / dt | via actions | clean command |
| MPPI-rollouts | sampling MPC K=200 H=30 on nominal kinematics, no terrain | terrain Sigma(x) | MPPI command |
| GC-diff-rollouts | goal-conditioned diffusion policy trained on the 20 demos, DDIM-10, own sampling noise (generator seed 70000+seed) | terrain Sigma(x) | executed action |
| MPC-relabel | bridge-slip states, MPPI labels recomputed on those states | terrain Sigma(x) | MPPI command |
| MPC-oracle | MPPI with the true friction and rain fields, rolled out in the true env under `slip` | true dynamics | executed command |

`MPC-rollout` in the prior suite is the non-oracle sampling MPC and is the same source
as MPPI-rollouts here (the prior code plans on pure kinematics with no terrain lookup).

Fixed: N_DEMO = 20, GEN_MULT = 4 (80 generated trajectories), STEPS = 8000 diffusion
training steps, DDPM-50 train / DDIM-10 test, chunk 8 / execute 4, 5-layer MLP 256,
BRIDGE_CFG = {n_pair 6000, steps0 1200, K 0, batch 512, lr 1e-3, n_sim 50}, MPPI
K = 200, H = 30, noise 0.5, lambda 0.01, cost = greedy (0.3 * distance to the next
marginal mean per horizon step + terminal) unless 0.5 changes it. Seeds 0-4, 200
evaluation episodes per cell, evaluation task seed 50000+seed.

Per-episode outcomes: `success` (alive and inside the 2-sigma set of rho_3),
`progress` in {0, 1/3, 2/3, 1} (handoffs reached alive with Mahalanobis <= 2),
`collision`, `energy`.

Which cells come from the prior run and which are new:

- From `skill_chains` g0 (seeds 0-4, both layouts): DEMO, NOISED, BRIDGE-slip, PD-noise,
  DART, MPPI-rollouts. From g1 (seeds 0-4, L1 only): BRIDGE-unicycle, PD-iso, MPC-relabel.
  These are not re-run. Their per-cell success values are read from the shards and put
  through the statistics below. The prior run has no per-episode records, so 0.3's
  CVaR is computed only on cells run here.
- New here (seeds 0-4): GC-diff-rollouts and MPC-oracle on both layouts;
  BRIDGE-unicycle, PD-iso and MPC-relabel on L2 (so every isolation has 6 disturbed
  cells). New here (seeds 5-9): every source, both layouts, as the replication set.
- Known confound in the prior g1 (recorded in ERRORS.md): for seeds 0-4 on L1/L2 the
  BRIDGE-slip nets were loaded from Phase 2 (n_pair 8000, steps0 1500) rather than
  trained at BRIDGE_CFG (n_pair 6000, steps0 1200). BRIDGE-unicycle was trained at
  BRIDGE_CFG. The replication set trains every bridge arm at BRIDGE_CFG; if F2 changes
  sign between the two sets the prior F2 is reported as confounded.

### 0.3 Statistics

Everything reported comes from `sb/stats.py`; no number is typed by hand.

- Cell estimates: mean and t-interval over seeds (ddof 1).
- Contrasts: paired over the seed index. Paired t-test p-value; t-interval and a 10000-
  resample percentile bootstrap interval, both over seeds. "CI clear of zero" means both
  intervals exclude zero (conservative at n = 5; the bootstrap has 126 distinct
  resamples at that n and is anti-conservative alone).
- Families, each BH-corrected at q = 0.05 over its 6 disturbed cells:
  F1 BRIDGE-slip - PD-noise (primary); F2 BRIDGE-slip - BRIDGE-unicycle;
  F3 PD-noise - PD-iso; F4 MPC-relabel - BRIDGE-slip; F5 MPC-oracle - BRIDGE-slip;
  F6 GC-diff-rollouts - PD-noise; F7 MPPI-rollouts - PD-noise.
- CVaR_0.1 of `progress` per seed on cells with per-episode records, paired as above.
- Demo efficiency is not measured in this section (no n_demo sweep is run here; the
  prior g2b sweep is reported descriptively if it is cited).
- Coverage: per generated set, distinct (x, y, theta-bin) cells on a 32x32x8 grid at
  SE(2) distance > 0.05 from every demo state. Spearman rank correlation between
  coverage and disturbed success across the 10 in-family sources within each seed, then
  the t-interval over seeds. MPC-oracle is excluded (different dynamics).

### 0.4 Dynamic-range rule

The prior g0 ran at N_DEMO = 20 and 8000 steps. Before any new cell is run, the prior
g0 cell table is checked: if the largest success over all sources is below 0.15 in every
one of the 6 disturbed cells, the design has no dynamic range at N_DEMO = 20. In that
case N_DEMO is re-registered at 100 (GEN_MULT stays 4) with a dated PLAN_CHANGES entry,
and all 11 sources are run here at seeds 0-4 as new cells (the prior g0 at N_DEMO = 20
is then a prior, not the gate). If that too sits below 0.15 everywhere, N_DEMO = 500
and STEPS = 10000 (the Phase 5 regime). The check is reported in FINDINGS section 2.

### 0.5 MPC-relabel procedure

Before any new cell is run, `python -m sb.cli diag relabel` reports on 64 demo
trajectories, L1, seed 0: (1) the ratio of mean planar speed of MPPI labels on demo
states to that of the demo labels; (2) cross-seed label consistency
RMS(U_seed5 - U_seed6) / RMS(U_seed5) per axis; (3) collision rate under `none` of a
300-step policy trained on the relabelled demos. Thresholds: ratio in [0.7, 1.4],
consistency < 0.3 on every axis, collision < 0.5.

If any threshold fails, the greedy cost is replaced for all three MPC-family sources
by a tracking cost (distance at each horizon step h to the geodesic interpolant
`SE2.interp(mean_k, mean_k+1, tau + h / T_SKILL)`, terminal cost at the skill end) and
the diagnostic is re-run. If it still fails, MPC-relabel leaves the decision rule and
the label isolation becomes the 2x2 {bridge, PD} states x {bridge, PD} labels
(PD-relabel = bridge states with PD labels; BRIDGE-on-PD = PD states with bridge
labels). Either change gets a dated PLAN_CHANGES entry before the affected cells run.
The prior g1 MPC-relabel cells are reported either way, labelled with the cost used.

### 0.6 Decision rule

Gate threshold +0.05 (the program's minimum effect for a claim is +0.10; the two are
kept separate).

- bridge-specific: F1 delta >= +0.05 with CI clear of zero and q < 0.05 in at least
  3 of 6 disturbed cells, AND NOT (F4 delta >= +0.10 with CI clear of zero in at least
  3 of 6).
- noise-model-specific: F1 |delta| < 0.05 with the upper t-bound < 0.10 in at least
  4 of 6, AND F3 delta >= +0.05 with CI clear of zero in at least 3 of 6.
- neither: anything else.

The verdict is computed by `sb.stats.gate_decision` from the contrast tables and
copied, not retyped, into FINDINGS_generator.md.

Program-level minimum effect, stated independently of the verdict: a success delta
>= +0.10 (F1, or the best source against DEMO), or a CVaR_0.1 delta >= 0.15, or 2x
demo efficiency at equal success; BH q < 0.05. Below that the effect is null whatever
the p-value.

What each verdict means for the `data` slot of the search grammar:
bridge-specific -> default BRIDGE-slip, reference genes (kind, kappa, sigma,
slip_scale) searchable; noise-model-specific -> default PD-noise, the noise model
(covariance source, scale, anisotropy) searchable, no bridge nets trained in the
search; neither -> the data slot is frozen to the best of {DEMO, MPPI-rollouts,
GC-diff-rollouts} by F6/F7 and the bridge-as-generator line is closed.

### 0.7 Replication

Seeds 5-9, fresh evaluation seeds, same commit, every source and both layouts, run in
this repo. Same machine, labelled "same-machine replication" in the report (weaker than
a second machine; stated). The verdict stands only if the decision rule applied to
seeds 5-9 alone gives the same verdict. The pooled 10-seed table is reported but is not
the decision. GPU training is not bit-reproducible here; replication is statistical.

### 0.8 Frozen

After the commit that adds this section nothing in 0.2-0.7 changes except through 0.4
and 0.5, each with a PLAN_CHANGES entry dated before the affected run.
