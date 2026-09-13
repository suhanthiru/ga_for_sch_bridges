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
- Demos: regenerated on CPU with `make_demos(layout, n=500, seed=4242)`, nominal
  success 0.998 on both layouts. sha256 (recorded before the first new cell ran):
  `data/demos_L1.npz` 2e60ef618b01265cefd7f6d6d99b25bd8f37f29c8c9b76f376ba8c115387fbb5,
  `data/demos_L2.npz` 3ae7a909044bf53bbf2cd944dbaa7e2361abc8f21bc166c53dd292c40926bf4f.

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

---

## 0.4 Costed pilot: throughput targets and the pruning rule

Registered before any benchmark in `bench/` has been run on a quiet GPU. Numbers
measured during the design pass (a separate session, GPU shared with other work) are
not evidence and do not appear in the pilot report; only `bench/results.json` written
by the scripts in this repo does.

Targets (from the program):

| quantity | target | script | counts as |
|---|---|---|---|
| environment steps per second per GPU | >= 50 000 | `bench/env_step.py` | the CUDA-graph replay at the batch the search will use (262 144 = population x seeds x episodes); eager reported alongside |
| batched bridge solves per second | >= 200 | `bench/sinkhorn_grid.py` (grid), `bench/bridge_fit.py` (neural) | reported separately for the two solver families; each is judged against the target on its own |
| population-parallel PPO updates per second | >= 128 | `bench/ppo_update.py` | genome-updates per second = P / seconds per full update (32 x 512 rollout, 4 epochs) at the largest P that fits |
| GPU-hours, actual vs plan | report | all | wall-clock summed by the harness |

Also recorded, no target: diffusion training and sampling throughput (`bench/diffusion_train.py`),
the covariance-steering gate at four precisions (`bench/gate_precision.py`; the solver
runs in fp32, the bf16 number says why), and the two-process bit-equality check
(`bench/determinism.py`).

Rule: if a measured quantity is below 50 % of its target, the grammar component that
depends on it is pruned from the search rungs before the remaining 90 % of the budget
runs, never replaced by something new. The pruning order if it comes to that: (1) the
MPC oracle to the validation rung only; (2) the per-mutant neural bridge to the grid
solver on rungs 0-1, neural bridges for elites and rung 2 only; (3) diffusion training
to 2000 steps on rung 0; (4) PPO to 250k steps on rung 0. Variants with zero rung-1
promotions in the pilot are removed. The grammar is not extended to compensate.

Measurement protocol: one process, GPU otherwise idle (checked with nvidia-smi and
recorded in `machine_info`), 3 warm-up and 5 timed episodes, median reported with the
10th/90th percentiles, peak memory recorded. A run with another CUDA process active is
labelled `shared_gpu` and repeated later.

---

## 1. Environment family

Registered before any environment beyond E1 is written. Each environment is a variant
of one codebase with the interface `reset(seed, descriptor) -> obs`,
`step(action) -> obs, reward, done, info`, `oracle()`, `demos(n)`, and a unit test that
its oracle succeeds >= 95 % undisturbed. E1 is the vendored SE(2) task; its fast step
(`sb/envs/fast.py`) is the kernel every variant extends.

| id | environment | what changes from E1 | descriptor axis it owns | oracle |
|---|---|---|---|---|
| E1 | terrain SE(2) | nothing (layouts L1, L2; slip / rain / push) | slip magnitude, slip anisotropy, push rate | MPPI with the true map (`MPCOracle`) |
| E2 | partial-observation terrain | the terrain map is hidden; obs = a local 5x5 patch of the property fields around the robot + pose | map correlation length (Voronoi seed count 4 / 14 / 40) | same planner, given the true map |
| E3 | multimodal goals | L2 extended to k in {2, 3, 4} gaps, terminal marginal a k-mode mixture; success = inside any mode | goal modality, mode separation | planner that picks the cheapest mode by expected slip along the straight path, then MPPI |
| E4 | contact | planar pusher: robot pushes a box through a gap; box-robot contact is a hybrid step with a friction cone | friction coefficient (0.3 / 0.6 / 0.9), box mass (1 / 2 / 4) | MPPI over the coupled robot-box kinematics |
| E5 | multi-agent | two SE(2) robots share a corridor; any collision fails both | corridor width over robot diameter (1.5 / 2.5 / 4) | joint MPPI over both robots |
| E6 | long horizon | 6-10 chained skills instead of 3; the time budget is 1.2x the oracle's | number of skills (6 / 8 / 10), time slack (1.2 / 1.5) | MPPI with per-skill time allocation proportional to path length |
| E7 | shifted demos | demos come from a mild terrain (slip_scale 0.35), test terrain is rough (1.4) | demo/test mismatch (0 / 2x / 4x) | MPPI on the test terrain |
| E8 | sparse demos | E1 with N_demo in {1, 2, 5} | N_demo | as E1 |

Additions applied to every environment:

- Terrain information given to every learned component in the same form the bridge's
  reference receives it: `terrain_info` in {none, oracle, estimated}. `estimated` is a
  running estimate of the local slip covariance from the last 50 steps' displacement
  residuals.
- Demo quality in {oracle MPC, oracle + 20 % action noise, scripted suboptimal (kp = 2
  tracker)}; human teleop is not available on this machine and is registered as absent.
- Raw-observation variants of E1 and E2 (64-beam lidar over the wall/pile geometry, or
  a 32x32 top-down occupancy image); the estimator slot and the latent-space bridge use
  these.
- Dynamics family in {unicycle (E1's body twist), Ackermann (curvature-limited), legged
  proxy (piecewise-holonomic with stance phases of 10 steps)}.
- Fidelity ladder: the fast batched step (rungs 0-1), a MuJoCo port of E1 and E4
  (rung 2), hardware (rung 3; registered unavailable).
- Oracle audit: for every environment the oracle's regret against a 10x larger MPPI
  (K = 2000, H = 30) is reported; an oracle more than 0.05 below the larger planner is
  replaced by it before the search starts.

Descriptor vector for the map (bins): env id (8) | slip magnitude (0.35, 0.7, 1.4, 2.8) |
slip anisotropy (1, 3, 10) | correlation length (4, 14, 40 seeds) | push rate multiplier
(0, 1, 3) | heading noise (0.05, 0.2, 0.5, 1.0) | N_demo (1, 5, 20, 100) | observability
(full, partial) | goal modality (1, 2, 3+) | horizon in skills (3, 6, 10) | terrain info
(none, oracle, estimated) | dynamics (unicycle, ackermann, legged) | demo quality (3).
CVT-MAP-Elites with 2000 centroids over the continuous embedding of these bins; the
centroid file is committed so the CVT seed is a sensitivity axis.

Sealed families E9 and E10: designed after the grammar is frozen by a fresh session that
gets only `sb/envs/base.py` and this table, never the archive; never evaluated during the
search; used only by the "why" model. Any archive row on E9/E10 before the freeze date
invalidates the "why" model and triggers the design of E11/E12.

Order of implementation: E1 (done) -> E2 (the estimator slot needs it, and the
`hidden_states` sibling repo supplies the belief machinery) -> E8 and E7 (parameter
changes only) -> E3 -> E6 -> E5 -> E4 (the only new physics). Each lands with its oracle
test and its bins in `sb/core/descriptor.py` before the next starts.

---

## 2. Grammar: genome, slots, components, fitness, ladder

Registered before any component in `sb/components/` exists. The grammar is frozen by
`python -m sb.cli freeze`, which runs every component's declared test, checks that every
bridge component has a nearest-non-bridge substitute for each slot it can fill, builds
the innovation table, and writes `grammar/manifest.json` with its hash. After the freeze
the only permitted change is shrink-only (`grammar/disabled.json`, with an ERRORS entry).
New ideas go to IDEAS_NEXT.md.

### 2.1 Genome

A genome is a directed graph of typed components filling typed slots (a slot may be
empty where the slot spec allows it), with continuous, integer and categorical
hyperparameters on the components, and the section-2.4 variation flags. Canonical form:
depth-first from the root in the fixed slot order, nodes renumbered; `gid` is the hash
of the canonical form, `sid` the same with parameter values stripped. The innovation
number of an edge is fixed at freeze from the sorted list of every legal
(parent component, slot, child component) triple, so it is the same in every island and
across resumes. One representation, three views: a fixed-length parameter vector for a
given `sid` (CMA-MAE, ES, the surrogate), the graph (structural mutation, NEAT-style
crossover aligned on innovation numbers), and an S-expression derivation tree in a DSL
whose nonterminals are the slot types and whose productions are the components (GGGP,
depth cap 6, node count as parsimony).

### 2.2 Slots

reference, planner, seam, controller, value, data, augment, trigger, noise, time-split,
manifold, estimator (E2 only), safety, adapt — each with the option sets of the program.
Following the gate (FINDINGS_generator, verdict "neither" on seeds 0-4; replication
pending at registration): the `data` slot carries no bridge option; it holds
{demos, noised, DART, PD-rollouts, PD-relabel (bridge states, PD labels), MPPI-rollouts,
GC-diffusion-rollouts, MPC-rollouts, mixture with learned ratios, RL-collected}.
PD-relabel keeps the bridge as a state generator, so its "bridge" tag stays for the
ablation delta (its substitute is PD-rollouts). If the replication reverses the verdict,
this paragraph is amended through PLAN_CHANGES before the freeze.

### 2.3 Bridge components

Every bridge is a `BridgeSolver`: `solve(mu0, mu1 | list, reference, cost, horizon) ->
Bridge` with `fwd_drift`, `bwd_drift`, `sample(x0, n, sampler, steps, gen)`, and must pass
the linear-Gaussian closed-form gate at 1e-5 in fp32 (tests/test_core.py) plus the
two-mode grid gate (a mixture pair with a known Sinkhorn solution). Each is labelled on
the four axes (reference, coupling, marginal constraint, solver/representation) and as
dynamic SB or entropic OT; reports are by axis. The variant list is the program's
(sb-brownian ... sb-score-bridge) with the solver flags on every variant: solver in
{DSBM, IPF, flow-iter0, dual}, IPF iterations in {0, 1, 2, 5}, epsilon log-uniform in
[1e-3, 1e-1], coupling in {independent, demo-paired, OT, minibatch-OT}, sampler in
{SDE-EM, ODE-Heun, ODE-RK4}, steps in {8, 16, 32, 64}. On rungs 0-1 the neural solvers
run at the pilot's budget or are replaced by the grid solver per section 0.4's pruning
rule; rung 2 always trains the neural solver at BRIDGE_CFG.

### 2.4 RL components

PPO or SAC (mutation picks), 100k steps at rung 0, 500k at rung 1, 2M at rung 2, reward
success - 0.01 |u|^2 unless the placement defines its own. Placements: RL-drift,
RL-residual-on-{PD, bridge, diffusion}, RL-cloud, RL-split, RL-switch, RL-noise,
RL-subgoal, RL-critic-as-value, bridge-shaped-RL, bridge-curriculum-RL,
bridge-exploration-RL, RL-finetune-diffusion. Every placement is warm-started (residuals
at zero; controllers from BC on the data slot; critics from distance or the bridge
log-density); rung 0 evaluates the warm start alone and after 100k steps and keeps the
placement if either beats its non-RL parent. RL placements are never pruned for lack of
steps alone. The population implementation is sb/rl/pop_ppo.py.

### 2.5 Variation flags

time-reversed execution, drift blending alpha(log-density), density gating, marginal
annealing, cost-in-reference vs cost-in-objective, per-skill epsilon, bridge-of-bridges,
ODE vs SDE sampling, learned intermediate marginals, action-space bridge. Each is a
mutation operator; all combinations are reachable.

### 2.6 Baselines in the menu

planner: MPPI, Diffuser, OT-flow-matching. controller: TD-MPC2-style, GC-RL with HER,
CVaR-MPC, DR-MPC. data: MPPI-rollouts, GC-diffusion-rollouts, MBRL-imagined. safety:
CBF-QP shield, bridge-density gate, DR-MPC. adapt: online bridge re-solve from the first
50 steps' slip estimate, online PD gain adaptation, online model update. Every baseline's
hyperparameters are tuned by the same budget through the no-bridge control search.

### 2.7 Fitness, constraints, secondary metrics

Fitness = mean success over the cell's disturbance set - 0.02 log(compute relative to
PD) - 0.05 collision rate, the three reported separately. Invalid (not low): inference
> 50 ms per step on CPU at batch 1; a failed component test; any read of the oracle at
test time (train-time only, and only in the data and value slots; enforced by the
capability object, a test wrapper without the oracle attribute, and a call counter).
Logged, not optimised: handoff error, energy, recovery rate, policy entropy, state
coverage, and for every bridge-containing genome the ablation delta = fitness minus the
fitness of the same genome with every bridge slot replaced by its nearest non-bridge
neighbour (bridge-mean -> spline, bridge-rollouts / PD-relabel -> PD-rollouts,
bridge-backward-drift -> distance, bridge-cloud -> waypoint, bridge-drift -> PD, bridge
plan -> MPPI, bridge-epsilon -> fixed Gaussian). NSGA-III objectives: success, CVaR_0.1,
worst-of-20, collision, energy, train compute, inference latency, demo count. Minimality
by greedy backward elimination of every validated elite; deployability = inference
<= 20 ms at a Jetson-class budget (reported, flagged).

### 2.8 Evaluation ladder

Rung 0: 1 seed, 30 episodes, 100k RL steps, E1, the 8 fixed cells (the 2x2x2 corners of
slip magnitude x push rate x observability mapped to centroids; stored in the manifest);
every mutant. Rung 1: 2 seeds, 100 episodes, 500k RL steps, the mutant's own cell;
promoted if rung-0 fitness >= current elite - 0.1. Rung 2: 10 seeds, 200 episodes, 2M RL
steps, the ablation delta, fresh disturbance seeds, fp32 with deterministic algorithms
and a two-run bit check before every batch; only elites that held a cell for >= 50
generations, and every elite at the end. Rung 3: hardware, registered unavailable. Every
evaluation at every rung is appended to `search/archive/` with its full descriptor,
fitness, secondary metrics and algorithm; nothing is deleted, only tagged.

### 2.9 Sealed families

After the freeze a fresh session receives `sb/envs/base.py`, `sb/envs/family.py` and the
section-1 table, nothing else, and designs E9 and E10. Their first archive rows carry the
freeze date; any earlier row invalidates the "why" model.

### 2.10 Pilot manifest

Frozen 2026-09-13 as `grammar_pilot/` (36 components, hash ef0314e95a24074f, 200 random
seed genomes) and re-frozen 2026-09-13 before any pilot row (hash 089779d1295a3a33,
component source hash 4574a117db226b49; PLAN_CHANGES 2026-09-13): the pilot's root
slots are manifold, seam, controller, trigger, noise, safety; the seam width and the
triggers now act on the stack. It is the pilot's grammar only; the main manifest is
frozen separately after stage C, and the "never extended mid-search" rule applies per
manifest.
