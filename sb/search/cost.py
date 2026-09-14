"""What a genome costs to train, as a deterministic function of the genome.

SEARCH_PLAN 2.7 scores a stack as success minus 0.02 log(compute relative to PD) minus
0.05 collision. "Compute" was read off the clock, which made the objective depend on
things that are not the genome: whether a cached net had already been trained by an
earlier evaluation (a grid bridge was charged 0.067 on its first evaluation and 0.0008 on
a later one - two thirds of the promotion margin, decided by evaluation order), how loaded
a GPU shared with other sessions happened to be, and which rung was running. It also made
fitness irreproducible, which contradicts the bit-check registered in 2.8, and it charged
nothing at all to the one stack that trains nothing - the PD, which is the baseline the
whole program compares bridges against (ERRORS 2026-09-14).

The cost here is computed from the components a genome contains and the parameters it
carries, with per-unit constants measured by `bench/` on this machine and recorded in the
plan. It is the same for every evaluation of the same genome at the same budget, it does
not depend on caches or neighbours, and the measured wall clock stays in the row as a
reported secondary metric.
"""
from sb.envs import task as TK

# seconds per unit of work, from bench/results.json on The_Tower (RTX 3080 Ti):
#   bridge_fit single_cuda 15.2 ms/solver step on GPU; the search fits on a CPU pool of
#   three, measured at 14.8 s for 3 skills x 2 directions x 300 steps -> 8.3e-3 s/step
#   diffusion_train single_b1024_graphed 0.687 ms/step
#   ppo at the evaluator's size: 0.144 s per update of 512 env-steps x 32 -> 3.8e-5 s/env-step
#   sinkhorn_grid float32: a 3-skill grid solve at 50 iterations, measured 0.04 s
BENCH_S = dict(bridge_solver_step=8.3e-3, diffusion_step=6.9e-4, rl_env_step=3.8e-5, grid_solve_50=0.04)
BRIDGE_STEPS0, BRIDGE_STEPS_IPF = 1200, 300      # SEARCH_BRIDGE_CFG; both directions are fit
PD_REFERENCE_S = 1.0                             # the denominator of "relative to PD": the PD trains nothing


def component_cost_s(comp, params, rl_steps, rung):
    """Seconds of training this component implies. Unknown components cost nothing, which
    is correct for every component that only reads demonstrations or wraps another."""
    p = dict(params)
    if comp == "controller.bridge_drift":
        steps = TK.N_SKILL * 2 * BRIDGE_STEPS0
        if rung >= 2:                                        # the validation rung honours the IPF gene
            steps += int(p.get("ipf", 0)) * TK.N_SKILL * 2 * BRIDGE_STEPS_IPF
        return steps * BENCH_S["bridge_solver_step"]
    if comp == "controller.grid_bridge":
        return BENCH_S["grid_solve_50"] * float(p.get("iters", 50)) / 50.0
    if comp == "controller.diffusion":
        return float(p.get("steps", 2000)) * BENCH_S["diffusion_step"]
    if comp in ("controller.ppo", "controller.rl_residual", "noise.rl"):
        return float(rl_steps) * BENCH_S["rl_env_step"]
    return 0.0


def train_cost_s(genome, grammar, rl_steps, rung=0):
    """The genome's training cost in seconds: every node, including sub-slots, so a
    residual policy pays for itself and for the base it corrects."""
    return sum(component_cost_s(n.comp, n.params, rl_steps, rung) for n in genome.nodes)
