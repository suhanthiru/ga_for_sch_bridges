"""The pilot's component set (SEARCH_PLAN section 2, stage B minimum). Every component
is a small class with `build(params, ctx) -> object` where ctx carries the task, the
demos and the capability object; what it returns is what the stack compiler wires into
the controller contract (g, k, tau, step) -> (u, extra). Heavier machinery lives in
sb.core / sb.gen / sb.policies; these classes are the registry's view of it.
"""
import torch

from sb.core import se2 as S
from sb.core.registry import P, SlotSpec, component
from sb.envs import task as TK

T = "tests/components/test_core_set.py::"


# ------------------------------------------------------------------ manifold
@component("manifold.se2", ("manifold",), test=T + "test_manifolds")
class ManifoldSE2:
    def build(self, params, ctx):
        return S.SE2


@component("manifold.flat", ("manifold",), test=T + "test_manifolds")
class ManifoldFlat:
    def build(self, params, ctx):
        return S.Flat


# ----------------------------------------------------------------- reference
@component("reference.geodesic", ("reference",), test=T + "test_references")
class RefGeodesic:
    """The straight geodesic through the marginal means (what the PD tracks)."""
    def build(self, params, ctx):
        tk = ctx["task"]
        return lambda g, k, tau: S.SE2.interp(tk.means[k].expand_as(g), tk.means[k + 1].expand_as(g), tau)


@component("reference.spline", ("reference",), params={"n_knots": P.int_uniform(2, 8)}, test=T + "test_references")
class RefSpline:
    """Piecewise-geodesic reference through n_knots points on the mean path (a stand-in
    for a smoothed route; equals the geodesic for 2 knots)."""
    def build(self, params, ctx):
        tk = ctx["task"]; n = int(params["n_knots"])

        def ref(g, k, tau):
            a, b = tk.means[k].expand_as(g), tk.means[k + 1].expand_as(g)
            s = (tau * (n - 1)).floor() / (n - 1)
            return S.SE2.interp(a, b, s + (tau - s))
        return ref


@component("reference.nearest_demo", ("reference",), test=T + "test_references")
class RefNearestDemo:
    def build(self, params, ctx):
        from sb.gen.controllers import PDTracker
        G, U = ctx["demos"]
        pd = PDTracker(G, U)
        return lambda g, k, tau: pd.G[pd.idx if pd.idx is not None else S.se2_dist2(g, pd.G[:, 0]).argmin(1), (k * TK.T_SKILL + (tau * TK.T_SKILL).long()).clamp(max=TK.T_SKILL * 3 - 1) + 1]


# ------------------------------------------------------------------- planner
@component("planner.mppi", ("planner",), params={"K": P.choice([64, 200, 512]), "H": P.choice([10, 30, 60]), "lam": P.loguniform(1e-3, 1e-1)},
           cost=dict(gpu=True), test=T + "test_planner")
class PlannerMPPI:
    def build(self, params, ctx):
        from sb.gen.controllers import MPPI
        return MPPI(ctx["task"], K=int(params["K"]), H=int(params["H"]), lam=float(params["lam"]), seed=ctx.get("seed", 0))


@component("planner.spline", ("planner",), params={"n_knots": P.int_uniform(2, 8)}, test=T + "test_planner")
class PlannerSpline(RefSpline):
    pass


# ---------------------------------------------------------------------- seam
@component("seam.fixed_clock", ("seam",), test=T + "test_seams")
class SeamFixedClock:
    """The registered handoff marginals as they are (width 1); the clock hands off at fixed steps."""
    def build(self, params, ctx):
        return dict(kind="fixed_clock", width=1.0)


@component("seam.waypoint", ("seam",), test=T + "test_seams")
class SeamWaypoint:
    """A point handoff: the bridge's handoff marginals shrink to a quarter width."""
    def build(self, params, ctx):
        return dict(kind="waypoint", width=0.25)


@component("seam.marginal_cloud", ("seam",), params={"width": P.loguniform(0.25, 4.0)}, test=T + "test_seams")
class SeamCloud:
    """A cloud handoff: the bridge is trained between handoff marginals scaled by `width`
    (the seam suite's knob). Controllers that track the mean path are unaffected."""
    def build(self, params, ctx):
        return dict(kind="cloud", width=float(params["width"]))


def seam_width(ctx):
    seam = ctx.get("seam") or {}
    return float(seam.get("width", 1.0))


# ---------------------------------------------------------------- controller
@component("controller.pd", ("controller",), params={"kp": P.loguniform(1.0, 30.0)},
           sub_slots={"reference": SlotSpec("reference")}, test=T + "test_controllers")
class CtlPD:
    def build(self, params, ctx):
        from sb.policies.nominal import Nominal
        kp = float(params["kp"]); ref = ctx.get("sub", {}).get("reference")
        if ref is None:
            return Nominal(ctx["task"], kp=kp)
        return lambda g, k, tau, step: (kp * S.between(g, ref(g, k, (tau + 1.0 / TK.T_SKILL).clamp(max=1.0))), None)


@component("controller.bridge_drift", ("controller",), params={"eps": P.loguniform(1e-3, 1e-1), "ipf": P.choice([0, 1, 2, 5]),
           "coupling": P.choice(["independent", "demo_paired", "ot", "minibatch_ot"]), "sampler": P.choice(["sde_em", "ode_heun", "ode_rk4"]),
           "steps": P.choice([8, 16, 32, 64]), "reference": P.choice(["brownian", "unicycle", "slip"])},
           sub_slots={"noise": SlotSpec("noise")}, tag="bridge", cost=dict(train_s_rung0=40, gpu=False),
           axes=dict(role="bridge_drift", reference="ou", coupling="param", marginal="hard_two", representation="neural_drift", family="dynamic_sb"),
           test=T + "test_controllers")
class CtlBridgeDrift:
    """Neural (DSBM iteration-0) bridge drift. The pilot pruned per-mutant neural solving
    from rungs 0-1 (PLAN_CHANGES 2026-09-13): the nets come from the per-cell cache
    trained once at BRIDGE_CFG for the chosen reference; the solver flags among the
    parameters are honoured at rung 2 only."""
    def build(self, params, ctx):
        from sb.core import solver as SV
        from sb.gen.bridges import SEARCH_BRIDGE_CFG, get_bridges
        tk = ctx["task"]
        nets = get_bridges(params["reference"], tk, tk.layout.name, ctx.get("seed", 0), tk.device, ctx["models"], mf=ctx["manifold"],
                           cfg=SEARCH_BRIDGE_CFG, width=seam_width(ctx))
        # (g, k, tau, step) -> (u, extra); extra[:, 0] is the forward/backward disagreement D
        return SV.BridgeController(nets, ctx["manifold"], tk, 0, tk.device, with_D=True)


@component("controller.grid_bridge", ("controller",), params={"eps": P.loguniform(2e-3, 5e-2), "grid": P.choice([32, 64]), "iters": P.choice([50, 200]),
           "kp_heading": P.loguniform(1.0, 10.0)}, tag="bridge", cost=dict(train_s_rung0=0.5, gpu=True),
           axes=dict(role="bridge_drift", reference="heat_kernel", coupling="independent", marginal="hard_two", representation="grid_sinkhorn", family="entropic_ot"),
           test=T + "test_controllers")
class CtlGridBridge:
    """Entropic-OT bridge on a grid between consecutive marginals, executed as a drift:
    the potentials give the conditional mean displacement at each cell and time, turned
    into a body twist with a heading PD toward the motion direction."""
    def build(self, params, ctx):
        from sb.core import grid_sinkhorn as GS
        tk = ctx["task"]; dev = tk.device; G = int(params["grid"]); eps = float(params["eps"]); kp = float(params["kp_heading"])
        w = seam_width(ctx)
        mus = [GS.gaussian_marginal(G, (float(m[0]), float(m[1])), float(torch.diag(c)[:2].sqrt().mean()) * (w if 0 < i < 3 else 1.0), dev)
               for i, (m, c) in enumerate(zip(tk.means, tk.covs))]
        sols = [GS.solve(mus[k][None], mus[k + 1][None], eps, int(params["iters"])) for k in range(TK.N_SKILL)]

        def act(g, k, tau, step):
            u_, v_, kern, _ = sols[k]
            t = float(tau[0].clamp(0.0, 0.95))
            field = GS.drift_field(u_, v_, kern, eps, tau=t)[0]                      # (G, G, 2) world units per unit time
            ix = (g[:, 0] * G).long().clamp(0, G - 1); iy = (g[:, 1] * G).long().clamp(0, G - 1)
            vw = field[iy, ix]                                                         # (n, 2) world-frame velocity
            R = S.rot(-g[:, 2]); vb = torch.einsum("nij,nj->ni", R, vw)
            head = torch.atan2(vw[:, 1], vw[:, 0]); dth = S.wrap(head - g[:, 2])
            w = kp * dth * (vw.norm(dim=1) > 1e-3).float()
            return torch.cat([vb, w[:, None]], 1), None
        return act


@component("controller.diffusion", ("controller",), params={"steps": P.choice([2000, 8000]), "hidden": P.choice([128, 256])},
           sub_slots={"data": SlotSpec("data", optional=False)}, cost=dict(gpu=True), test=T + "test_controllers")
class CtlDiffusion:
    def build(self, params, ctx):
        from sb.policies.diffusion import train_diffusion
        from sb.policies.evaluate import ChunkCtl
        G, U = ctx["sub"]["data"]
        from sb.policies.common import OBS_DIM, obs_of
        obs_fn, obs_dim = ctx.get("obs_fn", obs_of), ctx.get("obs_dim", OBS_DIM)
        pol = train_diffusion(ctx["task"], G, U, ctx.get("seed", 0), ctx["task"].device, int(params["steps"]), obs_fn=obs_fn, obs_dim=obs_dim)
        return ChunkCtl(pol, ctx["task"], obs_fn)


@component("controller.ppo", ("controller",), params={"lr": P.loguniform(1e-4, 1e-3), "clip": P.uniform(0.1, 0.3), "ent": P.loguniform(1e-4, 1e-2)},
           sub_slots={"data": SlotSpec("data", optional=False)}, tag="rl", cost=dict(gpu=True), test=T + "test_controllers")
class CtlPPO:
    """A Gaussian policy trained by PPO at the rung's step budget, warm-started by
    behaviour cloning on the data sub-slot (SEARCH_PLAN 2.4: every RL placement has a
    warm start). The evaluator does the training; build returns the recipe."""
    def build(self, params, ctx):
        return dict(kind="ppo", lr=float(params["lr"]), clip=float(params["clip"]), ent=float(params["ent"]), data=ctx["sub"]["data"])


@component("controller.rl_residual", ("controller",), params={"lr": P.loguniform(1e-4, 1e-3), "clip": P.uniform(0.1, 0.3), "ent": P.loguniform(1e-4, 1e-2),
           "bound": P.choice([0.1, 0.3, 0.6])}, sub_slots={"base": SlotSpec("controller", optional=False)}, tag="rl", cost=dict(gpu=True),
           test=T + "test_controllers")
class CtlRLResidual:
    """A bounded PPO residual on top of the base controller in the sub-slot; the residual
    starts at zero (its warm start), so the stack begins as the base and can only be
    trained away from it. The evaluator does the training."""
    def build(self, params, ctx):
        base = ctx["sub"]["base"]
        if isinstance(base, dict):
            raise ValueError("the base of a residual must be a trained controller, not another RL recipe")
        return dict(kind="ppo_residual", lr=float(params["lr"]), clip=float(params["clip"]), ent=float(params["ent"]),
                    bound=float(params["bound"]), base=base)


# ---------------------------------------------------------------------- data
def _src(name, extra=None):
    def build(self, params, ctx):
        from sb.gen.sources import build as build_source
        tk = ctx["task"]; kw = dict(n_demo=int(params.get("n_demo", 20)), mult=int(params.get("mult", 4)))
        G, U, meta, _ = build_source(name, tk, tk.layout.name, ctx.get("seed", 0), tk.device, ctx["demos"], ctx["models"], **kw)
        return G, U
    return build


for _name, _key in (("DEMO", "data.demos"), ("DART", "data.dart"), ("PD-noise", "data.pd_rollouts"), ("PD-relabel", "data.pd_relabel"),
                    ("MPPI-rollouts", "data.mppi_rollouts"), ("GC-diff-rollouts", "data.gc_diffusion"), ("MPC-oracle", "data.oracle_rollouts")):
    _tag = "bridge" if _key == "data.pd_relabel" else "none"
    _axes = dict(role="pd_relabel") if _key == "data.pd_relabel" else {}
    _oracle = "train_data" if _key in ("data.oracle_rollouts", "data.demos") else "none"
    component(_key, ("data",), params={"n_demo": P.choice([1, 5, 20, 100]), "mult": P.choice([1, 4, 16])}, tag=_tag, oracle=_oracle,
              axes=_axes, cost=dict(gpu=True), test=T + "test_data")(type(_key.replace(".", "_"), (), {"build": _src(_name)}))


# ------------------------------------------------------------------- augment
@component("augment.noised", ("augment",), params={"mult": P.choice([1, 4, 16])}, test=T + "test_augment")
class AugNoised:
    def build(self, params, ctx):
        from sb.gen.coverage import noised_copies
        G, U = ctx["demos"]; m = int(params["mult"]) * G.shape[0]
        Gn, Un = noised_copies(G, U, m, ctx.get("seed", 0), G.device)
        return torch.cat([G, Gn]), torch.cat([U, Un])


@component("augment.pd_rollouts", ("augment",), params={"mult": P.choice([1, 4, 16])}, test=T + "test_augment")
class AugPD:
    build = _src("PD-noise")


# ------------------------------------------------------------- trigger, noise
@component("trigger.distance", ("trigger",), params={"thr": P.loguniform(0.02, 0.5)}, test=T + "test_small_slots")
class TrigDistance:
    def build(self, params, ctx):
        return dict(kind="distance", thr=float(params["thr"]))


@component("trigger.bridge_disagreement", ("trigger",), params={"thr": P.loguniform(0.05, 1.0)}, tag="bridge",
           axes=dict(role="bridge_disagreement"), test=T + "test_small_slots")
class TrigDisagreement:
    """Restarts the skill clock where the controller's forward/backward drift disagreement
    D exceeds thr. Only a neural bridge controller exposes D; over any other controller
    the trigger never fires (recorded as such, not silently replaced by distance)."""
    def build(self, params, ctx):
        return dict(kind="disagreement", thr=float(params["thr"]))


@component("noise.fixed", ("noise",), params={"sigma": P.loguniform(1e-3, 1e-1)}, test=T + "test_small_slots")
class NoiseFixed:
    def build(self, params, ctx):
        return dict(kind="fixed", sigma=float(params["sigma"]))


@component("noise.rl", ("noise",), params={"lr": P.loguniform(1e-4, 1e-3), "clip": P.uniform(0.1, 0.3), "ent": P.loguniform(1e-4, 1e-2)},
           tag="rl", cost=dict(gpu=True), test=T + "test_small_slots")
class NoiseRL:
    """The learned-epsilon placement: a PPO policy sets the execution noise level of the
    controller from the state. Trained by the evaluator around the compiled controller."""
    def build(self, params, ctx):
        return dict(kind="rl_noise", lr=float(params["lr"]), clip=float(params["clip"]), ent=float(params["ent"]))


@component("noise.per_skill", ("noise",), params={"s0": P.loguniform(1e-3, 1e-1), "s1": P.loguniform(1e-3, 1e-1), "s2": P.loguniform(1e-3, 1e-1)},
           test=T + "test_small_slots")
class NoisePerSkill:
    def build(self, params, ctx):
        return dict(kind="per_skill", sigmas=[float(params[k]) for k in ("s0", "s1", "s2")])


# ---------------------------------------------------- value, estimator, safety
@component("value.distance", ("value",), test=T + "test_small_slots")
class ValueDistance:
    def build(self, params, ctx):
        tk = ctx["task"]
        return lambda g, k: -S.between(g, tk.means[k + 1].expand_as(g))[:, :2].norm(dim=1)


@component("estimator.ekf", ("estimator",), params={"window": P.int_uniform(10, 100)}, test=T + "test_small_slots")
class EstimatorEKF:
    """Running estimate of the local slip covariance from displacement residuals. Not a
    pilot root slot: nothing consumes it yet (stage C wires it into the observation)."""
    def build(self, params, ctx):
        return dict(kind="ekf", window=int(params["window"]))


@component("safety.clip", ("safety",), test=T + "test_small_slots")
class SafetyClip:
    def build(self, params, ctx):
        return lambda u: TK.clip_u(u)


@component("safety.none", ("safety",), test=T + "test_small_slots")
class SafetyNone:
    def build(self, params, ctx):
        return lambda u: u


@component("adapt.none", ("adapt",), test=T + "test_small_slots")
class AdaptNone:
    def build(self, params, ctx):
        return None


@component("time_split.equal", ("time_split",), test=T + "test_small_slots")
class SplitEqual:
    def build(self, params, ctx):
        return [TK.T_SKILL] * TK.N_SKILL
