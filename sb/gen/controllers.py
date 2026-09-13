"""Controllers the generators roll out. All return a body-frame command for (g, k, t).

  PDTracker    kp=10 + feed-forward on the nearest demo (by start pose)
  MPPI         sampling MPC on the reference kinematics; knows the kinematics, not the terrain
  MPCOracle    the same planner with the true friction/rain fields as its model
  gc_diffusion goal-conditioned diffusion policy rolled out with its own sampling noise
"""
import torch

from sb.core import se2 as S
from sb.core import solver as SV
from sb.envs import task as TK
from sb.envs import terrain as TR
from sb.gen.noise import DT, T_SKILL
from sb.policies.common import obs_gc
from sb.policies.evaluate import ChunkCtl
from sb.policies.nominal import Nominal

KP_PD = 10.0


class PDTracker:
    def __init__(self, demo_G, demo_U):
        self.G, self.U, self.idx = demo_G, demo_U, None

    def __call__(self, g, t):
        if t == 0 or self.idx is None:
            d = S.se2_dist2(g, self.G[:, 0]); self.idx = d.argmin(1)
        ref = self.G[self.idx, t + 1]
        return self.U[self.idx, t] + KP_PD * S.between(g, ref)


class MPPI:
    """K candidate sequences over horizon H, MPPI-weighted, warm-started.

    cost = "greedy": distance to the next marginal mean at every horizon step (the prior
    suite's cost). cost = "track": distance to the geodesic interpolant at the time each
    horizon step will be reached, so the labels move at the demonstrator's pace.
    """

    def __init__(self, tk, K=200, H=30, noise=0.5, lam=0.01, seed=0, cost="greedy"):
        self.tk, self.K, self.H, self.noise, self.lam, self.cost = tk, K, H, noise, lam, cost
        self.gen = torch.Generator(device=tk.device).manual_seed(seed)
        self.plan = None

    def _model_step(self, gs, u, step):
        return S.compose(gs, S.exp_se2(u * DT))

    def _target(self, k, t, h, n):
        goal = self.tk.means[k + 1].expand(n, 3)
        if self.cost == "greedy":
            return goal
        a = self.tk.means[k].expand(n, 3)
        tau = torch.full((n,), min(1.0, (t + h + 1) / T_SKILL), device=goal.device)
        return S.SE2.interp(a, goal, tau)

    @torch.no_grad()
    def act(self, g, k, t=0):
        n, d = g.shape[0], g.device
        goal = self.tk.means[k + 1].expand(n, 3)
        if self.plan is None or self.plan.shape[0] != n:
            self.plan = torch.zeros(n, self.H, 3, device=d)
        base = self.plan.clone(); base[:, -1] = TK.clip_u(2.0 * S.between(g, goal))
        U = base[:, None] + self.noise * torch.randn(n, self.K, self.H, 3, generator=self.gen, device=d) * torch.tensor([1, 1, 3.0], device=d)
        U[:, 0] = base
        U = TK.clip_u(U.reshape(-1, 3)).reshape(n, self.K, self.H, 3)
        gs = g[:, None].expand(n, self.K, 3).reshape(-1, 3)
        cost = torch.zeros(n * self.K, device=d)
        step0 = k * T_SKILL + t
        for h in range(self.H):
            u = U[:, :, h].reshape(-1, 3)
            gs = self._model_step(gs, u, step0 + h)
            tgt = self._target(k, t, h, n)[:, None].expand(n, self.K, 3).reshape(-1, 3)
            xi = S.between(gs, tgt); d2 = (xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2
            cost += 0.3 * d2 + 0.0005 * (u ** 2).sum(1) * DT
        tgt = self._target(k, t, self.H - 1, n)[:, None].expand(n, self.K, 3).reshape(-1, 3)
        xi = S.between(gs, tgt); cost += 10.0 * ((xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2)
        cost = cost.reshape(n, self.K)
        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / self.lam, 1)
        plan = (w[:, :, None, None] * U).sum(1)
        self.plan = torch.cat([plan[:, 1:], plan[:, -1:]], 1)
        return plan[:, 0]


class MPCOracle(MPPI):
    """MPPI whose internal model is the mean of the true dynamics: friction from the true
    map (rain fields after the onset), no noise. Used only to make the upper-bound data."""

    def _model_step(self, gs, u, step):
        mu = TR.lookup(self.tk.fields_at(step), gs[:, :2])[:, 3]
        return S.compose(gs, S.exp_se2(mu.unsqueeze(1) * u * DT))


# ------------------------------------------------------------ act closures
def bridge_act(nets, mf, tk):
    ctl = SV.BridgeController(nets, mf, tk, 0, tk.device, with_D=False)
    return lambda g, k, t: ctl(g, k, torch.full((g.shape[0],), t / T_SKILL, device=g.device), 0)[0]


def pd_act(demo_G, demo_U):
    pd = PDTracker(demo_G, demo_U)
    return lambda g, k, t: pd(g, k * T_SKILL + t)


def nominal_act(tk):
    nom = Nominal(tk)
    return lambda g, k, t: nom(g, k, torch.full((g.shape[0],), t / T_SKILL, device=g.device), 0)[0]


def mppi_act(tk, seed, cost="greedy"):
    m = MPPI(tk, seed=seed, cost=cost)
    return lambda g, k, t: m.act(g, k, t)


def oracle_act(tk_true, seed, cost="greedy"):
    m = MPCOracle(tk_true, seed=seed, cost=cost)
    return lambda g, k, t: m.act(g, k, t)


def gc_diffusion_act(pol, tk, seed):
    """Goal-conditioned chunk policy executed with its own sampling noise (seed 70000+seed)."""
    gen = torch.Generator(device=tk.device).manual_seed(70_000 + seed)
    ctl = ChunkCtl(pol, tk, obs_gc, gen)
    return lambda g, k, t: ctl(g, k, torch.full((g.shape[0],), t / T_SKILL, device=g.device), k * T_SKILL + t)[0]
