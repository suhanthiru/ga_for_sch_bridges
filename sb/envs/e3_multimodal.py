"""E3: multimodal goals. A wall with k gaps, the handoff-2 and goal marginals are k-mode
mixtures (one mode behind each gap), and success is being inside any goal mode. The
oracle picks, per robot, the gap with the least expected slip along the straight route
on the observed map and plans to that mode.

Descriptor axes: n_modes in {2, 3, 4}, mode separation (the spacing between gap centres).
"""
import math

import torch

from sb.core import se2 as S
from sb.envs import task as TK
from sb.envs import terrain as TR
from sb.envs.base import Caps, Descriptor
from sb.envs.e1_terrain import DEFAULTS, E1Terrain
from sb.envs.gen_task import GenTask
from sb.gen.controllers import MPCOracle

GAP_W = 0.16
WIDE = torch.diag(torch.tensor([0.050, 0.080, 0.250])) ** 2
NARROW = torch.diag(torch.tensor([0.025, 0.040, 0.150])) ** 2


class MultiGoalTask(GenTask):
    def __init__(self, n_modes, separation, disturbance, n, seed, device, **kw):
        super().__init__(TR.Layout("L2"), disturbance, n, seed, device, **kw)
        ys = [0.5 + (j - (n_modes - 1) / 2) * separation for j in range(n_modes)]
        self.layout = TR.Layout("L2"); self.layout.gaps = [(y, GAP_W) for y in ys]; self.layout.pile = None
        self.n_modes, self.ys = n_modes, ys
        m0 = torch.tensor([0.12, 0.5, 0.0]); m1 = torch.tensor([0.42, 0.5, 0.0])
        self.modes2 = [torch.tensor([0.58, y, 0.5 * math.atan2(y - 0.5, 0.16)], device=device) for y in ys]
        self.modes3 = [torch.tensor([0.88, y, 0.0], device=device) for y in ys]
        self.means = [m0.to(device), m1.to(device), self.modes2[0], self.modes3[0]]
        self.covs = [WIDE.to(device), NARROW.to(device), NARROW.to(device), WIDE.to(device)]
        self.covs = [c.clone() for c in self.covs]

    def _modes(self, k):
        return self.modes2 if k == 2 else self.modes3

    def sample(self, k, n):
        if k < 2:
            return super().sample(k, n)
        pick = torch.randint(self.n_modes, (n,), generator=self.gen, device=self.device)
        draws = torch.stack([S.sample_marginal(m, self.covs[k], n, self.device, self.gen) for m in self._modes(k)], 1)
        return draws[torch.arange(n, device=self.device), pick]

    def marginal_md(self, g, k):
        if k < 2:
            return S.mahalanobis(g, self.means[k], self.covs[k])
        return torch.stack([S.mahalanobis(g, m, self.covs[k]) for m in self._modes(k)], 1).min(1).values

    def expected_slip(self, a, b, n_pt=16):
        s = torch.linspace(0, 1, n_pt, device=self.device)[None, :, None]
        pts = a[:, None, :2] * (1 - s) + b[:, None, :2] * s
        p = TR.lookup(self.obs_fields, pts.reshape(-1, 2))
        return (p[:, :3] ** 2).sum(1).reshape(a.shape[0], n_pt).mean(1)

    def choose_mode(self, g):
        """Per-robot index of the gap with the least expected slip along the straight route."""
        n = g.shape[0]
        cost = torch.stack([self.expected_slip(g, m.expand(n, 3)) for m in self.modes2], 1)
        return cost.argmin(1)


class ModeOracle(MPCOracle):
    """MPPI with the true map toward the robot's chosen mode."""

    def __init__(self, tk, seed=0, cost="greedy"):
        super().__init__(tk, seed=seed, cost=cost)
        self.mode = None

    @torch.no_grad()
    def act(self, g, k, t=0):
        if k == 1 and t == 0:
            self.mode = self.tk.choose_mode(g)
        if k >= 1 and self.mode is not None:
            n = g.shape[0]
            modes = self.tk.modes2 if k == 1 else self.tk.modes3
            goal = torch.stack(modes)[self.mode]
            return self._act_to(g, k, t, goal)
        return super().act(g, k, t)

    def _act_to(self, g, k, t, goal):
        n, d = g.shape[0], g.device
        if self.plan is None or self.plan.shape[0] != n:
            self.plan = torch.zeros(n, self.H, 3, device=d)
        base = self.plan.clone(); base[:, -1] = self._clip(2.0 * S.between(g, goal))
        U = base[:, None] + self.noise * torch.randn(n, self.K, self.H, 3, generator=self.gen, device=d) * torch.tensor([1, 1, 3.0], device=d)
        U[:, 0] = base
        U = self._clip(U.reshape(-1, 3)).reshape(n, self.K, self.H, 3)
        gs = g[:, None].expand(n, self.K, 3).reshape(-1, 3); gg = goal[:, None].expand(n, self.K, 3).reshape(-1, 3)
        cost = torch.zeros(n * self.K, device=d)
        step0 = k * TK.T_SKILL + t
        for h in range(self.H):
            u = U[:, :, h].reshape(-1, 3)
            gs = self._model_step(gs, u, step0 + h)
            xi = S.between(gs, gg); d2 = (xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2
            cost += 0.3 * d2 + 0.0005 * (u ** 2).sum(1) * TK.DT
        xi = S.between(gs, gg); cost += 10.0 * ((xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2)
        cost = cost.reshape(n, self.K)
        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / self.lam, 1)
        plan = (w[:, :, None, None] * U).sum(1)
        self.plan = torch.cat([plan[:, 1:], plan[:, -1:]], 1)
        return plan[:, 0]

    @staticmethod
    def _clip(u):
        return TK.clip_u(u)


class E3Multimodal(E1Terrain):
    id = "E3"
    axes = ("disturbance", "slip_scale", "aniso", "n_voronoi", "push_mult", "n_modes", "separation")
    defaults = dict(DEFAULTS, n_modes=2, separation=0.24)

    def _task(self, seed, d, n, disturbance=None):
        return MultiGoalTask(int(d["n_modes"]), float(d["separation"]), disturbance or d["disturbance"], n, seed, self.device,
                             slip_scale=d["slip_scale"], aniso=d["aniso"], n_seed=d["n_voronoi"], push_mult=d["push_mult"])

    def reset(self, seed, descriptor=None):
        d = dict(self.defaults); d.update(descriptor.values if isinstance(descriptor, Descriptor) else (descriptor or {}))
        return super().reset(seed, d)

    def oracle(self, caps):
        assert isinstance(caps, Caps) and caps.oracle, "the oracle is train-time only"
        m = ModeOracle(self.tk, seed=self.seed)
        return lambda g, k, tau, step: (m.act(g, k, step % self.T), None)

    def invariant_geometry(self):
        geo = super().invariant_geometry()
        geo.update(modes2=[m.cpu() for m in self.tk.modes2], modes3=[m.cpu() for m in self.tk.modes3])
        return geo
