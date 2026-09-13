"""E5: two SE(2) robots cross an intersection. Robot A runs left to right along y = 0.5,
robot B bottom to top along x = 0.5, each inside a corridor of width `ratio` times the
robot diameter; any robot-robot contact fails both. Descriptor axis: ratio in {1.5, 2.5, 4}.

State is (2n, 3): the first n rows are the A robots, the last n their B partners. The
oracle plans both robots of a pair jointly with an MPPI over the 6-d action, penalising
corridor exits and contact.
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
from sb.policies.common import OBS_DIM, obs_of

DIAM = 0.04
WIDE = torch.diag(torch.tensor([0.050, 0.030, 0.250])) ** 2
NARROW = torch.diag(torch.tensor([0.025, 0.025, 0.150])) ** 2
ROUTE_A = [(0.12, 0.5, 0.0), (0.42, 0.5, 0.0), (0.58, 0.5, 0.0), (0.88, 0.5, 0.0)]
ROUTE_B = [(0.5, 0.12, math.pi / 2), (0.5, 0.42, math.pi / 2), (0.5, 0.58, math.pi / 2), (0.5, 0.88, math.pi / 2)]


class CrossTask(GenTask):
    def __init__(self, ratio, disturbance, n_pairs, seed, device, **kw):
        super().__init__(TR.Layout("none"), disturbance, 2 * n_pairs, seed, device, **kw)
        self.n_pairs, self.width = n_pairs, ratio * DIAM
        self.means_a = [torch.tensor(m, device=device) for m in ROUTE_A]
        self.means_b = [torch.tensor(m, device=device) for m in ROUTE_B]
        self.means = self.means_a
        lat = min(0.03, self.width / 8)                     # lateral spread stays inside the corridor (3 sigma)
        self.covs = [torch.diag(torch.tensor([c[0, 0], lat ** 2, c[2, 2]])).to(device) for c in (WIDE, NARROW, NARROW, WIDE)]
        self.covs_b = self.covs                             # tangent-frame covariances: forward/lateral, same for both routes

    def is_b(self, n):
        return torch.arange(n, device=self.device) >= n // 2

    def sample(self, k, n):
        h = n // 2
        a = S.sample_marginal(self.means_a[k], self.covs[k], h, self.device, self.gen)
        b = S.sample_marginal(self.means_b[k], self.covs_b[k], n - h, self.device, self.gen)
        return torch.cat([a, b])

    def marginal_md(self, g, k):
        n = g.shape[0]; h = n // 2
        return torch.cat([S.mahalanobis(g[:h], self.means_a[k], self.covs[k]), S.mahalanobis(g[h:], self.means_b[k], self.covs_b[k])])

    def corridor_exit(self, xy):
        n = xy.shape[0]; b = self.is_b(n)
        off = torch.where(b, xy[:, 0] - 0.5, xy[:, 1] - 0.5).abs()
        return off > self.width / 2

    def contact(self, g):
        h = g.shape[0] // 2
        d = (g[:h, :2] - g[h:, :2]).norm(dim=1) < DIAM
        return torch.cat([d, d])

    def dynamics(self, g, u, step):
        g_new, hit = super().dynamics(g, u, step)
        hit = hit | self.corridor_exit(g_new[:, :2]) | self.contact(g_new)
        g_new = torch.where(hit.unsqueeze(1), g, g_new)
        return g_new, hit


class PairOracle(MPCOracle):
    """Joint MPPI over both robots of every pair, true map, corridor and contact costs."""
    CONTACT, CORRIDOR = 10.0, 5.0

    HOLD = torch.tensor([0.5, 0.40, math.pi / 2])

    @torch.no_grad()
    def act(self, g, k, t=0):
        tk = self.tk; n = g.shape[0]; h = n // 2; d = g.device
        goal = torch.cat([tk.means_a[k + 1].expand(h, 3), tk.means_b[k + 1].expand(n - h, 3)])
        if k == 1:
            # priority at the intersection: B holds short of it until its A partner has crossed
            passed = g[:h, 0] > 0.5 + 2 * DIAM
            hold = self.HOLD.to(d).expand(h, 3)
            goal = torch.cat([goal[:h], torch.where(passed[:, None], goal[h:], hold)])
        if self.plan is None or self.plan.shape[0] != n:
            self.plan = torch.zeros(n, self.H, 3, device=d)
        base = self.plan.clone(); base[:, -1] = TK.clip_u(2.0 * S.between(g, goal))
        # one noise draw per pair-candidate: A and B of a pair share candidate index j
        noise = self.noise * torch.randn(n, self.K, self.H, 3, generator=self.gen, device=d) * torch.tensor([1, 1, 3.0], device=d)
        U = base[:, None] + noise; U[:, 0] = base
        U = TK.clip_u(U.reshape(-1, 3)).reshape(n, self.K, self.H, 3)
        gs = g[:, None].expand(n, self.K, 3).reshape(-1, 3); gg = goal[:, None].expand(n, self.K, 3).reshape(-1, 3)
        cost = torch.zeros(n * self.K, device=d)
        step0 = k * TK.T_SKILL + t
        for hh in range(self.H):
            u = U[:, :, hh].reshape(-1, 3)
            gs = self._model_step(gs, u, step0 + hh)
            xi = S.between(gs, gg); d2 = (xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2
            cost += 0.3 * d2 + 0.0005 * (u ** 2).sum(1) * TK.DT
            xy = gs[:, :2].reshape(n, self.K, 2)
            off = torch.where(tk.is_b(n)[:, None], xy[..., 0] - 0.5, xy[..., 1] - 0.5).abs()
            cost += self.CORRIDOR * (off > tk.width / 2).float().reshape(-1)
            dist = (xy[:h] - xy[h:]).norm(dim=2)                       # (h, K) same candidate index for both
            pen = self.CONTACT * (dist < 1.5 * DIAM).float()
            cost += torch.cat([pen, pen]).reshape(-1)
        xi = S.between(gs, gg); cost += 10.0 * ((xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2)
        cost = cost.reshape(n, self.K)
        cost = torch.cat([cost[:h] + cost[h:], cost[:h] + cost[h:]])    # a pair shares its candidate's total cost
        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / self.lam, 1)
        plan = (w[:, :, None, None] * U).sum(1)
        self.plan = torch.cat([plan[:, 1:], plan[:, -1:]], 1)
        return plan[:, 0]


class PriorityPD:
    """The E5 oracle: geodesic PD (kp = 6) along each robot's own route, B holding short of
    the intersection during the crossing skill until its A partner has passed. A lane task
    wants a tracker; the joint MPPI above stays as the secondary planner."""
    HOLD = torch.tensor([0.5, 0.40, math.pi / 2])

    def __init__(self, tk, kp=10.0):
        self.tk, self.kp = tk, kp

    def __call__(self, g, k, tau, step):
        tk = self.tk; n = g.shape[0]; h = n // 2
        s = (tau + 1.0 / TK.T_SKILL).clamp(max=1.0)
        ref_a = S.SE2.interp(tk.means_a[k].expand(h, 3), tk.means_a[k + 1].expand(h, 3), s[:h])
        ref_b = S.SE2.interp(tk.means_b[k].expand(n - h, 3), tk.means_b[k + 1].expand(n - h, 3), s[h:])
        if k == 1:
            # B yields only while A is about to use the intersection and B has not crossed yet
            conflict = (g[:h, 0] > 0.5 - 0.10) & (g[:h, 0] < 0.5 + 2 * DIAM) & (g[h:, 1] < 0.5)
            ref_b = torch.where(conflict[:, None], self.HOLD.to(g.device).expand(h, 3), ref_b)
        ref = torch.cat([ref_a, ref_b])
        return self.kp * S.between(g, ref), None


def obs_pair(tk, g, k, tau):
    """Own obs_of features plus the partner's pose in the own body frame -> (n, 27)."""
    h = g.shape[0] // 2
    partner = torch.cat([g[h:], g[:h]])
    return torch.cat([obs_of(tk, g, k, tau), S.between(g, partner)], 1)


class E5Cross(E1Terrain):
    id = "E5"
    axes = ("disturbance", "slip_scale", "aniso", "n_voronoi", "push_mult", "ratio")
    defaults = dict(DEFAULTS, ratio=2.5)

    def __init__(self, n_pairs, device):
        super().__init__(2 * n_pairs, device)
        self.n_pairs = n_pairs

    def _task(self, seed, d, n, disturbance=None):
        return CrossTask(float(d["ratio"]), disturbance or d["disturbance"], n // 2, seed, self.device,
                         slip_scale=d["slip_scale"], aniso=d["aniso"], n_seed=d["n_voronoi"], push_mult=d["push_mult"])

    def reset(self, seed, descriptor=None):
        d = dict(self.defaults); d.update(descriptor.values if isinstance(descriptor, Descriptor) else (descriptor or {}))
        return super().reset(seed, d)

    def obs_dim(self):
        return OBS_DIM + 3

    def obs(self):
        return obs_pair(self.tk, self.g, self.k, self.t.float() / self.T)

    def step(self, u):
        g_new, hit = self.tk.dynamics(self.g, u, int(self.k[0]) * self.T + int(self.t[0]))
        self.g = g_new; self.t = self.t + 1
        r = -0.01 * (TK.clip_u(u) ** 2).sum(1) - 0.001 - hit.float()
        ho = self.t >= self.T
        k = torch.where(ho, self.k + 1, self.k); t = torch.where(ho, torch.zeros_like(self.t), self.t)
        fin = ho & (k >= 3)
        success = fin & (self.tk.marginal_md(self.g, 3) <= 2.0)
        h = self.n // 2
        pair_ok = success[:h] & success[h:]; success = torch.cat([pair_ok, pair_ok])
        r = r + success.float()
        done = hit | fin
        h_done = done[:h] | done[h:]; done = torch.cat([h_done, h_done])           # a pair resets together
        fresh = self.tk.sample(0, self.n)
        self.g = torch.where(done[:, None], fresh, self.g)
        self.k = torch.where(done, torch.zeros_like(k), k); self.t = torch.where(done, torch.zeros_like(t), t)
        return self.obs(), r, done, dict(success=success, collision=hit)

    def oracle(self, caps):
        assert isinstance(caps, Caps) and caps.oracle, "the oracle is train-time only"
        return PriorityPD(self.tk)

    def demos(self, n, caps, kind="nominal"):
        raise NotImplementedError("E5 demos come from the joint oracle (data slot, stage 2)")

    @torch.no_grad()
    def evaluate(self, controller, n=None):
        out = self.tk.rollout(controller, n=n or self.n)
        h = out["success"].shape[0] // 2
        out["pair_success"] = out["success"][:h] & out["success"][h:]
        return out
