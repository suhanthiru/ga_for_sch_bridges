"""E6: long horizon. Six to ten chained skills along a zigzag route through the L1 gap,
each skill's clock sized to `slack` times the time a reference speed needs for its
segment. Descriptor axes: n_skills in {6, 8, 10}, slack in {1.2, 1.5}.

The task keeps its own rollout because the skill count and per-skill horizons vary;
the controller contract is unchanged: (g, k, tau, step) -> (u, extra).
"""
import math

import torch

from sb.core import se2 as S
from sb.envs import task as TK
from sb.envs import terrain as TR
from sb.envs.base import Caps, Descriptor
from sb.envs.e1_terrain import DEFAULTS, E1Terrain
from sb.envs.gen_task import GenTask
from sb.envs.oracles import WallOracle

V_REF = 0.5
WIDE = torch.diag(torch.tensor([0.050, 0.080, 0.250])) ** 2
NARROW = torch.diag(torch.tensor([0.030, 0.040, 0.150])) ** 2


def zigzag(n_skills, amp=0.15):
    """n_skills + 1 waypoints from x = 0.1 to 0.9; the one nearest the wall sits on the gap."""
    xs = torch.linspace(0.1, 0.9, n_skills + 1)
    ys = torch.tensor([0.5 + amp * (-1) ** i for i in range(n_skills + 1)])
    j = int((xs - 0.5).abs().argmin()); ys[j] = 0.5; ys[0] = 0.5; ys[-1] = 0.5
    bearing = [math.atan2(float(ys[i + 1] - ys[i]), float(xs[i + 1] - xs[i])) for i in range(n_skills)]
    th = torch.zeros(n_skills + 1)
    th[0] = bearing[0]
    for i in range(1, n_skills):                            # intermediate: mean of incoming and outgoing bearings
        th[i] = math.atan2(math.sin(bearing[i - 1]) + math.sin(bearing[i]), math.cos(bearing[i - 1]) + math.cos(bearing[i]))
    th[-1] = 0.0
    return torch.stack([xs, ys, th], 1)


class LongChainTask(GenTask):
    def __init__(self, n_skills, slack, disturbance, n, seed, device, **kw):
        super().__init__(TR.Layout("L1"), disturbance, n, seed, device, **kw)
        self.n_skills, self.slack = n_skills, slack
        wp = zigzag(n_skills).to(device)
        self.means = [wp[i] for i in range(n_skills + 1)]
        self.covs = [WIDE.to(device)] + [NARROW.to(device).clone() for _ in range(n_skills - 1)] + [WIDE.to(device)]
        seg = (wp[1:, :2] - wp[:-1, :2]).norm(dim=1)
        self.t_skill = [max(20, int(round(slack * float(d) / (V_REF * TK.DT)))) for d in seg]
        self.starts = [sum(self.t_skill[:k]) for k in range(n_skills)]
        self.n_steps = sum(self.t_skill)

    def marginal_md(self, g, k):
        return S.mahalanobis(g, self.means[k], self.covs[k])

    @torch.no_grad()
    def rollout(self, controller, n=None, record=False):
        n = n or self.n
        g = self.sample(0, n)
        alive = torch.ones(n, dtype=torch.bool, device=self.device); energy = torch.zeros(n, device=self.device)
        handoff, reached, traj = [], [], ([g.clone()] if record else None)
        step = 0
        for k in range(self.n_skills):
            T = self.t_skill[k]
            for t in range(T):
                tau = torch.full((n,), t / T, device=self.device)
                u, _ = controller(g, k, tau, step)
                energy = energy + (u ** 2).sum(1) * TK.DT * alive.float()
                g_new, hit = self.dynamics(g, u, step)
                g = torch.where(alive.unsqueeze(1), g_new, g); alive = alive & ~hit
                if record:
                    traj.append(g.clone())
                step += 1
            handoff.append(g.clone()); reached.append(alive & (self.marginal_md(g, k + 1) <= 2.0))
        goal_md = self.marginal_md(g, self.n_skills)
        succ = alive & (goal_md <= 2.0)
        out = dict(success=succ, alive=alive, energy=energy, goal_md=goal_md, final=g, handoff=handoff, reached=reached,
                   progress=torch.stack(reached, 1).float().mean(1))
        if record:
            out["traj"] = torch.stack(traj, 1)
        return out


def obs_long(tk, g, k, tau):
    """pose, tau, skill fraction, world-frame terrain probes -> (n, 4 + 2 + 16)."""
    if not torch.is_tensor(k):
        k = torch.full((g.shape[0],), int(k), dtype=torch.long, device=g.device)
    frac = (k.float() / tk.n_skills).unsqueeze(1)
    return torch.cat([g[:, :2], g[:, 2:3].cos(), g[:, 2:3].sin(), tau.unsqueeze(1), frac, TR.terrain_feats(tk.obs_fields, g, False)], 1)


class E6Long(E1Terrain):
    id = "E6"
    axes = ("disturbance", "slip_scale", "aniso", "n_voronoi", "push_mult", "n_skills", "slack")
    defaults = dict(DEFAULTS, n_skills=6, slack=1.2)

    def _task(self, seed, d, n, disturbance=None):
        return LongChainTask(int(d["n_skills"]), float(d["slack"]), disturbance or d["disturbance"], n, seed, self.device,
                             slip_scale=d["slip_scale"], aniso=d["aniso"], n_seed=d["n_voronoi"], push_mult=d["push_mult"])

    def reset(self, seed, descriptor=None):
        d = dict(self.defaults); d.update(descriptor.values if isinstance(descriptor, Descriptor) else (descriptor or {}))
        return super().reset(seed, d)

    def _clocks(self):
        return torch.tensor(self.tk.t_skill, device=self.device), torch.tensor(self.tk.starts, device=self.device)

    def obs_dim(self):
        return 4 + 2 + TR.N_TFEAT

    def obs(self):
        T = self._clocks()[0][self.k.clamp(max=self.tk.n_skills - 1)]
        return obs_long(self.tk, self.g, self.k, self.t.float() / T.float())

    def step(self, u):
        T_k, starts = self._clocks(); kk = self.k.clamp(max=self.tk.n_skills - 1)
        T, start = T_k[kk], starts[kk]
        g_new, hit = self.fe.step_random(self.g, u, start + self.t)
        self.g = g_new; self.t = self.t + 1
        r = -0.01 * (TK.clip_u(u) ** 2).sum(1) - 0.001 - hit.float()
        ho = self.t >= T
        k = torch.where(ho, self.k + 1, self.k); t = torch.where(ho, torch.zeros_like(self.t), self.t)
        fin = ho & (k >= self.tk.n_skills)
        success = fin & (self.tk.marginal_md(self.g, self.tk.n_skills) <= 2.0)
        r = r + success.float()
        done = hit | fin
        fresh = self.tk.sample(0, self.n)
        self.g = torch.where(done[:, None], fresh, self.g)
        self.k = torch.where(done, torch.zeros_like(k), k); self.t = torch.where(done, torch.zeros_like(t), t)
        return self.obs(), r, done, dict(success=success, collision=hit)

    def oracle(self, caps):
        assert isinstance(caps, Caps) and caps.oracle, "the oracle is train-time only"
        m = WallOracle(self.tk, seed=self.seed)
        tk = self.tk

        def ctl(g, k, tau, step):
            return m.act_to(g, tk.means[k + 1].expand(g.shape[0], 3), step), None
        return ctl

    def demos(self, n, caps, kind="nominal"):
        raise NotImplementedError("E6 demos come from the oracle on the undisturbed task (data slot, stage 2)")
