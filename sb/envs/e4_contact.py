"""E4: contact. A planar pusher (the SE(2) robot, disc of radius R) pushes a box (disc of
radius RB, mass ratio `mass`) through the L1 gap to a goal region. Quasi-static hybrid
model: when the discs touch, the box moves along the contact normal at the robot's
normal speed up to a cap that falls with mass (the robot cannot penetrate, so it is held
back to the same speed); the tangential component slides off unless it stays inside the
friction cone (|v_t| <= mu |v_n|), in which case the box also takes it. The box stops at the wall; the robot colliding with the wall fails the episode.

Descriptor axes: mu (friction coefficient) in {0.3, 0.6, 0.9}, mass in {1, 2, 4}.
Oracle audit (undisturbed, 64 episodes): 0.97 / 0.98 / 0.98 on the diagonal of the two axes
and 0.92 at the low-friction, heavy corner (mu 0.3, mass 4), which is below the family's
0.95 bar and is recorded as such.
State is (n, 6): robot pose (3) then box pose (x, y, theta).
"""
import math

import torch

from sb.core import se2 as S
from sb.envs import task as TK
from sb.envs import terrain as TR
from sb.envs.base import Caps, Descriptor
from sb.envs.e1_terrain import DEFAULTS, E1Terrain
from sb.envs.gen_task import GenTask
from sb.policies.common import obs_of

R, RB = 0.02, 0.03
BOX0 = (0.22, 0.5)                                  # box start (robot starts behind it at rho_0)
GOAL = (0.80, 0.5)                                  # box goal
GOAL_STD = 0.05
VPUSH = 0.8                                         # push speed of a unit-mass box, m/s


class PushTask(GenTask):
    def __init__(self, mu, mass, disturbance, n, seed, device, **kw):
        super().__init__(TR.Layout("L1"), disturbance, n, seed, device, **kw)
        self.mu, self.mass = mu, mass
        self.box_goal = torch.tensor(GOAL, device=device)
        self.means = [torch.tensor([0.12, 0.5, 0.0], device=device), torch.tensor([0.42, 0.5, 0.0], device=device),
                      torch.tensor([0.58, 0.5, 0.0], device=device), torch.tensor([0.88 - RB - R, 0.5, 0.0], device=device)]

    def sample_box(self, n):
        return torch.tensor(BOX0, device=self.device).expand(n, 2) + 0.01 * torch.randn(n, 2, generator=self.gen, device=self.device)

    def box_step(self, r_old, r_new, b):
        """Resolve the robot's move against the box. Returns (robot position after contact,
        new box position, blocked). Pushing along the contact normal moves both at the
        robot's normal speed up to the mass-dependent cap; the tangential part is shared
        only inside the friction cone, otherwise the robot slides along the box."""
        d0 = b - r_old[:, :2]; n = d0 / d0.norm(dim=1, keepdim=True).clamp_min(1e-9)
        v = r_new[:, :2] - r_old[:, :2]; vn = (v * n).sum(1); vt = v - vn[:, None] * n
        pen = (R + RB - (b - r_new[:, :2]).norm(dim=1)).clamp_min(0)
        pushing = (pen > 0) & (vn > 0)
        # mass caps the push speed: the box moves at most VPUSH / (1 + 0.25 (mass - 1)) per second
        # along the normal, and the robot, which cannot penetrate, is held back to match
        cap = VPUSH / (1.0 + 0.25 * (self.mass - 1.0)) * TK.DT
        step_n = torch.minimum(pen, torch.full_like(pen, cap))
        inside = vt.norm(dim=1) <= self.mu * vn.clamp_min(0)
        box_move = pushing[:, None].float() * (step_n[:, None] * n + inside[:, None].float() * vt)
        robot_back = pushing[:, None].float() * (pen - step_n)[:, None] * n
        b_new = b + box_move
        r_adj = r_new.clone(); r_adj[:, :2] = r_new[:, :2] - robot_back
        blocked = self.layout.collides(b, b_new) | (b_new < RB).any(1) | (b_new > 1 - RB).any(1)
        b_new = torch.where(blocked[:, None], b, b_new)
        r_adj[:, :2] = torch.where(blocked[:, None], r_old[:, :2] + vt, r_adj[:, :2])   # a blocked box stops the push
        return r_adj, b_new, blocked

    def marginal_md(self, g, k):
        if k < 3:
            return S.mahalanobis(g[:, :3], self.means[k], self.covs[k])
        return (g[:, 3:5] - self.box_goal).norm(dim=1) / GOAL_STD

    @torch.no_grad()
    def rollout(self, controller, n=None, record=False):
        n = n or self.n
        r = self.sample(0, n); b = self.sample_box(n); g = torch.cat([r, b, torch.zeros(n, 1, device=self.device)], 1)
        alive = torch.ones(n, dtype=torch.bool, device=self.device); energy = torch.zeros(n, device=self.device)
        handoff, reached, traj = [], [], ([g.clone()] if record else None)
        for k in range(TK.N_SKILL):
            for t in range(TK.T_SKILL):
                step = k * TK.T_SKILL + t
                tau = torch.full((n,), t / TK.T_SKILL, device=self.device)
                u, _ = controller(g, k, tau, step)
                energy = energy + (u ** 2).sum(1) * TK.DT * alive.float()
                r_new, hit = self.dynamics(g[:, :3], u, step)
                r_new, b_new, _ = self.box_step(g[:, :3], r_new, g[:, 3:5])
                g_new = torch.cat([r_new, b_new, g[:, 5:6]], 1)
                g = torch.where(alive[:, None], g_new, g); alive = alive & ~hit
                if record:
                    traj.append(g.clone())
            handoff.append(g.clone()); reached.append(alive & (self.marginal_md(g, k + 1) <= 2.0))
        goal_md = self.marginal_md(g, 3)
        succ = alive & (goal_md <= 2.0)
        out = dict(success=succ, alive=alive, energy=energy, goal_md=goal_md, final=g, handoff=handoff, reached=reached,
                   progress=torch.stack(reached, 1).float().mean(1))
        if record:
            out["traj"] = torch.stack(traj, 1)
        return out


class PushOracle:
    """Get behind the box on the line box -> goal, then push along that line; a PD on the
    robot's pose toward a moving target. Knows the box and the wall (the goal line passes
    through the gap)."""

    def __init__(self, tk, kp=8.0, bite=0.06):
        self.tk, self.kp, self.bite = tk, kp, bite

    def __call__(self, g, k, tau, step):
        r, b = g[:, :3], g[:, 3:5]
        goal = self.tk.box_goal.expand_as(b)
        to_goal = goal - b; dist = to_goal.norm(dim=1).clamp_min(1e-9); n = to_goal / dist[:, None]
        rel = r[:, :2] - b
        along = (rel * n).sum(1)                                          # > 0: in front of the box
        lateral = (rel[:, 0] * n[:, 1] - rel[:, 1] * n[:, 0]).abs()       # distance from the push line
        aligned = (along < -0.5 * (R + RB)) & (lateral < 0.3 * R)
        standoff = b - n * (R + RB + 0.03)                                # line up behind the box, no contact
        bite = b - n * (R + RB - self.bite)                               # push: a little into the box
        target_xy = torch.where(aligned[:, None], bite, standoff)
        target_xy = torch.where((dist < GOAL_STD * 0.5)[:, None], r[:, :2], target_xy)
        heading = torch.atan2(n[:, 1], n[:, 0])
        tgt = torch.cat([target_xy, heading[:, None]], 1)
        return self.kp * S.between(r, tgt), None


def obs_push(tk, g, k, tau):
    """robot obs_of features, box position relative to the robot in its body frame, box -> goal vector -> (n, 24 + 4)."""
    r, b = g[:, :3], g[:, 3:5]
    rel = torch.einsum("nij,nj->ni", S.rot(-r[:, 2]), b - r[:, :2])
    return torch.cat([obs_of(tk, r, k, tau), rel, tk.box_goal.expand_as(b) - b], 1)


class E4Contact(E1Terrain):
    id = "E4"
    axes = ("disturbance", "slip_scale", "aniso", "n_voronoi", "push_mult", "mu", "mass")
    defaults = dict(DEFAULTS, mu=0.6, mass=1.0)

    def _task(self, seed, d, n, disturbance=None):
        return PushTask(float(d["mu"]), float(d["mass"]), disturbance or d["disturbance"], n, seed, self.device,
                        slip_scale=d["slip_scale"], aniso=d["aniso"], n_seed=d["n_voronoi"], push_mult=d["push_mult"])

    def reset(self, seed, descriptor=None):
        d = dict(self.defaults); d.update(descriptor.values if isinstance(descriptor, Descriptor) else (descriptor or {}))
        self.d, self.seed = d, seed
        self.tk = self._task(seed, d, self.n)
        r = self.tk.sample(0, self.n); b = self.tk.sample_box(self.n)
        self.g = torch.cat([r, b, torch.zeros(self.n, 1, device=self.device)], 1)
        self.k = torch.zeros(self.n, dtype=torch.long, device=self.device); self.t = torch.zeros_like(self.k)
        return self.obs()

    def obs_dim(self):
        return 24 + 4

    def obs(self):
        return obs_push(self.tk, self.g, self.k, self.t.float() / self.T)

    def step(self, u):
        step = int(self.k[0]) * self.T + int(self.t[0])
        r_new, hit = self.tk.dynamics(self.g[:, :3], u, step)
        r_new, b_new, _ = self.tk.box_step(self.g[:, :3], r_new, self.g[:, 3:5])
        self.g = torch.cat([r_new, b_new, self.g[:, 5:6]], 1); self.t = self.t + 1
        rew = -0.01 * (TK.clip_u(u) ** 2).sum(1) - 0.001 - hit.float()
        ho = self.t >= self.T
        k = torch.where(ho, self.k + 1, self.k); t = torch.where(ho, torch.zeros_like(self.t), self.t)
        fin = ho & (k >= 3)
        success = fin & (self.tk.marginal_md(self.g, 3) <= 2.0)
        rew = rew + success.float()
        done = hit | fin
        fresh = torch.cat([self.tk.sample(0, self.n), self.tk.sample_box(self.n), torch.zeros(self.n, 1, device=self.device)], 1)
        self.g = torch.where(done[:, None], fresh, self.g)
        self.k = torch.where(done, torch.zeros_like(k), k); self.t = torch.where(done, torch.zeros_like(t), t)
        return self.obs(), rew, done, dict(success=success, collision=hit)

    def oracle(self, caps):
        assert isinstance(caps, Caps) and caps.oracle, "the oracle is train-time only"
        return PushOracle(self.tk)

    def demos(self, n, caps, kind="nominal"):
        raise NotImplementedError("E4 demos come from the push oracle (data slot, stage 2)")

    def invariant_geometry(self):
        geo = super().invariant_geometry()
        geo.update(box_radius=RB, robot_radius=R, box_goal=tuple(GOAL), goal_std=GOAL_STD)
        return geo
