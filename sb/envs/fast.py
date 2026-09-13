"""A branch-free version of Task.dynamics for large batches and CUDA-graph capture.

Every robot in the batch carries its own disturbance kind and rain onset as tensors, all
four disturbance branches are computed and masked, the layout's gaps live in a padded
tensor, and there is no Python control flow, no .item() and no cholesky (Sigma is
diagonal, so its square root is elementwise). Fed the same standard-normal and uniform
draws in the same order, it reproduces Task.dynamics to float precision (tests/test_fast.py).
"""
import math

import torch

from sb.core import se2 as S
from sb.envs import task as TK
from sb.envs import terrain as TR

KINDS = {"none": 0, "slip": 1, "rain": 2, "push": 3}


class FastEnv:
    def __init__(self, layout, fields, rain_fields, kind, rain_t, push_k, push_p, push_mag, slip_scale, means, covs, device):
        self.device = device
        self.fields2 = torch.stack([fields, rain_fields]).to(device)                 # (2, F, G, G)
        self.kind = kind.to(device); self.rain_t = rain_t.to(device)
        self.push_k, self.push_p, self.push_mag = push_k.to(device), push_p.to(device), float(push_mag)
        self.slip_scale = float(slip_scale)
        gaps = torch.tensor(layout.gaps, dtype=torch.float32)                          # (Ng, 2): centre, width
        self.gap_c, self.gap_w = gaps[:, 0].to(device), gaps[:, 1].to(device)
        px, py, pr = layout.pile if layout.pile is not None else (0.0, 0.0, 0.0)
        self.pile = torch.tensor([px, py, pr], dtype=torch.float32, device=device)
        self.means = torch.stack(means).to(device)
        self.cov_inv = torch.stack([torch.linalg.inv(c) for c in covs]).to(device)
        self.n = kind.shape[0]

    @classmethod
    def from_task(cls, tk, kinds=None):
        """Mirror a Task/GenTask; `kinds` (n,) overrides the task's single disturbance."""
        n = tk.n if kinds is None else kinds.shape[0]
        kind = torch.full((n,), KINDS[tk.dist], dtype=torch.long) if kinds is None else kinds.long()
        rain_t = torch.where(kind == KINDS["rain"], torch.full((n,), int(min(tk.rain_t, 10 ** 9)), dtype=torch.long),
                             torch.full((n,), 10 ** 9, dtype=torch.long))
        push = tk.push if tk.push is not None else TK.PushField(0, 0.25, torch.device("cpu"))
        return cls(tk.layout, tk.true_fields, tk.rain_fields, kind, rain_t, push.k, push.p, push.mag, tk.slip_scale,
                   tk.means, tk.covs, tk.device)

    # ------------------------------------------------------------------ pieces
    def in_gap(self, y):
        d = (y[:, None] - self.gap_c[None]).abs()
        return (d < self.gap_w[None] / 2).any(1)

    def collides(self, a, b):
        da, db = a[:, 0] - 0.5, b[:, 0] - 0.5
        cross = (da * db) < 0
        t = da / (da - db + 1e-12)
        y = a[:, 1] + t * (b[:, 1] - a[:, 1])
        hit = cross & ~self.in_gap(y)
        d = ((b[:, 0] - self.pile[0]) ** 2 + (b[:, 1] - self.pile[1]) ** 2).sqrt()
        hit |= d < self.pile[2]
        hit |= (b < 0).any(1) | (b > 1).any(1)
        return hit

    def push_field(self, xy):
        a = xy @ self.push_k.T
        return self.push_mag * torch.stack([torch.sin(a + self.push_p[:, 0]).mean(1), torch.cos(a + self.push_p[:, 1]).mean(1)], 1)

    # -------------------------------------------------------------------- step
    def step(self, g, u, step, z, r, kz):
        """One physics step given the draws: z, kz ~ N(0, I) (n, 3); r ~ U(0, 1) (n,).
        `step` may be an int or a 0-d / (n,) long tensor."""
        u = TK.clip_u(u)
        step = torch.as_tensor(step, device=g.device)
        rain = (step >= self.rain_t)
        p0 = TR.lookup(self.fields2[0], g[:, :2]); p1 = TR.lookup(self.fields2[1], g[:, :2])
        p = torch.where(rain[:, None], p1, p0)
        none = (self.kind == 0)[:, None]
        mu = torch.where(none[:, 0], torch.ones_like(p[:, 3]), p[:, 3])
        rate = torch.where(none[:, 0], torch.zeros_like(p[:, 4]), p[:, 4])
        sd = torch.where(none, torch.full_like(p[:, :3], TK.SIGMA_BASE), ((p[:, :3] * self.slip_scale) ** 2 + TK.SIGMA_BASE ** 2).sqrt())
        xi = mu[:, None] * u * TK.DT + math.sqrt(TK.DT) * sd * z
        push = (self.kind == 3)
        f = self.push_field(g[:, :2])
        R = S.rot(-g[:, 2])
        xi = xi + torch.cat([push[:, None].float() * torch.einsum("nij,nj->ni", R, f) * TK.DT, torch.zeros_like(xi[:, 2:3])], 1)
        thr = rate * torch.where(push, torch.full_like(rate, 3.0), torch.ones_like(rate))
        mag = torch.where(push, torch.full_like(rate, 0.05), torch.full_like(rate, 0.03))
        kick = (r < thr).float() * mag
        xi = xi + kick[:, None] * kz
        g_new = S.compose(g, S.exp_se2(xi))
        hit = self.collides(g[:, :2], g_new[:, :2])
        g_new = torch.where(hit[:, None], g, g_new)
        return g_new, hit

    def step_random(self, g, u, step, gen=None):
        n, d = g.shape[0], g.device
        z = torch.randn(n, 3, generator=gen, device=d); r = torch.rand(n, generator=gen, device=d); kz = torch.randn(n, 3, generator=gen, device=d)
        return self.step(g, u, step, z, r, kz)

    def mahalanobis(self, g, k):
        xi = S.log_se2(S.compose(S.inv(self.means[k].expand_as(g)), g))
        return torch.einsum("ni,ij,nj->n", xi, self.cov_inv[k], xi).sqrt()


def draws_like_task(tk, n, gen):
    """The draws Task.dynamics consumes for one step, in its order, for a single-kind task."""
    d = tk.device
    z = torch.randn(n, 3, generator=gen, device=d)
    if tk.dist == "none":
        return z, torch.ones(n, device=d), torch.zeros(n, 3, device=d)
    r = torch.rand(n, generator=gen, device=d); kz = torch.randn(n, 3, generator=gen, device=d)
    return z, r, kz
