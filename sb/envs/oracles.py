"""Privileged planners for the environment family (train-time only, behind Caps).

WallOracle is the true-map MPPI with the layout's wall in its cost and a per-robot goal;
E3 and E6 use it. E1 keeps gen.controllers.MPCOracle exactly as registered for the gate.
"""
import torch

from sb.core import se2 as S
from sb.envs import task as TK
from sb.gen.controllers import MPCOracle


class WallOracle(MPCOracle):
    WALL = 5.0

    @torch.no_grad()
    def act_to(self, g, goal, step0):
        """One MPPI step toward per-robot goals (n, 3), starting at absolute step `step0`."""
        n, d = g.shape[0], g.device
        if self.plan is None or self.plan.shape[0] != n:
            self.plan = torch.zeros(n, self.H, 3, device=d)
        base = self.plan.clone(); base[:, -1] = TK.clip_u(2.0 * S.between(g, goal))
        U = base[:, None] + self.noise * torch.randn(n, self.K, self.H, 3, generator=self.gen, device=d) * torch.tensor([1, 1, 3.0], device=d)
        U[:, 0] = base
        U = TK.clip_u(U.reshape(-1, 3)).reshape(n, self.K, self.H, 3)
        gs = g[:, None].expand(n, self.K, 3).reshape(-1, 3); gg = goal[:, None].expand(n, self.K, 3).reshape(-1, 3)
        cost = torch.zeros(n * self.K, device=d)
        for h in range(self.H):
            u = U[:, :, h].reshape(-1, 3)
            gs_new = self._model_step(gs, u, step0 + h)
            cost += self.WALL * self.tk.layout.collides(gs[:, :2], gs_new[:, :2]).float()
            gs = gs_new
            xi = S.between(gs, gg); d2 = (xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2
            cost += 0.3 * d2 + 0.0005 * (u ** 2).sum(1) * TK.DT
        xi = S.between(gs, gg); cost += 10.0 * ((xi[:, :2] ** 2).sum(1) + (S.HEADING_W * xi[:, 2]) ** 2)
        cost = cost.reshape(n, self.K)
        w = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / self.lam, 1)
        plan = (w[:, :, None, None] * U).sum(1)
        self.plan = torch.cat([plan[:, 1:], plan[:, -1:]], 1)
        return plan[:, 0]
