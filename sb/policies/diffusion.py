"""DDPM-50 / DDIM-10 chunk diffusion policy (the Phase 2/5 learner).

The 50-step schedule must reach abar_T ~ 0: beta up to 0.2 gives abar_T ~ 0.005. With the
1000-step DDPM betas (max 0.02) abar_T would be 0.6 and DDIM from pure noise would be
off-distribution; that was the defect in the earlier terrain-run baseline.
"""
import torch
import torch.nn as nn

from sb.envs import task as TK
from sb.policies.common import CHUNK, OBS_DIM, obs_of, sinusoidal


class DiffPolicy(nn.Module):
    def __init__(self, n_train=50, hidden=256, obs_dim=OBS_DIM):
        super().__init__()
        self.n_train, self.obs_dim = n_train, obs_dim
        self.net = nn.Sequential(nn.Linear(CHUNK * 3 + obs_dim + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, CHUNK * 3))
        self.register_buffer("abar", torch.cumprod(1 - torch.linspace(1e-4, 0.2, n_train), 0))
        self.scale = torch.tensor([TK.UMAX, TK.UMAX, TK.WMAX])

    def forward(self, a, obs, t):
        return self.net(torch.cat([a, obs, sinusoidal(t.float() / self.n_train, 16)], 1))

    @torch.no_grad()
    def sample(self, obs, n_ddim=10, gen=None):
        n = obs.shape[0]
        a = torch.randn(n, CHUNK * 3, device=obs.device, generator=gen)
        ts = torch.linspace(self.n_train - 1, 0, n_ddim + 1).long()[:-1]
        for i, t in enumerate(ts):
            ab = self.abar[t]; ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=obs.device)
            eps = self(a, obs, torch.full((n,), int(t), device=obs.device))
            a0 = ((a - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            a = ab_prev.sqrt() * a0 + (1 - ab_prev).sqrt() * eps
        return a.reshape(n, CHUNK, 3) * self.scale.to(obs.device)


def train_diffusion(tk, G, U, seed, device, steps, log=lambda m: None, obs_fn=obs_of, obs_dim=OBS_DIM):
    torch.manual_seed(seed)
    pol = DiffPolicy(obs_dim=obs_dim).to(device); opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    n = G.shape[0]; sc = pol.scale.to(device)
    for _ in range(steps):
        i = torch.randint(n, (1024,), device=device); t0 = torch.randint(300 - CHUNK + 1, (1024,), device=device)
        g = G[i, t0]; k = t0 // TK.T_SKILL; tau = (t0 % TK.T_SKILL).float() / TK.T_SKILL
        obs = obs_fn(tk, g, k, tau)
        idx = t0[:, None] + torch.arange(CHUNK, device=device)[None]
        a0 = (U[i[:, None], idx] / sc).clamp(-1, 1).reshape(1024, -1)
        t = torch.randint(pol.n_train, (1024,), device=device); ab = pol.abar[t][:, None]
        noise = torch.randn_like(a0)
        loss = ((pol(ab.sqrt() * a0 + (1 - ab).sqrt() * noise, obs, t) - noise) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  diffusion loss {loss.item():.4f}")
    pol.eval()
    return pol
