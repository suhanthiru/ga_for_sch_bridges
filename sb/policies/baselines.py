"""Non-diffusion chunk policies behind the same interface: train(kind, ...) -> policy.

  flow     conditional flow matching on the chunk, 10 Euler steps at test time
  bc       plain MLP regression on the chunk
  bc_gmm   MLP with a 5-component Gaussian-mixture head, NLL, executes the top component's mean
"""
import math

import torch
import torch.nn as nn

from sb.envs import task as TK
from sb.policies.common import CHUNK, OBS_DIM, SCALE, obs_of, sinusoidal


def windows(tk, G, U, batch, gen, device, obs_fn=obs_of):
    """Sample `batch` (obs, normalised action chunk) pairs from chained trajectories."""
    n = G.shape[0]
    i = torch.randint(n, (batch,), generator=gen, device=device)
    t0 = torch.randint(300 - CHUNK + 1, (batch,), generator=gen, device=device)
    obs = obs_fn(tk, G[i, t0], t0 // TK.T_SKILL, (t0 % TK.T_SKILL).float() / TK.T_SKILL)
    idx = t0[:, None] + torch.arange(CHUNK, device=device)[None]
    a0 = (U[i[:, None], idx] / SCALE.to(device)).clamp(-1, 1).reshape(batch, -1)
    return obs, a0


class FlowPolicy(nn.Module):
    def __init__(self, hidden=256, obs_dim=OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CHUNK * 3 + obs_dim + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, CHUNK * 3))

    def forward(self, a, obs, t):
        return self.net(torch.cat([a, obs, sinusoidal(t, 16)], 1))

    @torch.no_grad()
    def sample(self, obs, n_step=10, gen=None):
        n = obs.shape[0]
        a = torch.randn(n, CHUNK * 3, device=obs.device, generator=gen)
        for i in range(n_step):
            t = torch.full((n,), i / n_step, device=obs.device)
            a = a + self(a, obs, t) / n_step
        return a.clamp(-1, 1).reshape(n, CHUNK, 3) * SCALE.to(obs.device)


class BCPolicy(nn.Module):
    def __init__(self, hidden=256, gmm=0, obs_dim=OBS_DIM):
        super().__init__()
        self.gmm = gmm
        out = CHUNK * 3 if gmm == 0 else gmm * (2 * CHUNK * 3 + 1)
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out))

    def forward(self, obs):
        return self.net(obs)

    def nll(self, obs, a0):
        h = self(obs); n, D = a0.shape
        K = self.gmm
        logit = h[:, :K]; mu = h[:, K:K + K * D].reshape(n, K, D); ls = h[:, K + K * D:].reshape(n, K, D).clamp(-5, 2)
        lp = -0.5 * (((a0[:, None] - mu) / ls.exp()) ** 2 + 2 * ls + math.log(2 * math.pi)).sum(2)
        return -(torch.logsumexp(torch.log_softmax(logit, 1) + lp, 1)).mean()

    @torch.no_grad()
    def sample(self, obs, gen=None):
        h = self(obs); n = obs.shape[0]
        if self.gmm == 0:
            a = h
        else:
            K, D = self.gmm, CHUNK * 3
            logit = h[:, :K]; mu = h[:, K:K + K * D].reshape(n, K, D)
            a = mu[torch.arange(n), logit.argmax(1)]
        return a.clamp(-1, 1).reshape(n, CHUNK, 3) * SCALE.to(obs.device)


def train(kind, tk, G, U, seed, device, steps, log=lambda m: None, obs_fn=obs_of, obs_dim=OBS_DIM):
    """Same optimiser, batch and step budget for every policy class."""
    if kind == "diffusion":
        from sb.policies.diffusion import train_diffusion
        return train_diffusion(tk, G, U, seed, device, steps, log=log, obs_fn=obs_fn, obs_dim=obs_dim)
    torch.manual_seed(seed); gen = torch.Generator(device=device).manual_seed(seed)
    pol = (FlowPolicy(obs_dim=obs_dim) if kind == "flow" else BCPolicy(gmm=5 if kind == "bc_gmm" else 0, obs_dim=obs_dim)).to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    for _ in range(steps):
        obs, a0 = windows(tk, G, U, 1024, gen, device, obs_fn)
        if kind == "flow":
            t = torch.rand(1024, generator=gen, device=device); e = torch.randn_like(a0)
            at = (1 - t)[:, None] * e + t[:, None] * a0
            loss = ((pol(at, obs, t) - (a0 - e)) ** 2).mean()
        elif kind == "bc":
            loss = ((pol(obs) - a0) ** 2).mean()
        else:
            loss = pol.nll(obs, a0)
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  {kind} loss {loss.item():.4f}")
    pol.eval()
    return pol
