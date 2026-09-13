"""PPO for a population of P Gaussian policies at once.

Actors and critics are stacked (P, ...) parameter tensors; the rollout runs all P x n
environments as one FastEnv batch with branch-free auto-reset; GAE is vectorised over
(P, n); the update vmaps the clipped-surrogate loss over the population with per-genome
learning rate, clip range and entropy coefficient as (P,) tensors; Adam runs on the
stacked tensors. One population update launches roughly as many kernels as one
single-policy update.
"""
import copy
import math

import torch
import torch.nn as nn
from torch.func import functional_call, grad, stack_module_state, vmap

from sb.core import se2 as S
from sb.core.batched_fit import _adam_step
from sb.envs import task as TK
from sb.envs.fast import FastEnv
from sb.policies.common import OBS_DIM, obs_of

SCALE = torch.tensor([TK.UMAX, TK.UMAX, TK.WMAX])


def mlp(d_in, d_out, hidden=256, depth=4):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]; d = hidden
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class PopEnv:
    """P genomes x n environments on one FastEnv, with the skill clock as tensors.

    With `base` (a controller (g, k, tau, step) -> (u, extra)) the policy's action is a
    bounded residual added to the base command: u = base + bound * tanh(a). The same
    base serves every genome in the population; residuals start at zero."""

    def __init__(self, tk, P, n, kinds=None, base=None, bound=0.3, obs_fn=obs_of):
        self.tk, self.P, self.n, self.N, self.device = tk, P, n, P * n, tk.device
        self.obs_fn = obs_fn
        self.fe = FastEnv.from_task(tk, kinds=kinds if kinds is not None else torch.full((P * n,), FastEnv.from_task(tk).kind[0].item()))
        self.T = TK.T_SKILL
        self.base, self.bound = base, bound

    def reset(self):
        d = self.device
        self.g = self.tk.sample(0, self.N); self.k = torch.zeros(self.N, dtype=torch.long, device=d); self.t = torch.zeros_like(self.k)
        return self.obs()

    def obs(self):
        return self.obs_fn(self.tk, self.g, self.k, self.t.float() / self.T)

    def command(self, a):
        if self.base is None:
            return torch.tanh(a) * SCALE.to(a.device)
        k = int(self.k[0]); tau = self.t.float() / self.T          # the population shares one clock
        ub, _ = self.base(self.g, k, tau, k * self.T + int(self.t[0]))
        return TK.clip_u(ub) + self.bound * torch.tanh(a) * SCALE.to(a.device)

    def step(self, a):
        u = self.command(a)
        g_new, hit = self.fe.step_random(self.g, u, self.k * self.T + self.t)
        self.g = g_new; self.t = self.t + 1
        r = -0.01 * (u ** 2).sum(1) - 0.001 - hit.float()
        ho = self.t >= self.T
        k = torch.where(ho, self.k + 1, self.k); t = torch.where(ho, torch.zeros_like(self.t), self.t)
        fin = ho & (k >= 3)
        md = S.mahalanobis(self.g, self.tk.means[3], self.tk.covs[3])
        success = fin & (md <= 2.0)
        r = r + success.float()
        done = hit | fin
        fresh = self.tk.sample(0, self.N)
        self.g = torch.where(done[:, None], fresh, self.g)
        self.k = torch.where(done, torch.zeros_like(k), k); self.t = torch.where(done, torch.zeros_like(t), t)
        return self.obs(), r, done, success


class PopPPO:
    def __init__(self, P, device, hidden=256, depth=4, obs_dim=OBS_DIM, lr=3e-4, clip=0.2, ent=0.0, gamma=0.99, lam=0.95, seed=0, autocast=False):
        torch.manual_seed(seed)
        actors = [mlp(obs_dim, 3, hidden, depth) for _ in range(P)]; critics = [mlp(obs_dim, 1, hidden, depth) for _ in range(P)]
        self.actor_base = copy.deepcopy(actors[0]).to("meta"); self.critic_base = copy.deepcopy(critics[0]).to("meta")
        ap, _ = stack_module_state(actors); cp, _ = stack_module_state(critics)
        self.params = {**{"a." + k: v.to(device) for k, v in ap.items()}, **{"c." + k: v.to(device) for k, v in cp.items()},
                       "log_std": torch.full((P, 3), math.log(0.5), device=device)}
        self.params = {k: v.detach().clone() for k, v in self.params.items()}
        self.m = {k: torch.zeros_like(v) for k, v in self.params.items()}; self.v = {k: torch.zeros_like(v) for k, v in self.params.items()}
        as_p = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device).expand(P).clone()
        self.lr, self.clip, self.ent = as_p(lr), as_p(clip), as_p(ent)
        self.gamma, self.lam, self.P, self.device, self.step_count = gamma, lam, P, device, 0
        self.autocast = bool(autocast) and device.type == "cuda"      # bf16 on the search rungs only (SEARCH_PLAN 0.4)

    # ---------------------------------------------------------------- nets
    def _split(self, params):
        a = {k[2:]: v for k, v in params.items() if k.startswith("a.")}; c = {k[2:]: v for k, v in params.items() if k.startswith("c.")}
        return a, c, params["log_std"]

    def actor(self, params, obs):
        a, _, ls = self._split(params)
        mu = vmap(lambda p, x: functional_call(self.actor_base, p, (x,)))(a, obs)
        return mu, ls

    def critic(self, params, obs):
        _, c, _ = self._split(params)
        return vmap(lambda p, x: functional_call(self.critic_base, p, (x,)))(c, obs).squeeze(-1)

    @staticmethod
    def log_prob(mu, ls, a):
        std = ls.exp()[:, None, :]
        return (-0.5 * ((a - mu) / std) ** 2 - ls[:, None, :] - 0.5 * math.log(2 * math.pi)).sum(-1)

    # -------------------------------------------------------------- update
    def update(self, env, obs, rollout=32, epochs=4, minibatch=4096, gen=None):
        P, n, d = self.P, env.n, obs.shape[1]
        dev = self.device
        O = torch.zeros(rollout, P, n, d, device=dev); A = torch.zeros(rollout, P, n, 3, device=dev)
        LP = torch.zeros(rollout, P, n, device=dev); R = torch.zeros_like(LP); D = torch.zeros_like(LP); V = torch.zeros(rollout + 1, P, n, device=dev)
        succ = torch.zeros(P, device=dev); n_done = torch.zeros(P, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            for t in range(rollout):
                o = obs.view(P, n, d)
                mu, ls = self.actor(self.params, o); mu = mu.float()
                a = mu + ls.exp()[:, None, :] * torch.randn(P, n, 3, generator=gen, device=dev)
                O[t], A[t], LP[t], V[t] = o, a, self.log_prob(mu, ls, a), self.critic(self.params, o).float()
                obs, r, done, success = env.step(a.reshape(-1, 3))
                R[t], D[t] = r.view(P, n), done.view(P, n).float()
                succ += success.view(P, n).float().sum(1); n_done += done.view(P, n).float().sum(1)
            V[rollout] = self.critic(self.params, obs.view(P, n, d)).float()
        adv = torch.zeros_like(R); last = torch.zeros(P, n, device=dev)
        for t in reversed(range(rollout)):
            nonterm = 1 - D[t]
            delta = R[t] + self.gamma * V[t + 1] * nonterm - V[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + V[:rollout]
        b_obs = O.permute(1, 0, 2, 3).reshape(P, -1, d); b_a = A.permute(1, 0, 2, 3).reshape(P, -1, 3)
        b_lp = LP.permute(1, 0, 2).reshape(P, -1); b_adv = adv.permute(1, 0, 2).reshape(P, -1); b_ret = ret.permute(1, 0, 2).reshape(P, -1)
        b_adv = (b_adv - b_adv.mean(1, keepdim=True)) / (b_adv.std(1, keepdim=True) + 1e-8)
        N = b_obs.shape[1]; mb = min(minibatch, N)

        def loss_fn(params, obs_, a_, lp_, adv_, ret_, clip_, ent_):
            a_par = {k[2:]: v for k, v in params.items() if k.startswith("a.")}; c_par = {k[2:]: v for k, v in params.items() if k.startswith("c.")}
            mu = functional_call(self.actor_base, a_par, (obs_,)); ls = params["log_std"]
            std = ls.exp()
            lp = (-0.5 * ((a_ - mu) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
            ratio = (lp - lp_).exp()
            pg = -torch.minimum(ratio * adv_, ratio.clamp(1 - clip_, 1 + clip_) * adv_).mean()
            vl = ((functional_call(self.critic_base, c_par, (obs_,)).squeeze(-1) - ret_) ** 2).mean()
            entropy = (ls + 0.5 * math.log(2 * math.pi * math.e)).sum()
            return pg + 0.5 * vl - ent_ * entropy

        vgrad = vmap(grad(loss_fn))
        ar = torch.arange(P, device=dev)[:, None]
        for _ in range(epochs):
            perm = torch.argsort(torch.rand(P, N, generator=gen, device=dev), 1)
            for i in range(0, N, mb):
                idx = perm[:, i:i + mb]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                    grads = vgrad(self.params, b_obs[ar, idx], b_a[ar, idx], b_lp[ar, idx], b_adv[ar, idx], b_ret[ar, idx], self.clip, self.ent)
                grads = {k: g.float() for k, g in grads.items()}
                # per-genome global-norm clipping at 0.5
                sq = sum(g.reshape(P, -1).pow(2).sum(1) for g in grads.values())
                scale = (0.5 / (sq.sqrt() + 1e-6)).clamp(max=1.0)
                grads = {k: g * scale.view(-1, *([1] * (g.dim() - 1))) for k, g in grads.items()}
                self.step_count += 1
                _adam_step(self.params, grads, self.m, self.v, self.step_count, self.lr)
        return obs, dict(success=(succ / n_done.clamp_min(1)).detach(), episodes=n_done)

    def warm_start_bc(self, tk, G, U, steps=300, batch=512, lr=1e-3, obs_fn=obs_of, gen=None):
        """Behaviour-clone every actor's mean onto (obs, action) pairs from trajectories G, U
        (pre-tanh targets via atanh of the scaled action)."""
        from torch.func import functional_call, grad, vmap
        a_par = {k[2:]: v for k, v in self.params.items() if k.startswith("a.")}
        m = {k: torch.zeros_like(v) for k, v in a_par.items()}; vv = {k: torch.zeros_like(v) for k, v in a_par.items()}
        n = G.shape[0]; dev = self.device
        target_scale = SCALE.to(dev)

        def loss_fn(p, o, t):
            return ((functional_call(self.actor_base, p, (o,)) - t) ** 2).mean()

        vgrad = vmap(grad(loss_fn), in_dims=(0, None, None))
        for it in range(1, steps + 1):
            i = torch.randint(n, (batch,), generator=gen, device=dev); t0 = torch.randint(300, (batch,), generator=gen, device=dev)
            k = t0 // TK.T_SKILL; tau = (t0 % TK.T_SKILL).float() / TK.T_SKILL
            obs = obs_fn(tk, G[i, t0], k, tau)
            a = (U[i, t0] / target_scale).clamp(-0.999, 0.999); pre = torch.atanh(a)
            grads = vgrad(a_par, obs, pre)
            _adam_step(a_par, grads, m, vv, it, torch.full((self.P,), lr, device=dev))
        self.params.update({"a." + k: v for k, v in a_par.items()})

    @torch.no_grad()
    def act(self, obs_flat, n):
        """Deterministic actions for the flat (P*n, d) observation batch."""
        mu, _ = self.actor(self.params, obs_flat.view(self.P, n, -1))
        return torch.tanh(mu.reshape(-1, 3)) * SCALE.to(obs_flat.device)
