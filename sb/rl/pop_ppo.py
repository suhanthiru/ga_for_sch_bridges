"""PPO for a population of P Gaussian policies at once.

Actors and critics are stacked (P, ...) parameter tensors; the rollout runs all P x n
environments as one FastEnv batch with branch-free auto-reset; GAE is vectorised over
(P, n); the update vmaps the clipped-surrogate loss over the population with per-genome
learning rate, clip range and entropy coefficient as (P,) tensors; Adam runs on the
stacked tensors. One population update launches roughly as many kernels as one
single-policy update.

On CUDA the update can be graph-captured (`update(..., graphed=True)`): the whole
rollout (env steps, policy forwards, GAE, batch assembly) is one CUDA graph and a
minibatch gradient step is another, replayed per minibatch with the permutation copied
into a static index buffer. The environment's state lives in fixed buffers so the
captured step replays correctly; the user generators are registered with the graphs so
the draws stay reproducible from the seed. Plain-action PPO only: a residual over a
Python base controller cannot be captured and keeps the eager loop.
"""
import copy
import gc
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
_SCALE_ON = {}


def scale_on(device):
    """The action scale resident on `device` (no host copy per call: graph-capturable)."""
    key = str(device)
    if key not in _SCALE_ON:
        _SCALE_ON[key] = SCALE.to(device)
    return _SCALE_ON[key]


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

    def __init__(self, tk, P, n, kinds=None, base=None, bound=0.3, obs_fn=obs_of, mode="action", sigma_range=(1e-3, 1e-1)):
        """mode: "action" (the policy commands the twist), "residual" (bounded correction on
        `base`), or "noise" (the policy's first coordinate sets the log execution noise
        added to `base`, the learned-epsilon placement)."""
        self.tk, self.P, self.n, self.N, self.device = tk, P, n, P * n, tk.device
        self.obs_fn, self.mode, self.sigma_range = obs_fn, mode, sigma_range
        self.fe = FastEnv.from_task(tk, kinds=kinds if kinds is not None else torch.full((P * n,), FastEnv.from_task(tk).kind[0].item()))
        self.T = TK.T_SKILL
        self.base, self.bound = base, bound
        self.L0 = torch.linalg.cholesky(tk.covs[0])           # start-marginal factor, precomputed (no solver call per step)
        self.g = self.k = self.t = None

    def _fresh(self):
        xi = torch.randn(self.N, 3, generator=self.tk.gen, device=self.device) @ self.L0.T
        return S.compose(self.tk.means[0].expand(self.N, 3), S.exp_se2(xi))

    def reset(self):
        d = self.device
        self.g = self._fresh().contiguous(); self.k = torch.zeros(self.N, dtype=torch.long, device=d); self.t = torch.zeros_like(self.k)
        return self.obs()

    def obs(self):
        return self.obs_fn(self.tk, self.g, self.k, self.t.float() / self.T)

    def sigma_of(self, a):
        lo, hi = self.sigma_range
        return torch.exp(0.5 * (math.log(lo) + math.log(hi)) + 0.5 * (math.log(hi) - math.log(lo)) * torch.tanh(a[:, :1]))

    def command(self, a):
        if self.base is None:
            return torch.tanh(a) * scale_on(a.device)
        k = int(self.k[0]); tau = self.t.float() / self.T          # the population shares one clock
        ub, _ = self.base(self.g, k, tau, k * self.T + int(self.t[0]))
        if self.mode == "noise":
            return TK.clip_u(ub) + self.sigma_of(a) * torch.randn_like(ub)
        return TK.clip_u(ub) + self.bound * torch.tanh(a) * scale_on(a.device)

    def step(self, a):
        u = self.command(a)
        g_new, hit = self.fe.step_random(self.g, u, self.k * self.T + self.t)
        t = self.t + 1
        r = -0.01 * (u ** 2).sum(1) - 0.001 - hit.float()
        ho = t >= self.T
        k = torch.where(ho, self.k + 1, self.k); t = torch.where(ho, torch.zeros_like(t), t)
        fin = ho & (k >= 3)
        md = self.fe.mahalanobis(g_new, 3)
        success = fin & (md <= 2.0)
        r = r + success.float()
        done = hit | fin
        fresh = self._fresh()
        # the state is written in place: fixed buffers are what a captured step replays on
        self.g.copy_(torch.where(done[:, None], fresh, g_new))
        self.k.copy_(torch.where(done, torch.zeros_like(k), k)); self.t.copy_(torch.where(done, torch.zeros_like(t), t))
        return self.obs(), r, done, success


def _make_loss(actor_base, critic_base):
    """The clipped-surrogate loss of one genome, closed over the meta modules only."""
    def loss(params, obs_, a_, lp_, adv_, ret_, clip_, ent_):
        a_par = {k[2:]: v for k, v in params.items() if k.startswith("a.")}; c_par = {k[2:]: v for k, v in params.items() if k.startswith("c.")}
        mu = functional_call(actor_base, a_par, (obs_,)); ls = params["log_std"]
        std = ls.exp()
        lp = (-0.5 * ((a_ - mu) / std) ** 2 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
        ratio = (lp - lp_).exp()
        pg = -torch.minimum(ratio * adv_, ratio.clamp(1 - clip_, 1 + clip_) * adv_).mean()
        vl = ((functional_call(critic_base, c_par, (obs_,)).squeeze(-1) - ret_) ** 2).mean()
        entropy = (ls + 0.5 * math.log(2 * math.pi * math.e)).sum()
        return pg + 0.5 * vl - ent_ * entropy
    return loss


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
        self.gamma, self.lam, self.P, self.device = gamma, lam, P, device
        self.t_adam = torch.zeros((), device=device)                    # Adam's step count as a tensor (graph-safe)
        self.autocast = bool(autocast) and device.type == "cuda"      # bf16 on the search rungs only (SEARCH_PLAN 0.4)
        self._vgrad = vmap(grad(_make_loss(self.actor_base, self.critic_base)))   # no reference to self: a PopPPO is freed by refcount
        self._ar = torch.arange(P, device=device)[:, None]
        self.graphs = {}

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


    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast, cache_enabled=False)

    # -------------------------------------------------------------- pieces
    def _bufs(self, rollout, n, d):
        P, dev = self.P, self.device
        return dict(O=torch.zeros(rollout, P, n, d, device=dev), A=torch.zeros(rollout, P, n, 3, device=dev),
                    LP=torch.zeros(rollout, P, n, device=dev), R=torch.zeros(rollout, P, n, device=dev),
                    D=torch.zeros(rollout, P, n, device=dev), V=torch.zeros(rollout + 1, P, n, device=dev))

    def _rollout(self, env, rollout, gen, bufs):
        """`rollout` steps from the env's current state into the buffers; returns the last
        observation, the flat (P, N, .) batch with normalised advantages, and the
        success / episode counts per genome."""
        P, n, dev = self.P, env.n, self.device
        O, A, LP, R, D, V = (bufs[k] for k in ("O", "A", "LP", "R", "D", "V"))
        succ = torch.zeros(P, device=dev); n_done = torch.zeros(P, device=dev)
        with torch.no_grad(), self._autocast():
            for t in range(rollout):
                o = env.obs().view(P, n, -1)
                mu, ls = self.actor(self.params, o); mu = mu.float()
                a = mu + ls.exp()[:, None, :] * torch.randn(P, n, 3, generator=gen, device=dev)
                O[t].copy_(o); A[t].copy_(a); LP[t].copy_(self.log_prob(mu, ls, a)); V[t].copy_(self.critic(self.params, o).float())
                obs, r, done, success = env.step(a.reshape(-1, 3))
                R[t].copy_(r.view(P, n)); D[t].copy_(done.view(P, n).float())
                succ += success.view(P, n).float().sum(1); n_done += done.view(P, n).float().sum(1)
            V[rollout].copy_(self.critic(self.params, obs.view(P, n, -1)).float())
        adv = torch.zeros_like(R); last = torch.zeros(P, n, device=dev)
        for t in reversed(range(rollout)):
            nonterm = 1 - D[t]
            delta = R[t] + self.gamma * V[t + 1] * nonterm - V[t]
            last = delta + self.gamma * self.lam * nonterm * last
            adv[t] = last
        ret = adv + V[:rollout]
        d = O.shape[-1]
        b_adv = adv.permute(1, 0, 2).reshape(P, -1)
        b = dict(obs=O.permute(1, 0, 2, 3).reshape(P, -1, d), a=A.permute(1, 0, 2, 3).reshape(P, -1, 3), lp=LP.permute(1, 0, 2).reshape(P, -1),
                 adv=(b_adv - b_adv.mean(1, keepdim=True)) / (b_adv.std(1, keepdim=True) + 1e-8), ret=ret.permute(1, 0, 2).reshape(P, -1))
        return obs, b, succ, n_done

    def _minibatch(self, b, idx):
        """One clipped-surrogate gradient step of every genome on its rows `idx` (P, mb)."""
        ar = self._ar
        with self._autocast():
            grads = self._vgrad(self.params, b["obs"][ar, idx], b["a"][ar, idx], b["lp"][ar, idx], b["adv"][ar, idx], b["ret"][ar, idx], self.clip, self.ent)
        grads = {k: g.float() for k, g in grads.items()}
        # per-genome global-norm clipping at 0.5
        sq = sum(g.reshape(self.P, -1).pow(2).sum(1) for g in grads.values())
        scale = (0.5 / (sq.sqrt() + 1e-6)).clamp(max=1.0)
        grads = {k: g * scale.view(-1, *([1] * (g.dim() - 1))) for k, g in grads.items()}
        self.t_adam.add_(1)
        _adam_step(self.params, grads, self.m, self.v, self.t_adam, self.lr)

    @staticmethod
    def _chunks(N, minibatch):
        """Equal minibatches (static shapes): n_mb chunks of mb rows; up to n_mb - 1 rows
        of each epoch's permutation are left out."""
        n_mb = max(1, math.ceil(N / minibatch)); mb = N // n_mb
        return n_mb, mb

    def _capture(self, env, rollout, n, mb, gen, d):
        """Warm up on a side stream, then capture the rollout and one minibatch step."""
        dev = self.device
        bufs = self._bufs(rollout, n, d); idx = torch.zeros(self.P, mb, dtype=torch.long, device=dev)
        gens = [g_ for g_ in (gen, getattr(env.tk, "gen", None)) if g_ is not None and g_.device.type == "cuda"]
        side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                                    # two warm-up updates, one minibatch each
            for _ in range(2):
                _, b, _, _ = self._rollout(env, rollout, gen, bufs)
                self._minibatch(b, torch.arange(mb, device=dev).expand(self.P, mb))
        torch.cuda.current_stream().wait_stream(side)
        del b
        # a CUDAGraph destroyed by the cyclic collector during a capture invalidates the capture:
        # collect first, then keep the collector off until both graphs are recorded
        gc.collect(); gc.disable()
        try:
            roll = torch.cuda.CUDAGraph()
            for g_ in gens:
                roll.register_generator_state(g_)
            with torch.cuda.graph(roll):
                obs, b, succ, n_done = self._rollout(env, rollout, gen, bufs)
            step = torch.cuda.CUDAGraph()
            with torch.cuda.graph(step, pool=roll.pool()):
                self._minibatch(b, idx)
        finally:
            gc.enable()
        return dict(roll=roll, step=step, idx=idx, obs=obs, succ=succ, n_done=n_done)

    # -------------------------------------------------------------- update
    def update(self, env, obs, rollout=32, epochs=4, minibatch=4096, gen=None, graphed=False):
        P, n, dev = self.P, env.n, self.device
        N = rollout * n; n_mb, mb = self._chunks(N, minibatch)
        graphed = bool(graphed) and dev.type == "cuda" and env.base is None
        if not graphed:
            obs, b, succ, n_done = self._rollout(env, rollout, gen, self._bufs(rollout, n, obs.shape[-1]))
            for _ in range(epochs):
                perm = torch.argsort(torch.rand(P, N, generator=gen, device=dev), 1)
                for i in range(n_mb):
                    self._minibatch(b, perm[:, i * mb:(i + 1) * mb])
            return obs, dict(success=(succ / n_done.clamp_min(1)).detach(), episodes=n_done)
        key = (rollout, n, mb, obs.shape[-1])
        G = self.graphs.get(key)
        if G is None:
            G = self.graphs[key] = self._capture(env, rollout, n, mb, gen, obs.shape[-1])
        G["roll"].replay()
        for _ in range(epochs):
            perm = torch.argsort(torch.rand(P, N, generator=gen, device=dev), 1)
            for i in range(n_mb):
                G["idx"].copy_(perm[:, i * mb:(i + 1) * mb]); G["step"].replay()
        return G["obs"], dict(success=(G["succ"] / G["n_done"].clamp_min(1)).detach().clone(), episodes=G["n_done"].clone())

    def warm_start_bc(self, tk, G, U, steps=300, batch=512, lr=1e-3, obs_fn=obs_of, gen=None):
        """Behaviour-clone every actor's mean onto (obs, action) pairs from trajectories G, U
        (pre-tanh targets via atanh of the scaled action)."""
        a_par = {k[2:]: v for k, v in self.params.items() if k.startswith("a.")}
        m = {k: torch.zeros_like(v) for k, v in a_par.items()}; vv = {k: torch.zeros_like(v) for k, v in a_par.items()}
        n = G.shape[0]; dev = self.device
        target_scale = scale_on(dev)

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
        return torch.tanh(mu.reshape(-1, 3)) * scale_on(obs_flat.device)
