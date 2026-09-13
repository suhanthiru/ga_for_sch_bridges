"""Bridge matching for a population of drift nets in one call.

The parameter-free part of a DSBM step (sampling the reference bridge and its drift
target, building the input features) is computed flat over population x batch; only the
MLP forward/backward is vmapped over stacked parameters with torch.func, and Adam runs
on the stacked tensors with a per-net learning rate. One population step launches a
few hundred kernels regardless of P, against ~500 per single-net step in solver.fit.
"""
import copy
import math

import torch
from torch.func import functional_call, grad, stack_module_state, vmap

from sb.core import bridge_fast as BF
from sb.core import solver as SV
from sb.envs import terrain as TR


def features(mf, g, tau, goal, fields):
    return torch.cat([mf.feats(g, goal.expand_as(g), tau), TR.terrain_feats(fields, g, mf.name != "flat"), SV.tau_emb(tau)], 1)


class Population:
    """P DriftNets with stacked parameters."""

    def __init__(self, nets):
        self.P = len(nets)
        self.base = copy.deepcopy(nets[0].net).to("meta")
        self.params, self.buffers = stack_module_state([n.net for n in nets])
        self.mf = nets[0].mf

    def forward(self, params, h):
        """h: (P, B, d_in) -> (P, B, 3)."""
        return vmap(lambda p, x: functional_call(self.base, (p, self.buffers), (x,)))(params, h)

    def state_dicts(self):
        return [{f"net.{k}": v[p].detach().clone() for k, v in self.params.items()} for p in range(self.P)]


def _adam_step(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    for k in params:
        g = grads[k]
        m[k].mul_(b1).add_(g, alpha=1 - b1)
        v[k].mul_(b2).addcmul_(g, g, value=1 - b2)
        mhat = m[k] / (1 - b1 ** t); vhat = v[k] / (1 - b2 ** t)
        step = lr.view(-1, *([1] * (g.dim() - 1))) * mhat / (vhat.sqrt() + eps)
        params[k] = params[k] - step


def fit_population(nets, pairs, ref, mf, goal, fields, steps, batch=512, lr=1e-3, gen=None, autocast=False, backward=False):
    """nets: list of P DriftNet; pairs: list of P (x0, x1) endpoint sets (n, 3) each.
    Returns per-net state dicts and the last population loss (P,)."""
    P = len(nets); device = pairs[0][0].device
    pop = Population(nets)
    params = {k: v.detach().clone().requires_grad_(False) for k, v in pop.params.items()}
    m = {k: torch.zeros_like(v) for k, v in params.items()}; vv = {k: torch.zeros_like(v) for k, v in params.items()}
    lr_t = torch.as_tensor(lr, dtype=torch.float32, device=device).expand(P).clone() if not torch.is_tensor(lr) else lr.to(device)
    a_list, b_list = zip(*[((x1, x0) if backward else (x0, x1)) for x0, x1 in pairs])
    n = a_list[0].shape[0]
    preps = [BF.prepare(ref, a, b, mf) for a, b in zip(a_list, b_list)]
    mode = preps[0]["mode"]
    a_all = torch.cat(a_list); vnom = torch.cat([p["vnom"] for p in preps]); Q10i = torch.cat([p["Q10i"] for p in preps])
    offs = (torch.arange(P, device=device) * n)[:, None]

    def loss_fn(p, h, t):
        out = functional_call(pop.base, (p, pop.buffers), (h,))
        return ((out - t) ** 2).sum(1).mean()

    vgrad = vmap(grad(loss_fn))
    vloss = vmap(loss_fn)
    last = None
    for it in range(1, steps + 1):
        i = (torch.randint(n, (P, batch), generator=gen, device=device) + offs).reshape(-1)
        s = torch.rand(P * batch, generator=gen, device=device).clamp(1e-3, 1 - 1e-3)
        p = dict(mode=mode, vnom=vnom[i], Q10i=Q10i[i])
        g, target = BF.sample_and_target(ref, a_all[i], p, s, mf, gen)
        tau = (1 - s) if backward else s
        h = features(mf, g, tau, goal, fields).reshape(P, batch, -1); t = target.reshape(P, batch, 3)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
            grads = vgrad(params, h, t)
        grads = {k: v.float() for k, v in grads.items()}
        _adam_step(params, grads, m, vv, it, lr_t)
        if it == steps:
            with torch.no_grad():
                last = vloss(params, h, t)
    pop.params = params
    return pop.state_dicts(), last
