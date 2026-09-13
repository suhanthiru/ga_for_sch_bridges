"""The shared noise stream and the reference kinematics every generator rolls out on.

g <- g . exp(u dt + sqrt(dt) Sigma(x)^1/2 z): no friction, no base noise, Sigma(x) the
slip-reference covariance (or an isotropic match), z a stream of standard-normal draws
fixed by the seed. Two generators with the same seed see the same z at every step, so
only the controller differs between them.
"""
import math

import torch

from sb.core import bridge_fast as BF
from sb.core.sde import psd_sqrt
from sb.envs import task as TK

DT, T_SKILL, N_SKILL = TK.DT, TK.T_SKILL, TK.N_SKILL


class NoiseStream:
    """Standard-normal draws z[t] for n episodes, fixed by the seed."""

    def __init__(self, n, seed, device, T=N_SKILL * T_SKILL):
        g = torch.Generator(device=device).manual_seed(90_000 + seed)
        self.z = torch.randn(n, T, 3, generator=g, device=device)

    def __call__(self, t):
        return self.z[:, t]


def noise_step(ref, mf, g, z, iso_var=None):
    """sqrt(dt) Sigma(g)^1/2 z in the manifold's tangent coordinates."""
    if iso_var is not None:
        return math.sqrt(DT) * math.sqrt(iso_var) * z
    mode = BF.mode_of(ref, mf)
    Sg = BF.cov_at(ref, g, mf, mode)
    if mode == "scalar":
        return math.sqrt(DT) * Sg.sqrt()[:, None] * z
    if mode == "diag":
        return math.sqrt(DT) * Sg.sqrt() * z
    return math.sqrt(DT) * torch.einsum("nij,nj->ni", psd_sqrt(Sg), z)


def step_kin(mf, g, u, xi_noise):
    """Reference kinematics on the manifold: clipped body command plus tangent noise."""
    u = TK.clip_u(u)
    if mf.name == "flat":
        F = mf.frame(g)
        xi = torch.einsum("nij,nj->ni", F, u) * DT + xi_noise
        return mf.retract(g, xi), u
    return mf.retract(g, u * DT + xi_noise), u


def iso_variance(ref, mf, G):
    """Mean total variance tr(Sigma(x))/3 over the states of a rollout set."""
    mode = BF.mode_of(ref, mf)
    x = G.reshape(-1, 3)
    Sg = BF.cov_at(ref, x, mf, mode)
    tr = 3 * Sg if mode == "scalar" else (Sg.sum(1) if mode == "diag" else torch.diagonal(Sg, dim1=1, dim2=2).sum(1))
    return float(tr.mean() / 3)
