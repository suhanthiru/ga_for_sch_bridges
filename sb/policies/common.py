"""Observation features and constants shared by every downstream policy class.

obs_of: x, y, cos, sin, one-hot skill, tau, 16 world-frame terrain probes -> (n, 24).
obs_gc: obs_of plus the tangent offset to the next marginal mean -> (n, 27); used by the
        goal-conditioned diffusion generator only.
"""
import math

import torch

from sb.core import se2 as S
from sb.envs import terrain as TR
from sb.envs import task as TK

CHUNK, EXEC = 8, 4
OBS_DIM = 4 + 3 + 1 + TR.N_TFEAT
OBS_GC_DIM = OBS_DIM + 3
SCALE = torch.tensor([TK.UMAX, TK.UMAX, TK.WMAX])


def sinusoidal(v, n_freq=16):
    """v: (n,) -> (n, 2*n_freq)."""
    f = 2.0 ** torch.arange(n_freq, device=v.device, dtype=v.dtype) * math.pi
    a = v.unsqueeze(1) * f
    return torch.cat([a.sin(), a.cos()], 1)


def _k_tensor(k, n, device):
    if not torch.is_tensor(k):
        k = torch.full((n,), int(k), dtype=torch.long, device=device)
    return k


def obs_of(tk, g, k, tau):
    k = _k_tensor(k, g.shape[0], g.device)
    k1h = torch.nn.functional.one_hot(k.clamp(max=2), 3).float()
    return torch.cat([g[:, :2], g[:, 2:3].cos(), g[:, 2:3].sin(), k1h, tau.unsqueeze(1),
                      TR.terrain_feats(tk.obs_fields, g, False)], 1)


def obs_gc(tk, g, k, tau):
    k = _k_tensor(k, g.shape[0], g.device)
    means = torch.stack(tk.means[1:4])                       # (3, 3) next-marginal means
    goal = means[k.clamp(max=2)]
    return torch.cat([obs_of(tk, g, k, tau), S.between(g, goal)], 1)
