"""E2: the terrain map is hidden. The observation carries pose, skill clock and a local
5x5 patch of the property fields sampled around the robot (a short-range terrain sensor)
instead of the 16 world-frame probes over the known map. Descriptor axis: map correlation
length through the Voronoi seed count.

E7 and E8 are E1 with different defaults on axes the data slot reads (generator slip
scale, demo count); they exist so the descriptor bins have an environment id to point at.
"""
import torch

from sb.envs import terrain as TR
from sb.envs.e1_terrain import DEFAULTS, E1Terrain

PATCH = 5
PATCH_R = 0.06
OBS_PARTIAL_DIM = 4 + 3 + 1 + 4 * PATCH * PATCH
_OFFSETS = {}


def _offsets(device, patch=PATCH, radius=PATCH_R):
    key = (str(device), patch, radius)
    if key not in _OFFSETS:
        o = torch.linspace(-radius, radius, patch, device=device)
        _OFFSETS[key] = torch.stack(torch.meshgrid(o, o, indexing="ij"), -1).reshape(-1, 2)
    return _OFFSETS[key]


def obs_partial(tk, g, k, tau, patch=PATCH, radius=PATCH_R):
    """pose, one-hot skill, tau, and a patch x patch sample of slip stds and friction from the
    true fields around the robot: what a policy sees when the map is hidden. (n, 108)."""
    if not torch.is_tensor(k):
        k = torch.full((g.shape[0],), int(k), dtype=torch.long, device=g.device)
    n = g.shape[0]
    k1h = torch.nn.functional.one_hot(k.clamp(max=2), 3).float()
    pts = (g[:, None, :2] + _offsets(g.device, patch, radius)[None]).reshape(-1, 2)
    pr = TR.lookup(tk.true_fields, pts)[:, :4].reshape(n, -1)       # the local sensor reads the terrain as it is
    return torch.cat([g[:, :2], g[:, 2:3].cos(), g[:, 2:3].sin(), k1h, tau.unsqueeze(1), pr], 1)


class E2Partial(E1Terrain):
    id = "E2"
    axes = E1Terrain.axes + ("n_voronoi",)

    def __init__(self, n, device, patch=PATCH, radius=PATCH_R):
        super().__init__(n, device)
        self.patch, self.radius = patch, radius
        o = torch.linspace(-radius, radius, patch, device=device)
        self.offsets = torch.stack(torch.meshgrid(o, o, indexing="ij"), -1).reshape(-1, 2)      # (patch*patch, 2)

    def obs_dim(self):
        return 4 + 3 + 1 + 4 * self.patch * self.patch

    def obs(self):
        return obs_partial(self.tk, self.g, self.k, self.t.float() / self.T, self.patch, self.radius)


class E7Shifted(E1Terrain):
    id = "E7"
    axes = E1Terrain.axes + ("slip_scale_gen",)
    defaults = dict(DEFAULTS, slip_scale=1.4, slip_scale_gen=0.35)

    def reset(self, seed, descriptor=None):
        d = dict(self.defaults); d.update(descriptor.values if hasattr(descriptor, "values") else (descriptor or {}))
        return super().reset(seed, d)


class E8Sparse(E1Terrain):
    id = "E8"
    axes = E1Terrain.axes + ("n_demo",)
    defaults = dict(DEFAULTS, n_demo=2)

    def reset(self, seed, descriptor=None):
        d = dict(self.defaults); d.update(descriptor.values if hasattr(descriptor, "values") else (descriptor or {}))
        return super().reset(seed, d)
