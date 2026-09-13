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
        g, n = self.g, self.g.shape[0]
        k1h = torch.nn.functional.one_hot(self.k.clamp(max=2), 3).float()
        pts = (g[:, None, :2] + self.offsets[None]).reshape(-1, 2)
        fields = self.tk.fields_at(int(self.k[0]) * self.T + int(self.t.float().median()))
        p = TR.lookup(fields, pts)[:, :4].reshape(n, -1)                                 # 3 slip stds + friction
        return torch.cat([g[:, :2], g[:, 2:3].cos(), g[:, 2:3].sin(), k1h, (self.t.float() / self.T).unsqueeze(1), p], 1)


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
