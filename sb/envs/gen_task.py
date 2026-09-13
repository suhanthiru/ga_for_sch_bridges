"""The task with the generator suite's environment axes: slip anisotropy (lateral std x
aniso), terrain correlation length (Voronoi seed count), push-rate multiplier, marginal
heading std, and a class-flip rate for the map the generator is allowed to see.
"""
import torch

from sb.core.sde import Reference
from sb.envs import task as TK
from sb.envs import terrain as TR


class GenTask(TK.Task):
    def __init__(self, layout, disturbance, n, seed, device, slip_scale=0.7, aniso=1.0, n_seed=14,
                 push_mult=1.0, heading_std=None, map_flip=0.0, layout_id=0, route="lower", push_mag=0.25):
        self.layout, self.dist, self.n, self.device = layout, disturbance, n, device
        self.cmap = TR.class_map(layout_id, n_seed=n_seed)
        tf = TR.property_fields(self.cmap); rf = TR.property_fields(self.cmap, rain=True)
        for f in (tf, rf):
            f[1] *= aniso; f[4] *= push_mult
        self.true_fields, self.rain_fields = torch.tensor(tf, device=device), torch.tensor(rf, device=device)
        if map_flip > 0:
            of = TR.property_fields(TR.class_map(layout_id, n_seed=n_seed, flip_rate=map_flip)); of[1] *= aniso; of[4] *= push_mult
            self.obs_fields = torch.tensor(of, device=device)
        else:
            self.obs_fields = self.true_fields
        self.means, self.covs = TR.marginals(layout, route)
        self.means = [m.to(device) for m in self.means]; self.covs = [c.to(device).clone() for c in self.covs]
        if heading_std is not None:
            for c in self.covs:
                c[2, 2] = heading_std ** 2
        self.slip_scale = slip_scale
        self.push = TK.PushField(layout_id, push_mag, device) if disturbance == "push" else None
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.rain_t = int(torch.randint(60, 240, (1,), generator=self.gen, device=device)) if disturbance == "rain" else 10 ** 9


def make_ref(kind, tk, slip_scale=None):
    return Reference(kind, sigma=0.05, kappa=2.0, slip_scale=slip_scale if slip_scale is not None else tk.slip_scale,
                     fields=tk.obs_fields)


def to_cpu(tk, seed):
    """A CPU copy of a task (bridge nets train on CPU: the solver is launch-bound on the GPU)."""
    cpu = torch.device("cpu")
    tkc = type(tk).__new__(type(tk))
    tkc.__dict__.update({k: (v.to(cpu) if torch.is_tensor(v) else v) for k, v in tk.__dict__.items()})
    tkc.means = [m.to(cpu) for m in tk.means]; tkc.covs = [c.to(cpu) for c in tk.covs]; tkc.device = cpu
    tkc.gen = torch.Generator(device=cpu).manual_seed(1000 + seed)
    if hasattr(tkc, "body_std"):
        tkc.body_std = tk.body_std.to(cpu)
    return tkc
