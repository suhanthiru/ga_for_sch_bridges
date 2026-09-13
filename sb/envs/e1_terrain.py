"""E1: the SE(2) terrain task behind the family interface. Steps run on FastEnv with
tensor skill clocks and auto-reset; the oracle is the MPPI planner with the true map."""
import torch

from sb.core import se2 as S
from sb.envs import task as TK
from sb.envs import terrain as TR
from sb.envs.base import Caps, Descriptor, Env
from sb.envs.fast import FastEnv
from sb.envs.gen_task import GenTask
from sb.gen.controllers import MPCOracle
from sb.policies.common import OBS_DIM, obs_of
from sb.policies.nominal import Nominal

DEFAULTS = dict(layout="L1", disturbance="slip", slip_scale=0.7, aniso=1.0, n_voronoi=14, push_mult=1.0, heading_std=None)


class E1Terrain(Env):
    id = "E1"
    axes = ("layout", "disturbance", "slip_scale", "aniso", "n_voronoi", "push_mult", "heading_std")

    def __init__(self, n, device):
        self.n, self.device, self.T = n, device, TK.T_SKILL

    def _task(self, seed, d, n, disturbance=None):
        return GenTask(TR.Layout(d["layout"]), disturbance or d["disturbance"], n, seed, self.device, slip_scale=d["slip_scale"],
                       aniso=d["aniso"], n_seed=d["n_voronoi"], push_mult=d["push_mult"], heading_std=d["heading_std"])

    def reset(self, seed, descriptor=None):
        d = dict(DEFAULTS); d.update(descriptor.values if isinstance(descriptor, Descriptor) else (descriptor or {}))
        self.d, self.seed = d, seed
        self.tk = self._task(seed, d, self.n)
        self.fe = FastEnv.from_task(self.tk)
        self.g = self.tk.sample(0, self.n)
        self.k = torch.zeros(self.n, dtype=torch.long, device=self.device); self.t = torch.zeros_like(self.k)
        return self.obs()

    def obs(self):
        return obs_of(self.tk, self.g, self.k, self.t.float() / self.T)

    def obs_dim(self):
        return OBS_DIM

    def step(self, u):
        g_new, hit = self.fe.step_random(self.g, u, self.k * self.T + self.t)
        self.g = g_new; self.t = self.t + 1
        r = -0.01 * (TK.clip_u(u) ** 2).sum(1) - 0.001 - hit.float()
        ho = self.t >= self.T
        k = torch.where(ho, self.k + 1, self.k); t = torch.where(ho, torch.zeros_like(self.t), self.t)
        fin = ho & (k >= 3)
        success = fin & (S.mahalanobis(self.g, self.tk.means[3], self.tk.covs[3]) <= 2.0)
        r = r + success.float()
        done = hit | fin
        fresh = self.tk.sample(0, self.n)
        self.g = torch.where(done[:, None], fresh, self.g)
        self.k = torch.where(done, torch.zeros_like(k), k); self.t = torch.where(done, torch.zeros_like(t), t)
        return self.obs(), r, done, dict(success=success, collision=hit)

    def oracle(self, caps):
        assert isinstance(caps, Caps) and caps.oracle, "the oracle is train-time only"
        m = MPCOracle(self.tk, seed=self.seed)
        return lambda g, k, tau, step: (m.act(g, k, step % self.T), None)

    def demos(self, n, caps, kind="nominal"):
        """Demonstrations on the undisturbed task with clean labels (see policies.demos)."""
        assert isinstance(caps, Caps)
        tk = self._task(4242 + self.seed, self.d, n, disturbance="none")
        out = tk.rollout(Nominal(tk), n=n, record=True)
        G = out["traj"]; ctl = Nominal(tk); U = torch.zeros(n, 3 * self.T, 3, device=self.device)
        for s in range(3 * self.T):
            k = s // self.T; tau = torch.full((n,), (s % self.T) / self.T, device=self.device)
            U[:, s] = TK.clip_u(ctl(G[:, s], k, tau, s)[0])
        return G, U

    def invariant_geometry(self):
        lay = self.tk.layout
        return dict(wall_x=0.5, gaps=list(lay.gaps), pile=lay.pile, box=(0.0, 1.0), means=[m.cpu() for m in self.tk.means],
                    covs=[c.cpu() for c in self.tk.covs])

    @torch.no_grad()
    def evaluate(self, controller, n=None):
        """Full chained episodes with Task.rollout semantics (frozen on collision)."""
        return self.tk.rollout(controller, n=n or self.n)
