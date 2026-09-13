"""Per-cell features for the "why" model (SEARCH_PLAN / program section 7.1): the
regressors that are supposed to predict where a bridge helps.

  w2_demo_to_target      W2 (SE(2) metric) from the demos' endpoint cloud to the target marginal
  sigma_condition        condition number of the terrain slip covariance along the mean route
  demo_multimodality     number of GMM components the BIC prefers for the demo endpoints (1 = unimodal)
  pd_reachability        success of the nominal PD from the demos' handoff states under the cell's disturbance
  shell_fraction         fraction of the target's 2-sigma set farther than `thr` from every demo state
  kl_demo_reference      KL between a Gaussian fit of the demo endpoints and one of the reference
                         bridge's endpoints (Brownian-bridge samples from the same starts)

Everything is a function of (task, demos); nothing reads a search result.
"""
import math

import numpy as np
import torch

from sb.core import se2 as S
from sb.core.sde import Reference
from sb.envs import terrain as TR
from sb.policies.nominal import Nominal


def w2_demo_to_target(tk, G, n=400):
    ends = G[:, -1]
    ref = S.sample_marginal(tk.means[-1], tk.covs[-1], n, tk.device, torch.Generator(device=tk.device).manual_seed(0))
    return float(S.w2_se2(ends[:n].cpu(), ref.cpu()))


def sigma_condition(tk, n_pt=32):
    pts = torch.cat([tk.means[k][None, :2] * (1 - s) + tk.means[k + 1][None, :2] * s
                     for k in range(len(tk.means) - 1) for s in torch.linspace(0, 1, n_pt, device=tk.device)[:, None]])
    p = TR.lookup(tk.obs_fields, pts)[:, :3] * tk.slip_scale
    var = p ** 2 + 1e-8
    return float((var.max(1).values / var.min(1).values).mean())


def demo_multimodality(G, max_k=4, seed=0):
    from sklearn.mixture import GaussianMixture
    X = G[:, -1, :2].cpu().numpy()
    if len(X) < 8:
        return 1
    best, best_bic = 1, np.inf
    for k in range(1, min(max_k, len(X) // 4) + 1):
        bic = GaussianMixture(k, random_state=seed).fit(X).bic(X)
        if bic < best_bic - 1e-9:
            best, best_bic = k, bic
    return int(best)


@torch.no_grad()
def pd_reachability(tk_cell, G, kp=6.0):
    """Start the nominal PD at the demos' first handoff states and score the remaining
    two skills under the cell's disturbance: can feedback alone take the demonstration
    manifold to the target?"""
    from sb.envs import task as TK
    n = G.shape[0]
    g = G[:, TK.T_SKILL].clone().to(tk_cell.device)
    ctl = Nominal(tk_cell, kp=kp); alive = torch.ones(n, dtype=torch.bool, device=g.device)
    for k in (1, 2):
        for t in range(TK.T_SKILL):
            tau = torch.full((n,), t / TK.T_SKILL, device=g.device)
            u, _ = ctl(g, k, tau, k * TK.T_SKILL + t)
            g_new, hit = tk_cell.dynamics(g, u, k * TK.T_SKILL + t)
            g = torch.where(alive[:, None], g_new, g); alive &= ~hit
    return float((alive & (tk_cell.marginal_md(g, len(tk_cell.means) - 1) <= 2.0)).float().mean())


def shell_fraction(tk, G, n=2000, thr=0.05):
    from scipy.spatial import cKDTree
    ref = S.sample_marginal(tk.means[-1], tk.covs[-1], n, tk.device, torch.Generator(device=tk.device).manual_seed(1)).cpu().numpy()
    D = G.reshape(-1, 3).cpu().numpy()
    emb = lambda a: np.stack([a[:, 0], a[:, 1], S.HEADING_W * np.cos(a[:, 2]), S.HEADING_W * np.sin(a[:, 2])], 1)
    d, _ = cKDTree(emb(D)).query(emb(ref))
    return float((d > thr).mean())


def _gauss_kl(X, Y):
    """KL(N(mu_x, S_x) || N(mu_y, S_y)) with diagonal loading."""
    mx, my = X.mean(0), Y.mean(0); d = X.shape[1]
    Sx = np.cov(X.T) + 1e-6 * np.eye(d); Sy = np.cov(Y.T) + 1e-6 * np.eye(d)
    iSy = np.linalg.inv(Sy)
    return float(0.5 * (np.trace(iSy @ Sx) + (my - mx) @ iSy @ (my - mx) - d + np.log(np.linalg.det(Sy) / np.linalg.det(Sx))))


def kl_demo_reference(tk, G, sigma=0.05, n=None):
    """Demo endpoints against the endpoints of the Brownian-bridge reference started from
    the same demo starts and aimed at the target marginal."""
    n = n or G.shape[0]
    ref = Reference("brownian", sigma=sigma)
    g0 = G[:n, 0].to(tk.device); g1 = S.sample_marginal(tk.means[-1], tk.covs[-1], n, tk.device, torch.Generator(device=tk.device).manual_seed(2))
    tau = torch.full((n,), 0.95, device=tk.device)
    near_end = ref.bridge_sample(g0, g1, tau, S.SE2, torch.Generator(device=tk.device).manual_seed(3))
    X = G[:n, -1, :2].cpu().numpy(); Y = near_end[:, :2].cpu().numpy()
    return _gauss_kl(X, Y)


def features(tk_none, tk_cell, G):
    """All features for one cell: `tk_none` is the undisturbed task (route, map), `tk_cell`
    the cell's disturbed task, G the demo trajectories."""
    return dict(w2_demo_to_target=w2_demo_to_target(tk_none, G), sigma_condition=sigma_condition(tk_none),
                demo_multimodality=demo_multimodality(G), pd_reachability=pd_reachability(tk_cell, G),
                shell_fraction=shell_fraction(tk_none, G), kl_demo_reference=kl_demo_reference(tk_none, G))
