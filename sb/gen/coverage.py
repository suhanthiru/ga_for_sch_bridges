"""Off-path coverage of a generated trajectory set, and the noised-copies augmentation."""
import math

import numpy as np
import torch
from scipy.spatial import cKDTree

from sb.core import se2 as S


def noised_copies(G, U, m, seed, device):
    """m copies of random demos with Gaussian state jitter (0.03, 0.03, 0.1); actions kept."""
    g = torch.Generator(device=device).manual_seed(seed)
    idx = torch.randint(G.shape[0], (m,), generator=g, device=device)
    noise = torch.randn(m, G.shape[1], 3, generator=g, device=device) * torch.tensor([0.03, 0.03, 0.1], device=device)
    Gn = torch.cat([G[idx, :, :2] + noise[:, :, :2], S.wrap(G[idx, :, 2:3] + noise[:, :, 2:3])], 2)
    return Gn, U[idx]


def _embed(a):
    return np.stack([a[:, 0], a[:, 1], S.HEADING_W * np.cos(a[:, 2]), S.HEADING_W * np.sin(a[:, 2])], 1)


def coverage(G, demo_G, grid=32, nth=8, thr=0.05):
    """Distinct (x, y, heading-bin) cells visited by states farther than `thr` (SE(2)
    tangent metric) from any demo state; the off-path fraction; the mean displacement;
    and the occupancy histogram itself for the record."""
    X = G.reshape(-1, 3).cpu().numpy(); Dm = demo_G.reshape(-1, 3).cpu().numpy()
    d, _ = cKDTree(_embed(Dm)).query(_embed(X))
    off = X[d > thr]
    ix = (off[:, 0] * grid).astype(np.int64).clip(0, grid - 1)
    iy = (off[:, 1] * grid).astype(np.int64).clip(0, grid - 1)
    ih = (((off[:, 2] + math.pi) / (2 * math.pi)) * nth).astype(np.int64).clip(0, nth - 1)
    hist = np.zeros((grid, grid, nth), dtype=np.int64)
    np.add.at(hist, (ix, iy, ih), 1)
    return dict(cov_cells=int((hist > 0).sum()), off_frac=float((d > thr).mean()), mean_disp=float(d.mean())), hist
