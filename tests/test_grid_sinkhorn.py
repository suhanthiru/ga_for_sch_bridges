import math

import torch

from sb.core import grid_sinkhorn as GS


def test_scalings_reproduce_marginals():
    G = 32
    mu0 = torch.stack([GS.gaussian_marginal(G, (0.2, 0.5), 0.05), GS.gaussian_marginal(G, (0.3, 0.3), 0.08)])
    mu1 = torch.stack([GS.gaussian_marginal(G, (0.8, 0.5), 0.05), GS.gaussian_marginal(G, (0.7, 0.7), 0.04)])
    u, v, k, err = GS.solve(mu0, mu1, eps=0.01, iters=300)
    m0, m1 = GS.marginals(u, v, k)
    assert float((m0 - mu0).abs().sum((1, 2)).max()) < 1e-4
    assert float((m1 - mu1).abs().sum((1, 2)).max()) < 1e-4
    assert err.shape == (2,)


def test_batched_equals_per_item():
    G = 16
    mu0 = torch.stack([GS.gaussian_marginal(G, (0.2, 0.5), 0.06), GS.gaussian_marginal(G, (0.5, 0.2), 0.06)])
    mu1 = torch.stack([GS.gaussian_marginal(G, (0.8, 0.5), 0.06), GS.gaussian_marginal(G, (0.5, 0.8), 0.06)])
    u, v, k, _ = GS.solve(mu0, mu1, 0.02, iters=100)
    for i in range(2):
        ui, vi, _, _ = GS.solve(mu0[i:i + 1], mu1[i:i + 1], 0.02, iters=100)
        assert torch.allclose(u[i], ui[0], rtol=1e-5, atol=1e-8) and torch.allclose(v[i], vi[0], rtol=1e-5, atol=1e-8)


def test_coupling_matches_scalings_and_moves_mass_right():
    G = 8
    mu0 = GS.gaussian_marginal(G, (0.25, 0.5), 0.08)[None]; mu1 = GS.gaussian_marginal(G, (0.75, 0.5), 0.08)[None]
    u, v, k, _ = GS.solve(mu0, mu1, 0.05, iters=300)
    pi = GS.coupling(u, v, k)[0]
    assert torch.allclose(pi.sum(1).reshape(G, G), mu0[0], atol=1e-5)
    assert torch.allclose(pi.sum(0).reshape(G, G), mu1[0], atol=1e-5)
    x = (torch.arange(G).float() + 0.5) / G
    xs = x[None, :].expand(G, G).reshape(-1)                                   # x coordinate of each flat cell
    mean_shift = (pi * (xs[None, :] - xs[:, None])).sum()
    assert mean_shift > 0.4


def test_drift_field_points_toward_target():
    G = 32
    mu0 = GS.gaussian_marginal(G, (0.2, 0.5), 0.05)[None]; mu1 = GS.gaussian_marginal(G, (0.8, 0.5), 0.05)[None]
    u, v, k, _ = GS.solve(mu0, mu1, 0.01, iters=200)
    b = GS.drift_field(u, v, k, 0.01, tau=0.5)[0]
    mid = b[G // 2, G // 2]
    assert mid[0] > 0 and abs(mid[1]) < 0.05 * mid[0]
    assert math.isfinite(float(b.abs().max()))
