"""Batched discrete Schrodinger bridges on a G x G grid, solved by Sinkhorn scaling.

The reference is a separable heat kernel K = k(x) (x) k(y) with bandwidth eps (a
Brownian reference over the horizon), so the two matrix products per iteration are
G x G matmuls and thousands of bridges fit in memory at once. Optional wall masking
turns the kernel into k sub-steps of a masked 3x3 stencil (the reference cannot cross
the wall), at the price of a dense per-layout kernel.

solve(mu0, mu1, eps, iters) -> (u, v) scalings with pi = diag(u) K diag(v) the bridge
coupling; marginals(u, v) reproduces mu0, mu1 to the convergence tolerance. Everything
runs in fp32; bf16 loses the marginals at the 1e-3 level (bench/gate_precision.py).
"""
import math

import torch


def heat_kernel_1d(G, eps, device=None, dtype=torch.float32):
    x = (torch.arange(G, dtype=dtype, device=device) + 0.5) / G
    d2 = (x[:, None] - x[None]) ** 2
    return torch.exp(-d2 / (2 * eps))


def apply_kernel(k, v):
    """(K2d v) for a separable kernel: v (P, G, G) -> k v k^T per batch element."""
    return k @ v @ k.T


def solve(mu0, mu1, eps, iters=200, k=None, tol=None):
    """mu0, mu1: (P, G, G) nonnegative, each summing to one per batch element.
    Returns scalings u, v (P, G, G) and the final marginal error."""
    P, G, _ = mu0.shape
    k = heat_kernel_1d(G, eps, mu0.device, mu0.dtype) if k is None else k
    u = torch.ones_like(mu0); v = torch.ones_like(mu1)
    err = torch.full((P,), float("inf"), device=mu0.device)
    for i in range(iters):
        u = mu0 / apply_kernel(k, v).clamp_min(1e-30)
        v = mu1 / apply_kernel(k, u).clamp_min(1e-30)
        if tol is not None and (i % 10 == 9):
            err = (u * apply_kernel(k, v) - mu0).abs().sum((1, 2))
            if bool((err < tol).all()):
                break
    m0 = u * apply_kernel(k, v)
    err = (m0 - mu0).abs().sum((1, 2))
    return u, v, k, err


def marginals(u, v, k):
    return u * apply_kernel(k, v), v * apply_kernel(k, u)


def coupling(u, v, k):
    """Dense coupling (P, G*G, G*G); only for small G or single bridges."""
    P, G, _ = u.shape
    K2 = torch.kron(k, k)                                   # (G*G, G*G), row-major (y, x) flattening
    return u.reshape(P, -1, 1) * K2[None] * v.reshape(P, 1, -1)


def gaussian_marginal(G, mean, std, device=None):
    """A discretised isotropic Gaussian on the grid, normalised. mean, std in [0, 1] units."""
    x = (torch.arange(G, dtype=torch.float32, device=device) + 0.5) / G
    gx = torch.exp(-0.5 * ((x - mean[0]) / std) ** 2); gy = torch.exp(-0.5 * ((x - mean[1]) / std) ** 2)
    m = gy[:, None] * gx[None]
    return m / m.sum()


def drift_field(u, v, k, eps, tau=0.5):
    """Forward drift of the bridge at time tau on the grid, from the potentials: the
    conditional mean displacement under the coupling divided by the remaining time.
    Returns (P, G, G, 2) in grid units."""
    P, G, _ = u.shape
    device = u.device
    x = (torch.arange(G, dtype=torch.float32, device=device) + 0.5) / G
    # K_tau-weighted expectation of the terminal position given the current cell
    k1 = heat_kernel_1d(G, eps * (1 - tau), device)
    vk = apply_kernel(k1, v)                                 # normaliser (P, G, G)
    ex = apply_kernel(k1, v * x[None, None, :]) / vk.clamp_min(1e-30)
    ey = apply_kernel(k1, v * x[None, :, None]) / vk.clamp_min(1e-30)
    dx = (ex - x[None, None, :]) / max(1 - tau, 1e-6)
    dy = (ey - x[None, :, None]) / max(1 - tau, 1e-6)
    return torch.stack([dx, dy], -1)
