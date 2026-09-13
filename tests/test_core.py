"""The gate on the core: linear-Gaussian covariance steering against the closed form,
SE(2) group operations, terrain lookup, W2, the reference bridge reducing to the Brownian
bridge, the fast path agreeing with the reference implementation, and equivariance.

The covariance-steering test runs IPF on propagated moments rather than samples, so its
tolerance measures the solver's logic and not Monte-Carlo or neural error. It always runs
in float64.
"""
import math

import numpy as np
import pytest
import torch

from sb.core import bridge_fast as BF
from sb.core import sde as SD
from sb.core import se2 as S
from sb.envs import terrain as TR


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0); np.random.seed(0)


def imf_step(S0, S1, C, eps, n_t=4000, t_end=1 - 1e-5):
    """One Iterative-Markovian-Fitting step on Gaussian moments: reciprocal process from
    the coupling C, Markovian projection, simulate, return the new coupling."""
    d = len(S0); I = np.eye(d)

    def A(t):
        St = (1 - t) ** 2 * S0 + t ** 2 * S1 + t * (1 - t) * (C + C.T) + eps * t * (1 - t) * I
        K = ((1 - t) * C.T + t * S1) @ np.linalg.inv(St)
        return (K - I) / (1 - t)

    J = S0.copy(); ts = np.linspace(0, t_end, n_t + 1)
    for i in range(n_t):
        t, h = ts[i], ts[i + 1] - ts[i]
        k1 = J @ A(t).T
        k2 = (J + h / 2 * k1) @ A(t + h / 2).T
        k3 = (J + h / 2 * k2) @ A(t + h / 2).T
        k4 = (J + h * k3) @ A(t + h).T
        J = J + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return J


S0 = np.array([[0.040, 0.012], [0.012, 0.015]])
S1 = np.array([[0.008, -0.004], [-0.004, 0.030]])
EPS = 0.05


def test_closed_form_is_a_valid_joint():
    C = SD.gaussian_sb_coupling(S0, S1, EPS)
    J = np.block([[S0, C], [C.T, S1]])
    assert np.linalg.eigvalsh(J).min() > -1e-10


def test_imf_fixed_point_at_closed_form():
    C = SD.gaussian_sb_coupling(S0, S1, EPS)
    assert np.abs(imf_step(S0, S1, C, EPS) - C).max() < 1e-3


def test_imf_converges_from_independent_coupling():
    Cstar = SD.gaussian_sb_coupling(S0, S1, EPS); C = np.zeros_like(S0)
    for _ in range(12):
        C = imf_step(S0, S1, C, EPS)
    assert np.abs(C - Cstar).max() < 1e-3
    for t in (0.1, 0.25, 0.5, 0.75, 0.9):
        St_ipf = (1 - t) ** 2 * S0 + t ** 2 * S1 + t * (1 - t) * (C + C.T) + EPS * t * (1 - t) * np.eye(2)
        assert np.abs(St_ipf - SD.gaussian_sb_marginal(S0, S1, Cstar, EPS, t)).max() < 1e-3


def test_eps_to_zero_recovers_bures():
    C0 = SD.gaussian_sb_coupling(S0, S1, 1e-9)
    R = SD._sqrtm(S0)
    bures = R @ SD._sqrtm(R @ S1 @ R) @ np.linalg.inv(R)
    assert np.abs(C0 - bures).max() < 1e-5


def test_se2_exp_log_and_group():
    n = 4000
    xi = torch.randn(n, 3) * torch.tensor([0.5, 0.5, 1.5])
    xi[:, 2] = S.wrap(xi[:, 2]).clamp(-math.pi + 1e-6, math.pi - 1e-6)
    xi[:1000, 2] *= 1e-5; xi[1000:2000, 2] = math.pi - 1e-4
    assert (S.log_se2(S.exp_se2(xi)) - xi).abs().max().item() < 1e-4
    g = S.exp_se2(torch.randn(n, 3)); h = S.exp_se2(torch.randn(n, 3))
    assert (S.compose(g, S.inv(g))).abs().max().item() < 1e-5
    assert (S.compose(g, S.exp_se2(S.between(g, h))) - h).abs().max().item() < 1e-5
    assert (S.SE2.interp(g, h, torch.zeros(n)) - g).abs().max().item() < 1e-5
    assert (S.SE2.interp(g, h, torch.ones(n)) - h).abs().max().item() < 1e-5
    a = S.exp_se2(torch.randn(1, 3)).expand(n, 3).contiguous()
    lhs = S.SE2.interp(S.compose(a, g), S.compose(a, h), torch.full((n,), 0.37))
    rhs = S.compose(a, S.SE2.interp(g, h, torch.full((n,), 0.37)))
    assert (lhs - rhs).abs().max().item() < 1e-5


def test_mahalanobis_across_the_pi_cut():
    m = torch.tensor([[0.5, 0.5, math.pi]])
    gs = m.repeat(2, 1); gs[0, 2] = math.pi - 0.05; gs[1, 2] = -math.pi + 0.05
    cov = torch.diag(torch.tensor([0.05, 0.05, 0.1])) ** 2
    assert float(S.mahalanobis(gs, m, cov).max()) < 1.5


def test_terrain_lookup_and_structure():
    cm = TR.class_map(0); F = torch.tensor(TR.property_fields(cm))
    g = (torch.arange(TR.GRID) + 0.5) / TR.GRID
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    assert (TR.lookup(F, pts) - F.reshape(5, -1).T).abs().max().item() < 1e-5
    assert (cm[:, :-1] == cm[:, 1:]).mean() > 0.9
    cov = TR.slip_cov_body(F, pts[:64])
    assert float(torch.linalg.eigvalsh(cov).min()) > 0
    assert float((cov[:, 1, 1] / cov[:, 0, 0]).max()) > 3
    rain = torch.tensor(TR.property_fields(cm, rain=True))
    assert abs((rain[3] / F[3]).min().item() - 0.6) < 1e-5


def test_w2_se2():
    n, s, d = 400, 0.05, 0.50
    gen = torch.Generator().manual_seed(0)
    a = torch.zeros(n, 3); a[:, :2] = torch.randn(n, 2, generator=gen) * s
    b = a.clone(); b[:, 0] += d
    assert abs(S.w2_se2(a, b) - d) < 1e-4
    b2 = torch.zeros(n, 3); b2[:, :2] = torch.randn(n, 2, generator=gen) * s; b2[:, 0] += d
    assert abs(S.w2_se2(a, b2) - d) < 0.05 * d
    assert S.w2_se2(a, a.clone()) < 1e-6
    c = a.clone(); c[:, 2] += 0.4
    assert abs(S.w2_se2(a, c) - S.HEADING_W * 0.4) < 1e-4


def test_reference_bridge_reduces_to_brownian_bridge():
    n = 20000; ref = SD.Reference("brownian", sigma=0.08); mf = S.Flat
    gen = torch.Generator().manual_seed(0)
    g0 = torch.randn(n, 3, generator=gen) * 0.1
    g1 = torch.randn(n, 3, generator=gen) * 0.1 + torch.tensor([0.6, 0.0, 0.0])
    for tau_v in (0.15, 0.5, 0.85):
        tau = torch.full((n,), tau_v)
        g = ref.bridge_sample(g0, g1, tau, mf, gen)
        emp = (g - mf.interp(g0, g1, tau)).var(0).mean().item()
        assert abs(emp / (ref.sigma ** 2 * tau_v * (1 - tau_v)) - 1) < 0.05
        b = ref.bridge_drift(g, g0, g1, tau, mf)
        assert (b - mf.logmap(g, g1) / (1 - tau_v)).abs().max().item() < 1e-3
    ou = SD.Reference("unicycle", sigma=0.08, kappa=4.0); tau = torch.full((n,), 0.5)
    dev_b = (ref.bridge_sample(g0, g1, tau, mf, gen) - mf.interp(g0, g1, tau)).var(0).mean().item()
    dev_o = (ou.bridge_sample(g0, g1, tau, mf, gen) - mf.interp(g0, g1, tau)).var(0).mean().item()
    assert dev_o < dev_b * 0.9


@pytest.mark.parametrize("kind,mf", [("brownian", S.Flat), ("unicycle", S.SE2), ("slip", S.SE2), ("slip", S.Flat)])
def test_fast_path_matches_reference(kind, mf):
    n = 4000
    fields = torch.tensor(TR.property_fields(TR.class_map(0)))
    ref = SD.Reference(kind, sigma=0.06, kappa=2.0, fields=fields)
    g0 = torch.randn(n, 3) * 0.05 + torch.tensor([0.30, 0.5, 0.0])
    g1 = torch.randn(n, 3) * 0.05 + torch.tensor([0.60, 0.5, 0.0])
    tau = torch.rand(n).clamp(0.05, 0.95)
    prep = BF.prepare(ref, g0, g1, mf)
    gA, tA = BF.sample_and_target(ref, g0, prep, tau, mf, torch.Generator().manual_seed(3))
    tB = ref.bridge_drift(gA, g0, g1, tau, mf)
    assert (tA - tB).abs().max().item() / tB.abs().max().item() < 2e-3
    vA = (mf.logmap(mf.interp(g0, g1, tau), gA) ** 2).mean().item()
    gC = ref.bridge_sample(g0, g1, tau, mf, torch.Generator().manual_seed(3))
    vC = (mf.logmap(mf.interp(g0, g1, tau), gC) ** 2).mean().item()
    assert abs(vA / vC - 1) < 0.08


@pytest.mark.parametrize("kind", ["brownian", "unicycle"])
def test_se2_bridge_equivariant_and_flat_not(kind):
    n = 3000
    a = S.exp_se2(torch.tensor([[0.3, -0.2, 1.1]])).expand(n, 3).contiguous()
    ref = SD.Reference(kind, sigma=0.06, kappa=2.0)
    g0 = torch.randn(n, 3) * 0.05 + torch.tensor([0.30, 0.5, 0.2])
    g1 = torch.randn(n, 3) * 0.05 + torch.tensor([0.60, 0.5, -0.1])
    tau = torch.rand(n).clamp(0.05, 0.95)
    for mf, want_equi in ((S.SE2, True), (S.Flat, False)):
        p = BF.prepare(ref, g0, g1, mf)
        gA, _ = BF.sample_and_target(ref, g0, p, tau, mf, torch.Generator().manual_seed(7))
        pg = BF.prepare(ref, S.compose(a, g0), S.compose(a, g1), mf)
        gB, _ = BF.sample_and_target(ref, S.compose(a, g0), pg, tau, mf, torch.Generator().manual_seed(7))
        e = S.between(S.compose(a, gA), gB).abs().max().item()
        assert (e < 1e-4) if want_equi else (e > 1e-2)
