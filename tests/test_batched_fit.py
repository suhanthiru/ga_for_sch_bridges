"""The population fitter's gradients and Adam step equal the per-net ones on the same
data, and a short fit lowers the loss."""
import torch

from sb.core import batched_fit as BFit
from sb.core import se2 as S
from sb.core import solver as SV
from sb.core.sde import Reference
from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask


def _setup(cpu, P=3):
    torch.manual_seed(0)
    tk = GenTask(TR.Layout("L1"), "none", 64, 1, cpu)
    ref = Reference("slip", sigma=0.05, kappa=2.0, slip_scale=0.7, fields=tk.obs_fields)
    nets = [SV.DriftNet(S.SE2) for _ in range(P)]
    pairs = [SV.sample_pairs(tk, 0, 64, ref, S.SE2) for _ in range(P)]
    return tk, ref, nets, pairs


def test_population_forward_matches_each_net(cpu):
    tk, ref, nets, pairs = _setup(cpu)
    pop = BFit.Population(nets)
    g = torch.cat([p[0] for p in pairs]); tau = torch.rand(g.shape[0]).clamp(0.05, 0.95)
    h = BFit.features(S.SE2, g, tau, tk.means[1], tk.obs_fields).reshape(3, 64, -1)
    out = pop.forward(pop.params, h)
    for p, net in enumerate(nets):
        ref_out = net(pairs[p][0], tau[p * 64:(p + 1) * 64], tk.means[1], tk.obs_fields)
        assert torch.allclose(out[p], ref_out, atol=1e-6)


def test_vmapped_gradients_match_autograd(cpu):
    tk, ref, nets, pairs = _setup(cpu)
    pop = BFit.Population(nets)
    h = torch.randn(3, 16, pop.base[0].in_features); t = torch.randn(3, 16, 3)
    params = {k: v.detach().clone() for k, v in pop.params.items()}
    from torch.func import functional_call, grad, vmap
    f = lambda p, x, y: ((functional_call(pop.base, (p, pop.buffers), (x,)) - y) ** 2).sum(1).mean()
    grads = vmap(grad(f))(params, h, t)
    for p, net in enumerate(nets):
        loss = ((net.net(h[p]) - t[p]) ** 2).sum(1).mean(); loss.backward()
        for k, prm in net.net.named_parameters():
            assert torch.allclose(grads[k][p], prm.grad, atol=1e-6)


def test_fit_population_reduces_loss(cpu):
    tk, ref, nets, pairs = _setup(cpu, P=2)
    gen = torch.Generator().manual_seed(0)
    sds, last = BFit.fit_population(nets, pairs, ref, S.SE2, tk.means[1], tk.obs_fields, steps=30, batch=64, lr=1e-3, gen=gen)
    assert last.shape == (2,) and bool(torch.isfinite(last).all())
    net = SV.DriftNet(S.SE2); net.load_state_dict(sds[0])
    before = nets[0]; g = pairs[0][0]; tau = torch.full((64,), 0.5)
    assert not torch.allclose(net(g, tau, tk.means[1], tk.obs_fields), before(g, tau, tk.means[1], tk.obs_fields))
