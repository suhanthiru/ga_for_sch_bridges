"""The graphed diffusion trainer: same interface as the eager loop, reproducible from the
seed, and a usable policy at the end. GPU only; the eager path is exercised everywhere."""
import pytest
import torch

from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask
from sb.policies.diffusion import train_diffusion

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@cuda
def test_graphed_training_is_deterministic_and_trains(small_demos):
    dev = torch.device("cuda")
    tk = GenTask(TR.Layout("L1"), "none", 8, 1, dev)
    G, U = (x.to(dev) for x in small_demos)
    losses = []
    pols = [train_diffusion(tk, G, U, 3, dev, 60, log=lambda m: losses.append(m), graphed=True) for _ in range(2)]
    p0, p1 = (dict(p.named_parameters()) for p in pols)
    assert all(torch.equal(p0[k], p1[k]) for k in p0)
    assert losses[0] == losses[1] and float(losses[0].split()[-1]) < 1.0
    eager = train_diffusion(tk, G, U, 3, dev, 60, graphed=False)
    pe = dict(eager.named_parameters())
    assert all(torch.isfinite(pe[k]).all() and torch.isfinite(p0[k]).all() for k in pe)
    a = pols[0].sample(torch.zeros(4, pols[0].obs_dim, device=dev), gen=torch.Generator(device=dev).manual_seed(0))
    assert a.shape == (4, 8, 3) and torch.isfinite(a).all()
