import pytest
import torch

from sb.envs.base import Caps, Descriptor
from sb.envs.e3_multimodal import E3Multimodal


@pytest.mark.parametrize("k", [2, 3, 4])
def test_modes_and_any_mode_success(cpu, k):
    env = E3Multimodal(8, cpu)
    env.reset(0, Descriptor(dict(n_modes=k, separation=0.2, disturbance="none")))
    tk = env.tk
    assert len(tk.layout.gaps) == k and len(tk.modes3) == k
    # a robot sitting on any goal mode is inside the goal set; one between modes is not
    g = torch.stack(tk.modes3)
    assert bool((tk.marginal_md(g, 3) < 0.5).all())
    mid = (tk.modes3[0] + tk.modes3[1]) / 2
    assert float(tk.marginal_md(mid[None], 3)) > 1.0
    # the mixture sampler draws from every mode
    s = tk.sample(3, 400)
    d = torch.stack([(s[:, 1] - m[1]).abs() for m in tk.modes3], 1).argmin(1)
    assert len(set(d.tolist())) == k


def test_choose_mode_is_per_robot(cpu):
    env = E3Multimodal(16, cpu); env.reset(1, Descriptor(dict(n_modes=3, disturbance="slip")))
    g = env.tk.sample(1, 16)
    m = env.tk.choose_mode(g)
    assert m.shape == (16,) and int(m.max()) <= 2


@pytest.mark.slow
@pytest.mark.parametrize("k", [2, 4])
def test_e3_oracle_succeeds_undisturbed(cpu, k):
    env = E3Multimodal(64, cpu)
    env.reset(0, Descriptor(dict(n_modes=k, separation=0.2, disturbance="none")))
    out = env.evaluate(env.oracle(Caps()))
    assert float(out["success"].float().mean()) >= 0.95
