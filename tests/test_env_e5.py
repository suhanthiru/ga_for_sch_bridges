import pytest
import torch

from sb.envs.base import Caps, Descriptor
from sb.envs.e5_multiagent import DIAM, E5Cross


def test_contact_and_corridor_fail_both(cpu):
    env = E5Cross(4, cpu); env.reset(0, Descriptor(dict(ratio=2.5, disturbance="none")))
    tk = env.tk
    g = tk.sample(0, 8)
    g[0, :2] = torch.tensor([0.5, 0.5]); g[4, :2] = torch.tensor([0.5 + 0.5 * DIAM, 0.5])       # pair 0 touching
    c = tk.contact(g)
    assert bool(c[0]) and bool(c[4]) and not bool(c[1])
    xy = torch.tensor([[0.3, 0.5 + 0.6 * tk.width], [0.5 + 0.6 * tk.width, 0.3]])              # A off its corridor, B off its corridor
    full = torch.zeros(8, 2); full[1] = xy[0]; full[5] = xy[1]; full[2] = torch.tensor([0.3, 0.5]); full[6] = torch.tensor([0.5, 0.3])
    ex = tk.corridor_exit(full)
    assert bool(ex[1]) and bool(ex[5]) and not bool(ex[2]) and not bool(ex[6])
    obs = env.obs()
    assert obs.shape == (8, env.obs_dim())


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [1.5, 4.0])
def test_e5_oracle_succeeds_undisturbed(cpu, ratio):
    env = E5Cross(64, cpu)
    env.reset(0, Descriptor(dict(ratio=ratio, disturbance="none")))
    out = env.evaluate(env.oracle(Caps()))
    assert float(out["pair_success"].float().mean()) >= 0.95
