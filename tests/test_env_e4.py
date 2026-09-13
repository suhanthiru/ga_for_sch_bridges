import math

import pytest
import torch

from sb.envs.base import Caps, Descriptor
from sb.envs.e4_contact import R, RB, E4Contact


def test_box_moves_only_when_pushed_inside_the_cone(cpu):
    env = E4Contact(4, cpu); env.reset(0, Descriptor(dict(mu=0.5, mass=1.0, disturbance="none")))
    tk = env.tk
    b = torch.tensor([[0.3, 0.5]] * 4)
    r_old = torch.tensor([[0.3 - (R + RB) - 0.001, 0.5, 0.0]] * 4)
    # 0: straight push; 1: same push but not touching; 2: push at 60 deg (outside cone mu=0.5); 3: pulling away
    r_new = r_old.clone()
    r_new[0, 0] += 0.012
    r_new[1, 0] += 0.004; r_old[1, 0] -= 0.02; r_new[1, 0] -= 0.02
    r_new[2, 0] += 0.004; r_new[2, 1] += 0.004 * math.tan(math.radians(60))
    r_new[3, 0] -= 0.004
    r_adj, b_new, blocked = tk.box_step(r_old, r_new, b)
    assert b_new[0, 0] > b[0, 0] and abs(float(b_new[0, 1] - 0.5)) < 1e-6
    assert torch.allclose(b_new[1], b[1]) and torch.allclose(b_new[3], b[3])
    assert b_new[2, 0] > b[2, 0] and abs(float(b_new[2, 1] - 0.5)) < 1e-6           # tangential part slid off
    assert not blocked.any()
    assert float(r_adj[0, 0]) < float(r_new[0, 0])                                   # the pusher is slowed by the box
    obs = env.obs(); assert obs.shape == (4, env.obs_dim())


def test_heavier_box_moves_less(cpu):
    out = []
    for mass in (1.0, 4.0):
        env = E4Contact(1, cpu); env.reset(0, Descriptor(dict(mu=0.5, mass=mass, disturbance="none")))
        b = torch.tensor([[0.3, 0.5]]); r_old = torch.tensor([[0.3 - (R + RB) - 0.001, 0.5, 0.0]]); r_new = r_old.clone(); r_new[0, 0] += 0.009
        out.append(float(env.tk.box_step(r_old, r_new, b)[1][0, 0] - 0.3))
    assert out[0] > out[1] > 0


@pytest.mark.slow
@pytest.mark.parametrize("mu,mass", [(0.3, 1.0), (0.9, 4.0)])
def test_e4_oracle_succeeds_undisturbed(cpu, mu, mass):
    env = E4Contact(64, cpu)
    env.reset(0, Descriptor(dict(mu=mu, mass=mass, disturbance="none")))
    out = env.evaluate(env.oracle(Caps()))
    assert float(out["success"].float().mean()) >= 0.95
