import pytest
import torch

from sb.envs.base import Caps, Descriptor
from sb.envs.e1_terrain import E1Terrain


def test_interface_shapes_and_autoreset(cpu):
    env = E1Terrain(16, cpu)
    obs = env.reset(0, Descriptor(dict(layout="L2", disturbance="push")))
    assert obs.shape == (16, env.obs_dim())
    for _ in range(5):
        obs, r, done, info = env.step(torch.randn(16, 3))
    assert obs.shape == (16, 24) and r.shape == (16,) and done.dtype == torch.bool and "success" in info
    G, U = env.demos(4, Caps())
    assert G.shape == (4, 301, 3) and U.shape == (4, 300, 3)
    geo = env.invariant_geometry()
    assert geo["pile"] is not None and len(geo["gaps"]) == 2


def test_oracle_needs_caps(cpu):
    env = E1Terrain(4, cpu); env.reset(0)
    with pytest.raises(AssertionError):
        env.oracle(Caps(oracle=False))


@pytest.mark.slow
@pytest.mark.parametrize("layout", ["L1", "L2"])
def test_oracle_succeeds_undisturbed(cpu, layout):
    env = E1Terrain(64, cpu)
    env.reset(0, Descriptor(dict(layout=layout, disturbance="none")))
    out = env.evaluate(env.oracle(Caps()))
    assert float(out["success"].float().mean()) >= 0.95
