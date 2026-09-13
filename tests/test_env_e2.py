import pytest
import torch

from sb.envs.base import Caps, Descriptor
from sb.envs.e2_partial import E2Partial, E7Shifted, E8Sparse


def test_partial_obs_is_local_and_map_free(cpu):
    env = E2Partial(8, cpu)
    obs = env.reset(0, Descriptor(dict(layout="L1", disturbance="slip", n_voronoi=40)))
    assert obs.shape == (8, env.obs_dim()) and env.obs_dim() == 4 + 3 + 1 + 100
    # moving a robot far away changes its patch; two robots at the same pose see the same patch
    env.g[1] = env.g[0]
    o = env.obs()
    assert torch.allclose(o[0, 8:], o[1, 8:])
    env.g[1, 0] = (env.g[0, 0] + 0.5) % 1.0
    assert not torch.allclose(env.obs()[0, 8:], env.obs()[1, 8:])


def test_variants_carry_their_axes(cpu):
    e7 = E7Shifted(4, cpu); e7.reset(0)
    assert e7.d["slip_scale_gen"] == 0.35 and e7.d["slip_scale"] == 1.4 and "slip_scale_gen" in e7.axes
    e8 = E8Sparse(4, cpu); e8.reset(0, Descriptor(dict(n_demo=5)))
    assert e8.d["n_demo"] == 5


@pytest.mark.slow
def test_e2_oracle_succeeds_undisturbed(cpu):
    env = E2Partial(64, cpu)
    env.reset(0, Descriptor(dict(layout="L2", disturbance="none")))
    out = env.evaluate(env.oracle(Caps()))
    assert float(out["success"].float().mean()) >= 0.95
