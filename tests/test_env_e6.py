import pytest
import torch

from sb.envs.base import Caps, Descriptor
from sb.envs.e6_long import E6Long, zigzag


@pytest.mark.parametrize("n", [6, 8, 10])
def test_route_passes_the_gap_and_clocks_scale_with_slack(cpu, n):
    wp = zigzag(n)
    assert wp.shape == (n + 1, 3) and abs(float(wp[0, 0]) - 0.1) < 1e-6 and abs(float(wp[-1, 0]) - 0.9) < 1e-6
    j = int((wp[:, 0] - 0.5).abs().argmin()); assert float(wp[j, 1]) == 0.5
    e12 = E6Long(4, cpu); e12.reset(0, Descriptor(dict(n_skills=n, slack=1.2, disturbance="none")))
    e15 = E6Long(4, cpu); e15.reset(0, Descriptor(dict(n_skills=n, slack=1.5, disturbance="none")))
    assert len(e12.tk.t_skill) == n and e15.tk.n_steps > e12.tk.n_steps
    obs, r, done, info = e12.step(torch.zeros(4, 3))
    assert obs.shape == (4, e12.obs_dim())


@pytest.mark.slow
@pytest.mark.parametrize("n", [6, 10])
def test_e6_oracle_succeeds_undisturbed(cpu, n):
    env = E6Long(64, cpu)
    env.reset(0, Descriptor(dict(n_skills=n, slack=1.2, disturbance="none")))
    out = env.evaluate(env.oracle(Caps()))
    assert float(out["success"].float().mean()) >= 0.95
