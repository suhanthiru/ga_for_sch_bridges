"""FastEnv.step reproduces Task.dynamics given the same draws, on both layouts, every
disturbance, before and after the rain onset; and a mixed-kind batch equals the
per-kind results."""
import pytest
import torch

from sb.core import se2 as S
from sb.envs import terrain as TR
from sb.envs.fast import KINDS, FastEnv, draws_like_task
from sb.envs.gen_task import GenTask


@pytest.mark.parametrize("layout", ["L1", "L2"])
@pytest.mark.parametrize("dist", ["none", "slip", "rain", "push"])
def test_fast_step_matches_task_dynamics(cpu, layout, dist):
    n = 512
    tk = GenTask(TR.Layout(layout), dist, n, 11, cpu)
    fe = FastEnv.from_task(tk)
    g = tk.sample(0, n); g[:, 0] = torch.rand(n, generator=tk.gen) * 0.9 + 0.05       # spread across the wall
    u = torch.randn(n, 3, generator=tk.gen) * 2
    for step in (10, 250):
        state = tk.gen.get_state()
        g1, h1 = tk.dynamics(g, u, step)
        tk.gen.set_state(state)
        z, r, kz = draws_like_task(tk, n, tk.gen)
        g2, h2 = fe.step(g, u, step, z, r, kz)
        assert torch.equal(h1, h2)
        assert (S.between(g1, g2)).abs().max().item() < 1e-5


def test_mixed_kind_batch_equals_per_kind(cpu):
    n = 256
    tasks = {d: GenTask(TR.Layout("L2"), d, n, 5, cpu) for d in KINDS}
    for d in tasks:
        tasks[d].rain_t = 100 if d == "rain" else 10 ** 9
    kinds = torch.arange(4).repeat_interleave(n // 4)
    base = tasks["push"]
    fe = FastEnv.from_task(base, kinds=kinds)
    fe.rain_t = torch.where(kinds == KINDS["rain"], torch.full_like(kinds, 100), torch.full_like(kinds, 10 ** 9))
    g = base.sample(0, n); u = torch.randn(n, 3, generator=base.gen)
    z = torch.randn(n, 3, generator=base.gen); r = torch.rand(n, generator=base.gen); kz = torch.randn(n, 3, generator=base.gen)
    gm, hm = fe.step(g, u, 150, z, r, kz)
    for d, k in KINDS.items():
        sel = kinds == k
        fk = FastEnv.from_task(tasks[d], kinds=torch.full((int(sel.sum()),), k))
        gk, hk = fk.step(g[sel], u[sel], 150, z[sel], r[sel], kz[sel])
        assert torch.equal(hm[sel], hk) and torch.allclose(gm[sel], gk, atol=1e-6)
