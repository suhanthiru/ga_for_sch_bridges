"""Generator-suite invariants: the shared noise stream, DART's action noise, the planners,
the goal-conditioned generator, the bridge cache key, and the diffusion schedule."""
import torch

from sb.core import bridge_fast as BF
from sb.core import se2 as S
from sb.envs import task as TK
from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask, make_ref
from sb.gen import controllers as C
from sb.gen.bridges import BRIDGE_CFG, bridge_path
from sb.gen.noise import NoiseStream, noise_step
from sb.gen.rollout import rollout
from sb.policies.common import OBS_GC_DIM, obs_gc
from sb.policies.diffusion import DiffPolicy, train_diffusion


def test_shared_noise_stream(cpu, l1_task):
    z1, z2 = NoiseStream(16, 7, cpu), NoiseStream(16, 7, cpu)
    assert torch.equal(z1.z, z2.z)
    assert not torch.equal(z1.z, NoiseStream(16, 8, cpu).z)
    ref = make_ref("slip", l1_task)
    g = l1_task.sample(0, 8); z = z1(0)[:8]
    a = noise_step(ref, S.SE2, g, z); b = noise_step(ref, S.SE2, g, z)
    assert torch.equal(a, b)
    want = (TK.DT * BF.cov_at(ref, g, S.SE2, "diag")).sqrt() * z
    assert float((a - want).abs().max()) < 1e-6


def test_dart_action_noise_matches_slip_covariance(cpu):
    n = 20000
    tk = GenTask(TR.Layout("L1"), "none", n, 2, cpu); ref = make_ref("slip", tk)
    g = tk.sample(1, n); g[:, :2] = 0.5
    z = torch.randn(n, 3)
    eta = noise_step(ref, S.SE2, g, z) / TK.DT
    emp = (eta * TK.DT).var(0); want = TK.DT * BF.cov_at(ref, g, S.SE2, "diag")[0]
    assert float(((emp - want).abs() / want).max()) < 0.05


def test_relabel_is_reproducible(cpu, small_demos):
    tk = GenTask(TR.Layout("L1"), "none", 8, 3, cpu)
    G = small_demos[0][:8]
    ref = make_ref("slip", tk); z = NoiseStream(8, 0, cpu)
    _, U1 = rollout(tk, S.SE2, C.mppi_act(tk, 5), 8, z, ref, states=G)
    _, U2 = rollout(tk, S.SE2, C.mppi_act(tk, 5), 8, z, ref, states=G)
    assert float((U1 - U2).abs().max()) < 1e-3
    u_direct = C.MPPI(tk, seed=5).act(G[:, 0], 0, 0)
    assert float((U1[:, 0] - TK.clip_u(u_direct)).abs().max()) < 1e-3


def test_mppi_reads_no_terrain(cpu, l1_task):
    tk = GenTask(TR.Layout("L1"), "none", 8, 1, cpu)
    g = l1_task.sample(0, 8)
    tk.obs_fields = tk.true_fields = tk.rain_fields = None      # any terrain lookup would now fail
    u = C.MPPI(tk, seed=1).act(g, 0, 0)
    assert u.shape == (8, 3)


def test_track_cost_targets_the_interpolant(cpu, l1_task):
    m = C.MPPI(l1_task, cost="track")
    n = 4
    early = m._target(0, 0, 0, n); late = m._target(0, 90, 29, n)
    assert torch.allclose(late, l1_task.means[1].expand(n, 3))
    assert not torch.allclose(early, late)
    assert torch.allclose(C.MPPI(l1_task, cost="greedy")._target(0, 0, 0, n), l1_task.means[1].expand(n, 3))


def test_oracle_model_uses_true_friction(cpu):
    tk = GenTask(TR.Layout("L1"), "slip", 8, 1, cpu)
    g = tk.sample(0, 8); u = torch.tensor([[0.5, 0.0, 0.0]]).expand(8, 3)
    o = C.MPCOracle(tk, seed=0)
    tk.true_fields = tk.true_fields.clone(); tk.true_fields[3] = 1.0
    d1 = S.between(g, o._model_step(g, u, 0))[:, 0]
    tk.true_fields[3] = 0.5
    d2 = S.between(g, o._model_step(g, u, 0))[:, 0]
    assert torch.allclose(d2, 0.5 * d1, atol=1e-6)
    assert torch.allclose(d1, torch.full((8,), 0.5 * TK.DT), atol=1e-6)


def test_gc_generator_rollouts(cpu, small_demos, l1_task):
    Gd, Ud = small_demos
    pol = train_diffusion(l1_task, Gd, Ud, 0, cpu, 5, obs_fn=obs_gc, obs_dim=OBS_GC_DIM)
    from sb.gen.sources import start_states
    ref = make_ref("slip", l1_task); z = NoiseStream(8, 0, cpu); g0 = start_states(l1_task, 8, 0, cpu)
    G1, U1 = rollout(l1_task, S.SE2, C.gc_diffusion_act(pol, l1_task, 0), 8, z, ref, starts=g0)
    G2, U2 = rollout(l1_task, S.SE2, C.gc_diffusion_act(pol, l1_task, 0), 8, z, ref, starts=g0)
    G3, U3 = rollout(l1_task, S.SE2, C.gc_diffusion_act(pol, l1_task, 1), 8, z, ref, starts=g0)
    assert G1.shape == (8, 301, 3) and U1.shape == (8, 300, 3)
    assert torch.equal(U1, U2) and torch.equal(G1, G2)
    assert not torch.equal(U1, U3)
    assert obs_gc(l1_task, Gd[:4, 0], 0, torch.zeros(4)).shape == (4, OBS_GC_DIM)


def test_bridge_cache_key_includes_cfg(tmp_path):
    a = bridge_path(tmp_path, "slip", "L1", 0)
    b = bridge_path(tmp_path, "slip", "L1", 0, cfg=dict(BRIDGE_CFG, steps0=1500))
    assert a != b and a.parent == b.parent


def test_diffusion_schedule_reaches_noise():
    pol = DiffPolicy()
    assert float(pol.abar[-1]) < 0.05
    assert bool((pol.abar[1:] < pol.abar[:-1]).all())


def test_rollout_progress_is_graded(cpu, small_demos):
    tk = GenTask(TR.Layout("L1"), "none", 8, 1, cpu)
    from sb.policies.nominal import Nominal
    out = tk.rollout(Nominal(tk), n=8)
    p = out["progress"]
    assert p.shape == (8,) and bool(((p * 3).round() - p * 3).abs().max() < 1e-6)
    assert torch.equal(out["reached"][2], out["success"])       # the last handoff is the goal
    assert bool((p[out["success"]] >= 1 / 3 - 1e-6).all())
