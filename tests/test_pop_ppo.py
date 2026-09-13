import torch

from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask
from sb.rl.pop_ppo import PopEnv, PopPPO


def test_population_update_runs_and_moves_every_policy(cpu):
    P, n = 3, 8
    tk = GenTask(TR.Layout("L1"), "slip", P * n, 3, cpu)
    env = PopEnv(tk, P, n)
    ppo = PopPPO(P, cpu, hidden=32, depth=2, lr=torch.tensor([1e-3, 1e-3, 0.0]), seed=0)
    obs = env.reset()
    before = {k: v.clone() for k, v in ppo.params.items()}
    gen = torch.Generator().manual_seed(0)
    obs, info = ppo.update(env, obs, rollout=6, epochs=2, minibatch=16, gen=gen)
    assert obs.shape == (P * n, 24) and info["success"].shape == (P,)
    for k, v in ppo.params.items():
        assert torch.isfinite(v).all()
        if k.startswith("a.") or k.startswith("c."):
            assert not torch.allclose(v[0], before[k][0]) and not torch.allclose(v[1], before[k][1])
            assert torch.equal(v[2], before[k][2])                       # lr = 0 leaves the third policy alone


def test_log_prob_matches_torch_normal(cpu):
    ppo = PopPPO(2, cpu, hidden=8, depth=1)
    mu = torch.randn(2, 5, 3); ls = torch.randn(2, 3) * 0.1; a = torch.randn(2, 5, 3)
    ref = torch.distributions.Normal(mu, ls.exp()[:, None, :]).log_prob(a).sum(-1)
    assert torch.allclose(ppo.log_prob(mu, ls, a), ref, atol=1e-6)


def test_graphed_update_is_reproducible_and_matches_the_eager_path_in_shape():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    dev = torch.device("cuda"); P, n = 2, 16
    outs = []
    for _ in range(2):
        tk = GenTask(TR.Layout("L1"), "slip", P * n, 3, dev)
        env = PopEnv(tk, P, n); ppo = PopPPO(P, dev, hidden=32, depth=2, seed=0, autocast=True)
        obs = env.reset(); gen = torch.Generator(device=dev).manual_seed(0)
        for _ in range(4):
            obs, info = ppo.update(env, obs, rollout=8, epochs=2, minibatch=64, gen=gen, graphed=True)
        assert len(ppo.graphs) == 1 and obs.shape == (P * n, 24) and info["success"].shape == (P,)
        outs.append({k: v.clone() for k, v in ppo.params.items()})
    assert all(torch.equal(outs[0][k], outs[1][k]) for k in outs[0])
    assert all(torch.isfinite(v).all() for v in outs[0].values())
    tk = GenTask(TR.Layout("L1"), "slip", P * n, 3, dev); env = PopEnv(tk, P, n); ppo = PopPPO(P, dev, hidden=32, depth=2, seed=0)
    obs = env.reset(); obs, info = ppo.update(env, obs, rollout=8, epochs=2, minibatch=64, gen=torch.Generator(device=dev).manual_seed(0), graphed=False)
    assert obs.shape == (P * n, 24) and float(ppo.t_adam) == 4
