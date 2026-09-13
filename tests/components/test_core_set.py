"""Each registered component builds and returns something of the expected shape. The
test node ids here are the ones the components declare; the freeze runs them."""
import numpy as np
import pytest
import torch

from sb.components import load_all
from sb.core.grammar import Grammar
from sb.core.substitute import coverage_check
from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask


@pytest.fixture(scope="module")
def reg():
    return load_all()


@pytest.fixture
def ctx(cpu, small_demos, tmp_path):
    tk = GenTask(TR.Layout("L1"), "none", 8, 1, cpu)
    return dict(task=tk, demos=small_demos, models=tmp_path, seed=0, manifold=None, sub={})


def _build(reg, key, ctx, **over):
    spec = reg[key]; rng = np.random.default_rng(0)
    params = {k: p.sample(rng) for k, p in spec.params.items()}; params.update(over)
    return spec.cls().build(params, ctx)


def test_manifolds(reg, ctx):
    assert _build(reg, "manifold.se2", ctx).name == "se2" and _build(reg, "manifold.flat", ctx).name == "flat"


def test_references(reg, ctx):
    g = ctx["task"].sample(0, 8); tau = torch.full((8,), 0.3)
    for key in ("reference.geodesic", "reference.spline", "reference.nearest_demo"):
        assert _build(reg, key, ctx)(g, 0, tau).shape == (8, 3)


def test_planner(reg, ctx):
    m = _build(reg, "planner.mppi", ctx, K=64, H=10)
    assert m.act(ctx["task"].sample(0, 8), 0, 0).shape == (8, 3)
    assert _build(reg, "planner.spline", ctx)(ctx["task"].sample(0, 8), 0, torch.full((8,), 0.5)).shape == (8, 3)


def test_seams(reg, ctx):
    assert _build(reg, "seam.marginal_cloud", ctx)["width"] > 0 and _build(reg, "seam.fixed_clock", ctx)["kind"] == "fixed_clock"


def test_controllers(reg, ctx):
    g = ctx["task"].sample(0, 8); tau = torch.full((8,), 0.2)
    u, _ = _build(reg, "controller.pd", ctx, kp=6.0)(g, 0, tau, 20)
    assert u.shape == (8, 3)
    rec = _build(reg, "controller.ppo", dict(ctx, sub={"data": ctx["demos"]}))
    assert rec["kind"] == "ppo" and rec["data"][0].shape == ctx["demos"][0].shape
    spec = reg["controller.bridge_drift"]
    assert spec.tag == "bridge" and spec.axes["role"] == "bridge_drift" and "reference" in spec.params


def test_data(reg, ctx):
    G, U = _build(reg, "data.demos", ctx, n_demo=5, mult=1)
    assert G.shape[0] == 5 and U.shape == (5, 300, 3)
    G, U = _build(reg, "data.dart", ctx, n_demo=4, mult=1)
    assert G.shape[0] == 8
    assert reg["data.pd_relabel"].tag == "bridge" and reg["data.oracle_rollouts"].oracle == "train_data"


def test_augment(reg, ctx):
    G, U = _build(reg, "augment.noised", ctx, mult=1)
    assert G.shape[0] == 2 * ctx["demos"][0].shape[0]


def test_small_slots(reg, ctx):
    assert _build(reg, "trigger.distance", ctx)["kind"] == "distance" and _build(reg, "noise.fixed", ctx)["sigma"] > 0
    assert len(_build(reg, "noise.per_skill", ctx)["sigmas"]) == 3
    assert _build(reg, "value.distance", ctx)(ctx["task"].sample(0, 4), 0).shape == (4,)
    assert _build(reg, "safety.clip", ctx)(torch.ones(2, 3) * 5).abs().max() <= 3.0
    assert _build(reg, "time_split.equal", ctx) == [100, 100, 100] and _build(reg, "adapt.none", ctx) is None


def test_grammar_over_the_set_is_consistent(reg):
    G = Grammar(reg)
    assert coverage_check(G) == [], coverage_check(G)
    rng = np.random.default_rng(0)
    for _ in range(30):
        g = G.random_genome(rng, 0.7)
        assert G.validate(g) == [], G.validate(g)
    assert all(s.test for s in reg.values())
