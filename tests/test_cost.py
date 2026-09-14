"""The training cost is a function of the genome, not of the clock (ERRORS 2026-09-14)."""
import pytest

from sb.components import load_all
from sb.core.genome import ROOT, Edge, Genome, Node
from sb.core.grammar import Grammar
from sb.search.cost import train_cost_s
from sb.search.evaluate import fitness_of


def _g(G, *nodes_and_slots):
    nodes = tuple(n for n, _ in nodes_and_slots)
    edges = tuple(Edge(ROOT, slot, n.nid) for n, slot in nodes_and_slots)
    return Genome(nodes, edges).canonical(G.slot_order)


@pytest.fixture(scope="module")
def G():
    return Grammar(load_all())


def test_a_pd_trains_nothing_and_a_neural_bridge_costs_the_most(G):
    pd = _g(G, (Node("a", "manifold.se2"), "manifold"), (Node("b", "controller.pd", (("kp", 6.0),)), "controller"))
    grid = _g(G, (Node("a", "manifold.flat"), "manifold"),
              (Node("b", "controller.grid_bridge", (("eps", 0.02), ("grid", 32), ("iters", 50), ("kp_heading", 4.0))), "controller"))
    neural = _g(G, (Node("a", "manifold.se2"), "manifold"),
                (Node("b", "controller.bridge_drift", (("coupling", "independent"), ("eps", 0.01), ("ipf", 0),
                                                       ("reference", "slip"), ("sampler", "sde_em"), ("steps", 32))), "controller"))
    c = lambda g, rung=0: train_cost_s(g, G, 100_000, rung)
    assert c(pd) == 0.0
    assert 0 < c(grid) < 1.0 < c(neural)
    assert c(neural) == pytest.approx(3 * 2 * 1200 * 8.3e-3)
    assert fitness_of(0.8, 0.0, c(pd)) > fitness_of(0.8, 0.0, c(grid)) > fitness_of(0.8, 0.0, c(neural))


def test_the_cost_does_not_move_between_identical_evaluations_or_with_cache_state(G):
    neural = _g(G, (Node("a", "manifold.se2"), "manifold"),
                (Node("b", "controller.bridge_drift", (("coupling", "ot"), ("eps", 0.01), ("ipf", 2),
                                                       ("reference", "slip"), ("sampler", "sde_em"), ("steps", 16))), "controller"))
    first = train_cost_s(neural, G, 100_000, 0)
    assert all(train_cost_s(neural, G, 100_000, 0) == first for _ in range(5))
    # the IPF gene is honoured at the validation rung, and costs more there
    assert train_cost_s(neural, G, 100_000, 2) > first == train_cost_s(neural, G, 100_000, 1)


def test_learned_controllers_scale_with_their_declared_work(G):
    d2000, d8000 = (_g(G, (Node("a", "manifold.se2"), "manifold"),
                       (Node("b", "controller.diffusion", (("hidden", 128), ("steps", n))), "controller")) for n in (2000, 8000))
    assert train_cost_s(d8000, G, 0, 0) == pytest.approx(4 * train_cost_s(d2000, G, 0, 0))
    ppo = _g(G, (Node("a", "manifold.se2"), "manifold"),
             (Node("b", "controller.ppo", (("clip", 0.2), ("ent", 1e-3), ("lr", 3e-4))), "controller"))
    assert train_cost_s(ppo, G, 250_000, 1) == pytest.approx(2.5 * train_cost_s(ppo, G, 100_000, 0))
    assert train_cost_s(ppo, G, 0, 0) == 0.0


def test_a_residual_pays_for_itself_and_for_its_base(G):
    base = Node("c", "controller.grid_bridge", (("eps", 0.02), ("grid", 32), ("iters", 50), ("kp_heading", 4.0)))
    res = Node("b", "controller.rl_residual", (("bound", 0.3), ("clip", 0.2), ("ent", 1e-3), ("lr", 3e-4)))
    g = Genome((Node("a", "manifold.se2"), res, base),
               (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge("b", "base", "c"))).canonical(G.slot_order)
    only_res = 100_000 * 3.8e-5
    assert train_cost_s(g, G, 100_000, 0) == pytest.approx(only_res + 0.04)      # the sub-slot base is counted too


def test_fitness_is_reproducible_for_the_same_genome_and_outcome(cpu, small_demos, tmp_path):
    from sb.search.cells import rung0_cells
    from sb.search.evaluate import evaluate
    G_ = Grammar(load_all())
    g = _g(G_, (Node("a", "manifold.se2"), "manifold"), (Node("b", "controller.pd", (("kp", 9.0),)), "controller"))
    cell = rung0_cells()[0]["cell"]; cell.disturbances = ("none", "slip")
    a = evaluate(g, cell, 0, 0, G_, models_dir=tmp_path, device=cpu, episodes=8, demos=small_demos)
    b = evaluate(g, cell, 0, 0, G_, models_dir=tmp_path, device=cpu, episodes=8, demos=small_demos)
    assert a.fitness == b.fitness and a.train_cost_s == b.train_cost_s == 0.0
    assert a.train_s >= 0.0 and "train_cost_s" in a.row()
