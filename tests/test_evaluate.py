import numpy as np
import torch

from sb.components import load_all
from sb.core.genome import ROOT, Edge, Genome, Node
from sb.core.grammar import Grammar
from sb.search.evaluate import Cell, OracleGuard, Stack, eval_id_of, evaluate, fitness_of


def _pd_genome(G, kp=6.0, safety=True):
    nodes = [Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", kp),))]
    edges = [Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b")]
    if safety:
        nodes.append(Node("c", "safety.clip")); edges.append(Edge(ROOT, "safety", "c"))
    return Genome(tuple(nodes), tuple(edges)).canonical(G.slot_order)


def test_pd_stack_evaluates_and_is_deterministic(cpu, small_demos, tmp_path):
    G = Grammar(load_all()); g = _pd_genome(G)
    cell = Cell(0, "L1", ("none", "slip"))
    r1 = evaluate(g, cell, 0, 0, G, models_dir=tmp_path, device=cpu, episodes=16, demos=small_demos)
    r2 = evaluate(g, cell, 0, 0, G, models_dir=tmp_path, device=cpu, episodes=16, demos=small_demos)
    assert r1.valid, r1.invalid_reason + r1.error
    assert r1.success == r2.success and r1.eval_id == r2.eval_id == eval_id_of(g, 0, 0, 0)
    assert set(r1.per_disturbance) == {"none", "slip"} and r1.per_disturbance["none"] > 0.8
    assert np.isfinite(r1.fitness) and r1.infer_ms < 50 and not r1.has_bridge and not r1.has_rl
    row = r1.row(); assert "success_slip" in row and "per_disturbance" not in row


def test_invalid_genome_is_logged_not_dropped(cpu, small_demos, tmp_path):
    G = Grammar(load_all())
    bad = Genome((Node("a", "manifold.se2"),), (Edge(ROOT, "manifold", "a"),))
    r = evaluate(bad, Cell(0), 0, 0, G, models_dir=tmp_path, device=cpu, episodes=4, demos=small_demos)
    assert not r.valid and "controller" in r.invalid_reason


def test_oracle_guard_counts_only_when_armed():
    OracleGuard.armed, OracleGuard.count = False, 0
    OracleGuard.touch(); assert OracleGuard.count == 0
    OracleGuard.armed = True; OracleGuard.touch(); assert OracleGuard.count == 1
    OracleGuard.armed = False


def test_fitness_penalises_compute_and_collisions():
    assert fitness_of(0.8, 0.0, 1.0) > fitness_of(0.8, 0.0, 100.0) > fitness_of(0.8, 0.5, 100.0)


def test_noise_and_safety_wrap_the_controller(cpu, small_demos, tmp_path):
    G = Grammar(load_all())
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 6.0),)), Node("c", "safety.clip"), Node("d", "noise.fixed", (("sigma", 0.05),)))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "safety", "c"), Edge(ROOT, "noise", "d"))
    g = Genome(nodes, edges).canonical(G.slot_order)
    from sb.envs import terrain as TR
    from sb.envs.gen_task import GenTask
    from sb.envs.base import Caps
    tk = GenTask(TR.Layout("L1"), "none", 8, 1, cpu)
    st = Stack(g, G, tk, small_demos, tmp_path, 0, Caps())
    torch.manual_seed(0); u1, _ = st.act(tk.sample(0, 8), 0, torch.zeros(8), 0)
    torch.manual_seed(1); u2, _ = st.act(tk.sample(0, 8), 0, torch.zeros(8), 0)
    assert not torch.equal(u1, u2) and u1[:, :2].norm(dim=1).max() <= 1.0 + 1e-6


def test_ppo_controller_trains_from_a_bc_warm_start(cpu, small_demos, tmp_path):
    G = Grammar(load_all())
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.ppo", (("clip", 0.2), ("ent", 1e-3), ("lr", 3e-4))),
             Node("c", "data.demos", (("mult", 1), ("n_demo", 5))))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge("b", "data", "c"))
    g = Genome(nodes, edges).canonical(G.slot_order)
    assert G.validate(g) == [], G.validate(g)
    r = evaluate(g, Cell(0, "L1", ("none",)), 0, 0, G, models_dir=tmp_path, device=cpu, episodes=8, demos=small_demos, rl_steps=256)
    assert r.valid, r.invalid_reason + r.error
    assert r.has_rl and np.isfinite(r.fitness) and r.train_s > 0
