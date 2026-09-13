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
    assert "inv_wall_penetration" in row and not r1.quarantined and r1.invariants["action_bounds"] == 0.0


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


def test_rung0_cells_are_the_eight_corners():
    from sb.search.cells import rung0_cells
    cells = rung0_cells()
    assert len(cells) == 8 and len({tuple(c["vector"]) for c in cells}) == 8
    assert sum(c["env"] == "E2" for c in cells) == 4


def test_residual_controller_starts_as_its_base_and_trains(cpu, small_demos, tmp_path):
    G = Grammar(load_all())
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.rl_residual", (("bound", 0.3), ("clip", 0.2), ("ent", 1e-3), ("lr", 3e-4))),
             Node("c", "controller.pd", (("kp", 6.0),)))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge("b", "base", "c"))
    g = Genome(nodes, edges).canonical(G.slot_order)
    assert G.validate(g) == [], G.validate(g)
    r = evaluate(g, Cell(0, "L1", ("none",)), 0, 0, G, models_dir=tmp_path, device=cpu, episodes=8, demos=small_demos, rl_steps=256)
    assert r.valid, r.invalid_reason + r.error
    assert r.has_rl and r.per_disturbance["none"] > 0.5            # a zero residual on a PD that succeeds undisturbed


def test_partial_observation_cell_runs_learned_controllers(cpu, small_demos, tmp_path):
    G = Grammar(load_all())
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.ppo", (("clip", 0.2), ("ent", 1e-3), ("lr", 3e-4))),
             Node("c", "data.demos", (("mult", 1), ("n_demo", 5))))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge("b", "data", "c"))
    g = Genome(nodes, edges).canonical(G.slot_order)
    r = evaluate(g, Cell(0, "L1", ("none",), env="E2"), 0, 0, G, models_dir=tmp_path, device=cpu, episodes=8, demos=small_demos, rl_steps=256)
    assert r.valid, r.invalid_reason + r.error
    from sb.search.cells import rung0_cells
    assert {c["cell"].env for c in rung0_cells()} == {"E1", "E2"}


def test_rl_noise_placement_trains_around_the_controller(cpu, small_demos, tmp_path):
    G = Grammar(load_all())
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 6.0),)), Node("c", "noise.rl", (("clip", 0.2), ("ent", 1e-3), ("lr", 3e-4))))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "noise", "c"))
    g = Genome(nodes, edges).canonical(G.slot_order)
    assert G.validate(g) == [] and G.has_tag(g, "rl")
    r = evaluate(g, Cell(0, "L1", ("none",)), 0, 0, G, models_dir=tmp_path, device=cpu, episodes=8, demos=small_demos, rl_steps=256)
    assert r.valid, r.invalid_reason + r.error
    assert r.per_disturbance["none"] > 0.5


def test_cell_from_vector_maps_bins_to_the_evaluator():
    from sb.search.cells import cell_from_vector, rung0_cells
    for c in rung0_cells():
        cell = cell_from_vector(c["vector"], c["cell"].cell_id)
        assert cell.env == c["env"] and cell.descriptor["slip_scale"] == c["values"]["slip_scale"]
    import pytest
    from sb.envs.family import AXES, descriptor_vector
    with pytest.raises(ValueError):
        cell_from_vector(descriptor_vector(dict(env="E4")))


def test_distance_trigger_restarts_the_clock_of_a_robot_off_the_reference(cpu, small_demos, tmp_path):
    G = Grammar(load_all())
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 6.0),)), Node("c", "trigger.distance", (("thr", 0.05),)))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "trigger", "c"))
    g = Genome(nodes, edges).canonical(G.slot_order)
    from sb.envs import terrain as TR
    from sb.envs.gen_task import GenTask
    from sb.envs.base import Caps
    from sb.envs import task as TK
    tk = GenTask(TR.Layout("L1"), "none", 8, 1, cpu)
    st = Stack(g, G, tk, small_demos, tmp_path, 0, Caps())
    from sb.core import se2 as S
    t = TK.T_SKILL // 2
    x = S.SE2.interp(tk.means[0].expand(8, 3), tk.means[1].expand(8, 3), torch.full((8,), t / TK.T_SKILL)).clone()
    x[4:, 0] += 0.3                                                       # four robots on the reference, four far off it
    tau, fire = st._replan_clock(x, 0, None, t, None)
    assert fire.tolist() == [False] * 4 + [True] * 4
    assert torch.allclose(tau[:4], torch.full((4,), t / TK.T_SKILL)) and torch.all(tau[4:] == 0)
    tau2, _ = st._replan_clock(x, 0, None, t + 5, None)                 # the restarted clocks advance from the restart
    assert torch.allclose(tau2[4:], torch.full((4,), 5 / TK.T_SKILL))
    u, _ = st.act(x, 0, torch.zeros(8), t)
    assert u.shape == (8, 3)


def test_seam_width_scales_the_bridge_and_is_cached_per_level(tmp_path):
    from sb.gen.bridges import BRIDGE_CFG, bridge_path, quantise_width
    assert quantise_width(1.3) == 1.0 and quantise_width(1.5) == 2.0 and quantise_width(0.3) == 0.25 and quantise_width(3.9) == 4.0
    p1 = bridge_path(tmp_path, "slip", "L1", 0, "", cfg=BRIDGE_CFG); p2 = bridge_path(tmp_path, "slip", "L1", 0, "_w2", cfg=BRIDGE_CFG)
    assert p1 != p2 and "_w2" in p2.name


def test_search_bridge_cache_trains_the_backward_drift_and_exposes_d(cpu, tmp_path):
    from sb.core import se2 as S
    from sb.core import solver as SV
    from sb.envs import terrain as TR
    from sb.envs.gen_task import GenTask
    from sb.gen.bridges import SEARCH_BRIDGE_CFG, get_bridges
    tk = GenTask(TR.Layout("L1"), "none", 8, 1, cpu)
    tiny = dict(SEARCH_BRIDGE_CFG, n_pair=32, steps0=2, steps_ipf=1, batch=16, n_sim=2)
    nets = get_bridges("slip", tk, "L1", 0, cpu, tmp_path, mf=S.SE2, cfg=tiny, width=2.0)
    assert (0, "bwd") in nets[0] and any("_w2" in p.name for p in tmp_path.iterdir())
    ctl = SV.BridgeController(nets, S.SE2, tk, 0, cpu, with_D=True)
    u, extra = ctl(tk.sample(0, 4), 0, torch.full((4,), 0.3), 0)
    assert u.shape == (4, 3) and extra.shape == (4, 2) and torch.isfinite(extra).all()
