"""The ablation's counterfactual: deterministic substitution, no transfer between unlike
parameter spaces, and a tuned substitute at the validation rung (ERRORS 2026-09-14)."""
import json

import numpy as np
import pytest
import torch

from sb.components import load_all
from sb.core.genome import ROOT, Edge, Genome, Node
from sb.core.grammar import Grammar
from sb.core.registry import P
from sb.core.substitute import ablate_bridges, ablation_sites
from sb.search.ablation import points, tune, tuned_ablation, with_param


def _trigger_genome(G, thr=0.050576, kp=21.9):
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", kp),)), Node("c", "trigger.bridge_disagreement", (("thr", thr),)))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "trigger", "c"))
    return Genome(nodes, edges).canonical(G.slot_order)


def test_param_spaces_midpoint_and_identity():
    assert P.loguniform(0.02, 0.5).midpoint() == pytest.approx(0.1)
    assert P.uniform(0.1, 0.3).midpoint() == pytest.approx(0.2)
    assert P.int_uniform(10, 100).midpoint() == 55
    assert P.choice([32, 64]).midpoint() == 64
    assert P.loguniform(0.02, 0.5).same_space(P.loguniform(0.02, 0.5))
    assert not P.loguniform(0.02, 0.5).same_space(P.loguniform(0.05, 1.0))   # the two trigger thresholds
    assert not P.loguniform(0.02, 0.5).same_space(P.uniform(0.02, 0.5))


def test_threshold_does_not_transfer_between_unlike_spaces_and_the_result_is_deterministic():
    G = Grammar(load_all())
    g = _trigger_genome(G)
    a1, a2 = ablate_bridges(g, G), ablate_bridges(g, G)
    assert a1.gid == a2.gid and ablate_bridges(a1, G).gid == a1.gid          # deterministic and idempotent
    node = next(n for n in a1.nodes if n.comp == "trigger.distance")
    assert dict(node.params)["thr"] == pytest.approx(0.1)                     # the midpoint, not the copied 0.0506
    assert ablation_sites(g, a1) == [node.nid]
    # a gain that is not in the map is the substitute's midpoint too, never a random draw
    gb = Genome((Node("a", "manifold.flat"), Node("b", "controller.grid_bridge", (("eps", 0.02), ("grid", 32), ("iters", 50), ("kp_heading", 9.6)))),
                (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"))).canonical(G.slot_order)
    ab = ablate_bridges(gb, G)
    pd = next(n for n in ab.nodes if n.comp == "controller.pd")
    assert dict(pd.params)["kp"] == pytest.approx(G.spec("controller.pd").params["kp"].midpoint())


def test_tuning_walks_to_the_best_point_and_caches_per_cell_and_structure(tmp_path):
    G = Grammar(load_all())
    g = _trigger_genome(G); ab = ablate_bridges(g, G); sites = ablation_sites(g, ab)
    seen = []

    def score(cand):                                       # a synthetic landscape: the largest threshold wins
        thr = dict(next(n for n in cand.nodes if n.comp == "trigger.distance").params)["thr"]
        seen.append(thr); return float(thr)

    best, f, used = tune(ab, G, sites, score)
    assert f == pytest.approx(0.5) and used <= 8 and 0.1 in seen            # started at the midpoint
    assert dict(next(n for n in best.nodes if n.comp == "trigger.distance").params)["thr"] == pytest.approx(0.5)
    b2, f2, used2 = tuned_ablation(ab, G, sites, 3, score, tmp_path)
    assert used2 > 0 and (tmp_path / "ablation_tuning.json").exists()
    n_calls = len(seen)
    b3, f3, used3 = tuned_ablation(ab, G, sites, 3, score, tmp_path)        # second seed: cache hit, no evaluations
    assert used3 == 0 and len(seen) == n_calls and b3.gid == b2.gid
    cache = json.loads((tmp_path / "ablation_tuning.json").read_text())
    assert list(cache) == ["3|" + ab.sid] and cache["3|" + ab.sid]["evals"] == used2


def test_points_cover_the_ends_and_with_param_is_a_pure_edit():
    G = Grammar(load_all())
    assert points(P.loguniform(0.02, 0.5)) == [0.02, pytest.approx(0.1), 0.5]
    assert points(P.choice(["a", "b", "c", "d"])) == ["a", "b", "c"]
    g = _trigger_genome(G)
    nid = next(n.nid for n in g.nodes if n.comp == "controller.pd")
    h = with_param(g, nid, "kp", 3.0, G.slot_order)
    assert dict(h.node(nid).params)["kp"] == 3.0 and dict(g.node(nid).params)["kp"] == pytest.approx(21.9)
    assert not G.validate(h) and h.gid != g.gid


def test_trigger_activity_is_recorded_and_an_inert_bridge_trigger_shows_as_such(cpu, small_demos, tmp_path):
    from sb.search.cells import rung0_cells
    from sb.search.evaluate import evaluate
    G = Grammar(load_all())
    cell = rung0_cells()[0]["cell"]; cell.disturbances = ("none", "slip")
    r = evaluate(_trigger_genome(G), cell, 0, 0, G, models_dir=tmp_path, device=cpu, episodes=8, demos=small_demos)
    assert r.valid and r.has_bridge
    assert r.trigger_fire_rate == 0.0 and r.trigger_d_seen == 0.0            # a PD reports no disagreement: the trigger cannot act
    row = r.row(); assert "trigger_fire_rate" in row and "ablation_tuned" in row
    # a distance trigger at a small threshold does act, and that is visible
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 21.9),)), Node("c", "trigger.distance", (("thr", 0.03),)))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "trigger", "c"))
    d = evaluate(Genome(nodes, edges).canonical(G.slot_order), cell, 0, 0, G, models_dir=tmp_path, device=cpu, episodes=8, demos=small_demos)
    assert d.valid and d.trigger_fire_rate > 0 and not d.has_bridge
