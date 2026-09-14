"""A cell's identity comes from its descriptor, so anything cached or keyed per cell - the
ablation's tuned counterfactual, the evaluator's eval ids - is actually per cell. Every
evaluation used to run with cell_id 0 (ERRORS 2026-09-14)."""
import pytest

from sb.search.cells import cell_from_vector, descriptor_id, rung0_cells
from sb.search.evaluate import eval_id_of


def test_the_eight_fixed_cells_get_distinct_stable_identities():
    cs = rung0_cells()
    ids = [cell_from_vector(c["vector"]).cell_id for c in cs]
    assert len(set(ids)) == len(cs) == 8
    assert ids == [cell_from_vector(c["vector"]).cell_id for c in cs]          # stable across calls
    assert [c["cell"].cell_id for c in cs] == list(range(8))                   # the fixed cells keep readable ids


def test_identity_follows_the_descriptor_not_the_call():
    a = [0.0] * 13
    b = list(a); b[1] = 1.0
    assert descriptor_id(a) == descriptor_id(list(a)) != descriptor_id(b)
    assert descriptor_id([0.3333333]) == descriptor_id([0.33333334])           # rounded to six digits
    assert cell_from_vector(a, cell_id=7).cell_id == 7                         # an explicit id still wins


def test_eval_ids_of_one_genome_differ_between_cells():
    from sb.components import load_all
    from sb.core.genome import ROOT, Edge, Genome, Node
    from sb.core.grammar import Grammar
    G = Grammar(load_all())
    g = Genome((Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 6.0),))),
               (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"))).canonical(G.slot_order)
    cs = rung0_cells()
    ids = {eval_id_of(g, cell_from_vector(c["vector"]).cell_id, 2, 10) for c in cs}
    assert len(ids) == 8


def test_the_ablation_cache_key_is_per_cell(tmp_path):
    from sb.components import load_all
    from sb.core.grammar import Grammar
    from sb.core.substitute import ablate_bridges, ablation_sites
    from sb.core.genome import ROOT, Edge, Genome, Node
    from sb.search.ablation import tuned_ablation
    G = Grammar(load_all())
    g = Genome((Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 6.0),)), Node("c", "trigger.bridge_disagreement", (("thr", 0.1),))),
               (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "trigger", "c"))).canonical(G.slot_order)
    ab = ablate_bridges(g, G); sites = ablation_sites(g, ab)
    calls = []

    def score(cand):
        calls.append(cand.gid); return 0.5

    cs = rung0_cells()
    for c in cs[:2]:
        tuned_ablation(ab, G, sites, cell_from_vector(c["vector"]).cell_id, score, tmp_path)
    import json
    cache = json.loads((tmp_path / "ablation_tuning.json").read_text())
    assert len(cache) == 2, "two cells must tune their own counterfactual, not share one"
    assert len({k.split("|")[0] for k in cache}) == 2
