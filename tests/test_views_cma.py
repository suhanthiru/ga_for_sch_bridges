import numpy as np

from sb.components import load_all
from sb.core.genome import ROOT, Edge, Genome, Node
from sb.core.grammar import Grammar
from sb.core.views import ParamView
from sb.search.algos.cma_emitter import CMAEmitter


def _g(G, kp=6.0, sigma=0.01):
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", kp),)), Node("c", "noise.fixed", (("sigma", sigma),)))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "noise", "c"))
    return Genome(nodes, edges).canonical(G.slot_order)


def test_param_view_round_trips_in_log_space():
    G = Grammar(load_all()); g = _g(G)
    v = ParamView(g, G)
    assert v.dim == 2 and abs(v.vector()[0] - np.log10(6.0)) < 1e-12 or abs(v.vector()[1] - np.log10(6.0)) < 1e-12
    back = v.apply(v.vector())
    assert back.gid == g.gid
    lo, hi = v.bounds(); assert (lo < hi).all()
    far = v.apply(np.array([10.0, 10.0]))                       # clipped to the parameter ranges
    assert G.validate(far) == []


def test_cma_emitter_improves_a_smooth_objective():
    G = Grammar(load_all()); g = _g(G, kp=2.0, sigma=0.05)
    em = CMAEmitter(g, G, sigma0=0.3, seed=0, popsize=8)
    assert em.active

    def f(genome):                                                # peak at kp = 12, sigma = 1e-3
        p = {n.comp: dict(n.params) for n in genome.nodes}
        return -((np.log10(p["controller.pd"]["kp"]) - np.log10(12)) ** 2 + (np.log10(p["noise.fixed"]["sigma"]) + 3) ** 2)

    best0 = f(g); best = best0; elite = best0
    for gen in range(25):
        cands = em.ask(gen)
        assert cands and all(G.validate(c) == [] for c in cands) and all(c.sid == g.sid for c in cands)
        fs = [f(c) for c in cands]
        em.tell(fs, elite)
        best = max(best, max(fs)); elite = max(elite, max(fs))
    assert best > best0 + 0.5
