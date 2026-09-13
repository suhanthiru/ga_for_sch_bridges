"""Genome canonical form and hashing, registry parameter spaces, grammar validation,
random genomes, mutation, crossover and the control filters — on a dummy component set."""
import numpy as np
import pytest

from sb.core import registry as RG
from sb.core.genome import ROOT, Edge, Genome, Node
from sb.core.grammar import NO_BRIDGE, NO_RL, Grammar
from sb.core.registry import P, SlotSpec


@pytest.fixture
def reg():
    r = {}
    def add(key, slots, sub=None, params=None, tag="none", oracle="none", disabled=""):
        r[key] = RG.ComponentSpec(key, tuple(slots), sub or {}, params or {}, tag, oracle, {}, {}, f"tests/x.py::{key}", disabled, None)
    add("manifold.se2", ["manifold"]); add("manifold.flat", ["manifold"])
    add("controller.pd", ["controller"], params={"kp": P.loguniform(1, 30)})
    add("controller.bridge_drift", ["controller"], sub={"noise": SlotSpec("noise")}, params={"eps": P.loguniform(1e-3, 1e-1)}, tag="bridge")
    add("controller.rl_residual", ["controller"], sub={"base": SlotSpec("controller", optional=False)}, params={"steps": P.int_uniform(1, 5)}, tag="rl")
    add("noise.fixed", ["noise"], params={"sigma": P.uniform(0, 1)}); add("noise.rl", ["noise"], tag="rl")
    add("data.demos", ["data"], oracle="train_data", params={"n": P.choice([1, 5, 20])})
    add("planner.mppi", ["planner"]); add("planner.old", ["planner"], disabled="broken")
    return r


def _pd(kp=6.0):
    return Genome((Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", kp),))),
                  (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b")))


def test_canonical_form_is_order_independent(reg):
    G = Grammar(reg)
    g1 = _pd().canonical(G.slot_order)
    g2 = Genome((Node("zz", "controller.pd", (("kp", 6.0),)), Node("q", "manifold.se2")),
                (Edge(ROOT, "controller", "zz"), Edge(ROOT, "manifold", "q"))).canonical(G.slot_order)
    assert g1.gid == g2.gid and g1.sid == g2.sid and g1.nodes == g2.nodes
    assert _pd(6.0).canonical().sid == _pd(9.0).canonical().sid and _pd(6.0).canonical().gid != _pd(9.0).canonical().gid


def test_json_and_dsl_round_trip(reg):
    G = Grammar(reg); g = G.random_genome(np.random.default_rng(1))
    back = Genome.from_json(g.to_json())
    assert back.gid == g.gid and back.provenance == g.provenance
    assert g.dsl().startswith("(stack ") and "controller" in g.dsl()


def test_validate_catches_the_usual_mistakes(reg):
    G = Grammar(reg)
    assert G.validate(_pd()) == []
    assert any("required slot controller" in v for v in G.validate(Genome((Node("a", "manifold.se2"),), (Edge(ROOT, "manifold", "a"),))))
    bad = Genome((Node("a", "manifold.se2"), Node("b", "noise.fixed", (("sigma", 0.5),))), (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b")))
    assert any("cannot fill" in v for v in G.validate(bad))
    assert any("out of range" in v for v in G.validate(_pd(kp=100.0)))
    dis = Genome((Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 6.0),)), Node("c", "planner.old")),
                 (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "planner", "c")))
    assert any("disabled" in v for v in G.validate(dis))
    orc = Genome((Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 6.0),)), Node("c", "data.demos", (("n", 5),))),
                 (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "planner", "c")))
    assert any("oracle" in v for v in G.validate(orc))


def test_random_genomes_are_valid_and_mutations_stay_valid(reg):
    G = Grammar(reg); rng = np.random.default_rng(0)
    for _ in range(50):
        g = G.random_genome(rng, p_fill=0.8)
        assert G.validate(g) == [], G.validate(g)
        for op in ("replace", "param", "fill", "empty", "flag"):
            m = G.mutate(g, rng, op)
            assert G.validate(m) == [], (op, G.validate(m))
            assert m.provenance.parent_ids == (g.gid,) and m.provenance.op == op
    a, b = G.random_genome(rng, 0.9), G.random_genome(rng, 0.9)
    c = G.crossover(a, b, rng)
    assert G.validate(c) == [] and c.provenance.parent_ids == (a.gid, b.gid)


def test_required_sub_slots_are_filled_and_depth_is_bounded(reg):
    G = Grammar(reg); rng = np.random.default_rng(3)
    seen = False
    for _ in range(100):
        g = G.random_genome(rng, p_fill=1.0)
        if any(n.comp == "controller.rl_residual" for n in g.nodes):
            seen = True
            nid = next(n.nid for n in g.nodes if n.comp == "controller.rl_residual")
            assert g.child_in(nid, "base") is not None
        assert G.depth(g) <= 6
    assert seen


def test_filters_and_innovation_table(reg):
    rng = np.random.default_rng(0)
    nb = Grammar(reg, filt=NO_BRIDGE); nr = Grammar(reg, filt=NO_RL); full = Grammar(reg)
    for _ in range(40):
        assert not nb.has_tag(nb.random_genome(rng, 1.0), "bridge")
        assert not nr.has_tag(nr.random_genome(rng, 1.0), "rl")
    g = full.random_genome(rng, 1.0)
    assert all(e.innov >= 0 for e in g.edges)
    assert full.innov(None, "controller", "controller.pd") == Grammar(reg).innov(None, "controller", "controller.pd")
    assert full.hash != nb.hash and len(full.manifest()["components"]) == len(reg)
    assert all(v for v in (full.validate(g) == [],))


def test_param_specs_sample_mutate_and_clip():
    rng = np.random.default_rng(0)
    lg = P.loguniform(1e-3, 1e-1)
    xs = [lg.sample(rng) for _ in range(200)]
    assert min(xs) >= 1e-3 and max(xs) <= 1e-1 and lg.valid(lg.mutate(0.05, rng))
    it = P.int_uniform(1, 5)
    assert isinstance(it.mutate(3, rng), int) and 1 <= it.mutate(5, rng) <= 5
    ch = P.choice(["a", "b"], p_flip=1.0)
    assert ch.mutate("zzz", rng) in ("a", "b") and not ch.valid("zzz")


def test_innovation_crossover_and_speciation_distance(reg):
    G = Grammar(reg); rng = np.random.default_rng(5)
    for _ in range(30):
        a, b = G.random_genome(rng, 0.9), G.random_genome(rng, 0.9)
        c = G.crossover_innov(a, b, 0.7, 0.4, rng)
        assert G.validate(c) == [], G.validate(c)
        assert c.provenance.op == "xover_innov" and c.provenance.parent_ids == (a.gid, b.gid)
        assert c.innovs() <= (a.innovs() | b.innovs())
    a = G.random_genome(rng, 0.9)
    assert G.compat_distance(a, a) == 0.0
    m = G.mutate(a, rng, "param")
    assert 0.0 <= G.compat_distance(a, m) < 1.0
    r = G.mutate(G.mutate(a, rng, "replace"), rng, "fill")
    assert G.compat_distance(a, r) >= G.compat_distance(a, m) or G.compat_distance(a, r) > 0


def test_rl_never_nests_inside_rl(reg):
    G = Grammar(reg); rng = np.random.default_rng(9)
    for _ in range(200):
        g = G.random_genome(rng, 1.0)
        for e in g.edges:
            if e.parent != ROOT:
                assert not (G.spec(g.node(e.child).comp).tag == "rl" and G.spec(g.node(e.parent).comp).tag == "rl")
        m = G.mutate(g, rng, "replace"); assert G.validate(m) == []
    bad = Genome((Node("a", "manifold.se2"), Node("b", "controller.rl_residual", (("steps", 2),)), Node("c", "controller.rl_residual", (("steps", 3),)),
                  Node("d", "controller.pd", (("kp", 6.0),))),
                 (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge("b", "base", "c"), Edge("c", "base", "d")))
    assert any("inside RL" in v for v in G.validate(bad))


def test_manifest_carries_a_source_hash_and_the_pilot_roots_are_the_consumed_slots():
    from sb.components import load_all
    from sb.core.grammar import PILOT_ROOT_SLOTS
    G = Grammar(load_all()); man = G.manifest()
    assert len(man["source"]) == 16 and man["source"] == Grammar.source_hash()
    assert set(PILOT_ROOT_SLOTS) == {"manifold", "seam", "controller", "trigger", "noise", "safety"}
    P = Grammar(load_all(), root_slots=PILOT_ROOT_SLOTS)
    assert P.hash != G.hash and set(P.manifest()["slots"]) == set(PILOT_ROOT_SLOTS)
