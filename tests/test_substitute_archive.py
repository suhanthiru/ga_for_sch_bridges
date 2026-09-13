import numpy as np
import pytest

from sb.core import registry as RG
from sb.core.genome import ROOT, Edge, Genome, Node
from sb.core.grammar import Grammar
from sb.core.novelty import SeedSet, novelty
from sb.core.registry import P, SlotSpec
from sb.core.substitute import ablate_bridges, coverage_check
from sb.search.archive import Archive


@pytest.fixture
def reg():
    r = {}
    def add(key, slots, sub=None, params=None, tag="none", axes=None):
        r[key] = RG.ComponentSpec(key, tuple(slots), sub or {}, params or {}, tag, "none", {}, axes or {}, f"t::{key}", "", None)
    add("manifold.se2", ["manifold"])
    add("controller.pd", ["controller"], params={"kp": P.loguniform(1, 30)})
    add("controller.bridge_drift", ["controller"], sub={"noise": SlotSpec("noise")}, params={"eps": P.loguniform(1e-3, 1e-1)}, tag="bridge",
        axes={"role": "bridge_drift"})
    add("noise.fixed", ["noise"], params={"sigma": P.uniform(0, 1)})
    add("noise.bridge_eps", ["noise"], params={"eps": P.uniform(0, 1)}, tag="bridge", axes={"role": "bridge_eps"})
    add("data.pd_rollouts", ["data"], params={"mult": P.int_uniform(1, 8)})
    add("data.bridge_rollouts", ["data"], params={"mult": P.int_uniform(1, 8)}, tag="bridge", axes={"role": "bridge_rollouts"})
    add("planner.mppi", ["planner"])
    return r


def test_coverage_check_and_ablation(reg):
    G = Grammar(reg)
    assert coverage_check(G) == []
    reg2 = dict(reg); reg2["seam.bridge_cloud"] = RG.ComponentSpec("seam.bridge_cloud", ("seam",), {}, {}, "bridge", "none", {}, {"role": "bridge_cloud"}, "t", "", None)
    assert any("no substitute" in p or "not registered" in p for p in coverage_check(Grammar(reg2)))
    g = Genome((Node("a", "manifold.se2"), Node("b", "controller.bridge_drift", (("eps", 0.01),)), Node("c", "noise.bridge_eps", (("eps", 0.3),)),
                Node("d", "data.bridge_rollouts", (("mult", 4),))),
               (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge("b", "noise", "c"), Edge(ROOT, "data", "d"))).canonical(G.slot_order)
    assert G.validate(g) == [] and G.has_tag(g, "bridge")
    a = ablate_bridges(g, G)
    assert G.validate(a) == [] and not G.has_tag(a, "bridge")
    comps = {n.comp for n in a.nodes}
    assert comps == {"manifold.se2", "controller.pd", "data.pd_rollouts"}       # the bridge's noise sub-slot went with it
    assert dict(a.node(next(n.nid for n in a.nodes if n.comp == "data.pd_rollouts")).params)["mult"] == 4
    assert ablate_bridges(a, G).gid == a.gid                                     # idempotent, identity without bridges
    assert a.provenance.op == "ablate" and a.provenance.parent_ids == (g.gid,)


def test_novelty_levels(reg):
    G = Grammar(reg); rng = np.random.default_rng(0)
    seeds = [G.random_genome(rng, 0.5) for _ in range(20)]
    S = SeedSet.from_genomes(seeds)
    assert novelty(seeds[0], S)[0] == 0
    S2 = SeedSet.from_json(S.to_json()); assert S2 == S
    g = seeds[0]
    m = G.mutate(g, rng, "param")
    assert novelty(m, S)[0] == 0
    # a slot-edge no seed used: force the bridge controller if the seeds never had it
    seen = {e for s in seeds for e in s.innovs()}
    fresh = Genome((Node("a", "manifold.se2"), Node("b", "controller.bridge_drift", (("eps", 0.01),))),
                   (Edge(ROOT, "manifold", "a", G.innov(None, "manifold", "manifold.se2")), Edge(ROOT, "controller", "b", G.innov(None, "controller", "controller.bridge_drift")))).canonical(G.slot_order)
    lvl, dist = novelty(fresh, S)
    assert lvl in (1, 2, 3) and dist >= 0
    if G.innov(None, "controller", "controller.bridge_drift") not in seen:
        assert lvl >= 2


def test_archive_append_tags_checkpoint_resume(tmp_path):
    A = Archive(tmp_path / "search", rows_per_shard=3)
    for i in range(7):
        A.append(dict(eval_id=f"e{i}", gid=f"g{i % 2}", fitness=i / 10))
    assert len(list((tmp_path / "search" / "archive").glob("*.parquet"))) == 2 and len(A._buf) == 1
    A.tag("e2", "exclude:quarantine", "invariant tripped")
    df = A.frame(); assert len(df) == 6 and "e2" not in set(df.eval_id)
    assert len(A.frame(include_excluded=True)) == 7
    d = A.checkpoint(dict(generation=5))
    assert A.verify(d) and (tmp_path / "search" / "LATEST").read_text() == d.name
    A.append(dict(eval_id="e7", gid="g0", fitness=0.7)); A.flush()
    B = Archive(tmp_path / "search", rows_per_shard=3)
    state, replayed = B.resume()
    assert state["generation"] == 5 and replayed == 1 and len(B.frame(include_excluded=True)) == 8
    # corrupt the checkpoint: resume must refuse it and fall back to nothing
    (d / "state.json").write_text("{}")
    assert not B.verify(d)
    state2, _ = B.resume()
    assert state2 is None
    # duplicate eval ids are read once
    B.append(dict(eval_id="e7", gid="g0", fitness=0.9)); B.flush()
    assert len(B.frame(include_excluded=True)) == 8
