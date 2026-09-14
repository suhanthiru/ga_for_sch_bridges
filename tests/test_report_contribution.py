"""The ablation delta is attributed to the slot that holds the bridge, and a bridge
component that never acted is counted apart from one that did (ERRORS 2026-09-14)."""
import numpy as np
import pandas as pd

from sb.components import load_all
from sb.core.genome import ROOT, Edge, Genome, Node
from sb.core.grammar import Grammar
from sb.search.report import bridge_slots, contribution_by_slot, inert_bridge, source_census


def _row(g, delta, fire=np.nan, **kw):
    return dict(eval_id=f"e{abs(hash(g.gid + str(delta))) % 10 ** 8}", genome_json=g.to_json(), has_bridge=True,
                ablation_delta=delta, trigger_fire_rate=fire, rung=2, **kw)


def _pd_with_bridge_trigger(G, thr=0.05):
    nodes = (Node("a", "manifold.se2"), Node("b", "controller.pd", (("kp", 6.0),)), Node("c", "trigger.bridge_disagreement", (("thr", thr),)))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"), Edge(ROOT, "trigger", "c"))
    return Genome(nodes, edges).canonical(G.slot_order)


def _bridge_controller(G):
    nodes = (Node("a", "manifold.flat"), Node("b", "controller.grid_bridge", (("eps", 0.02), ("grid", 32), ("iters", 50), ("kp_heading", 4.0))))
    edges = (Edge(ROOT, "manifold", "a"), Edge(ROOT, "controller", "b"))
    return Genome(nodes, edges).canonical(G.slot_order)


def test_only_bridge_holding_slots_are_credited():
    G = Grammar(load_all())
    g = _pd_with_bridge_trigger(G)
    assert bridge_slots(g, G) == [("trigger", "trigger.bridge_disagreement")]
    df = pd.DataFrame([_row(g, 0.57, fire=0.0)])
    cb = contribution_by_slot(df, list(G.root_slots), G)
    assert list(cb.slot) == ["trigger"]                       # not manifold, not controller
    assert int(cb.n.iloc[0]) == 1 and int(cb.n_inert.iloc[0]) == 1 and np.isnan(cb.mean_delta_active.iloc[0])


def test_active_and_inert_bridges_are_separated():
    G = Grammar(load_all())
    inert, active = _pd_with_bridge_trigger(G), _bridge_controller(G)
    assert inert_bridge(dict(trigger_fire_rate=0.0), inert, G)
    assert not inert_bridge(dict(trigger_fire_rate=0.01), inert, G)
    assert not inert_bridge(dict(trigger_fire_rate=np.nan), active, G)     # a bridge controller always acts
    df = pd.DataFrame([_row(inert, 0.57, fire=0.0), _row(active, -0.41), _row(active, -0.30)])
    cb = contribution_by_slot(df, list(G.root_slots), G).set_index("slot")
    assert int(cb.loc["trigger", "n_inert"]) == 1 and np.isnan(cb.loc["trigger", "mean_delta_active"])
    assert int(cb.loc["controller", "n"]) == 2 and cb.loc["controller", "mean_delta_active"] == -0.355
    assert int(cb.loc["controller", "n_inert"]) == 0


def test_source_census_separates_rows_across_a_measurement_fix():
    df = pd.DataFrame([dict(eval_id="a", grammar_hash="g1", source_hash="s1", control="none", rung=0),
                       dict(eval_id="b", grammar_hash="g1", source_hash="s1", control="none", rung=2),
                       dict(eval_id="c", grammar_hash="g1", source_hash="s2", control="none", rung=2)])
    sc = source_census(df)
    assert len(sc) == 2 and set(sc.source_hash) == {"s1", "s2"} and int(sc[sc.source_hash == "s1"].rows.iloc[0]) == 2
    assert int(sc[sc.source_hash == "s1"].rung2.iloc[0]) == 1
