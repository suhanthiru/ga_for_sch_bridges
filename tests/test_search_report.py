import numpy as np

from sb.components import load_all
from sb.core.grammar import Grammar
from sb.search.loop import Search, dummy_evaluate
from sb.search.report import algorithm_census, component_census, elites_frame, interim, slot_census


def test_interim_report_from_a_dummy_search(tmp_path):
    G = Grammar(load_all()); rng = np.random.default_rng(2)
    seeds = [G.random_genome(rng, 0.5) for _ in range(5)]
    s = Search(G, tmp_path / "s", seeds, dummy_evaluate, n_cells=25, log=lambda m: None)
    s.run(budget=150, stop_after_flat=10_000)
    ef = elites_frame(s)
    assert len(ef) == s.map.filled() and "slot_controller" in ef
    sc = slot_census(ef, list(G.root_slots))
    assert abs(sc.set_index("slot").loc["controller"][["bridge", "rl", "other", "empty"]].sum() - 1.0) < 1e-9
    cc, never = component_census(ef, list(G.root_slots), G)
    assert set(cc.component) & set(G.registry) and all(k in G.registry for k in never)
    ac = algorithm_census(s.archive.frame())
    assert ac.evaluations.sum() == len(s.archive.frame()) and (s.archive.frame().rung == 0).sum() == 150
    txt = interim(s, tmp_path / "interim.md", figure=tmp_path / "map.png")
    assert "Slot census" in txt and "Algorithm census" in txt and (tmp_path / "interim.md").exists()
