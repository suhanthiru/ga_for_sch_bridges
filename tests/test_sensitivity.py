import numpy as np

from sb.components import load_all
from sb.core.grammar import Grammar
from sb.search.loop import dummy_evaluate
from sb.search.sensitivity import run_study, stability, variants


def test_sensitivity_study_runs_every_variant_and_scores_stability(tmp_path):
    G = Grammar(load_all()); rng = np.random.default_rng(0)
    seeds = [G.random_genome(rng, 0.5) for _ in range(4)]
    maps, st = run_study(G, tmp_path, seeds, dummy_evaluate, budget=60, n_cells=12)
    assert set(maps) == set(variants()) and all(maps[k] for k in maps)
    assert st["n_cells"] == len(maps["base"]) and 0.0 <= st["stable_fraction"] <= 1.0
    same = stability({"base": maps["base"], "copy": maps["base"]})
    assert same["stable_fraction"] == 1.0
