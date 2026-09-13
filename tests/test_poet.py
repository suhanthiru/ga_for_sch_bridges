import numpy as np

from sb.components import load_all
from sb.core.grammar import Grammar
from sb.envs.family import AXES, descriptor_vector
from sb.search.algos.poet import POET, perturb_env
from sb.search.loop import dummy_evaluate


def test_poet_admits_within_the_band_and_transfers(tmp_path):
    G = Grammar(load_all()); rng = np.random.default_rng(0)
    seeds = [G.random_genome(rng, 0.5) for _ in range(4)]
    p = POET(G, dummy_evaluate, seeds, rng, max_active=6, children_per_gen=3)
    p.start(descriptor_vector({}))
    for _ in range(12):
        p.step()
    assert 1 <= len(p.active) <= 6 and p.n_evals > 12
    keys = {e.key() for e in p.active}; assert len(keys) == len(p.active)
    em = p.edge_map()
    assert all(0.0 <= x <= 1.0 for e in em for x in e["desc"]) and all("has_bridge" in e for e in em)
    d = perturb_env(descriptor_vector({}), rng)
    assert len(d) == len(AXES) and sum(abs(a - b) > 1e-9 for a, b in zip(d, descriptor_vector({}))) <= 1
    for e in p.active:
        assert e.elite is not None and G.validate(e.elite) == []
