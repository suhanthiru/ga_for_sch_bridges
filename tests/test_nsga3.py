import numpy as np

from sb.components import load_all
from sb.core.grammar import Grammar
from sb.search.algos.nsga3 import NSGA3, das_dennis, dominates, nondominated_sort, objectives
from sb.search.loop import dummy_evaluate


def test_reference_directions_and_sorting():
    d = das_dennis(3, 2)
    assert d.shape == (6, 3) and np.allclose(d.sum(1), 1.0)
    F = np.array([[0, 1], [1, 0], [1, 1], [0.5, 0.5], [2, 2]])
    fr = nondominated_sort(F)
    assert sorted(fr[0]) == [0, 1, 3] and 4 in fr[-1] and dominates(F[0], F[2]) and not dominates(F[0], F[1])


def test_nsga3_keeps_a_nondominated_front_of_bounded_size():
    G = Grammar(load_all()); rng = np.random.default_rng(0)
    algo = NSGA3(G, rng, pop_size=12, partitions=1)
    desc = [0.5] * 13
    for gen in range(6):
        gs = algo.ask(12)
        assert all(G.validate(g) == [] for g in gs)
        rows = [dummy_evaluate(g, desc, 0, 0, G) for g in gs]
        algo.tell(gs, rows)
        assert len(algo.pop) <= 12
    front = algo.front()
    F = np.stack([f[1] for f in front])
    for i in range(len(F)):
        assert not any(dominates(F[j], F[i]) for j in range(len(F)) if j != i)
    assert objectives(dict(success=0.9, collision=0.1))[0] == -0.9 and objectives(dict(success=None))[0] == 0.0
