import numpy as np

from sb.components import load_all
from sb.core.grammar import Grammar
from sb.search.loop import Search, dummy_evaluate
from sb.search.surrogate import Surrogate


def test_surrogate_trains_prescreens_and_logs_accuracy(tmp_path):
    G = Grammar(load_all()); rng = np.random.default_rng(0)
    seeds = [G.random_genome(rng, 0.5) for _ in range(5)]
    s = Search(G, tmp_path / "s", seeds, dummy_evaluate, n_cells=20, log=lambda m: None)
    s.surrogate.min_rows, s.surrogate.retrain_every = 120, 100          # small budgets for the test
    s.run(budget=320, stop_after_flat=10_000)
    assert s.surrogate.ready and np.isfinite(s.surrogate.r2) and s.surrogate.trained_at >= 120
    st = s.state(); assert "surrogate_r2" in st and st["surrogate_trained_at"] == s.surrogate.trained_at
    cands = [G.mutate(seeds[0], rng) for _ in range(10)]
    picked = s.surrogate.prescreen(cands, s.random_cell_desc(), rng)
    assert len(picked) == 4 and all(any(p.gid == c.gid for c in cands) for p in picked)
    sur = Surrogate(G); assert sur.prescreen(cands[:3], [0.0] * 13, rng) == cands[:3]      # not ready: pass-through
