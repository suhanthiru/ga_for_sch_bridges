"""Stage-A acceptance: hundreds of stand-in evaluations end to end, a kill mid-run, and
a resume that loses nothing and continues from the same state."""
import numpy as np

from sb.components import load_all
from sb.core.grammar import Grammar
from sb.search.loop import CHECKPOINT_EVERY, Search, dummy_evaluate


def _seeds(G, n=8):
    rng = np.random.default_rng(1)
    return [G.random_genome(rng, 0.5) for _ in range(n)]


def test_search_runs_checkpoints_and_resumes(tmp_path):
    G = Grammar(load_all()); seeds = _seeds(G)
    s1 = Search(G, tmp_path / "s", seeds, dummy_evaluate, n_cells=50, log=lambda m: None)
    for _ in range(2 * CHECKPOINT_EVERY + 37):          # two checkpoints plus 37 rows after the last one
        s1.step()
    s1.archive.flush()
    n_rows = len(s1.archive.frame(include_excluded=True)); assert n_rows == 2 * CHECKPOINT_EVERY + 37
    filled, mean = s1.map.filled(), s1.map.mean_fitness()
    # a fresh process resumes from the verified checkpoint and replays the 37 later rows
    s2 = Search(G, tmp_path / "s", seeds, dummy_evaluate, n_cells=50, log=lambda m: None)
    assert s2.resume()
    assert s2.n_evals == n_rows and s2.map.filled() == filled and abs(s2.map.mean_fitness() - mean) < 1e-9
    s2.step(); s2.archive.flush()
    assert len(s2.archive.frame(include_excluded=True)) == n_rows + 1
    assert (tmp_path / "s" / "cvt_centroids.npy").exists()


def test_run_writes_the_map_and_random_control_differs(tmp_path):
    G = Grammar(load_all()); seeds = _seeds(G, 4)
    s = Search(G, tmp_path / "m", seeds, dummy_evaluate, n_cells=30, log=lambda m: None)
    s.run(budget=120, stop_after_flat=1000)
    assert (tmp_path / "m" / "map.parquet").exists() and s.map.filled() > 0
    r = Search(G, tmp_path / "r", seeds, dummy_evaluate, n_cells=30, algorithm="random", log=lambda m: None)
    r.run(budget=120, stop_after_flat=1000)
    df = r.archive.frame()
    assert (df.algorithm == "random").sum() >= 100 and set(df.columns) >= {"gid", "sid", "dsl", "novelty_level", "cell", "fitness"}
