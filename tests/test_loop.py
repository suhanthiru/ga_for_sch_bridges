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
    df1 = s1.archive.frame(include_excluded=True); n_rows = len(df1)
    assert s1.n_evals == 2 * CHECKPOINT_EVERY + 37 and (df1.rung == 0).sum() == s1.n_evals and n_rows > s1.n_evals
    filled, mean = s1.map.filled(), s1.map.mean_fitness()
    # a fresh process resumes from the verified checkpoint and replays the rows written after it
    s2 = Search(G, tmp_path / "s", seeds, dummy_evaluate, n_cells=50, log=lambda m: None)
    assert s2.resume()
    assert s2.n_evals == s1.n_evals and s2.map.filled() == filled and abs(s2.map.mean_fitness() - mean) < 1e-9
    s2.step(); s2.archive.flush()
    assert (s2.archive.frame(include_excluded=True).rung == 0).sum() == n_rows - (n_rows - s1.n_evals) + 1
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


def test_cma_batches_appear_and_improve_structures(tmp_path):
    G = Grammar(load_all()); seeds = _seeds(G, 6)
    s = Search(G, tmp_path / "c", seeds, dummy_evaluate, n_cells=20, log=lambda m: None)
    s.run(budget=400, stop_after_flat=10_000)
    df = s.archive.frame()
    assert (df.algorithm == "cma_mae").sum() > 20 and s.emitters
    assert set(df[df.algorithm == "cma_mae"].sid) <= set(df.sid)


def test_ladder_rung1_seeds_the_map_and_rung2_validates(tmp_path):
    G = Grammar(load_all()); seeds = _seeds(G, 4)
    s = Search(G, tmp_path / "l", seeds, dummy_evaluate, n_cells=15, log=lambda m: None)
    s.rung_seeds = {1: (1, 2), 2: (10, 11)}
    s.run(budget=80, stop_after_flat=10_000)
    df = s.archive.frame()
    assert set(df.rung) >= {0, 1} and (df.rung == 1).sum() >= 2
    assert s.validated and all(v["fitness"] is not None for v in s.validated.values())
    assert (df.rung == 2).sum() >= 2 * len(s.validated)
    st = s.state(); assert "validated" in st


def test_fixed_cells_become_the_map(tmp_path):
    from sb.search.cells import rung0_cells
    G = Grammar(load_all()); seeds = _seeds(G, 3)
    fixed = [c["vector"] for c in rung0_cells()]
    s = Search(G, tmp_path / "f", seeds, dummy_evaluate, cells=fixed, log=lambda m: None)
    assert s.map.n == 8 and all(s.map.cell_of(v) == i for i, v in enumerate(fixed))
    s.run(budget=30, stop_after_flat=10_000)
    assert set(s.archive.frame().cell) <= set(range(8))


def test_control_searches_never_propose_the_excluded_tag_and_keep_the_hash(tmp_path):
    import pytest
    G = Grammar(load_all()); seeds = _seeds(G, 12)
    assert any(G.has_tag(g, "bridge") for g in seeds)                     # the seed set does carry bridges
    msgs = []
    s = Search(G, tmp_path / "nb", seeds, dummy_evaluate, n_cells=30, log=msgs.append, control="no_bridge")
    assert len(s.seeds) < len(seeds) and all(not G.has_tag(g, "bridge") for g in s.seeds) and "control no_bridge" in msgs[0]
    s.run(budget=80, stop_after_flat=1000)
    df = s.archive.frame(include_excluded=True)
    assert len(df) >= 80 and not df["has_bridge"].any() and (df["control"] == "no_bridge").all() and (df["grammar_hash"] == G.hash).all()
    r = Search(G, tmp_path / "nb", seeds, dummy_evaluate, n_cells=30, log=lambda m: None, control="no_bridge")
    assert r.resume()
    with pytest.raises(AssertionError):
        Search(G, tmp_path / "nb", seeds, dummy_evaluate, n_cells=30, log=lambda m: None, control="none").resume()
    nr = Search(G, tmp_path / "nr", seeds, dummy_evaluate, n_cells=30, log=lambda m: None, control="no_rl")
    nr.run(budget=40, stop_after_flat=1000)
    assert not nr.archive.frame(include_excluded=True)["has_rl"].any()


def test_rows_carry_the_source_hash_and_a_source_change_resumes_with_a_note(tmp_path, monkeypatch):
    G = Grammar(load_all()); seeds = _seeds(G, 4)
    s = Search(G, tmp_path / "s", seeds, dummy_evaluate, n_cells=20, log=lambda m: None)
    for _ in range(CHECKPOINT_EVERY):
        s.step()
    s.archive.flush()
    df = s.archive.frame(include_excluded=True)
    assert (df["source_hash"] == G.source_hash()).all() and (df["grammar_hash"] == G.hash).all()
    msgs = []
    r = Search(G, tmp_path / "s", seeds, dummy_evaluate, n_cells=20, log=msgs.append)
    monkeypatch.setattr(type(G), "source_hash", staticmethod(lambda: "0" * 16))
    assert r.resume() and r.n_evals == CHECKPOINT_EVERY                      # the grammar is unchanged: the run continues
    assert any("component-source change" in m for m in msgs)
