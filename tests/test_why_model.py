"""The why model on synthetic rows: a linear mechanism is recovered across held-out
families, noise is not, the sealed verdict follows the registered threshold, and the
clause table has one row per clause combination."""
import numpy as np
import pandas as pd

from sb.envs.family import AXES, BINS
from sb.stats import why_model as M


def _rows(n, rng, mechanism=True, families=("E1", "E2", "E3")):
    rows = []
    for i in range(n):
        fam = families[i % len(families)]
        d = {f"d_{a}": 0.0 for a in AXES}
        d["d_env"] = BINS["env"].index(fam) / (len(BINS["env"]) - 1)
        d["d_goal_modality"] = rng.choice([0.0, 0.5, 1.0]); d["d_terrain_info"] = rng.choice([0.0, 0.5, 1.0]); d["d_observability"] = rng.choice([0.0, 1.0])
        f = dict(w2_demo_to_target=rng.uniform(0, 0.5), sigma_condition=rng.uniform(1, 10), demo_multimodality=rng.integers(1, 4),
                 pd_reachability=rng.uniform(0, 1), shell_fraction=rng.uniform(0, 1), kl_demo_reference=rng.uniform(0, 5))
        delta = (0.4 * f["w2_demo_to_target"] - 0.3 * f["pd_reachability"] + 0.02 * f["kl_demo_reference"]) if mechanism else 0.0
        rows.append(dict(gid=f"g{i}", cell=i, ablation_delta=delta + 0.01 * rng.normal(), **d, **f))
    df = pd.DataFrame(rows)
    df["terrain_info_level"] = [M.terrain_info_level(r) for _, r in df.iterrows()]
    cl = pd.DataFrame([M.clauses(r) for _, r in df.iterrows()])
    return pd.concat([df, cl], axis=1)


def test_a_linear_mechanism_is_recovered_out_of_family_and_noise_is_not():
    rng = np.random.default_rng(0)
    loo = M.fit_loo(_rows(240, rng), kind="ridge")
    assert loo["pooled_r2"] > 0.8 and len(loo["per_group"]) == 3 and all(v > 0.5 for v in loo["per_group"].values())
    noise = M.fit_loo(_rows(240, rng, mechanism=False), kind="ridge")
    assert not (noise["pooled_r2"] > M.WHY_R2_MIN)
    hgb = M.fit_loo(_rows(240, rng), kind="hgb")
    assert hgb["pooled_r2"] > 0.5


def test_sealed_verdict_follows_the_registered_threshold():
    # E7/E8 stand in for the sealed families here; E9/E10 join BINS["env"] when designed
    rng = np.random.default_rng(1)
    train = _rows(200, rng); sealed = _rows(60, rng, families=("E7", "E8"))
    s = M.sealed_test(train, sealed)
    assert s["verdict"] == "supported" and s["r2"] > 0.8 and abs(s["slope"] - 1.0) < 0.2
    s2 = M.sealed_test(train, _rows(60, rng, mechanism=False, families=("E7", "E8")))
    assert s2["verdict"] == "unsupported"


def test_clause_table_has_every_combination_and_the_markdown_carries_the_numbers():
    rng = np.random.default_rng(2)
    df = _rows(120, rng)
    t = M.clause_table(df)
    assert len(t) == 8 and t["n"].sum() == 120 and set(t.columns) >= {"dist_match", "ref_informed", "not_pd_reachable", "n", "mean", "lo", "hi"}
    loo = M.fit_loo(df); txt = M.markdown(loo, t, M.sealed_test(df, _rows(30, rng, families=("E7",))))
    assert f"{loo['pooled_r2']:.3f}" in txt and "Clause table" in txt


def test_single_family_gives_in_sample_only_and_validated_rows_average_seeds():
    rng = np.random.default_rng(3)
    df = _rows(40, rng, families=("E1",))
    loo = M.fit_loo(df)
    assert np.isnan(loo["pooled_r2"]) and np.isfinite(loo["in_sample_r2"]) and loo["groups"] == ["E1"]
    arch = pd.DataFrame([dict(gid="a", cell=0, rung=2, seed=s, valid=True, quarantined=False, has_bridge=True, ablation_delta=0.1 * (s + 1), fitness=0.5,
                              **{f"d_{a}": 0.0 for a in AXES}) for s in range(3)]
                        + [dict(gid="b", cell=0, rung=0, seed=0, valid=True, quarantined=False, has_bridge=True, ablation_delta=np.nan, fitness=0.2,
                                **{f"d_{a}": 0.0 for a in AXES})])
    v = M.validated_rows(arch)
    assert len(v) == 1 and abs(float(v["ablation_delta"].iloc[0]) - 0.2) < 1e-9 and int(v["n_seeds"].iloc[0]) == 3


def test_cell_features_run_on_a_pilot_cell_and_an_empty_archive_reports_nothing(cpu, small_demos, tmp_path):
    from sb.search.cells import rung0_cells
    from sb.search.why_report import cell_features, why_report
    c = rung0_cells()[0]
    f = cell_features(c["vector"], 0, cpu, n=8, demos=small_demos)
    assert set(f) >= set(k for k in M.FEATURES if k != "terrain_info_level") and 0.0 <= f["pd_reachability"] <= 1.0
    (tmp_path / "archive").mkdir()
    txt, rows = why_report(tmp_path, device=cpu, demos=small_demos)
    assert "No validated bridge rows" in txt and len(rows) == 0
