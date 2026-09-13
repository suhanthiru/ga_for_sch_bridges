"""The gate's fixed design (SEARCH_PLAN 0.2) and its row schema."""
import numpy as np
import pandas as pd

from sb.gen.sources import SOURCES as _BASE

# SEARCH_PLAN 0.5 fired (PLAN_CHANGES 2026-09-13): the 2x2 label isolation joins the sources
SOURCES = _BASE + ("PD-relabel", "BRIDGE-on-PD")

N_DEMO, GEN_MULT, N_EVAL, STEPS = 20, 4, 200, 8000
LAYOUTS = ("L1", "L2")
DISTS = ("none", "slip", "rain", "push")
SEEDS_MAIN = (0, 1, 2, 3, 4)
SEEDS_REP = (5, 6, 7, 8, 9)

# cells the prior run in skill_chains already covers (read from its shards, never re-run)
PRIOR_G0 = ("DEMO", "NOISED", "BRIDGE-slip", "PD-noise", "DART", "MPPI-rollouts")
PRIOR_G1 = ("BRIDGE-unicycle", "PD-iso", "MPC-relabel")
PRIOR_RENAME = {"MPC-rollout": "MPPI-rollouts"}

FAMILIES = (("F1", "BRIDGE-slip", "PD-noise"), ("F2", "BRIDGE-slip", "BRIDGE-unicycle"),
            ("F3", "PD-noise", "PD-iso"), ("F4a", "PD-relabel", "BRIDGE-slip"), ("F4b", "BRIDGE-on-PD", "PD-noise"),
            ("F4", "MPC-relabel", "BRIDGE-slip"),
            ("F5", "MPC-oracle", "BRIDGE-slip"), ("F6", "GC-diff-rollouts", "PD-noise"),
            ("F7", "MPPI-rollouts", "PD-noise"))


def is_prior(source, layout, seed):
    if seed not in SEEDS_MAIN:
        return False
    if source in PRIOR_G0:
        return True
    return source in PRIOR_G1 and layout == "L1"


def cells_for(seed, sources=SOURCES, layouts=LAYOUTS):
    """(source, layout) pairs this repo runs for a seed: the ones the prior run lacks."""
    return [(s, l) for l in layouts for s in sources if not is_prior(s, l, seed)]


SCHEMA = dict(run="gate", phase="gate", source="", layout="", disturbance="", seed=0, policy="diffusion",
              n_demo=N_DEMO, gen_mult=GEN_MULT, steps=STEPS, n_eval=N_EVAL, slip_scale_eval=0.7, slip_scale_gen=0.7,
              aniso=1.0, n_voronoi=14, push_mult=1.0, heading_std=np.nan, demo_noise=0.0, map_flip=0.0,
              manifold="se2", route="L", mppi_cost="greedy", n_gen=0, cov_cells=np.nan, off_frac=np.nan,
              mean_disp=np.nan, iso_var=np.nan, gen_collision=np.nan, gen_success=np.nan, train_s=np.nan,
              success=np.nan, collision=np.nan, handoff1_md=np.nan, handoff2_md=np.nan, w2_goal=np.nan,
              energy=np.nan, commit="")
DTYPES = {k: (str if isinstance(v, str) else (np.int64 if isinstance(v, int) else np.float64)) for k, v in SCHEMA.items()}


def make_row(**kw):
    r = dict(SCHEMA)
    for k, v in kw.items():
        if k not in r:
            raise KeyError(f"{k} is not a schema column")
        r[k] = v
    return r


def frame(rows):
    df = pd.DataFrame(rows, columns=list(SCHEMA))
    for k, t in DTYPES.items():
        df[k] = df[k].astype(t) if t is not str else df[k].astype("string")
    return df


def load_prior(prior_dir):
    """The prior g0/g1 shards mapped onto this schema (source renamed, run tagged)."""
    import glob
    from pathlib import Path
    fs = sorted(glob.glob(str(Path(prior_dir) / "phaseg[01]_seed[0-4].parquet")))
    if not fs:
        return frame([])
    dfs = []
    for f in fs:
        d = pd.read_parquet(f)
        d["source"] = d["source"].replace(PRIOR_RENAME)
        d["run"] = "prior-" + d["phase"].astype(str)
        d = d.rename(columns={"n_seed": "n_voronoi"})
        d["heading_std"] = pd.to_numeric(d["heading_std"], errors="coerce")
        for k in SCHEMA:
            if k not in d.columns:
                d[k] = SCHEMA[k]
        dfs.append(d[list(SCHEMA)])
    out = frame(pd.concat(dfs, ignore_index=True).to_dict("records"))
    # g1 re-evaluated g0's BRIDGE-slip and PD-noise policies; keep the g0 rows only
    dup = (out.run == "prior-g1") & out.source.isin(("BRIDGE-slip", "PD-noise"))
    return out[~dup].reset_index(drop=True)
