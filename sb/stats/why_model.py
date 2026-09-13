"""The "why" model (SEARCH_PLAN 7.1): does a small set of per-cell features predict the
validated ablation delta, and does the prediction survive a family it never saw?

Rows: validated (rung-2) bridge genomes, one per (genome, cell), the ablation delta
averaged over seeds. Features: the six of `why_features` plus the terrain-information
level. Fit: ridge (standardised) and a gradient-boosted regressor, each with one
environment family held out in turn; the pooled out-of-fold R2 is the number reported.
The sealed test fits on every searched family and scores E9/E10 once; R2 below
WHY_R2_MIN there means H is unsupported at the mechanistic level (SEARCH_PLAN 9).

Clauses of H, one indicator each, from the cell's descriptor and features:
  (i)   distribution match, not point reach:   goal_modality > 1
  (ii)  the reference knows what the controller does not:
        terrain_info != "none" and observability == "partial"
  (iii) the target is not reachable by feedback from the demo manifold:
        pd_reachability < PD_REACH_THR
H predicts a positive delta only where all three hold; the 2x2x2 table is the test.
"""
import numpy as np
import pandas as pd

from sb.envs.family import AXES, BINS
from sb.stats.core import ci95

FEATURES = ("w2_demo_to_target", "sigma_condition", "demo_multimodality", "pd_reachability", "shell_fraction", "kl_demo_reference",
            "terrain_info_level")
WHY_R2_MIN = 0.3
PD_REACH_THR = 0.5
CLAUSES = ("dist_match", "ref_informed", "not_pd_reachable")


# ------------------------------------------------------------------ descriptors
def level_of(row, axis):
    """Decode the archive's d_<axis> fraction back to the axis level."""
    levels = BINS[axis]; x = float(row[f"d_{axis}"])
    i = int(round(x * (len(levels) - 1)))
    return levels[min(max(i, 0), len(levels) - 1)]


def terrain_info_level(row):
    return float(BINS["terrain_info"].index(level_of(row, "terrain_info")))


def clauses(row):
    return dict(dist_match=bool(level_of(row, "goal_modality") > 1),
                ref_informed=bool(level_of(row, "terrain_info") != "none" and level_of(row, "observability") == "partial"),
                not_pd_reachable=bool(float(row["pd_reachability"]) < PD_REACH_THR))


# ------------------------------------------------------------------ rows
def validated_rows(df):
    """One row per (gid, cell) for validated bridge genomes: rung 2, valid, not quarantined,
    a finite ablation delta; the delta and fitness averaged over seeds."""
    d = df[(df["rung"] == 2) & df["valid"].astype(bool) & ~df["quarantined"].astype(bool) & df["has_bridge"].astype(bool)]
    d = d[np.isfinite(d["ablation_delta"].astype(float))]
    if d.empty:
        return d
    keep = ["gid", "cell"] + [f"d_{a}" for a in AXES]
    agg = d.groupby(["gid", "cell"], as_index=False).agg(ablation_delta=("ablation_delta", "mean"), fitness=("fitness", "mean"), n_seeds=("seed", "nunique"))
    first = d.drop_duplicates(["gid", "cell"])[keep]
    return agg.merge(first, on=["gid", "cell"])


def attach_features(rows, feats_by_cell):
    """feats_by_cell: {cell: dict of why_features}. Adds the feature columns and the clauses."""
    out = rows.copy()
    for k in FEATURES:
        if k == "terrain_info_level":
            out[k] = [terrain_info_level(r) for _, r in out.iterrows()]
        else:
            out[k] = [float(feats_by_cell[c][k]) for c in out["cell"]]
    cl = pd.DataFrame([clauses(r) for _, r in out.iterrows()], index=out.index)
    return pd.concat([out, cl], axis=1)


# ------------------------------------------------------------------ fitting
def _model(kind, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=5, random_state=seed)


def r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    ss = float(((y - y.mean()) ** 2).sum())
    return float("nan") if ss == 0 else float(1.0 - ((y - yhat) ** 2).sum() / ss)


def family_of(df):
    return np.array([str(level_of(r, "env")) for _, r in df.iterrows()])


def fit_loo(df, features=FEATURES, target="ablation_delta", group="family", kind="ridge", seed=0):
    """Leave-one-group-out: returns dict(pooled_r2, per_group {g: r2}, n, groups, in_sample_r2).
    The default group is the environment family decoded from the descriptor. With a
    single group the out-of-fold numbers are nan and only in_sample_r2 is set."""
    X = df[list(features)].to_numpy(float); y = df[target].to_numpy(float)
    g = family_of(df) if group == "family" else df[group].to_numpy()
    groups = sorted(set(g.tolist()))
    out = dict(n=int(len(df)), groups=[str(x) for x in groups], kind=kind, per_group={}, pooled_r2=float("nan"), in_sample_r2=float("nan"))
    if len(df) >= 3:
        m = _model(kind, seed).fit(X, y); out["in_sample_r2"] = r2(y, m.predict(X))
    if len(groups) < 2:
        return out
    oof = np.full(len(y), np.nan)
    for gv in groups:
        tr, te = g != gv, g == gv
        if tr.sum() < 3:
            continue
        m = _model(kind, seed).fit(X[tr], y[tr]); oof[te] = m.predict(X[te])
        out["per_group"][str(gv)] = r2(y[te], oof[te]) if te.sum() >= 2 else float("nan")
    ok = np.isfinite(oof)
    out["pooled_r2"] = r2(y[ok], oof[ok]) if ok.sum() >= 3 else float("nan")
    return out


def sealed_test(train, sealed, features=FEATURES, target="ablation_delta", kind="ridge", seed=0):
    """Fit on the searched families, score the sealed ones once: R2, calibration
    (slope and intercept of observed on predicted) and the pre-registered verdict."""
    m = _model(kind, seed).fit(train[list(features)].to_numpy(float), train[target].to_numpy(float))
    yhat = m.predict(sealed[list(features)].to_numpy(float)); y = sealed[target].to_numpy(float)
    slope, intercept = (np.polyfit(yhat, y, 1) if len(y) >= 3 and np.ptp(yhat) > 0 else (float("nan"), float("nan")))
    score = r2(y, yhat)
    return dict(r2=score, slope=float(slope), intercept=float(intercept), n=int(len(y)), kind=kind,
                verdict=("supported" if np.isfinite(score) and score >= WHY_R2_MIN else "unsupported"))


# ------------------------------------------------------------------ the clause table
def clause_table(df, target="ablation_delta"):
    """The 2x2x2 table: for every combination of the three clauses, the mean delta with its
    seed-level t-CI (over rows) and the count; H predicts a positive mean only in the
    all-true cell."""
    rows = []
    for a in (False, True):
        for b in (False, True):
            for c in (False, True):
                sel = df[(df["dist_match"] == a) & (df["ref_informed"] == b) & (df["not_pd_reachable"] == c)]
                e = ci95(sel[target].to_numpy(float)) if len(sel) else None
                rows.append(dict(dist_match=a, ref_informed=b, not_pd_reachable=c, n=int(len(sel)),
                                 mean=(e.mean if e else float("nan")), lo=(e.lo if e else float("nan")), hi=(e.hi if e else float("nan"))))
    return pd.DataFrame(rows)


def markdown(loo, table, sealed=None):
    parts = [f"## Why model ({loo['kind']}): n = {loo['n']}, families {', '.join(loo['groups'])}\n",
             f"Pooled held-out R2 {loo['pooled_r2']:.3f}; in-sample R2 {loo['in_sample_r2']:.3f}."]
    if loo["per_group"]:
        parts.append("Per held-out family: " + ", ".join(f"{k} {v:.3f}" for k, v in loo["per_group"].items()) + ".")
    if sealed:
        parts.append(f"Sealed families: R2 {sealed['r2']:.3f} (threshold {WHY_R2_MIN}), calibration slope {sealed['slope']:.2f}, "
                     f"intercept {sealed['intercept']:.3f}, n = {sealed['n']}: H {sealed['verdict']}.")
    parts.append("\n### Clause table\n\n" + table.to_markdown(index=False, floatfmt=".3f") + "\n")
    return "\n".join(parts)
