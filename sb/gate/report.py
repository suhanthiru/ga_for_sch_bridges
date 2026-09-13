"""Fill FINDINGS_generator.md from the shards through sb.stats, and nothing else.

Every block between <!-- tables:X --> and <!-- /tables:X --> is replaced by the table
named X; all tables are also written to results/gate/tables.md.
"""
import json
import re
from dataclasses import replace
from pathlib import Path

import pandas as pd

from sb import settings
from sb import stats as ST
from sb.gate import cells as CL
from sb.gate.cells import load_prior
from sb.gate.run_gate import load_episodes, load_gate


def combined(out_dir, prior_dir):
    prior = load_prior(prior_dir); gate = load_gate(out_dir)
    df = pd.concat([prior, gate], ignore_index=True)
    return df, prior, gate


def dynamic_range(prior):
    """SEARCH_PLAN 0.4 on the prior g0 cell table: max over sources per disturbed cell."""
    d = prior[prior.disturbance.isin(ST.DISTURBED) & (prior.run == "prior-g0")]
    if d.empty:
        return {}, None
    ct = ST.cell_table(d)
    mx = ct.groupby(["layout", "disturbance"])["mean"].max().reset_index()
    ok = bool((mx["mean"] >= 0.15).any())
    return dict(cells_at_floor=int((mx["mean"] < 0.15).sum()), n_cells=len(mx), design_has_range=ok), mx


def families(df, seeds):
    d = df[df.seed.isin(seeds)]
    out = {}
    for name, a, b in CL.FAMILIES:
        if (d.source == a).any() and (d.source == b).any():
            fam = ST.contrast_family(d, a, b, name)
            out[name] = [replace(c, verdict=ST.min_effect(c)) for c in fam]
        else:
            out[name] = []
    return out


def cvar_family(eps, seeds):
    """Paired CVaR_0.1 of progress per seed, only for cells where both sources have episodes."""
    e = eps[eps.seed.isin(seeds) & eps.disturbance.isin(ST.DISTURBED)]
    if e.empty:
        return []
    by = e.groupby(["source", "layout", "disturbance", "seed"])["progress"].apply(lambda v: ST.cvar(v.values)).reset_index()
    by = by.rename(columns={"progress": "cvar"})
    out = []
    for name, a, b in CL.FAMILIES:
        if (by.source == a).any() and (by.source == b).any():
            out += [replace(c, verdict=ST.min_effect(c, "cvar")) for c in ST.contrast_family(by, a, b, name + "-cvar", value="cvar")]
    return out


def coverage_table(df, seeds):
    d = df[df.seed.isin(seeds) & df.source.isin(CL.SOURCES) & (df.source != "MPC-oracle") & df.disturbance.isin(ST.DISTURBED)]
    d = d.dropna(subset=["cov_cells"])
    if d.empty:
        return "", None
    cov = d.groupby(["source", "layout"]).agg(cov_cells=("cov_cells", "mean"), off_frac=("off_frac", "mean"),
                                              success=("success", "mean")).reset_index()
    per = d.groupby(["seed", "source", "layout"]).agg(cov_cells=("cov_cells", "mean"), success=("success", "mean")).reset_index()
    per["seed_layout"] = per.seed.astype(str) + per.layout
    est, rs = ST.spearman_by_seed(per, "cov_cells", "success", group="seed_layout")
    txt = ST.report_table(cov.rename(columns={"cov_cells": "mean", "off_frac": "lo", "success": "hi"}).assign(n=len(seeds)), "cell")
    txt = txt.replace("| n | mean | 95% CI |", "| n | off-path cells | [off-path fraction, disturbed success] |")
    txt += "\nSpearman(coverage, disturbed success) across sources within (seed, layout): " + \
        f"mean {est.mean:.3f}, 95% CI [{est.lo:.3f}, {est.hi:.3f}], n = {est.n}\n"
    return txt, est


def build_tables(out_dir, prior_dir, diag_path=None):
    df, prior, gate = combined(out_dir, prior_dir)
    eps = load_episodes(out_dir)
    T = {}
    dr, mx = dynamic_range(prior)
    T["dr"] = (ST.report_table(dr, "kv") + "\n" + ST.report_table(mx.rename(columns={"mean": "mean"}).assign(n=5, lo=float("nan"), hi=float("nan")), "cell")) if mx is not None else "not run\n"
    diag_path = Path(diag_path or (Path(out_dir) / "diag_relabel.json"))
    T["diag"] = ST.report_table(json.loads(diag_path.read_text()), "kv") if diag_path.exists() else "not run\n"
    main = df[df.seed.isin(CL.SEEDS_MAIN)]
    T["cells"] = ST.report_table(ST.cell_table(main), "cell") if not main.empty else "not run\n"
    fam = families(df, CL.SEEDS_MAIN)
    for k, v in fam.items():
        T[k] = ST.report_table(v, "contrast") if v else "not run\n"
    T["cov"], _ = coverage_table(df, CL.SEEDS_MAIN)
    T["cov"] = T["cov"] or "not run\n"
    cv = cvar_family(eps, CL.SEEDS_MAIN)
    T["cvar"] = ST.report_table(cv, "contrast") if cv else "no cells with per-episode records yet\n"
    dec = ST.gate_decision(fam["F1"], fam["F4a"], fam["F3"]) if fam["F1"] else None
    T["decision"] = ST.report_table(dec, "decision") if dec else "not run\n"
    rep = df[df.seed.isin(CL.SEEDS_REP)]
    if not rep.empty:
        famr = families(df, CL.SEEDS_REP); decr = ST.gate_decision(famr["F1"], famr["F4a"], famr["F3"]) if famr["F1"] else None
        T["rep"] = ST.report_table(ST.cell_table(rep), "cell") + "\n" + (ST.report_table(famr["F1"], "contrast") if famr["F1"] else "") + \
            ("\n" + ST.report_table(decr, "decision") if decr else "") + \
            (f"\nAgreement with seeds 0-4: {'yes' if dec and decr and dec['verdict'] == decr['verdict'] else 'no'}\n" if dec and decr else "")
    else:
        T["rep"] = "not run\n"
    me = []
    for k in ("F1", "F5", "F6", "F7"):
        me += fam.get(k, [])
    T["mineffect"] = ST.report_table(me, "contrast") if me else "not run\n"
    tables = "\n".join(f"### {k}\n\n{v}" for k, v in T.items())
    Path(out_dir, "tables.md").write_text(tables, encoding="utf-8", newline="
")
    return T, dec


def fill_findings(T, path=None, commit=""):
    path = Path(path or settings.ROOT / "FINDINGS_generator.md")
    txt = path.read_text(encoding="utf-8")
    for k, v in T.items():
        txt = re.sub(rf"(<!-- tables:{k} -->\n).*?(<!-- /tables:{k} -->)", lambda m: m.group(1) + v + m.group(2), txt, flags=re.S)
    txt = re.sub(r"Commit: .*?\.", f"Commit: {commit}.", txt, count=1)
    path.write_text(txt, encoding="utf-8", newline="
")
