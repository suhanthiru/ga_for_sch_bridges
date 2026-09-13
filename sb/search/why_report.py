"""The why report for a search directory (SEARCH_PLAN 7.1): per-cell features from the
archive's descriptors, the validated bridge rows, the held-out-family fit and the clause
table, written as markdown. `python -m sb.cli report why --root <dir> [--pilot]`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sb import settings
from sb.envs import terrain as TR
from sb.envs.family import AXES
from sb.envs.gen_task import GenTask
from sb.policies.demos import load_demos
from sb.search.archive import Archive
from sb.search.cells import cell_from_vector
from sb.stats import why_features as W
from sb.stats import why_model as M

_TASK_KEYS = ("slip_scale", "aniso", "n_seed", "push_mult", "heading_std")


def cell_features(vec, cell_id, device, n=64, seed=0, demos=None):
    """Features of one descriptor vector: the undisturbed task carries the route and the
    map, PD reachability is averaged over the cell's disturbances (all but `none`)."""
    cell = cell_from_vector(vec, cell_id)
    kw = {k: v for k, v in cell.descriptor.items() if k in _TASK_KEYS}
    tk0 = GenTask(TR.Layout(cell.layout), "none", n, 50_000 + seed, device, **kw)
    G, _ = demos if demos is not None else load_demos(settings.DATA / f"demos_{cell.layout}.npz", device)
    dist = [d for d in cell.disturbances if d != "none"] or ["none"]
    tks = [GenTask(TR.Layout(cell.layout), d, n, 50_000 + seed, device, **kw) for d in dist]
    f = W.features(tk0, tks[0], G)
    f["pd_reachability"] = float(np.mean([W.pd_reachability(t, G) for t in tks]))
    return f


def features_by_cell(rows, device, n=64, demos=None):
    out = {}
    for _, r in rows.drop_duplicates("cell").iterrows():
        vec = [float(r[f"d_{a}"]) for a in AXES]
        out[int(r["cell"])] = cell_features(vec, int(r["cell"]), device, n=n, demos=demos)
    return out


def why_report(root, out_path=None, device=None, n=64, demos=None):
    root = Path(root); device = device or settings.device()
    df = Archive(root).frame()
    rows = M.validated_rows(df) if len(df) else df
    out_path = Path(out_path or root / "why_model.md")
    if len(rows) == 0:
        txt = "# Why model\n\nNo validated bridge rows yet.\n"
        out_path.write_text(txt, encoding="utf-8", newline="\n"); return txt, rows
    feats = features_by_cell(rows, device, n=n, demos=demos)
    rows = M.attach_features(rows, feats)
    parts = ["# Why model\n"]
    for kind in ("ridge", "hgb"):
        loo = M.fit_loo(rows, kind=kind)
        parts.append(M.markdown(loo, M.clause_table(rows)))
    parts.append("## Rows\n\n" + rows[["gid", "cell", "ablation_delta", "n_seeds", *M.FEATURES, *M.CLAUSES]].to_markdown(index=False, floatfmt=".3f") + "\n")
    txt = "\n".join(parts)
    out_path.write_text(txt, encoding="utf-8", newline="\n")
    return txt, rows
