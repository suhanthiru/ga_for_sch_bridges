"""Interim and final search reports from the archive and the elite map, in the order
FINDINGS_search.md asks for: the map, the slot census, contribution by slot, algorithm
census, novelty, negative space. Tables go through sb.stats.report_table where they are
statistics; censuses are counts."""
from pathlib import Path

import numpy as np
import pandas as pd

from sb import stats as ST
from sb.core.genome import ROOT, Genome
from sb.envs.family import AXES


def elites_frame(search):
    """One row per elite cell with its genome's slot contents."""
    rows = []
    for cell, e in search.map.elite.items():
        g = Genome.from_json(search.archive.genome_json(e["gid"]))
        row = dict(cell=cell, gid=e["gid"], fitness=e["fitness"], since_gen=e["since_gen"])
        for edge in g.children(ROOT):
            spec = search.G.spec(g.node(edge.child).comp)
            row[f"slot_{edge.slot}"] = spec.key; row[f"tag_{edge.slot}"] = spec.tag
        rows.append(row)
    return pd.DataFrame(rows)


def slot_census(ef, slots):
    """Per slot: fraction of elites holding a bridge, an RL component, another component, or nothing."""
    out = []
    n = max(len(ef), 1)
    for s in slots:
        col = f"tag_{s}"
        tags = ef[col] if col in ef else pd.Series([None] * len(ef))
        out.append(dict(slot=s, bridge=float((tags == "bridge").sum() / n), rl=float((tags == "rl").sum() / n),
                        other=float((tags == "none").sum() / n), empty=float(tags.isna().sum() / n)))
    return pd.DataFrame(out)


def component_census(ef, slots, grammar):
    """Which components appear in validated elites and how often; the negative space is
    every enabled component that never appears."""
    seen = {}
    for s in slots:
        col = f"slot_{s}"
        if col in ef:
            for k, c in ef[col].dropna().value_counts().items():
                seen[k] = seen.get(k, 0) + int(c)
    never = sorted(k for k, sp in grammar.registry.items() if not sp.disabled and k not in seen)
    return pd.DataFrame(sorted(seen.items(), key=lambda kv: -kv[1]), columns=["component", "elites"]), never


def algorithm_census(df):
    """Per algorithm: evaluations, valid fraction, mean fitness, and share of rows that
    improved a cell at the time (first elite per cell)."""
    g = df.groupby("algorithm")
    out = g.agg(evaluations=("eval_id", "count"), valid=("valid", "mean"), mean_fitness=("fitness", "mean")).reset_index()
    first = df.sort_values("generation").drop_duplicates("cell", keep="first").algorithm.value_counts()
    out["cells_first_filled"] = out.algorithm.map(first).fillna(0).astype(int)
    return out


def bridge_slots(g, grammar):
    """Every (slot, component) in a genome whose component is bridge-tagged, root slots and
    sub-slots alike: the places the ablation actually substitutes."""
    return [(e.slot, g.node(e.child).comp) for e in g.edges if grammar.spec(g.node(e.child).comp).tag == "bridge"]


def inert_bridge(row, g, grammar):
    """True when the genome's only bridge components could not act in this stack. Measured,
    not assumed: a bridge trigger over a controller that reports no forward/backward
    disagreement never fires, and the evaluation records that (ERRORS 2026-09-14)."""
    bs = bridge_slots(g, grammar)
    if not bs or any(slot != "trigger" for slot, _ in bs):
        return False                                    # a bridge elsewhere in the stack does act
    return float(row.get("trigger_fire_rate", np.nan) or 0.0) == 0.0


def contribution_by_slot(df, slots, grammar):
    """Mean ablation delta by the slot that *holds the bridge* - not by every slot in the
    genome, which credited the manifold and the controller for a trigger's delta. Rows
    whose bridge component never acted are counted separately: their delta is a property
    of the counterfactual, not of a bridge."""
    d = df[df.has_bridge & np.isfinite(df.get("ablation_delta", np.nan))] if "ablation_delta" in df else df.iloc[0:0]
    rows = []
    for _, r in d.iterrows():
        g = Genome.from_json(r.genome_json)
        inert = inert_bridge(r, g, grammar)
        for slot, comp in bridge_slots(g, grammar):
            rows.append(dict(slot=slot, comp=comp, delta=r.ablation_delta, inert=inert))
    if not rows:
        return pd.DataFrame(columns=["slot", "n", "mean_delta", "n_inert"])
    x = pd.DataFrame(rows)
    act = x[~x.inert]
    out = x.groupby("slot").agg(n=("delta", "count"), n_inert=("inert", "sum")).reset_index()
    mean = act.groupby("slot").delta.mean().rename("mean_delta_active")
    return out.merge(mean, on="slot", how="left")


def source_census(df):
    """Rows by the grammar and the component source that produced them: an archive that
    crossed a measurement fix says so instead of averaging over both."""
    cols = [c for c in ("grammar_hash", "source_hash", "control") if c in df]
    if not cols:
        return pd.DataFrame()
    return df.groupby(cols).agg(rows=("eval_id", "count"), rung2=("rung", lambda r: int((r == 2).sum()))).reset_index()


def map_figure(search, path, axes=("slip_scale", "push_mult")):
    """Heatmap of elite fitness over two descriptor axes (max over the other axes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i, j = AXES.index(axes[0]), AXES.index(axes[1])
    grid = {}
    for cell, e in search.map.elite.items():
        if e.get("desc") is None:
            continue
        key = (round(e["desc"][i], 2), round(e["desc"][j], 2)); grid[key] = max(grid.get(key, -np.inf), e["fitness"])
    if not grid:
        return None
    xs = sorted({k[0] for k in grid}); ys = sorted({k[1] for k in grid})
    Z = np.full((len(ys), len(xs)), np.nan)
    for (x, y), v in grid.items():
        Z[ys.index(y), xs.index(x)] = v
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs); ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys)
    ax.set_xlabel(axes[0]); ax.set_ylabel(axes[1]); fig.colorbar(im, ax=ax, label="elite fitness")
    Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def interim(search, out_path, slots=None, figure=None):
    slots = slots or list(search.G.root_slots)
    df = search.archive.frame(); ef = elites_frame(search)
    parts = [f"# Search interim report at {search.n_evals} evaluations\n",
             f"Cells filled: {search.map.filled()} of {search.map.n}. Mean elite fitness: {search.map.mean_fitness():.3f}. Rows: {len(df)}.\n"]
    parts.append("## Slot census (fraction of elites)\n\n" + slot_census(ef, slots).to_markdown(index=False) + "\n")
    cc, never = component_census(ef, slots, search.G)
    parts.append("## Components in elites\n\n" + (cc.to_markdown(index=False) if len(cc) else "none") + "\n")
    parts.append("## Negative space (enabled, never in an elite)\n\n" + ("\n".join(f"- {k}" for k in never) if never else "none") + "\n")
    parts.append("## Algorithm census\n\n" + algorithm_census(df).to_markdown(index=False) + "\n")
    cb = contribution_by_slot(df, slots, search.G)
    parts.append("## Bridge contribution by slot (ablation delta, bridge-holding slots only)\n\n"
                 + (cb.to_markdown(index=False) if len(cb) else "no rung-2 ablations yet") + "\n")
    sc = source_census(df)
    if len(sc) > 1:
        parts.append("## Grammar and component source of these rows\n\n" + sc.to_markdown(index=False) + "\n")
    if "trigger_fire_rate" in df:
        t = df[np.isfinite(df.trigger_fire_rate)]
        if len(t):
            parts.append(f"## Trigger activity\n\nRows with a trigger: {len(t)}; never fired: "
                         f"{int((t.trigger_fire_rate == 0).sum())}; median fire rate when it fired: "
                         f"{float(t[t.trigger_fire_rate > 0].trigger_fire_rate.median()) if (t.trigger_fire_rate > 0).any() else float('nan'):.4f}.\n")
    if "novelty_level" in df:
        nv = df.groupby("novelty_level").size().rename("rows").reset_index()
        parts.append("## Novelty levels\n\n" + nv.to_markdown(index=False) + "\n")
    if figure:
        p = map_figure(search, figure)
        if p:
            parts.append(f"## Map\n\n![map]({Path(p).as_posix()})\n")
    txt = "\n".join(parts)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); Path(out_path).write_text(txt, encoding="utf-8", newline="\n")
    return txt
