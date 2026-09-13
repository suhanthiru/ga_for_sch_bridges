"""Search-sensitivity study (SEARCH_PLAN 5.11): rerun the primary search at a fraction
of the budget with the fitness weights scaled by +-50 %, the promotion threshold moved by
+-0.05 and a different CVT seed, and report which cells keep the same elite structure.
Only cells whose elite is stable across the variants enter the findings.

The evaluator is called with a `weights` keyword so the fitness recombination happens
inside it; the stand-in evaluator honours it the same way the real one does.
"""
import json
from pathlib import Path

import numpy as np

from sb.search.loop import Search

BASE_WEIGHTS = dict(compute=0.02, collision=0.05)


def variants():
    return {
        "base": dict(weights=BASE_WEIGHTS, cvt_seed=0, margin=0.1),
        "weights_x0.5": dict(weights={k: 0.5 * v for k, v in BASE_WEIGHTS.items()}, cvt_seed=0, margin=0.1),
        "weights_x1.5": dict(weights={k: 1.5 * v for k, v in BASE_WEIGHTS.items()}, cvt_seed=0, margin=0.1),
        "margin_-0.05": dict(weights=BASE_WEIGHTS, cvt_seed=0, margin=0.05),
        "margin_+0.05": dict(weights=BASE_WEIGHTS, cvt_seed=0, margin=0.15),
        "cvt_seed_1": dict(weights=BASE_WEIGHTS, cvt_seed=1, margin=0.1),
    }


def run_study(grammar, root, seeds, evaluate, budget, n_cells=200, log=lambda m: None):
    """Runs every variant, returns the per-variant elite maps and the stability table."""
    root = Path(root); maps = {}
    for name, v in variants().items():
        ev = lambda g, d, r, s, G, _w=v["weights"]: evaluate(g, d, r, s, G, weights=_w)
        s = Search(grammar, root / name, seeds, ev, n_cells=n_cells, cvt_seed=v["cvt_seed"], log=log)
        s.run(budget, stop_after_flat=10 ** 9)
        maps[name] = {c: dict(sid=None, gid=e["gid"], desc=e["desc"], fitness=e["fitness"]) for c, e in s.map.elite.items()}
        for c, e in maps[name].items():
            from sb.core.genome import Genome
            e["sid"] = Genome.from_json(s.archive.genome_json(e["gid"])).sid
    return maps, stability(maps)


def stability(maps, base="base"):
    """For each base cell: the fraction of variants (matched by the elite's descriptor to the
    variant's nearest cell) whose elite has the same structure id. Cells at >= 0.8 are stable."""
    b = maps[base]; rows = []
    others = {k: v for k, v in maps.items() if k != base}
    for c, e in b.items():
        agree = 0
        for name, m in others.items():
            if not m:
                continue
            # match by descriptor: the variant's elite whose descriptor is nearest to this cell's
            desc = np.asarray(e["desc"]); near = min(m.values(), key=lambda x: np.sum((np.asarray(x["desc"]) - desc) ** 2))
            agree += int(near["sid"] == e["sid"])
        rows.append(dict(cell=c, sid=e["sid"], agreement=agree / max(len(others), 1)))
    stable = [r for r in rows if r["agreement"] >= 0.8]
    return dict(rows=rows, n_cells=len(rows), n_stable=len(stable), stable_fraction=(len(stable) / len(rows) if rows else float("nan")))


def write(study, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(study, indent=1) + "\n", newline="\n")
