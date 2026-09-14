"""python search.py --plan SEARCH_PLAN.md [--budget N] [--resume] [--algorithm mapelites|random] [--dummy]

Runs the search over the frozen grammar in grammar/ (or grammar_pilot/ with --pilot),
seeding from its seeds.jsonl, appending every evaluation to results/search/<algorithm>/.
--dummy uses the stand-in evaluator (plumbing checks only; never a result).
"""
import argparse
import sys
from pathlib import Path

from sb import settings
from sb.core.genome import Genome
from sb.search.evaluate import Cell, evaluate
from sb.search.freeze import load_frozen
from sb.search.loop import Search, dummy_evaluate


def real_evaluate(genome, cell_desc, rung, seed, grammar, weights=None):
    from sb.search.cells import cell_from_vector
    cell = cell_from_vector(cell_desc)          # identity derived from the descriptor: the
                                                # ablation cache and the eval ids are per cell
    r = evaluate(genome, cell, rung, seed, grammar, weights=weights)
    return r.row()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="SEARCH_PLAN.md"); ap.add_argument("--budget", type=int, default=20_000)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--algorithm", default="mapelites", choices=["mapelites", "random"])
    ap.add_argument("--pilot", action="store_true"); ap.add_argument("--dummy", action="store_true"); ap.add_argument("--cells", type=int, default=2000)
    ap.add_argument("--out", default=""); ap.add_argument("--rung2-seeds", type=int, default=10)
    ap.add_argument("--control", default="none", choices=["none", "no_bridge", "no_rl"])
    a = ap.parse_args()
    if not Path(a.plan).exists():
        sys.exit(f"{a.plan} not found; the plan is pre-registered before the search runs")
    G, man = load_frozen(pilot=a.pilot)
    gdir = settings.ROOT / ("grammar_pilot" if a.pilot else "grammar")
    seeds = [Genome.from_json(l) for l in (gdir / "seeds.jsonl").read_text().splitlines() if l.strip()]
    root = Path(a.out) if a.out else settings.RESULTS / ("search_dummy" if a.dummy else "search") / (a.algorithm if a.control == "none" else a.control)
    from sb.search.cells import rung0_cells
    fixed = [c["vector"] for c in rung0_cells()]                 # rung 0 proposes on the eight fixed cells
    s = Search(G, root, seeds, dummy_evaluate if a.dummy else real_evaluate, n_cells=a.cells, algorithm=a.algorithm, cells=fixed, control=a.control)
    s.rung_seeds[2] = tuple(range(10, 10 + a.rung2_seeds))
    if a.resume:
        s.resume()
    s.run(a.budget)


if __name__ == "__main__":
    main()
