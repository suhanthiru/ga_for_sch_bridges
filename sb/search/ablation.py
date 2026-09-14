"""The ablation's counterfactual: the nearest non-bridge stack, tuned.

SEARCH_PLAN 2.7 defines the ablation delta as the genome's fitness minus the fitness of
the same genome with every bridge node replaced by its nearest non-bridge neighbour. It
does not say where the substitute's own free parameters go, and the first five validated
cells showed why that matters: a threshold copied across incompatible units, or a PD gain
drawn at random, made the delta a measure of the substitute's luck (ERRORS 2026-09-14).

Two changes fix it. `substitute.ablate_bridges` is now deterministic and neutral (a
parameter transfers only between identical spaces, everything else takes its midpoint),
and at the validation rung the substitute is *tuned* here: a short coordinate sweep from
that neutral point, scored at rung-0 cost on the same cell at a seed the validation does
not use, cached per (cell, structure). The delta then compares the bridge against the
best non-bridge counterpart the same budget can find, which is the comparison the map is
supposed to report - and the strict direction, since it can only shrink a bridge's
apparent contribution.
"""
import json
from pathlib import Path

from sb.core.genome import Genome, Node

MAX_EVALS = 8                        # rung-0 evaluations per (cell, structure), cached
TUNE_SEED = 0                        # never one of the rung-2 validation seeds


def points(spec):
    """Up to three values per parameter: the ends of the space and its midpoint."""
    if spec.kind == "choice":
        return list(spec.choices)[:3]
    if spec.kind == "int":
        return sorted({int(spec.lo), int(spec.midpoint()), int(spec.hi)})
    return sorted({float(spec.lo), float(spec.midpoint()), float(spec.hi)})


def with_param(g, nid, name, value, slot_order=None):
    nodes = [Node(n.nid, n.comp, tuple(sorted({**dict(n.params), name: value}.items()))) if n.nid == nid else n for n in g.nodes]
    return Genome(tuple(nodes), g.edges, g.flags, g.provenance, g.grammar_hash).canonical(slot_order)


def tune(ab, grammar, sites, score, max_evals=MAX_EVALS, log=lambda m: None):
    """Coordinate sweep over the substituted nodes' parameters, starting at the neutral
    stack and keeping every improvement. Returns (genome, fitness, evaluations used)."""
    best, best_f, used = ab, score(ab), 1
    for nid in sites:
        try:
            spec = grammar.spec(best.node(nid).comp)
        except KeyError:                                   # a substituted parent dropped this node
            continue
        for name in sorted(spec.params):
            for v in points(spec.params[name]):
                if used >= max_evals:
                    log(f"ablation tuning stopped at the {max_evals}-evaluation cap"); return best, best_f, used
                cand = with_param(best, nid, name, v, grammar.slot_order)
                if cand.gid == best.gid:
                    continue
                f = score(cand); used += 1
                if f > best_f:
                    best, best_f = cand, f
    return best, best_f, used


def _cache_path(models_dir):
    return Path(models_dir) / "ablation_tuning.json"


def load_cache(models_dir):
    p = _cache_path(models_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def tuned_ablation(ab, grammar, sites, cell_id, score, models_dir, max_evals=MAX_EVALS, log=lambda m: None):
    """The tuned counterfactual for one (cell, structure), read from or written to the
    cache beside the models so the ten validation seeds pay for one sweep."""
    key = f"{cell_id}|{ab.sid}"
    cache = load_cache(models_dir)
    hit = cache.get(key)
    if hit is not None:
        out = ab
        for nid, params in hit["params"].items():
            for name, v in params.items():
                out = with_param(out, nid, name, v, grammar.slot_order)
        return out, hit.get("fitness"), 0
    best, f, used = tune(ab, grammar, sites, score, max_evals, log)
    cache[key] = dict(params={nid: dict(best.node(nid).params) for nid in sites if any(n.nid == nid for n in best.nodes)},
                      fitness=f, evals=used, dsl=best.dsl())
    p = _cache_path(models_dir); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n", newline="\n")
    log(f"ablation tuned for cell {cell_id}: {best.dsl()[:110]} (fitness {f:.3f}, {used} evaluations)")
    return best, f, used
