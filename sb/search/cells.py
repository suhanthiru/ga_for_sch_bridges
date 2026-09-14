"""Rung-0's eight fixed cells (SEARCH_PLAN 2.8): the 2x2x2 corners of slip magnitude x
push rate x observability in E1 terms, each mapped to its descriptor vector. The centroid
ids are resolved against the committed CVT file at freeze time."""
import hashlib
from itertools import product

from sb.envs.family import descriptor_vector
from sb.search.evaluate import Cell

CORNERS = dict(slip_scale=(0.35, 1.4), push_mult=(0.0, 3.0), observability=("full", "partial"))


def rung0_cells():
    cells = []
    for i, (sl, pm, ob) in enumerate(product(*CORNERS.values())):
        env = "E2" if ob == "partial" else "E1"
        desc = dict(env=env, slip_scale=sl, push_mult=pm, observability=ob)
        cells.append(dict(cell=Cell(i, "L1", descriptor=dict(slip_scale=sl, push_mult=pm), env=env), env=env, vector=descriptor_vector(desc), values=desc))
    return cells


def descriptor_id(vec, digits=6):
    """A stable integer identity for a descriptor vector: the evaluator's cell id. The map's
    cell index depends on the CVT centroids, so it is not the right key for anything cached
    across runs; the conditions a genome is evaluated under are exactly this vector."""
    key = ",".join(f"{float(x):.{digits}f}" for x in vec)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def cell_from_vector(vec, cell_id=None):
    """The evaluator's Cell for a descriptor vector: environment id (E1 or E2 in the pilot),
    slip magnitude and push rate from their bins; the other axes are carried in the
    descriptor dict for the record. Raises for families the evaluator cannot run yet."""
    from sb.envs.family import AXES, BINS
    cell_id = descriptor_id(vec) if cell_id is None else cell_id
    vals = {}
    for ax, x in zip(AXES, vec):
        levels = BINS[ax]; i = int(round(float(x) * (len(levels) - 1)))
        vals[ax] = levels[min(max(i, 0), len(levels) - 1)]
    env = vals["env"] if vals["observability"] == "full" else "E2"
    if env not in ("E1", "E2"):
        raise ValueError(f"the evaluator runs E1 and E2 only; cell asked for {env}")
    return Cell(cell_id, "L1", descriptor=dict(slip_scale=float(vals["slip_scale"]), push_mult=float(vals["push_mult"]),
                                               n_seed=int(vals["n_voronoi"]), aniso=float(vals["aniso"])), env=env)
