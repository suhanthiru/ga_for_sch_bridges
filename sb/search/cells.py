"""Rung-0's eight fixed cells (SEARCH_PLAN 2.8): the 2x2x2 corners of slip magnitude x
push rate x observability in E1 terms, each mapped to its descriptor vector. The centroid
ids are resolved against the committed CVT file at freeze time."""
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
