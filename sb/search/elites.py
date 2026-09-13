"""CVT elite map: k-means centroids over the descriptor space (fixed seed, committed to
disk so the CVT seed is a sensitivity axis), one elite per centroid, and the
'generations held' counter the rung-2 rule needs."""
import json
from pathlib import Path

import numpy as np


def cvt_centroids(dim, n_cells, seed=0, n_samples=50_000, iters=30):
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, dim))
    C = X[rng.choice(n_samples, n_cells, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(n_cells):
            m = a == j
            if m.any():
                C[j] = X[m].mean(0)
    return C


class EliteMap:
    def __init__(self, centroids):
        self.C = np.asarray(centroids, dtype=float)
        self.n = len(self.C)
        self.elite = {}                    # cell -> dict(gid, fitness, eval_id, since_gen, desc)

    @classmethod
    def load_or_make(cls, path, dim, n_cells, seed=0):
        path = Path(path)
        if path.exists():
            return cls(np.load(path))
        C = cvt_centroids(dim, n_cells, seed); path.parent.mkdir(parents=True, exist_ok=True); np.save(path, C)
        return cls(C)

    def cell_of(self, desc):
        d = ((self.C - np.asarray(desc, dtype=float)[None]) ** 2).sum(1)
        return int(d.argmin())

    def insert(self, cell, gid, fitness, eval_id, gen, desc=None):
        """Returns True if the genome became (or improved) the cell's elite."""
        cur = self.elite.get(cell)
        if cur is None or fitness > cur["fitness"]:
            self.elite[cell] = dict(gid=gid, fitness=float(fitness), eval_id=eval_id, since_gen=gen, desc=list(desc) if desc is not None else None)
            return True
        return False

    def held_for(self, cell, gen):
        e = self.elite.get(cell)
        return None if e is None else gen - e["since_gen"]

    def filled(self):
        return len(self.elite)

    def mean_fitness(self):
        return float(np.mean([e["fitness"] for e in self.elite.values()])) if self.elite else float("nan")

    def state(self):
        return {str(k): v for k, v in self.elite.items()}

    def load_state(self, s):
        self.elite = {int(k): v for k, v in s.items()}

    def to_frame(self):
        import pandas as pd
        rows = [dict(cell=k, **{kk: vv for kk, vv in v.items() if kk != "desc"}) for k, v in self.elite.items()]
        return pd.DataFrame(rows)
