"""NSGA-III over the genome (SEARCH_PLAN 5.4), hand-rolled so the operators are the
grammar's own. Objectives, all minimised: -success, -cvar_01, -worst_of_20, collision,
energy, train compute, inference latency, demo count. Survival: non-dominated sorting,
then reference-direction niching (Das-Dennis directions) on the last admitted front.
The front per descriptor cell is what the findings report, not a scalarised elite."""
from itertools import combinations

import numpy as np

OBJECTIVES = (("success", -1.0), ("cvar_01", -1.0), ("worst_of_20", -1.0), ("collision", 1.0), ("energy", 1.0),
              ("train_s", 1.0), ("infer_ms", 1.0), ("n_demo", 1.0))


def objectives(row):
    out = []
    for k, sgn in OBJECTIVES:
        v = row.get(k, 0.0)
        v = 0.0 if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
        out.append(sgn * v)
    return np.asarray(out, dtype=float)


def das_dennis(m, p):
    """Reference directions on the simplex in m dimensions with p partitions."""
    out = []
    for c in combinations(range(m + p - 1), m - 1):
        pts = [-1] + list(c) + [m + p - 1]
        out.append([(pts[i + 1] - pts[i] - 1) / p for i in range(m)])
    return np.asarray(out, dtype=float)


def dominates(a, b):
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_sort(F):
    n = len(F); S = [[] for _ in range(n)]; cnt = np.zeros(n, dtype=int); fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(F[i], F[j]):
                    S[i].append(j)
                elif dominates(F[j], F[i]):
                    cnt[i] += 1
        if cnt[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        k += 1; fronts.append(nxt)
    return [f for f in fronts if f]


def niche_select(F, chosen, last, dirs, n_pick, rng):
    """Associate normalised points with reference directions; fill from the last front
    preferring directions with the fewest members among the already chosen."""
    if n_pick <= 0:
        return []
    ideal = F.min(0); span = np.where(F.max(0) - ideal > 1e-12, F.max(0) - ideal, 1.0)
    N = (F - ideal) / span
    def assoc(i):
        d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-12)
        proj = N[i] @ d.T
        perp = np.linalg.norm(N[i][None] - proj[:, None] * d, axis=1)
        j = int(perp.argmin()); return j, float(perp[j])
    counts = np.zeros(len(dirs), dtype=int)
    for i in chosen:
        counts[assoc(i)[0]] += 1
    pool = {i: assoc(i) for i in last}
    picked = []
    while len(picked) < n_pick and pool:
        j = int(np.flatnonzero(counts == counts.min())[rng.integers((counts == counts.min()).sum())])
        members = [i for i, (jj, _) in pool.items() if jj == j]
        if not members:
            counts[j] = 10 ** 9; continue
        i = min(members, key=lambda i: pool[i][1]) if counts[j] == 0 else members[int(rng.integers(len(members)))]
        picked.append(i); counts[j] += 1; pool.pop(i)
    return picked


class NSGA3:
    def __init__(self, grammar, rng, pop_size=None, partitions=2):
        self.G, self.rng = grammar, rng
        self.dirs = das_dennis(len(OBJECTIVES), partitions)
        self.pop_size = pop_size or len(self.dirs)
        self.pop = []                     # list of (genome, objective vector, row)

    def ask(self, n=None):
        n = n or self.pop_size
        if not self.pop:
            return [self.G.random_genome(self.rng, 0.6) for _ in range(n)]
        out = []
        for _ in range(n):
            a = self.pop[int(self.rng.integers(len(self.pop)))][0]
            if len(self.pop) > 1 and self.rng.random() < 0.3:
                b = self.pop[int(self.rng.integers(len(self.pop)))][0]
                out.append(self.G.crossover(a, b, self.rng))
            else:
                out.append(self.G.mutate(a, self.rng))
        return out

    def tell(self, genomes, rows):
        cand = self.pop + [(g, objectives(r), r) for g, r in zip(genomes, rows) if r.get("valid", True)]
        if not cand:
            return
        F = np.stack([c[1] for c in cand])
        fronts = nondominated_sort(F)
        chosen = []
        for f in fronts:
            if len(chosen) + len(f) <= self.pop_size:
                chosen += f
            else:
                chosen += niche_select(F, chosen, f, self.dirs, self.pop_size - len(chosen), self.rng); break
        self.pop = [cand[i] for i in chosen]

    def front(self):
        if not self.pop:
            return []
        F = np.stack([c[1] for c in self.pop]); first = nondominated_sort(F)[0]
        return [self.pop[i] for i in first]
