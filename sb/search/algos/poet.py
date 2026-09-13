"""POET-style environment / stack coevolution (SEARCH_PLAN 5.7).

Environments are descriptor vectors on the family's bins. Each active environment keeps
its own elite stack; every generation a few environments spawn children by moving one
axis a bin, a child is admitted only if the current best stacks score between 0.2 and
0.8 on it (the minimal criterion, checked on three stacks), and elites are transferred
across environments so a stack evolved on one is tried on the others. At most
`max_active` environments; retired ones stay in the archive with their history. The
output is the edge map: the environments where the best stack sits in the 0.2-0.8 band,
and whether bridge stacks are the ones that pushed past an edge.
"""
from dataclasses import dataclass, field

import numpy as np

from sb.core.genome import Genome, Provenance
from sb.envs.family import AXES, BINS


@dataclass
class EnvGenome:
    desc: list
    parent: int = -1
    born: int = 0
    elite_gid: str = ""
    elite_fitness: float = -np.inf
    elite: object = None
    history: list = field(default_factory=list)
    retired_at: int = -1

    def key(self):
        return tuple(round(x, 3) for x in self.desc)


def perturb_env(desc, rng):
    """Move one axis by one bin (descriptor values are bin indices scaled to [0, 1])."""
    d = list(desc); i = int(rng.integers(len(AXES)))
    n = len(BINS[AXES[i]]); step = 1.0 / max(n - 1, 1)
    d[i] = float(np.clip(d[i] + step * (1 if rng.random() < 0.5 else -1), 0.0, 1.0))
    return d


class POET:
    def __init__(self, grammar, evaluate, seed_stacks, rng, max_active=200, lo=0.2, hi=0.8, children_per_gen=2, log=lambda m: None):
        self.G, self.evaluate, self.rng, self.log = grammar, evaluate, rng, log
        self.max_active, self.lo, self.hi, self.children_per_gen = max_active, lo, hi, children_per_gen
        self.seed_stacks = list(seed_stacks)
        self.active, self.retired, self.gen, self.n_evals = [], [], 0, 0
        self.rows = []

    # --------------------------------------------------------------- score
    def score(self, genome, env, rung=0):
        r = self.evaluate(genome, env.desc, rung, 0, self.G); self.n_evals += 1
        self.rows.append(dict(gid=genome.gid, env=env.key(), gen=self.gen, algorithm="poet", **{k: r[k] for k in ("valid", "fitness", "success")}))
        return r["success"] if r["valid"] else 0.0

    def admit(self, desc):
        """Minimal criterion on three stacks: the current best elites (or seeds)."""
        env = EnvGenome(desc=desc, born=self.gen)
        pool = sorted(self.active, key=lambda e: -e.elite_fitness)[:3]
        stacks = [e.elite for e in pool if e.elite is not None] or self.seed_stacks[:3]
        scores = [self.score(s, env) for s in stacks[:3]]
        ok = any(self.lo <= v <= self.hi for v in scores) and not all(v > self.hi for v in scores) and not all(v < self.lo for v in scores)
        return env, ok, scores

    # ---------------------------------------------------------------- loop
    def start(self, init_desc):
        env = EnvGenome(desc=list(init_desc), born=0)
        for s in self.seed_stacks:
            self._offer(env, s)
        self.active.append(env)

    def _offer(self, env, genome):
        f = self.score(genome, env)
        env.history.append((self.gen, genome.gid, f))
        if f > env.elite_fitness:
            env.elite_fitness, env.elite_gid, env.elite = f, genome.gid, genome
            return True
        return False

    def step(self):
        self.gen += 1
        # 1. optimise each active env's stack by one mutation
        for env in self.active:
            parent = env.elite if env.elite is not None else self.seed_stacks[int(self.rng.integers(len(self.seed_stacks)))]
            child = self.G.mutate(parent, self.rng)
            self._offer(env, child)
        # 2. transfer: every elite is tried on one other environment
        if len(self.active) > 1:
            for env in self.active:
                other = self.active[int(self.rng.integers(len(self.active)))]
                if other is not env and env.elite is not None:
                    self._offer(other, env.elite)
        # 3. spawn children of random active environments under the minimal criterion
        for _ in range(self.children_per_gen):
            if not self.active:
                break
            src = self.active[int(self.rng.integers(len(self.active)))]
            child, ok, scores = self.admit(perturb_env(src.desc, self.rng))
            if ok and child.key() not in {e.key() for e in self.active}:
                child.parent = self.active.index(src)
                self.active.append(child)
                self.log(f"gen {self.gen}: admitted env {child.key()} scores {[round(v, 2) for v in scores]}")
        # 4. retire: keep the population bounded; the oldest fully-solved or unsolvable envs go first
        while len(self.active) > self.max_active:
            e = min(self.active, key=lambda e: (self.lo <= e.elite_fitness <= self.hi, -e.born))
            e.retired_at = self.gen; self.retired.append(e); self.active.remove(e)

    def edge_map(self):
        """Environments where the best stack scores inside the band, with whether that
        stack contains a bridge."""
        out = []
        for e in self.active + self.retired:
            if e.elite is None:
                continue
            out.append(dict(env=e.key(), desc=e.desc, fitness=e.elite_fitness, in_band=self.lo <= e.elite_fitness <= self.hi,
                            has_bridge=self.G.has_tag(e.elite, "bridge"), gid=e.elite_gid, retired=e.retired_at >= 0))
        return out
