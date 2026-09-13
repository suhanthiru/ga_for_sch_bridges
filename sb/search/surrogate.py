"""Surrogate pre-screening (SEARCH_PLAN 5.10): after enough evaluations, a gradient-
boosted model from (one-hot slot contents + binned parameters + descriptor) to rung-0
fitness ranks candidate mutants; the search evaluates the top few plus one at random.
Accuracy on a held-out fifth is logged at every retrain, and a diversity guard rejects a
pre-screened pick when its structure already fills most of the recent batch."""
import numpy as np
import pandas as pd

from sb.core.genome import ROOT, Genome
from sb.envs.family import AXES

MIN_ROWS = 2000
RETRAIN_EVERY = 500


class Surrogate:
    def __init__(self, grammar, min_rows=MIN_ROWS, retrain_every=RETRAIN_EVERY, seed=0):
        self.G, self.min_rows, self.retrain_every, self.seed = grammar, min_rows, retrain_every, seed
        self.keys = sorted((slot, s.key) for s in grammar.registry.values() for slot in grammar.root_slots if slot in s.slots)
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.model, self.trained_at, self.r2 = None, 0, float("nan")

    # ------------------------------------------------------------ features
    def features(self, genome, desc):
        x = np.zeros(len(self.keys) + len(AXES) + 2, dtype=float)
        for e in genome.children(ROOT):
            k = (e.slot, genome.node(e.child).comp)
            if k in self.index:
                x[self.index[k]] = 1.0
        x[len(self.keys):len(self.keys) + len(AXES)] = desc
        x[-2] = len(genome.nodes); x[-1] = sum(1 for n in genome.nodes if self.G.spec(n.comp).tag == "bridge")
        return x

    def _frame_features(self, df):
        X = np.stack([self.features(Genome.from_json(r.genome_json), [r[f"d_{a}"] for a in AXES]) for _, r in df.iterrows()])
        return X, df.fitness.values.astype(float)

    # ------------------------------------------------------------- train
    def maybe_train(self, df, n_evals):
        d = df[df.valid & np.isfinite(df.fitness)] if "valid" in df else df
        if len(d) < self.min_rows or (self.model is not None and n_evals - self.trained_at < self.retrain_every):
            return False
        from sklearn.ensemble import HistGradientBoostingRegressor
        X, y = self._frame_features(d)
        rng = np.random.default_rng(self.seed); idx = rng.permutation(len(y)); cut = len(y) // 5
        te, tr = idx[:cut], idx[cut:]
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=self.seed).fit(X[tr], y[tr])
        pred = m.predict(X[te]); ss = ((y[te] - y[te].mean()) ** 2).sum()
        self.r2 = float(1 - ((y[te] - pred) ** 2).sum() / ss) if ss > 0 else float("nan")
        self.model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=self.seed).fit(X, y)
        self.trained_at = n_evals
        return True

    @property
    def ready(self):
        return self.model is not None

    def predict(self, genomes, desc):
        X = np.stack([self.features(g, desc) for g in genomes])
        return self.model.predict(X)

    def prescreen(self, candidates, desc, rng, top=3, random=1, recent_sids=(), guard=0.3):
        """From `candidates` (genomes), return the top-`top` by prediction plus `random` at
        random; a top pick whose structure fills more than `guard` of the recent batch is
        replaced by the next best."""
        if not self.ready or len(candidates) <= top + random:
            return list(candidates)
        pred = self.predict(candidates, desc); order = list(np.argsort(-pred))
        recent = list(recent_sids); n_recent = max(len(recent), 1)
        picked = []
        for i in order:
            sid = candidates[i].sid
            if recent.count(sid) / n_recent > guard and len(order) > top + random:
                continue
            picked.append(i)
            if len(picked) == top:
                break
        rest = [i for i in order if i not in picked]
        picked += [rest[j] for j in rng.choice(len(rest), size=min(random, len(rest)), replace=False)] if rest else []
        return [candidates[i] for i in picked]
