"""CMA-ES over the parameter vector of one structure (sid): the continuous half of
CMA-MAE. One emitter per elite structure; ask() returns genomes decoded from CMA samples
clipped to the bounds, tell() feeds back the improvement over the cell's elite (the
"imp" ranker of CMA-MAE: a sample that does not improve its cell counts as the worst).
Structures with no continuous parameters get no emitter."""
import numpy as np

from sb.core.genome import Provenance
from sb.core.views import ParamView

try:
    import cma
except ImportError:                                     # pragma: no cover
    cma = None


class CMAEmitter:
    def __init__(self, genome, grammar, sigma0=0.3, seed=0, popsize=None):
        self.view = ParamView(genome, grammar); self.sid = genome.sid; self.G = grammar
        lo, hi = self.view.bounds()
        self.lo, self.hi = lo, hi
        x0 = np.clip(self.view.vector(), lo, hi)
        scale = np.where(hi > lo, hi - lo, 1.0)
        self.scale, self.x0 = scale, x0
        opts = dict(seed=seed + 1, verbose=-9, bounds=[list((lo - x0) / scale), list((hi - x0) / scale)])
        if popsize:
            opts["popsize"] = popsize
        # cma needs at least two dimensions; one-parameter structures are left to the graph operators
        self.es = cma.CMAEvolutionStrategy(np.zeros(self.view.dim), sigma0, opts) if (cma is not None and self.view.dim >= 2) else None
        self._pending = None

    @property
    def active(self):
        return self.es is not None and not self.es.stop()

    def ask(self, gen=0):
        if not self.active:
            return []
        zs = self.es.ask(); self._pending = zs
        genomes = []
        for z in zs:
            x = np.clip(self.x0 + np.asarray(z) * self.scale, self.lo, self.hi)
            genomes.append(self.view.apply(x, Provenance(parent_ids=(self.view.g.gid,), algorithm="cma_mae", generation=gen, op="cma")))
        return genomes

    def tell(self, fitnesses, elite_fitness):
        """fitnesses: one per genome from ask(), in order. Improvement ranking: the objective
        CMA minimises is -(f - elite) for improvers and a large constant for the rest."""
        if self._pending is None:
            return
        base = elite_fitness if elite_fitness is not None else -np.inf
        costs = [(-(f - base) if f > base else 1.0 + (base - f if np.isfinite(base) else 0.0)) for f in fitnesses]
        self.es.tell(self._pending, costs); self._pending = None
