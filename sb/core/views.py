"""ParamView: the continuous parameters of a genome with a fixed structure as one
vector, in the space each parameter is mutated in (log10 for log-uniform, raw for
linear and integer, none for choices), with bounds; and the inverse. CMA emitters and
the surrogate work on this vector; structure moves stay with the grammar's graph ops."""
import math
from dataclasses import replace

import numpy as np

from sb.core.genome import Genome, Node


class ParamView:
    def __init__(self, genome, grammar):
        self.g, self.G = genome, grammar
        self.keys = []                                  # (nid, name, spec)
        for n in genome.nodes:
            spec = grammar.spec(n.comp)
            for name, p in sorted(spec.params.items()):
                if p.kind in ("log", "lin", "int"):
                    self.keys.append((n.nid, name, p))
        self.dim = len(self.keys)

    # ------------------------------------------------------------ encode
    @staticmethod
    def _enc(p, v):
        return math.log10(v) if p.kind == "log" else float(v)

    @staticmethod
    def _dec(p, x):
        if p.kind == "log":
            return float(p.clip(10 ** x))
        if p.kind == "int":
            return int(p.clip(int(round(x))))
        return float(p.clip(x))

    def vector(self, genome=None):
        g = genome or self.g
        return np.array([self._enc(p, dict(g.node(nid).params)[name]) for nid, name, p in self.keys], dtype=float)

    def bounds(self):
        lo = np.array([math.log10(p.lo) if p.kind == "log" else p.lo for _, _, p in self.keys], dtype=float)
        hi = np.array([math.log10(p.hi) if p.kind == "log" else p.hi for _, _, p in self.keys], dtype=float)
        return lo, hi

    def apply(self, x, provenance=None):
        """A genome with the same structure and the vector's values."""
        vals = {}
        for (nid, name, p), xi in zip(self.keys, x):
            vals.setdefault(nid, {})[name] = self._dec(p, float(xi))
        nodes = []
        for n in self.g.nodes:
            pr = dict(n.params); pr.update(vals.get(n.nid, {}))
            nodes.append(Node(n.nid, n.comp, tuple(sorted(pr.items()))))
        out = Genome(tuple(nodes), self.g.edges, self.g.flags, provenance or self.g.provenance, self.g.grammar_hash)
        return out.canonical(self.G.slot_order)
