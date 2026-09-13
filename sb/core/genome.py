"""The genome: a directed graph of typed components in typed slots, with hyperparameters
and variation flags, in one canonical form.

Nodes are renumbered by a depth-first walk from the root in the grammar's slot order, so
two genomes with the same structure and values have the same canonical form whatever
order they were built in. `gid` hashes the canonical form; `sid` hashes it with the
parameter values stripped and is the structure id species and CMA emitters key on.
"""
import hashlib
import json
from dataclasses import dataclass, field, replace

ROOT = "root"


@dataclass(frozen=True)
class Node:
    nid: str
    comp: str
    params: tuple = ()                 # sorted (name, value) pairs


@dataclass(frozen=True)
class Edge:
    parent: str                        # nid or ROOT
    slot: str
    child: str
    innov: int = -1


@dataclass(frozen=True)
class Provenance:
    parent_ids: tuple = ()
    algorithm: str = "seed"
    island: int = 0
    generation: int = 0
    op: str = ""
    warm_start_from: str = ""


@dataclass(frozen=True)
class Genome:
    nodes: tuple = ()
    edges: tuple = ()
    flags: tuple = ()                  # sorted (name, value) pairs
    provenance: Provenance = field(default_factory=Provenance)
    grammar_hash: str = ""

    # -------------------------------------------------------------- queries
    def node(self, nid):
        for n in self.nodes:
            if n.nid == nid:
                return n
        raise KeyError(nid)

    def children(self, parent):
        return sorted((e for e in self.edges if e.parent == parent), key=lambda e: e.slot)

    def child_in(self, parent, slot):
        for e in self.edges:
            if e.parent == parent and e.slot == slot:
                return e.child
        return None

    def params_of(self, nid):
        return dict(self.node(nid).params)

    def flag(self, name, default=None):
        return dict(self.flags).get(name, default)

    # ------------------------------------------------------------ canonical
    def canonical(self, slot_order=None):
        """Renumber nodes by depth-first order from the root, slots in `slot_order`
        (alphabetical if None). Unreachable nodes are dropped."""
        order = {s: i for i, s in enumerate(slot_order)} if slot_order else {}
        key = lambda e: (order.get(e.slot, 10 ** 6), e.slot)
        mapping, seq = {}, []
        stack = [ROOT]
        while stack:
            p = stack.pop()
            kids = sorted((e for e in self.edges if e.parent == p), key=key, reverse=True)
            for e in kids:
                if e.child not in mapping:
                    mapping[e.child] = f"n{len(mapping):02d}"; seq.append(e.child); stack.append(e.child)
        nodes = tuple(replace(self.node(old), nid=mapping[old], params=tuple(sorted(self.node(old).params))) for old in seq)
        edges = tuple(sorted((Edge(mapping.get(e.parent, ROOT) if e.parent != ROOT else ROOT, e.slot, mapping[e.child], e.innov)
                              for e in self.edges if e.child in mapping and (e.parent == ROOT or e.parent in mapping)),
                             key=lambda e: (e.parent, e.slot)))
        return replace(self, nodes=nodes, edges=edges, flags=tuple(sorted(self.flags)))

    def _payload(self, with_params=True):
        return dict(nodes=[[n.nid, n.comp, [[k, _round(v)] for k, v in n.params] if with_params else []] for n in self.nodes],
                    edges=[[e.parent, e.slot, e.child] for e in self.edges], flags=[[k, v] for k, v in self.flags])

    @property
    def gid(self):
        return hashlib.sha256(json.dumps(self._payload(True), sort_keys=True).encode()).hexdigest()[:16]

    @property
    def sid(self):
        return hashlib.sha256(json.dumps(self._payload(False), sort_keys=True).encode()).hexdigest()[:16]

    def innovs(self):
        return frozenset(e.innov for e in self.edges)

    # ---------------------------------------------------------------- json
    def to_json(self):
        return json.dumps(dict(nodes=[[n.nid, n.comp, list(map(list, n.params))] for n in self.nodes],
                               edges=[[e.parent, e.slot, e.child, e.innov] for e in self.edges], flags=list(map(list, self.flags)),
                               provenance=self.provenance.__dict__, grammar_hash=self.grammar_hash), sort_keys=True)

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        return cls(nodes=tuple(Node(n[0], n[1], tuple((k, v) for k, v in n[2])) for n in d["nodes"]),
                   edges=tuple(Edge(*e) for e in d["edges"]), flags=tuple((k, v) for k, v in d["flags"]),
                   provenance=Provenance(**{k: (tuple(v) if k == "parent_ids" else v) for k, v in d["provenance"].items()}),
                   grammar_hash=d["grammar_hash"])

    def dsl(self):
        """S-expression view: (slot (component :param value ... (subslot ...)))."""
        def rec(nid):
            n = self.node(nid)
            parts = [n.comp] + [f":{k} {_round(v)}" for k, v in n.params]
            for e in self.children(nid):
                parts.append(f"({e.slot} {rec(e.child)})")
            return "(" + " ".join(parts) + ")"
        return "(stack " + " ".join(f"({e.slot} {rec(e.child)})" for e in self.children(ROOT)) + ")"


def _round(v):
    return round(v, 6) if isinstance(v, float) else v
