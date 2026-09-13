"""The grammar: root slots, the registry, an optional filter (the no-bridge / no-RL
controls), validation, random valid genomes, mutation, and the freeze manifest.
"""
import hashlib
import json
from dataclasses import replace

import numpy as np

from sb.core.genome import ROOT, Edge, Genome, Node, Provenance
from sb.core.registry import REGISTRY, SlotSpec

# the stack's slots in canonical order; controller and manifold are required
ROOT_SLOTS = {
    "manifold": SlotSpec("manifold", optional=False),
    "reference": SlotSpec("reference"),
    "planner": SlotSpec("planner"),
    "seam": SlotSpec("seam"),
    "controller": SlotSpec("controller", optional=False),
    "value": SlotSpec("value"),
    "data": SlotSpec("data"),
    "augment": SlotSpec("augment"),
    "trigger": SlotSpec("trigger"),
    "noise": SlotSpec("noise"),
    "time_split": SlotSpec("time_split"),
    "estimator": SlotSpec("estimator"),
    "safety": SlotSpec("safety"),
    "adapt": SlotSpec("adapt"),
}
ORACLE_SLOTS = ("data", "value")
FLAGS = {"time_reversed": (False, True), "drift_blend": (False, True), "density_gating": (False, True),
         "marginal_annealing": (False, True), "cost_in": ("objective", "reference"), "per_skill_eps": (False, True),
         "bridge_of_bridges": (False, True), "sampling": ("sde", "ode"), "learned_intermediates": (False, True),
         "action_space_bridge": (False, True)}
MAX_DEPTH = 6

NO_BRIDGE = lambda spec: spec.tag != "bridge"
NO_RL = lambda spec: spec.tag != "rl"


class Grammar:
    def __init__(self, registry=None, root_slots=ROOT_SLOTS, filt=None, flags=FLAGS):
        self.registry = dict(registry if registry is not None else REGISTRY)
        self.root_slots, self.filt, self.flags = dict(root_slots), filt, dict(flags)
        self.slot_order = list(root_slots)
        self._innov = None

    # ------------------------------------------------------------- options
    def options(self, slot_type):
        return [s for s in self.registry.values() if slot_type in s.slots and not s.disabled and (self.filt is None or self.filt(s))]

    def spec(self, key):
        return self.registry[key]

    def slot_spec(self, parent_comp, slot):
        if parent_comp is None:
            return self.root_slots.get(slot)
        return self.spec(parent_comp).sub_slots.get(slot)

    # ------------------------------------------------------------ innovation
    def innovation_table(self):
        if self._innov is None:
            triples = set()
            for s in self.registry.values():
                if s.disabled:
                    continue
                for slot in [sl for sl in self.root_slots if s.key in {o.key for o in self.options(sl)}]:
                    triples.add((ROOT, slot, s.key))
                for par in self.registry.values():
                    for name, sub in par.sub_slots.items():
                        if sub.type in s.slots:
                            triples.add((par.key, name, s.key))
            self._innov = {t: i for i, t in enumerate(sorted(triples))}
        return self._innov

    def innov(self, parent_comp, slot, child_comp):
        return self.innovation_table().get((ROOT if parent_comp is None else parent_comp, slot, child_comp), -1)

    def manifest(self):
        m = dict(slots={k: [v.type, v.optional] for k, v in self.root_slots.items()}, flags={k: list(v) for k, v in self.flags.items()},
                 components=[s.manifest() for s in sorted(self.registry.values(), key=lambda s: s.key)],
                 innovation=[[list(k), v] for k, v in sorted(self.innovation_table().items(), key=lambda kv: kv[1])],
                 filter=getattr(self.filt, "__name__", "none") if self.filt else "none")
        return m

    @property
    def hash(self):
        return hashlib.sha256(json.dumps(self.manifest(), sort_keys=True).encode()).hexdigest()[:16]

    # ------------------------------------------------------------- validate
    def validate(self, g):
        v = []
        nids = {n.nid for n in g.nodes}
        parent_comp = {e.child: (None if e.parent == ROOT else g.node(e.parent).comp) for e in g.edges if e.parent == ROOT or e.parent in nids}
        for n in g.nodes:
            if n.comp not in self.registry:
                v.append(f"unknown component {n.comp}"); continue
            s = self.spec(n.comp)
            if s.disabled:
                v.append(f"{n.comp} is disabled: {s.disabled}")
            if self.filt is not None and not self.filt(s):
                v.append(f"{n.comp} excluded by the grammar filter")
            for k, val in n.params:
                if k not in s.params:
                    v.append(f"{n.comp} has no parameter {k}")
                elif not s.params[k].valid(val):
                    v.append(f"{n.comp}.{k}={val} out of range")
            for k in s.params:
                if k not in dict(n.params):
                    v.append(f"{n.comp} missing parameter {k}")
        seen = set()
        for e in g.edges:
            if e.child not in nids:
                v.append(f"edge to unknown node {e.child}"); continue
            if (e.parent, e.slot) in seen:
                v.append(f"slot {e.slot} of {e.parent} filled twice")
            seen.add((e.parent, e.slot))
            pc = None if e.parent == ROOT else (g.node(e.parent).comp if e.parent in nids else None)
            if e.parent != ROOT and e.parent not in nids:
                v.append(f"edge from unknown node {e.parent}"); continue
            ss = self.slot_spec(pc, e.slot)
            if ss is None:
                v.append(f"{pc or ROOT} has no slot {e.slot}"); continue
            cs = self.registry.get(g.node(e.child).comp)
            if cs is not None and ss.type not in cs.slots:
                v.append(f"{cs.key} cannot fill slot {e.slot} ({ss.type})")
            if cs is not None and cs.oracle != "none" and e.parent == ROOT and e.slot not in ORACLE_SLOTS:
                v.append(f"{cs.key} reads the oracle and sits outside the data/value slots")
        for slot, ss in self.root_slots.items():
            if not ss.optional and g.child_in(ROOT, slot) is None:
                v.append(f"required slot {slot} is empty")
        for n in g.nodes:
            if n.nid not in parent_comp:
                v.append(f"node {n.nid} is unreachable")
            s = self.registry.get(n.comp)
            if s is not None:
                for name, sub in s.sub_slots.items():
                    if not sub.optional and g.child_in(n.nid, name) is None:
                        v.append(f"required sub-slot {name} of {n.comp} is empty")
        if self.depth(g) > MAX_DEPTH:
            v.append(f"depth {self.depth(g)} exceeds {MAX_DEPTH}")
        for k, val in g.flags:
            if k not in self.flags or val not in self.flags[k]:
                v.append(f"bad flag {k}={val}")
        return v

    def level_of(self, g, nid):
        lvl, cur = 0, nid
        while cur != ROOT:
            cur = next(e.parent for e in g.edges if e.child == cur); lvl += 1
        return lvl

    def depth(self, g, nid=ROOT, d=0):
        kids = g.children(nid)
        return d if not kids else max(self.depth(g, e.child, d + 1) for e in kids)

    # --------------------------------------------------------------- build
    def _fill(self, nodes, edges, parent_nid, parent_comp, slot, ss, rng, depth, p_fill):
        """Place a component in `slot` at level depth + 1 (root children are level 1)."""
        level = depth + 1
        opts = self.options(ss.type)
        if level >= MAX_DEPTH:                      # the last level: only components with no required sub-slots
            opts = [o for o in opts if not any(not sub.optional for sub in o.sub_slots.values())]
        if not opts or level > MAX_DEPTH or (ss.optional and (level >= MAX_DEPTH or rng.random() > p_fill)):
            return
        s = opts[int(rng.integers(len(opts)))]
        nid = f"t{len(nodes)}"
        nodes.append(Node(nid, s.key, tuple(sorted((k, p.sample(rng)) for k, p in s.params.items()))))
        edges.append(Edge(parent_nid, slot, nid, self.innov(parent_comp, slot, s.key)))
        for name, sub in s.sub_slots.items():
            self._fill(nodes, edges, nid, s.key, name, sub, rng, depth + 1, p_fill)

    def random_genome(self, rng, p_fill=0.5, provenance=None):
        nodes, edges = [], []
        for slot, ss in self.root_slots.items():
            self._fill(nodes, edges, ROOT, None, slot, ss, rng, 0, p_fill)
        flags = tuple((k, vals[int(rng.integers(len(vals)))]) for k, vals in self.flags.items())
        g = Genome(tuple(nodes), tuple(edges), flags, provenance or Provenance(algorithm="random"), self.hash)
        return g.canonical(self.slot_order)

    # -------------------------------------------------------------- mutate
    def mutate(self, g, rng, op=None):
        """One of: replace a node's component (same slot type), perturb one parameter,
        fill an empty slot, empty an optional slot, flip a flag. Returns a canonical genome."""
        ops = ["replace", "param", "fill", "empty", "flag"]
        op = op or ops[int(rng.integers(len(ops)))]
        nodes, edges, flags = list(g.nodes), list(g.edges), dict(g.flags)
        if op == "param" and nodes:
            n = nodes[int(rng.integers(len(nodes)))]; s = self.spec(n.comp)
            if s.params:
                k = list(s.params)[int(rng.integers(len(s.params)))]
                pr = dict(n.params); pr[k] = s.params[k].mutate(pr[k], rng)
                nodes[nodes.index(n)] = replace(n, params=tuple(sorted(pr.items())))
        elif op == "replace" and nodes:
            e = edges[int(rng.integers(len(edges)))]
            pc = None if e.parent == ROOT else g.node(e.parent).comp
            ss = self.slot_spec(pc, e.slot); opts = [o for o in self.options(ss.type) if o.key != g.node(e.child).comp]
            if opts:
                s = opts[int(rng.integers(len(opts)))]
                old = g.node(e.child)
                keep = {k: v for k, v in old.params if k in s.params and s.params[k].valid(v)}
                params = tuple(sorted((k, keep.get(k, p.sample(rng))) for k, p in s.params.items()))
                nodes[nodes.index(old)] = Node(old.nid, s.key, params)
                edges[edges.index(e)] = replace(e, innov=self.innov(pc, e.slot, s.key))
                # drop sub-slot children the new component does not expose, fill the required ones it adds
                bad = {c.child for c in edges if c.parent == old.nid and c.slot not in s.sub_slots}
                edges = [c for c in edges if c.child not in bad]; nodes = [n for n in nodes if n.nid not in bad]
                lvl = self.level_of(g, old.nid)
                for name, sub in s.sub_slots.items():
                    if not sub.optional and not any(c.parent == old.nid and c.slot == name for c in edges):
                        self._fill(nodes, edges, old.nid, s.key, name, sub, rng, lvl, 1.0)
        elif op == "fill":
            empties = [(ROOT, None, slot, ss) for slot, ss in self.root_slots.items() if g.child_in(ROOT, slot) is None]
            for n in nodes:
                for name, sub in self.spec(n.comp).sub_slots.items():
                    if g.child_in(n.nid, name) is None:
                        empties.append((n.nid, n.comp, name, sub))
            if empties:
                pn, pc, slot, ss = empties[int(rng.integers(len(empties)))]
                self._fill(nodes, edges, pn, pc, slot, ss, rng, 0 if pn == ROOT else self.level_of(g, pn), 1.0)
        elif op == "empty":
            cands = [e for e in edges if (self.slot_spec(None if e.parent == ROOT else g.node(e.parent).comp, e.slot) or SlotSpec("x")).optional]
            if cands:
                e = cands[int(rng.integers(len(cands)))]
                edges.remove(e)
        elif op == "flag":
            k = list(self.flags)[int(rng.integers(len(self.flags)))]
            vals = [v for v in self.flags[k] if v != flags.get(k)]
            flags[k] = vals[int(rng.integers(len(vals)))]
        prov = replace(g.provenance, parent_ids=(g.gid,), op=op)
        out = Genome(tuple(nodes), tuple(edges), tuple(sorted(flags.items())), prov, self.hash)
        return out.canonical(self.slot_order)

    def crossover(self, a, b, rng):
        """Uniform over root slots: each slot's subtree comes from one parent."""
        nodes, edges = [], []
        for slot in self.root_slots:
            src = a if rng.random() < 0.5 else b
            c = src.child_in(ROOT, slot)
            if c is None:
                continue
            self._copy_subtree(src, c, ROOT, slot, nodes, edges, f"{slot}_")
        prov = Provenance(parent_ids=(a.gid, b.gid), op="xover")
        out = Genome(tuple(nodes), tuple(edges), a.flags if rng.random() < 0.5 else b.flags, prov, self.hash)
        return out.canonical(self.slot_order)

    def _copy_subtree(self, src, nid, parent, slot, nodes, edges, prefix):
        n = src.node(nid); new = prefix + nid
        nodes.append(replace(n, nid=new))
        e = next(e for e in src.edges if e.child == nid)
        edges.append(Edge(parent, slot, new, e.innov))
        for c in src.children(nid):
            self._copy_subtree(src, c.child, new, c.slot, nodes, edges, prefix)

    def has_tag(self, g, tag):
        return any(self.spec(n.comp).tag == tag for n in g.nodes)
