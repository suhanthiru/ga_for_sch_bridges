"""Nearest-non-bridge substitution: the ablation delta is the fitness of a genome minus
the fitness of the same genome with every bridge component replaced by its nearest
non-bridge neighbour in that slot. The table is keyed by (slot type, role), where a
bridge component declares its role in `axes["role"]`; the freeze refuses a bridge
component that has no entry for a slot it can fill.
"""
from dataclasses import replace

from sb.core.genome import ROOT, Edge, Genome, Node

# (slot type, role) -> (substitute component key, {bridge param: substitute param})
NEAREST_NON_BRIDGE = {
    ("planner", "bridge_mean"): ("planner.spline", {"steps": "n_knots"}),
    ("planner", "bridge_sample"): ("planner.spline", {}),
    ("planner", "bridge_branching"): ("planner.mppi", {}),
    ("augment", "bridge_rollouts"): ("augment.pd_rollouts", {"mult": "mult"}),
    ("data", "bridge_rollouts"): ("data.pd_rollouts", {"mult": "mult"}),
    ("data", "pd_relabel"): ("data.pd_rollouts", {"mult": "mult"}),
    ("value", "bridge_backward_drift"): ("value.distance", {}),
    ("value", "bridge_log_density"): ("value.distance", {}),
    ("seam", "bridge_cloud"): ("seam.waypoint", {}),
    ("controller", "bridge_drift"): ("controller.pd", {}),
    ("controller", "bridge_drift_pd"): ("controller.pd", {}),
    ("noise", "bridge_eps"): ("noise.fixed", {"eps": "sigma"}),
    ("trigger", "bridge_disagreement"): ("trigger.distance", {"thr": "thr"}),
    ("trigger", "bridge_log_density_drop"): ("trigger.distance", {"thr": "thr"}),
    ("safety", "bridge_density_gate"): ("safety.none", {}),
    ("adapt", "bridge_resolve"): ("adapt.none", {}),
    ("estimator", "bridge_belief"): ("estimator.ekf", {}),
    ("reference", "bridge_reference"): ("reference.spline", {}),
}


def role_of(spec):
    return spec.axes.get("role", spec.key.split(".", 1)[-1])


def coverage_check(grammar, table=NEAREST_NON_BRIDGE):
    """Every bridge component must have a substitute for every slot type it fills, and the
    substitute must exist and not be a bridge. Returns the list of problems."""
    out = []
    for s in grammar.registry.values():
        if s.tag != "bridge" or s.disabled:
            continue
        for slot in s.slots:
            ent = table.get((slot, role_of(s)))
            if ent is None:
                out.append(f"{s.key}: no substitute for slot {slot} (role {role_of(s)})"); continue
            sub = grammar.registry.get(ent[0])
            if sub is None:
                out.append(f"{s.key}: substitute {ent[0]} is not registered")
            elif sub.tag == "bridge":
                out.append(f"{s.key}: substitute {ent[0]} is itself a bridge")
            elif slot not in sub.slots:
                out.append(f"{s.key}: substitute {ent[0]} cannot fill {slot}")
    return out


def ablate_bridges(g, grammar, table=NEAREST_NON_BRIDGE):
    """The same genome with every bridge-tagged node replaced by its table entry. A mapped
    parameter is carried over only when the two components declare the *same space* (kind
    and range): a disagreement threshold and a distance threshold share a name and nothing
    else. Every other free parameter of the substitute takes its midpoint, so the
    counterfactual is a fixed, neutral stack rather than a lucky draw (ERRORS 2026-09-14).
    Returns a canonical genome; deterministic; idempotent; identity without a bridge node."""
    nodes, edges = list(g.nodes), list(g.edges)
    changed = False
    for n in list(nodes):
        s = grammar.spec(n.comp)
        if s.tag != "bridge" or n not in nodes:              # dropped with a replaced parent's sub-slots
            continue
        e = next(e for e in edges if e.child == n.nid)
        pc = None if e.parent == ROOT else g.node(e.parent).comp
        slot_type = grammar.slot_spec(pc, e.slot).type
        key, pmap = table[(slot_type, role_of(s))]
        sub = grammar.spec(key)
        old = dict(n.params)
        params = {}
        for k, p in sub.params.items():
            src = next((bk for bk, sk in pmap.items() if sk == k), None)
            v = old.get(src) if src is not None else None
            transfers = v is not None and p.valid(v) and src in s.params and p.same_space(s.params[src])
            params[k] = v if transfers else p.midpoint()
        nodes[nodes.index(n)] = Node(n.nid, key, tuple(sorted(params.items())))
        edges[edges.index(e)] = Edge(e.parent, e.slot, e.child, grammar.innov(pc, e.slot, key))
        bad = {c.child for c in edges if c.parent == n.nid and c.slot not in sub.sub_slots}
        while bad:
            edges = [c for c in edges if c.child not in bad]; nodes = [x for x in nodes if x.nid not in bad]
            bad = {c.child for c in edges if c.parent in bad}
        changed = True
    if not changed:
        return g.canonical(grammar.slot_order)
    out = Genome(tuple(nodes), tuple(edges), g.flags, replace(g.provenance, op="ablate", parent_ids=(g.gid,)), g.grammar_hash)
    out = out.canonical(grammar.slot_order)
    if grammar.has_tag(out, "bridge"):
        return ablate_bridges(out, grammar, table)
    return out


def ablation_sites(g, ab):
    """The node ids where `ab = ablate_bridges(g)` differs from `g`: the substituted nodes.
    Substitution keeps node ids and structure, so the canonical numbering is shared."""
    comps = {n.nid: n.comp for n in g.nodes}
    return [n.nid for n in ab.nodes if comps.get(n.nid) != n.comp]
