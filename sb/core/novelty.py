"""How new an architecture is relative to the seed population.

level 0: structure id seen among the seeds
level 1: new structure, but every slot-edge (innovation) appeared in some seed
level 2: contains a slot-edge no seed had
level 3: level 2 and at least `far` innovations away from every seed (symmetric difference)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedSet:
    sids: frozenset
    innovs: frozenset
    per_seed: tuple                    # tuple of frozensets, one per seed genome

    @classmethod
    def from_genomes(cls, seeds):
        return cls(frozenset(g.sid for g in seeds), frozenset().union(*(g.innovs() for g in seeds)) if seeds else frozenset(),
                   tuple(g.innovs() for g in seeds))

    def to_json(self):
        return dict(sids=sorted(self.sids), innovs=sorted(self.innovs), per_seed=[sorted(s) for s in self.per_seed])

    @classmethod
    def from_json(cls, d):
        return cls(frozenset(d["sids"]), frozenset(d["innovs"]), tuple(frozenset(s) for s in d["per_seed"]))


def novelty(g, seeds, far=3):
    """Returns (level, distance to the nearest seed in innovation-set symmetric difference)."""
    inn = g.innovs()
    dist = min((len(inn ^ s) for s in seeds.per_seed), default=len(inn))
    if g.sid in seeds.sids:
        return 0, dist
    if inn <= seeds.innovs:
        return 1, dist
    return (3 if dist >= far else 2), dist
