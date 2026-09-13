"""Rung promotion (SEARCH_PLAN 2.8): rung 0 for every mutant on the fixed cells; rung 1
for a mutant whose rung-0 fitness is within 0.1 of its cell's elite; rung 2 for elites
that held a cell for at least HOLD generations, and for every elite at the end.
Statistical anomalies (a jump above JUMP over the previous elite) are re-run at rung-1
sample size before being accepted (program section 6)."""
from dataclasses import dataclass

PROMOTE_MARGIN = 0.1
HOLD = 50
JUMP = 0.15


@dataclass
class Decision:
    rung: int
    reason: str


def after_rung0(fitness, elite_fitness):
    if elite_fitness is None or fitness >= elite_fitness - PROMOTE_MARGIN:
        return Decision(1, "within margin of the cell elite" if elite_fitness is not None else "empty cell")
    return Decision(0, "below the promotion margin")


def is_jump(fitness, elite_fitness):
    return elite_fitness is not None and fitness - elite_fitness > JUMP


def due_for_validation(elite_map, gen, final=False):
    """Cells whose elite has held for HOLD generations (or all cells at the end)."""
    return [c for c in elite_map.elite if final or (elite_map.held_for(c, gen) or 0) >= HOLD]
