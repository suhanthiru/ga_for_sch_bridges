# ga_for_sch_bridges

Where does a Schrodinger bridge belong in a robotics control stack, if anywhere?

This repo runs that question as a gated program: first a controlled test of the one
result from the earlier experiments that held up (bridge rollouts as training data for
a diffusion policy), then a costed throughput pilot, then a quality-diversity search
over stack architectures whose output is a map of where the bridge earns its place.

The prior experiments live in the sibling repo `skill_chains`; their tested core is
vendored into `sb/` and not modified there.

## Layout

```
SEARCH_PLAN.md          pre-registration, one section per stage
FINDINGS_generator.md   gate results (tables are generated, not typed)
ERRORS.md PLAN_CHANGES.md IDEAS_NEXT.md EXPLOITS.md INCIDENTS.md   append-only logs
sb/core       SE(2) geometry, reference SDEs, bridge solver
sb/envs       terrain model and the SE(2) chained-skill task
sb/policies   diffusion / flow / BC policies, demos, evaluation
sb/gen        training-data generators (bridge, PD, DART, MPPI, oracle, ...)
sb/stats.py   every reported number comes from here
sb/gate       the generator gate: cells, runner, report
tests/        pytest
bench/        throughput measurements for the pilot
```

## Running

```
pip install -e .[dev]
pytest
python -m sb.cli demos
python -m sb.cli diag relabel
python -m sb.cli gate --seed 0
python -m sb.cli report gate
```
