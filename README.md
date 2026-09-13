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
pytest                                   # fast suite; add -m slow for the oracle gates
python -m sb.cli demos                   # regenerate the demo sets (sha256 in SEARCH_PLAN)
python -m sb.cli gate --seed 0           # gate cells for one seed (see SEARCH_PLAN 0)
python -m sb.cli report gate             # tables and FINDINGS_generator.md
bash bench/run_all.sh                    # throughput pilot on a quiet GPU; then bench/report.py
python -m sb.cli audit                   # every family's oracle against a 10x planner
python -m sb.cli freeze --pilot          # component tests, then grammar_pilot/manifest.json
python search.py --pilot --budget 1000 --out results/search/pilot_t1   # a search tranche
python -m sb.cli report search --pilot --root results/search/pilot_t1  # interim report
```

## Where things stand

- Gate (SEARCH_PLAN 0): verdict "neither" on seeds 0-4 and on the replication; the
  bridge's edge as a data generator is its states, not its labels (FINDINGS_generator.md).
- Pilot bench (0.4): environment step and grid bridges pass by orders of magnitude;
  neural per-mutant bridges and population PPO are pruned per the rule (PLAN_CHANGES).
- Environment family E1-E8 behind one interface, every oracle audited; grammar frozen
  for the pilot; search tranche 1 running.
