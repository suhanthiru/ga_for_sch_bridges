# Search interim report at 16 evaluations

Cells filled: 2 of 64. Mean elite fitness: 0.428. Rows: 34.

## Slot census (fraction of elites)

| slot       |   bridge |   rl |   other |   empty |
|:-----------|---------:|-----:|--------:|--------:|
| manifold   |        0 |    0 |     1   |     0   |
| reference  |        0 |    0 |     0   |     1   |
| planner    |        0 |    0 |     0.5 |     0.5 |
| seam       |        0 |    0 |     0   |     1   |
| controller |        0 |    0 |     1   |     0   |
| value      |        0 |    0 |     0   |     1   |
| data       |        0 |    0 |     0.5 |     0.5 |
| augment    |        0 |    0 |     1   |     0   |
| trigger    |        0 |    0 |     0.5 |     0.5 |
| noise      |        0 |    0 |     0.5 |     0.5 |
| time_split |        0 |    0 |     0.5 |     0.5 |
| estimator  |        0 |    0 |     0.5 |     0.5 |
| safety     |        0 |    0 |     0.5 |     0.5 |
| adapt      |        0 |    0 |     1   |     0   |

## Components in elites

| component           |   elites |
|:--------------------|---------:|
| controller.pd       |        2 |
| augment.pd_rollouts |        2 |
| adapt.none          |        2 |
| manifold.flat       |        1 |
| manifold.se2        |        1 |
| planner.spline      |        1 |
| data.pd_rollouts    |        1 |
| trigger.distance    |        1 |
| noise.fixed         |        1 |
| time_split.equal    |        1 |
| estimator.ekf       |        1 |
| safety.none         |        1 |

## Negative space (enabled, never in an elite)

- augment.noised
- controller.bridge_drift
- controller.diffusion
- controller.grid_bridge
- controller.ppo
- controller.rl_residual
- data.dart
- data.demos
- data.gc_diffusion
- data.mppi_rollouts
- data.oracle_rollouts
- data.pd_relabel
- noise.per_skill
- noise.rl
- planner.mppi
- reference.geodesic
- reference.nearest_demo
- reference.spline
- safety.clip
- seam.fixed_clock
- seam.marginal_cloud
- seam.waypoint
- trigger.bridge_disagreement
- value.distance

## Algorithm census

| algorithm   |   evaluations |    valid |   mean_fitness |   cells_first_filled |
|:------------|--------------:|---------:|---------------:|---------------------:|
| random      |            34 | 0.970588 |       0.439436 |                    2 |

## Bridge contribution by slot (ablation delta)

no rung-2 ablations yet

## Novelty levels

|   novelty_level |   rows |
|----------------:|-------:|
|               0 |     34 |

## Map

![map](results/search_probe/map.png)
