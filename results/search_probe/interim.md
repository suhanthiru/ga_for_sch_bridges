# Search interim report at 16 evaluations

Cells filled: 7 of 8. Mean elite fitness: 0.487. Rows: 58.

## Slot census (fraction of elites)

| slot       |   bridge |       rl |    other |    empty |
|:-----------|---------:|---------:|---------:|---------:|
| manifold   | 0        | 0        | 1        | 0        |
| reference  | 0        | 0        | 0.571429 | 0.428571 |
| planner    | 0        | 0        | 0.857143 | 0.142857 |
| seam       | 0        | 0        | 0.571429 | 0.428571 |
| controller | 0.285714 | 0        | 0.714286 | 0        |
| value      | 0        | 0        | 0.142857 | 0.857143 |
| data       | 0        | 0        | 0.285714 | 0.714286 |
| augment    | 0        | 0        | 0.285714 | 0.714286 |
| trigger    | 0.571429 | 0        | 0.285714 | 0.142857 |
| noise      | 0        | 0.142857 | 0.428571 | 0.428571 |
| time_split | 0        | 0        | 0.571429 | 0.428571 |
| estimator  | 0        | 0        | 0.428571 | 0.571429 |
| safety     | 0        | 0        | 0.428571 | 0.571429 |
| adapt      | 0        | 0        | 0.714286 | 0.285714 |

## Components in elites

| component                   |   elites |
|:----------------------------|---------:|
| planner.spline              |        5 |
| controller.pd               |        5 |
| adapt.none                  |        5 |
| manifold.se2                |        4 |
| trigger.bridge_disagreement |        4 |
| time_split.equal            |        4 |
| manifold.flat               |        3 |
| reference.spline            |        3 |
| estimator.ekf               |        3 |
| seam.marginal_cloud         |        2 |
| data.pd_rollouts            |        2 |
| augment.pd_rollouts         |        2 |
| trigger.distance            |        2 |
| noise.fixed                 |        2 |
| safety.none                 |        2 |
| reference.geodesic          |        1 |
| planner.mppi                |        1 |
| seam.fixed_clock            |        1 |
| seam.waypoint               |        1 |
| controller.grid_bridge      |        1 |
| controller.bridge_drift     |        1 |
| value.distance              |        1 |
| noise.rl                    |        1 |
| noise.per_skill             |        1 |
| safety.clip                 |        1 |

## Negative space (enabled, never in an elite)

- augment.noised
- controller.diffusion
- controller.ppo
- controller.rl_residual
- data.dart
- data.demos
- data.gc_diffusion
- data.mppi_rollouts
- data.oracle_rollouts
- data.pd_relabel
- reference.nearest_demo

## Algorithm census

| algorithm   |   evaluations |    valid |   mean_fitness |   cells_first_filled |
|:------------|--------------:|---------:|---------------:|---------------------:|
| random      |            58 | 0.982759 |       0.424156 |                    7 |

## Bridge contribution by slot (ablation delta)

| slot       |   n |   mean_delta |
|:-----------|----:|-------------:|
| adapt      |   6 | -0.196291    |
| controller |  10 | -0.11774     |
| data       |   2 | -5.16755e-08 |
| estimator  |   4 |  0.00221872  |
| manifold   |  10 | -0.11774     |
| noise      |   6 | -0.115998    |
| planner    |  10 | -0.11774     |
| reference  |   8 | -0.147176    |
| safety     |   4 |  0.00230435  |
| seam       |   8 | -0.0590252   |
| time_split |   6 | -0.0787003   |
| trigger    |  10 | -0.11774     |
| value      |   2 | -0.352601    |

## Novelty levels

|   novelty_level |   rows |
|----------------:|-------:|
|               0 |     58 |

## Map

![map](results/search_probe/map.png)
