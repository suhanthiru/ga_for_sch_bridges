# Generator gate: findings

Status: final. Seeds 0-4 and the same-machine replication (seeds 5-9) both give the same verdict. Commit: a80c89b. Pre-registration:
SEARCH_PLAN.md section 0 at the commit that introduced it.

Every table below is written by `python -m sb.cli report gate` from `sb/stats.py`.
Nothing between the table markers is edited by hand.

## 1. Setup

SEARCH_PLAN section 0 fixes everything below. Eleven training-data sources for the same
diffusion policy (plus the two sources the 0.5 fallback added), two layouts, four
evaluation disturbances, seeds 0-4. The cells the prior run in skill_chains already had
(g0 on both layouts, g1 on L1) are read from its shards; the rest were run here. Every
in-family source is rolled out on the reference kinematics under one shared noise stream
per seed, from the same starting poses, so the sources differ only in the controller
that produced the states and labels. Tables are generated from `sb/stats.py`; the prose
here only reads them.

## 2. Dynamic-range check

<!-- tables:dr -->
| quantity | value |
|---|---|
| cells_at_floor | 0 |
| n_cells | 6 |
| design_has_range | 1 |

| layout | disturbance | n | mean | 95% CI |
|---|---|---|---|---|
| L1 | push | 5 | 0.513 | [-, -] |
| L1 | rain | 5 | 0.350 | [-, -] |
| L1 | slip | 5 | 0.745 | [-, -] |
| L2 | push | 5 | 0.152 | [-, -] |
| L2 | rain | 5 | 0.193 | [-, -] |
| L2 | slip | 5 | 0.444 | [-, -] |
<!-- /tables:dr -->

## 3. MPC-relabel diagnostic

<!-- tables:diag -->
| quantity | value |
|---|---|
| cost | greedy |
| speed_ratio | 2.833 |
| demo_speed | 0.254 |
| relabel_speed | 0.720 |
| consistency_vx | 0.190 |
| consistency_vy | 1.346 |
| consistency_w | 1.376 |
| collision | 1.000 |
| success_none | 0.000 |
| pass | 0 |
| seconds | 56.600 |
<!-- /tables:diag -->

## 4. Success by source, layout and disturbance

<!-- tables:cells -->
| source | layout | disturbance | n | mean | 95% CI |
|---|---|---|---|---|---|
| DEMO | L1 | none | 5 | 0.894 | [0.772, 1.016] |
| DEMO | L1 | slip | 5 | 0.174 | [0.110, 0.238] |
| DEMO | L1 | rain | 5 | 0.081 | [0.066, 0.096] |
| DEMO | L1 | push | 5 | 0.068 | [0.049, 0.087] |
| NOISED | L1 | none | 5 | 0.739 | [0.529, 0.949] |
| NOISED | L1 | slip | 5 | 0.100 | [0.082, 0.118] |
| NOISED | L1 | rain | 5 | 0.054 | [0.030, 0.078] |
| NOISED | L1 | push | 5 | 0.028 | [0.009, 0.047] |
| BRIDGE-slip | L1 | none | 5 | 0.993 | [0.981, 1.005] |
| BRIDGE-slip | L1 | slip | 5 | 0.505 | [0.470, 0.540] |
| BRIDGE-slip | L1 | rain | 5 | 0.183 | [0.134, 0.232] |
| BRIDGE-slip | L1 | push | 5 | 0.196 | [0.168, 0.224] |
| PD-noise | L1 | none | 5 | 0.988 | [0.974, 1.002] |
| PD-noise | L1 | slip | 5 | 0.618 | [0.496, 0.740] |
| PD-noise | L1 | rain | 5 | 0.295 | [0.260, 0.330] |
| PD-noise | L1 | push | 5 | 0.513 | [0.447, 0.579] |
| DART | L1 | none | 5 | 0.994 | [0.987, 1.001] |
| DART | L1 | slip | 5 | 0.745 | [0.698, 0.792] |
| DART | L1 | rain | 5 | 0.331 | [0.270, 0.392] |
| DART | L1 | push | 5 | 0.476 | [0.419, 0.533] |
| MPPI-rollouts | L1 | none | 5 | 0.990 | [0.980, 1.000] |
| MPPI-rollouts | L1 | slip | 5 | 0.686 | [0.655, 0.717] |
| MPPI-rollouts | L1 | rain | 5 | 0.350 | [0.276, 0.424] |
| MPPI-rollouts | L1 | push | 5 | 0.493 | [0.455, 0.531] |
| DEMO | L2 | none | 5 | 0.329 | [0.212, 0.446] |
| DEMO | L2 | slip | 5 | 0.059 | [0.028, 0.090] |
| DEMO | L2 | rain | 5 | 0.029 | [0.013, 0.045] |
| DEMO | L2 | push | 5 | 0.026 | [0.003, 0.049] |
| NOISED | L2 | none | 5 | 0.357 | [0.217, 0.497] |
| NOISED | L2 | slip | 5 | 0.054 | [0.011, 0.097] |
| NOISED | L2 | rain | 5 | 0.033 | [0.024, 0.042] |
| NOISED | L2 | push | 5 | 0.013 | [0.003, 0.023] |
| BRIDGE-slip | L2 | none | 5 | 0.944 | [0.880, 1.008] |
| BRIDGE-slip | L2 | slip | 5 | 0.318 | [0.280, 0.356] |
| BRIDGE-slip | L2 | rain | 5 | 0.112 | [0.084, 0.140] |
| BRIDGE-slip | L2 | push | 5 | 0.076 | [0.045, 0.107] |
| PD-noise | L2 | none | 5 | 0.697 | [0.463, 0.931] |
| PD-noise | L2 | slip | 5 | 0.333 | [0.185, 0.481] |
| PD-noise | L2 | rain | 5 | 0.151 | [0.091, 0.211] |
| PD-noise | L2 | push | 5 | 0.148 | [0.107, 0.189] |
| DART | L2 | none | 5 | 0.910 | [0.773, 1.047] |
| DART | L2 | slip | 5 | 0.444 | [0.327, 0.561] |
| DART | L2 | rain | 5 | 0.172 | [0.114, 0.230] |
| DART | L2 | push | 5 | 0.152 | [0.074, 0.230] |
| MPPI-rollouts | L2 | none | 5 | 0.881 | [0.774, 0.988] |
| MPPI-rollouts | L2 | slip | 5 | 0.440 | [0.389, 0.491] |
| MPPI-rollouts | L2 | rain | 5 | 0.193 | [0.130, 0.256] |
| MPPI-rollouts | L2 | push | 5 | 0.125 | [0.085, 0.165] |
| BRIDGE-brownian | L1 | none | 5 | 0.953 | [0.921, 0.985] |
| BRIDGE-brownian | L1 | slip | 5 | 0.167 | [0.145, 0.189] |
| BRIDGE-brownian | L1 | rain | 5 | 0.078 | [0.064, 0.092] |
| BRIDGE-brownian | L1 | push | 5 | 0.036 | [0.013, 0.059] |
| BRIDGE-unicycle | L1 | none | 5 | 0.949 | [0.924, 0.974] |
| BRIDGE-unicycle | L1 | slip | 5 | 0.158 | [0.096, 0.220] |
| BRIDGE-unicycle | L1 | rain | 5 | 0.055 | [0.031, 0.079] |
| BRIDGE-unicycle | L1 | push | 5 | 0.040 | [0.021, 0.059] |
| PD-iso | L1 | none | 5 | 0.995 | [0.989, 1.001] |
| PD-iso | L1 | slip | 5 | 0.708 | [0.662, 0.754] |
| PD-iso | L1 | rain | 5 | 0.327 | [0.304, 0.350] |
| PD-iso | L1 | push | 5 | 0.536 | [0.499, 0.573] |
| MPC-relabel | L1 | none | 5 | 0.002 | [-0.001, 0.005] |
| MPC-relabel | L1 | slip | 5 | 0.198 | [0.122, 0.274] |
| MPC-relabel | L1 | rain | 5 | 0.226 | [0.174, 0.278] |
| MPC-relabel | L1 | push | 5 | 0.069 | [0.033, 0.105] |
| BRIDGE-on-PD | L1 | none | 5 | 0.944 | [0.897, 0.991] |
| BRIDGE-on-PD | L1 | slip | 5 | 0.291 | [0.242, 0.340] |
| BRIDGE-on-PD | L1 | rain | 5 | 0.123 | [0.099, 0.147] |
| BRIDGE-on-PD | L1 | push | 5 | 0.072 | [0.043, 0.101] |
| BRIDGE-on-PD | L2 | none | 5 | 0.760 | [0.655, 0.865] |
| BRIDGE-on-PD | L2 | slip | 5 | 0.246 | [0.228, 0.264] |
| BRIDGE-on-PD | L2 | rain | 5 | 0.121 | [0.080, 0.162] |
| BRIDGE-on-PD | L2 | push | 5 | 0.023 | [0.012, 0.034] |
| BRIDGE-unicycle | L2 | none | 5 | 0.730 | [0.638, 0.822] |
| BRIDGE-unicycle | L2 | slip | 5 | 0.160 | [0.103, 0.217] |
| BRIDGE-unicycle | L2 | rain | 5 | 0.064 | [0.045, 0.083] |
| BRIDGE-unicycle | L2 | push | 5 | 0.017 | [0.008, 0.026] |
| GC-diff-rollouts | L1 | none | 5 | 0.992 | [0.979, 1.005] |
| GC-diff-rollouts | L1 | slip | 5 | 0.480 | [0.417, 0.543] |
| GC-diff-rollouts | L1 | rain | 5 | 0.240 | [0.217, 0.263] |
| GC-diff-rollouts | L1 | push | 5 | 0.369 | [0.334, 0.404] |
| GC-diff-rollouts | L2 | none | 5 | 0.363 | [0.115, 0.611] |
| GC-diff-rollouts | L2 | slip | 5 | 0.162 | [0.080, 0.244] |
| GC-diff-rollouts | L2 | rain | 5 | 0.072 | [0.029, 0.115] |
| GC-diff-rollouts | L2 | push | 5 | 0.040 | [0.030, 0.050] |
| MPC-oracle | L1 | none | 5 | 0.999 | [0.996, 1.002] |
| MPC-oracle | L1 | slip | 5 | 0.754 | [0.725, 0.783] |
| MPC-oracle | L1 | rain | 5 | 0.392 | [0.304, 0.480] |
| MPC-oracle | L1 | push | 5 | 0.493 | [0.465, 0.521] |
| MPC-oracle | L2 | none | 5 | 0.841 | [0.686, 0.996] |
| MPC-oracle | L2 | slip | 5 | 0.466 | [0.436, 0.496] |
| MPC-oracle | L2 | rain | 5 | 0.215 | [0.149, 0.281] |
| MPC-oracle | L2 | push | 5 | 0.130 | [0.108, 0.152] |
| MPC-relabel | L2 | none | 5 | 0.005 | [-0.004, 0.014] |
| MPC-relabel | L2 | slip | 5 | 0.136 | [0.095, 0.177] |
| MPC-relabel | L2 | rain | 5 | 0.094 | [0.081, 0.107] |
| MPC-relabel | L2 | push | 5 | 0.032 | [0.023, 0.041] |
| PD-iso | L2 | none | 5 | 0.871 | [0.768, 0.974] |
| PD-iso | L2 | slip | 5 | 0.368 | [0.243, 0.493] |
| PD-iso | L2 | rain | 5 | 0.171 | [0.116, 0.226] |
| PD-iso | L2 | push | 5 | 0.175 | [0.108, 0.242] |
| PD-relabel | L1 | none | 5 | 0.996 | [0.988, 1.004] |
| PD-relabel | L1 | slip | 5 | 0.759 | [0.686, 0.832] |
| PD-relabel | L1 | rain | 5 | 0.401 | [0.351, 0.451] |
| PD-relabel | L1 | push | 5 | 0.645 | [0.573, 0.717] |
| PD-relabel | L2 | none | 5 | 0.900 | [0.759, 1.041] |
| PD-relabel | L2 | slip | 5 | 0.385 | [0.301, 0.469] |
| PD-relabel | L2 | rain | 5 | 0.188 | [0.138, 0.238] |
| PD-relabel | L2 | push | 5 | 0.159 | [0.139, 0.179] |
<!-- /tables:cells -->

## 5. Primary contrast F1: BRIDGE-slip minus PD-noise

<!-- tables:F1 -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F1 (BRIDGE-slip - PD-noise) | L1 | push | 5 | -0.317 | [-0.394, -0.240] | [-0.367, -0.270] | 0.000 | 0.002 | null |
| F1 (BRIDGE-slip - PD-noise) | L1 | rain | 5 | -0.112 | [-0.166, -0.058] | [-0.145, -0.078] | 0.005 | 0.014 | null |
| F1 (BRIDGE-slip - PD-noise) | L1 | slip | 5 | -0.113 | [-0.221, -0.005] | [-0.186, -0.050] | 0.044 | 0.066 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | push | 5 | -0.072 | [-0.122, -0.022] | [-0.102, -0.041] | 0.016 | 0.033 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | rain | 5 | -0.039 | [-0.107, 0.029] | [-0.086, -0.004] | 0.185 | 0.222 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | slip | 5 | -0.015 | [-0.151, 0.121] | [-0.097, 0.074] | 0.774 | 0.774 | inconclusive |
<!-- /tables:F1 -->

## 6. Isolations

### 6.1 Reference covariance (F2, F3)

<!-- tables:F2 -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F2 (BRIDGE-slip - BRIDGE-unicycle) | L1 | push | 5 | 0.156 | [0.125, 0.187] | [0.137, 0.178] | 0.000 | 0.001 | pass |
| F2 (BRIDGE-slip - BRIDGE-unicycle) | L1 | rain | 5 | 0.128 | [0.092, 0.164] | [0.104, 0.149] | 0.001 | 0.001 | pass |
| F2 (BRIDGE-slip - BRIDGE-unicycle) | L1 | slip | 5 | 0.347 | [0.264, 0.430] | [0.299, 0.402] | 0.000 | 0.001 | pass |
| F2 (BRIDGE-slip - BRIDGE-unicycle) | L2 | push | 5 | 0.059 | [0.019, 0.099] | [0.035, 0.085] | 0.015 | 0.018 | null |
| F2 (BRIDGE-slip - BRIDGE-unicycle) | L2 | rain | 5 | 0.048 | [0.005, 0.091] | [0.022, 0.075] | 0.035 | 0.035 | null |
| F2 (BRIDGE-slip - BRIDGE-unicycle) | L2 | slip | 5 | 0.158 | [0.100, 0.216] | [0.118, 0.191] | 0.002 | 0.002 | pass |
<!-- /tables:F2 -->

<!-- tables:F3 -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F3 (PD-noise - PD-iso) | L1 | push | 5 | -0.023 | [-0.086, 0.040] | [-0.062, 0.016] | 0.367 | 0.440 | null |
| F3 (PD-noise - PD-iso) | L1 | rain | 5 | -0.032 | [-0.076, 0.012] | [-0.059, -0.004] | 0.116 | 0.404 | null |
| F3 (PD-noise - PD-iso) | L1 | slip | 5 | -0.090 | [-0.230, 0.050] | [-0.176, 0.004] | 0.150 | 0.404 | null |
| F3 (PD-noise - PD-iso) | L2 | push | 5 | -0.027 | [-0.097, 0.043] | [-0.075, 0.006] | 0.345 | 0.440 | null |
| F3 (PD-noise - PD-iso) | L2 | rain | 5 | -0.020 | [-0.088, 0.048] | [-0.058, 0.029] | 0.463 | 0.463 | null |
| F3 (PD-noise - PD-iso) | L2 | slip | 5 | -0.035 | [-0.099, 0.029] | [-0.074, 0.006] | 0.202 | 0.404 | null |
<!-- /tables:F3 -->

### 6.2 Action labels (F4a, F4b; F4 reported, not in the rule)

<!-- tables:F4a -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F4a (PD-relabel - BRIDGE-slip) | L1 | push | 5 | 0.449 | [0.354, 0.544] | [0.382, 0.500] | 0.000 | 0.001 | pass |
| F4a (PD-relabel - BRIDGE-slip) | L1 | rain | 5 | 0.218 | [0.168, 0.268] | [0.188, 0.248] | 0.000 | 0.001 | pass |
| F4a (PD-relabel - BRIDGE-slip) | L1 | slip | 5 | 0.254 | [0.176, 0.332] | [0.205, 0.303] | 0.001 | 0.002 | pass |
| F4a (PD-relabel - BRIDGE-slip) | L2 | push | 5 | 0.083 | [0.041, 0.125] | [0.055, 0.111] | 0.005 | 0.008 | inconclusive |
| F4a (PD-relabel - BRIDGE-slip) | L2 | rain | 5 | 0.076 | [0.025, 0.127] | [0.050, 0.111] | 0.014 | 0.017 | inconclusive |
| F4a (PD-relabel - BRIDGE-slip) | L2 | slip | 5 | 0.067 | [-0.046, 0.180] | [-0.005, 0.141] | 0.176 | 0.176 | inconclusive |
<!-- /tables:F4a -->

<!-- tables:F4b -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F4b (BRIDGE-on-PD - PD-noise) | L1 | push | 5 | -0.441 | [-0.532, -0.350] | [-0.502, -0.389] | 0.000 | 0.001 | null |
| F4b (BRIDGE-on-PD - PD-noise) | L1 | rain | 5 | -0.172 | [-0.214, -0.130] | [-0.198, -0.146] | 0.000 | 0.001 | null |
| F4b (BRIDGE-on-PD - PD-noise) | L1 | slip | 5 | -0.327 | [-0.442, -0.212] | [-0.404, -0.263] | 0.001 | 0.003 | null |
| F4b (BRIDGE-on-PD - PD-noise) | L2 | push | 5 | -0.125 | [-0.173, -0.077] | [-0.157, -0.098] | 0.002 | 0.003 | null |
| F4b (BRIDGE-on-PD - PD-noise) | L2 | rain | 5 | -0.030 | [-0.071, 0.011] | [-0.052, -0.001] | 0.115 | 0.138 | null |
| F4b (BRIDGE-on-PD - PD-noise) | L2 | slip | 5 | -0.087 | [-0.242, 0.068] | [-0.185, 0.011] | 0.195 | 0.195 | null |
<!-- /tables:F4b -->

<!-- tables:F4 -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F4 (MPC-relabel - BRIDGE-slip) | L1 | push | 5 | -0.127 | [-0.159, -0.095] | [-0.144, -0.106] | 0.000 | 0.002 | null |
| F4 (MPC-relabel - BRIDGE-slip) | L1 | rain | 5 | 0.043 | [-0.031, 0.117] | [0.000, 0.092] | 0.181 | 0.181 | inconclusive |
| F4 (MPC-relabel - BRIDGE-slip) | L1 | slip | 5 | -0.307 | [-0.407, -0.207] | [-0.370, -0.244] | 0.001 | 0.003 | null |
| F4 (MPC-relabel - BRIDGE-slip) | L2 | push | 5 | -0.044 | [-0.069, -0.019] | [-0.061, -0.030] | 0.008 | 0.012 | null |
| F4 (MPC-relabel - BRIDGE-slip) | L2 | rain | 5 | -0.018 | [-0.047, 0.011] | [-0.038, -0.001] | 0.163 | 0.181 | null |
| F4 (MPC-relabel - BRIDGE-slip) | L2 | slip | 5 | -0.182 | [-0.247, -0.117] | [-0.222, -0.142] | 0.001 | 0.003 | null |
<!-- /tables:F4 -->

### 6.3 State coverage

<!-- tables:cov -->
| source | layout | n | off-path cells | [off-path fraction, disturbed success] |
|---|---|---|---|---|
| BRIDGE-on-PD | L1 | 5 | 41.000 | [0.004, 0.162] |
| BRIDGE-on-PD | L2 | 5 | 28.200 | [0.003, 0.130] |
| BRIDGE-slip | L1 | 5 | 416.400 | [0.289, 0.295] |
| BRIDGE-slip | L2 | 5 | 388.600 | [0.261, 0.169] |
| BRIDGE-unicycle | L1 | 5 | 374.600 | [0.299, 0.084] |
| BRIDGE-unicycle | L2 | 5 | 367.600 | [0.281, 0.080] |
| DART | L1 | 5 | 28.800 | [0.003, 0.517] |
| DART | L2 | 5 | 25.400 | [0.003, 0.256] |
| GC-diff-rollouts | L1 | 5 | 159.000 | [0.047, 0.363] |
| GC-diff-rollouts | L2 | 5 | 461.000 | [0.481, 0.091] |
| MPC-relabel | L1 | 5 | 421.800 | [0.311, 0.164] |
| MPC-relabel | L2 | 5 | 456.200 | [0.342, 0.087] |
| MPPI-rollouts | L1 | 5 | 177.600 | [0.090, 0.510] |
| MPPI-rollouts | L2 | 5 | 178.600 | [0.105, 0.253] |
| NOISED | L1 | 5 | 290.400 | [0.092, 0.061] |
| NOISED | L2 | 5 | 295.000 | [0.086, 0.033] |
| PD-iso | L1 | 5 | 11.400 | [0.001, 0.524] |
| PD-iso | L2 | 5 | 5.600 | [0.000, 0.238] |
| PD-noise | L1 | 5 | 45.200 | [0.004, 0.475] |
| PD-noise | L2 | 5 | 35.400 | [0.004, 0.211] |
| PD-relabel | L1 | 5 | 472.200 | [0.376, 0.602] |
| PD-relabel | L2 | 5 | 456.200 | [0.342, 0.244] |

Spearman(coverage, disturbed success) across sources within (seed, layout): mean -0.317, 95% CI [-0.421, -0.212], n = 10
<!-- /tables:cov -->

## 7. Upper bound and the other generators (F5, F6, F7)

<!-- tables:F5 -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F5 (MPC-oracle - BRIDGE-slip) | L1 | push | 5 | 0.297 | [0.246, 0.348] | [0.262, 0.324] | 0.000 | 0.001 | pass |
| F5 (MPC-oracle - BRIDGE-slip) | L1 | rain | 5 | 0.209 | [0.134, 0.284] | [0.158, 0.258] | 0.002 | 0.002 | pass |
| F5 (MPC-oracle - BRIDGE-slip) | L1 | slip | 5 | 0.249 | [0.196, 0.302] | [0.212, 0.279] | 0.000 | 0.001 | pass |
| F5 (MPC-oracle - BRIDGE-slip) | L2 | push | 5 | 0.054 | [0.009, 0.099] | [0.024, 0.081] | 0.028 | 0.030 | null |
| F5 (MPC-oracle - BRIDGE-slip) | L2 | rain | 5 | 0.103 | [0.016, 0.190] | [0.048, 0.160] | 0.030 | 0.030 | pass |
| F5 (MPC-oracle - BRIDGE-slip) | L2 | slip | 5 | 0.148 | [0.095, 0.201] | [0.121, 0.185] | 0.001 | 0.002 | pass |
<!-- /tables:F5 -->

<!-- tables:F6 -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F6 (GC-diff-rollouts - PD-noise) | L1 | push | 5 | -0.144 | [-0.208, -0.080] | [-0.185, -0.105] | 0.003 | 0.007 | null |
| F6 (GC-diff-rollouts - PD-noise) | L1 | rain | 5 | -0.055 | [-0.073, -0.037] | [-0.066, -0.044] | 0.001 | 0.004 | null |
| F6 (GC-diff-rollouts - PD-noise) | L1 | slip | 5 | -0.138 | [-0.219, -0.057] | [-0.196, -0.098] | 0.009 | 0.011 | null |
| F6 (GC-diff-rollouts - PD-noise) | L2 | push | 5 | -0.108 | [-0.146, -0.070] | [-0.136, -0.090] | 0.001 | 0.004 | null |
| F6 (GC-diff-rollouts - PD-noise) | L2 | rain | 5 | -0.079 | [-0.125, -0.033] | [-0.105, -0.049] | 0.009 | 0.011 | null |
| F6 (GC-diff-rollouts - PD-noise) | L2 | slip | 5 | -0.171 | [-0.291, -0.051] | [-0.248, -0.092] | 0.017 | 0.017 | null |
<!-- /tables:F6 -->

<!-- tables:F7 -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F7 (MPPI-rollouts - PD-noise) | L1 | push | 5 | -0.020 | [-0.109, 0.069] | [-0.083, 0.032] | 0.568 | 0.568 | null |
| F7 (MPPI-rollouts - PD-noise) | L1 | rain | 5 | 0.055 | [-0.046, 0.156] | [-0.013, 0.119] | 0.205 | 0.409 | inconclusive |
| F7 (MPPI-rollouts - PD-noise) | L1 | slip | 5 | 0.068 | [-0.053, 0.189] | [-0.007, 0.143] | 0.194 | 0.409 | inconclusive |
| F7 (MPPI-rollouts - PD-noise) | L2 | push | 5 | -0.023 | [-0.083, 0.037] | [-0.056, 0.017] | 0.349 | 0.451 | null |
| F7 (MPPI-rollouts - PD-noise) | L2 | rain | 5 | 0.042 | [-0.075, 0.159] | [-0.027, 0.110] | 0.376 | 0.451 | inconclusive |
| F7 (MPPI-rollouts - PD-noise) | L2 | slip | 5 | 0.107 | [-0.054, 0.268] | [0.010, 0.213] | 0.139 | 0.409 | inconclusive |
<!-- /tables:F7 -->

## 8. Tail: CVaR_0.1 of progress

The worst tenth of episodes fails before the first handoff for every source in almost
every disturbed cell, so the paired CVaR deltas are zero with zero-width intervals; the
one cell with any spread (L1 slip) points the same way as the mean. A tail criterion
needs a finer per-episode outcome than handoffs reached; noted in IDEAS_NEXT.

<!-- tables:cvar -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F1-cvar (BRIDGE-slip - PD-noise) | L1 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F1-cvar (BRIDGE-slip - PD-noise) | L1 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F1-cvar (BRIDGE-slip - PD-noise) | L1 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F1-cvar (BRIDGE-slip - PD-noise) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F1-cvar (BRIDGE-slip - PD-noise) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F1-cvar (BRIDGE-slip - PD-noise) | L2 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F2-cvar (BRIDGE-slip - BRIDGE-unicycle) | L1 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F2-cvar (BRIDGE-slip - BRIDGE-unicycle) | L1 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F2-cvar (BRIDGE-slip - BRIDGE-unicycle) | L1 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F2-cvar (BRIDGE-slip - BRIDGE-unicycle) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F2-cvar (BRIDGE-slip - BRIDGE-unicycle) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F2-cvar (BRIDGE-slip - BRIDGE-unicycle) | L2 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F3-cvar (PD-noise - PD-iso) | L1 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F3-cvar (PD-noise - PD-iso) | L1 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F3-cvar (PD-noise - PD-iso) | L1 | slip | 5 | -0.077 | [-0.138, -0.015] | [-0.113, -0.033] | 0.026 | 0.154 | null |
| F3-cvar (PD-noise - PD-iso) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F3-cvar (PD-noise - PD-iso) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F3-cvar (PD-noise - PD-iso) | L2 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4a-cvar (PD-relabel - BRIDGE-slip) | L1 | push | 5 | 0.037 | [-0.015, 0.088] | [0.007, 0.070] | 0.119 | 0.358 | null |
| F4a-cvar (PD-relabel - BRIDGE-slip) | L1 | rain | 5 | 0.003 | [-0.006, 0.013] | [0.000, 0.010] | 0.374 | 0.748 | null |
| F4a-cvar (PD-relabel - BRIDGE-slip) | L1 | slip | 5 | 0.160 | [0.092, 0.228] | [0.113, 0.193] | 0.003 | 0.017 | pass |
| F4a-cvar (PD-relabel - BRIDGE-slip) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4a-cvar (PD-relabel - BRIDGE-slip) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4a-cvar (PD-relabel - BRIDGE-slip) | L2 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4b-cvar (BRIDGE-on-PD - PD-noise) | L1 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4b-cvar (BRIDGE-on-PD - PD-noise) | L1 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4b-cvar (BRIDGE-on-PD - PD-noise) | L1 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4b-cvar (BRIDGE-on-PD - PD-noise) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4b-cvar (BRIDGE-on-PD - PD-noise) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4b-cvar (BRIDGE-on-PD - PD-noise) | L2 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4-cvar (MPC-relabel - BRIDGE-slip) | L1 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4-cvar (MPC-relabel - BRIDGE-slip) | L1 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4-cvar (MPC-relabel - BRIDGE-slip) | L1 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4-cvar (MPC-relabel - BRIDGE-slip) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4-cvar (MPC-relabel - BRIDGE-slip) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F4-cvar (MPC-relabel - BRIDGE-slip) | L2 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F5-cvar (MPC-oracle - BRIDGE-slip) | L1 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F5-cvar (MPC-oracle - BRIDGE-slip) | L1 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F5-cvar (MPC-oracle - BRIDGE-slip) | L1 | slip | 5 | 0.150 | [0.025, 0.275] | [0.070, 0.223] | 0.029 | 0.174 | inconclusive |
| F5-cvar (MPC-oracle - BRIDGE-slip) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F5-cvar (MPC-oracle - BRIDGE-slip) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F5-cvar (MPC-oracle - BRIDGE-slip) | L2 | slip | 5 | 0.003 | [-0.006, 0.013] | [0.000, 0.010] | 0.374 | 1.000 | null |
| F6-cvar (GC-diff-rollouts - PD-noise) | L1 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F6-cvar (GC-diff-rollouts - PD-noise) | L1 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F6-cvar (GC-diff-rollouts - PD-noise) | L1 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F6-cvar (GC-diff-rollouts - PD-noise) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F6-cvar (GC-diff-rollouts - PD-noise) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F6-cvar (GC-diff-rollouts - PD-noise) | L2 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F7-cvar (MPPI-rollouts - PD-noise) | L1 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F7-cvar (MPPI-rollouts - PD-noise) | L1 | rain | 5 | 0.007 | [-0.012, 0.025] | [0.000, 0.020] | 0.374 | 1.000 | null |
| F7-cvar (MPPI-rollouts - PD-noise) | L1 | slip | 5 | 0.127 | [0.029, 0.224] | [0.057, 0.180] | 0.023 | 0.137 | inconclusive |
| F7-cvar (MPPI-rollouts - PD-noise) | L2 | push | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F7-cvar (MPPI-rollouts - PD-noise) | L2 | rain | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
| F7-cvar (MPPI-rollouts - PD-noise) | L2 | slip | 5 | 0.000 | [0.000, 0.000] | [0.000, 0.000] | 1.000 | 1.000 | null |
<!-- /tables:cvar -->

## 9. Replication (seeds 5-9)

Every source, both layouts, seeds 5-9, run here at the registered budgets (all bridge
arms trained at BRIDGE_CFG, so the prior run's bridge-net confound is absent). The
decision rule applied to these seeds alone gives the same verdict as seeds 0-4; the
agreement line below is computed, not typed. Two things the replication adds: BRIDGE-slip
trained at the registered budget is weaker than the prior run's Phase-2 nets (compare the
L1 slip cell here with section 4), which is the confound ERRORS.md recorded and does not
change any sign; and the tail statistic (section 8) is floored for every source, so the
program's CVaR criterion cannot separate them in this family.

<!-- tables:rep -->
| source | layout | disturbance | n | mean | 95% CI |
|---|---|---|---|---|---|
| BRIDGE-on-PD | L1 | none | 5 | 0.812 | [0.632, 0.992] |
| BRIDGE-on-PD | L1 | slip | 5 | 0.287 | [0.245, 0.329] |
| BRIDGE-on-PD | L1 | rain | 5 | 0.119 | [0.094, 0.144] |
| BRIDGE-on-PD | L1 | push | 5 | 0.057 | [0.037, 0.077] |
| BRIDGE-on-PD | L2 | none | 5 | 0.689 | [0.423, 0.955] |
| BRIDGE-on-PD | L2 | slip | 5 | 0.238 | [0.177, 0.299] |
| BRIDGE-on-PD | L2 | rain | 5 | 0.108 | [0.066, 0.150] |
| BRIDGE-on-PD | L2 | push | 5 | 0.024 | [0.004, 0.044] |
| BRIDGE-slip | L1 | none | 5 | 0.983 | [0.949, 1.017] |
| BRIDGE-slip | L1 | slip | 5 | 0.399 | [0.360, 0.438] |
| BRIDGE-slip | L1 | rain | 5 | 0.153 | [0.124, 0.182] |
| BRIDGE-slip | L1 | push | 5 | 0.126 | [0.113, 0.139] |
| BRIDGE-slip | L2 | none | 5 | 0.826 | [0.687, 0.965] |
| BRIDGE-slip | L2 | slip | 5 | 0.229 | [0.162, 0.296] |
| BRIDGE-slip | L2 | rain | 5 | 0.086 | [0.068, 0.104] |
| BRIDGE-slip | L2 | push | 5 | 0.037 | [0.020, 0.054] |
| BRIDGE-unicycle | L1 | none | 5 | 0.903 | [0.821, 0.985] |
| BRIDGE-unicycle | L1 | slip | 5 | 0.166 | [0.090, 0.242] |
| BRIDGE-unicycle | L1 | rain | 5 | 0.066 | [0.041, 0.091] |
| BRIDGE-unicycle | L1 | push | 5 | 0.036 | [0.016, 0.056] |
| BRIDGE-unicycle | L2 | none | 5 | 0.683 | [0.559, 0.807] |
| BRIDGE-unicycle | L2 | slip | 5 | 0.144 | [0.062, 0.226] |
| BRIDGE-unicycle | L2 | rain | 5 | 0.067 | [0.022, 0.112] |
| BRIDGE-unicycle | L2 | push | 5 | 0.020 | [0.004, 0.036] |
| DART | L1 | none | 5 | 0.993 | [0.985, 1.001] |
| DART | L1 | slip | 5 | 0.721 | [0.701, 0.741] |
| DART | L1 | rain | 5 | 0.335 | [0.296, 0.374] |
| DART | L1 | push | 5 | 0.503 | [0.413, 0.593] |
| DART | L2 | none | 5 | 0.739 | [0.494, 0.984] |
| DART | L2 | slip | 5 | 0.380 | [0.269, 0.491] |
| DART | L2 | rain | 5 | 0.162 | [0.125, 0.199] |
| DART | L2 | push | 5 | 0.099 | [0.044, 0.154] |
| DEMO | L1 | none | 5 | 0.911 | [0.789, 1.033] |
| DEMO | L1 | slip | 5 | 0.221 | [0.159, 0.283] |
| DEMO | L1 | rain | 5 | 0.130 | [0.084, 0.176] |
| DEMO | L1 | push | 5 | 0.124 | [0.079, 0.169] |
| DEMO | L2 | none | 5 | 0.114 | [0.077, 0.151] |
| DEMO | L2 | slip | 5 | 0.038 | [0.031, 0.045] |
| DEMO | L2 | rain | 5 | 0.019 | [0.007, 0.031] |
| DEMO | L2 | push | 5 | 0.018 | [0.011, 0.025] |
| GC-diff-rollouts | L1 | none | 5 | 0.994 | [0.989, 0.999] |
| GC-diff-rollouts | L1 | slip | 5 | 0.482 | [0.421, 0.543] |
| GC-diff-rollouts | L1 | rain | 5 | 0.237 | [0.171, 0.303] |
| GC-diff-rollouts | L1 | push | 5 | 0.395 | [0.364, 0.426] |
| GC-diff-rollouts | L2 | none | 5 | 0.190 | [0.099, 0.281] |
| GC-diff-rollouts | L2 | slip | 5 | 0.078 | [0.044, 0.112] |
| GC-diff-rollouts | L2 | rain | 5 | 0.036 | [0.023, 0.049] |
| GC-diff-rollouts | L2 | push | 5 | 0.055 | [0.039, 0.071] |
| MPC-oracle | L1 | none | 5 | 0.994 | [0.986, 1.002] |
| MPC-oracle | L1 | slip | 5 | 0.739 | [0.658, 0.820] |
| MPC-oracle | L1 | rain | 5 | 0.393 | [0.317, 0.469] |
| MPC-oracle | L1 | push | 5 | 0.481 | [0.416, 0.546] |
| MPC-oracle | L2 | none | 5 | 0.724 | [0.400, 1.048] |
| MPC-oracle | L2 | slip | 5 | 0.444 | [0.336, 0.552] |
| MPC-oracle | L2 | rain | 5 | 0.212 | [0.180, 0.244] |
| MPC-oracle | L2 | push | 5 | 0.137 | [0.063, 0.211] |
| MPC-relabel | L1 | none | 5 | 0.146 | [-0.154, 0.446] |
| MPC-relabel | L1 | slip | 5 | 0.310 | [0.140, 0.480] |
| MPC-relabel | L1 | rain | 5 | 0.273 | [0.164, 0.382] |
| MPC-relabel | L1 | push | 5 | 0.116 | [0.032, 0.200] |
| MPC-relabel | L2 | none | 5 | 0.012 | [-0.018, 0.042] |
| MPC-relabel | L2 | slip | 5 | 0.139 | [0.048, 0.230] |
| MPC-relabel | L2 | rain | 5 | 0.095 | [0.073, 0.117] |
| MPC-relabel | L2 | push | 5 | 0.038 | [-0.001, 0.077] |
| MPPI-rollouts | L1 | none | 5 | 0.989 | [0.981, 0.997] |
| MPPI-rollouts | L1 | slip | 5 | 0.665 | [0.573, 0.757] |
| MPPI-rollouts | L1 | rain | 5 | 0.370 | [0.304, 0.436] |
| MPPI-rollouts | L1 | push | 5 | 0.484 | [0.412, 0.556] |
| MPPI-rollouts | L2 | none | 5 | 0.861 | [0.709, 1.013] |
| MPPI-rollouts | L2 | slip | 5 | 0.431 | [0.365, 0.497] |
| MPPI-rollouts | L2 | rain | 5 | 0.207 | [0.179, 0.235] |
| MPPI-rollouts | L2 | push | 5 | 0.104 | [0.078, 0.130] |
| NOISED | L1 | none | 5 | 0.669 | [0.477, 0.861] |
| NOISED | L1 | slip | 5 | 0.111 | [0.080, 0.142] |
| NOISED | L1 | rain | 5 | 0.052 | [0.026, 0.078] |
| NOISED | L1 | push | 5 | 0.041 | [0.023, 0.059] |
| NOISED | L2 | none | 5 | 0.355 | [0.202, 0.508] |
| NOISED | L2 | slip | 5 | 0.056 | [0.014, 0.098] |
| NOISED | L2 | rain | 5 | 0.033 | [0.017, 0.049] |
| NOISED | L2 | push | 5 | 0.010 | [0.002, 0.018] |
| PD-iso | L1 | none | 5 | 0.994 | [0.991, 0.997] |
| PD-iso | L1 | slip | 5 | 0.721 | [0.689, 0.753] |
| PD-iso | L1 | rain | 5 | 0.365 | [0.343, 0.387] |
| PD-iso | L1 | push | 5 | 0.556 | [0.523, 0.589] |
| PD-iso | L2 | none | 5 | 0.767 | [0.656, 0.878] |
| PD-iso | L2 | slip | 5 | 0.333 | [0.239, 0.427] |
| PD-iso | L2 | rain | 5 | 0.164 | [0.139, 0.189] |
| PD-iso | L2 | push | 5 | 0.160 | [0.092, 0.228] |
| PD-noise | L1 | none | 5 | 0.984 | [0.966, 1.002] |
| PD-noise | L1 | slip | 5 | 0.577 | [0.498, 0.656] |
| PD-noise | L1 | rain | 5 | 0.272 | [0.216, 0.328] |
| PD-noise | L1 | push | 5 | 0.486 | [0.431, 0.541] |
| PD-noise | L2 | none | 5 | 0.568 | [0.426, 0.710] |
| PD-noise | L2 | slip | 5 | 0.269 | [0.217, 0.321] |
| PD-noise | L2 | rain | 5 | 0.125 | [0.068, 0.182] |
| PD-noise | L2 | push | 5 | 0.125 | [0.090, 0.160] |
| PD-relabel | L1 | none | 5 | 0.994 | [0.986, 1.002] |
| PD-relabel | L1 | slip | 5 | 0.795 | [0.731, 0.859] |
| PD-relabel | L1 | rain | 5 | 0.425 | [0.350, 0.500] |
| PD-relabel | L1 | push | 5 | 0.654 | [0.609, 0.699] |
| PD-relabel | L2 | none | 5 | 0.882 | [0.803, 0.961] |
| PD-relabel | L2 | slip | 5 | 0.372 | [0.298, 0.446] |
| PD-relabel | L2 | rain | 5 | 0.197 | [0.163, 0.231] |
| PD-relabel | L2 | push | 5 | 0.170 | [0.135, 0.205] |

| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F1 (BRIDGE-slip - PD-noise) | L1 | push | 5 | -0.360 | [-0.419, -0.301] | [-0.399, -0.327] | 0.000 | 0.000 | null |
| F1 (BRIDGE-slip - PD-noise) | L1 | rain | 5 | -0.119 | [-0.177, -0.061] | [-0.148, -0.078] | 0.005 | 0.009 | null |
| F1 (BRIDGE-slip - PD-noise) | L1 | slip | 5 | -0.178 | [-0.291, -0.065] | [-0.250, -0.115] | 0.012 | 0.018 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | push | 5 | -0.088 | [-0.129, -0.047] | [-0.114, -0.062] | 0.004 | 0.009 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | rain | 5 | -0.039 | [-0.097, 0.019] | [-0.075, -0.004] | 0.136 | 0.163 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | slip | 5 | -0.040 | [-0.107, 0.027] | [-0.080, 0.000] | 0.171 | 0.171 | null |

**Verdict: neither**

| count | value |
|---|---|
| f1_positive | 0 |
| f4_large | 6 |
| f1_null | 2 |
| f3_positive | 0 |
| n_f1 | 6 |
| n_f4 | 6 |
| n_f3 | 6 |
| n_cells | 6 |

Agreement with seeds 0-4: yes
<!-- /tables:rep -->

## 10. Verdict

<!-- tables:decision -->
**Verdict: neither**

| count | value |
|---|---|
| f1_positive | 0 |
| f4_large | 3 |
| f1_null | 1 |
| f3_positive | 0 |
| n_f1 | 6 |
| n_f4 | 6 |
| n_f3 | 6 |
| n_cells | 6 |
<!-- /tables:decision -->

On seeds 0-4 the rule gives **neither**, and it is not a close call. The primary
contrast (section 5) is negative in every disturbed cell and null against the minimum
effect in five of six. The tracker does not care about the noise shape (section 6.1,
F3: PD-iso is if anything better than PD-noise), so the noise model cannot be what
matters either. The reference covariance is a real bridge-specific mechanism (F2: the
slip reference beats the unicycle reference by a wide margin inside the bridge), but it
only stops the bridge from being worse; it never produces a lead over the tracker.

The 2x2 label isolation (section 6.2) says where the effect lives. Put the tracker's
labels on the bridge's states and the result is the best in-family source on L1 (F4a,
above the minimum effect in all three L1 cells, BH-significant; L2 positive but below
the minimum effect). Put the bridge's labels on the tracker's states and the result
drops below PD-noise by about as much (F4b). The bridge's states are not the problem;
its drift labels are. That is consistent with the coverage table: wide off-path coverage
correlates negatively with success across sources (Spearman well below zero), except
for the one source that combines wide coverage with corrective labels.

The oracle upper bound (F5) sits above the bridge everywhere. The two other
generators: a goal-conditioned diffusion policy rolled out under its own sampling noise
is worse than the tracker in every cell (F6), and the sampling MPC is indistinguishable
from it (F7).

For the `data` slot of the grammar (SEARCH_PLAN 0.6 as amended in PLAN_CHANGES): the
slot drops BRIDGE-* and MPC-relabel and keeps the non-bridge sources; the default is the
best of them by the cell table, which on these seeds is PD-relabel on L1 and DART /
MPC-oracle territory on L2. PD-relabel is worth carrying forward as its own idea: the
bridge as a state generator with a tracker as the labeller (IDEAS_NEXT.md).

The replication (section 9) returns the same verdict on its own seeds, with the primary
contrast negative in every disturbed cell and the labels-on-bridge-states contrast above
the minimum effect in all six. The verdict stands.

## 11. Minimum-effect statement

<!-- tables:mineffect -->
| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F1 (BRIDGE-slip - PD-noise) | L1 | push | 5 | -0.317 | [-0.394, -0.240] | [-0.367, -0.270] | 0.000 | 0.002 | null |
| F1 (BRIDGE-slip - PD-noise) | L1 | rain | 5 | -0.112 | [-0.166, -0.058] | [-0.145, -0.078] | 0.005 | 0.014 | null |
| F1 (BRIDGE-slip - PD-noise) | L1 | slip | 5 | -0.113 | [-0.221, -0.005] | [-0.186, -0.050] | 0.044 | 0.066 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | push | 5 | -0.072 | [-0.122, -0.022] | [-0.102, -0.041] | 0.016 | 0.033 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | rain | 5 | -0.039 | [-0.107, 0.029] | [-0.086, -0.004] | 0.185 | 0.222 | null |
| F1 (BRIDGE-slip - PD-noise) | L2 | slip | 5 | -0.015 | [-0.151, 0.121] | [-0.097, 0.074] | 0.774 | 0.774 | inconclusive |
| F5 (MPC-oracle - BRIDGE-slip) | L1 | push | 5 | 0.297 | [0.246, 0.348] | [0.262, 0.324] | 0.000 | 0.001 | pass |
| F5 (MPC-oracle - BRIDGE-slip) | L1 | rain | 5 | 0.209 | [0.134, 0.284] | [0.158, 0.258] | 0.002 | 0.002 | pass |
| F5 (MPC-oracle - BRIDGE-slip) | L1 | slip | 5 | 0.249 | [0.196, 0.302] | [0.212, 0.279] | 0.000 | 0.001 | pass |
| F5 (MPC-oracle - BRIDGE-slip) | L2 | push | 5 | 0.054 | [0.009, 0.099] | [0.024, 0.081] | 0.028 | 0.030 | null |
| F5 (MPC-oracle - BRIDGE-slip) | L2 | rain | 5 | 0.103 | [0.016, 0.190] | [0.048, 0.160] | 0.030 | 0.030 | pass |
| F5 (MPC-oracle - BRIDGE-slip) | L2 | slip | 5 | 0.148 | [0.095, 0.201] | [0.121, 0.185] | 0.001 | 0.002 | pass |
| F6 (GC-diff-rollouts - PD-noise) | L1 | push | 5 | -0.144 | [-0.208, -0.080] | [-0.185, -0.105] | 0.003 | 0.007 | null |
| F6 (GC-diff-rollouts - PD-noise) | L1 | rain | 5 | -0.055 | [-0.073, -0.037] | [-0.066, -0.044] | 0.001 | 0.004 | null |
| F6 (GC-diff-rollouts - PD-noise) | L1 | slip | 5 | -0.138 | [-0.219, -0.057] | [-0.196, -0.098] | 0.009 | 0.011 | null |
| F6 (GC-diff-rollouts - PD-noise) | L2 | push | 5 | -0.108 | [-0.146, -0.070] | [-0.136, -0.090] | 0.001 | 0.004 | null |
| F6 (GC-diff-rollouts - PD-noise) | L2 | rain | 5 | -0.079 | [-0.125, -0.033] | [-0.105, -0.049] | 0.009 | 0.011 | null |
| F6 (GC-diff-rollouts - PD-noise) | L2 | slip | 5 | -0.171 | [-0.291, -0.051] | [-0.248, -0.092] | 0.017 | 0.017 | null |
| F7 (MPPI-rollouts - PD-noise) | L1 | push | 5 | -0.020 | [-0.109, 0.069] | [-0.083, 0.032] | 0.568 | 0.568 | null |
| F7 (MPPI-rollouts - PD-noise) | L1 | rain | 5 | 0.055 | [-0.046, 0.156] | [-0.013, 0.119] | 0.205 | 0.409 | inconclusive |
| F7 (MPPI-rollouts - PD-noise) | L1 | slip | 5 | 0.068 | [-0.053, 0.189] | [-0.007, 0.143] | 0.194 | 0.409 | inconclusive |
| F7 (MPPI-rollouts - PD-noise) | L2 | push | 5 | -0.023 | [-0.083, 0.037] | [-0.056, 0.017] | 0.349 | 0.451 | null |
| F7 (MPPI-rollouts - PD-noise) | L2 | rain | 5 | 0.042 | [-0.075, 0.159] | [-0.027, 0.110] | 0.376 | 0.451 | inconclusive |
| F7 (MPPI-rollouts - PD-noise) | L2 | slip | 5 | 0.107 | [-0.054, 0.268] | [0.010, 0.213] | 0.139 | 0.409 | inconclusive |
<!-- /tables:mineffect -->

## 12. Deviations from the plan

- PLAN_CHANGES 2026-09-13 (0.6 wording): the "neither" candidate set for the data slot
  was widened to every non-bridge source before any cell ran.
- PLAN_CHANGES 2026-09-13 (0.5 fired): MPC-relabel failed the diagnostic under both
  costs and left the decision rule; the 2x2 label isolation replaced it. MPC-relabel is
  still tabulated (F4, greedy cost) but not used.
- ERRORS.md 2026-09-13: the prior g1 loaded Phase 2's bridge nets for BRIDGE-slip on
  seeds 0-4, a different training budget from the other bridge arm; the replication
  set trains every arm at BRIDGE_CFG.
- CVaR (section 8) needs per-episode records for both sources of a contrast; the prior
  cells have none, so it is reported from the replication set only.
