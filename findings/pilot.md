# Pilot throughput report

## The_Tower|NVIDIA GeForce RTX 3080 Ti — quiet GPU

measured 2026-09-13 18:57:12, commit b20c412

| target | registered | measured | ratio | decision |
|---|---|---|---|---|
| env_steps_per_s | 50,000 | 51,705,485.1 | 1034.11x | pass |
| grid_bridges_per_s | 200 | 9,867.3 | 49.34x | pass |
| neural_bridges_per_s | 200 | 2.3 | 0.01x | prune (< 50 %) |
| ppo_genome_updates_per_s | 128 | 48.9 | 0.38x | prune (< 50 %) |

Configurations: env cudagraph_n262144, grid float32_P4096, neural P256_bf16, ppo P128_bf16.

Gate at fp32: pass; bf16 max abs err 0.03512045478978856. Determinism (two processes): {'cuda': True}. Graph replay bit-equal on repeat: {'cudagraph_n4096': True, 'cudagraph_n32768': True, 'cudagraph_n262144': True}. Diffusion: 41.3 s per 8000 steps.


# Contended pass (kept for the record)

## The_Tower|NVIDIA GeForce RTX 3080 Ti — shared GPU (another CUDA process was active; provisional)

measured 2026-09-13 15:03:35, commit 137665a

| target | registered | measured | ratio | decision |
|---|---|---|---|---|
| env_steps_per_s | 50,000 | 1,552,932.8 | 31.06x | pass |
| grid_bridges_per_s | 200 | 9,698.1 | 48.49x | pass |
| neural_bridges_per_s | 200 | 1.6 | 0.01x | prune (< 50 %) |
| ppo_genome_updates_per_s | 128 | 25.5 | 0.20x | prune (< 50 %) |

Configurations: env cudagraph_n4096, grid float32_P4096, neural P256_bf16, ppo P32_fp32.

Gate at fp32: pass; bf16 max abs err 0.03512045478978856. Determinism (two processes): {'cuda': True}. Graph replay bit-equal on repeat: {'cudagraph_n4096': True}. Diffusion: 34.3 s per 8000 steps.

Pruning follows SEARCH_PLAN 0.4: any target below 50 % of its value prunes the component that depends on it; the decision goes to PLAN_CHANGES.md.
