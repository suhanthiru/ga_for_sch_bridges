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

## Addendum 2026-09-13: diffusion training graph-captured

The diffusion policy's training step (batch draw, noise, forward, backward, Adam) is
captured once as a CUDA graph and replayed (`sb/policies/diffusion.py`; the CPU path is
the eager loop as before). `bench/diffusion_train.py` with pilot tranche 1 running on the
same GPU: eager 5.16 ms/step (41 s per 8000 steps), graphed 0.69 ms/step (5.5 s). The
quiet-machine row above (34.3 s) predates the change; the bench records both rows now.
The graphed trainer is deterministic from the seed (tests/test_diffusion_graph.py) and
is what the search rungs use from the next tranche on; the tranche running now is pinned
to its worktree and keeps the eager trainer.

## Addendum 2026-09-13: PPO rollout and update graph-captured

`sb/rl/pop_ppo.py`: the whole 32-step rollout (env steps, policy forwards, GAE, batch
assembly) is one CUDA graph and a minibatch gradient step another, replayed per
minibatch with the epoch's permutation copied into a static index buffer; the
environment state lives in fixed buffers and the user generators are registered with
the graphs, so a graphed run is reproducible from its seed (tests/test_pop_ppo.py).
Residual and learned-noise placements wrap a Python base controller and keep the eager
loop. Measured with tranche 1 on the same GPU (rollout 32, minibatch 1024, bf16): the
evaluator's setting P=1, n=120 went from 1.72 s to 0.14 s per update (a rung-0 PPO from
~45 s to ~4 s); P=32, n=64 gives 173 genome-updates/s, above the 128 registered, on a
contended GPU. `bench/ppo_update.py` now records eager and graphed rows.
