"""Diffusion policy: ms per training step at batch 1024 (single policy, the gate's learner)
and DDIM-10 sampling throughput with the policy in the loop for 512 and 4096 robots.

    python bench/diffusion_train.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import machine_info, peak_mb, record, reset_peak, timeit  # noqa: E402

from sb.envs import terrain as TR  # noqa: E402
from sb.envs.gen_task import GenTask  # noqa: E402
from sb.policies.common import CHUNK, obs_of  # noqa: E402
from sb.policies.diffusion import DiffPolicy, train_diffusion  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(machine_info())
    tk = GenTask(TR.Layout("L1"), "none", 64, 1, device)
    G = torch.rand(100, 301, 3, device=device); U = torch.randn(100, 300, 3, device=device).clamp(-1, 1)
    steps = 200
    for name, graphed in (("single_b1024", False), ("single_b1024_graphed", device.type == "cuda")):
        reset_peak()
        t = timeit(lambda: train_diffusion(tk, G, U, 0, device, steps, graphed=graphed), warmup=1, iters=3)
        m = dict(batch=1024, graphed=graphed, ms_per_step=t["median"] / steps * 1e3, s_per_8000_steps=t["median"] / steps * 8000, peak_mb=peak_mb())
        record("diffusion_train", name, m); print(f"train {name}: {m['ms_per_step']:.2f} ms/step -> {m['s_per_8000_steps']:.0f} s per 8000 steps")
    pol = DiffPolicy().to(device).eval()
    for n in (512, 4096):
        g = tk.sample(0, n); obs = obs_of(tk, g, 0, torch.zeros(n, device=device))
        reset_peak()
        t = timeit(lambda: pol.sample(obs), warmup=3, iters=10)
        m = dict(n=n, ms_per_chunk=t["median"] * 1e3, env_steps_per_s_policy_in_loop=n * 4 / t["median"], peak_mb=peak_mb())
        record("diffusion_sample", f"ddim10_n{n}", m)
        print(f"sample n={n}: {m['ms_per_chunk']:.1f} ms per chunk of {CHUNK} -> {m['env_steps_per_s_policy_in_loop'] / 1e3:.0f}k env-steps/s executing 4")


if __name__ == "__main__":
    main()
