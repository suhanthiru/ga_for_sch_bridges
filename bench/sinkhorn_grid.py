"""Batched grid bridges per second: P in {256, 1024, 4096} on a 64x64 grid, 200 Sinkhorn
iterations, fp32 (the bf16 number is recorded with its marginal error, not used).

    python bench/sinkhorn_grid.py [--sizes 256,1024,4096] [--iters 200]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import machine_info, peak_mb, record, reset_peak, timeit  # noqa: E402

from sb.core import grid_sinkhorn as GS  # noqa: E402


def make(P, G, device, gen):
    c0 = torch.rand(P, 2, generator=gen) * 0.6 + 0.2; c1 = torch.rand(P, 2, generator=gen) * 0.6 + 0.2
    mu0 = torch.stack([GS.gaussian_marginal(G, tuple(c0[i].tolist()), 0.05) for i in range(P)]).to(device)
    mu1 = torch.stack([GS.gaussian_marginal(G, tuple(c1[i].tolist()), 0.05) for i in range(P)]).to(device)
    return mu0, mu1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="256,1024,4096"); ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--grid", type=int, default=64)
    a = ap.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(machine_info())
    gen = torch.Generator().manual_seed(0)
    for P in (int(x) for x in a.sizes.split(",")):
        mu0, mu1 = make(P, a.grid, device, gen)
        for dtype in (torch.float32, torch.bfloat16):
            m0, m1 = mu0.to(dtype), mu1.to(dtype)
            reset_peak()
            t = timeit(lambda: GS.solve(m0, m1, 0.005, a.iters), warmup=2, iters=5)
            _, _, _, err = GS.solve(m0, m1, 0.005, a.iters)
            m = dict(P=P, grid=a.grid, iters=a.iters, dtype=str(dtype).split(".")[-1], s_per_batch=t["median"],
                     bridges_per_s=P / t["median"], peak_mb=peak_mb(), marginal_l1_err_max=float(err.float().max()),
                     kb_per_bridge=peak_mb() * 1024 / P)
            record("sinkhorn_grid", f"{m['dtype']}_P{P}", m)
            print(f"{m['dtype']:9s} P={P:5d}  {m['bridges_per_s']:8.0f} bridges/s  err {m['marginal_l1_err_max']:.1e}  peak {m['peak_mb']} MB")


if __name__ == "__main__":
    main()
