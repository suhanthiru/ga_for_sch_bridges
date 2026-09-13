"""Throughput of the SE(2) step: eager Python loop vs CUDA-graph replay of the branch-free
FastEnv step with a PD controller inside the step, for batches of 4k / 32k / 262k robots
of mixed disturbance kinds. Reports steps/s, microseconds per step, peak memory and
whether eager and graph replay agree on the final state from the same seed.

    python bench/env_step.py [--sizes 4096,32768,262144] [--steps 300]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import machine_info, peak_mb, record, reset_peak, timeit  # noqa: E402

from sb.core import se2 as S  # noqa: E402
from sb.envs import task as TK  # noqa: E402
from sb.envs import terrain as TR  # noqa: E402
from sb.envs.fast import FastEnv  # noqa: E402
from sb.envs.gen_task import GenTask  # noqa: E402


def make(n, device):
    tk = GenTask(TR.Layout("L2"), "push", n, 0, device)
    kinds = torch.arange(4).repeat_interleave(n // 4)
    fe = FastEnv.from_task(tk, kinds=kinds)
    fe.rain_t = torch.where(kinds == 2, torch.full_like(kinds, 150), torch.full_like(kinds, 10 ** 9)).to(device)
    return tk, fe


def pd_step(fe, g, step):
    """Nominal PD toward the geodesic between the skill's marginal means, all in tensors."""
    k = torch.clamp(step // TK.T_SKILL, max=2)
    tau = ((step % TK.T_SKILL).float() / TK.T_SKILL + 1.0 / TK.T_SKILL).clamp(max=1.0)
    a = fe.means[k].expand_as(g); b = fe.means[k + 1].expand_as(g)
    ref = S.SE2.interp(a, b, tau.expand(g.shape[0]))
    return 6.0 * S.between(g, ref)


def episode_eager(fe, g0, steps):
    g = g0.clone(); alive = torch.ones(g.shape[0], dtype=torch.bool, device=g.device)
    for s in range(steps):
        step = torch.tensor(s, device=g.device)
        u = pd_step(fe, g, step)
        g_new, hit = fe.step_random(g, u, step)
        g = torch.where(alive[:, None], g_new, g); alive &= ~hit
    return g, alive


def episode_graph(fe, g0, steps):
    """Capture one step into a CUDA graph with static buffers and replay it `steps` times."""
    g = g0.clone(); alive = torch.ones(g.shape[0], dtype=torch.bool, device=g.device)
    step = torch.zeros((), dtype=torch.long, device=g.device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):                                  # warm up allocations on the side stream
            u = pd_step(fe, g, step); g_new, hit = fe.step_random(g, u, step)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        u = pd_step(fe, g, step)
        g_new, hit = fe.step_random(g, u, step)
        g_out = torch.where(alive[:, None], g_new, g)
        alive_out = alive & ~hit
    g.copy_(g0); alive.fill_(True); step.zero_()
    for _ in range(steps):
        graph.replay()
        g.copy_(g_out); alive.copy_(alive_out); step += 1
    return g, alive


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="4096,32768,262144"); ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    device = torch.device("cuda")
    print(machine_info())
    for n in (int(x) for x in a.sizes.split(",")):
        tk, fe = make(n, device)
        g0 = tk.sample(0, n)
        for mode, fn in (("eager", episode_eager), ("cudagraph", episode_graph)):
            reset_peak()
            torch.manual_seed(0); torch.cuda.manual_seed(0)
            t = timeit(lambda: fn(fe, g0, a.steps), warmup=2, iters=5)
            torch.manual_seed(0); torch.cuda.manual_seed(0); gA, aA = fn(fe, g0, a.steps)
            torch.manual_seed(0); torch.cuda.manual_seed(0); gB, aB = fn(fe, g0, a.steps)
            per_step = t["median"] / a.steps
            m = dict(n=n, steps=a.steps, s_per_episode=t["median"], us_per_step=per_step * 1e6, steps_per_s=n / per_step,
                     peak_mb=peak_mb(), repeat_bit_equal=bool(torch.equal(gA, gB) and torch.equal(aA, aB)),
                     alive_frac=float(aA.float().mean()))
            record("env_step", f"{mode}_n{n}", m)
            print(f"{mode:9s} n={n:7d}  {m['us_per_step']:9.1f} us/step  {m['steps_per_s'] / 1e6:8.2f} M steps/s  peak {m['peak_mb']} MB  alive {m['alive_frac']:.2f}")


if __name__ == "__main__":
    main()
