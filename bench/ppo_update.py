"""Population PPO: seconds per full update (32 x 512 rollout plus 4 epochs of minibatches)
and genome-updates per second, for P in {32, 128, 256}, actor+critic 4x256, fp32.

    python bench/ppo_update.py [--sizes 32,128,256] [--minibatch 4096]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import machine_info, peak_mb, record, reset_peak, timeit  # noqa: E402

from sb.envs import terrain as TR  # noqa: E402
from sb.envs.gen_task import GenTask  # noqa: E402
from sb.rl.pop_ppo import PopEnv, PopPPO  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="32,128,256"); ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--rollout", type=int, default=32)
    a = ap.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(machine_info())
    for P in (int(x) for x in a.sizes.split(",")):
        try:
            tk = GenTask(TR.Layout("L1"), "slip", P * a.n, 0, device)
            env = PopEnv(tk, P, a.n); ppo = PopPPO(P, device); obs = env.reset()
            gen = torch.Generator(device=device).manual_seed(0)
            reset_peak()
            state = {"obs": obs}

            def one():
                state["obs"], _ = ppo.update(env, state["obs"], rollout=a.rollout, epochs=4, minibatch=a.minibatch, gen=gen)

            t = timeit(one, warmup=1, iters=3)
            m = dict(P=P, n_envs=a.n, rollout=a.rollout, minibatch=a.minibatch, s_per_update=t["median"],
                     genome_updates_per_s=P / t["median"], env_steps_per_s=P * a.n * a.rollout / t["median"], peak_mb=peak_mb())
            record("ppo_update", f"P{P}_fp32", m)
            print(f"P={P:4d}: {m['s_per_update']:.2f} s/update  {m['genome_updates_per_s']:.0f} genome-updates/s  peak {m['peak_mb']} MB")
        except torch.cuda.OutOfMemoryError:
            record("ppo_update", f"P{P}_fp32", dict(P=P, oom=True)); print(f"P={P}: OOM"); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
