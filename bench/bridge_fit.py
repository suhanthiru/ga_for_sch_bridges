"""Population-batched DSBM fitting: ms per population step, bridge-steps per second, and
the implied bridges per second at the 1200-step BRIDGE_CFG budget, for P in
{32, 128, 256} at batch 512, fp32 and bf16-autocast on the MLP. Also the single-net
solver.fit step for reference.

    python bench/bridge_fit.py [--sizes 32,128,256] [--steps 20]
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import machine_info, peak_mb, record, reset_peak, timeit  # noqa: E402

from sb.core import batched_fit as BFit  # noqa: E402
from sb.core import se2 as S  # noqa: E402
from sb.core import solver as SV  # noqa: E402
from sb.core.sde import Reference  # noqa: E402
from sb.envs import terrain as TR  # noqa: E402
from sb.envs.gen_task import GenTask  # noqa: E402
from sb.gen.bridges import BRIDGE_CFG  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sizes", default="32,128,256"); ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(machine_info())
    tk = GenTask(TR.Layout("L1"), "none", 64, 1, device)
    ref = Reference("slip", sigma=0.05, kappa=2.0, slip_scale=0.7, fields=tk.obs_fields)
    gen = torch.Generator(device=device).manual_seed(0)
    # single-net reference point
    net = SV.DriftNet(S.SE2).to(device); x0, x1 = SV.sample_pairs(tk, 0, 6000, ref, S.SE2)
    t = timeit(lambda: SV.fit(net, x0, x1, ref, S.SE2, tk.means[1], tk.obs_fields, a.steps, 512, 1e-3, gen=gen), warmup=1, iters=3)
    m = dict(P=1, ms_per_step=t["median"] / a.steps * 1e3, bridge_steps_per_s=a.steps / t["median"],
             bridges_per_s_at_cfg=1.0 / (t["median"] / a.steps * BRIDGE_CFG["steps0"]), device=device.type)
    record("bridge_fit", f"single_{device.type}", m); print(f"single net ({device.type}): {m['ms_per_step']:.1f} ms/step")
    for P in (int(x) for x in a.sizes.split(",")):
        nets = [SV.DriftNet(S.SE2).to(device) for _ in range(P)]
        pairs = [SV.sample_pairs(tk, 0, 6000, ref, S.SE2) for _ in range(P)]
        for ac in (False, True):
            if ac and device.type != "cuda":
                continue
            reset_peak()
            t = timeit(lambda: BFit.fit_population(nets, pairs, ref, S.SE2, tk.means[1], tk.obs_fields, a.steps, 512, 1e-3, gen, autocast=ac), warmup=1, iters=3)
            per = t["median"] / a.steps
            m = dict(P=P, autocast=ac, ms_per_pop_step=per * 1e3, bridge_steps_per_s=P / per,
                     bridges_per_s_at_cfg=P / (per * BRIDGE_CFG["steps0"]), peak_mb=peak_mb())
            record("bridge_fit", f"P{P}_{'bf16' if ac else 'fp32'}", m)
            print(f"P={P:4d} {'bf16' if ac else 'fp32'}: {m['ms_per_pop_step']:7.1f} ms/pop-step  {m['bridge_steps_per_s']:8.0f} bridge-steps/s  "
                  f"{m['bridges_per_s_at_cfg']:6.2f} bridges/s at {BRIDGE_CFG['steps0']} steps  peak {m['peak_mb']} MB")


if __name__ == "__main__":
    main()
