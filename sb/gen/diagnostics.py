"""The MPC-relabel diagnostic registered in SEARCH_PLAN 0.5.

Three numbers on demo trajectories: the speed of the relabelled commands relative to the
demo commands, the consistency of the labels across two planner seeds, and the collision
rate of a short-trained policy on the relabelled demos.
"""
import torch

from sb.core import se2 as S
from sb.envs.gen_task import GenTask, make_ref
from sb.gen import controllers as C
from sb.gen.noise import NoiseStream
from sb.gen.rollout import rollout
from sb.envs import terrain as TR
from sb.policies.diffusion import train_diffusion
from sb.policies.evaluate import evaluate

THRESH = dict(speed_ratio=(0.7, 1.4), consistency=0.3, collision=0.5)


def relabel_diag(demos, layout="L1", seed=0, n=64, device=torch.device("cpu"), cost="greedy", steps=300, n_eval=200):
    Gd, Ud = demos[0][:n], demos[1][:n]
    tk = GenTask(TR.Layout(layout), "none", n, 1000 + seed, device)
    ref = make_ref("slip", tk); z = NoiseStream(n, seed, device)
    _, U5 = rollout(tk, S.SE2, C.mppi_act(tk, 5, cost), n, z, ref, states=Gd)
    _, U6 = rollout(tk, S.SE2, C.mppi_act(tk, 6, cost), n, z, ref, states=Gd)
    speed = lambda U: float(U[:, :, :2].norm(dim=2).mean())
    rms = lambda U: U.pow(2).mean((0, 1)).sqrt()
    cons = (rms(U5 - U6) / rms(U5).clamp_min(1e-9)).tolist()
    pol = train_diffusion(tk, Gd, U5, seed, device, steps)
    tke = GenTask(TR.Layout(layout), "none", n_eval, 50_000 + seed, device)
    scal, _ = evaluate(pol, tke, n_eval)
    out = dict(cost=cost, speed_ratio=speed(U5) / speed(Ud), demo_speed=speed(Ud), relabel_speed=speed(U5),
               consistency_vx=cons[0], consistency_vy=cons[1], consistency_w=cons[2],
               collision=scal["collision"], success_none=scal["success"])
    lo, hi = THRESH["speed_ratio"]
    out["pass"] = bool(lo <= out["speed_ratio"] <= hi and max(cons) < THRESH["consistency"] and out["collision"] < THRESH["collision"])
    return out
