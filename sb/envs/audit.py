"""Oracle audit (SEARCH_PLAN section 1): every family's oracle against a ten-times larger
sampling planner on the undisturbed task. An oracle more than 0.05 below the larger
planner is replaced by it before the search starts; the table goes to
findings/oracle_audit.md.

    python -m sb.cli audit [--n 64] [--device cuda]
"""
import time

import torch

from sb.envs.base import Caps, Descriptor
from sb.envs.e3_multimodal import ModeOracle
from sb.envs.e5_multiagent import PairOracle
from sb.envs.family import FAMILY
from sb.envs.oracles import WallOracle
from sb.gen.controllers import MPCOracle

BIG = dict(K=2000, H=30)
CASES = {  # env id -> (descriptor values, how to build the 10x comparator or None)
    "E1": (dict(layout="L2", disturbance="none"), lambda env: _wrap(MPCOracle(env.tk, seed=env.seed, **BIG), env)),
    "E2": (dict(layout="L2", disturbance="none"), lambda env: _wrap(MPCOracle(env.tk, seed=env.seed, **BIG), env)),
    "E3": (dict(n_modes=3, separation=0.2, disturbance="none"), lambda env: _wrap(ModeOracle(env.tk, seed=env.seed, K=2000, H=30), env)),
    "E4": (dict(mu=0.6, mass=2.0, disturbance="none"), None),
    "E5": (dict(ratio=2.5, disturbance="none"), lambda env: _wrap(PairOracle(env.tk, seed=env.seed, **BIG), env)),
    "E6": (dict(n_skills=8, slack=1.2, disturbance="none"), lambda env: _wall(env)),
    "E7": (dict(disturbance="none"), lambda env: _wrap(MPCOracle(env.tk, seed=env.seed, **BIG), env)),
    "E8": (dict(disturbance="none"), lambda env: _wrap(MPCOracle(env.tk, seed=env.seed, **BIG), env)),
}


def _wrap(m, env):
    return lambda g, k, tau, step: (m.act(g, k, step % env.T), None)


def _wall(env):
    m = WallOracle(env.tk, seed=env.seed, **BIG); tk = env.tk
    return lambda g, k, tau, step: (m.act_to(g, tk.means[k + 1].expand(g.shape[0], 3), step), None)


def _success(env, out):
    return float((out["pair_success"] if "pair_success" in out else out["success"]).float().mean())


def audit(n=64, device=None, ids=None, log=print):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    rows = []
    for eid in ids or FAMILY:
        cls = FAMILY[eid]; desc, big = CASES[eid]
        env = cls(n // 2, device) if eid == "E5" else cls(n, device)
        env.reset(0, Descriptor(desc))
        t0 = time.time(); s_or = _success(env, env.evaluate(env.oracle(Caps()))); t_or = time.time() - t0
        if big is None:
            rows.append(dict(env=eid, oracle=s_or, big=None, regret=None, oracle_s=t_or, note="no sampling comparator (rule-based pusher)"))
        else:
            env.reset(0, Descriptor(desc))
            t0 = time.time(); s_big = _success(env, env.evaluate(big(env))); t_big = time.time() - t0
            rows.append(dict(env=eid, oracle=s_or, big=s_big, regret=s_big - s_or, oracle_s=t_or, big_s=t_big,
                             note=("REPLACE" if s_big - s_or > 0.05 else "ok")))
        log(f"{eid}: oracle {s_or:.3f}" + (f", 10x {rows[-1]['big']:.3f}, regret {rows[-1]['regret']:+.3f} -> {rows[-1]['note']}" if big else f" ({rows[-1]['note']})"))
    return rows


def render(rows, n):
    lines = [f"# Oracle audit\n\nUndisturbed, {n} episodes per environment, oracle against the same planner at K = 2000 (ten times its samples). "
             "Regret above 0.05 replaces the oracle before the search starts.\n",
             "| env | oracle success | 10x planner | regret | note |", "|---|---|---|---|---|"]
    for r in rows:
        big = "-" if r["big"] is None else f"{r['big']:.3f}"; reg = "-" if r["regret"] is None else f"{r['regret']:+.3f}"
        lines.append(f"| {r['env']} | {r['oracle']:.3f} | {big} | {reg} | {r['note']} |")
    return "\n".join(lines) + "\n"
