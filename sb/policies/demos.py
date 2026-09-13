"""Demo sets: rollouts of the nominal demonstrator under no disturbance, with the
demonstrator re-evaluated on the recorded poses so the labels carry no process noise
(recovering u from consecutive poses would bake sigma_base/sqrt(dt) ~ 0.2 into them).
"""
from pathlib import Path

import numpy as np
import torch

from sb.envs import task as TK
from sb.envs import terrain as TR
from sb.policies.nominal import Nominal


@torch.no_grad()
def make_demos(layout, n, path, device=torch.device("cpu"), seed=4242):
    tk = TK.Task(TR.Layout(layout), "none", n, seed, device, layout_id=0)
    out = tk.rollout(Nominal(tk), n=n, record=True)
    G = out["traj"]                                     # (n, 301, 3)
    ctl = Nominal(tk); U = torch.zeros(n, 300, 3, device=device)
    for s_ in range(300):
        k = s_ // TK.T_SKILL; tau = torch.full((n,), (s_ % TK.T_SKILL) / TK.T_SKILL, device=device)
        U[:, s_] = TK.clip_u(ctl(G[:, s_], k, tau, s_)[0])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, G=G.cpu().numpy(), U=U.cpu().numpy(), success=out["success"].cpu().numpy())
    return float(out["success"].float().mean())


def load_demos(path, device):
    z = np.load(path)
    return torch.tensor(z["G"], device=device), torch.tensor(z["U"], device=device)
