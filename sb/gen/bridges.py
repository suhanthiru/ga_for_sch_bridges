"""Iteration-0 bridge drift nets for a reference kind, trained on CPU and cached.

The cache key carries a hash of the training config, so two arms of an isolation can
never silently come from different budgets (the prior suite loaded Phase 2's nets for
one arm and trained the other fresh).
"""
import hashlib
import json
from pathlib import Path

import torch

from sb.core import se2 as S
from sb.core import solver as SV
from sb.core.sde import Reference
from sb.envs import task as TK
from sb.envs.gen_task import to_cpu

BRIDGE_CFG = dict(n_pair=6000, steps0=1200, steps_ipf=300, batch=512, lr=1e-3, n_sim=50)
# the search's cached nets also carry the backward drift so the disagreement trigger has a D
SEARCH_BRIDGE_CFG = dict(BRIDGE_CFG, with_bwd=True)


def quantise_width(width):
    """Seam widths for the neural cache are powers of two: a continuous width would train
    a fresh set of nets for every mutant, which is the per-mutant solve the pilot pruned."""
    import math
    return float(2.0 ** round(math.log2(max(float(width), 1e-6))))


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def bridge_path(models_dir, kind, layout, seed, tag="", mf=S.SE2, cfg=BRIDGE_CFG):
    return Path(models_dir) / f"bridge_{kind}_{layout}{tag}_{mf.name}_{cfg_hash(cfg)}_s{seed}.pt"


def get_bridges(kind, tk, layout, seed, device, models_dir, tag="", mf=S.SE2, cfg=BRIDGE_CFG, width=1.0):
    """`width` scales the handoff marginals (rho_1, rho_2) the bridge is trained between:
    the seam slot's cloud width. It is part of the cache key."""
    width = quantise_width(width)
    if width != 1.0:
        tag = f"{tag}_w{width:g}"
    p = bridge_path(models_dir, kind, layout, seed, tag, mf, cfg)
    if p.exists():
        return torch.load(p, map_location=device, weights_only=True)
    torch.manual_seed(seed); cpu = torch.device("cpu")
    tkc = to_cpu(tk, seed)
    if width != 1.0:
        tkc.covs = [c.clone() for c in tkc.covs]
        for k in (1, 2):
            tkc.covs[k] = tkc.covs[k] * width ** 2
    ref_body = tk.body_std ** 2 if hasattr(tk, "body_std") else None
    ref = Reference(kind, sigma=0.05, kappa=2.0, slip_scale=tk.slip_scale if hasattr(tk, "slip_scale") else 1.0,
                    fields=tkc.obs_fields, body_cov=ref_body.to(cpu) if ref_body is not None else None)
    nets = {}
    for k in range(TK.N_SKILL):
        nets[k], _ = SV.train_skill(tkc, k, ref, mf, seed * 17 + k, cpu, K=0, log=lambda m: None, **dict(dict(with_bwd=False), **cfg))
    nets = {k: {key: {pn: t.to(device) for pn, t in sd.items()} for key, sd in n_.items()} for k, n_ in nets.items()}
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(nets, p)
    return nets
