"""Receding-horizon execution of a chunk policy and its evaluation on a task."""
import numpy as np
import torch

from sb.envs import task as TK
from sb.policies.common import EXEC, obs_of


class ChunkCtl:
    """Execute EXEC of the CHUNK predicted commands, then re-plan; also at each skill start."""

    def __init__(self, pol, tk, obs_fn=obs_of, gen=None):
        self.pol, self.tk, self.obs_fn, self.gen = pol, tk, obs_fn, gen
        self.buf, self.ptr = None, EXEC

    def __call__(self, g, k, tau, step):
        if self.buf is None or self.ptr >= EXEC or step % TK.T_SKILL == 0:
            self.buf, self.ptr = self.pol.sample(self.obs_fn(self.tk, g, k, tau), gen=self.gen), 0
        u = self.buf[:, self.ptr]; self.ptr += 1
        return u, None


@torch.no_grad()
def evaluate(pol, tk_eval, n, obs_fn=obs_of):
    """Scalar metrics over n episodes plus the per-episode outcomes needed for tail stats."""
    out = tk_eval.rollout(ChunkCtl(pol, tk_eval, obs_fn), n=n)
    m = tk_eval.metrics(out)
    scal = dict(success=m["success"], collision=m["collision"], handoff1_md=m["handoff1_md"],
                handoff2_md=m["handoff2_md"], w2_goal=m["w2_goal"], energy=m["energy"])
    eps = dict(success=out["success"].cpu().numpy().astype(np.int8),
               progress=out["progress"].cpu().numpy().astype(np.float32),
               collision=(~out["alive"]).cpu().numpy().astype(np.int8),
               energy=out["energy"].cpu().numpy().astype(np.float32))
    return scal, eps
