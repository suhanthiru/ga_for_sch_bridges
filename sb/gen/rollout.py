"""Chain three skills on the reference kinematics (all in-family sources), or in the true
environment (the oracle upper bound)."""
import torch

from sb.core import se2 as S
from sb.envs import task as TK
from sb.gen.noise import DT, N_SKILL, T_SKILL, noise_step, step_kin


@torch.no_grad()
def rollout(tk, mf, act_fn, n, z, ref, iso_var=None, action_noise_ref=None, states=None, world_heading=None, starts=None):
    """act_fn(g, k, t) -> body command.
    starts: (n, 3) initial poses; if None they are drawn from the task's own generator.
    action_noise_ref: DART mode (noise on the action, none on the displacement).
    states: relabel mode - impose the given states, only record act_fn's commands."""
    if states is not None:
        g = states[:, 0].clone()
    elif starts is not None:
        g = starts.clone()
    else:
        g = tk.sample(0, n)
    G, U = [g.clone()], []
    for k in range(N_SKILL):
        for t in range(T_SKILL):
            s = k * T_SKILL + t
            u = act_fn(g, k, t)
            U.append(TK.clip_u(u))
            if states is not None:
                g = states[:, s + 1]; G.append(g.clone()); continue
            if action_noise_ref is not None:
                eta = noise_step(action_noise_ref, mf, g, z(s)) / DT
                g, _ = step_kin(mf, g, u + eta, torch.zeros_like(g))
            elif world_heading is not None:
                # body covariance rotated at the skill's mean heading, applied in the world
                # frame regardless of the robot's actual heading
                gbar = torch.tensor([0.0, 0.0, world_heading[k]], device=g.device).expand(g.shape[0], 3)
                xi_w = noise_step(ref, S.Flat, gbar, z(s))
                xi_b = torch.einsum("nij,nj->ni", S.rot3(g[:, 2]).transpose(1, 2), xi_w)
                g, _ = step_kin(mf, g, u, xi_b)
            else:
                g, _ = step_kin(mf, g, u, noise_step(ref, mf, g, z(s), iso_var))
            G.append(g.clone())
    return torch.stack(G, 1), torch.stack(U, 1)


class _Log:
    def __init__(self, act_fn):
        self.act_fn, self.U = act_fn, []

    def __call__(self, g, k, tau, step):
        u = TK.clip_u(self.act_fn(g, k, step % T_SKILL)); self.U.append(u.clone())
        return u, None


@torch.no_grad()
def rollout_true(tk, act_fn, n):
    """Roll act_fn out in the true environment (friction, terrain noise, kicks) and record
    the executed commands. Collided robots freeze, as in Task.rollout."""
    log = _Log(act_fn)
    out = tk.rollout(log, n=n, record=True)
    return out["traj"], torch.stack(log.U, 1), out
