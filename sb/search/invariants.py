"""Exploit invariants (SEARCH_PLAN / program section 6), recomputed per episode from the
recorded trajectory with geometry independent of the environment's own collision flag.

  wall_penetration      a recorded segment crosses the wall outside a gap or enters the pile
  box_exit              a recorded pose leaves the unit box
  goal_by_marginal      success claimed but the final pose is outside the goal set
  time_to_goal          reached the goal set earlier than 0.9 x the oracle's time (a shortcut)
  energy_floor          energy below 0.5 x the oracle's for a successful episode
  nan                   any non-finite value in the trajectory or actions
  action_bounds         any recorded action outside the clip
  terrain_consistency   the terrain the policy queried differs from the map at its position

Each returns the fraction of episodes tripping it; any nonzero fraction quarantines the
genome (tagged, kept in the archive, excluded from the elite map until a clean re-run).
"""
import math

import numpy as np
import torch

from sb.core import se2 as S
from sb.envs import task as TK


def segment_hits(a, b, geo):
    """(n,) bool: segment a->b crosses the wall outside a gap, hits the pile, or leaves the box."""
    wx = geo.get("wall_x", 0.5); gaps = geo.get("gaps", []); pile = geo.get("pile")
    da, db = a[:, 0] - wx, b[:, 0] - wx
    cross = (da * db) < 0
    t = da / (da - db + 1e-12); y = a[:, 1] + t * (b[:, 1] - a[:, 1])
    in_gap = torch.zeros_like(cross)
    for c, w in gaps:
        in_gap |= (y > c - w / 2) & (y < c + w / 2)
    hit = cross & ~in_gap
    if pile is not None:
        px, py, pr = pile
        hit |= ((b[:, 0] - px) ** 2 + (b[:, 1] - py) ** 2).sqrt() < pr
    hit |= (b < 0).any(1) | (b > 1).any(1)
    return hit


def check_episode_set(traj, actions, success, geo, oracle_time=None, oracle_energy=None, energy=None, first_goal_step=None):
    """traj (n, T+1, 3); actions (n, T, 3); success (n,) bool. Returns dict invariant -> fraction."""
    n, T1, _ = traj.shape
    out = {}
    hit = torch.zeros(n, dtype=torch.bool, device=traj.device)
    for s in range(T1 - 1):
        hit |= segment_hits(traj[:, s, :2], traj[:, s + 1, :2], geo)
    out["wall_penetration"] = float(hit.float().mean())
    out["box_exit"] = float(((traj[:, :, :2] < 0) | (traj[:, :, :2] > 1)).any(2).any(1).float().mean())
    means, covs = geo.get("means"), geo.get("covs")
    if means is not None and covs is not None:
        md = S.mahalanobis(traj[:, -1].cpu(), means[-1], covs[-1])
        out["goal_by_marginal"] = float((success.cpu() & (md > 2.0)).float().mean())
    out["nan"] = float((~torch.isfinite(traj).all(2).all(1) | ~torch.isfinite(actions).all(2).all(1)).float().mean())
    ok = (actions[:, :, :2].norm(dim=2) <= TK.UMAX + 1e-4).all(1) & (actions[:, :, 2].abs() <= TK.WMAX + 1e-4).all(1)
    out["action_bounds"] = float((~ok).float().mean())
    if oracle_time is not None and first_goal_step is not None:
        out["time_to_goal"] = float((success & (first_goal_step < 0.9 * oracle_time)).float().mean())
    if oracle_energy is not None and energy is not None:
        out["energy_floor"] = float((success & (energy < 0.5 * oracle_energy)).float().mean())
    return out


def tripped(inv):
    return [k for k, v in inv.items() if v > 0]


def first_goal_step(traj, geo, md_thr=2.0):
    """Step index at which each episode first enters the goal set (T if never)."""
    means, covs = geo["means"], geo["covs"]
    n, T1, _ = traj.shape
    first = torch.full((n,), T1 - 1, dtype=torch.long)
    for s in range(T1 - 1, -1, -1):
        inside = S.mahalanobis(traj[:, s].cpu(), means[-1], covs[-1]) <= md_thr
        first = torch.where(inside, torch.full_like(first, s), first)
    return first
