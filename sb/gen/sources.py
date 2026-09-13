"""Training-data sources for the gate. build() returns (G, U, meta, hist): the demos
concatenated with the generated set, coverage of the generated part, and its occupancy
histogram.

  DEMO              the n_demo demos only
  NOISED            demos + Gaussian-jittered copies, actions kept
  BRIDGE-<kind>     iteration-0 bridge drift with reference <kind>
  PD-noise          PD tracker of the nearest demo under the slip covariance
  PD-iso            the same tracker under isotropic noise of matched total variance
  PD-world          the tracker with the body covariance applied in a fixed world frame
  DART              the kp=6 demonstrator with action noise, clean command recorded
  MPPI-rollouts     sampling MPC on nominal kinematics (no terrain)
  GC-diff-rollouts  goal-conditioned diffusion policy trained on the demos, own sampling noise
  MPC-relabel       BRIDGE-slip states, MPPI labels
  PD-relabel        BRIDGE-slip states, PD labels          (fallback 2x2, SEARCH_PLAN 0.5)
  BRIDGE-on-PD      PD-noise states, bridge labels          (fallback 2x2, SEARCH_PLAN 0.5)
  MPC-oracle        MPPI with the true map, rolled out in the true env under slip
"""
from pathlib import Path

import torch

from sb.core import se2 as S
from sb.envs.gen_task import GenTask, make_ref
from sb.gen import controllers as C
# the gate trained its nets sequentially (registered 0.2); its path stays sequential
from sb.gen.bridges import BRIDGE_CFG, get_bridges
from sb.gen.coverage import coverage, noised_copies
from sb.gen.noise import NoiseStream, iso_variance
from sb.gen.rollout import rollout, rollout_true
from sb.policies.common import OBS_GC_DIM, obs_gc
from sb.policies.diffusion import DiffPolicy, train_diffusion

SOURCES = ("DEMO", "NOISED", "BRIDGE-slip", "BRIDGE-unicycle", "PD-noise", "PD-iso", "DART",
           "MPPI-rollouts", "GC-diff-rollouts", "MPC-relabel", "MPC-oracle")
EXTRA = ("BRIDGE-brownian", "PD-world", "PD-relabel", "BRIDGE-on-PD")
IN_FAMILY = tuple(s for s in SOURCES if s != "MPC-oracle")


def start_states(tk, m, seed, device):
    """rho_0 samples shared by every in-family source for a given seed, so the sources
    differ only in the controller (the task's own generator would advance between builds)."""
    g = torch.Generator(device=device).manual_seed(80_000 + seed)
    return S.sample_marginal(tk.means[0], tk.covs[0], m, device, g)


def gc_generator(tk, Gd, Ud, layout, seed, device, models_dir, steps, tag=""):
    p = Path(models_dir) / f"gcgen_{layout}{tag}_n{Gd.shape[0]}_st{steps}_s{seed}.pt"
    pol = DiffPolicy(obs_dim=OBS_GC_DIM).to(device)
    if p.exists():
        pol.load_state_dict(torch.load(p, map_location=device, weights_only=True)); pol.eval()
        return pol
    pol = train_diffusion(tk, Gd, Ud, seed, device, steps, obs_fn=obs_gc, obs_dim=OBS_GC_DIM)
    p.parent.mkdir(parents=True, exist_ok=True); torch.save(pol.state_dict(), p)
    return pol


def build(source, tk, layout, seed, device, demos, models_dir, n_demo=20, mult=4, ref_kind="slip", mf=S.SE2,
          tag="", world_heading=None, cache=None, gc_steps=8000, mppi_cost="greedy", cfg=BRIDGE_CFG):
    cache = cache if cache is not None else {}
    Gd, Ud = demos[0][:n_demo], demos[1][:n_demo]
    m = mult * n_demo
    z = NoiseStream(m, seed, device)
    starts = start_states(tk, m, seed, device)
    if source == "DEMO":
        return Gd, Ud, dict(n_gen=0), None
    if source == "NOISED":
        Gn, Un = noised_copies(Gd, Ud, m, seed, device)
        cov, hist = coverage(Gn, demos[0])
        return torch.cat([Gd, Gn]), torch.cat([Ud, Un]), dict(n_gen=m, **cov), hist
    ref = make_ref(ref_kind, tk)
    if hasattr(tk, "body_std"):
        ref.body_cov = tk.body_std ** 2
    extra = {}

    def bridge_states():
        key = ("bridge_states", layout, seed, tag)
        if key not in cache:
            nets = get_bridges("slip", tk, layout, seed, device, models_dir, tag, mf, cfg, workers=1)
            cache[key] = rollout(tk, mf, C.bridge_act(nets, mf, tk), m, z, ref, starts=starts)[0]
        return cache[key]

    if source.startswith("BRIDGE-on-PD"):
        Gp, _ = rollout(tk, mf, C.pd_act(Gd, Ud), m, z, ref, starts=starts)
        nets = get_bridges("slip", tk, layout, seed, device, models_dir, tag, mf, cfg, workers=1)
        Gg, Ug = rollout(tk, mf, C.bridge_act(nets, mf, tk), m, z, ref, states=Gp)
    elif source.startswith("BRIDGE"):
        kind = source.split("-")[1]
        r = make_ref(kind, tk)
        if hasattr(tk, "body_std"):
            r.body_cov = tk.body_std ** 2
        nets = get_bridges(kind, tk, layout, seed, device, models_dir, tag, mf, cfg, workers=1)
        Gg, Ug = rollout(tk, mf, C.bridge_act(nets, mf, tk), m, z, r, starts=starts)
    elif source == "PD-noise":
        Gg, Ug = rollout(tk, mf, C.pd_act(Gd, Ud), m, z, ref, starts=starts)
    elif source == "PD-world":
        Gg, Ug = rollout(tk, mf, C.pd_act(Gd, Ud), m, z, ref, world_heading=world_heading, starts=starts)
    elif source == "PD-iso":
        v = iso_variance(ref, mf, bridge_states())
        extra["iso_var"] = v
        Gg, Ug = rollout(tk, mf, C.pd_act(Gd, Ud), m, z, ref, iso_var=v, starts=starts)
    elif source == "DART":
        Gg, Ug = rollout(tk, mf, C.nominal_act(tk), m, z, ref, action_noise_ref=ref, starts=starts)
    elif source == "MPPI-rollouts":
        Gg, Ug = rollout(tk, mf, C.mppi_act(tk, seed, mppi_cost), m, z, ref, starts=starts)
    elif source == "GC-diff-rollouts":
        pol = gc_generator(tk, Gd, Ud, layout, seed, device, models_dir, gc_steps, tag)
        Gg, Ug = rollout(tk, mf, C.gc_diffusion_act(pol, tk, seed), m, z, ref, starts=starts)
    elif source == "MPC-relabel":
        Gg, Ug = rollout(tk, mf, C.mppi_act(tk, seed, mppi_cost), m, z, ref, states=bridge_states())
    elif source == "PD-relabel":
        Gg, Ug = rollout(tk, mf, C.pd_act(Gd, Ud), m, z, ref, states=bridge_states())
    elif source == "MPC-oracle":
        tk_true = GenTask(tk.layout, "slip", m, 60_000 + seed, device, slip_scale=tk.slip_scale)
        Gg, Ug, out = rollout_true(tk_true, C.oracle_act(tk_true, seed, mppi_cost), m)
        extra["gen_collision"] = float(1 - out["alive"].float().mean())
        extra["gen_success"] = float(out["success"].float().mean())
    else:
        raise ValueError(source)
    cov, hist = coverage(Gg, demos[0])
    return torch.cat([Gd, Gg]), torch.cat([Ud, Ug]), dict(n_gen=m, **cov, **extra), hist
