"""The evaluation contract: evaluate(genome, cell, rung, seed) -> EvalResult.

Stack.compile turns a genome into a controller for a task: the manifold, the controller
(with its sub-slot reference / data), the safety wrapper and the execution noise. Train-
time components get the capability object; the rollout that scores the stack never sees
it (the oracle counter is armed around the episodes). The ablation delta at rung 2 is
the same evaluation on ablate_bridges(genome).
"""
import hashlib
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from sb import settings
from sb.core.genome import ROOT
from sb.core.substitute import ablate_bridges
from sb.envs import task as TK
from sb.envs import terrain as TR
from sb.envs.base import Caps
from sb.envs.gen_task import GenTask
from sb.policies.common import obs_of
from sb.policies.demos import load_demos
from sb.rl.pop_ppo import PopEnv, PopPPO
from sb.search.invariants import check_episode_set, tripped

DISTS = ("none", "slip", "rain", "push")
RUNGS = {0: dict(seeds=1, episodes=30, rl_steps=100_000), 1: dict(seeds=2, episodes=100, rl_steps=500_000),
         2: dict(seeds=10, episodes=200, rl_steps=2_000_000)}


@dataclass
class Cell:
    cell_id: int
    layout: str = "L1"
    disturbances: tuple = DISTS
    descriptor: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    eval_id: str
    gid: str
    sid: str
    cell_id: int
    rung: int
    seed: int
    valid: bool = True
    invalid_reason: str = ""
    fitness: float = float("nan")
    success: float = float("nan")
    collision: float = float("nan")
    energy: float = float("nan")
    cvar_01: float = float("nan")
    worst_of_20: float = float("nan")
    per_disturbance: dict = field(default_factory=dict)
    ablation_delta: float = float("nan")
    ablation_eval_id: str = ""
    has_bridge: bool = False
    has_rl: bool = False
    train_s: float = 0.0
    eval_s: float = 0.0
    infer_ms: float = float("nan")
    oracle_reads_at_test: int = 0
    invariants: dict = field(default_factory=dict)
    quarantined: bool = False
    error: str = ""

    def row(self):
        d = asdict(self); d.pop("per_disturbance"); d.pop("invariants")
        for k, v in self.per_disturbance.items():
            d[f"success_{k}"] = v
        for k, v in self.invariants.items():
            d[f"inv_{k}"] = v
        return d


class OracleGuard:
    """Counts oracle reads while armed; the evaluator arms it around test episodes."""
    count = 0
    armed = False

    @classmethod
    def touch(cls):
        if cls.armed:
            cls.count += 1


def train_rl(recipe, tk, seed, steps, device, n_envs=512, rollout=32, log=lambda m: None):
    """PPO for one genome: BC warm start on the recipe's data, then `steps` environment
    steps of PPO. Returns a controller (g, k, tau, step) -> (u, None)."""
    G, U = recipe["data"]
    env = PopEnv(tk, 1, n_envs)
    ppo = PopPPO(1, device, lr=recipe["lr"], clip=recipe["clip"], ent=recipe["ent"], seed=seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    ppo.warm_start_bc(tk, G, U, gen=gen)
    obs = env.reset()
    for _ in range(max(1, steps // (n_envs * rollout))):
        obs, info = ppo.update(env, obs, rollout=rollout, epochs=4, minibatch=4096, gen=gen)

    def ctl(g, k, tau, step):
        return ppo.act(obs_of(tk, g, k, tau), g.shape[0]), None
    return ctl


class Stack:
    """Compiled genome: build every node's object bottom-up, wire the controller."""

    def __init__(self, genome, grammar, task, demos, models, seed, caps):
        self.g, self.grammar, self.task, self.demos, self.models, self.seed, self.caps = genome, grammar, task, demos, models, seed, caps
        self.built = {}
        self.manifold = self._build_slot(ROOT, "manifold")
        self.controller = self._build_slot(ROOT, "controller")
        self.safety = self._build_slot(ROOT, "safety") or (lambda u: u)
        self.noise = self._build_slot(ROOT, "noise")
        self.trigger = self._build_slot(ROOT, "trigger"); self.seam = self._build_slot(ROOT, "seam")
        self.value = self._build_slot(ROOT, "value")

    def _ctx(self, sub):
        return dict(task=self.task, demos=self.demos, models=self.models, seed=self.seed, manifold=getattr(self, "manifold", None), sub=sub, caps=self.caps)

    def _build_slot(self, parent, slot):
        nid = self.g.child_in(parent, slot)
        return None if nid is None else self._build_node(nid)

    def _build_node(self, nid):
        if nid in self.built:
            return self.built[nid]
        node = self.g.node(nid); spec = self.grammar.spec(node.comp)
        sub = {name: self._build_node(self.g.child_in(nid, name)) for name in spec.sub_slots if self.g.child_in(nid, name) is not None}
        if spec.oracle != "none":
            OracleGuard.touch()
        obj = spec.cls().build(dict(node.params), self._ctx(sub))
        self.built[nid] = obj
        return obj

    def act(self, g, k, tau, step):
        ctl = self.controller
        if isinstance(ctl, dict):                            # RL placeholders are trained by the evaluator, not here
            raise RuntimeError("RL controllers are trained by evaluate(); compile got an untrained one")
        u, extra = ctl(g, k, tau, step)
        if self.noise is not None:
            sig = self.noise["sigma"] if self.noise["kind"] == "fixed" else self.noise["sigmas"][min(k, 2)]
            u = u + sig * torch.randn_like(u)
        return self.safety(u), extra


def eval_id_of(genome, cell_id, rung, seed):
    return hashlib.sha256(f"{genome.gid}|{cell_id}|{rung}|{seed}".encode()).hexdigest()[:16]


class _Recorder:
    """Wraps the stack's act to keep the executed actions for the invariant checks."""

    def __init__(self, act):
        self.act, self.U = act, []

    def __call__(self, g, k, tau, step):
        u, extra = self.act(g, k, tau, step); self.U.append(TK.clip_u(u).detach().clone()); return u, extra


@torch.no_grad()
def score(stack, cell, seed, episodes, device):
    """Episodes over the cell's disturbances with the oracle guard armed, plus the exploit
    invariants on the recorded trajectories and actions (worst fraction over disturbances)."""
    per, succ_all, coll, energy, inv = {}, [], [], [], {}
    OracleGuard.armed, OracleGuard.count = True, 0
    t0 = time.time()
    try:
        for d in cell.disturbances:
            tk = GenTask(TR.Layout(cell.layout), d, episodes, 50_000 + seed, device, **{k: v for k, v in cell.descriptor.items() if k in ("slip_scale", "aniso", "n_seed", "push_mult", "heading_std")})
            rec = _Recorder(stack.act)
            out = tk.rollout(rec, n=episodes, record=True)
            per[d] = float(out["success"].float().mean())
            succ_all.append(out["progress"].cpu().numpy()); coll.append(float(1 - out["alive"].float().mean())); energy.append(float(out["energy"].mean()))
            geo = dict(wall_x=0.5, gaps=list(tk.layout.gaps), pile=tk.layout.pile, means=[m.cpu() for m in tk.means], covs=[c.cpu() for c in tk.covs])
            chk = check_episode_set(out["traj"], torch.stack(rec.U, 1), out["success"], geo)
            for k, v in chk.items():
                inv[k] = max(inv.get(k, 0.0), v)
    finally:
        OracleGuard.armed = False
    prog = np.concatenate(succ_all)
    k20 = max(1, len(prog) // 20)
    return dict(per=per, success=float(np.mean(list(per.values()))), collision=float(np.mean(coll)), energy=float(np.mean(energy)),
                cvar_01=float(np.sort(prog)[:max(1, int(np.ceil(0.1 * len(prog))))].mean()), worst_of_20=float(np.sort(prog)[:k20].mean()),
                eval_s=time.time() - t0, oracle_reads=OracleGuard.count, invariants=inv)


def latency_ms(stack, device):
    tk = stack.task; g = tk.sample(0, 1); tau = torch.zeros(1, device=device)
    torch.set_num_threads(1)
    for _ in range(5):
        stack.act(g, 0, tau, 0)
    t0 = time.perf_counter()
    for i in range(50):
        stack.act(g, 0, tau, i)
    return (time.perf_counter() - t0) / 50 * 1e3


def fitness_of(success, collision, train_s, pd_train_s=1.0):
    return success - 0.02 * np.log(max(train_s, 1e-3) / pd_train_s + 1.0) - 0.05 * collision


def evaluate(genome, cell, rung, seed, grammar, models_dir=None, device=None, episodes=None, ablate=None, demos=None, rl_steps=None):
    device = device or settings.device(); cfg = RUNGS[rung]; episodes = episodes or cfg["episodes"]
    models_dir = models_dir or (settings.RESULTS / "search" / "models"); models_dir.mkdir(parents=True, exist_ok=True)
    res = EvalResult(eval_id_of(genome, cell.cell_id, rung, seed), genome.gid, genome.sid, cell.cell_id, rung, seed,
                     has_bridge=grammar.has_tag(genome, "bridge"), has_rl=grammar.has_tag(genome, "rl"))
    v = grammar.validate(genome)
    if v:
        res.valid, res.invalid_reason = False, "; ".join(v); return res
    try:
        tk = GenTask(TR.Layout(cell.layout), "none", 64, 1000 + seed, device)
        demos = demos if demos is not None else load_demos(settings.demo_path(cell.layout), device)
        t0 = time.time()
        stack = Stack(genome, grammar, tk, demos, models_dir, seed, Caps())
        if isinstance(stack.controller, dict) and stack.controller.get("kind") == "ppo":
            stack.controller = train_rl(stack.controller, tk, seed, rl_steps if rl_steps is not None else cfg["rl_steps"], device,
                                        n_envs=min(512, max(8, episodes * 4)))
        res.train_s = time.time() - t0
        res.infer_ms = latency_ms(stack, device) if device.type == "cpu" else float("nan")
        if np.isfinite(res.infer_ms) and res.infer_ms > 50:
            res.valid, res.invalid_reason = False, f"inference {res.infer_ms:.1f} ms > 50 ms"; return res
        sc = score(stack, cell, seed, episodes, device)
        res.per_disturbance, res.success, res.collision, res.energy = sc["per"], sc["success"], sc["collision"], sc["energy"]
        res.cvar_01, res.worst_of_20, res.eval_s, res.oracle_reads_at_test = sc["cvar_01"], sc["worst_of_20"], sc["eval_s"], sc["oracle_reads"]
        res.invariants = sc["invariants"]; res.quarantined = bool(tripped(sc["invariants"]))
        if sc["oracle_reads"]:
            res.valid, res.invalid_reason = False, "oracle read at test time"; return res
        res.fitness = fitness_of(res.success, res.collision, res.train_s)
        if (ablate if ablate is not None else rung >= 2) and res.has_bridge:
            ab = evaluate(ablate_bridges(genome, grammar), cell, rung, seed, grammar, models_dir, device, episodes, ablate=False, demos=demos, rl_steps=rl_steps)
            res.ablation_delta, res.ablation_eval_id = res.fitness - ab.fitness, ab.eval_id
    except Exception as e:                                   # logged, never silently dropped
        res.valid, res.invalid_reason, res.error = False, "exception", f"{type(e).__name__}: {e}"
    return res
