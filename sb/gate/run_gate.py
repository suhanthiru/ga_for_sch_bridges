"""Run the gate cells for one seed, one shard per (source, layout), crash-safe and resumable.

preflight() refuses to start unless the fast test suite is green and the tree is clean
(or --allow-dirty), and writes provenance.json next to the shards.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sb import settings
from sb.envs import terrain as TR
from sb.envs.gen_task import GenTask
from sb.gate import cells as CL
from sb.gen.sources import build
from sb.policies.demos import load_demos
from sb.policies.diffusion import train_diffusion
from sb.policies.evaluate import evaluate


def git(*args):
    return subprocess.run(["git", *args], cwd=settings.ROOT, capture_output=True, text=True).stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dirty():
    """Uncommitted changes outside results/ (shards and timing files are written by the run itself)."""
    return git("status", "--porcelain", "--", ".", ":!results")


def preflight(out_dir, allow_dirty=False, skip_tests=False, extra=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if dirty() and not allow_dirty:
        raise SystemExit("working tree is dirty; commit first or pass --allow-dirty")
    if not skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow"], cwd=settings.ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("tests are red; refusing to run the gate\n" + r.stdout[-2000:])
    prov = dict(commit=git("rev-parse", "HEAD"), dirty=bool(dirty()), python=sys.version.split()[0],
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                n_demo=CL.N_DEMO, gen_mult=CL.GEN_MULT, steps=CL.STEPS, n_eval=CL.N_EVAL, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                demos={l: sha256(settings.demo_path(l)) for l in CL.LAYOUTS if settings.demo_path(l).exists()})
    prov.update(extra or {})
    p = out_dir / "provenance.json"
    old = json.loads(p.read_text()) if p.exists() else []
    old.append(prov); p.write_text(json.dumps(old, indent=1) + "
", newline="
")
    return prov


def _timing(out_dir, key, secs):
    p = Path(out_dir) / "timing.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old[key] = round(secs, 1); p.write_text(json.dumps(old, indent=1) + "
", newline="
")


def run_seed(seed, out_dir, models_dir, cells=None, steps=CL.STEPS, n_eval=CL.N_EVAL, n_demo=CL.N_DEMO, mult=CL.GEN_MULT,
             device=None, mppi_cost="greedy", run="gate", log=print):
    device = device or settings.device()
    out_dir, models_dir = Path(out_dir), Path(models_dir)
    sdir = out_dir / f"seed{seed}"; (sdir / "episodes").mkdir(parents=True, exist_ok=True); (sdir / "coverage").mkdir(exist_ok=True)
    commit = git("rev-parse", "--short", "HEAD")
    cells = cells if cells is not None else CL.cells_for(seed)
    done = []
    for layout in CL.LAYOUTS:
        todo = [s for s, l in cells if l == layout]
        if not todo:
            continue
        tk = GenTask(TR.Layout(layout), "none", 64, 1000 + seed, device)
        demos = load_demos(settings.demo_path(layout), device)
        evals = [(d, GenTask(TR.Layout(layout), d, n_eval, 50_000 + seed, device)) for d in CL.DISTS]
        cache = {}
        for source in todo:
            shard = sdir / f"{source}_{layout}.parquet"
            if shard.exists():
                log(f"[gate s{seed} {layout} {source}] shard exists, skipping"); continue
            t0 = time.time()
            G, U, meta, hist = build(source, tk, layout, seed, device, demos, models_dir, n_demo=n_demo, mult=mult,
                                     cache=cache, gc_steps=steps, mppi_cost=mppi_cost)
            pol = train_diffusion(tk, G, U, seed, device, steps)
            torch.save(pol.state_dict(), models_dir / f"pol_{source}_{layout}_s{seed}.pt")
            train_s = time.time() - t0
            rows, eps = [], []
            for name, tke in evals:
                scal, ep = evaluate(pol, tke, n_eval)
                rows.append(CL.make_row(run=run, source=source, layout=layout, disturbance=name, seed=seed, n_demo=n_demo,
                                        gen_mult=mult, steps=steps, n_eval=n_eval, mppi_cost=mppi_cost, train_s=train_s,
                                        commit=commit, **{k: v for k, v in meta.items() if k in CL.SCHEMA}, **scal))
                eps.append(pd.DataFrame(dict(disturbance=name, episode=np.arange(n_eval), **ep)))
            pd.concat(eps, ignore_index=True).to_parquet(sdir / "episodes" / f"{source}_{layout}.parquet", index=False)
            if hist is not None:
                np.savez_compressed(sdir / "coverage" / f"{source}_{layout}.npz", hist=hist)
            CL.frame(rows).to_parquet(shard, index=False)
            _timing(out_dir, f"s{seed}_{layout}_{source}", time.time() - t0)
            log(f"[gate s{seed} {layout} {source}] " + " ".join(f"{r['disturbance']}:{r['success']:.2f}" for r in rows)
                + f" cov {meta.get('cov_cells', '-')} ({time.time() - t0:.0f}s)")
            done.append((source, layout))
    return done


def load_gate(out_dir):
    import glob
    fs = sorted(glob.glob(str(Path(out_dir) / "seed*" / "*.parquet")))
    if not fs:
        return CL.frame([])
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def load_episodes(out_dir):
    import glob
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "seed*" / "episodes" / "*.parquet"))):
        p = Path(f); seed = int(p.parent.parent.name[4:]); source, layout = p.stem.rsplit("_", 1)
        d = pd.read_parquet(f); d["seed"], d["source"], d["layout"] = seed, source, layout
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["seed", "source", "layout", "disturbance", "episode", "success", "progress", "collision", "energy"])
