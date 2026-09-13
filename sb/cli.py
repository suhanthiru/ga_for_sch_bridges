"""python -m sb.cli <command>

  demos                 regenerate the demo sets on CPU (seed 4242) and print their sha256
  ingest-prior          copy the prior g0/g1 shards from skill_chains into results/gate/prior
  diag relabel          the MPC-relabel diagnostic from SEARCH_PLAN 0.5
  dr-check              the dynamic-range check from SEARCH_PLAN 0.4 on the prior g0 table
  gate --seed S         run this seed's cells (only those the prior run lacks for seeds 0-4)
  report gate           write results/gate/tables.md and fill FINDINGS_generator.md
  freeze [--pilot] [--filter none|no_bridge|no_rl]   run component tests, write grammar/manifest.json
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from sb import settings

PRIOR_DEFAULT = Path(os.environ.get("SB_PRIOR", r"D:\s_bridges\skill_chains\results_generator"))


def prior_dir():
    local = settings.GATE / "prior"
    return local if any(local.glob("phaseg*_seed*.parquet")) else PRIOR_DEFAULT


def cmd_demos(a):
    import hashlib
    from sb.policies.demos import make_demos
    for layout in ("L1", "L2"):
        p = settings.demo_path(layout)
        s = make_demos(layout, 500, p, seed=4242)
        print(f"{layout}: nominal success {s:.3f}  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}  {p}")


def cmd_ingest_prior(a):
    dst = settings.GATE / "prior"; dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(PRIOR_DEFAULT.glob("phaseg[01]_seed[0-4].parquet")):
        shutil.copy2(f, dst / f.name); n += 1
    print(f"copied {n} prior shards to {dst}")


def cmd_diag(a):
    from sb.envs.gen_task import GenTask  # noqa: F401  (import check)
    from sb.gen.diagnostics import relabel_diag
    from sb.policies.demos import load_demos
    dev = settings.device()
    demos = load_demos(settings.demo_path("L1"), dev)
    t0 = time.time(); out = relabel_diag(demos, "L1", 0, 64, dev, cost=a.cost)
    out["seconds"] = round(time.time() - t0, 1)
    settings.GATE.mkdir(parents=True, exist_ok=True)
    p = settings.GATE / f"diag_relabel{'' if a.cost == 'greedy' else '_' + a.cost}.json"
    p.write_text(json.dumps(out, indent=1) + "\n", newline="\n"); print(json.dumps(out, indent=1)); print("->", p)


def cmd_dr_check(a):
    from sb.gate.report import dynamic_range
    from sb.gate.cells import load_prior
    dr, mx = dynamic_range(load_prior(prior_dir()))
    print(mx.to_string() if mx is not None else "no prior g0 shards"); print(dr)


def cmd_gate(a):
    from sb.gate import cells as CL
    from sb.gate.run_gate import preflight, run_seed
    out, models = settings.GATE, settings.GATE / "models"
    models.mkdir(parents=True, exist_ok=True)
    cells = None
    if a.sources:
        cells = [(s, l) for l in CL.LAYOUTS for s in a.sources.split(",")]
    if a.smoke:
        out = settings.RESULTS / "gate_smoke"; models = out / "models"; models.mkdir(parents=True, exist_ok=True)
    preflight(out, allow_dirty=a.allow_dirty, skip_tests=a.skip_tests, extra=dict(seed=a.seed, smoke=a.smoke, mppi_cost=a.cost))
    kw = dict(steps=300, n_eval=30) if a.smoke else {}
    t0 = time.time()
    done = run_seed(a.seed, out, models, cells=cells, mppi_cost=a.cost, run="smoke" if a.smoke else "gate", **kw)
    print(f"seed {a.seed}: {len(done)} cells in {time.time() - t0:.0f}s")


def cmd_report(a):
    if a.what == "gate":
        from sb.gate.report import build_tables, fill_findings
        from sb.gate.run_gate import git
        T, dec = build_tables(settings.GATE, prior_dir())
        fill_findings(T, commit=git("rev-parse", "--short", "HEAD"))
        print("verdict:", dec["verdict"] if dec else "not available"); print("tables ->", settings.GATE / "tables.md")
        return
    # search: rebuild the search state from a results directory and write the interim report
    from pathlib import Path
    from sb.core.genome import Genome
    from sb.search.freeze import load_frozen
    from sb.search.loop import Search, dummy_evaluate
    from sb.search.report import interim
    root = Path(a.root); G, man = load_frozen(pilot=a.pilot)
    gdir = settings.ROOT / ("grammar_pilot" if a.pilot else "grammar")
    seeds = [Genome.from_json(l) for l in (gdir / "seeds.jsonl").read_text().splitlines() if l.strip()]
    s = Search(G, root, seeds, dummy_evaluate, n_cells=len(__import__("numpy").load(root / "cvt_centroids.npy")), log=lambda m: None)
    if not s.resume():
        raise SystemExit("no verified checkpoint under " + str(root))
    out = root / "interim.md"
    interim(s, out, figure=root / "map.png")
    print("interim ->", out)


def cmd_audit(a):
    from pathlib import Path
    from sb.envs.audit import audit, render
    import torch
    rows = audit(n=a.n, device=torch.device(a.device) if a.device else None, ids=a.ids.split(",") if a.ids else None)
    out = Path(settings.ROOT / "findings" / "oracle_audit.md"); out.parent.mkdir(exist_ok=True)
    out.write_text(render(rows, a.n), encoding="utf-8", newline="\n"); print("->", out)


def cmd_freeze(a):
    from sb.search.freeze import freeze
    G, out = freeze(filt=a.filter, pilot=a.pilot, run_tests=not a.skip_tests)
    print(f"frozen {len(G.registry)} components, hash {G.hash} -> {out}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demos").set_defaults(f=cmd_demos)
    sub.add_parser("ingest-prior").set_defaults(f=cmd_ingest_prior)
    d = sub.add_parser("diag"); d.add_argument("what", choices=["relabel"]); d.add_argument("--cost", default="greedy", choices=["greedy", "track"]); d.set_defaults(f=cmd_diag)
    sub.add_parser("dr-check").set_defaults(f=cmd_dr_check)
    g = sub.add_parser("gate"); g.add_argument("--seed", type=int, required=True); g.add_argument("--sources", default="")
    g.add_argument("--smoke", action="store_true"); g.add_argument("--allow-dirty", action="store_true"); g.add_argument("--skip-tests", action="store_true")
    g.add_argument("--cost", default="greedy", choices=["greedy", "track"]); g.set_defaults(f=cmd_gate)
    r = sub.add_parser("report"); r.add_argument("what", choices=["gate", "search"]); r.add_argument("--root", default="results/search/mapelites")
    r.add_argument("--pilot", action="store_true"); r.set_defaults(f=cmd_report)
    au = sub.add_parser("audit"); au.add_argument("--n", type=int, default=64); au.add_argument("--device", default=""); au.add_argument("--ids", default="")
    au.set_defaults(f=cmd_audit)
    f = sub.add_parser("freeze"); f.add_argument("--pilot", action="store_true"); f.add_argument("--filter", default="none", choices=["none", "no_bridge", "no_rl"])
    f.add_argument("--skip-tests", action="store_true"); f.set_defaults(f=cmd_freeze)
    a = ap.parse_args(); a.f(a)


if __name__ == "__main__":
    main()
