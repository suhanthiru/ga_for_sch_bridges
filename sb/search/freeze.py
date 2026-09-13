"""Freeze the grammar: every component's declared test must pass, every bridge component
needs a substitute for each slot it fills, and the manifest with its hash, the innovation
table and the seed population are written to grammar/. After this only
grammar/disabled.json may change (shrink-only), and every archive row carries the hash.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sb import settings
from sb.components import load_all
from sb.core.grammar import NO_BRIDGE, NO_RL, PILOT_ROOT_SLOTS, ROOT_SLOTS, Grammar
from sb.core.novelty import SeedSet
from sb.core.substitute import coverage_check

FILTERS = {"none": None, "no_bridge": NO_BRIDGE, "no_rl": NO_RL}


def run_component_tests(registry):
    ids = sorted({s.test for s in registry.values() if s.test})
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *ids], cwd=settings.ROOT, capture_output=True, text=True)
    return r.returncode == 0, r.stdout[-3000:]


def freeze(out_dir=None, filt="none", n_random=200, seed=0, run_tests=True, pilot=False):
    reg = load_all()
    missing = [k for k, s in reg.items() if not s.test]
    if missing:
        raise SystemExit(f"components without a test are not in the grammar: {missing}")
    G = Grammar(reg, root_slots=PILOT_ROOT_SLOTS if pilot else ROOT_SLOTS, filt=FILTERS[filt])
    problems = coverage_check(G)
    if problems:
        raise SystemExit("substitution table incomplete:\n" + "\n".join(problems))
    if run_tests:
        ok, log = run_component_tests(reg)
        if not ok:
            raise SystemExit("component tests failed; grammar not frozen\n" + log)
    rng = np.random.default_rng(seed)
    seeds = [G.random_genome(rng, 0.6) for _ in range(n_random)]
    out = Path(out_dir or settings.ROOT / ("grammar_pilot" if pilot else "grammar")); out.mkdir(parents=True, exist_ok=True)
    man = G.manifest(); man.update(hash=G.hash, filter=filt, frozen=time.strftime("%Y-%m-%d %H:%M:%S"), pilot=pilot, n_random_seeds=n_random)
    (out / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", newline="\n")
    (out / "manifest.sha256").write_text(G.hash + "\n", newline="\n")
    (out / "seeds.jsonl").write_text("".join(g.to_json() + "\n" for g in seeds), newline="\n")
    (out / "seed_set.json").write_text(json.dumps(SeedSet.from_genomes(seeds).to_json()) + "\n", newline="\n")
    if not (out / "disabled.json").exists():
        (out / "disabled.json").write_text("[]\n", newline="\n")
    return G, out


def load_frozen(out_dir=None, pilot=False):
    """The grammar as frozen, with disabled.json applied; refuses to run if the registry's
    hash no longer matches the manifest (the code changed after the freeze)."""
    from dataclasses import replace
    out = Path(out_dir or settings.ROOT / ("grammar_pilot" if pilot else "grammar"))
    man = json.loads((out / "manifest.json").read_text())
    reg = dict(load_all())
    G = Grammar(reg, root_slots=PILOT_ROOT_SLOTS if pilot else ROOT_SLOTS, filt=FILTERS[man["filter"]])
    if G.hash != man["hash"]:
        raise SystemExit(f"registry hash {G.hash} differs from the frozen manifest {man['hash']}; the grammar changed after the freeze")
    for d in json.loads((out / "disabled.json").read_text()):
        reg[d["key"]] = replace(reg[d["key"]], disabled=d["reason"])
    return Grammar(reg, root_slots=PILOT_ROOT_SLOTS if pilot else ROOT_SLOTS, filt=FILTERS[man["filter"]]), man
