"""The search driver: frozen grammar, elite map, archive, checkpoints, resume.

    python search.py --plan SEARCH_PLAN.md --budget 20000 --resume

One process for now (the island model and the GPU server come with the pilot); the
evaluation function is injected so the loop can be exercised end-to-end with a stand-in
evaluator, which is what the kill-and-resume acceptance test does.
"""
import json
import time
from pathlib import Path

import numpy as np

from sb.core.genome import Genome, Provenance
from sb.core.novelty import SeedSet, novelty
from sb.envs.family import AXES
from sb.search.algos.cma_emitter import CMAEmitter
from sb.search.archive import Archive
from sb.search.elites import EliteMap
from sb.search.ladder import HOLD, after_rung0, due_for_validation, is_jump
from sb.search.surrogate import Surrogate

CHECKPOINT_EVERY = 100


def dummy_evaluate(genome, cell_desc, rung, seed, grammar, weights=None):
    """A stand-in with the evaluator's row shape: success from a hash of the genome and
    the cell (deterministic and structured), a synthetic collision rate and compute, and
    the fitness recombined with the same weights the real evaluator uses."""
    from sb.search.evaluate import fitness_of
    h = int(genome.gid[:8], 16) ^ int(abs(hash(tuple(round(x, 3) for x in cell_desc))) % (1 << 30))
    rng = np.random.default_rng(h % (1 << 32))
    succ = float(rng.beta(2, 3)) + 0.02 * len(genome.nodes) + 0.02 * np.random.default_rng(seed + 7).normal()
    coll = float(rng.beta(1, 8)); train_s = float(1 + 20 * rng.random())
    f = fitness_of(succ, coll, train_s, weights=weights)
    hb = grammar.has_tag(genome, "bridge")
    return dict(valid=True, fitness=f, success=succ, collision=coll, energy=1.0, cvar_01=succ / 2, worst_of_20=succ / 3, train_s=train_s, eval_s=0.0,
                has_bridge=hb, has_rl=grammar.has_tag(genome, "rl"), invalid_reason="", error="", quarantined=False,
                ablation_delta=(0.05 if (rung >= 2 and hb) else float("nan")))


class Search:
    def __init__(self, grammar, root, seeds, evaluate, n_cells=2000, cvt_seed=0, algorithm="mapelites", seed=0, log=print, cells=None):
        """cells: optional list of descriptor vectors the search may propose (rung 0 uses the
        eight fixed cells); None draws uniformly over the bins."""
        self.G, self.root, self.evaluate, self.log = grammar, Path(root), evaluate, log
        self.cells = [list(map(float, c)) for c in cells] if cells else None
        self.archive = Archive(self.root)
        self.map = EliteMap.load_or_make(self.root / "cvt_centroids.npy", len(AXES), n_cells, cvt_seed)
        self.seeds = list(seeds); self.seed_set = SeedSet.from_genomes(self.seeds)
        self.rng = np.random.default_rng(seed); self.algorithm = algorithm
        self.gen, self.n_evals, self.pending = 0, 0, list(range(len(self.seeds)))
        self.emitters, self.queue, self.batch = {}, [], None      # CMA-MAE: per-structure emitters, queued samples, open batch
        self.p_cma = 0.3 if algorithm == "mapelites" else 0.0
        self.surrogate = Surrogate(grammar, seed=seed) if algorithm == "mapelites" else None
        self.recent_sids = []
        # seeds per rung (SEARCH_PLAN 2.8); tests shrink these
        self.rung_seeds = {1: (1, 2), 2: tuple(range(10, 20))}
        self.validated = {}                                    # cell -> dict(gid, fitness, ablation_delta, eval_ids)

    # ---------------------------------------------------------------- cells
    def random_cell_desc(self):
        if self.cells:
            return list(self.cells[int(self.rng.integers(len(self.cells)))])
        return [float(self.rng.integers(0, 4)) / 3 for _ in AXES]

    # ----------------------------------------------------------------- step
    def _cma_batch(self):
        """Ask one structure's emitter for a batch aimed at that elite's cell."""
        cells = list(self.map.elite)
        c = cells[int(self.rng.integers(len(cells)))]; e = self.map.elite[c]
        parent = Genome.from_json(self.archive.genome_json(e["gid"]))
        em = self.emitters.get(parent.sid)
        if em is None:
            em = CMAEmitter(parent, self.G, seed=int(self.rng.integers(1 << 30)))
            if len(self.emitters) >= 64:
                self.emitters.pop(next(iter(self.emitters)))
            self.emitters[parent.sid] = em
        if not em.active:
            return False
        genomes = em.ask(self.gen)
        self.batch = dict(emitter=em, cell=c, elite_fitness=e["fitness"], fits=[], n=len(genomes))
        self.queue = [(g, e["desc"]) for g in genomes]
        return bool(genomes)

    def propose(self):
        """Next genome and cell: seeds first; then CMA batches on an elite's structure, or
        mutations / crossovers of random elites."""
        if self.pending:
            g = self.seeds[self.pending.pop(0)]; desc = self.random_cell_desc()
            return g, desc
        if self.queue:
            return self.queue.pop(0)
        if not self.map.elite or self.algorithm == "random":
            return self.G.random_genome(self.rng, 0.6, Provenance(algorithm=self.algorithm, generation=self.gen)), self.random_cell_desc()
        if self.rng.random() < self.p_cma and self._cma_batch():
            return self.queue.pop(0)
        cells = list(self.map.elite)
        c = cells[int(self.rng.integers(len(cells)))]; parent = Genome.from_json(self.archive.genome_json(self.map.elite[c]["gid"]))
        desc = self.map.elite[c]["desc"] if self.rng.random() < 0.7 else self.random_cell_desc()
        if self.surrogate is not None and self.surrogate.ready:
            # pre-screen ten mutants: the best three by the surrogate plus one at random
            cands = [self._offspring(parent, cells) for _ in range(10)]
            self.queue = [(g, desc) for g in self.surrogate.prescreen(cands, desc, self.rng, recent_sids=self.recent_sids[-50:])]
            return self.queue.pop(0)
        return self._offspring(parent, cells), desc

    def _offspring(self, parent, cells):
        if len(cells) > 1 and self.rng.random() < 0.2:
            c2 = cells[int(self.rng.integers(len(cells)))]; other = Genome.from_json(self.archive.genome_json(self.map.elite[c2]["gid"]))
            return self.G.crossover(parent, other, self.rng)
        return self.G.mutate(parent, self.rng)

    def _row(self, g, desc, cell, rung, seed, r):
        lvl, dist = novelty(g, self.seed_set)
        row = dict(eval_id=f"{g.gid}-{cell}-r{rung}-s{seed}-{self.n_evals}", gid=g.gid, sid=g.sid, genome_json=g.to_json(), dsl=g.dsl(),
                   grammar_hash=self.G.hash, algorithm=g.provenance.algorithm or self.algorithm, generation=self.gen, rung=rung, seed=seed,
                   cell=cell, novelty_level=lvl, seed_dist=dist, ts=time.time(), **{f"d_{a}": v for a, v in zip(AXES, desc)})
        row.update({k: v for k, v in r.items() if k not in row})          # the loop's identifiers win over the evaluator's
        self.archive.append(row, genome=g)
        return row

    def _rung(self, g, desc, cell, rung, seeds):
        """Evaluate at a rung over its seeds; mean fitness, or None if any seed is invalid or quarantined."""
        rows = [self._row(g, desc, cell, rung, sd, self.evaluate(g, desc, rung, sd, self.G)) for sd in seeds]
        if not all(r["valid"] for r in rows) or any(r.get("quarantined", False) for r in rows):
            return None, rows
        return float(np.mean([r["fitness"] for r in rows])), rows

    def step(self):
        g, desc = self.propose()
        cell = self.map.cell_of(desc)
        r = self.evaluate(g, desc, 0, 0, self.G)
        row = self._row(g, desc, cell, 0, 0, r)
        improved = False
        elite = self.map.elite.get(cell); elite_f = elite["fitness"] if elite else None
        if r["valid"] and not r.get("quarantined", False) and after_rung0(r["fitness"], elite_f).rung == 1:
            f1, _ = self._rung(g, desc, cell, 1, self.rung_seeds[1])
            if f1 is not None and is_jump(f1, elite_f):
                # a statistical anomaly is re-run on fresh seeds before it is believed (program section 10.2)
                f1b, _ = self._rung(g, desc, cell, 1, tuple(sd + 100 for sd in self.rung_seeds[1]))
                f1 = None if f1b is None else 0.5 * (f1 + f1b)
            if f1 is not None:
                improved = self.map.insert(cell, g.gid, f1, row["eval_id"], self.gen, desc)
        if self.batch is not None and g.provenance.op == "cma":
            self.batch["fits"].append(r["fitness"] if r["valid"] else -np.inf)
            if len(self.batch["fits"]) >= self.batch["n"]:
                self.batch["emitter"].tell(self.batch["fits"], self.batch["elite_fitness"]); self.batch = None
        self.n_evals += 1; self.gen += 1
        self.recent_sids.append(g.sid); self.recent_sids = self.recent_sids[-200:]
        if self.surrogate is not None and self.n_evals % 50 == 0 and self.surrogate.maybe_train(self.archive.frame(), self.n_evals):
            self.log(f"surrogate retrained at {self.n_evals}: held-out R2 {self.surrogate.r2:.3f}")
        if self.n_evals % CHECKPOINT_EVERY == 0:
            self.checkpoint()
        return row, improved

    # ---------------------------------------------------------- checkpoint
    def state(self):
        return dict(gen=self.gen, n_evals=self.n_evals, pending=self.pending, rng=self.rng.bit_generator.state, elites=self.map.state(),
                    algorithm=self.algorithm, grammar_hash=self.G.hash, validated={str(k): v for k, v in self.validated.items()},
                    surrogate_r2=(self.surrogate.r2 if self.surrogate is not None else None),
                    surrogate_trained_at=(self.surrogate.trained_at if self.surrogate is not None else None))

    def validate_due(self, final=False):
        """Rung 2 for elites that held their cell long enough (all of them when final)."""
        for c in due_for_validation(self.map, self.gen, final):
            e = self.map.elite[c]
            if self.validated.get(c, {}).get("gid") == e["gid"]:
                continue
            g = Genome.from_json(self.archive.genome_json(e["gid"]))
            f2, rows = self._rung(g, e["desc"], c, 2, self.rung_seeds[2])
            deltas = [r["ablation_delta"] for r in rows if r.get("ablation_delta") is not None and np.isfinite(r.get("ablation_delta", np.nan))]
            self.validated[c] = dict(gid=e["gid"], fitness=f2, ablation_delta=(float(np.mean(deltas)) if deltas else None),
                                     eval_ids=[r["eval_id"] for r in rows], gen=self.gen)
            self.log(f"validated cell {c}: rung-2 fitness {f2}, ablation delta {self.validated[c]['ablation_delta']}")

    def checkpoint(self):
        self.validate_due()
        d = self.archive.checkpoint(self.state())
        self.log(f"checkpoint {d.name}: {self.n_evals} evals, {self.map.filled()} cells, mean elite {self.map.mean_fitness():.3f}")
        return d

    def resume(self):
        st, replayed = self.archive.resume()
        if st is None:
            return False
        assert st["grammar_hash"] == self.G.hash, "the archive was built with another grammar"
        self.gen, self.n_evals, self.pending = st["gen"], st["n_evals"], list(st["pending"])
        self.rng.bit_generator.state = st["rng"]; self.map.load_state(st["elites"])
        self.validated = {int(k): v for k, v in st.get("validated", {}).items()}
        if replayed:
            # rows written after the checkpoint: re-insert their elites so nothing evaluated is lost
            df = self.archive.frame(include_excluded=True)
            tail = df.iloc[-replayed:]
            # rung-1 evaluations were inserted as the mean over their seeds; rebuild the same
            # groups (a pair may straddle the checkpoint, so group over the whole frame)
            r1 = df[(df.rung == 1) & df.valid]
            keys = {(r.gid, int(r.cell), int(r.generation)) for _, r in tail[tail.rung == 1].iterrows()}
            for gid, cell, gen in keys:
                grp = r1[(r1.gid == gid) & (r1.cell == cell) & (r1.generation == gen)]
                if len(grp):
                    r0 = df[(df.gid == gid) & (df.cell == cell) & (df.generation == gen) & (df.rung == 0)]
                    eid = r0.eval_id.iloc[0] if len(r0) else grp.eval_id.iloc[0]
                    self.map.insert(cell, gid, float(grp.fitness.mean()), eid, gen, [grp.iloc[0][f"d_{a}"] for a in AXES])
            n0 = int((tail.rung == 0).sum()) if "rung" in tail else len(tail)     # only rung-0 rows are proposals
            self.n_evals += n0; self.gen += n0
        self.log(f"resumed at {self.n_evals} evals ({replayed} replayed), {self.map.filled()} cells")
        return True

    def run(self, budget, stop_after_flat=500):
        last_improve = self.n_evals
        while self.n_evals < budget:
            _, improved = self.step()
            if improved:
                last_improve = self.n_evals
            if self.n_evals - last_improve >= stop_after_flat and self.map.filled() > 0:
                self.log(f"no cell improved for {stop_after_flat} evaluations; stopping"); break
        self.validate_due(final=True)
        self.checkpoint()
        (self.root / "map.parquet").parent.mkdir(exist_ok=True)
        self.map.to_frame().to_parquet(self.root / "map.parquet", index=False)
