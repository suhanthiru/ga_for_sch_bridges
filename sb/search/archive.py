"""Append-only archive of every evaluation, with tags instead of deletions, hashed
checkpoints and resume.

    root/archive/part-NNNNNNN.parquet    immutable shards (<= rows_per_shard rows)
    root/tags/part-NNNNNNN.parquet       append-only tag log (eval_id, tag, reason, author, ts)
    root/genomes/<gid>.json              genome bodies
    root/ckpt/ckpt-NNNNNNN/              snapshot: archive.parquet, tags.parquet, state.json, MANIFEST.json (sha256)
    root/LATEST                          name of the newest verified checkpoint (atomic swap)

Rows are never modified. `excluded` is derived at read time from tags starting with
"exclude:". Resume verifies every file in the checkpoint against its manifest, falls
back one checkpoint on a mismatch, and replays every shard newer than the checkpoint.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Archive:
    def __init__(self, root, rows_per_shard=100):
        self.root = Path(root); self.rows_per_shard = rows_per_shard
        for d in ("archive", "tags", "genomes", "ckpt"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self._buf, self._tagbuf = [], []
        self.seq = 1 + max((int(p.stem.split("-")[1]) for p in (self.root / "archive").glob("part-*.parquet")), default=-1)
        self.tagseq = 1 + max((int(p.stem.split("-")[1]) for p in (self.root / "tags").glob("part-*.parquet")), default=-1)

    # -------------------------------------------------------------- writes
    def append(self, row, genome=None):
        if genome is not None:
            p = self.root / "genomes" / f"{genome.gid}.json"
            if not p.exists():
                p.write_text(genome.to_json(), newline="\n")
        self._buf.append(dict(row))
        if len(self._buf) >= self.rows_per_shard:
            self.flush()

    def flush(self):
        out = []
        if self._buf:
            p = self.root / "archive" / f"part-{self.seq:07d}.parquet"
            pd.DataFrame(self._buf).to_parquet(p, index=False); self.seq += 1; self._buf = []; out.append(p)
        if self._tagbuf:
            p = self.root / "tags" / f"part-{self.tagseq:07d}.parquet"
            pd.DataFrame(self._tagbuf).to_parquet(p, index=False); self.tagseq += 1; self._tagbuf = []; out.append(p)
        return out

    def tag(self, eval_id, tag, reason="", author="search"):
        self._tagbuf.append(dict(eval_id=eval_id, tag=tag, reason=reason, author=author, ts=time.time()))

    # --------------------------------------------------------------- reads
    def _read(self, sub):
        fs = sorted((self.root / sub).glob("part-*.parquet"))
        return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

    def frame(self, include_excluded=False):
        df = self._read("archive")
        if self._buf:
            df = pd.concat([df, pd.DataFrame(self._buf)], ignore_index=True)
        if df.empty:
            return df
        df = df.drop_duplicates("eval_id", keep="first")
        tags = self.tags()
        ex = set(tags.loc[tags.tag.str.startswith("exclude:"), "eval_id"]) if not tags.empty else set()
        df["excluded"] = df.eval_id.isin(ex)
        return df if include_excluded else df[~df.excluded].reset_index(drop=True)

    def tags(self):
        t = self._read("tags")
        if self._tagbuf:
            t = pd.concat([t, pd.DataFrame(self._tagbuf)], ignore_index=True)
        return t

    def genome_json(self, gid):
        return (self.root / "genomes" / f"{gid}.json").read_text()

    # ---------------------------------------------------------- checkpoint
    def checkpoint(self, state):
        """Flush, snapshot the compacted archive and tags with the search state, write a
        sha256 manifest, and point LATEST at it atomically."""
        self.flush()
        n = len(self._read("archive"))
        d = self.root / "ckpt" / f"ckpt-{n:07d}"; d.mkdir(parents=True, exist_ok=True)
        self._read("archive").to_parquet(d / "archive.parquet", index=False)
        self._read("tags").to_parquet(d / "tags.parquet", index=False)
        (d / "state.json").write_text(json.dumps(dict(state, n_rows=n, shard_seq=self.seq, tag_seq=self.tagseq, ts=time.time()), sort_keys=True), newline="\n")
        man = {f.name: sha256(f) for f in sorted(d.iterdir()) if f.name != "MANIFEST.json"}
        (d / "MANIFEST.json").write_text(json.dumps(man, sort_keys=True, indent=1), newline="\n")
        tmp = self.root / "LATEST.tmp"; tmp.write_text(d.name); os.replace(tmp, self.root / "LATEST")
        return d

    def verify(self, d):
        d = Path(d)
        if not (d / "MANIFEST.json").exists():
            return False
        man = json.loads((d / "MANIFEST.json").read_text())
        return all((d / k).exists() and sha256(d / k) == v for k, v in man.items())

    def resume(self):
        """Returns (state, replayed_rows): the newest checkpoint whose manifest verifies,
        plus the number of rows in shards written after it (they are replayed by reading:
        the archive is the shards, the checkpoint is only a verified state)."""
        cks = sorted((self.root / "ckpt").glob("ckpt-*"), reverse=True)
        latest = (self.root / "LATEST").read_text().strip() if (self.root / "LATEST").exists() else None
        if latest and (self.root / "ckpt" / latest) in cks:
            cks.remove(self.root / "ckpt" / latest); cks.insert(0, self.root / "ckpt" / latest)
        for d in cks:
            if self.verify(d):
                state = json.loads((d / "state.json").read_text())
                replayed = max(0, len(self._read("archive")) - state["n_rows"])
                return state, replayed
        return None, len(self._read("archive"))
