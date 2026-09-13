"""Every reported number comes from here.

Seed is the unit of analysis. Cell estimates are t-intervals over seeds; contrasts are
paired over the seed index with a t-interval, a percentile bootstrap over seeds and a
paired t-test; families of cells are Benjamini-Hochberg corrected; the pre-registered
minimum-effect check and the gate's decision rule are functions of those objects and
nothing else. report_table is the only formatter.

All index and rank arithmetic goes through int64 (numpy's default int is 32-bit on
Windows and rank sums overflow it).
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

N_BOOT = 10_000
ALPHA = 0.05
Q_FDR = 0.05
MIN_EFFECT = dict(success=0.10, efficiency=2.0, cvar=0.15)
GATE_EFFECT = 0.05
BOOT_SEED = 0
DISTURBED = ("slip", "rain", "push")


def _i64(a):
    return np.asarray(a, dtype=np.int64)


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int
    method: str


@dataclass(frozen=True)
class Contrast:
    name: str
    a: str
    b: str
    cell: dict
    delta: float
    t_lo: float
    t_hi: float
    p_t: float
    boot_lo: float
    boot_hi: float
    n_pairs: int
    seeds: tuple
    q: Optional[float] = None
    verdict: Optional[str] = None

    def clear_positive(self):
        return self.t_lo > 0 and self.boot_lo > 0

    def clear_negative(self):
        return self.t_hi < 0 and self.boot_hi < 0

    def clear_of_zero(self):
        return self.clear_positive() or self.clear_negative()


# ------------------------------------------------------------------ estimates
def ci95(x):
    """Mean and t-interval over seeds (ddof 1); non-finite values dropped; n < 2 gives nan bounds."""
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), 0, "t")
    m = float(x.mean())
    if n < 2:
        return Estimate(m, float("nan"), float("nan"), n, "t")
    h = float(sps.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return Estimate(m, m - h, m + h, n, "t")


def cvar(x, alpha=0.1):
    """Mean of the lowest ceil(alpha n) values."""
    x = np.sort(np.asarray(x, dtype=float)); k = int(np.ceil(alpha * x.size))
    return float(x[:max(k, 1)].mean())


def cvar_by_seed(episodes, value="progress", alpha=0.1):
    """episodes: DataFrame with columns seed and `value`, one row per episode -> Series by seed."""
    return episodes.groupby("seed")[value].apply(lambda v: cvar(v.values, alpha))


# ------------------------------------------------------------------ contrasts
def paired_t(a, b):
    j = a.index.intersection(b.index)
    d = (a[j] - b[j]).values.astype(float)
    if len(j) < 2:
        return float(d.mean()) if len(j) else float("nan"), float("nan")
    if np.allclose(d, d[0]):
        return float(d.mean()), 1.0 if abs(d[0]) < 1e-12 else 0.0
    return float(d.mean()), float(sps.ttest_rel(a[j], b[j]).pvalue)


def paired_diff(a, b, name, cell, n_boot=N_BOOT, seed=BOOT_SEED):
    """a, b: Series indexed by seed. Paired over the intersection of seeds."""
    assert n_boot >= 10_000, "the plan fixes the bootstrap at 10000 resamples or more"
    j = a.index.intersection(b.index)
    d = (a[j] - b[j]).values.astype(float)
    n = int(d.size)
    est = ci95(d)
    delta, p = paired_t(a, b)
    if n >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
        means = d[idx].mean(1)
        blo, bhi = (float(v) for v in np.percentile(means, [2.5, 97.5]))
    else:
        blo = bhi = float("nan")
    return Contrast(name=name, a=str(a.name) if a.name is not None else "a", b=str(b.name) if b.name is not None else "b",
                    cell=dict(cell), delta=delta, t_lo=est.lo, t_hi=est.hi, p_t=p, boot_lo=blo, boot_hi=bhi,
                    n_pairs=n, seeds=tuple(int(s) for s in j))


def bh_adjust(p):
    """Benjamini-Hochberg q-values (monotone step-up). nan p-values stay nan."""
    p = np.asarray(p, dtype=float); q = np.full_like(p, np.nan)
    ok = np.isfinite(p); m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok]); ps = p[ok][order]
    ranks = _i64(np.arange(1, m + 1))
    adj = ps * m / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(adj, 0, 1)
    q[ok] = out
    return q


def bh_reject(p, q=Q_FDR):
    return np.nan_to_num(bh_adjust(p), nan=1.0) < q


def bh_table(contrasts, q=Q_FDR):
    """One family per call: fills each Contrast's q-value."""
    qs = bh_adjust([c.p_t for c in contrasts])
    return [replace(c, q=float(qv)) for c, qv in zip(contrasts, qs)]


def contrast_family(df, a, b, name, value="success", cells=None, key=("layout", "disturbance")):
    """Build and BH-correct one family of paired contrasts a - b from a long table with
    columns source, seed, `value` and the cell keys."""
    key = list(key)
    if cells is None:
        cells = sorted({tuple(r) for r in df.loc[df.disturbance.isin(DISTURBED), key].drop_duplicates().itertuples(index=False)})
    out = []
    for cell in cells:
        cd = dict(zip(key, cell))
        sub = df
        for k, v in cd.items():
            sub = sub[sub[k] == v]
        sa = sub[sub.source == a].set_index("seed")[value].rename(a)
        sb = sub[sub.source == b].set_index("seed")[value].rename(b)
        out.append(paired_diff(sa, sb, name, cd))
    return bh_table(out)


# ------------------------------------------------------------ effect checks
def min_effect(c, kind="success", q=Q_FDR):
    """pass: the pre-registered minimum effect is met with q < Q and both CIs clear of zero;
    null: the minimum effect is excluded by the t-interval; else inconclusive."""
    thr = MIN_EFFECT[kind]
    if c.delta >= thr and c.q is not None and c.q < q and c.clear_positive():
        return "pass"
    if np.isfinite(c.t_hi) and c.t_hi < thr:
        return "null"
    return "inconclusive"


def demo_efficiency(curve_a, curve_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """curve_*: DataFrame with columns n_demo, seed, success. Demo-efficiency ratio of a over
    b: the demos b needs to reach a's success at a's largest n, divided by a's n there,
    interpolated monotonically on log n per seed; ratio bootstrapped over seeds. None if
    either curve has fewer than two demo counts."""
    if curve_a.n_demo.nunique() < 2 or curve_b.n_demo.nunique() < 2:
        return None
    seeds = sorted(set(curve_a.seed) & set(curve_b.seed))
    ratios = []
    for s in seeds:
        ca = curve_a[curve_a.seed == s].sort_values("n_demo"); cb = curve_b[curve_b.seed == s].sort_values("n_demo")
        na = float(ca.n_demo.max()); target = float(ca[ca.n_demo == ca.n_demo.max()].success.mean())
        xb = np.log(cb.n_demo.values.astype(float)); yb = np.maximum.accumulate(cb.success.values.astype(float))
        if target > yb.max():
            nb = float(np.exp(xb.max())) * (1 + (target - yb.max()) / max(yb.max(), 1e-9))
        else:
            nb = float(np.exp(np.interp(target, yb, xb)))
        ratios.append(nb / na)
    r = np.asarray(ratios, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(n_boot, r.size), dtype=np.int64)
    bs = r[idx].mean(1)
    return dict(ratio=float(r.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                target=MIN_EFFECT["efficiency"], n=int(r.size))


def _disturbed(cs):
    return [c for c in cs if c.cell.get("disturbance") in DISTURBED]


def gate_decision(primary, relabel, iso_pd, n_cells=6, q=Q_FDR):
    """SEARCH_PLAN 0.6, literally, from BH-corrected contrast lists:
    primary = F1 BRIDGE-slip - PD-noise, relabel = F4 MPC-relabel - BRIDGE-slip,
    iso_pd = F3 PD-noise - PD-iso."""
    f1, f4, f3 = _disturbed(primary), _disturbed(relabel), _disturbed(iso_pd)
    f1_pos = [c.delta >= GATE_EFFECT and c.clear_positive() and c.q is not None and c.q < q for c in f1]
    f4_big = [c.delta >= MIN_EFFECT["success"] and c.clear_positive() for c in f4]
    f1_null = [abs(c.delta) < GATE_EFFECT and np.isfinite(c.t_hi) and c.t_hi < MIN_EFFECT["success"] for c in f1]
    f3_pos = [c.delta >= GATE_EFFECT and c.clear_positive() for c in f3]
    counts = dict(f1_positive=int(sum(f1_pos)), f4_large=int(sum(f4_big)), f1_null=int(sum(f1_null)),
                  f3_positive=int(sum(f3_pos)), n_f1=len(f1), n_f4=len(f4), n_f3=len(f3), n_cells=n_cells)
    if counts["f1_positive"] >= 3 and counts["f4_large"] < 3:
        verdict = "bridge-specific"
    elif counts["f1_null"] >= 4 and counts["f3_positive"] >= 3:
        verdict = "noise-model-specific"
    else:
        verdict = "neither"
    evidence = dict(f1=[(c.cell, bool(x), bool(y)) for c, x, y in zip(f1, f1_pos, f1_null)],
                    f4=[(c.cell, bool(x)) for c, x in zip(f4, f4_big)],
                    f3=[(c.cell, bool(x)) for c, x in zip(f3, f3_pos)])
    return dict(verdict=verdict, counts=counts, evidence=evidence)


def spearman_by_seed(df, x, y, group="seed"):
    """Spearman rank correlation of x and y within each group, then the t-interval over groups."""
    rs = []
    for _, g in df.groupby(group):
        if len(g) < 3:
            continue
        rx = _i64(sps.rankdata(g[x].values)); ry = _i64(sps.rankdata(g[y].values))
        rs.append(float(np.corrcoef(rx, ry)[0, 1]))
    return ci95(rs), rs


# ------------------------------------------------------------------ tables
def cell_table(df, value="success", by=("source", "layout", "disturbance")):
    rows = []
    for keys, g in df.groupby(list(by), sort=False):
        e = ci95(g[value].values)
        rows.append(dict(zip(by, keys), n=e.n, mean=e.mean, lo=e.lo, hi=e.hi))
    return pd.DataFrame(rows)


def _f(v, digits):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:+.{digits}f}" if isinstance(v, float) and v < 0 else f"{v:.{digits}f}"


def report_table(rows, kind="contrast", digits=3):
    """The single markdown formatter. kind: contrast (list[Contrast]) | cell (DataFrame from
    cell_table) | decision (dict from gate_decision) | kv (dict of scalars)."""
    if kind == "contrast":
        head = "| contrast | layout | disturbance | n | delta | t 95% CI | boot 95% CI | p | q | verdict |\n|---|---|---|---|---|---|---|---|---|---|\n"
        body = ""
        for c in rows:
            body += (f"| {c.name} ({c.a} - {c.b}) | {c.cell.get('layout', '-')} | {c.cell.get('disturbance', '-')} | {c.n_pairs} "
                     f"| {_f(c.delta, digits)} | [{_f(c.t_lo, digits)}, {_f(c.t_hi, digits)}] | [{_f(c.boot_lo, digits)}, {_f(c.boot_hi, digits)}] "
                     f"| {_f(c.p_t, digits)} | {_f(c.q, digits)} | {c.verdict or '-'} |\n")
        return head + body
    if kind == "cell":
        cols = [c for c in rows.columns if c not in ("n", "mean", "lo", "hi")]
        head = "| " + " | ".join(cols) + " | n | mean | 95% CI |\n|" + "---|" * (len(cols) + 3) + "\n"
        body = ""
        for _, r in rows.iterrows():
            body += "| " + " | ".join(str(r[c]) for c in cols) + f" | {int(r.n)} | {_f(float(r['mean']), digits)} | [{_f(float(r.lo), digits)}, {_f(float(r.hi), digits)}] |\n"
        return head + body
    if kind == "decision":
        out = f"**Verdict: {rows['verdict']}**\n\n| count | value |\n|---|---|\n"
        for k, v in rows["counts"].items():
            out += f"| {k} | {v} |\n"
        return out
    if kind == "kv":
        out = "| quantity | value |\n|---|---|\n"
        for k, v in rows.items():
            out += f"| {k} | {_f(v, digits) if isinstance(v, (int, float, np.floating, np.integer)) else v} |\n"
        return out
    raise ValueError(kind)
