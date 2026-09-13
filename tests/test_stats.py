import re

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from sb import stats as ST


def test_ci95_matches_scipy():
    x = np.array([0.3, 0.5, 0.4, 0.6, 0.45])
    e = ST.ci95(x)
    lo, hi = sps.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=sps.sem(x))
    assert abs(e.lo - lo) < 1e-12 and abs(e.hi - hi) < 1e-12 and e.n == 5


def test_ci95_small_n_and_nonfinite():
    e = ST.ci95([0.4])
    assert e.n == 1 and e.mean == 0.4 and np.isnan(e.lo)
    e = ST.ci95([0.4, np.nan, 0.6])
    assert e.n == 2 and abs(e.mean - 0.5) < 1e-12


def _series(vals, seeds, name):
    return pd.Series(vals, index=pd.Index(seeds, name="seed"), name=name)


def test_paired_diff_aligns_on_seed_intersection():
    a = _series([0.5, 0.6, 0.7, 0.8], [0, 1, 2, 3], "A")
    b = _series([0.4, 0.4, 0.5, 0.9], [1, 2, 3, 9], "B")
    c = ST.paired_diff(a, b, "F", dict(layout="L1", disturbance="slip"))
    assert c.n_pairs == 3 and c.seeds == (1, 2, 3)
    assert abs(c.delta - np.mean([0.2, 0.3, 0.3])) < 1e-12


def test_paired_diff_bootstrap_is_deterministic_and_contains_delta():
    a = _series([0.5, 0.6, 0.7, 0.8, 0.9], range(5), "A"); b = _series([0.4, 0.4, 0.5, 0.5, 0.6], range(5), "B")
    c1 = ST.paired_diff(a, b, "F", {}); c2 = ST.paired_diff(a, b, "F", {})
    assert c1.boot_lo == c2.boot_lo and c1.boot_hi == c2.boot_hi
    assert c1.boot_lo <= c1.delta <= c1.boot_hi and c1.t_lo <= c1.delta <= c1.t_hi


def test_paired_diff_rejects_small_bootstrap():
    a = _series([0.5, 0.6], [0, 1], "A"); b = _series([0.4, 0.4], [0, 1], "B")
    with pytest.raises(AssertionError):
        ST.paired_diff(a, b, "F", {}, n_boot=100)


def test_paired_t_shift_and_identity():
    a = _series(np.linspace(0.3, 0.7, 6), range(6), "A")
    _, p = ST.paired_t(a + 0.2 + 1e-3 * np.arange(6), a)
    assert p < 1e-6
    _, p = ST.paired_t(a, a.copy())
    assert p == 1.0


def test_bh_canonical_1995_example():
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
         0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    rej = ST.bh_reject(p, 0.05)
    assert rej.sum() == 4 and rej[:4].all()
    q = ST.bh_adjust(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12) and np.all(q >= np.asarray(p) - 1e-12)


def test_bh_controls_fdr_monte_carlo():
    rng = np.random.default_rng(1); m, m1, reps = 50, 10, 300
    fdps = []
    for _ in range(reps):
        z = rng.normal(size=m); z[:m1] += 3.0
        p = 2 * sps.norm.sf(np.abs(z))
        rej = ST.bh_reject(p, 0.05)
        fdps.append(rej[m1:].sum() / max(rej.sum(), 1))
    assert np.mean(fdps) <= 0.07


def test_cvar_known_value():
    x = np.arange(100) / 100
    assert abs(ST.cvar(x, 0.1) - 0.045) < 1e-12
    df = pd.DataFrame(dict(seed=[0] * 100 + [1] * 100, progress=np.r_[x, x[::-1]]))
    s = ST.cvar_by_seed(df)
    assert list(s.index) == [0, 1] and abs(s[0] - 0.045) < 1e-12


def test_demo_efficiency_ratio_two():
    rows = []
    for s in range(3):
        for n in (1, 2, 4, 8, 16):
            rows.append(dict(seed=s, n_demo=n, success=min(1.0, 0.1 * np.log2(n) + 0.3), src="A"))
            rows.append(dict(seed=s, n_demo=n, success=min(1.0, 0.1 * np.log2(2 * n) + 0.3), src="B"))
    df = pd.DataFrame(rows)
    r = ST.demo_efficiency(df[df.src == "A"], df[df.src == "B"])
    assert abs(r["ratio"] - 0.5) < 0.05
    assert ST.demo_efficiency(df[(df.src == "A") & (df.n_demo == 4)], df[df.src == "B"]) is None


def _contrast(delta, lo, hi, q, blo=None, bhi=None, disturbance="slip"):
    return ST.Contrast("F", "A", "B", dict(layout="L1", disturbance=disturbance), delta, lo, hi, 0.01,
                       lo if blo is None else blo, hi if bhi is None else bhi, 5, (0, 1, 2, 3, 4), q=q)


def test_min_effect_three_outcomes():
    assert ST.min_effect(_contrast(0.2, 0.1, 0.3, 0.01)) == "pass"
    assert ST.min_effect(_contrast(0.02, -0.01, 0.05, 0.4)) == "null"
    assert ST.min_effect(_contrast(0.12, 0.02, 0.22, 0.2)) == "inconclusive"


def test_gate_decision_three_verdicts():
    cells = [("L1", d) for d in ST.DISTURBED] + [("L2", d) for d in ST.DISTURBED]

    def fam(delta, lo, hi, q):
        return [ST.Contrast("F", "A", "B", dict(layout=l, disturbance=d), delta, lo, hi, 0.01, lo, hi, 5, (0, 1, 2, 3, 4), q=q)
                for l, d in cells]

    f1 = fam(0.15, 0.05, 0.25, 0.01); f4 = fam(0.0, -0.05, 0.05, 0.5); f3 = fam(0.0, -0.05, 0.05, 0.5)
    assert ST.gate_decision(f1, f4, f3)["verdict"] == "bridge-specific"
    f1 = fam(0.01, -0.03, 0.05, 0.6); f3 = fam(0.12, 0.04, 0.2, 0.01)
    assert ST.gate_decision(f1, f4, f3)["verdict"] == "noise-model-specific"
    f1 = fam(-0.15, -0.25, -0.05, 0.01)
    assert ST.gate_decision(f1, f4, f3)["verdict"] == "neither"
    f1 = fam(0.15, 0.05, 0.25, 0.01); f4 = fam(0.3, 0.2, 0.4, 0.01)
    assert ST.gate_decision(f1, f4, f3)["verdict"] == "neither"


def test_contrast_family_from_long_table():
    rows = []
    for l in ("L1", "L2"):
        for d in ("none", "slip", "rain", "push"):
            for s in range(5):
                rows.append(dict(source="A", layout=l, disturbance=d, seed=s, success=0.5 + 0.02 * s))
                rows.append(dict(source="B", layout=l, disturbance=d, seed=s, success=0.4 + 0.02 * s))
    fam = ST.contrast_family(pd.DataFrame(rows), "A", "B", "F1")
    assert len(fam) == 6 and all(abs(c.delta - 0.1) < 1e-12 for c in fam) and all(c.q is not None for c in fam)


def test_report_table_prints_only_input_numbers():
    c = _contrast(0.123456, 0.0123, 0.2345, 0.0456, 0.02, 0.22)
    c = ST.replace(c, verdict="pass")
    txt = ST.report_table([c], "contrast", digits=3)
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+", txt)]
    allowed = {round(v, 3) for v in (c.delta, c.t_lo, c.t_hi, c.boot_lo, c.boot_hi, c.p_t, c.q)}
    assert all(round(v, 3) in allowed for v in nums) and len(nums) == 7


def test_int64_guard_rank_sum():
    r = ST._i64(sps.rankdata(np.arange(70000)))
    assert int(r.sum()) == 2_450_035_000
    idx = np.random.default_rng(0).integers(0, 5, size=(10, 5), dtype=np.int64)
    assert idx.dtype == np.int64


def test_spearman_by_seed():
    df = pd.DataFrame(dict(seed=[0] * 4 + [1] * 4, x=[1, 2, 3, 4] * 2, y=[1, 2, 3, 4, 4, 3, 2, 1]))
    e, rs = ST.spearman_by_seed(df, "x", "y")
    assert rs == [1.0, -1.0] and abs(e.mean) < 1e-12
