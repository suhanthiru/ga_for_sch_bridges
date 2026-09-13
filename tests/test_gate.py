"""Schema round-trip, cell bookkeeping, the report filler, and the rule that every number
in FINDINGS_generator.md comes from the generated tables."""
import re

import numpy as np
import pandas as pd

from sb import settings
from sb.gate import cells as CL
from sb.gate.report import fill_findings


def test_schema_round_trips_through_parquet(tmp_path):
    df = CL.frame([CL.make_row(source="DEMO", layout="L1", disturbance="none", seed=0, success=0.5),
                   CL.make_row(source="PD-iso", layout="L2", disturbance="push", seed=1, iso_var=0.0086, success=0.1)])
    p = tmp_path / "x.parquet"; df.to_parquet(p, index=False)
    back = pd.read_parquet(p)
    assert list(back.columns) == list(CL.SCHEMA)
    assert back.heading_std.dtype == np.float64 and back.heading_std.isna().all()
    assert back.seed.dtype == np.int64 and float(back.iso_var[1]) == 0.0086


def test_make_row_rejects_unknown_columns():
    try:
        CL.make_row(nope=1)
    except KeyError:
        return
    raise AssertionError


def test_cells_for_seed_skip_the_prior_run():
    main = CL.cells_for(0); rep = CL.cells_for(5)
    assert ("DEMO", "L1") not in main and ("PD-iso", "L1") not in main
    assert ("PD-iso", "L2") in main and ("GC-diff-rollouts", "L1") in main and ("MPC-oracle", "L2") in main
    assert len(rep) == len(CL.SOURCES) * len(CL.LAYOUTS) and ("PD-relabel", "L1") in main and ("BRIDGE-on-PD", "L2") in rep


def test_fill_findings_replaces_only_marked_blocks(tmp_path):
    p = tmp_path / "F.md"
    p.write_text("Commit: none.\n\n<!-- tables:F1 -->\nold\n<!-- /tables:F1 -->\nkeep\n<!-- tables:F2 -->\n<!-- /tables:F2 -->\n", encoding="utf-8")
    fill_findings({"F1": "| a |\n", "F2": "| b |\n"}, p, commit="abc1234")
    t = p.read_text(encoding="utf-8")
    assert "old" not in t and "| a |" in t and "| b |" in t and "keep" in t and "Commit: abc1234." in t


def test_findings_numbers_come_from_tables():
    f = settings.ROOT / "FINDINGS_generator.md"; t = settings.GATE / "tables.md"
    txt = f.read_text(encoding="utf-8")
    blocks = re.findall(r"<!-- tables:\w+ -->\n(.*?)<!-- /tables:\w+ -->", txt, flags=re.S)
    nums = set()
    for b in blocks:
        nums |= set(re.findall(r"[-+]?\d*\.\d+", b))
    if not nums:
        return
    assert t.exists(), "FINDINGS has numbers but results/gate/tables.md does not exist"
    allowed = set(re.findall(r"[-+]?\d*\.\d+", t.read_text(encoding="utf-8")))
    assert nums <= allowed, sorted(nums - allowed)[:10]
