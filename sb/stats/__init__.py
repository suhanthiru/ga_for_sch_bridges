"""Statistics package. The reporting primitives live in sb.stats.core (re-exported here so
`from sb import stats as ST` keeps working); the "why" model's features are in
sb.stats.why_features."""
from sb.stats.core import *  # noqa: F401,F403
from sb.stats.core import (ALPHA, BOOT_SEED, DISTURBED, GATE_EFFECT, MIN_EFFECT, N_BOOT, Q_FDR, Contrast, Estimate,  # noqa: F401
                           bh_adjust, bh_reject, bh_table, cell_table, ci95, contrast_family, cvar, cvar_by_seed,
                           demo_efficiency, gate_decision, min_effect, paired_diff, paired_t, replace, report_table,
                           spearman_by_seed, _i64)
