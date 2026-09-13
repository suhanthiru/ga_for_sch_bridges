# Generator gate: findings

Status: template, nothing run yet. Commit: (filled by the report). Pre-registration:
SEARCH_PLAN.md section 0 at the commit that introduced it.

Every table below is written by `python -m sb.cli report gate` from `sb/stats.py`.
Nothing between the table markers is edited by hand.

## 1. Setup

One paragraph pointing at SEARCH_PLAN section 0. No numbers here.

## 2. Dynamic-range check

<!-- tables:dr -->
<!-- /tables:dr -->

## 3. MPC-relabel diagnostic

<!-- tables:diag -->
<!-- /tables:diag -->

## 4. Success by source, layout and disturbance

<!-- tables:cells -->
<!-- /tables:cells -->

## 5. Primary contrast F1: BRIDGE-slip minus PD-noise

<!-- tables:F1 -->
<!-- /tables:F1 -->

## 6. Isolations

### 6.1 Reference covariance (F2, F3)

<!-- tables:F2 -->
<!-- /tables:F2 -->

<!-- tables:F3 -->
<!-- /tables:F3 -->

### 6.2 Action labels (F4)

<!-- tables:F4 -->
<!-- /tables:F4 -->

### 6.3 State coverage

<!-- tables:cov -->
<!-- /tables:cov -->

## 7. Upper bound and the other generators (F5, F6, F7)

<!-- tables:F5 -->
<!-- /tables:F5 -->

<!-- tables:F6 -->
<!-- /tables:F6 -->

<!-- tables:F7 -->
<!-- /tables:F7 -->

## 8. Tail: CVaR_0.1 of progress

<!-- tables:cvar -->
<!-- /tables:cvar -->

## 9. Replication (seeds 5-9)

<!-- tables:rep -->
<!-- /tables:rep -->

## 10. Verdict

<!-- tables:decision -->
<!-- /tables:decision -->

What this means for the `data` slot of the grammar: (filled once the verdict is in;
the three cases are spelled out in SEARCH_PLAN 0.6).

## 11. Minimum-effect statement

<!-- tables:mineffect -->
<!-- /tables:mineffect -->

## 12. Deviations from the plan

Links to PLAN_CHANGES.md entries and ERRORS.md entries raised during this section.
