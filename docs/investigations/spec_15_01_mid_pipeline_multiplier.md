# Spec 15 Item 1 — Mid-Pipeline Risk Multiplier Investigation

**Date:** 2026-04-20
**Status:** RESOLVED — code is correct, docs updated to match.

---

## Problem Statement

`protection_engine.py` uses `multiplier = 1.10` for structural methods in the mid-pipeline risk check. Earlier documentation referenced `1.5`. At a 5% target:
- 1.10 → early-exit when ReID ≤ 5.5%
- 1.50 → early-exit when ReID ≤ 7.5%

Hypothesis: at 1.10, the check almost never fires, making it dead code.

## Investigation

### Code Location

`sdc_engine/sdc/protection_engine.py`, lines 1142–1163 in `run_pipeline()`:

```python
if method in _STRUCTURAL:
    multiplier = 1.10
else:
    multiplier = 1.20  # GENERALIZE or unknown
if mid_reid <= (risk_target_raw * multiplier):
    # skip remaining pipeline steps
    break
```

### Git History

The `1.10` value was introduced in the initial SDC Engine commit (`462bc16`, 2026-04-19). No commit message or comment explains the choice. The value has never been changed.

### Empirical Data

Ran `run_pipeline()` instrumented on all 8 harness datasets with a forced `[kANON, LOCSUPR]` pipeline and `risk_target=0.05`:

| Dataset | Mid-ReID after kANON | Target | 1.10 fires? | 1.15+ fires? |
|---------|---------------------|--------|-------------|--------------|
| testdata | 0.1429 | 0.05 | no | no |
| CASCrefmicrodata | 1.0000 | 0.05 | no | no |
| free1 | 1.0000 | 0.05 | no | no |
| adult_low_reid | 0.0556 | 0.05 | no | YES (0.0575) |
| adult_mid_reid | 0.0345 | 0.05 | YES | YES |
| adult_high_reid | 0.0833 | 0.05 | no | no |
| greek_low_reid | 0.0333 | 0.05 | YES | YES |
| greek_mid_reid | 0.0909 | 0.05 | no | no |

### Multiplier Fire Rates

| Multiplier | Fires | Rate |
|-----------|-------|------|
| 1.10 | 2/8 | 25.0% |
| 1.15 | 3/8 | 37.5% |
| 1.20 | 3/8 | 37.5% |
| 1.25 | 3/8 | 37.5% |
| 1.30 | 3/8 | 37.5% |
| 1.50 | 3/8 | 37.5% |

The distribution is binary — kANON either fully solves the problem (reid < target, the 2 cases) or leaves reid >> target (the 5 cases that never fire regardless of multiplier). The one edge case (adult_low_reid: mid_reid=0.0556) sits exactly between 1.10× (0.055) and 1.15× (0.0575) thresholds.

### Key Observation

In production, none of the standard harness datasets actually trigger a pipeline rule. The DYN pipeline requires reid_95 > 0.20 + mixed types + outliers + high_risk_rate conditions that combine to be rare. The mid-check is a low-traffic code path regardless of multiplier value.

## Decision

Per Spec 15 decision rule: **"If 1.10 fires in 5–25% of pipeline runs: current behaviour is defensible. Keep the code, update docs from 1.5 → 1.10."**

- 1.10 fires at 25.0% — right at the boundary, defensible.
- Raising to 1.15 would gain one marginal case but no meaningful behavioural improvement.
- The user_guide already documents `1.1×` correctly (line 2411).
- The empirical_validation_checklist previously flagged this as drift — now resolved.

## Actions Taken

1. Updated `docs/empirical_validation_checklist.md`:
   - Line 580: confidence raised from "Medium — tighter than documented 1.5" to "High — fires ~25% of pipeline mid-checks (Spec 15 validated)"
   - Line 679: status changed from "🟡 Partial — code-vs-spec drift" to "✅ Validated"
   - Line 682: conclusion updated to note resolution
2. No code changes — multiplier remains at 1.10.
3. Investigation script: `tests/empirical/investigate_mid_pipeline.py` (not committed to main).
