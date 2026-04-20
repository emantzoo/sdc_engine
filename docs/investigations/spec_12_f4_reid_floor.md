# Spec 12 Finding 4 — Mathematical Floor on ReID Reduction

**Date:** 2026-04-20
**Status:** Documented (architectural finding, no code fix)
**Source:** Family 1 sensitivity analysis (Spec 12), empirical harness observations

---

## What the Floor Is

Every dataset has a minimum achievable reid_95 determined by its structural properties. Below this floor, no amount of kANON escalation, LOCSUPR suppression, or retry-loop iterations can reduce reid_95 further without destroying the data (suppression > utility floor).

The floor is **not a bug** — it's a combinatorial property of the dataset.

---

## The Mechanism

### Core Formula

```
expected_eq_size = n_rows / combination_space
```

Where `combination_space` = product of unique values across all QI columns.

For reid_95 ≤ T (target):
- Need 95th-percentile equivalence class size ≥ `1/T`
- At target T=0.05: need eq classes of ≥ 20 records at the 95th percentile

If `expected_eq_size` is much smaller than `1/T`, generalization-only methods (kANON) cannot create enough large equivalence classes without suppressing a large fraction of the data.

### Why Generalization Alone Fails

kANON merges QI values into coarser groups to create larger equivalence classes. But:

1. Each generalization step reduces combination_space by a factor (e.g., collapsing 50 values to 10 categories = 5× reduction per QI)
2. The reduction compounds multiplicatively across QIs
3. But excessive generalization destroys utility (information loss), hitting the utility floor

The tension: deep reid reduction requires **aggressive** generalization, but the utility floor prevents it.

### Why the Retry Engine Hits a "Plateau"

The retry engine escalates: k=3 → 5 → 7 ��� 10 → 15 → 20. Each step forces larger minimum equivalence classes, which requires more generalization/suppression. At some k, either:
- The suppression rate exceeds the acceptable limit (utility floor rejects)
- Or the generalization hierarchies are exhausted (all QIs at max level)

The engine reports "target not met" — but this isn't a tuning failure, it's hitting the structural floor.

---

## Example Datasets Where the Floor Was Hit

### Family 1 Synthetic Data (Spec 12)

| Property | Value |
|----------|-------|
| Rows | 5,000 |
| QI combination space | 1,250 |
| Expected eq size | 4.0 |
| Required for T=0.05 | ≥20 at 95th percentile |
| Best reid_after achieved | 0.083 (83% reduction) |
| Target | 0.05 (requires 90% reduction) |
| Result | 0/135 runs met target |

The math: even with perfect k-anonymity at k=12 (which would require collapsing from 1250 to ~417 equivalence classes), the 95th-percentile class would still be too small for reid_95 ≤ 0.05. The engine achieves k=5 effectively (reid=0.083 ≈ 1/12) but cannot go further.

### CASCrefmicrodata (Real Data)

| Property | Value |
|----------|-------|
| Rows | 1,080 |
| QIs | 5 continuous financial variables |
| Combination space | ~1,080 (essentially all unique) |
| Expected eq size | 1.0 |
| reid_before | 1.0 (100% unique) |
| reid_after | 1.0 (unchanged) |
| Floor | Cannot reduce — every record is structurally unique |

With 5 continuous QIs and 1080 records, the QI space is so large that even aggressive binning can't create meaningful equivalence classes. The QR0 rule correctly routes this to `GENERALIZE_FIRST`, but the floor persists. (Note: QR0 was silently config-blocked before Fix 0 (2026-04-20) because GENERALIZE_FIRST was missing from `METRIC_ALLOWED_METHODS`. This claim is valid post-Fix 0.)

### free1 (Real Data)

| Property | Value |
|----------|-------|
| Rows | 4,000 |
| QIs | 7 (5 categorical + 2 continuous) |
| Combination space | Very large (7 QIs) |
| reid_before | 1.0 |
| reid_after | 0.50 (50% floor) |
| Suppression rate | 60% |

The engine achieves 50% reduction but at 60% suppression — already at the utility floor. Further reduction would require even more suppression.

### Contrast: Datasets That DO Meet Target

| Dataset | QIs | Rows | Combo space | Expected eq | reid_after | Target met? |
|---------|-----|------|-------------|-------------|------------|-------------|
| adult_mid_reid | 4 | 30K | ~3,000 | 10.0 | 0.038 | Yes |
| greek_mid_reid | 4 | 35K | ~2,000 | 17.5 | 0.019 | Yes |
| testdata | 7 | 4,580 | ~4,000 | 1.1 | 0.043 | Yes (barely) |

The difference: datasets with `expected_eq_size ≥ 5` tend to meet T=0.05; those with `expected_eq_size ≤ 4` tend to hit the floor.

---

## Implication for Users

When a user sees "target not met" after the retry engine exhausts all escalation steps, the cause is usually one of:

1. **Too many QIs** — each additional QI multiplies combination space
2. **High-cardinality QIs** — continuous or many-valued categoricals
3. **Too few rows** — not enough records to form large equivalence classes
4. **Target too ambitious** — T=0.05 is aggressive for datasets with expected_eq < 5

The retry engine's "plateau" (reid improvement < 0.001 across escalation steps) is the symptom. The floor is the cause.

### User-Facing Guidance

- If `diagnose_qis()` reports `status=PARTIAL` or `INFEASIBLE`, the dataset may be at or near the floor
- The `max_achievable_k` from feasibility diagnosis is the best proxy for the floor
- Reducing QIs (dropping the least useful one) or accepting a higher target are the two practical remedies
- The engine already reports this via the QR0 rule ("k-anonymity infeasible") and the GENERALIZE_FIRST pathway

---

## Implication for the Engine

### Existing Mitigations

1. **`diagnose_qis()`** — computes `combination_space`, `expected_eq_size`, `max_achievable_k` upfront
2. **QR0 rule** — detects infeasible cases (combo space > rows) and routes to GENERALIZE_FIRST
3. **k-pruning** — `_prune_schedule_by_max_k()` removes impossible escalation steps
4. **Plateau detection** — stops escalation if reid improvement < 0.001

### Potential Future Enhancement (Not Scoped)

A cheap upfront estimator could warn users before they run 10+ iterations into a wall:

```
if expected_eq_size < (1 / reid_target) * 4:
    warn("Dataset structure limits maximum reid reduction. "
         "Consider reducing QIs or relaxing target.")
```

This is ~5 lines in `run_rules_engine_protection()` after `diagnose_qis()`. Not implementing now — flagged for future consideration if user complaints arise.

---

## Cross-References

- Family 1 report: `tests/empirical/sensitivity/reports/family_1_concentration.md` §F4
- Analysis notes: `tests/empirical/sensitivity/reports/family_1_analysis_notes.md` §5
- Feasibility diagnosis: `sdc_engine/sdc/preprocessing/diagnose.py`
- k-pruning: `sdc_engine/sdc/protection_engine.py:746` (`_prune_schedule_by_max_k`)
- QR0 rule: `sdc_engine/sdc/selection/rules.py` (`k_anonymity_feasibility_rules`)
