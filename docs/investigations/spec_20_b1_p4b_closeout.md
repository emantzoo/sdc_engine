# Spec 20 B1 — P4b Restoration Check: Closeout

**Verdict:** Confirm deletion. P4b is not restored.

## Investigation

P4b (`P4b_Skewed_Sensitive_Targeted`) was deleted in commit `45f2cdf` (Spec 19 Phase 2).

**P4b requirements:**
1. `skewed_count >= 2` (skewed continuous QIs)
2. `has_sensitive_attributes == True`
3. `n_qis >= 2`
4. `sensitive_column_diversity <= 10`

**G4 fixture** (`fixture_g4_p4.csv`): 10K rows, QIs: income/wealth/sex, sensitive: disease.
- `skewed_columns = ['income', 'wealth']` (gate 1 met)
- `has_sensitive_attributes = False` (gate 2 NOT met — hardcoded in build_data_features)
- `sensitive_column_diversity = 5` when injected (gate 4 would be met)

## Current engine behavior on G4

Without injection: `RC1_Risk_Dominated → LOCSUPR` (reid_95=0.33, backward elimination data available).

With sensitive injection: Still `RC1_Risk_Dominated → LOCSUPR`. RC1 has higher priority than pipeline rules in the factory chain.

## Why P4b is not restorable

1. **Unwired:** `has_sensitive_attributes` is hardcoded `False` in `build_data_features()` (line 281). Until this is wired, P4b cannot fire in production.
2. **Preempted:** Even when wired, RC1 fires first on any data with backward-elimination results and reid_95 > 15%. G4's reid_95=33% guarantees RC1 preemption.
3. **No utility comparison possible:** Since P4b can't fire through the chain (even with injection), there's no production path to compare against.

## Conclusion

P4b's deletion was correct. Restoring it would produce dead code (preempted by RC1). If sensitive-column-targeted PRAM is needed in the future, it should be implemented as a new rule at the appropriate priority level with `has_sensitive_attributes` wired in `build_data_features`.
