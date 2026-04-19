# Empirical Validation Summary

**Date:** 2026-04-20
**Mode:** Python-only fallback (deterministic, no R/rpy2 variance)
**Risk target:** 0.05

## Datasets

| Dataset | Rows | QIs | cat_ratio | reid_95 | Source | Relevant thresholds |
|---------|------|-----|-----------|---------|--------|-------------------|
| testdata | 4580 | 7 (all categorical) | 1.00 | 0.50 | sdcMicro | T1, T2, T4 |
| CASCrefmicrodata | 1080 | 5 (all continuous) | 0.00 | 1.00 | sdcMicro | T1, T3 |
| free1 | 4000 | 7 (5 cat + 2 cont) | 0.71 | 1.00 | sdcMicro | T1, T2, T3, T4 |
| adult_low_reid | 30162 | 3 (age/sex/race) | 0.67 | 0.067 | UCI Adult | T4 |
| adult_mid_reid | 30162 | 4 (+marital_status) | 0.75 | 0.25 | UCI Adult | T2, T3 |
| adult_high_reid | 30162 | 4 (+education) | 0.75 | 0.50 | UCI Adult | T3 |
| greek_low_reid | 41742 | 3 (all categorical) | 1.00 | 0.125 | Greek RE | T4 |
| greek_mid_reid | 35154 | 4 (all categorical) | 1.00 | 0.33 | Greek RE | T2, T3 |

## Thresholds tested

| ID | Name | Current | Range tested |
|----|------|---------|-------------|
| T1 | RC1 dominated cutoff | 0.40 | 0.30 - 0.50 |
| T2 | CAT1 categorical ratio | 0.70 | 0.60 - 0.80 |
| T3 | QR2 suppression gate | 0.25 | 0.15 - 0.35 |
| T4 | LOW1 reid_95 gate | 0.10 | 0.05 - 0.15 |

## Key Finding

**Rule-level thresholds affect routing but not outcomes under the reid95 metric.** The
engine's metric-method compatibility filter correctly converts perturbative selections
(PRAM, NOISE) to kANON when reid95 is the active risk metric, because perturbation
cannot reduce structural re-identification risk. Under other metrics (k-anonymity,
l-diversity, uniqueness), the same thresholds would produce different method outputs.
Validation under reid95 therefore cannot distinguish threshold effects from the metric
filter -- future validation rounds should test under each risk metric independently.

## Run 3 results: 80 runs, 0 errors, 3 crossovers

### Crossovers detected

| Threshold | Dataset | Crossover at | Current | Shift | Low rule | High rule | Initial method change |
|-----------|---------|-------------|---------|-------|----------|-----------|----------------------|
| T4 | adult_low_reid | 0.075 | 0.10 | -2.5 pp | LOW3_Mixed (kANON) | LOW1_Categorical (PRAM) | kANON -> PRAM |
| T4 | greek_low_reid | 0.125 | 0.10 | +2.5 pp | LOW3_Mixed (kANON) | LOW1_Categorical (PRAM) | kANON -> PRAM |
| T2 | adult_mid_reid | 0.80 | 0.70 | +10 pp | CAT1_Categorical_Dominant (PRAM) | CAT2_Mixed (NOISE) | PRAM -> NOISE |

### Outcome quality across crossovers

**T4 on adult_low_reid** (reid_95 = 0.067, crossover at threshold = 0.075):

| Threshold value | Rule fired | Initial method | Final method | reid_after | utility | target_met |
|----------------|------------|---------------|--------------|-----------|---------|-----------|
| 0.050 | LOW3_Mixed | kANON | kANON | 0.010 | 0.9993 | Yes |
| 0.075 | LOW1_Categorical | PRAM | kANON | 0.050 | 1.0000 | Yes |
| 0.100 | LOW1_Categorical | PRAM | kANON | 0.050 | 1.0000 | Yes |
| 0.125 | LOW1_Categorical | PRAM | kANON | 0.050 | 1.0000 | Yes |
| 0.150 | LOW1_Categorical | PRAM | kANON | 0.050 | 1.0000 | Yes |

Interpretation: The crossover is real -- LOW1 fires when the gate opens past reid_95=0.067.
LOW1 selects PRAM, but the override rule converts it to kANON. Both sides meet the risk
target. The reid_after difference (0.01 vs 0.05) reflects different kANON parameterizations
(k=5 hybrid vs k=3), not the rule's method choice.

**T4 on greek_low_reid** (reid_95 = 0.125, crossover at threshold = 0.125):

| Threshold value | Rule fired | Initial method | Final method | reid_after | utility | target_met |
|----------------|------------|---------------|--------------|-----------|---------|-----------|
| 0.050 | LOW3_Mixed | kANON | kANON | 0.038 | 0.9793 | Yes |
| 0.075 | LOW3_Mixed | kANON | kANON | 0.038 | 0.9793 | Yes |
| 0.100 | LOW3_Mixed | kANON | kANON | 0.038 | 0.9793 | Yes |
| 0.125 | LOW1_Categorical | PRAM | kANON | 0.034 | 0.9788 | Yes |
| 0.150 | LOW1_Categorical | PRAM | kANON | 0.034 | 0.9788 | Yes |

Interpretation: Crossover at the exact reid_95 boundary (0.125). Outcomes are nearly
identical on both sides -- the override rule masks the method difference. The rule
routing changes but the final outcome does not.

**T2 on adult_mid_reid** (cat_ratio = 0.75, crossover at threshold = 0.80):

| Threshold value | Rule fired | Initial method | Final method | reid_after | utility | target_met |
|----------------|------------|---------------|--------------|-----------|---------|-----------|
| 0.60 | CAT1_Categorical_Dominant | PRAM | kANON | 0.0455 | 0.9995 | Yes |
| 0.65 | CAT1_Categorical_Dominant | PRAM | kANON | 0.0455 | 0.9995 | Yes |
| 0.70 | CAT1_Categorical_Dominant | PRAM | kANON | 0.0455 | 0.9995 | Yes |
| 0.75 | CAT1_Categorical_Dominant | PRAM | kANON | 0.0455 | 0.9995 | Yes |
| 0.80 | CAT2_Mixed_Categorical_Majority | NOISE | kANON | 0.0455 | 0.9995 | Yes |

Interpretation: CAT1 fires when cat_ratio (0.75) >= threshold. At threshold=0.80,
cat_ratio < threshold so CAT2 fires instead. Both select different initial methods
(PRAM vs NOISE) but the override rule converts both to kANON, producing identical
outcomes.

## Analysis

### What the crossovers tell us

1. **Rules are reachable.** T2 and T4 patchers correctly vary which rule fires.
   The rule engine's routing logic is functional -- thresholds control rule selection
   as designed.

2. **The override rule masks method diversity.** All three crossovers show a pattern:
   the initial method changes (kANON->PRAM, PRAM->NOISE) but the final output is
   always kANON. The `_override_to_kanon_if_reid_target` mechanism converts PRAM and
   NOISE selections to kANON whenever a reid_target is set. This means **rule-level
   thresholds cannot influence the final protection method** in the current architecture.

3. **Outcomes are indistinguishable across crossovers.** reid_after and utility_score
   are identical (T2) or nearly identical (T4) on both sides of every crossover.
   The thresholds route to different rules, the rules select different methods, but
   the override produces the same output. The thresholds are routing control that
   currently has no effect on outcomes.

4. **T4 crossover placement matches expectations.** On adult_low_reid (reid_95=0.067),
   the crossover appears at threshold=0.075 -- LOW1 fires when the gate (reid_95 <= value)
   opens past the actual reid_95. On greek_low_reid (reid_95=0.125), the crossover
   appears at threshold=0.125. Both are exactly at the reid_95 boundary, confirming
   the patcher is correctly varying the gate.

### Thresholds not exercised

- **T1 (RC1 risk concentration)** is structurally unreachable. `build_data_features()`
  returns empty `var_priority`, so `classify_risk_concentration()` always returns
  `'unknown'` pattern. RC1 never fires regardless of threshold value. Validating T1
  requires populating backward-elimination risk data.

  **Finding for follow-up:** RC1-RC4 rules depend on `features['var_priority']` which
  is not populated by `build_data_features()`. The entire risk-concentration rule family
  may be dormant in production. Requires investigation separate from this validation.

- **T3 (QR2 suppression gate)** showed no crossovers. On all tested datasets, the
  QR2 rule either doesn't fire (low-risk datasets) or fires consistently regardless
  of the suppression gate value (high-risk datasets like CASC/free1 where reid_95=1.0
  triggers higher-priority rules like HR1/HR3 first).

- **High-reid configurations** (adult_high_reid, CASCrefmicrodata, free1) did not
  exercise the tested thresholds because higher-priority rules (HR-series, QR-series)
  fire at reid_95 >= 0.50 regardless of mid-range threshold values. These datasets
  confirm the engine's escalation path works but do not contribute to threshold
  calibration.

### What this means for the engine

The thresholds are designed to route datasets to different SDC methods based on their
characteristics. However, the override rule (`_override_to_kanon_if_reid_target`)
effectively short-circuits this routing. The thresholds would only matter when:

- kANON is **not** available or fails to meet the risk target
- The override rule is disabled (no reid_target set)
- A future architecture change allows PRAM/NOISE to be the final method

With the current architecture, **the thresholds provide defense-in-depth** -- they
correctly select methods for edge cases, but the override rule provides a safety net
that currently dominates.

## Bugs found and fixed in the harness

Three bugs were identified and fixed in Run 3:

1. **Rule capture was broken.** `result.metadata` does not contain `rule_applied`.
   Fixed by parsing log entries with regex `Rule:\s+(\S+)\s+→\s+(\S+)` to extract
   both the rule name and the initial method selection.

2. **LOW1 patcher delegated to original.** The patched function fell through to
   `original(features)` which re-ran the hardcoded `reid_95 <= 0.10` gate, making
   the patcher a no-op. Fixed by re-implementing the full LOW1/LOW2/LOW3 logic in
   the patcher with only the LOW1 gate variable.

3. **CAT1 patcher delegated to original.** Same issue -- `original(features)`
   re-checked both the hardcoded 0.70 gate and `_has_dominant_categories`. Fixed by
   building the CAT1 result dict directly and intentionally removing the dominance
   check to isolate the threshold variable.

4. **Crossover detection compared `selected_method` (final output) instead of
   `selected_rule`.** Since the override rule converts PRAM/NOISE -> kANON, method-level
   comparison could never show crossovers. Fixed by comparing `selected_rule` and tracking
   `initial_method` separately.

## Recommendation

**No threshold changes at this time.** The validation produced no actionable calibration
signal because `_override_to_kanon_if_reid_target` converts perturbative selections to
kANON before outcomes are measured. Rule routing is verified to be functional; threshold
calibration cannot be evaluated until either (a) the override is scoped more narrowly,
or (b) the validation runs with a risk metric other than reid95.

## Files

### Run 3 (reports/run3_fixed/) -- current, with fixed harness
- `results.csv` -- 80 runs, 0 errors
- `crossovers.csv` -- 3 crossovers
- `report.md` -- per-threshold summary

### Historical (reports/latest/, reports/run2_boundary/)
- Runs 1-2 with broken harness (no rule capture, patcher bugs)
- Retained for reference but superseded by Run 3
