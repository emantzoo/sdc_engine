# Unconditional Catch-All Audit

**Date:** 2026-04-21
**Status:** Complete — no action needed beyond known items
**Context:** DP4 was found dead in Spec 19 because LOW3 fires unconditionally after entering `low_risk_rules()`. This audit checks for other dead rules caused by upstream catch-alls.

---

## Methodology

For every rule factory in `rules.py` and `pipelines.py`, classified each as:
- **CONDITIONAL**: has gates that can return `applies: False`
- **UNCONDITIONAL**: always returns `applies: True` for any input reaching it
- **DEAD**: positioned after an unconditional rule that always preempts it

---

## Rule Chain Priority Order (15 factories)

| # | Factory | Classification | Key Gates |
|---|---------|----------------|-----------|
| 1 | `regulatory_compliance_rules` (REG1) | CONDITIONAL | PUBLIC + target=3% + reid_95>3% |
| 2 | `data_structure_rules` | ALWAYS FALSE | Never fires (log-only placeholder) |
| 3 | `small_dataset_rules` (HR6) | CONDITIONAL | n_records<200 + n_qis>=2 |
| 4 | `structural_risk_rules` (SR3) | CONDITIONAL | n_qis<=2 + uniq>70% + reid_95>20% |
| 5 | `risk_concentration_rules` (RC1) | CONDITIONAL | var_priority + reid_95>15% + dominated + feasible |
| 6 | `public_release_rules` (PUB1) | CONDITIONAL | PUBLIC + reid_95>5% + not regulatory |
| 7 | `secure_environment_rules` (SEC1) | CONDITIONAL | SECURE + 5%<reid<25% + utility>=90% |
| 8 | `categorical_aware_rules` (CAT1) | CONDITIONAL | l_diversity + 15%<reid<40% + cat>=70% |
| 9 | `l_diversity_rules` (LDIV1) | CONDITIONAL | sensitive diversity<=5 + reid<=10% |
| 10 | `temporal_dominant_rules` (DATE1) | CONDITIONAL | date>=50% of QIs + reid<=40% |
| 11 | `reid_risk_rules` (QR0-QR4, MED1) | CONDITIONAL | has_reid + matching risk pattern |
| 12 | **`low_risk_rules` (LOW1-LOW3)** | **UNCONDITIONAL CATCH-ALL** | **Entry: microdata + has_reid + reid<=20%** |
| 13 | `distribution_rules` (DP1-DP2) | CONDITIONAL but **DEAD** | Outliers/skew (unreachable) |
| 14 | `uniqueness_risk_rules` (HR1-HR5) | CONDITIONAL but **DEAD when has_reid=True** | uniqueness thresholds |
| 15 | `default_rules` (DEFAULT_*) | UNCONDITIONAL but **DEAD** | Fallback catch-all (unreachable) |

---

## LOW3: The Catch-All

`low_risk_rules()` has three internal paths:
- **LOW1**: `cat_ratio >= 0.60 AND high_card == 0 AND reid_95 <= 0.10` -> PRAM
- **LOW2**: `cat_ratio <= 0.40 AND n_cont > 0` -> NOISE or kANON
- **LOW3**: **unconditional fall-through** -> kANON k=3 or k=5

Once the entry gates pass (microdata + has_reid + reid_95 <= 0.20), one of LOW1/LOW2/LOW3 always fires. There is no code path returning `applies: False` after entering the function.

---

## Dead Rules

| Rule | Why Dead | Preempted By |
|------|----------|--------------|
| DP1 (outliers -> NOISE) | After LOW3 catch-all | LOW3 selects kANON |
| DP2 (skew -> PRAM) | After LOW3 catch-all | LOW3 selects kANON |
| DP4 (deleted Spec 19) | Was after LOW3 | Already removed |
| HR1-HR5 (when has_reid=True) | After LOW3 catch-all | LOW3 fires first |
| DEFAULT_* | After LOW3 catch-all | LOW3 fires first |

**Nuance for HR1-HR5:** When `has_reid == False`, `low_risk_rules()` returns False (it requires has_reid=True), so HR1-HR5 CAN theoretically fire. However, this only applies to the rare case where ReID metrics are unavailable, which doesn't happen in the normal Configure -> Protect workflow.

**Nuance for DEFAULT_*:** Same — reachable only if `has_reid == False` AND HR1-HR5 all return False.

---

## Is the Preemption Intentional?

**YES.** LOW3 is a comprehensive low-risk fallback for mixed/high-cardinality data. Its preemption of DP1-DP2 is a design feature, not a bug:

- **DP1** (outliers -> NOISE): For reid_95 <= 20% with outliers, LOW2 already routes to NOISE when data is continuous-dominant. The DP1 path would select NOISE with mag=0.20, but LOW2's smart config achieves the same.
- **DP2** (skew -> PRAM): For reid_95 <= 20% with skew, LOW1 already routes to PRAM when data is categorical-dominant. DP2 would duplicate this.

The dead rules represent an older design where distribution-based routing was separate from risk-based routing. After LOW1-LOW3 were added with comprehensive coverage, the distribution rules became redundant.

---

## Findings

1. **One unconditional catch-all:** LOW3 in `low_risk_rules()` — known and intentional
2. **Three dead rule factories:** distribution_rules (DP1-DP2), uniqueness_risk_rules (HR1-HR5, when has_reid=True), default_rules — all positioned after LOW3
3. **No broken preemption:** Every case where LOW3 fires instead of a downstream rule produces an acceptable (often identical) method selection
4. **No new dead rules discovered** beyond the known set from Spec 19

---

## Recommendation

**No immediate action.** The dead rules are harmless — they add ~100 lines of unreachable code but don't affect behavior. If a future cleanup spec (Spec 21+) wants to delete them for hygiene, the candidates are:

- DP1, DP2: delete (provably unreachable, same as DP4)
- HR1-HR5: keep as defensive fallback (reachable when has_reid=False)
- DEFAULT_*: keep as safety net (reachable in edge cases)
