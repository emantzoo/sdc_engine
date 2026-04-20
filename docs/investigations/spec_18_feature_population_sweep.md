# Spec 18 — Feature-Population Sweep

**Date:** 2026-04-20
**Context:** During Spec 18 Item 5 (SR3 fixture), discovered that `max_qi_uniqueness`
is not populated by `build_data_features()`. This is the same class of issue as the
HR1-HR5 `uniqueness_rate` claim from Spec 15. A systematic sweep was ordered to find
all such gaps before proceeding with rule deletions.

**Correction:** The Spec 15/16 claim that `uniqueness_rate` is not populated was
**incorrect**. `build_data_features()` DOES populate `uniqueness_rate` (line 278 of
protection_engine.py). HR1-HR5 are untriggered because they sit in
`uniqueness_risk_rules()` at priority 13 (out of 15), and higher-priority rules
(reid_risk_rules, low_risk_rules, distribution_rules) fire first when `has_reid=True`.
The harness always computes ReID, so HR1-HR5 are preempted, not broken.

---

## Methodology

For every `features.get('KEY')` or `features['KEY']` access in `rules.py` and
`pipelines.py`, checked whether `build_data_features()` populates that key.

---

## Results: 44 feature keys accessed, 7 unwired

### Category 1: Config keys (set by caller, not by `build_data_features()`) — by design

These are NOT gaps. They're set by `select_method_suite()`, the protection engine
caller, or preprocessing. The rule chain expects them to be injected, not computed
from data.

| Key | Set by | Rules |
|-----|--------|-------|
| `_access_tier` | `select_method_suite()` parameter | PUB1, REG1, SEC1 |
| `_utility_floor` | protection engine caller | SEC1 |
| `_reid_target_raw` | protection engine caller | PUB1, REG1 |
| `_audit` | `select_method_suite()` env flag | RC sub-tracing |
| `qi_treatment` | preprocessing pipeline | treatment balance |
| `method_constraints` | optional caller override | select_method_suite |

### Category 2: Unwired features — rules depend on keys `build_data_features()` doesn't compute

| Key | Default when missing | Rules affected | Impact |
|-----|---------------------|----------------|--------|
| `max_qi_uniqueness` | `0` | **SR3** | SR3 gate `max_qi_uniq > 0.70` never satisfied |
| `integer_coded_qis` | `[]` | **DP4** | DP4 gate `integer_coded` falsy → never fires |
| `qi_type_counts` | `{}` | **DATE1**, LDIV1 merge | `n_date = 0` → DATE1 never fires, LDIV1+DATE1 merge dead |
| `geo_qis_by_granularity` | `{}` | **GEO1** pipeline | GEO1 gate needs ≥1 fine + ≥1 coarse geo QI → never fires |
| `sensitive_column_diversity` | `None` | **LDIV1**, P4a, P4b | LDIV1 checks `sens_div is None or sens_div > 5` — short-circuit `or` returns early when `None`, so gate rejects (applies=False). P4a/P4b never reach the check because `has_sensitive_attributes` is `False` |
| `has_sensitive_attributes` | Hardcoded `False` | **DP3**, P4a, P4b | `has_sensitive` always False → rules never fire |
| `min_l` | `None` (via `.get()`) | **LDIV1** | Accessed via `features.get('min_l')` with `None` fallback that triggers a heuristic estimate. Feature itself only set via `feature_overrides` in tests — heuristic always used in production |

### Category 3: Correctly wired — no gap

All other 31 keys are correctly populated by `build_data_features()`.

---

## Affected rules summary

| Rule | Unwired feature(s) | Would fire if wired? | Compute exists elsewhere? |
|------|-------------------|---------------------|--------------------------|
| **SR3** | `max_qi_uniqueness` | Yes (fixture G10 proves it) | `extract_data_features_with_reid()` computes it |
| **DP4** | `integer_coded_qis` | Possibly (needs low-reid integer data) | `extract_data_features_with_reid()` computes it |
| **DATE1** | `qi_type_counts` | Possibly (needs temporal QIs) | `extract_data_features_with_reid()` computes it |
| **GEO1** | `geo_qis_by_granularity` | Possibly (needs geo QIs) | `verify_fixtures.py` computes it via `_inject_geo_features()` |
| **LDIV1** | `sensitive_column_diversity`, `min_l` | Yes (fixture G1 proves it with overrides) | Computed in production path (protection engine), not in `build_data_features()` |
| **P4a, P4b** | `has_sensitive_attributes`, `sensitive_column_diversity` | Yes (fixture G4 proves P4b with sensitive_columns injection) | Requires explicit sensitive column metadata |
| **DP3** | `has_sensitive_attributes` | Possibly | Same as P4a/P4b |

---

## Root cause

`build_data_features()` in `protection_engine.py` and `extract_data_features_with_reid()`
in `features.py` were written at different times. `extract_data_features_with_reid()` computes
more features (max_qi_uniqueness, integer_coded_qis, qi_type_counts) but is used by tests and
legacy paths. `build_data_features()` is the production path but was not kept in sync when new
rules were added that depended on the features.py fields.

Additionally, `geo_qis_by_granularity` is not computed by either function — it's only injected
by `verify_fixtures.py::_inject_geo_features()`.

`sensitive_column_diversity` and `has_sensitive_attributes` are partially populated: the
production protection engine sets them during the full pipeline, but `build_data_features()`
alone does not, because sensitive column analysis happens later in the workflow.

---

## Recommendations (input for Spec 19)

1. **Consolidate the two feature-extraction paths.** `build_data_features()` and
   `extract_data_features_with_reid()` exist because they were written at different
   times. Spec 19 must decide which is canonical and merge them. All other
   recommendations below are downstream of this decision — wiring individual
   features into the wrong path wastes effort.

2. **Wire the 4 data-derived features** that have existing compute functions:
   `max_qi_uniqueness`, `integer_coded_qis`, `qi_type_counts`, `geo_qis_by_granularity`.

3. **Wire sensitive-column features** (`has_sensitive_attributes`, `sensitive_column_diversity`)
   — these require the caller to pass sensitive column info, which `build_data_features()`
   already accepts as a parameter but doesn't use for feature computation.

4. **Add a systemic test** (like `test_method_metric_coverage.py`) that asserts every
   `features.get()` key in rules.py and pipelines.py is present in the output of
   `build_data_features()` (with exemptions for config keys).

5. **Do not delete unwired rules.** SR3, DP4, DATE1, GEO1, LDIV1, P4a/P4b, DP3 are
   not dead code — they're unwired. The fix is to wire them, not delete them.

---

## Spec 16 verdict impact — rules requiring revision

Cross-reference of every unwired rule against its Spec 16 audit verdict. A new
verdict category **unwired** is introduced: the rule's gate logic is valid but
`build_data_features()` does not populate one or more feature keys the gate
depends on, so the rule cannot fire in the production path.

| Rule | Spec 16 verdict | Revised verdict | Reason for revision |
|------|----------------|-----------------|---------------------|
| **GEO1** | live-unverified | **unwired** | `geo_qis_by_granularity` not populated by either `build_data_features()` or `extract_data_features_with_reid()`. Only injected by `verify_fixtures.py::_inject_geo_features()`. Cannot fire in production |
| **P4a** | live-unverified | **unwired** | `has_sensitive_attributes` hardcoded `False` in `build_data_features()`. Gate `has_sensitive and ...` always False. Cannot fire without explicit injection |
| **P4b** | live-unverified | **unwired** | Same as P4a. Additionally depends on `sensitive_column_diversity` which is `None` |
| **DP3** | live-unverified | **unwired** | Same `has_sensitive_attributes` = False issue as P4a/P4b |
| **LDIV1** | live-unverified | **unwired** | `sensitive_column_diversity` not populated (gets `None`); `min_l` only set via `feature_overrides` in tests. G1 fixture proves it fires with injection but not without |
| **SR3** | niche (verified) | **unwired (niche if wired)** | `max_qi_uniqueness` not populated. G10 fixture fires correctly with injection. Rule logic is valid; feature path is broken |
| **DP4** | untriggered → tightened | **unwired + tightened** | `integer_coded_qis` not populated (defaults to `[]`, falsy). Tightening (0.30→0.20) was correct but insufficient — rule also can't fire because the feature is missing |
| **DATE1** | untriggered → widened | **unwired + widened** | `qi_type_counts` not populated (defaults to `{}`). `n_date = 0` always → DATE1 gate never satisfied regardless of threshold. Widening (0.80→0.50) was correct but insufficient |

### Rules whose Spec 16 verdicts remain correct

| Rule | Spec 16 verdict | Status | Notes |
|------|----------------|--------|-------|
| **HR1-HR5** | untriggered | **Correct** | Verdict is right (has_reid=False gate). But Spec 15 Item 3's *explanation* was wrong — claimed `uniqueness_rate` wasn't populated. It IS populated. See correction below |
| **RC2, RC3, RC4** | preempted-always | **Correct** | 50% floor on contribution metric. Not affected by feature-population gaps |
| **DYN_CAT** | unreachable | **Correct** | NOISE/l_diversity contradiction is the primary blocker. Unwired features are secondary |
| **CAT2** | preempted | **Correct** | Preemption by DYN_CAT/DYN_Pipeline is the primary issue |

---

## HR1-HR5 status correction

The Spec 15 Item 3 / Spec 16 claim that HR1-HR5 are dormant due to `uniqueness_rate`
not being populated is **incorrect**. `build_data_features()` DOES compute `uniqueness_rate`
(line 278). HR1-HR5 are untriggered because:

1. They sit at priority 13 in the 15-factory chain
2. Higher-priority rules (reid_risk_rules at priority 11, low_risk_rules at 12,
   distribution_rules at 12.5) fire first when `has_reid=True`
3. The harness always provides ReID

HR1-HR5 ARE reachable on the no-ReID path (when risk computation fails or is skipped).
They're defensive fallbacks, not dead code.
