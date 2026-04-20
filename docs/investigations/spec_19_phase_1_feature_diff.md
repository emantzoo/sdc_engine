# Spec 19 Phase 1.1 — Feature Extractor Diff

**Date:** 2026-04-20
**Script:** `scripts/spec_19_feature_diff.py`

## Test datasets

| Dataset | Rows | QIs | Sensitive cols | Purpose |
|---------|------|-----|---------------|---------|
| **adult** | 32,561 | age, workclass, education | none | Harness dataset — production-representative |
| **synthetic sensitive** | 500 | region, age_group, income_level | diagnosis | Exercises sensitive-column path (G1/G4 shape) |
| **synthetic G10** | 200 | id_code (near-unique), category | none | Exercises high-uniqueness path (G10 shape) |

Both functions were called with equivalent inputs. `build_data_features` received
pre-computed ReID via `calculate_reid()`. `extract_data_features_with_reid`
received an `analysis` dict from `analyze_data()` (with sensitive columns
patched in where applicable).

---

## Signature comparison

| Parameter | `build_data_features` | `extract_data_features_with_reid` |
|-----------|----------------------|----------------------------------|
| data | `data: DataFrame` | `data: DataFrame` |
| QIs | `quasi_identifiers: List[str]` | `quasi_identifiers: Optional[List[str]]` |
| risk input | `reid: Optional[Dict]` (pre-computed) | Computed internally via `calculate_reid()` |
| data analysis | N/A — analyzes data inline | `analysis: Dict` (from `analyze_data()`) |
| active columns | `active_cols: Optional[List[str]]` | N/A — uses all columns |
| var priority | `var_priority: Optional[Dict]` | N/A — not accepted |
| column types | `column_types: Optional[Dict]` | N/A — not accepted |
| risk metric | `risk_metric: Optional[str]` | N/A — always uses reid95 |
| risk target | `risk_target_raw: Optional[float]` | N/A |
| sensitive cols | `sensitive_columns: Optional[List[str]]` | Via `analysis['sensitive_columns']` |
| preprocess meta | `preprocess_metadata: Optional[Dict]` | N/A |

**Signature mismatch summary:** The two functions have fundamentally different
interfaces. `build_data_features` is self-contained (takes raw parameters),
while `extract_data_features_with_reid` depends on a pre-built `analysis`
dict. Migration will require either:
- Rewriting callers to pass raw parameters instead of an analysis dict, or
- Adding an adapter that unpacks `analysis` into `build_data_features` parameters.

The `analysis` dict wraps: `continuous_variables`, `categorical_variables`,
`sensitive_columns`, `uniqueness_rate`, `risk_level`, `data_type`, and
`quasi_identifiers`. Most of these are re-derived by `build_data_features`
from the raw data, making the `analysis` dict redundant for callers that
have the DataFrame.

---

## Full diff table

Legend:
- **build** = `build_data_features` in `protection_engine.py`
- **reid** = `extract_data_features_with_reid` in `selection/features.py`
- **A** = adult, **S** = synthetic sensitive, **G** = synthetic G10

### Keys present in both functions

| Key | Type | Match across datasets | Value differences | Notes |
|-----|------|----------------------|-------------------|-------|
| `_risk_metric_type` | str | exact (A,S,G) | — | Both hardcode `'reid95'` |
| `categorical_vars` | list | **mismatch (A)**, exact (S,G) | build: QI-only; reid: all columns from `analysis` | See Finding 1 |
| `continuous_vars` | list | **mismatch (A,G)**, exact (S) | build: QI-only; reid: all columns from `analysis` | See Finding 1 |
| `data_type` | str | exact (A,S,G) | — | Both return `'microdata'` |
| `estimated_suppression` | dict | exact (A,S,G) | — | Same groupby logic |
| `estimated_suppression_k5` | float | exact (A,S,G) | — | |
| `expected_eq_size` | float | exact (A,S,G) | — | |
| `has_outliers` | bool | **mismatch (A)**, exact (S,G) | build=False, reid=True on adult | reid scans all continuous vars (3 cols); build scans only QI continuous vars (1 col). See Finding 1 |
| `has_reid` | bool | exact (A,S,G) | — | Both True when ReID succeeds |
| `has_sensitive_attributes` | bool | **mismatch (S,G)**, exact (A) | build: always False; reid: from `analysis` | See Finding 2 |
| `high_cardinality_count` | int | exact (A,S,G) | — | |
| `high_cardinality_qis` | list | exact (A,S,G) | — | |
| `high_risk_count` | int | **mismatch (A,G)**, exact (S) | build: from eq_sizes; reid: from `risk_scores` (always 0) | See Finding 3 |
| `high_risk_rate` | float/int | **mismatch (A,G)**, type mismatch (S) | build: float; reid: int(0) | See Finding 3 |
| `k_anonymity_feasibility` | str | exact (A,S,G) | — | |
| `low_cardinality_qis` | list | exact (A,S,G) | — | |
| `max_achievable_k` | int | exact (A,S,G) | — | |
| `max_risk` | float | exact (A,S,G) | — | |
| `mean_risk` | float | **mismatch (A,S)**, exact (G) | build: `(reid_50+reid_95)/2`; reid: from `calculate_reid().mean_risk` | See Finding 4 |
| `n_categorical` | int | **mismatch (A)**, exact (S,G) | build: QI-only count; reid: all-column count | See Finding 1 |
| `n_columns` | int | exact (A,S,G) | — | |
| `n_continuous` | int | **mismatch (A,G)**, exact (S) | build: QI-only count; reid: all-column count | See Finding 1 |
| `n_qis` | int | exact (A,S,G) | — | |
| `n_records` | int | exact (A,S,G) | — | |
| `qi_cardinalities` | dict | exact (A,S,G) | — | |
| `qi_cardinality_product` | int | exact (A,S,G) | — | |
| `qi_max_category_freq` | dict | exact (A,S,G) | — | |
| `quasi_identifiers` | list | exact (A,S,G) | — | |
| `recommended_qi_to_remove` | str/None | **mismatch (G)** | build: always None; reid: sets it when infeasible | See Finding 5 |
| `reid_50` | float | exact (A,S,G) | — | |
| `reid_95` | float | exact (A,S,G) | — | |
| `reid_99` | float | exact (A,S,G) | — | |
| `risk_level` | str | **mismatch (A,S,G)** | build: from `risk_pattern`; reid: from `analysis['risk_level']` | See Finding 6 |
| `risk_pattern` | str | exact (A,S,G) | — | Same classification logic |
| `sensitive_columns` | dict | **mismatch (S,G)**, exact (A) | build: always `{}`; reid: from `analysis` | See Finding 2 |
| `skewed_columns` | list | **mismatch (A)**, exact (S,G) | build: QI-only; reid: all continuous vars | See Finding 1 |
| `small_cells_rate` | float/int | **type mismatch** (A,S,G) | build: `float(0.0)`; reid: `int(0)` | Trivial — both mean zero. reid initializes as int literal `0` |
| `uniqueness_rate` | float | **mismatch (A,S)**, exact (G) | build: computes from QI groupby; reid: reads from `analysis` dict | See Finding 7 |

### Keys in `build_data_features` only

| Key | Type | Value | Notes |
|-----|------|-------|-------|
| `_risk_assessment` | RiskAssessment/None | None (when reid pre-computed) | Internal to production pipeline. Not consumed by any rule |
| `risk_concentration` | dict | `{'pattern': 'unknown', ...}` when no var_priority | Consumed by RC rules. reid version doesn't compute this |
| `var_priority` | dict | `{}` when not provided | Passed through from caller. reid version doesn't accept this parameter |
| `recommended_qi_to_remove` | str/None | Always None | build always sets it to None; reid sets it conditionally. Both have the key but build never populates it meaningfully |

### Keys in `extract_data_features_with_reid` only

| Key | Type | Value (example) | Rules that consume it | Notes |
|-----|------|-----------------|----------------------|-------|
| `geo_qis_by_granularity` | dict | `{'region': 'coarse'}` | **GEO1** | Classifies geo QIs as fine/coarse by cardinality threshold (50) |
| `integer_coded_qis` | list | `[]` | **DP4** | QIs that are numeric with ≤15 unique integer values |
| `l_diversity` | dict | `{'l_achieved': 3, ...}` | (not directly) | Full l-diversity result dict. Only computed when sensitive cols exist and diversity ≤ 10 |
| `max_qi_uniqueness` | float | `0.002` to `1.0` | **SR3** | max(nunique/n_records) across QIs |
| `min_l` | int/None | `3` | **LDIV1** | Achieved l-diversity value. Only computed when sensitive cols with low diversity exist |
| `n_geo_qis` | int | `0` or `1` | (not directly) | Count of geo-type QIs. Convenience derived from `qi_type_counts` |
| `qi_type_counts` | dict | `{'date':0,'geo':1,...}` | **DATE1**, LDIV1 merge | Classifies QIs by type using name-based heuristics |
| `sensitive_column_diversity` | int/None | `5` or `None` | **LDIV1**, P4a, P4b | min nunique across sensitive columns |

---

## Findings

### Finding 1: `continuous_vars` / `categorical_vars` scope divergence

**build_data_features** classifies only QI columns as continuous or categorical.
**extract_data_features_with_reid** reads `continuous_variables` and `categorical_variables`
from `analyze_data()`, which classifies ALL columns in the dataset.

This means:
- `n_continuous`, `n_categorical` differ when dataset has non-QI columns
- `has_outliers` and `skewed_columns` scan different column sets (QI-only vs all)
- On adult: build finds 1 continuous (age), reid finds 3 (age, fnlwgt, capital.loss)
- On adult: build finds no outliers in age; reid finds outliers in capital.loss

**Impact on rules:** Rules that check `has_outliers` or `n_continuous` will get
different inputs depending on which function was called. The build version's
QI-only scope is intentional — method selection should only consider QI columns.
The reid version's all-column scope is a legacy behavior from `analyze_data()`.

**Phase 1.2 decision:** Keep build's QI-only scope. It's correct for method selection.

### Finding 2: `has_sensitive_attributes` and `sensitive_columns` — hardcoded vs computed

**build_data_features** hardcodes `has_sensitive_attributes = False` and
`sensitive_columns = {}` regardless of the `sensitive_columns` parameter.
This is the Spec 18 unwired feature — the parameter is accepted but unused
for feature computation.

**extract_data_features_with_reid** reads `analysis.get('sensitive_columns')`
(from `auto_detect_sensitive_columns`) and sets `has_sensitive_attributes = bool(...)`.

Note: On the G10 dataset, `auto_detect_sensitive_columns` incorrectly flags
`id_code` as sensitive (because it's a unique identifier), giving
`has_sensitive_attributes = True` and `sensitive_column_diversity = 200`.
This is a false positive from the auto-detection heuristic.

**Phase 1.2 decision:** Wire `has_sensitive_attributes = bool(sensitive_columns)`
and `sensitive_columns` passthrough. Don't rely on auto-detection — require
explicit caller input (which `build_data_features` already accepts).

### Finding 3: `high_risk_count` / `high_risk_rate` — different computation

**build_data_features** computes `high_risk_rate` from equivalence class sizes:
records in classes < 5 (i.e., individual risk > 0.20). This works correctly.

**extract_data_features_with_reid** reads `reid.get('risk_scores', [])` and
counts entries > 0.20. But `calculate_reid()` in `sdc_utils.py` does NOT
return a `risk_scores` key — it returns scalar percentiles. So
`risk_scores` is always `[]`, and `high_risk_count` is always 0.

This is a **bug in the reid version**: it silently produces `high_risk_count=0`
and `high_risk_rate=0` on every dataset. The build version's equivalence-class
approach is correct.

**Phase 1.2 decision:** Keep build's computation. No port needed.

### Finding 4: `mean_risk` — different formula

**build_data_features:** `mean_risk = (reid_50 + reid_95) / 2` — an approximation.

**extract_data_features_with_reid:** `mean_risk = reid.get('mean_risk', reid['reid_50'])` —
reads the actual mean from `calculate_reid()`, which computes `individual_risk.mean()`.

The reid version is mathematically correct. The build version uses a
percentile-based approximation that can diverge significantly (on adult:
build=0.2619, reid=0.1060).

**Phase 1.2 decision:** Port the reid version's approach — read `mean_risk` from
the pre-computed reid dict. `calculate_reid()` already returns this key.

### Finding 5: `recommended_qi_to_remove` — conditional vs hardcoded None

**build_data_features:** Always returns `None`.

**extract_data_features_with_reid:** Sets it to the highest-cardinality QI
when `k_anonymity_feasibility == 'infeasible'` (on G10: `id_code`).

**Phase 1.2 decision:** Port the conditional logic from reid version.

### Finding 6: `risk_level` — different sources

**build_data_features:** Sets `risk_level = risk_pattern` (the pattern classification
it computes: severe_tail, uniform_high, etc.).

**extract_data_features_with_reid:** Reads `analysis.get('risk_level', 'medium')` —
a different classification from `analyze_data()` (low/medium/high based on
disclosure risk thresholds).

These are **two different risk classification systems** producing values with
the same key name. On adult: build says "severe_tail", reid says "medium".

**Phase 1.2 decision:** Keep build's behavior (risk_level = risk_pattern).
Rules check `risk_pattern` directly; `risk_level` is not consumed by any rule
gate. It's informational only.

### Finding 7: `uniqueness_rate` — different computations

**build_data_features:** Computes uniqueness as
`data[qis].drop_duplicates().shape[0] / n_rows` — the fraction of unique
QI combinations.

**extract_data_features_with_reid:** Reads `analysis.get('uniqueness_rate', 0)` —
which comes from `calculate_disclosure_risk()` via `analyze_data()`.
`calculate_disclosure_risk` computes uniqueness as records where
`group_size == 1` divided by total records (fraction of singletons).

These are **semantically different metrics**:
- build: fraction of unique QI value combinations (row-level)
- reid: fraction of records in singleton equivalence classes

On adult: build=0.1060 (10.6% unique combos), reid=0.0360 (3.6% singletons).
On G10 (fully unique): both=1.0 (equivalent when all records are singletons).

**Phase 1.2 decision:** Keep build's computation. It's the one HR1-HR5 consume
(confirmed in Spec 18 sweep — `uniqueness_rate` IS correctly populated by
`build_data_features`). The singleton-fraction from `analyze_data()` is a
different metric that no rule depends on.

---

## Summary

### Reid-only keys (not in `build_data_features`)

**8 keys** are computed by `extract_data_features_with_reid` only:

| Key | Rule(s) | Port needed | Complexity |
|-----|---------|-------------|------------|
| `max_qi_uniqueness` | SR3 | Yes | Low — copy 3-line computation |
| `integer_coded_qis` | DP4 | Yes | Low — copy 8-line loop |
| `qi_type_counts` | DATE1, LDIV1 merge | Yes | Medium — name-based heuristic with Greek locale hints |
| `geo_qis_by_granularity` | GEO1 | Yes | Medium — same name heuristic + cardinality threshold |
| `n_geo_qis` | (convenience) | Yes | Trivial — derived from `qi_type_counts` |
| `sensitive_column_diversity` | LDIV1, P4a, P4b | Yes | Low — min nunique across sensitive cols |
| `min_l` | LDIV1 | Yes | Medium — requires `check_l_diversity` call |
| `l_diversity` | (not directly by rules) | Optional | Medium — full l-diversity computation |

This matches the Spec 18 sweep's 7 unwired keys, plus 2 bonus keys
(`n_geo_qis` as a convenience accessor, `l_diversity` as a full result dict).

### Build-only keys (not in `extract_data_features_with_reid`)

**3 keys** are computed by `build_data_features` only:

| Key | Rule(s) | Notes |
|-----|---------|-------|
| `_risk_assessment` | (internal) | RiskAssessment object for multi-metric support. Not consumed by rules |
| `risk_concentration` | RC1-RC4 | Delegates to `classify_risk_concentration()`. reid version doesn't compute this because it doesn't accept `var_priority` |
| `var_priority` | RC1-RC4, risk_concentration | Passthrough from caller. reid version doesn't accept this parameter |

These do not need porting INTO the reid version (it's being deleted).
But they confirm that `extract_data_features_with_reid` could never support
RC rules even if called — it lacks `var_priority`.

### Value mismatches where both populate (landmines for Phase 1.2)

| Key | Issue | Resolution |
|-----|-------|------------|
| `mean_risk` | build uses `(reid_50+reid_95)/2` approximation; reid reads actual mean from `calculate_reid()` | Port reid's approach — read from reid dict |
| `high_risk_count` / `high_risk_rate` | reid version is BROKEN (reads missing `risk_scores` key, always 0) | Keep build's equivalence-class computation |
| `continuous_vars` / `categorical_vars` | build: QI-only; reid: all columns | Keep build's QI-only scope |
| `has_outliers` / `skewed_columns` | build: QI-only scan; reid: all-column scan | Keep build's QI-only scope |
| `uniqueness_rate` | Different semantic metric (unique combos vs singleton fraction) | Keep build's computation (rules depend on it) |
| `risk_level` | Different classification system | Keep build's (= risk_pattern) |
| `recommended_qi_to_remove` | build: always None; reid: conditional | Port reid's conditional logic |
| `small_cells_rate` | Type mismatch: float(0.0) vs int(0) | Trivial — keep build's float |

### Signature mismatch impact on migration

`extract_data_features_with_reid` takes `(data, analysis, quasi_identifiers)`
while `build_data_features` takes individual parameters. Callers of the reid
version will need to be refactored to pass raw parameters instead of an
`analysis` dict. The `analysis` dict is mostly redundant — `build_data_features`
re-derives the same information from the raw DataFrame.

The one piece of information that callers will need to supply explicitly
(rather than relying on `analyze_data()` auto-detection) is `sensitive_columns`.
This is the correct design — sensitive columns should be explicitly declared,
not auto-detected.

---

## Keys consumed by rules but populated by neither function

None found. Every `features.get()` / `features['KEY']` access in `rules.py`
and `pipelines.py` is populated by at least one of the two functions (or is
a config key injected by `select_method_suite()`). The Spec 18 sweep's count
of 7 unwired keys is confirmed — no additional gaps discovered.

---

## Phase 1.2 input summary

**Port from reid version into build_data_features (8 keys):**
1. `max_qi_uniqueness` — 3-line computation
2. `integer_coded_qis` — 8-line loop
3. `qi_type_counts` — name-based heuristic (~15 lines)
4. `geo_qis_by_granularity` — same heuristic + cardinality threshold (~10 lines)
5. `n_geo_qis` — 1-line derived from `qi_type_counts`
6. `sensitive_column_diversity` — min nunique loop (~8 lines)
7. `min_l` — l-diversity computation (~10 lines, plus import)
8. `l_diversity` — full result dict (computed alongside `min_l`)

**Fix in build_data_features (3 keys):**
1. `has_sensitive_attributes` — change from `False` to `bool(sensitive_columns)`
2. `sensitive_columns` — pass through from parameter instead of `{}`
3. `mean_risk` — read from reid dict instead of `(reid_50+reid_95)/2`

**Port conditionally (1 key):**
1. `recommended_qi_to_remove` — set when feasibility is infeasible

**No action needed (type-only fix):**
1. `small_cells_rate` — already float in build version
