# Spec 19 Phase 1.3 — Parity Report

**Date:** 2026-04-21
**Prerequisite:** Phase 1.2 complete (8 commits, 0 regressions)

---

## Overview

Ran both `build_data_features()` (post-Phase-1.2) and
`extract_data_features_with_reid()` (pre-deletion canonical version) on 6
datasets with equivalent inputs. Diffed all output keys.

**Script:** `scripts/spec_19_parity_check.py`

---

## Datasets

| # | Dataset | Rows | QIs | Sensitive | Purpose |
|---|---------|------|-----|-----------|---------|
| 1 | synthetic micro | 500 | 3 cat | none | baseline categorical |
| 2 | synthetic sensitive | 500 | 3 cat | 1 (diagnosis) | sensitive column path |
| 3 | synthetic G10 | 200 | 2 (near-unique) | none | high-uniqueness edge case |
| 4 | synthetic date+geo | 300 | 4 (year, region, city, age) | none | date/geo name hints |
| 5 | test_small_150 | 150 | 3 (age, gender, city) | none | real data, small |
| 6 | test_census_like_1K | 1000 | 3 (age, gender, race) | none | real data, medium |

---

## Ported Keys — Exact Match Check

The 8 keys ported in Phase 1.2 must produce **identical values** from both
functions on the same data.

| Key | D1 | D2 | D3 | D4 | D5 | D6 | Verdict |
|-----|----|----|----|----|----|----|---------|
| `max_qi_uniqueness` | PASS | PASS | PASS | PASS | PASS | PASS | Clean |
| `integer_coded_qis` | PASS | PASS | PASS | PASS | PASS | PASS | Clean |
| `qi_type_counts` | PASS | PASS | PASS | PASS | PASS | PASS | Clean |
| `n_geo_qis` | PASS | PASS | PASS | PASS | PASS | PASS | Clean |
| `geo_qis_by_granularity` | PASS | PASS | PASS | PASS | PASS | PASS | Clean |
| `sensitive_column_diversity` | PASS | PASS | *DIVERGE* | PASS | PASS | PASS | See below |
| `min_l` | PASS | PASS | PASS | PASS | PASS | PASS | Clean |
| `l_diversity` | PASS | PASS | PASS | PASS | PASS | PASS | Clean |

### Divergence: `sensitive_column_diversity` on G10

- **build:** `None` (no sensitive_columns parameter passed)
- **reid:** `200` (auto_detect_sensitive_columns falsely flags `id_code`)

**Verdict: NOT a port bug.** This is the Phase 1.1 Surprise 3 (G10 false-positive
on sensitive detection). The reid version reads `analysis.get('sensitive_columns')`
which includes auto-detected columns; `id_code` (a near-unique numeric column)
triggers the auto-detector. The build version correctly returns `None` because no
sensitive columns were explicitly specified.

Confirmation: Dataset 2 (synthetic sensitive) passes sensitive columns explicitly
to both functions — and `sensitive_column_diversity` matches exactly (`5`, the
nunique of `diagnosis`). The port logic is correct.

---

## Fixed Keys — Expected Divergences

| Key | D1 | D2 | D3 | D4 | D5 | D6 | Notes |
|-----|----|----|----|----|----|----|-------|
| `mean_risk` | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH | Both use `reid.get('mean_risk', reid_50)` now |
| `recommended_qi_to_remove` | MATCH | MATCH | MATCH | MATCH | MATCH | MATCH | Both None (no infeasible cases in test data) |
| `has_sensitive_attributes` | MATCH | MATCH | DIVERGE | MATCH | MATCH | MATCH | G10: build=False (correct), reid=True (auto-detect FP) |
| `sensitive_columns` | MATCH | DIVERGE | DIVERGE | MATCH | MATCH | MATCH | D2: same keys, different values (detection reasons). D3: G10 FP |

D2 `sensitive_columns` divergence detail: both have `{'diagnosis': ...}` but the
value strings differ — build: `'sensitive (diagnosis)'`, reid:
`'explicit sensitive column (diagnosis)'`. Downstream consumers use `.keys()` only
(confirmed by grep: rules.py:1182, pipelines.py:238). No behavioral difference.

---

## Other Keys — Known Divergences

These are pre-existing differences documented in Phase 1.1, unrelated to Phase 1.2
ports. Listed for completeness.

| Key | Divergence | Documented in |
|-----|-----------|---------------|
| `continuous_vars` | build=QI-only scope, reid=all columns | Phase 1.1 diff |
| `categorical_vars` | follows continuous_vars scope | Phase 1.1 diff |
| `n_continuous` | follows continuous_vars scope | Phase 1.1 diff |
| `n_categorical` | follows categorical_vars scope | Phase 1.1 diff |
| `has_outliers` | build checks QI continuous only, reid checks all | Phase 1.1 diff (scope) |
| `skewed_columns` | build checks QI continuous only, reid checks all | Phase 1.1 diff (scope) |
| `n_columns` | build=active_cols count, reid=all data.columns | Phase 1.1 diff |
| `uniqueness_rate` | different formulas (QI-combo vs analysis) | Phase 1.1 diff |
| `risk_level` | build=risk_pattern copy, reid=analysis.risk_level | Phase 1.1 diff |
| `high_risk_count` | reid reads nonexistent `risk_scores` key (always 0) | Phase 1.1 diff |
| `high_risk_rate` | reid derives from `risk_scores` (always 0) | Phase 1.1 diff |
| `_risk_metric_type` | build=enum value, reid=literal "reid95" | Phase 1.1 diff |

---

## `sensitive_columns.values()` Consumer Check

Per user's flag: confirmed no downstream consumer iterates `.values()`.

- `rules.py:1182` — `list(features.get('sensitive_columns', {}).keys())` — keys only
- `pipelines.py:238` — `list(sens_cols.keys())[:5]` — keys only
- No other references to `features['sensitive_columns']` in rules/pipelines

The dict-value shape difference (detection reason strings) is invisible to
all consumers.

---

## Adaptation Audit (Phase 1.2 ports)

Three of the 8 ports required adaptation due to the interface difference
(analysis dict vs direct parameters):

| Commit | Adaptation | Impact |
|--------|-----------|--------|
| 5 (sensitive_column_diversity) | Iterates `List[str]` param instead of `dict.keys()` | Same min(nunique) result |
| 6 (min_l + l_diversity) | Passes list directly instead of `list(dict.keys())` | Same `check_l_diversity` call |
| 7 (has_sensitive_attributes) | Constructs dict from list instead of passthrough | Same `.keys()` for consumers |

---

## Verdict

**PARITY CHECK: PASSED**

- 47/48 ported-key checks pass exactly (8 keys x 6 datasets)
- 1 divergence (`sensitive_column_diversity` on G10) is a known auto-detection
  false positive, not a port bug — confirmed by exact match on Dataset 2 which
  exercises the same code path with explicit sensitive columns
- 0 unexpected differences after accounting for documented Phase 1.1 divergences
- `sensitive_columns.values()` consumer check: clean (no consumers iterate values)

Phase 1.4 (caller migration) is cleared pending human review.
