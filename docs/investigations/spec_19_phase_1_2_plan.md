# Spec 19 Phase 1.2 — Pre-Port Plan

**Date:** 2026-04-21
**Prerequisite:** Phase 1.1 diff (`spec_19_phase_1_feature_diff.md`)

---

## Pre-Phase-1.2 Baseline

**242 pass / 8 fail.** The 8 failures in `test_cross_metric.py` are regressions
from commit `aa1f943` (Spec 12 PRAM-gating fix, 2026-04-20), not from Spec 18
or Spec 19. Tests need updating to match new PRAM-gating behavior; tracked
separately. Affected tests: `TestMethodSuiteMetricFiltering` (5 tests) and
`TestHighRiskForcesStructural` (3 tests).

**Follow-up task (not in Spec 19):** Update the 8 tests in
`test_cross_metric.py` to match PRAM-gating behavior from `aa1f943`. Low
priority, isolated to test file. Candidate for Spec 20 or cleanup batch.

---

## Check A — `mean_risk` consumer audit

**Result:** `mean_risk` is NOT consumed by any rule.

Zero matches for `mean_risk` in `rules.py` and `pipelines.py`. The key is
informational — shown in UI reports and diagnostic output. Fixing the formula
is cosmetic: corrects the displayed number but does not change any rule's
firing behavior.

---

## Check B — `extract_data_features_with_reid` caller inventory

### Definitions (3 copies exist)

| File | Line | Status |
|------|------|--------|
| `sdc/selection/features.py:58` | **Canonical.** Used by rules.py and selection/__init__.py | Active |
| `sdc/select_method.py:56` | **Dead code.** Different signature (adds `structural_risk`, `risk_metric`, `risk_target_raw`). No caller imports it | Delete in Phase 1.5 |
| (none — the deprecated `select_method_by_features` in `select_method.py:629` delegates to `rules.select_method_by_features`, which calls the canonical copy) | | |

### Callers

| # | File | Line | Call site | Purpose | Migration complexity |
|---|------|------|-----------|---------|---------------------|
| 1 | `sdc/selection/rules.py` | 92 | `from .features import extract_data_features_with_reid` | Import for `select_method_by_features()` | Low — change import source |
| 2 | `sdc/selection/rules.py` | 1337 | `features = extract_data_features_with_reid(data, analysis, quasi_identifiers)` | **Actual call** inside `select_method_by_features()`. Passes `(data, analysis, qis)` | **Medium** — must unpack `analysis` dict into `build_data_features` params |
| 3 | `sdc/selection/__init__.py` | 37 | `from .features import extract_data_features_with_reid` | Re-export for public API | Low — change import or remove |
| 4 | `sdc/selection/__init__.py` | 66 | `'extract_data_features_with_reid'` in `__all__` | Public API listing | Low — remove entry |

### Non-callers (docstrings / comments / examples only)

| File | Line | Context |
|------|------|---------|
| `sdc/selection/pipelines.py` | 312, 381, 403-404 | Docstrings referencing `extract_data_features_with_reid` as the feature source |
| `sdc/preprocessing/__init__.py` | 55-56 | Example code in module docstring |
| `sdc/selection/rules.py` | 1134 | Comment mentioning the function |
| `sdc/selection/__init__.py` | 19 | Module docstring workflow example |
| `docs/Method_Selection_Guide.md` | 356 | Doc reference |
| `docs/user_guide.md` | 2421 | Doc reference |

### Migration summary

**1 actual call site** (rules.py:1337) plus imports/re-exports.

The call site is `select_method_by_features(data, analysis, qis)` — a legacy
entry point that takes `(data, analysis, quasi_identifiers)`. This function
calls `extract_data_features_with_reid` to build features, then runs the
rule chain.

The production path (`protection_engine.py`) already uses `build_data_features`
directly and passes the result to `select_method_suite`. It does NOT call
`select_method_by_features` or `extract_data_features_with_reid`.

Migration options for the one call site:
- **Option A:** Rewrite `select_method_by_features` to call `build_data_features`
  instead, unpacking the `analysis` dict into raw parameters. This preserves
  the legacy function's signature for any external callers.
- **Option B:** Delete `select_method_by_features` entirely and update callers
  to use `select_method_suite` (the production entry point). This is cleaner
  but higher blast radius.
- **Recommended: Option A** for Phase 1.4. Option B can follow in Phase 2 or
  a later cleanup spec.

Phase 1.4 is **small** — one call site plus import cleanup plus docstring updates.

---

## Check C — Phase 1.2 change plan

### Port changes (reid → build): 8 keys

| # | Key | Direction | Rationale | Source lines in features.py | Commit message skeleton |
|---|-----|-----------|-----------|---------------------------|------------------------|
| 1 | `max_qi_uniqueness` | port from reid | SR3 gate. 3-line max(nunique/n_records) computation | 136-140 | `feat: wire max_qi_uniqueness into build_data_features` |
| 2 | `integer_coded_qis` | port from reid | DP4 gate. 8-line loop detecting numeric cols with ≤15 unique ints | 142-150 | `feat: wire integer_coded_qis into build_data_features` |
| 3 | `qi_type_counts` | port from reid | DATE1 gate + LDIV1 merge. Name-based heuristic with Greek locale hints (~15 lines) | 152-171 | `feat: wire qi_type_counts into build_data_features` |
| 4 | `geo_qis_by_granularity` | port from reid | GEO1 pipeline gate. Same name heuristic + cardinality threshold (~10 lines). Also adds `n_geo_qis` convenience key | 174-183 | `feat: wire geo_qis_by_granularity into build_data_features` |
| 5 | `sensitive_column_diversity` | port from reid | LDIV1, P4a, P4b gate. min nunique across sensitive cols (~8 lines) | 307-316 | `feat: wire sensitive_column_diversity into build_data_features` |
| 6 | `min_l` + `l_diversity` | port from reid | LDIV1 gate. Requires `check_l_diversity` import. Computed when sensitive cols with low diversity exist (~10 lines) | 318-329 | `feat: wire min_l and l_diversity into build_data_features` |

### Fix changes (build-side corrections): 4 keys

| # | Key | Direction | Rationale | Commit message skeleton |
|---|-----|-----------|-----------|------------------------|
| 7 | `has_sensitive_attributes` | fix in build | Currently hardcoded `False`. Change to `bool(sensitive_columns)` | `fix: compute has_sensitive_attributes from sensitive_columns param` |
| 8 | `sensitive_columns` | fix in build | Currently hardcoded `{}`. Pass through from parameter (as dict or convert list→dict) | (same commit as #7 — single logical change) |
| 9 | `mean_risk` | fix in build | Currently `(reid_50+reid_95)/2` approximation. Read `reid.get('mean_risk', ...)` instead. UI-only — no rule consumes it | `fix: use actual mean_risk from reid dict instead of percentile approximation` |
| 10 | `recommended_qi_to_remove` | conditional port | Currently always `None`. Set to highest-cardinality QI when feasibility is infeasible | `fix: set recommended_qi_to_remove when k-anonymity infeasible` |

### No-action items (confirmed keep-build)

| # | Key | Direction | Rationale |
|---|-----|-----------|-----------|
| 11 | `high_risk_count` / `high_risk_rate` | keep build | Reid version is BROKEN (reads missing `risk_scores` key, always 0). Build's eq-size approach is correct |
| 12 | `continuous_vars` / `categorical_vars` / `n_continuous` / `n_categorical` | keep build | Build's QI-only scope is correct for method selection. Reid's all-column scope is legacy |
| 13 | `has_outliers` / `skewed_columns` | keep build | Same scope issue as #12 — build scans QI columns only |
| 14 | `uniqueness_rate` | keep build | Semantically correct metric (unique QI combos). HR1-HR5 consume it. Reid's singleton-fraction is a different metric |
| 15 | `risk_level` | keep build | Build sets `risk_level = risk_pattern`. Reid reads a different classification from `analyze_data()`. No rule gates on `risk_level` directly |
| 16 | `small_cells_rate` | keep build | Already float(0.0) in build. Reid uses int(0). Trivial type difference |

### Commit sequence

Phase 1.2 produces **8 commits** (not 13 — items 7+8 are one logical change,
and no-action items produce no commits):

1. `feat: wire max_qi_uniqueness into build_data_features`
2. `feat: wire integer_coded_qis into build_data_features`
3. `feat: wire qi_type_counts into build_data_features`
4. `feat: wire geo_qis_by_granularity into build_data_features`
5. `feat: wire sensitive_column_diversity into build_data_features`
6. `feat: wire min_l and l_diversity into build_data_features`
7. `fix: compute has_sensitive_attributes and sensitive_columns from param`
8. `fix: mean_risk from reid dict + recommended_qi_to_remove when infeasible`

Each commit is followed by the full regression suite run. The spec says
"commit after each feature ported, run full regression suite after each commit."

### Appendix A — 13-to-8 commit mapping

The Phase 1.1 diff identified 13 actionable items. This table maps each to
its commit (or documents why it produces no commit).

| Diff item | Key(s) | Direction | Commit # | Rationale |
|-----------|--------|-----------|----------|-----------|
| 1 | `max_qi_uniqueness` | port from reid | **Commit 1** | |
| 2 | `integer_coded_qis` | port from reid | **Commit 2** | |
| 3 | `qi_type_counts` | port from reid | **Commit 3** | |
| 4 | `geo_qis_by_granularity` + `n_geo_qis` | port from reid | **Commit 4** | `n_geo_qis` is derived from `qi_type_counts`, ported alongside the geo computation |
| 5 | `sensitive_column_diversity` | port from reid | **Commit 5** | |
| 6 | `min_l` + `l_diversity` | port from reid | **Commit 6** | Two keys from same computation block |
| 7 | `has_sensitive_attributes` | fix in build | **Commit 7** | Folded with item 8 — single logical change |
| 8 | `sensitive_columns` | fix in build | **Commit 7** | Same commit as item 7 — both are "wire the sensitive_columns param" |
| 9 | `mean_risk` | fix in build | **Commit 8** | Folded with item 10 — both are cosmetic fixes to existing keys |
| 10 | `recommended_qi_to_remove` | conditional port | **Commit 8** | Same commit as item 9 |
| 11 | `high_risk_count` / `high_risk_rate` | keep build | **No commit** | Reid version is broken. Build is correct. No change needed |
| 12 | `continuous_vars` / `categorical_vars` / `n_continuous` / `n_categorical` | keep build | **No commit** | Build's QI-only scope is correct. No change needed |
| 13 | `has_outliers` / `skewed_columns` | keep build | **No commit** | Same scope issue as item 12. No change needed |
| — | `uniqueness_rate` | keep build | **No commit** | Different semantic metric. Build's is what rules consume |
| — | `risk_level` | keep build | **No commit** | Different classification system. No rule gates on it |
| — | `small_cells_rate` | keep build | **No commit** | Already float in build. Trivial type difference |

**Summary:** 13 actionable items → 6 ports (commits 1-6) + 2 fixes (commits 7-8)
= 8 commits. 6 no-action items confirmed as keep-build with explicit rationale.
No item dropped.

### Ordering rationale

- Ports 1-4 are pure data-derived features with no parameter dependencies.
  They can be ported in any order.
- Port 5-6 depend on `sensitive_columns` being properly wired, so fix 7
  (has_sensitive_attributes + sensitive_columns passthrough) should logically
  come before ports 5-6. **However**, the spec says ports first, fixes second.
  Since ports 5-6 will compute `sensitive_column_diversity` from
  `sensitive_columns` (which `build_data_features` already accepts as a param),
  the computation can be added first and the `has_sensitive_attributes` flag
  fixed afterward. The computation works independently of the flag.
- Fix 8 (mean_risk + recommended_qi_to_remove) is independent of everything
  else and goes last.

### Test expectations per commit

| Commit | Expected test impact |
|--------|---------------------|
| 1 (max_qi_uniqueness) | No test changes expected. SR3 may now fire on datasets with near-unique QIs |
| 2 (integer_coded_qis) | No test changes expected. DP4 may fire on datasets with integer-coded QIs |
| 3 (qi_type_counts) | No test changes expected. DATE1 may fire on datasets with date-named QIs |
| 4 (geo_qis_by_granularity) | No test changes expected. GEO1 may fire on datasets with geo-named QIs |
| 5 (sensitive_column_diversity) | No test changes expected unless sensitive_columns are passed |
| 6 (min_l + l_diversity) | No test changes expected. LDIV1 may fire when sensitive cols + low diversity |
| 7 (has_sensitive_attributes fix) | DP3, P4a, P4b may now fire when sensitive_columns is non-empty |
| 8 (mean_risk + recommended_qi) | No behavioral change (mean_risk is UI-only). recommended_qi_to_remove now populated when infeasible |

**Key risk:** Commits 1-7 wire features that were previously missing, which
means rules that were "unwired" (couldn't fire) may now fire. This could
change the rule chain's output on existing test datasets if those datasets
happen to satisfy the newly-wired gates. The parity check at Phase 1.3
specifically looks for this. If a test fails, it means the rule was correctly
wired and is now producing a different (potentially better) recommendation.

---

## Phase 1.3 note

After all 8 commits, Phase 1.3 runs the diff script again to verify that
`build_data_features` now produces a superset of what
`extract_data_features_with_reid` produces for all non-config keys.
Any remaining differences are bugs in the port.

---

## Phase 1.4 preview

**Scope:** 1 actual call site (rules.py:1337), 2 imports to change,
1 `__all__` entry to remove, ~6 docstring/comment references to update.

This is a half-day task, not the "2-4 hours depending on caller count"
the spec estimated. The caller count is 1.

---

## Appendix B — Dead copy diff (`select_method.py:56` vs canonical `features.py:58`)

The dead copy in `select_method.py` was inspected to determine whether it
contains unique computations that would be lost on deletion.

### Signature differences

| Parameter | Canonical (`features.py`) | Dead copy (`select_method.py`) |
|-----------|--------------------------|-------------------------------|
| `structural_risk: float = 0.0` | Not present | Present — stored as `features['structural_risk']` |
| `risk_metric: Optional[str]` | Not present | Present — used to select `RiskMetricType` enum |
| `risk_target_raw: Optional[float]` | Not present | Present — passed to `compute_risk()` |

### Computation differences

| Area | Canonical (`features.py`) | Dead copy (`select_method.py`) | Verdict |
|------|--------------------------|-------------------------------|---------|
| **Risk computation** | Calls `sdc_utils.calculate_reid()` directly | Calls `metrics.risk_metric.compute_risk()` + `risk_to_reid_compat()` | Dead copy uses the newer multi-metric API. **But** `build_data_features` already uses this same API (lines 89-93). No unique value |
| **structural_risk** | Not computed | Stored as `features['structural_risk']` | Not consumed by any rule in `rules.py` or `pipelines.py` (grep confirms zero matches). Abandoned feature |
| **reid_90** | Not computed | Stored as `features['reid_90']` | Not consumed by any rule. Legacy percentile |
| **reid dict** | Not stored | Stored as `features['reid']` (full dict) | Not consumed by any rule directly. Informational |
| **_risk_assessment** | Not computed | Stored as `features['_risk_assessment']` | Already computed by `build_data_features`. No unique value |
| **reid_error** | Not computed | Stored on exception as `features['reid_error']` | Diagnostic key. Not consumed by rules. Could be useful but `build_data_features` already handles the exception path differently |
| **Outlier threshold** | 1.5 × IQR | 3 × IQR | Dead copy is less sensitive. `build_data_features` uses 1.5 × IQR (same as canonical). No unique value |
| **Skewness detection** | Numeric skew > 1.5 on continuous vars | Categorical frequency > 70% on categorical vars | **Different metric entirely.** Dead copy checks categorical dominance, canonical checks numeric skewness. Neither is consumed by rules that check `skewed_columns` — those rules only check the list length |
| **Extended features** | `max_qi_uniqueness`, `integer_coded_qis`, `qi_type_counts`, `geo_qis_by_granularity`, `sensitive_column_diversity`, `min_l`, `l_diversity` | None of these | Dead copy is the **oldest** version — predates the extended features added to canonical |
| **QI cardinality** | Full analysis: `qi_cardinalities` dict, `qi_cardinality_product`, `expected_eq_size`, `k_anonymity_feasibility`, `max_achievable_k`, `recommended_qi_to_remove`, `qi_max_category_freq`, `estimated_suppression` | Only `high_cardinality_qis` / `low_cardinality_qis` lists | Dead copy is severely stripped down |

### Verdict

The dead copy is an **older, stripped-down version** that predates the canonical
version's extended features. Its three extra parameters represent abandoned
scaffolding:

- `structural_risk`: Not consumed by any rule. Abandoned feature from an early
  design that was superseded by `var_priority` and `risk_concentration`.
- `risk_metric` / `risk_target_raw`: Multi-metric support. Already present in
  `build_data_features` (which accepts `risk_metric` and `risk_target_raw`
  as parameters and uses the same `compute_risk()` API).

The categorical-frequency skewness detection (checking if a category has > 70%
dominance) is semantically different from the canonical version's numeric
skewness detection — but no rule gates on the specific contents of
`skewed_columns`. Rules only check `len(features.get('skewed_columns', []))`.
Both approaches would produce the same boolean "has skewed columns or not"
in different cases, but neither is critical for rule behavior.

**Conclusion:** No unique computations worth preserving. Deletion is clean.
The `structural_risk`, `risk_metric`, and `risk_target_raw` params are either
abandoned or already present in `build_data_features`. The categorical skewness
heuristic is different but unused by rules in a way that distinguishes it.

---

## Appendix C — Phase 1.5 scope (deletion + cleanup)

Phase 1.5 deletes `extract_data_features_with_reid` and cleans up all
references. Full scope:

### Code deletions

| File | What to delete |
|------|---------------|
| `sdc/selection/features.py:58-332` | Canonical `extract_data_features_with_reid` function |
| `sdc/select_method.py:56-178` | Dead copy of `extract_data_features_with_reid` |
| `sdc/select_method.py:181-196` | Dead `_classify_risk_pattern` helper (only used by dead copy) |

### Import/export removals

| File | Line | What to remove |
|------|------|---------------|
| `sdc/selection/rules.py` | 92 | `extract_data_features_with_reid` from import statement |
| `sdc/selection/__init__.py` | 37 | `from .features import extract_data_features_with_reid` |
| `sdc/selection/__init__.py` | 66 | `'extract_data_features_with_reid'` from `__all__` |
| `sdc/selection/__init__.py` | 19 | Workflow example in module docstring |

### Docstring/comment updates

| File | Line(s) | What to update |
|------|---------|---------------|
| `sdc/selection/pipelines.py` | 312, 381 | Change "Output from extract_data_features_with_reid()" to "Output from build_data_features()" |
| `sdc/selection/pipelines.py` | 403-404 | Update example code in docstring |
| `sdc/preprocessing/__init__.py` | 55-56 | Update example code to use `build_data_features` |
| `sdc/selection/rules.py` | 1134 | Update comment reference |

### Documentation updates

| File | Line | What to update |
|------|------|---------------|
| `docs/Method_Selection_Guide.md` | 356 | Update function reference in table |
| `docs/user_guide.md` | 2421 | Update function reference |

### Verification

- Run full test suite after deletion
- Grep for any remaining references to `extract_data_features_with_reid`
- Confirm zero matches outside investigation docs and git history
