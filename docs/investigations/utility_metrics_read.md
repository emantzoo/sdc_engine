# Utility Metrics — Structural Read

**Date:** 2026-04-22
**Scope:** Read-only investigation. No code changes.
**Patterns checked:** dual paths, unwired features, metric-method mismatches, data-contract violations, edge cases.

---

## Files read and scope covered

| File | Lines | What was checked |
|------|-------|-----------------|
| `sdc_engine/sdc/utility.py` | 1–1286 (full) | Core utility: `compute_utility`, `compute_per_variable_utility`, `compute_il1s`, `compute_benchmark_analysis`, `compute_distributional_metrics`, `compute_composite_utility`, helpers (`_cramers_v`, `_eta_squared`, `_categorical_preservation`) |
| `sdc_engine/sdc/metrics/utility.py` | 1–441 (full) | Legacy utility: `calculate_information_loss`, `calculate_utility_metrics` (6-component weighted score) |
| `sdc_engine/sdc/metrics/ml_utility.py` | 1–343 (full) | ML utility: `compute_ml_utility` (LogisticRegression accuracy ratio) |
| `sdc_engine/sdc/sdc_utils.py` | 1344–1399 | Duplicate `calculate_information_loss` |
| `sdc_engine/sdc/post_protection_diagnostics.py` | 28–104 | `compare_qi_utility` (per-QI delta with verdict) |
| `sdc_engine/sdc/protection_engine.py` | 605–651 | `_utility_ok`, `_get_utility_score` (escalation gates) |
| `sdc_engine/sdc/smart_defaults.py` | 700–760 | Tier loop utility check (GENERALIZE + protection per tier) |
| `sdc_engine/sdc/select_method.py` | 1793, 2259, 3083–3252, 3711 | Legacy method selector utility calls |
| `streamlit_app/pages/3_Protect.py` | 234–254, 349–529 | UI utility display |
| `streamlit_app/pages/4_Download.py` | 313–372 | HTML report utility sections |

---

## Red flags found

### RF-1: Two independent utility scoring formulas (dual-path)

**Severity:** Medium — affects internal engine decisions, not user-visible numbers.

Two completely different utility scoring functions exist:

| Function | File | Formula | Callers |
|----------|------|---------|---------|
| `compute_utility()` | `utility.py:93` | Pearson r (numeric) + row preservation (categorical), simple mean | UI (3_Protect.py), smart_defaults tier loop, GENERALIZE per-QI gate |
| `calculate_utility_metrics()` | `metrics/utility.py:167` | 6-component weighted: 0.25*(1-IL) + 0.15*corr + 0.20*mean + 0.15*dist + 0.15*(1-supp) + 0.10*retention | `select_method.py` (legacy method selector, 8 call sites) |

The two functions measure different things with different weights. `compute_utility` measures sensitive-column preservation (correlation + row match). `calculate_utility_metrics` measures QI-column statistical properties (information loss + distribution + suppression). They cannot produce the same number on the same data.

**Impact assessment:**
- **User-visible numbers:** The UI uses `compute_utility` → `compute_composite_utility` exclusively. Users see a single, consistent score.
- **Internal engine decisions:** `select_method.py` uses `calculate_utility_metrics` for `max_info_loss` gating. This is the legacy method selector; the current production path (`run_rules_engine_protection` → `select_method_suite`) does NOT call `calculate_utility_metrics` directly — it reads `result.utility_score` set by apply-method handlers (which call `compute_composite_utility`).
- **Risk:** If `select_method.py`'s `recommend_method` is ever called from a production path, its utility gating uses a different formula than the UI displays. Currently, it's only called from the interactor's `smart_protect()` method, which is a secondary API surface.

### RF-2: Duplicate `calculate_information_loss` in two files

**Severity:** Low — cosmetic, creates import confusion.

- `sdc_engine/sdc/sdc_utils.py:1344` — exported via `sdc.__init__`
- `sdc_engine/sdc/metrics/utility.py:16` — exported via `sdc.metrics.__init__`

Both compute the same thing (per-column MAE for numeric, change rate for categorical). Two entry points to the same logic creates confusion about which is canonical.

### RF-3: `compute_utility` returns 1.0 when no columns score

**Severity:** Low — edge case, not observed in production.

At `utility.py:151-152`: if all columns fail to compute (correlation error, type mismatch), `col_scores` is empty and the function returns `1.0` ("perfect utility"). This is arguably wrong — failing to measure is not the same as measuring perfection. The `0.0` fallback when `cols` is empty (line 127) contradicts this: no columns → 0.0, but columns-with-errors → 1.0.

**Impact:** This would only trigger if every column's Pearson correlation AND string comparison both fail, which is unlikely in practice.

---

## Nothing-found confirmations

| Pattern | Result | Evidence |
|---------|--------|----------|
| Unwired features consumed by utility | **Clean.** All utility functions take explicit DataFrames + column lists. No feature-dict consumption. | All signatures in utility.py take `original, processed` as required args |
| Method-dependent formula differences | **Clean.** Utility is computed on output DataFrames regardless of which protection method produced them. No method-specific branches. | `compute_utility`, `compute_composite_utility` are method-agnostic |
| Scale/direction inconsistency in user-visible numbers | **Clean.** All user-visible numbers go through `compute_utility` (0-1, higher=better), displayed as percentages. | 3_Protect.py lines 241, 249-252; 4_Download.py line 372 |
| NaN/infinity on edge cases | **Clean for production paths.** `compute_utility` handles NaN via `pd.notna()` check (line 137), row-count mismatch via `min(len, len)` (line 143), empty columns (returns 0.0). Range-binned values handled via midpoint extraction in `compute_per_variable_utility` (lines 190-214). | Inline try/except blocks around every computation |
| Original data requirement | **Clean.** All utility functions require both original and processed DataFrames. Callers all pass both. The GENERALIZE utility lambda captures `original_data` in closure (smart_defaults.py:708). Protection engine's `_utility_ok` reads pre-computed `result.utility_score` (no original needed). | Lambda at smart_defaults.py:708; gate at protection_engine.py:605 |

---

## User-visible number audit

Every utility number that reaches users:

| Number | UI Label | Computed by | Display location | Scale | Direction |
|--------|----------|-------------|-----------------|-------|-----------|
| Overall utility | "Utility" | `compute_utility()` → `compute_composite_utility()` | 3_Protect.py:353 (metric card), 4_Download.py:372 (HTML) | 0-100% | higher=better |
| Per-variable utility | Per-QI table | `compute_per_variable_utility()` | 3_Protect.py:493-507 (expander table) | 0-100% per col | higher=better |
| IL1s | "Information Loss (IL1s)" | `compute_il1s()` | 3_Protect.py:520-521 (JSON) | 0-1 per var | lower=better |
| Benchmark | "Cross-Tab Benchmark" | `compute_benchmark_analysis()` | 3_Protect.py:510-511 (JSON) | various | various |
| Distributional | "Distributional Comparison" | `compute_distributional_metrics()` | 3_Protect.py:516-517 (JSON) | KL:[0,∞), Hellinger:[0,1] | lower=better |
| QI delta | Per-QI bar chart | `compare_qi_utility()` | 3_Protect.py:525-529 (chart) | ±% delta | smaller=better |
| Retry trajectory | "Utility" (y-axis) | `compute_utility()` per tier | 3_Protect.py:417-425 (line chart) | 0-100% | higher=better |
| Scenario radar | "Utility" (axis) | `result.utility_score` | 3_Protect.py:741 (radar) | 0-1 | higher=better |
| ML accuracy ratio | (diagnostic) | `compute_ml_utility_multi()` | auto_diagnostics only | 0-1 | higher=better |
| AI estimate | "Estimated utility" | Cerebras API (external) | 3_Protect.py:1067 (text) | free text | n/a |

**Scale consistency:** All user-facing utility scores are 0-1 (displayed as percentages), higher=better. The IL1s and distributional metrics have reversed direction but are displayed in JSON expanders clearly labeled. No scale mixing detected.

---

## Hardcoded thresholds in utility computation

| Threshold | Value | File:line | Configurable? | Notes |
|-----------|-------|-----------|---------------|-------|
| Cross-tab min group size | 3 | utility.py:541, 879 | No | Used for η² and Cramér's V subgroup analysis |
| η² weak association | 0.02 | utility.py:580 | No | Below this, cross-tab pair skipped |
| Cramér's V weak | 0.05 | utility.py:652 | No | Below this, cross-tab pair skipped |
| Relationship preservation floor | 0.40 | utility.py:1273 | No | Prevents composite collapse |
| Post-check degradation threshold | 0.60 | utility.py:1107 | Yes (parameter) | Default; callers can override |
| GENERALIZE per-QI utility gate | 0.60 | smart_defaults.py:711 | Yes (`min_utility`) | `max(min_utility, 0.60)` — 0.60 is hard floor |
| Escalation utility floor | 0.88 | config.py:919 | Yes (via `info_loss_max`) | `1.0 - info_loss_max` |
| Perturbative challenge gain | 0.02 | protection_engine.py:~960 | No | Min 2% utility gain to accept PRAM over structural |
| KL divergence epsilon | 1e-10 | utility.py:1058 | No | Prevents log(0) |
| Cramér's V cardinality cap | 500 | utility.py:~820 | No | Above this, column skipped |

None of these thresholds are problematic. The 0.60 hard floor for per-QI utility in GENERALIZE is the most impactful (controls binning rollback), and it has a defensible rationale (below 60% per-QI utility means the variable is analytically useless).

---

## Recommendations

| Finding | Action | Priority |
|---------|--------|----------|
| RF-1: Dual utility formulas | **No action now.** The legacy `calculate_utility_metrics` in `metrics/utility.py` is only called from `select_method.py` (legacy method selector). If `select_method.py` is eventually retired, `calculate_utility_metrics` becomes dead code. Document in the meantime. | Low — monitor during future cleanup |
| RF-2: Duplicate `calculate_information_loss` | **No action now.** Cosmetic. Could clean up when `sdc_utils.py` is refactored. | Low |
| RF-3: Empty col_scores → 1.0 | **No action now.** Edge case unlikely to trigger. If it does, the 1.0 would be visible in the UI as "perfect utility" when the data is actually unmeasurable — but this scenario requires every column to fail both Pearson and string comparison, which implies broken data rather than a normal protection path. | Low |
