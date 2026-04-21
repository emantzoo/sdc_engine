# Spec 22 — Systemic Test Suite Summary

## Overview

Six systemic test files added to prevent known bug classes from recurring.
Total: **359 new tests**, all passing. Regression suite: **475 passing, 0 failures**.

Prior systemic tests (Spec 18): `test_method_metric_coverage.py`, `test_feature_population_coverage.py`.
Spec 22 adds six more, each targeting a distinct structural failure mode.

## Test Files

| # | File | Tests | What it guards |
|---|------|------:|----------------|
| 1 | `test_systemic_rule_coverage.py` | 44 | Every rule factory has a positive-path test reaching `applies=True` |
| 2 | `test_systemic_rule_ordering.py` | 46 | Factory eval order matches code; no rule unreachable after catch-all |
| 3 | `test_systemic_data_contracts.py` | 136 | Shared data structures conform to canonical formats |
| 4 | `test_systemic_metric_consistency.py` | 32 | Named computations are deterministic with documented boundaries |
| 5 | `test_systemic_rule_metric.py` | 92 | Every rule's method is compatible with the active metric |
| 6 | `test_systemic_signature_stability.py` | 9 | Gatekeeper function signatures don't break callers |

Shared fixture: `tests/fixtures/canonical_rules.py` — `FACTORY_ORDER` (16 factories),
`CANONICAL_RULES` (~43 rules), with AST-based staleness guard.

## Architectural Findings

### 1. severe_tail boundary dominance (Test 4)
`classify_risk_pattern` checks `severe_tail` (tail_ratio > 10 AND reid_99 > 0.30)
before `uniform_low` (reid_50 < 0.05). Input with reid_50=0.01, reid_99=0.50
produces `severe_tail`, not `tail` or `uniform_low`. Codified as observed behavior.

### 2. EMERGENCY_FALLBACK narrow reachability (Test 2)
`EMERGENCY_FALLBACK` is only reachable when:
- `data_type != 'microdata'` (bypasses most rule factories)
- `_risk_metric_type = 'k_anonymity'` (blocks PRAM via `_is_allowed()`)
- This causes `DEFAULT_Fallback` (unconditional PRAM catch-all) to be skipped

Without the metric gate, `DEFAULT_Fallback` always fires first.

### 3. access_tier parameter vs features key (Test 2)
`select_method_suite` overwrites `features['_access_tier']` from its `access_tier`
parameter (default `'SCIENTIFIC'`) at line 347 of pipelines.py. Tests that set
`_access_tier` in the features dict must also pass the `access_tier` parameter
explicitly, or the value gets overwritten silently.

### 4. cat_ratio is computed-not-stored (Test 3)
`cat_ratio` is not a key in the features dict. Every consumer independently
recomputes `n_categorical / (n_categorical + n_continuous)` from shared primitive
keys. Less fragile than stored-value divergence, but still a dual-path risk.

### 5. DP1-DP3 dead by position (Test 2)
`LOW3` is an unconditional catch-all in `low_risk_rules()`. Distribution rules
(DP1-DP3) follow it in the factory chain and can never fire. Confirmed dead both
by individual factory call (LOW3 preempts) and full-chain sweep.

### 6. LDIV1 reid95 gate holds (Test 5)
Spec 19's LDIV1 gate (`reid_95 <= 0.10` required for `l_diversity` metric) is
confirmed working. At reid_95=0.20, LDIV1 does not fire.

## Regression Baseline

**475 tests passing, 0 failures** (up from 116 pre-Spec 22).

8 pre-existing failures in `test_cross_metric.py` excluded from count:
- 5 PRAM-gating (aa1f943 regression)
- 3 GENERALIZE_FIRST test-expectation mismatch

## Spec 20 Backlog Items

These findings should be addressed in Spec 20 (cleanup/consolidation):

1. **Pre-Spec-14 PRAM-gating expectations** — The 8 `test_cross_metric.py` failures
   likely test pre-Spec-14 expectations. Test 5's full-chain sweep exercises
   `_is_allowed()` correctly. Fix may be updating test expectations, not engine code.

2. **cat_ratio dual-path** — Multiple consumers computing the same ratio from shared
   primitives. Consider extracting to a features dict key or helper function.

3. **access_tier parameter/features inconsistency** — `select_method_suite` silently
   overwrites the features key from its parameter. Document or consolidate.

## Spec 21 Backlog Items

1. **DP1-DP3 deletion** — Confirmed dead by position after LOW3 catch-all. Annotated
   in `canonical_rules.py` with `dead_by_position: True`. Safe to remove.

2. **EMERGENCY_FALLBACK consolidation** — Inline in both `rules.py` and `pipelines.py`,
   hidden from systematic rule-chain analysis. Consider making it a factory output.

## Commits

```
5958f34 Spec 22 Test 1: positive-path rule coverage (44 tests)
e01a368 Spec 22 Test 3: data contract validation (136 tests)
b487df3 Spec 22 Test 5: rule-metric appropriateness (92 tests)
7d5ca36 Spec 22 Test 4: metric consistency (32 tests)
2a211f7 Spec 22 Test 2: rule ordering validation (46 tests)
4a16c21 Spec 22 Test 6: signature stability for gatekeeper functions (9 tests)
```
