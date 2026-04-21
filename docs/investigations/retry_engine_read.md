# Retry Engine — Structural Read

**Date:** 2026-04-22
**Scope:** Read-only investigation. No code changes.
**Patterns checked:** dual paths, unwired features, metric-method mismatches, data-contract violations between iterations.

---

## Files read and scope covered

| File | Lines | What was checked |
|------|-------|-----------------|
| `sdc_engine/sdc/protection_engine.py` | 797–811, 1254–1450 | `_pick_escalation_start()`, `run_rules_engine_protection()` 3-phase retry, metric filtering |
| `sdc_engine/sdc/smart_defaults.py` | 130–369, 514–760 | `calculate_smart_defaults()` method selection, `apply_smart_workflow_with_adaptive_retry()` 4-tier loop |
| `sdc_engine/sdc/config.py` | 625–682, 937–942 | `PARAMETER_TUNING_SCHEDULES`, `METHOD_FALLBACK_ORDER`, `METRIC_ALLOWED_METHODS`, `GENERALIZE_TIERS` |
| `streamlit_app/pages/3_Protect.py` | entry points | Three UI modes: Auto (→ protection_engine), Smart Combo (→ smart_defaults), AI (→ LLM + fallback) |

---

## Architecture: two independent retry loops

The engine has two retry mechanisms that run **sequentially**, not competitively:

1. **Preprocessing retry** (`smart_defaults.py:514`, `apply_smart_workflow_with_adaptive_retry`):
   Escalates GENERALIZE preprocessing through 4 tiers (light → moderate → aggressive → very_aggressive). Each tier increases `max_categories` aggressiveness. Measures risk fresh after each tier. Feeds into protection method (PRAM or kANON).

2. **Method retry** (`protection_engine.py:1254`, `run_rules_engine_protection`):
   3-phase escalation: (a) pipeline methods, (b) primary + parameter tuning, (c) fallbacks + parameter tuning. Uses `_pick_escalation_start()` to skip early phases when structural risk is high. Filters all methods through `METRIC_ALLOWED_METHODS` at line 1384–1393.

The preprocessing retry runs first (via `calculate_smart_defaults` in the Smart Combo path). The method retry runs in the Auto path via `run_rules_engine_protection`. They don't compete because the UI dispatches to one or the other based on mode selection.

---

## Red flags found

### RF-1: Metric-awareness gap in `calculate_smart_defaults()`

**File:** `smart_defaults.py:337–365`
**Severity:** Medium (mitigated by downstream filter, but causes wasted work)

`calculate_smart_defaults()` selects a protection method based purely on `n_qis`:
- `n_qis <= 4` → PRAM
- `n_qis > 4` → kANON

No check against `METRIC_ALLOWED_METHODS`. Under k_anonymity or uniqueness metrics, PRAM is blocked — but `calculate_smart_defaults` will still recommend it. The adaptive retry loop (`smart_defaults.py:714`) then applies PRAM, measures risk, and reports success/failure without knowing the method is disallowed for the active metric.

**Mitigation:** When the Auto path calls `run_rules_engine_protection`, the metric filter at `protection_engine.py:1384–1393` catches and removes disallowed methods. But in the Smart Combo UI path, the method from `calculate_smart_defaults` is used directly — there's no downstream metric gate.

**Impact:** Smart Combo mode can produce PRAM-protected output under k_anonymity metric. The risk measurement is still valid (risk is metric-aware), but the method choice violates the engine's own `METRIC_ALLOWED_METHODS` contract.

### RF-2: No time guard in adaptive retry

**File:** `smart_defaults.py:514–760`
**Severity:** Low

`run_rules_engine_protection` has a 30-second `ESC_TIME_BUDGET` circuit breaker. The adaptive retry loop in `smart_defaults.py` has no equivalent time guard — it will run all `max_attempts` tiers regardless of elapsed time. With large datasets (100K+ rows) and 4 tiers, each running GENERALIZE + protection + risk measurement, this could take minutes.

**Mitigation:** The loop is bounded by `max_attempts` (default 4), which limits worst-case to 4 iterations. Not a production risk today, but differs from the protection_engine's design philosophy.

---

## Nothing-found confirmations

| Pattern | Result | Evidence |
|---------|--------|----------|
| Dual retry paths (competing loops) | **Clean.** Two loops are sequential, dispatched by UI mode. No competition. | UI mode selector in `3_Protect.py`; Auto → `run_rules_engine_protection`, Smart Combo → `apply_smart_workflow_with_adaptive_retry` |
| Unwired features consumed by retry | **Clean.** All features consumed by both retry paths (`var_priority`, `reid_95`, `n_qis`, `structural_risk`) are populated by `build_data_features` with sensible defaults. | `build_data_features` returns complete dict; `calculate_smart_defaults` uses `.get()` with fallbacks throughout |
| Metric-method mismatch in method escalation | **Clean in Auto path.** `protection_engine.py:1384–1393` filters fallbacks through `is_method_allowed_for_metric()`. Cross-method escalation respects the metric. | `suite['fallbacks']` list comprehension at line 1386–1389 |
| Data-contract violations between iterations | **Clean.** Each tier iteration starts from `original_data.copy()` (line 654–655). Risk is re-measured fresh after each tier's GENERALIZE (line 711) and after protection (line 730). No stale-state accumulation. | Tier loop at `smart_defaults.py:650–760` |

---

## Recommendations

| Finding | Action | Priority |
|---------|--------|----------|
| RF-1: Metric-awareness gap | **Fixed.** Added `METRIC_ALLOWED_METHODS` check in `calculate_smart_defaults()` after method selection. If PRAM is blocked for the active metric, falls back to kANON. Test 5 Layer 6 (13 tests) guards against regression. | Medium — was a silent contract violation in Smart Combo path |
| RF-2: No time guard | No action now. 4-tier bound is sufficient. Document as known difference if retry loops are ever unified. | Low |
