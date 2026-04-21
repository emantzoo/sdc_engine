# Contribution Metric Consumer Audit

**Date:** 2026-04-21
**Status:** Complete — one cosmetic issue found (emoji labels in UI)
**Context:** The backward elimination contribution metric `(1 - 1/k) * 100%` has a 50% floor for any QI with k>=2. Contributions are non-normalized (don't sum to 100%). RC2/RC3/RC4 were deleted because `top_pct < 40%` is unreachable. This audit checks whether other consumers are affected.

---

## Methodology

Grepped for all references to `var_priority`, `contrib_pct`, `_compute_var_priority`, `risk_drop_pct`, `risk_concentration`, and `classify_risk_concentration`. Read the consuming code for each match to determine whether non-normalization causes problems.

---

## Consumer Inventory

| # | Consumer | File | What It Uses | Status |
|---|----------|------|--------------|--------|
| 1 | `classify_risk_concentration()` | selection/features.py | `top_pct >= 40%` threshold | SAFE |
| 2 | `risk_concentration_rules()` (RC1) | selection/rules.py | pattern string from #1 | SAFE |
| 3 | GENERALIZE ordering | GENERALIZE.py:264 | Label ranking (HIGH->LOW), secondary by -pct | SAFE |
| 4 | `compute_risk_weighted_limits()` | GENERALIZE.py:1105 | Label only, ignores pct | SAFE |
| 5 | `auto_classify()` dual-signal fusion | auto_classify.py:135 | Relative magnitude (pct / max_contribution) | SAFE |
| 6 | LOCSUPR importance weights | LOCSUPR.py:72 | Ordering only | SAFE |
| 7 | `build_data_features()` | protection_engine.py | Passes var_priority through to consumers | SAFE |
| 8 | `ReidentificationRisk.initialize()` | risk_calculation.py:200 | **Different metric** — normalized risk_drop_pct | SAFE (distinct) |
| 9 | **Configure page labels** | **2_Configure.py:40** | **Emoji-prefixed labels** | **MISLEADING** |
| 10 | **Configure page chart** | **2_Configure.py:276** | **Raw pct in bar chart** | **MISLEADING** |

---

## Detailed Analysis

### SAFE Consumers

**#1-2: Risk concentration (features.py + rules.py)**
Uses `top_pct >= 40%` to classify as "dominated". Since the 50% floor ensures `top_pct >= 50%` always, this threshold always fires — which is the intended behavior after RC2/RC3/RC4 deletion.

**#3-4: GENERALIZE (ordering + limits)**
Uses priority labels (`HIGH`, `MED-HIGH`, `MODERATE`, `LOW`) for ordering and per-QI cardinality limits. The actual `pct` value is used only as a tiebreaker in sort order. Non-normalization doesn't affect label assignment (thresholds are 15%/8%/3%, all below the 50% floor for k>=2 QIs).

**#5: auto_classify() dual-signal fusion**
Normalizes contributions against `max_contribution` (the largest single value), not against total. This produces correct relative rankings regardless of whether raw values sum to 100%.

**#6: LOCSUPR importance weights**
Uses importance ordering (sort by weight), not absolute percentages.

**#8: ReidentificationRisk (legacy path)**
This is a **different** metric — `risk_drop_pct` in `risk_calculation.py` IS normalized (divides by total_drop, sums to 100%). It feeds the Configure page's backward elimination display, NOT the rule chain's `_compute_var_priority`. The two metrics coexist but serve different consumers.

### MISLEADING Consumers

**#9: Configure page labels (2_Configure.py lines 40-47)**

```python
label = "\U0001f534 HIGH"      # 🔴 HIGH
label = "\U0001f7e0 MED-HIGH"  # 🟠 MED-HIGH
label = "\U0001f7e1 MODERATE"  # 🟡 MODERATE
label = "\u26aa LOW"           # ⚪ LOW
```

Issue: Emoji prefixes violate the data contract (canonical labels are plain `HIGH` / `MED-HIGH` / `MODERATE` / `LOW` per MEMORY.md). This was previously flagged (commit c5661cf: "fix: remove emoji prefixes from var_priority labels") — but the fix was applied to the **engine** labels, not the **UI** labels. The UI's `_run_backward_elimination()` re-adds emojis.

Severity: Cosmetic. The emojis only appear in the UI's internal `var_priority` dict (which is then passed to `build_data_features()`). However, `build_data_features()` receives the emoji-prefixed labels and strips them during label matching, so no downstream logic breaks.

**#10: Configure page chart (2_Configure.py lines 272-301)**

The Variable Importance bar chart displays raw non-normalized percentages without explanation. Users may see:
```
Age:    65%
Gender: 55%
Region: 60%
```
...and wonder why the values exceed 100% total. No legend explains the leave-one-out methodology.

Severity: Low. The chart is informational and doesn't affect protection outcomes.

---

## Findings

1. **0 BROKEN consumers** — no code assumes contributions are normalized
2. **1 cosmetic issue** — UI emoji labels in `_run_backward_elimination()` (2_Configure.py)
3. **1 UX gap** — Variable Importance chart shows non-normalized percentages without explanation
4. **All engine consumers use ordering/labels, not absolute percentages** — the 50% floor is harmless

---

## Recommendations

### Fix Now (targeted commit, no spec needed)

1. **Remove emoji prefixes from UI labels** in `_run_backward_elimination()`. Use plain `'HIGH'`, `'MED-HIGH'`, `'MODERATE'`, `'LOW'`. Present emojis in the chart display layer only (color-coded bars already convey severity).

### Consider for Spec 21+

2. **Add chart legend** explaining leave-one-out methodology and why percentages don't sum to 100%.

3. **Investigate the two `risk_drop_pct` paths**: `ReidentificationRisk` (normalized, legacy) vs `_compute_var_priority` (non-normalized, modern). If both feed the UI, users may see inconsistent numbers. Unifying them is a larger task.

---

## Cross-References

- 50% floor analysis: `tests/empirical/sensitivity/reports/family_1_concentration.md`
- RC family preemption: `docs/investigations/spec_16_readiness_rc_family_preemption.md`
- Emoji label fix (engine): commit c5661cf
- Data contract: `var_priority` labels are `HIGH` / `MED-HIGH` / `MODERATE` / `LOW` (no emoji)
