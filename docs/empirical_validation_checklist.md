# Empirical Validation Checklist

All heuristic thresholds across both branches — `auto_sdc_v2_manly` (Python-native engine) and `feature/enhanced-risk-views_MicroSDC` (R/sdcMicro bridge) — with rationale and what testing would confirm.

All thresholds were tuned against Greek property/demographic datasets during development. None are derived from theory.

---

## Validation Priority

| Priority | Category | Impact if wrong |
|---|---|---|
| P1 | Method routing — wrong method selected | High — wasted iterations, suboptimal protection or unnecessary utility loss |
| P2 | Parameter calibration — right method, wrong starting params | Medium — retry engine compensates, but wastes 1-3 iterations |
| P3 | Classification boundary — cosmetic or low-impact | Low — downstream logic handles it |

---

## Branch: `auto_sdc_v2_manly` (Python-Native Engine)

## P1: Method Routing Thresholds

### 1.1 Risk Concentration Cutpoints

**File:** `selection/features.py` — `classify_risk_concentration()`

| Threshold | Value | Routes to | Rationale |
|---|---|---|---|
| Dominated | top QI ≥ 40% | RC1 → LOCSUPR k=5 | One column drives nearly half the risk — targeted suppression is more surgical |

> **Note (Spec 19 Phase 2):** RC2 (Concentrated) and RC3 (Spread High) were deleted — structurally unreachable under the current backward elimination contribution metric (minimum contribution per QI is ~50%, so the `dominated` pattern always matches first). Only RC1 remains.

**What to test:** Run datasets where the top QI contributes 30%, 35%, 40%, 45%. Compare LOCSUPR (RC1 path) vs kANON (QR path) outcomes at each level. Find the crossover point where LOCSUPR starts outperforming kANON. If it's consistently at 35% rather than 40%, adjust.

**Risk if wrong:** At 40%, a dataset where one QI contributes 38% misses RC1 and falls to generic QR rules → kANON instead of LOCSUPR. If LOCSUPR would have been better, the engine wastes iterations on kANON before falling back. The retry engine catches this, but at the cost of time and potentially suboptimal utility.

**Current confidence:** High. 40% is a strong concentration signal. The margin of error matters most in the 35-45% range.

---

### 1.2 Categorical Ratio for PRAM (CAT1)

**File:** `selection/rules.py` — `categorical_aware_rules()`

> **Metric gate (Spec 12 F3a):** CAT1, CAT2, and DYN_CAT are gated to `l_diversity` metric only. PRAM invalidates frequency-count-based metrics (reid_95, k_anonymity, uniqueness). When metric is not l_diversity, these rules return `applies: False`.

| Threshold | Value | Routes to |
|---|---|---|
| CAT1 gate | **l_diversity** + cat_ratio ≥ 0.70 | PRAM p=0.25-0.35 |
| CAT2 gate | **l_diversity** + 0.50 < cat_ratio < 0.70 | Pipeline: NOISE → PRAM |
| Below 0.50 or non-l_diversity metric | — | Falls through to QR rules → kANON |

**What to test:** Under l_diversity metric: datasets with 60%, 65%, 70%, 75% categorical QIs at moderate risk (ReID 15-30%). Compare PRAM vs kANON outcomes. Under reid_95/k_anonymity: verify CAT1 never fires.

**Risk if wrong:** Under l_diversity, a dataset with 68% categorical QIs and ReID=25% misses CAT1, falls to QR4 → kANON k=7. PRAM might have achieved the same risk reduction with zero suppression and better utility. Under reid_95, this is working as designed — PRAM would fragment equivalence classes and inflate the metric.

**Current confidence:** High for metric gate (confirmed against sdcMicro docs). Medium-high for 70% threshold under l_diversity.

---

### 1.3 LOW1 PRAM Gate vs LOW Outer Gate

**File:** `selection/rules.py` — `low_risk_rules()`

| Threshold | Value | Effect |
|---|---|---|
| LOW outer gate | reid_95 > 0.20 → skip | LOW rules only fire at ≤20% |
| LOW1 PRAM | reid_95 ≤ 0.10 + cat_ratio ≥ 0.60 | PRAM p=0.15-0.20 |
| LOW2/LOW3 | reid_95 ≤ 0.20 | kANON k=3-5 or NOISE |

**The dead zone:** Datasets with 10-20% risk and ≥60% categorical QIs get LOW3 (kANON k=3-5), not PRAM. Is kANON always better than PRAM at 15% risk with high categorical ratio?

**What to test:** Categorical-dominant datasets at ReID 12%, 15%, 18%. Compare PRAM p=0.20 vs kANON k=5. If PRAM consistently achieves target with better utility, extend LOW1's gate from ≤0.10 to ≤0.15 or ≤0.20.

**Risk if wrong:** Mild. kANON k=3-5 at low risk usually works fine, just with slightly more suppression than PRAM would cause. The utility difference is small.

**Current confidence:** Medium. The 10% gate is conservative — PRAM at 15% risk is probably fine for categorical-dominant data, but it can't reduce ReID (only kANON/LOCSUPR can guarantee k-anonymity), so the conservative choice to use structural methods above 10% has a defensible safety rationale.

---

### 1.4 RC Gate at 15%

**File:** `selection/rules.py` — `risk_concentration_rules()`

| Threshold | Value | Effect |
|---|---|---|
| RC gate | reid_95 > 0.15 | RC1 can fire |
| Original | reid_95 > 0.20 | Lowered to 0.15 (originally for RC4, now only RC1 remains) |

**Concern:** RC1 (LOCSUPR k=5) can now fire at 16% risk if one QI contributes ≥40%. Is targeted suppression warranted at relatively low risk just because risk is concentrated?

**What to test:** Datasets with ReID 16-20% and dominated concentration. Compare RC1 path (LOCSUPR k=5) vs what QR rules would pick (likely MED1 → kANON k=5). If kANON performs comparably at this risk level, the lower gate adds no value.

**Risk if wrong:** Low. LOCSUPR k=5 at 16% risk is not harmful — it's just potentially unnecessary when kANON k=5 would also work. Both achieve similar protection; the question is which has better utility for the specific data shape.

**Current confidence:** Medium-high. The gate at 0.15 only affects RC1 in practice.

> **Post-investigation note (Spec 19 Phase 2):** RC2/RC3/RC4 deleted — structurally unreachable under the current backward elimination contribution metric (minimum contribution ~50%, so `dominated` always matches first). The gate lowering to 0.15 only benefits RC1. See `docs/investigations/spec_16_readiness_rc_family_preemption.md`.

---

## P2: Parameter Calibration Thresholds

### 2.1 Suppression Gate

**File:** `selection/rules.py` — `_suppression_gated_kanon()`

| Threshold | Value | Effect |
|---|---|---|
| Switch threshold | > 0.25 | Switch from kANON to PRAM/LOCSUPR |
| Clamp threshold | 0.30 | `_clamp_k_by_suppression` walks down k schedule |

**Rationale:** 25% suppression means losing a quarter of records. Industry considers >15% concerning, >30% unacceptable. 25% as a switch point is between "warning" and "unacceptable."

**What to test:** Datasets where estimated suppression at proposed k is 20%, 25%, 30%. Compare outcomes: kANON at that k vs switching to PRAM. Is 25% the right inflection point where PRAM + zero suppression gives better overall outcome than kANON + 25% record loss?

**Risk if wrong:** If 25% is too high, the engine runs kANON and suppresses 24% of records when PRAM would have been better. If too low, it switches to PRAM prematurely and misses the structural guarantee.

**Current confidence:** High. 25% is a well-reasoned midpoint. The exact value matters less because the retry engine's fallback chain includes the alternative method anyway.

---

### 2.2 Dataset Size Adjustment

**File:** `selection/rules.py` — `_size_adjusted_k()`

| Range | Adjustment | Rationale |
|---|---|---|
| ≥ 5,000 | No change | Large enough for structural methods |
| 2,000-5,000 | k≥10→7, k≥7→5 | Moderate reduction |
| 1,000-2,000 | k reduced by 2 (min 3) | More aggressive reduction |
| 500-1,000 | k reduced by 2 (min 3) | Same |
| < 500 | Cap at k=3 | Structural methods are destructive |
| < 200 | HR6 fires | LOCSUPR k=3 only |

**What to test:** Datasets at 800, 1500, 3000, 6000 rows with the same QI structure. Compare suppression rates at proposed k vs size-adjusted k. Verify that the adjustment actually prevents catastrophic suppression without under-protecting.

**Risk if wrong:** If too aggressive (reducing k too much), small datasets get insufficient protection. If too conservative (not reducing enough), they get excessive suppression. The retry engine compensates either way, but starting closer to optimal saves iterations.

**Current confidence:** High. The schedule follows a logical progression and the floor at k=3 prevents under-protection.

---

### 2.3 Smart Config Method Switch Thresholds

**File:** `sdc/smart_method_config.py` — `suggest_kanon_config()`

| Threshold | Value | Effect |
|---|---|---|
| kANON → PRAM switch | est. suppression > 25% | Pre-application swap |
| LOCSUPR → kANON switch | est. cell suppression > 10% | Pre-application swap |
| PRAM → kANON switch | >50% QIs LOW effectiveness | Pre-application swap |

**What to test:** The PRAM effectiveness check ("50% of QIs have LOW effectiveness") is the least grounded. What constitutes "LOW effectiveness" for PRAM on a specific column? This likely depends on cardinality and category dominance — worth verifying the definition matches real-world PRAM performance.

**Current confidence:** kANON and LOCSUPR switches: high. PRAM switch: medium.

---

### 2.4 Cross-Method Starting Points

**File:** `sdc/protection_engine.py`

| kANON failed at k= | PRAM starts at p= | NOISE starts at mag= |
|---|---|---|
| 3 | 0.10 | 0.05 |
| 5 | 0.15 | 0.10 |
| 7-10 | 0.20 | 0.15 |
| 15+ | 0.25 | 0.20 |

**Rationale:** These are from a cross-method equivalence table — "kANON k=5 provides roughly the same protection as PRAM p=0.15." This equivalence is approximate and dataset-dependent.

**What to test:** Run kANON at k=5 and PRAM at p=0.15 on the same datasets. Compare ReID reduction. If PRAM p=0.15 consistently achieves less reduction than kANON k=5, the starting point should be higher (p=0.20).

**Risk if wrong:** Low. If the starting point is too low, the fallback's escalation schedule corrects within 1-2 steps. If too high, it wastes perturbation budget on the first attempt.

**Current confidence:** Medium. The equivalence is reasonable but hasn't been empirically calibrated.

---

## P3: Classification Boundaries

### 3.1 Risk Pattern Classification

**File:** `sdc/metrics/reid.py` — `classify_risk_pattern()`

| Pattern | Condition | Concern |
|---|---|---|
| uniform_high | reid_50 > 0.20 | Solid — half of records at >20% risk is clearly "uniform high" |
| severe_tail | tail_ratio > 10 AND reid_99 > 0.30 | Why 10× and not 8× or 15×? The AND condition is reasonable but the ratio threshold is arbitrary |
| tail | reid_95 > 0.30 OR (reid_99 > 0.50 AND tail_ratio > 5) | Two paths — the OR could fire on different data shapes. Worth testing which path fires more often |
| bimodal | abs(mean - median) > 0.15 | Weakest classifier. Mean-median divergence of 0.15 could trigger on right-skewed but unimodal distributions |
| uniform_low | reid_50 < 0.05 AND reid_99 ≤ 0.20 | Solid — median risk below 5% with no extreme tail |

**What to test:** Generate synthetic risk distributions with known shapes (true bimodal, true tail, true uniform). Run `classify_risk_pattern()` and check if classifications match ground truth. The bimodal detector is most likely to misfire.

**Risk if wrong:** Misclassifying "tail" as "moderate" routes to MED1 (kANON k=5) instead of QR1 (LOCSUPR k=5). Both are reasonable methods — the impact is suboptimal rather than wrong.

**Current confidence:** uniform_high, uniform_low: high. severe_tail: medium-high. tail: medium. bimodal: medium-low.

---

### 3.2 Sensitive Detection Tiers

**File:** `sdc/auto_classify.py`

| Tier | Score | Confidence |
|---|---|---|
| High | ≥ 0.50 | Strong signals — almost certainly sensitive |
| Medium | ≥ 0.35 | Some ambiguity |
| Low | ≥ 0.20 | Genuinely ambiguous |

**What to test:** Columns classified as Sensitive with score 0.20-0.35 (Low confidence). How often are these correctly Sensitive vs actually Unassigned or QI? If most Low-confidence Sensitives turn out to be Unassigned in user review, raise the floor to 0.25 or 0.30.

**Risk if wrong:** Low. Sensitive classification doesn't affect protection — sensitive columns are preserved by default. The only impact is whether the user sees a column pre-labelled as Sensitive vs Unassigned in the review table.

**Current confidence:** High (0.50), Medium (0.35), Medium-low (0.20).

---

### 3.3 QI Fusion Formula Weights

**File:** `sdc/auto_classify.py`

```
fused = 0.30 × keyword_confidence + 0.70 × risk_contribution_normalized
```

**The question:** Is 30/70 the right weight split? This says risk contribution is 2.3× more important than keyword matching.

**What to test:** Datasets where keyword scoring disagrees with risk contribution. Example: column named "municipality" (DEFINITE keyword → high score) but with very low risk contribution (2%). The DEFINITE floor protects this case. But for PROBABLE keywords at low risk — is 30/70 right? A column named "occupation" (PROBABLE, score ~0.70) with 3% risk contribution: fused = 0.30×0.70 + 0.70×0.03/max = 0.21 + ~0.02 = 0.23. That's below the 0.35 QI threshold — not classified as QI. Is that correct? Occupation at 3% risk probably should be QI.

**Risk if wrong:** Medium. The DEFINITE floor catches the obvious QIs. The 30/70 weighting most affects PROBABLE-tier columns with low-moderate risk contribution — these are exactly the ambiguous cases the user is expected to review.

**Current confidence:** Medium. The formula works well when risk contribution is available and meaningful. When risk contribution is noisy (small datasets, few columns), keyword weight should probably be higher.

---

### 3.4 Domain Boosters

**File:** `sdc/auto_classify.py`

| Domain | Boost |
|---|---|
| Date/temporal | +0.15 |
| Geographic | +0.15 |
| Demographic | +0.10 |

**The question:** Why dates and geography get +0.15 but demographics only +0.10? Demographic variables (age, gender, education) are equally classic QIs.

**What to test:** Check if any demographic columns are missed as QIs that would have been caught with +0.15. If yes, equalise the boost.

**Risk if wrong:** Very low. The boost is additive on top of keyword + risk scoring. The difference between +0.10 and +0.15 only matters for columns right at the 0.35 threshold boundary.

**Current confidence:** High. The relative ordering (date/geo slightly above demographics) reflects that dates and geography have higher re-identification power per variable in most datasets.

---

## P3: Preprocessing Thresholds

### 3.5 Structural Risk Tier Overrides

**File:** `sdc/smart_defaults.py`

| SR | Starting Tier | Utility Floor Adjustment |
|---|---|---|
| > 50% | Aggressive (max_cat=5) | −10pp (min 55%) |
| > 80% | Aggressive | −20pp (min 50%) |
| > 30% | Moderate (max_cat=10) | No adjustment |
| ≤ 30% | Light (max_cat=15) | No adjustment |

**What to test:** Datasets with SR=35%, SR=55%, SR=85%. Does skipping directly to Aggressive at SR=55% actually save iterations compared to starting at Moderate? The assumption is "light/moderate will fail so skip them" — but if Moderate succeeds at SR=55%, we've over-generalised.

**Risk if wrong:** Over-generalisation reduces utility unnecessarily. Under-generalisation wastes retry iterations. The utility floor adjustment at SR>50% (-10pp) could be too aggressive for datasets where SR is high but the data is otherwise well-structured.

**Current confidence:** Medium-high. The tier skip is a time optimisation, not a correctness issue. Starting too aggressive is always safe (protection-wise), just potentially wasteful of utility.

---

### 3.6 Risk-Weighted Per-QI Limits

**File:** `sdc/smart_defaults.py` — `compute_risk_weighted_limits()`

| Risk Tier | max_categories |
|---|---|
| HIGH (≥ 15%) | max(5, global // 2) |
| MED-HIGH (≥ 8%) | max(5, int(global × 0.8)) |
| MODERATE (≥ 3%) | global (unchanged) |
| LOW (< 3%) | min(20, int(global × 1.5)) |

**What to test:** The 15%/8%/3% boundaries. Does a QI contributing 14% really need the same treatment as one contributing 4%? The jump from "global × 0.8" to "global // 2" at the 15% boundary is significant.

**Risk if wrong:** Over-binning a HIGH QI (global//2 when ×0.8 would suffice) causes unnecessary utility loss on that variable. Under-binning means more retry iterations.

**Current confidence:** Medium-high. The multipliers are conservative — the floor at max(5, ...) prevents extreme under-binning.

---

## Summary: What to Test First

| Priority | Threshold | Test |
|---|---|---|
| 1 | RC cutpoints (40/60/3+) | LOCSUPR vs kANON at varying concentration levels |
| 2 | CAT1 ratio (70%) | PRAM vs kANON at 60-75% categorical |
| 3 | LOW1 PRAM gate (10%) | PRAM vs kANON at 10-20% risk, categorical-dominant |
| 4 | Bimodal detection (mean-median > 0.15) | Synthetic distributions — does it misfire on skewed-unimodal? |
| 5 | Cross-method starting points | kANON k=5 ↔ PRAM p=0.15 equivalence check |
| 6 | QI fusion 30/70 weighting | PROBABLE keywords at low risk — are real QIs being missed? |
| 7 | Suppression gate (25%) | kANON at 20-30% suppression vs PRAM switch |

Items 1-3 are the highest impact because they determine which method is selected as primary. Items 4-7 are lower impact because the retry engine or user review compensates.

### Mitigation: Perturbative Challenge Rule

The three most dangerous thresholds (items 1-3) share the same failure mode: kANON is selected as primary, succeeds through suppression, and the engine never discovers that PRAM would have achieved the same target with better utility. A single post-success rule — the **Perturbative Challenge** — neutralises all three without changing any existing thresholds.

After a structural method (kANON/LOCSUPR) succeeds, if the data is ≥50% categorical, ReID ≤ 30%, and suppression > 3%, the engine tries PRAM once at a calibrated p. If PRAM also meets the target with > 3% utility improvement, it replaces the structural result. One extra method call, < 1 second on typical datasets.

This means the thresholds in items 1-3 become less critical — they still affect which method is tried first (faster convergence), but a wrong choice no longer locks in suboptimal utility. See `perturbative_challenge_spec.md` for full implementation details.

---

## Testing Approach

For each threshold:

1. **Assemble 5-10 datasets** spanning the boundary (e.g., concentration at 30%, 35%, 40%, 45%, 50%)
2. **Run both paths** — the method the rule selects vs the alternative it doesn't
3. **Compare** ReID reduction, utility score, suppression rate, number of retry iterations
4. **Find the crossover** — at what value does the alternative start outperforming?
5. **If crossover differs from threshold by > 5 percentage points** — adjust the threshold

The retry engine means a "wrong" threshold choice is self-correcting within 1-3 iterations. The goal of validation isn't to prove thresholds are perfect — it's to confirm they're close enough that the engine starts near the optimal point.

---

## Branch: `feature/enhanced-risk-views_MicroSDC` (R/sdcMicro Bridge)

This branch has no retry engine — the user manually adjusts and re-runs. Wrong thresholds cost the user an extra manual iteration, not just compute time.

### M1: Classification (`classify_suggest.py`)

| Threshold | Value | What it gates | Confidence | Concern |
|---|---|---|---|---|
| Identifier uniqueness | > 95% | Column flagged as Identifier | High | Well-established. Could miss some IDs at 90-95% uniqueness (e.g., address strings with shared apartment numbers) |
| Priority HIGH | ≥ 1.5× mean contribution | QI / High confidence | Medium | Why 1.5× and not 1.2× or 2.0×? Affects which columns get auto-classified as QI vs require user review |
| Priority MED-HIGH | ≥ 0.8× mean | QI / Medium confidence | Medium | Same question — the mean is dataset-dependent, so the threshold's meaning shifts with dataset composition |
| Priority MODERATE | ≥ 0.3× mean | QI or Sensitive | Medium-low | At 0.3× mean, a column contributing very little risk can still be classified as QI. May over-classify |
| Rare category | < 1% frequency | Merge rare trigger | High | Standard practice. Some domains need 2% or 5% — but 1% is conservative and safe |

**What to test:** The mean-relative thresholds (1.5×, 0.8×, 0.3×) on datasets with varying numbers of columns. With 5 columns, mean contribution is ~20%, so HIGH fires at 30%. With 15 columns, mean is ~7%, so HIGH fires at 10.5%. Is a column at 11% contribution on a 15-column dataset really "HIGH priority"? The threshold's meaning is not stable across dataset sizes.

**Alternative:** Fixed percentage thresholds (e.g., HIGH ≥ 15%, MED-HIGH ≥ 8%, MODERATE ≥ 3%) — same as the auto_sdc_v2_manly branch uses. These are stable regardless of column count.

---

### M2: Preprocessing Plan (`protect.py → _generate_plan`)

| Threshold | Value | What it gates | Confidence | Concern |
|---|---|---|---|---|
| Numeric needs binning | unique > 50 | Bin action | High | 50 is reasonable — above this, equivalence classes are too small |
| High skew | \|skew\| > 5 | Log-scale bins | Medium-high | Extreme skew — log-scale is the right call. But the 5 vs 3 boundary hasn't been tested |
| Moderate skew | \|skew\| > 2 | Quantile bins + top/bottom | Medium-high | Standard statistical threshold for "substantially skewed" |
| Date high cardinality | unique > 100 | Generalize to month | Medium | 100 unique dates → monthly. But what about 80 unique dates spanning 2 years? Still high risk. Could be 50 |
| Date moderate | unique > 10 | Generalize to quarter | High | 10 unique values is a reasonable "needs some reduction" signal |
| Categorical high card | unique > 100 | Merge rare | High | Above 100 categories, merging is clearly needed |
| Geo hierarchy | ≥ 3× reduction + ≥ 95% clean mapping | Coarsen to parent | High | Both conditions are strict — this avoids false hierarchy detection |
| Child min cardinality | unique < 20 | Skip hierarchy detection | Medium | If child has < 20 values, hierarchy detection is pointless. But 20 is arbitrary — could be 15 or 25 |
| Missing warning | > 10% null | Show warning | High | Standard data quality threshold |
| Near-constant | unique ≤ 2 | Show warning | High | Binary or constant — clearly low utility |

**Highest-impact to test:** Date thresholds (100 → month, 10 → quarter). A dataset with 60 unique dates spanning 5 years has high re-identification power but wouldn't trigger monthly generalisation. Run datasets with 30, 50, 80, 120 unique dates and compare k-anonymity outcomes with and without date generalisation.

**Skewness boundary (5 vs 2):** Run numeric QIs with skewness 2, 3, 4, 5, 7. Compare log-scale vs quantile binning outcomes. The 5→log boundary matters because log-scale bins are more aggressive — if quantile bins suffice at skew=5, the threshold should be higher.

---

### M3: Risk Pattern (`_classify_risk_pattern`)

| Threshold | Value | Pattern | Confidence | Concern |
|---|---|---|---|---|
| Tail detection | group violation < 20% + record rate > 5% | Tail | Medium | Two-condition gate. The 20% group / 5% record thresholds are sensible but untested at boundaries |
| Uniform high | group violation > 50% | Uniform high | High | More than half of groups violate — clearly uniform. Solid threshold |

**What to test:** Datasets near the tail boundary — 18% group violation + 6% record rate vs 22% + 4%. Do they need different methods? If both respond well to the same method, the boundary doesn't matter much.

**Current confidence:** Higher than the auto_sdc branch's `classify_risk_pattern()` because this version uses simpler, more direct conditions (group violations vs percentile ratios).

---

### M4: Method Recommendation (`_recommend_method`)

| Threshold | Value | Decision | Confidence | Concern |
|---|---|---|---|---|
| Small dataset | < 200 rows | Perturbative preferred | High | Same rationale as auto branch HR6 — structural methods are destructive at this size |
| High violations | > 30% | Generalization | Medium-high | Why 30% and not 25% or 40%? This determines whether the user sees "generalization" or "local suppression" as the default |
| Moderate violations | > 10% | Local suppression | Medium-high | Below 10%, the recommendation drops to NOISE/PRAM. Is 10% the right point where structural methods become necessary? |
| High risk QI | > 30% contribution | Importance-weighted note | High | Informational only — doesn't change method. Low risk if wrong |
| All categorical | cat_ratio == 1.0 | PRAM | High | Exact 1.0 — no partial threshold. If one numeric QI exists, PRAM is not recommended. Could arguably be ≥ 0.90 |
| k already met | k_met == True | Preprocessing only | High | Correct — no method needed |

**Highest-impact to test:** The 30% violation threshold for generalization vs local suppression. Run datasets with 25%, 30%, 35% violation rates. Compare generalization outcomes vs local suppression. If local suppression handles 35% violations well, the threshold could be higher (reducing unnecessary generalization).

**The all-categorical strict equality:** A dataset with 8 categorical QIs and 1 numeric QI (cat_ratio = 0.89) falls out of the PRAM recommendation. If the numeric QI is low-risk, PRAM might still be the best choice. Consider relaxing to ≥ 0.85 or ≥ 0.90.

---

### M5: Adaptive Scaling (`_compute_adaptive_params`)

| Threshold | Value | What it does | Confidence | Concern |
|---|---|---|---|---|
| Bin count floor | 5 | Minimum bins for any QI | High | Below 5 bins, the variable loses most analytical value |
| Bin count cap | 20 | Maximum bins | Medium-high | 20 is generous — might be too many for small datasets. But the risk-weighted adjustment halves this for HIGH QIs |
| Risk high | > 50% contribution | Halve bins (floor 5) | Medium | 50% is very concentrated — halving is aggressive. Would 40% be a better trigger? |
| Risk low | < 10% contribution | 1.5× bins (cap 30) | High | Low-risk QIs get more bins — preserves utility. Conservative and safe |
| Merge percentage | clamp(1/budget, 0.5%, 5%) | Merge rare threshold | Medium-high | Budget-derived — scales with dataset. The 0.5% floor and 5% cap are sensible |

**What to test:** The 50% risk threshold for bin halving. With risk-weighted limits, a QI contributing 45% gets normal bins while one at 55% gets halved. Is this discontinuity justified? Run datasets where the top QI contributes 40-60% and compare utility with and without bin halving.

---

### M6: Auto-Retry (`_auto_retry`)

| Threshold | Value | What it does | Confidence | Concern |
|---|---|---|---|---|
| Bin tightening | halve (floor 3) | More aggressive grouping | Medium-high | Halving is a strong adjustment. Could over-correct if first attempt was close |
| Merge tightening | 2× frequency | Absorb more rare categories | High | Doubling merge threshold is standard escalation |
| Activate skipped cats | unique > 20 | Turn on merge_rare | Medium | Why 20 and not 15 or 30? This determines which skipped categorical QIs get retroactively processed |
| Perturbative bump | +0.05 | Slightly stronger p_change/magnitude | High | Small increment — safe escalation step |

**What to test:** The bin halving vs a gentler reduction (×0.7). Halving may over-shoot on the retry, producing unnecessarily coarse bins when a 30% reduction would have sufficed. Run the retry on datasets where first attempt missed by 5%, 10%, 20% — does halving always improve the outcome, or does it sometimes over-correct?

---

### M7: Release Context Thresholds

| Context | k | Source | Confidence |
|---|---|---|---|
| Public | 10 | Hundepool et al. 2012 | High — literature standard |
| Research (SUF) | 5 | SUF standard practice | High — widely used |
| Internal | 3 | Minimum meaningful k | Medium-high — k=3 provides limited protection; some argue k=5 should be the floor |

These are the only thresholds grounded in published SDC literature rather than empirical tuning. k=5 for research and k=10 for public release are standard across the field (Hundepool et al. 2012, Eurostat guidelines). k=3 for internal is the only debatable one — it provides minimal grouping that could be broken with auxiliary information.

---

## MicroSDC: Summary — What to Test First

| Priority | Threshold | Test |
|---|---|---|
| 1 | Violation rate 30% for generalization | Compare generalization vs local suppression at 25-40% violations |
| 2 | Mean-relative classification (1.5×/0.8×/0.3×) | Stability across datasets with 5 vs 15 columns |
| 3 | Date cardinality 100 → month | Datasets with 50-120 unique dates |
| 4 | Skewness boundary 5 vs 2 | Log-scale vs quantile bins at skew 2-7 |
| 5 | All-categorical strict equality (1.0) | PRAM effectiveness at cat_ratio 0.85-1.0 |
| 6 | Risk-weighted bin halving at 50% | Utility comparison at 40-60% contribution |
| 7 | Auto-retry bin halving vs gentler reduction | Over-correction check on near-miss datasets |

---

## P1 (continued): QR Risk Pattern Rules

**File:** `selection/rules.py` — `reid_risk_rules()`

| Rule | Condition | Method | Confidence |
|---|---|---|---|
| QR0 | k-anonymity infeasible (EQ < 3) | GENERALIZE_FIRST | High — math-based |
| QR1 | Severe tail: reid_50 < 5%, reid_99 ≥ 50% | LOCSUPR k=5 | Medium-high — 5%/50% split is reasonable but boundary untested |
| QR2 heavy | Tail + reid_95 > 40% | kANON k=7 or LOCSUPR k=5 (suppression-gated at 25%) | High |
| QR2 moderate | Tail + 30% ≤ reid_95 ≤ 40% | LOCSUPR k=3 | Medium-high |
| QR3 | Uniform high: reid_50 ≥ 20% | kANON k=10 (suppression-gated) | High |
| QR4 high | Widespread: reid_95 > 50% | kANON k=10 | High |
| QR4 moderate | Widespread: reid_50 > 15%, reid_95 30-50% | kANON k=7 | Medium-high |
| MED1 | Moderate: reid_95 > 20%, reid_50 < 10% OR bimodal OR high_risk_rate > 10% | kANON k=5 (suppression-gated) | Medium |

**What to test:** QR1 boundary — datasets with reid_50=4% vs 6%, reid_99=48% vs 52%. Does the severe_tail classification change outcomes meaningfully?

---

## P1 (continued): Heuristic/Distribution/Default Rules

**File:** `selection/rules.py`

| Rule | Condition | Method | Confidence |
|---|---|---|---|
| HR1 | Uniqueness > 20% | LOCSUPR k=5 | High |
| HR2 | Uniqueness 10-20% | kANON k=7 | Medium-high |
| HR3 | Uniqueness > 5% + ≥2 QIs | kANON k=5 | Medium-high |
| HR4 | < 100 rows | PRAM p=0.30 | High — very small datasets |
| HR5 | 100-500 rows + uniqueness > 3% | NOISE mag=0.15 or PRAM p=0.25 | Medium |
| DP1 | Outliers present + continuous > 0 | NOISE mag=0.20 | High |
| DP2 | ≥2 skewed columns | PRAM p=0.20 | Medium |
| DP4 | Integer-coded categoricals | PRAM p=0.20-0.30 | Medium |
| DATE1 | ≥50% temporal QIs + reid ≤ 40% | PRAM on binned dates | Medium-low — rarely fires |
| LDIV1 | Sensitive diversity ≤ 5 | Advisory: add PRAM on sensitive | Medium |
| DEFAULT | Catch-all | kANON k=5 / PRAM p=0.20 / NOISE mag=0.15 | High — safe fallbacks |

---

## P1 (continued): Pipeline Rules

**File:** `selection/pipelines.py`

| Pipeline | Trigger | Methods | Confidence |
|---|---|---|---|
| DYN_CAT | 50-70% categorical + ≥1 continuous + reid > 15% | NOISE → PRAM | Medium-high — the 15% reid gate untested |
| GEO1 | ≥2 geo QIs (fine + coarse) | GENERALIZE → kANON k=5 | High — geo detection is strict |
| DYN | reid > 20% + mixed types + outliers | kANON/NOISE/LOCSUPR | Medium — complex multi-condition |
| P4 | ≥2 skewed + sensitive | kANON ± PRAM | Medium — skew threshold ≥2 |
| P5 | density < 5 + mixed + uniqueness > 15% | NOISE → PRAM | Medium-low — density<5 is rare |

**Pipeline p_change/magnitude values:**

| Context | Value | Confidence |
|---|---|---|
| DYN_CAT PRAM p | 0.25-0.30 (scales with reid) | Medium |
| DYN_CAT NOISE mag | 0.15-0.20 | Medium |
| P5 NOISE mag | Scales with uniqueness | Medium-low |
| P5 PRAM p | 0.30 | Medium |

---

## P2 (continued): Treatment Balance

**File:** `selection/rules.py` — `_apply_treatment_balance()`

| Threshold | Value | Effect | Confidence |
|---|---|---|---|
| Heavy trigger | ≥60% of QIs set Heavy | k+2, p+0.05, mag+0.05 | Medium-high |
| Light trigger | ≥60% of QIs set Light | k-1, p-0.05, mag-0.03 | Medium-high |

---

## P2 (continued): Escalation Schedules

**File:** `sdc/config.py` — `get_tuning_schedule()`

| Method | Schedule | Confidence |
|---|---|---|
| kANON | [3, 5, 7, 10, 15, 20, 25, 30] | High — standard k progression |
| PRAM | [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50] | Medium-high — 0.50 max is aggressive |
| NOISE | [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] | Medium-high — same concern at 0.50 |
| LOCSUPR | [3, 5, 7, 10, 15, 20] | High |

**What to test:** Are the top-end values (k=30, p=0.50, mag=0.50) ever reached? If so, do they produce usable data? Likely too destructive — confirm with real datasets.

---

## P2 (continued): Protection Context Targets

**File:** `sdc/config.py` — `PROTECTION_THRESHOLDS`

| Context | reid_95_max | reid_99_max | k_min | utility_floor | l_min | Confidence |
|---|---|---|---|---|---|---|
| Public | 0.01 | 0.03 | 10 | 85% | 3 | High — Eurostat |
| Scientific | 0.05 | 0.10 | 5 | 90% | 2 | High — SUF standard |
| Internal | 0.10 | 0.20 | 3 | 92% | 2 | Medium-high |
| Regulatory | 0.03 | 0.05 | 5 | 88% | 3 | Medium-high |
| Default | 0.05 | 0.10 | 5 | 88% | 2 | High |

---

## P2 (continued): GENERALIZE Tier Configuration

**File:** `sdc/config.py` — `GENERALIZE_TIERS`

| Tier | max_categories | Confidence |
|---|---|---|
| Light | 15 | Medium-high |
| Moderate | 10 | High — standard |
| Aggressive | 5 | High |
| Very Aggressive | 3 | Medium — may over-generalise |

---

## P2 (continued): Protection Engine Controls

**File:** `sdc/protection_engine.py`

| Threshold | Value | What it gates | Confidence |
|---|---|---|---|
| Mid-pipeline risk multiplier (structural) | 1.10 | Early exit if reid < target × 1.10 | High — fires ~25% of pipeline mid-checks (Spec 15 validated) |
| Mid-pipeline risk multiplier (other) | 1.20 | Early exit if reid < target × 1.20 | Medium |
| Escalation time budget | ~30s per phase | Stops escalation if phase exceeds | High — prevents runaway |
| Max fallbacks | 5 | Cap on fallback attempts | High |
| Plateau detection | reid improvement < 0.001 | Skip to next method if stalled | Medium-high — often indicates structural floor (see [F4 investigation](investigations/spec_12_f4_reid_floor.md)) |
| Perturbative filter | best_reid > target | Strip PRAM/NOISE from fallbacks when structural needed | High |
| k step-down gap | achieved_k > target_k + 2 | Trigger step-down check | Medium-high |
| k step-down suppression | > 2% | Only step down if there's suppression to save | High |
| Perturbative challenge cat_ratio | ≥ 0.50 | Only challenge if enough categoricals for PRAM | High |
| Perturbative challenge reid | ≤ 0.30 | Don't challenge at high risk | High |
| Perturbative challenge suppression | > 3% | Only challenge if primary caused meaningful suppression | High |
| Perturbative challenge utility gain | > 3% | Accept PRAM only if meaningfully better | High |
| k step-down utility gain | > 2% | Accept lower k only if meaningfully better | High |

---

## P2 (continued): Smart Method Config

**File:** `sdc/smart_method_config.py`

| Threshold | Value | What it gates | Confidence |
|---|---|---|---|
| kANON strategy: suppression-dominant | < 10% violations | Suppress outliers only | High |
| kANON strategy: hybrid | 10-40% violations | Generalise + suppress | High |
| kANON strategy: generalisation-dominant | > 40% violations | Generalise everything | High |
| kANON → PRAM switch | est. suppression > 25% | Pre-application method swap | High |
| LOCSUPR → kANON switch | est. cell suppression > 10% | Pre-application method swap | Medium-high |
| PRAM → kANON switch | >50% QIs LOW effectiveness | Pre-application method swap | Medium |
| PRAM dominance threshold | >80% single category | Flag as LOW effectiveness | High |
| NOISE correlation warning | r > 0.70 | Warn about correlation destruction | Medium-high |
| NOISE distribution risk: HIGH | noise_std/col_std > 50% | Flag as high distortion | Medium |
| NOISE distribution risk: MODERATE | noise_std/col_std > 30% | Flag as moderate | Medium |

---

## P3 (continued): Auto-Classify Details

**File:** `sdc/auto_classify.py`

| Threshold | Value | What it gates | Confidence |
|---|---|---|---|
| Near-constant | >95% one value | → Unassigned | High |
| Zero-variance | std/(abs(mean)+ε) < 0.01 | → Unassigned | High |
| High missingness (strong) | >70% null | → Unassigned | Medium-high |
| High missingness (moderate) | 30-70% null | Demote confidence one tier | Medium |
| Continuous + low risk ramp | rc=0%→+0.40, rc=10%→+0.25, rc≥15%→0 | Sensitive score boost | Medium — smooth ramp, well-designed |
| High entropy | >4 bits | +0.25 toward Sensitive | Medium |
| Skewness signal | \|skew\| > 2 | +0.15 toward Sensitive | High |
| Moderate cardinality numeric | 20-500 unique + <5% risk | +0.20 toward Sensitive | Medium |
| Binary/few category | ≤5 categories + <3% risk | +0.15 to +0.20 toward Sensitive | Medium |
| Ratio/percentage | 0-1 or 0-100 range | +0.25 toward Sensitive | Medium-high |
| High-cardinality non-numeric penalty | >50% unique | -0.30 from Sensitive | High |
| Admin keyword penalty | code, type, κωδικός | -0.15 from Sensitive | High |
| High risk contribution penalty | >10% + non-continuous | -0.20 from Sensitive | Medium-high |
| Low-cardinality numeric penalty | <20 unique + not count | -0.15 from Sensitive | Medium |

---

## P3 (continued): Method-Level Thresholds

### NOISE (`sdc/NOISE.py`)

| Threshold | Value | What it gates | Confidence |
|---|---|---|---|
| Per-value cap | 25% of original value | Prevents small values from being destroyed | High — well-reasoned |
| Distributional correction | Proportional scaling on capped values | Restores pre-cap mean | High |
| preserve_sign guard | original ≥ 0 | Prevents sign flipping | High |

### GENERALIZE (`sdc/GENERALIZE.py`)

| Threshold | Value | What it gates | Confidence |
|---|---|---|---|
| Quantile binning trigger | \|skew\| > 3.0 | Routes to pd.qcut instead of pd.cut | Medium-high |
| Value range guard | range / bin_size < 2 | Adjusts bin_size to prevent <2 bins | High |
| Constant column guard | nunique ≤ 1 | Skip generalisation | High |

### kANON (`sdc/kANON.py`)

| Threshold | Value | What it gates | Confidence |
|---|---|---|---|
| Max suppression rate | Configurable, typically 30% | Stops if too many records suppressed | High |
| Beam search width | Configurable | Number of generalisation paths explored | Medium |
| l-diversity enforcement | min_l per equivalence class | Post-generalisation check | High |
| t-closeness enforcement | EMD/TVD ≤ threshold | Distribution distance check | High |

---

## Impact Assessment: Newly Added Thresholds

Of the ~90 thresholds added above, **8 are higher impact than P3**:

| Threshold | Value | Risk | Mitigated by |
|---|---|---|---|
| QR2 heavy tail gate | reid_95 > 40% | kANON k=7 vs LOCSUPR k=5 — wrong method | ✅ Fallback chain (kANON → LOCSUPR), 1 wasted iteration |
| QR4 moderate | reid_50 > 15% + reid_95 30-50% | k=7 vs k=10 — big suppression diff | ✅ **k step-down** catches this post-success |
| DYN_CAT reid gate | reid > 15% | Pipeline fires or doesn't — multi vs single method | ✅ Pipeline failure falls to single-method rules |
| HR4/HR5 small dataset | <100 / 100-500 rows | Wrong method on tiny data | ✅ HR6 catches <200, HR4 <100 |
| kANON strategy | 10%/40% violation split | Wrong internal strategy | ✅ Retry escalates through strategies |
| PRAM dominance | >80% single category | PRAM useless on dominant cats | ✅ Smart config switches to kANON pre-application |
| Mid-pipeline risk multiplier | 1.10 | Premature early exit | ✅ Validated — fires 25% of pipeline mid-checks; docs updated to match code |
| Escalation top-end | k=30 / p=0.50 | Data destruction if reached | ✅ Utility floor gate rejects |

**Conclusion:** The retry engine + fallback chains + k step-down + perturbative challenge together cover all 8. No new mitigation needed. The mid-pipeline multiplier drift (previously flagged as 1.10 vs 1.5) is resolved — Spec 15 investigation confirmed 1.10 fires in ~25% of pipeline mid-checks, placing it in the "defensible" range (5–25%).

The remaining ~82 additions are genuinely P3 (low impact) — either well-established values, user-reviewable classifications, or downstream-compensated parameter choices.

## Logging Note

The Python-engine branches have extensive existing logging (`log.info("[Protection] ...")`, `log.info("[Rule] ...")`) that captures rule firing, method selection, escalation steps, and fallback attempts. The structured `[Threshold]` format (as added to MicroSDC's `protect.py`) is not present — threshold decisions must be reconstructed from the rule/method log trail. This is sufficient for debugging but not for systematic threshold analysis. Adding structured `[Threshold]` logging across all 130 thresholds is a mechanical task (~200 edits) deferred for future work.

---

## Cross-Branch: Shared Concerns

Two thresholds appear in both branches with the same value:

| Threshold | Value | Both branches | Concern |
|---|---|---|---|
| Small dataset | < 200 rows | HR6 (auto) / perturbative-preferred (MicroSDC) | Consistent. Well-grounded in equivalence class math |
| Identifier uniqueness | > 95% | auto_classify (auto) / classify_suggest (MicroSDC) | Consistent. Standard practice |

One threshold differs between branches and should be reconciled:

| Threshold | auto_sdc_v2_manly | MicroSDC | Which is better? |
|---|---|---|---|
| HIGH QI classification | Fixed ≥ 15% contribution | Relative ≥ 1.5× mean | Fixed is more stable across dataset sizes. MicroSDC should consider adopting fixed thresholds |

---

## Related Test Suites

### Known-Case Regression Tests (`tests/test_rule_selection_known_cases.py`)

42 tests across 12 test classes verifying that each rule fires as designed on synthetic data. Each test constructs a minimal dataset targeting exactly one rule (or a specific guard condition). Covers QR1, QR2, QR4, MED1, RC1 (both injected and organic), CAT1, CAT2 via DYN_CAT_Pipeline, LOW1, LOW2, LOW3, SR3, HR6, HR1/HR3 (injected), PUB1/SEC1/REG1 (context-aware), rule priority ordering, dominance guard, and metric-filter edge cases.

Key findings during builder construction (2026-04-20):
- **RC1 now fires organically** for small-to-medium datasets (RC2/RC3/RC4 deleted in Spec 19 Phase 2 — structurally unreachable). Spec 07 added lazy `var_priority` computation to `build_data_features()` — for datasets up to 10,000 rows with ≤8 QIs, per-QI risk contribution is computed via leave-one-out reid_95 and the resulting `var_priority` populates `features['var_priority']` and `features['risk_concentration']`. For larger datasets, the performance guard skips the computation and RC1 remains dormant — the engine then falls through to QR/LOW rules.
- **HR1-HR5 remain dormant** — they depend on `uniqueness_rate` which is not populated in the feature pipeline. Tests for HR1-HR5 inject the feature manually via feature-injection (see `TestUniquenessRiskRules` in the known-case suite).
- **DYN_CAT_Pipeline preempts CAT2.** The pipeline_rules check fires before rule_factories, so `CAT2_Mixed_Categorical_Majority` is unreachable when `DYN_CAT_Pipeline` has the same condition.
- **QR0 (GENERALIZE_FIRST) ~~is skipped under reid95~~.** Pre-Fix 0 data — `GENERALIZE_FIRST` was missing from `METRIC_ALLOWED_METHODS` for all metrics, so QR0 was silently config-blocked. Fixed 2026-04-20: GENERALIZE and GENERALIZE_FIRST added to all 4 metric lists. See `tests/empirical/fixtures/README.md` Change History.

Run: `python -m pytest tests/test_rule_selection_known_cases.py -v` (~1.5s, no R/sdcMicro required).

### Empirical Threshold Validation (`tests/empirical/`)

80-run harness testing 4 thresholds across 8 real datasets. See `tests/empirical/reports/SUMMARY.md` for full results and crossover analysis.

**Cross-Metric Validation (`tests/empirical/reports/`):** Two validation runs documented — `reid95/` (80 runs, 3 crossovers documented at rule level) and `k_anonymity_latest/` (80 runs, 15 passing, findings documented in SUMMARY.md). Cross-metric comparison in `COMBINED_SUMMARY.md` addresses whether rule-level thresholds produce different outcomes under different risk metrics.

### Regression Test Additions (Hunts 1–4, 2026-04)

153 additional regression tests covering dtype fuzz (int32/float32/object on each method), degenerate inputs (single-row, single-QI, all-null), and cross-metric matrix (every config/method path tested under reid95/k_anonymity/uniqueness). One latent bug surfaced (NOISE single-row NaN std bypass — fixed). Files: `tests/test_dtype_fuzz.py`, `tests/test_degenerate_inputs.py`, `tests/test_cross_metric.py`.

**Total test count (as of 2026-04-20): 246 tests** across all test files (42 known-case + 28 dtype fuzz + 28 degenerate + 97 cross-metric + 10 harness + 8 parity + 33 other).
