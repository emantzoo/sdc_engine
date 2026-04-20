# Family 1 — Risk Concentration Sensitivity Analysis

**Date:** 2026-04-20
**Datasets:** 27 synthetic (5000 rows, 4 QIs, reid_95=0.50) + 3 real
**Runs:** 135 synthetic (27 datasets x 5 counterfactuals) + 9 real-dataset verifications
**Status:** Complete. No threshold adjustments proposed.

## 1. Scope and methodology

### What this study tests

Family 1 tests the **routing layer**: given a dataset classified as pattern X
by `classify_risk_concentration()`, does the rules engine correctly route it
to the expected protection method?

- `dominated` (top_pct >= 40%) -> RC1 -> LOCSUPR k=5
- `concentrated` (top2_pct >= 60%) -> RC2 -> kANON hybrid
- `spread_high` (n_high >= 3) -> RC3 -> kANON k=7/10
- `balanced` (else) -> fall-through to QR/CAT/LOW rules

### What this study does NOT test

This study does **not** test whether the specific 40%/60% cutoffs are
well-calibrated. The four classify_risk_concentration rules interact:
below the 40% dominated cutoff, whether a dataset classifies as 'balanced'
or 'spread_high' depends on the n_high >= 3 rule. The 40% threshold
cannot be swept in isolation without controlling or acknowledging the
others. See "Threshold interaction finding" below.

### Approach

27 synthetic datasets with controlled var_priority dicts that produce
specific top_pct and top2_pct values. Each dataset is run through
5 counterfactual paths (natural, dominated, concentrated, spread_high,
balanced) by overriding the pattern string in the features dict.
RC rules check pattern as a string match (rules.py lines 378, 400, 426),
not by re-evaluating underlying conditions.

RC4 is not tested — it's gated by n_high_risk, not top_pct/top2_pct,
and its GENERALIZE->kANON pipeline is architecturally distinct.

## 2. Threshold interaction finding

During builder development, datasets with top_pct=30% (below the 40%
dominated cutoff) were classified as `spread_high` instead of `balanced`
when contributions were distributed organically. Root cause: with
top_pct=30%, the remaining ~70% spread across 3 secondary QIs gave each
~23% contribution, all labeled HIGH (>= 15%), triggering n_high >= 3.

**This is informative, not a bug.** It means:

1. The "balanced" classification requires contributions below 15% per QI
   (the HIGH threshold). A dataset with 4 QIs where each contributes
   15-25% is classified as spread_high, not balanced.

2. The 40% dominated threshold cannot be evaluated independently. Below
   40%, the classification depends on whether spread_high or concentrated
   fires first. Above 40%, dominated takes priority regardless.

**Implication:** Any future threshold calibration study must jointly
consider the dominated (40%), concentrated (60%), and spread_high
(n_high >= 3) rules as a coupled system.

## 3. Findings

### Finding 1: Routing is correct

Each pattern routes to its expected rule/method. The natural path shifts
behavior exactly at the boundary:

- top_pct < 40%: balanced -> QR4 -> reid=0.088, util=0.49
- top_pct >= 40%: dominated -> RC1 -> reid=0.167, util=0.66
- top2_pct < 60%: balanced -> QR4 -> reid=0.088, util=0.49
- top2_pct >= 60%: concentrated -> RC2 -> reid=0.167, util=0.66

Seed stability is excellent: reid_std < 0.005 across all groups,
signal/noise ratio ~18:1.

### Finding 2: RC rules and QR fallthrough are Pareto-incomparable (on synthetic data)

On synthetic uniform-random datasets, the two routing paths produce
qualitatively different tradeoffs:

| Cluster | Paths | reid_after | utility | reid reduction |
|---------|-------|-----------|---------|----------------|
| A | balanced, spread_high | 0.088 | 0.49 | 82% |
| B | dominated, concentrated | 0.167 | 0.66 | 67% |

Neither dominates the other. RC rules (Cluster B) prioritize utility
preservation; QR fallthrough (Cluster A) prioritizes reid reduction.
The right choice depends on the user's privacy/utility preference.

**Why?** RC1 selects LOCSUPR k=5, which over-suppresses on these
datasets, then falls back to kANON with a moderate k. QR4 (the balanced
fallthrough) selects kANON with a higher k, and the more aggressive
generalization achieves lower reid at the cost of more utility loss.

**Scope limitation:** This tradeoff is observed on synthetic datasets
with uniform-random data and no QI correlation structure. Real datasets
with correlated QIs could show different behavior — targeted suppression
(LOCSUPR) may outperform generic kANON when correlation allows it to
suppress fewer records.

### Finding 3: RC rules have very narrow activation scope in production

**Real dataset verification** (3 datasets, 3 paths each: natural, dominated, balanced):

| Dataset | Rows | QIs | reid_95 | var_priority | pattern | Outcome |
|---------|------|-----|---------|-------------|---------|---------|
| adult_mid | 30162 | 4 | 0.25 | **empty** | unknown | All paths identical (kANON, 0.039, 0.993) |
| testdata | 4580 | 7 | 0.50 | populated | dominated | All paths converge (LOCSUPR, 0.044, 1.000) |
| greek_mid | 41742 | 4 | 0.33 | **empty** | unknown | All paths identical (kANON, 0.037, 0.968) |

**Root cause:** `_compute_var_priority()` has a hard threshold at
`VAR_PRIORITY_COMPUTATION['max_n_records'] = 10,000` (config.py line 1081).
Datasets exceeding 10K rows skip backward elimination entirely, returning
`None` (masked as `{}` by the `or {}` fallback in build_data_features).
With empty var_priority, RC rules never fire (pattern=unknown, applies=False).

On the one dataset where var_priority IS populated organically (testdata,
4580 rows), all paths converge to the same outcome regardless of the
counterfactual override — the retry engine normalizes the starting-point
differences.

**Implication:** RC rules were unreachable on 2 of 3 real datasets
tested, both exceeding 10K rows, due to the `max_n_records=10,000`
guard. This suggests RC rules may be unreachable on most production-scale
datasets, though a wider survey would be needed to confirm. On the one
small dataset where RC rules do fire (testdata, 4580 rows), the retry
engine absorbs routing differences — all paths converge to the same
outcome. The Pareto tradeoff observed in synthetic data (Finding 2)
does not appear on real data.

### Finding 4: Mathematical floor on reid reduction

- reid_before = 0.50 (all synthetic datasets)
- Best path: reid_after = 0.083 (83% reduction)
- Target: 0.05 (requires 90% reduction)
- 0/135 target_met

The math: 5000 rows / 1250 QI combinations = 4 average equivalence class
size. For reid_95 <= 0.05, need 95th-percentile eq class >= 20 records.
Generalization-only methods cannot bridge this gap without massive
suppression. Same architectural limit as CASCrefmicrodata/free1.

## 4. Crossover analysis

10 crossovers detected — all are balanced/spread_high outperforming
the natural path on datasets where natural is dominated/concentrated:

| boundary | target | counterfactual | wins/seeds | reid improvement |
|----------|--------|----------------|------------|-----------------|
| top_pct | 40% | balanced | 3/3 | +0.078 |
| top_pct | 40% | spread_high | 3/3 | +0.078 |
| top_pct | 45% | balanced | 3/3 | +0.078 |
| top_pct | 45% | spread_high | 3/3 | +0.078 |
| top_pct | 50% | balanced | 3/3 | +0.078 |
| top_pct | 50% | spread_high | 3/3 | +0.078 |
| top2_pct | 60% | balanced | 3/3 | +0.078 |
| top2_pct | 60% | spread_high | 3/3 | +0.078 |
| top2_pct | 65% | balanced | 3/3 | +0.078 |
| top2_pct | 65% | spread_high | 3/3 | +0.078 |

These crossovers reflect Finding 2 (Cluster A outperforms Cluster B on
reid at the cost of utility). **Step 4 cross-metric regression is NOT
warranted** because:

1. The crossover reflects Pareto-incomparability, not Pareto-dominance.
   Balanced wins on reid; RC rules win on utility. No threshold change
   resolves a Pareto tradeoff.
2. Finding 3 shows that RC rules rarely fire on real data anyway.
   Adjusting thresholds for a code path that's unreachable in production
   has no practical effect.

## 5. Summary

### What we confirmed

- Routing works correctly. Thresholds at 40% and 60% shift behavior
  exactly as designed.
- The routing layer is deterministic — reid_std < 0.005 across seeds.
- Threshold interaction: the dominated/concentrated/spread_high/balanced
  rules form a coupled system; no single threshold can be evaluated alone.

### What we discovered

- **RC rules were unreachable on 2 of 3 real datasets tested** (both
  exceeding 10K rows). The `max_n_records=10,000` guard in
  `_compute_var_priority()` prevents backward elimination on larger
  datasets, leaving var_priority empty and RC rules dormant. A wider
  survey would be needed to confirm this is universal.
- **On small datasets where RC rules DO fire, the retry engine absorbs
  routing differences.** testdata (4580 rows) produces the same outcome
  regardless of counterfactual.
- **On synthetic data where we force-inject var_priority, RC rules and
  QR fallthrough produce Pareto-incomparable tradeoffs** (reid vs utility),
  not Pareto-superior ones. But this synthetic finding doesn't generalize
  to real data.

### Recommendations

1. **No threshold changes.** The boundaries work as designed.
2. **The `max_n_records=10,000` threshold warrants investigation as a
   follow-up (potential Spec 14).** Before raising it, three questions
   need answers: (a) what is the actual performance cost of
   `_compute_var_priority` on larger datasets, (b) would sampled
   var_priority produce the same pattern classification as full
   computation, and (c) would activating RC rules on large datasets
   improve outcomes compared to current fallthrough behavior. Current
   engine outcomes on large real datasets are already excellent
   (adult_mid: reid=0.039, util=0.993), so the case for raising the
   threshold is not obvious.
3. **Future calibration studies** should jointly test all four
   concentration patterns and use real datasets with organic correlation
   structure. Given Findings 2 and 3, it is unclear whether the 40%/60%
   thresholds are empirically calibratable — RC rules rarely fire on
   real data, and where they do fire (synthetic or small real datasets),
   outcomes are Pareto-incomparable with fallthrough paths. Future work
   may conclude that threshold calibration is not the right frame for
   evaluating the rules engine's risk concentration tier.

### Numbers

- Synthetic: 135 runs, 0 errors, 0/135 target_met
- Real verification: 9 runs (3 datasets x 3 paths), 0 errors, 9/9 target_met
- Elapsed: synthetic ~0.9s/run median, real ~1-3s/run

### Files

- `family_1_results.csv` — raw synthetic sweep data (135 rows)
- `family_1_analysis_notes.md` — detailed analysis of 5 investigation questions
- `family_1_concentration.md` — this report
