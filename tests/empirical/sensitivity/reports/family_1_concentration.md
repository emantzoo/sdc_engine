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

### Finding 3: RC rules are unreachable on large datasets; other rules fire instead

**Real dataset verification** — full log capture showing which rule
actually fires on each dataset:

| Dataset | Rows | Rule fired | Primary | Outcome |
|---------|------|-----------|---------|---------|
| adult_mid | 30162 | **MED1_Moderate_Structural** | kANON | reid=0.039, target met on first try |
| testdata | 4580 | **RC1_Risk_Dominated** | LOCSUPR | reid=0.044, escalated k=5→20 |
| greek_mid | 41742 | **CAT1_Categorical_Dominant** | PRAM→kANON | reid=0.037, PRAM failed, kANON fallback met target |

Rules fire on all three datasets — the engine is never in "pure fallthrough."
The question is which *tier* of rules fires:

- **testdata (4580 rows):** var_priority is populated (below 10K threshold),
  RC1 fires organically, selects LOCSUPR, escalates to k=20, meets target.
  This is RC rules working as designed.

- **adult_mid (30162 rows):** var_priority empty (above 10K threshold),
  RC rules skipped, MED1 fires instead. MED1 selects kANON which meets
  target on first attempt with reid=0.039.

- **greek_mid (41742 rows):** var_priority empty, RC rules skipped,
  CAT1 fires (cat_ratio=1.00, reid_95=0.33). CAT1 selects PRAM as
  primary, **which makes reid worse** — see Finding 3a below. The retry
  engine falls back to kANON, which meets target.

**Root cause of RC rule absence on large datasets:** `_compute_var_priority()`
has a hard threshold at `max_n_records=10,000` (config.py line 1081).
Datasets exceeding 10K rows skip backward elimination, leaving
var_priority empty and RC rules dormant. Other rule tiers (MED, CAT)
fire instead — the engine is not falling through to a default path.

**Implication:** RC rules were unreachable on 2 of 3 real datasets
tested, both exceeding 10K rows. MED and CAT rules handled these
datasets with the help of fallback logic. The Pareto tradeoff between
RC and QR paths observed in synthetic data (Finding 2) is irrelevant
in production — RC rules don't compete with other tiers because they're
gated out before evaluation.

### Finding 3a: CAT1 method selection concern — PRAM increases reid

**3a.1 — The observation.**

On both real datasets where CAT1 is eligible (cat_ratio >= 0.70),
forcing the CAT1 path shows PRAM increases reid_95 rather than
decreasing it:

| | greek_mid | adult_mid |
|--|----------|----------|
| Rows | 41,742 | 30,162 |
| cat_ratio | 1.00 | 0.75 |
| Natural rule | CAT1 | MED1 (not CAT1) |
| Cells changed by PRAM | 22,951 (10-17%/col) | 14,071 (5-20%/col) |
| Eq classes before | 2,611 | 1,690 |
| Eq classes after | 6,727 (+2.6x) | 2,304 (+1.4x) |
| Unique records before | 736 | 543 |
| Unique records after | 3,643 (+5x) | 763 (+1.4x) |
| reid_95 before | 0.333 | 0.250 |
| reid_95 after | **1.000** | **0.333** |

Consistent direction on N=2: PRAM modified QI columns but reid_95
increased in both cases. greek_mid's damage is catastrophic
(0.33 → 1.0); adult_mid's is moderate (0.25 → 0.33). The smaller
magnitude on adult_mid may reflect that its continuous QI (age,
nunique=72) already creates high baseline fragmentation, limiting
PRAM's marginal impact — though this would need testing on
additional datasets to confirm.

**3a.2 — Proposed mechanism (hypothesis).**

PRAM perturbs categorical values by probabilistically reassigning
them to other categories. Because reid_95 measures the size of each
record's equivalence class (records with identical QI values),
creating new QI combinations means smaller classes, which means
higher reid_95. PRAM moves records into combinations that previously
had zero occurrences, fragmenting the grouping structure.

This failure mode may be specific to the reid_95 metric (which
penalizes fragmentation directly) or may reflect a broader mismatch
between PRAM and microdata. PRAM is designed to make individual cell
values unreliable, which reduces frequency-based disclosure risk.
But on record-level microdata, the perturbation creates new
equivalence classes rather than consolidating existing ones. Whether
this matters depends on how risk is measured.

**3a.3 — Engine protection (the self-correcting story).**

The rules engine's priority ordering and retry engine together mask
most of the user-visible damage from CAT1's method choice:

- **adult_mid:** MED1 fires before CAT1 in natural routing (priority
  ordering), so users get MED1/kANON (reid=0.039) rather than
  CAT1/PRAM (reid=0.333). The engine avoids the problem entirely.

- **greek_mid:** CAT1 fires naturally (no higher-priority rule
  matches). PRAM fails catastrophically (reid 1.0), but the retry
  engine detects the failure ("need structural method") and falls
  back to kANON (reid=0.037). Users get correct protection.

Users of the engine are protected in both cases. CAT1's method
choice is a latent issue — it costs one wasted PRAM attempt per run
on CAT1-eligible datasets — not an active failure that produces bad
output.

**3a.4 — Scope and limitations.**

N=2 with consistent direction. Both datasets are categorical-heavy
microdata. The finding may not generalize to: (a) other risk metrics
like k-anonymity, where PRAM's perturbation may not fragment classes
in the same way; (b) datasets with different cardinality
distributions; (c) other PRAM configurations (different p_change,
different prior matrices). Wider testing on 5+ CAT1-eligible
datasets would strengthen or scope the finding.

**This observation changes Family 2's scope.** Calibrating the 0.70
cat_ratio threshold for a rule whose primary method increases reid
does not produce useful results — lowering the threshold routes more
datasets to the problem, raising it reduces exposure but doesn't fix
the method choice. The productive next step is investigating whether
CAT1 should route to PRAM at all on microdata under reid_95, which
is a method-selection question rather than a threshold-calibration
question.

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
  datasets, leaving var_priority empty and RC rules dormant. Other rule
  tiers (MED1, CAT1) fire instead — the engine is not falling through
  to a default path. A wider survey would be needed to confirm this
  pattern is universal.
- **On testdata (4580 rows), RC1 fires organically and works as
  designed** — selects LOCSUPR, escalates to k=20, meets target with
  reid=0.044 and util=1.000.
- **PRAM increases reid on both CAT1-eligible real datasets tested**
  (Finding 3a). On greek_mid: reid 0.33 → 1.0, eq classes 2.6x more
  fragmented. On adult_mid: reid 0.25 → 0.33, eq classes 1.4x.
  Consistent direction, N=2. The engine's priority ordering (MED1
  fires before CAT1 on adult_mid) and retry engine (kANON fallback
  on greek_mid) mask the user-visible damage. CAT1's method choice
  is a latent issue, not an active failure. Wider testing needed
  before concluding systematic. This changes Family 2's scope from
  threshold calibration to method-selection investigation.
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
3. **CAT1's method selection warrants investigation before Family 2
   threshold calibration proceeds.** PRAM increased reid on both
   CAT1-eligible real datasets tested (Finding 3a). Calibrating the
   0.70 cat_ratio threshold while CAT1's primary method increases
   reid does not produce useful results — lowering the threshold
   routes more datasets to the problem, raising it reduces exposure
   without fixing the method choice. The productive next step is
   establishing whether CAT1 should route to PRAM at all on microdata
   under reid_95. This is a method-selection question, separate from
   threshold calibration, and should be scoped as its own
   investigation before Family 2 proceeds.
4. **Future calibration studies** should jointly test all four
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
