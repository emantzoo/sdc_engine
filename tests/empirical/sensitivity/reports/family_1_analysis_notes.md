# Family 1 Analysis Notes

**Date:** 2026-04-20
**Datasets:** 27 synthetic, reid_95=0.50, 5000 rows, 4 QIs
**Runs:** 135 (27 datasets x 5 counterfactuals), 0 errors

## 1. Seed stability — is the tradeoff pattern real?

**Yes.** The pattern is extremely stable across seeds.

- **reid_std**: max 0.0044 across all (counterfactual, target) groups. Effect size is 0.078 (balanced vs RC1). Signal/noise ratio ~18:1. Stable.
- **utility_std**: max 0.0014 for non-natural groups. Stable.
- **Exception**: natural path at top_pct=50 shows reid_std=0.043, utility_std=0.091 — but this is because one seed lands at kANON and another at a different method via organic var_priority recomputation. The injected counterfactuals don't have this variance.

**Conclusion:** Tradeoff pattern is real, not noise. The two outcome clusters (reid~0.088/util~0.49 and reid~0.167/util~0.66) are deterministic given the counterfactual.

## 2. Boundary comparison — does threshold location matter?

**Yes, with a caveat.**

The natural path outcome shifts cleanly at both boundaries:
- top_pct < 40%: balanced -> QR4 -> reid=0.088, util=0.49
- top_pct >= 40%: dominated -> RC1 -> reid=0.167, util=0.66
- top2_pct < 60%: balanced -> QR4 -> reid=0.088, util=0.49
- top2_pct >= 60%: concentrated -> RC2 -> reid=0.167, util=0.66

The threshold location IS being tested here — the natural path changes behavior exactly at the cutoff. But the **direction** of the change is counterintuitive: crossing INTO dominated territory produces WORSE reid (0.167 vs 0.088) and BETTER utility (0.66 vs 0.49).

**Why?** RC1 selects LOCSUPR k=5, which over-suppresses on these datasets, then falls back to kANON with a moderate k. QR4 (the balanced fallthrough) selects kANON with a higher k (or different strategy), and the more aggressive generalization achieves lower reid at the cost of more utility loss.

The threshold IS meaningful — it routes to different rules with different k parameters, producing measurably different outcomes. But the "specialized" RC rules don't outperform the "generic" QR fallthrough on these datasets. Whether that's a problem depends on the target: if you prioritize reid reduction, balanced/QR4 wins. If you prioritize utility preservation, dominated/RC1 wins. They're Pareto-incomparable.

## 3. Retry iteration counts

**n_iterations = 0 for all runs.** The metadata field is not being populated by the engine's internal retry loop — it tracks outer retry attempts, not kANON's internal generalization iterations.

**Elapsed time** as a proxy: median ~0.9s across all paths except spread_high (median 1.24s). Spread_high is ~40% slower — likely because RC3 requests k=7 (higher than RC1's k=5 or QR4's default), and the extra generalization depth costs time.

**Conclusion:** Routing doesn't meaningfully affect wall-clock time on these datasets. All paths complete in under 3s (excluding first-run module import overhead).

## 4. Utility/risk tradeoff — is it consistent?

**Perfectly consistent. Two Pareto-incomparable clusters.**

- **Cluster A** (balanced, spread_high): reid=0.088, utility=0.49, composite=5.57
- **Cluster B** (dominated, concentrated): reid=0.167, utility=0.66, composite=3.95
- **Natural** lands in A or B depending on whether it's above/below the threshold

No path dominates another. Balanced/spread_high achieve 82% reid reduction at 49% utility. Dominated/concentrated achieve 67% reid reduction at 66% utility. The right choice depends on the user's privacy/utility preference.

Composite score (utility/reid) slightly favors Cluster A (5.57 vs 3.95), but this is sensitivity-analysis-specific: the metric weighs reid reduction proportionally more than utility, which is arbitrary.

**Per-dataset dominance:** balanced/spread_high win best-composite on 15/27 datasets; natural wins on 12/27 (those where natural IS balanced). This isn't dataset-dependent variation — it's just reflecting which datasets land above/below the threshold.

## 5. The target_met=False story

**Genuine floor, not an engine failure.**

- reid_before = 0.50 (all datasets)
- reid_after: best = 0.083, worst = 0.167
- target = 0.05
- Best path achieves 83% reduction, but target requires 90% reduction

The math: 5000 rows / 1250 combinations = 4 average equivalence class size. For reid_95 <= 0.05, need 95th-percentile eq class >= 20 records. Even with aggressive generalization collapsing combinations, you can't get from 4-avg to 20-at-95th-percentile without massive suppression or category merging that the engine caps.

**This is the same architectural limit seen in CASCrefmicrodata/free1** — too many combinations relative to rows for generalization-only methods to achieve deep reid reduction.

## What it suggests

### The two findings that matter:

1. **Threshold boundaries work correctly.** The natural path shifts behavior exactly at 40% (dominated) and 60% (concentrated). The routing layer does what it was designed to do.

2. **RC rules are not Pareto-superior to QR fallthrough on these datasets.** This is the more interesting finding. RC1/RC2 were designed to be "smarter" routing for dominated/concentrated risk profiles, but they produce HIGHER reid_after and HIGHER utility than the generic QR4 path. This isn't necessarily wrong — it's a design choice: RC rules prioritize utility preservation, QR rules prioritize reid reduction. But it means the RC rules are NOT "better methods for hard datasets." They're "different methods with different tradeoff points."

### What this does NOT tell us:

- Whether RC rules outperform on real (non-synthetic) datasets. Our datasets are uniform-random with no correlation structure. Real datasets have correlated QIs where targeted suppression (LOCSUPR) could genuinely outperform generic kANON.
- Whether the tradeoff reverses at different reid_95 starting points. At reid_95=0.30 or 0.70, the relative performance could differ.
- Whether RC4's pipeline approach (GENERALIZE -> kANON) produces better outcomes. We didn't test it.

### Possible actions:

- **No threshold changes needed.** The boundaries work as designed.
- **Consider: should RC1/RC2 use more aggressive k values?** They currently use k=5 (RC1/LOCSUPR) and k=5 (RC2/kANON hybrid). QR4 uses higher k via escalation. If the goal is reid minimization, RC rules could escalate faster.
- **The retry engine is the real story.** Different starting rules converge to similar final methods. The retry engine is the dominant force in outcome determination, not the initial rule selection.
