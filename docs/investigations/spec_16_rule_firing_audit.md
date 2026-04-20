# Spec 16 — Rule Firing Audit Report

**Date:** 2026-04-20
**Status:** Complete (Phases 1 + 1b only; Phases 2-3 not run)

---

## Executive Summary

The rule chain has 37 distinct rule gates across 15 rule-factory functions
plus pipeline rules. This audit checked every gate's threshold against the
observed metric ranges across 8 Layer A dataset configurations and 9 Layer B
fixtures (Spec 16a).

**37 gates checked. 28 inside observed range. 9 outside.**

Phases 2 (harness firing matrix) and 3 (counterfactual outcome measurement)
were not run. Consequently, live/niche/redundant distinctions among firing
rules are not measured — only structural reachability. Rules verdicted as
"live-unverified" below would need Phase 2 data to confirm firing frequency
and outcome delta.

---

## Methodology

### What was done

1. **Phase 1: Instrumentation** — `_rule_trace` added to `select_method_suite()`
   behind `SDC_RULE_AUDIT=1` env flag. Evaluates all 15 rule factories + pipeline
   without short-circuiting; records `{rule, applies, method}` per factory plus
   optional `{blocked, blocked_reason}` and `{sub_rules}` for RC-family. 11 tests.
   Committed as `b7b1bf2` and `7491cf5`.

2. **Phase 1b: Gate-metric calibration** — Static analysis of every gate condition
   against observed metric ranges. Identifies which thresholds lie inside vs outside
   the achievable range. Report: `spec_16_gate_metric_calibration.md`.

3. **Evidence from Spec 15 and Spec 16a readiness** — RC-family preemption
   investigation, Fix 0 (GENERALIZE config), fixture verification (9/9 PASS),
   dormant-rule taxonomy.

### What was NOT done

- **Phase 2: Harness firing matrix** — No systematic run of all datasets × metrics
  × tiers with audit flag. Fire counts and preempted-by maps are not measured.
- **Phase 3: Counterfactual outcome measurement** — No "what would have happened
  under the next-best rule" analysis. Cannot distinguish live vs niche vs redundant.

### Implication for verdicts

Rules that are inside the observed gate range and not structurally preempted
are verdicted **live-unverified** rather than **live**. The "unverified" suffix
means: the gate is reachable, but we haven't measured how often it fires or
whether it improves outcomes. Phase 2 would upgrade these to live/niche/redundant.

---

## Per-Rule Verdicts

### Verdict vocabulary (8 verdicts)

| Verdict | Meaning |
|---|---|
| **live-unverified** | Gate inside observed range, not preempted. Phases 2-3 needed for fire count and outcome delta |
| **niche** | Fires in <10% of runs but improves outcome. *Not assignable without Phase 2* |
| **redundant** | Fires but same outcome as next-best rule. *Not assignable without Phase 3* |
| **config-blocked** | Gate matches but method rejected by config filter. Fixed by Fix 0 for known cases |
| **preempted** | Gate matches on some data but higher-priority rule always fires first. Probabilistic |
| **preempted-always** | Structurally impossible given metric algebra. Not probabilistic |
| **untriggered** | Gate not met by any harness dataset. May fire on production data |
| **unreachable** | Code path logically impossible. Self-contradictory gate |

### Pipeline rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| DYN_Pipeline | **live-unverified** | Gate inside range. Fires on G3 fixture. Multiple datasets have reid_95 > 0.15 with mixed types | Phase 2 for fire count |
| DYN_CAT_Pipeline | **unreachable** | Gated to `l_diversity` metric but pipeline includes NOISE, which is blocked for l_diversity. Gate matches → pipeline rejected → fallthrough. Silent failure | Delete or replace NOISE with structural method in pipeline |
| GEO1 | **live-unverified** | Gate inside range (post-Fix 0). Fires on G2 fixture. Requires multi-level geo QIs | Phase 2 for fire count on real geo datasets |
| P4b_Skewed_Sensitive | **live-unverified** | Gate inside range. Fires on G4 fixture. Requires skewed continuous + low-diversity sensitive cols | Phase 2 |
| P4a_Skewed_Structural | **live-unverified** | Gate inside range. Sister case of P4b when sensitive cols have high diversity | Phase 2 |
| P5_Small_Dataset | **live-unverified** | Gate inside range. Fires on G5 fixture. Requires sparse density + mixed types | Phase 2 |

### Tier-gated rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| REG1_high | **live-unverified** | Gate inside range. Exercisable via access_tier=PUBLIC + reid_target=0.03 | Phase 2 with tier variation |
| REG1_moderate | **live-unverified** | Gate inside range. Same | Phase 2 |
| PUB1_high | **live-unverified** | Gate inside range. reid_95 > 0.20 at PUBLIC tier | Phase 2 |
| PUB1_moderate | **live-unverified** | Gate inside range. 0.05 < reid_95 ≤ 0.20 at PUBLIC | Phase 2 |
| SEC1_cat | **live-unverified** | Gate inside range. PRAM at SECURE tier. Config-blocked under k_anonymity/uniqueness (by design — PRAM incompatible) | Phase 2 |
| SEC1_cont | **live-unverified** | Gate inside range. NOISE at SECURE tier. Same metric blocking as SEC1_cat | Phase 2 |

### Structural / size rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| SR3 | **untriggered** | All 3 gate conditions (n_qis ≤ 2, max_uniq > 0.70, reid > 0.20) never simultaneously met in harness. Gate is valid — a 2-QI dataset with one high-cardinality continuous QI would hit it | Add fixture or accept as edge-case |
| HR6 | **untriggered** | No dataset < 200 rows in harness. Gate is valid defensive code | Accept as edge-case |

### Risk concentration rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| RC1 | **live-unverified** | Fires on adult, greek datasets via backward elimination. Only RC rule observed to fire. Pattern `dominated` always matches due to 50% floor | Phase 2 for fire count |
| RC2 | **preempted-always** | Requires `concentrated` pattern (top_pct < 40%). Contribution metric 50% floor means top_pct ≥ 50% always. RC1's `dominated` matches first | Delete or redesign contribution metric. See `spec_16_readiness_rc_family_preemption.md` |
| RC3 | **preempted-always** | Requires `spread_high` (top_pct < 40%). Same 50% floor blocker | Same as RC2 |
| RC4 | **preempted-always** | Requires `balanced` pattern. `dominated` always matches first. Additionally, the bottleneck shape (1 HIGH 15-39%, 3+ LOW) is incompatible with the contribution metric's range | Same as RC2. G9 fixture confirms rule-selection logic is correct when given injected inputs |

### Categorical / diversity rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| CAT1 | **live-unverified** | Gate inside range. Fires on G6 fixture under l_diversity. PRAM for categorical-dominant data at moderate risk | Phase 2. Note: PRAM invalidates frequency-count metrics — gate correctly restricts to l_diversity |
| CAT2 | **preempted** | Gate reachable (0.50 < cat_ratio < 0.70 under l_diversity) but pipeline rules (DYN_CAT) check the same window first. DYN_CAT's NOISE is blocked for l_diversity → falls through → but DYN_Pipeline may then fire. Narrow surviving window | Investigate whether CAT2 ever wins after both DYN_CAT and DYN_Pipeline fall through. Likely redundant with CAT1 at wider cat_ratio |
| LDIV1 | **live-unverified** | Gate inside range (fires on G1 fixture with injected min_l). Requires sensitive columns with diversity ≤ 5. No real harness dataset has this organically — only via fixture injection | Phase 2 with sensitive-column datasets |
| DATE1 | **untriggered** | date_ratio = 0.00 across entire harness. Threshold 0.80 requires ≥80% temporal QIs — no temporal QIs exist in any dataset. Gate is valid for longitudinal data | Add temporal fixture. Consider lowering threshold from 0.80 to 0.50 |

### ReID risk rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| QR0 | **live-unverified** | Gate inside range. testdata, CASC, free1, adult_4qi_b, greek_4qi, G8 all infeasible. Post-Fix 0: GENERALIZE_FIRST is now allowed | Phase 2 for fire count. Pre-Fix 0 runs fell through to other rules |
| QR1 | **live-unverified** | Gate inside range. risk_pattern=severe_tail on testdata, adult_4qi, adult_4qi_b, greek_3qi, greek_4qi | Phase 2 |
| QR2 (3 variants) | **live-unverified** | Gate inside range. risk_pattern=tail on adult_3qi. Heavy/moderate/low-suppression variants all reachable | Phase 2 |
| QR3 | **live-unverified** | Gate inside range. risk_pattern=uniform_high on CASC, free1 | Phase 2 |
| QR4 (2 variants) | **live-unverified** | Gate inside range. risk_pattern=widespread + reid_50 > 0.15 on G2, G5 | Phase 2 |
| MED1 | **live-unverified** | Gate inside range. Composite trigger: moderate spread OR bimodal OR high_risk_rate > 0.10. Multiple datasets match sub-conditions | Phase 2 |

### Low-risk rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| LOW1 | **live-unverified** | Gate inside range. adult_3qi: reid_95=0.067, cat_ratio=0.75. Selects PRAM | Phase 2. Note: PRAM at low risk — same metric-invalidation caveat as CAT1 |
| LOW2 (2 variants) | **live-unverified** | Gate inside range. G7 fixture: reid_95=0.064, cat_ratio=0.00. NOISE or kANON for continuous-dominant | Phase 2 |
| LOW3 | **live-unverified** | Gate inside range. Catch-all for mixed types at low risk | Phase 2 |

### Distribution rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| DP1 | **live-unverified** | Gate inside range. testdata, adult, greek all have outliers + continuous vars. But DP1 is priority 6 — likely preempted by higher-priority rules in practice | Phase 2 to measure actual fire rate. May be `preempted` |
| DP2 | **live-unverified** | Gate inside range. G4 has 2 skewed cols, greek has 7. Same preemption caveat as DP1 | Phase 2 |
| DP3 | **live-unverified** | Gate inside range via fixture injection. Requires sensitive columns | Phase 2 |
| DP4 | **untriggered** | integer_coded_qis present on testdata (7) and free1 (5), but reid_95 ≥ 0.50 on both. Gate requires reid_95 ≤ 0.30 AND integer-coded QIs simultaneously. Also priority 6 — higher rules fire first | Add fixture with 3 integer-coded QIs at moderate reid, or accept as edge-case |

### Uniqueness risk rules (no-ReID fallback)

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| HR1 | **untriggered** | Requires has_reid=False. Harness always computes ReID. Uniqueness threshold (> 0.20) is inside range | Exercise with has_reid=False path. Defensive code, not dead |
| HR2 | **untriggered** | Same has_reid=False dependency. Uniqueness > 0.10 inside range | Same |
| HR3 | **untriggered** | Same. Uniqueness > 0.05 inside range | Same |
| HR4 | **untriggered** | Requires has_reid=False AND n_records < 100. No dataset < 100 rows | Defensive code. Accept as edge-case |
| HR5 | **untriggered** | Requires has_reid=False AND 100 ≤ n_records < 500. No dataset in this window | Same |

### Default rules

| Rule | Verdict | Rationale | Recommended follow-up |
|---|---|---|---|
| DEFAULT | **live-unverified** | Always reachable as catch-all. Fires when no higher-priority rule matches | Phase 2 — should fire rarely if rule chain has good coverage |

---

## Summary Statistics

| Verdict | Count | Rules |
|---|---|---|
| live-unverified | 24 | DYN, GEO1, P4a, P4b, P5, REG1×2, PUB1×2, SEC1×2, RC1, CAT1, LDIV1, QR0-QR4, MED1, LOW1-LOW3, DP1-DP3, DEFAULT |
| preempted-always | 3 | RC2, RC3, RC4 |
| preempted | 1 | CAT2 |
| unreachable | 1 | DYN_CAT_Pipeline |
| untriggered | 8 | SR3, HR6, DATE1, DP4, HR1, HR2, HR3, HR4, HR5 |
| **Total** | **37** | |

Note: HR4 and HR5 counted separately from HR1-HR3 because they have
additional n_records constraints beyond the has_reid=False gate.

---

## Findings

### Finding 1: RC-family preemption (preempted-always × 3)

The backward elimination contribution metric has a 50% floor for QIs with
cardinality ≥ 2. This means `classify_risk_concentration()` always assigns
the `dominated` pattern, and RC1 fires first. RC2, RC3, RC4 are structurally
unreachable.

**Evidence:** Algebraic proof + empirical verification on adult dataset.
Full investigation: `spec_16_readiness_rc_family_preemption.md`.

**Follow-up:** Delete RC2/RC3/RC4 or redesign the contribution metric to
produce meaningful variation across patterns. Scoped for Spec 18.

### Finding 2: DYN_CAT_Pipeline is unreachable (unreachable × 1)

DYN_CAT is gated to `l_diversity` metric but its pipeline includes NOISE,
which is blocked for l_diversity by `METRIC_ALLOWED_METHODS`. The gate
matches → pipeline assembled → `_all_allowed()` rejects → falls through.

**Evidence:** G7 fixture history (original CAT2 target hit the same issue).
Fixtures README documents the discovery.

**Follow-up:** Either remove NOISE from DYN_CAT's pipeline (replace with
structural method) or remove the l_diversity gating. Scoped for Spec 18.

### Finding 3: CAT2 is likely preempted by pipeline rules (preempted × 1)

CAT2 and DYN_CAT occupy the same cat_ratio window (0.50-0.70) under
l_diversity. DYN_CAT fires first (pipeline rules precede rule factories).
When DYN_CAT is blocked (Finding 2), DYN_Pipeline may then fire if it
produces a ≥2-method pipeline. CAT2's surviving window is narrow.

**Evidence:** Phase 1b calibration analysis. G7 fixture redesign history
(CAT2 was abandoned as a fixture target).

**Follow-up:** Confirm via Phase 2 or delete alongside DYN_CAT fix.

### Finding 4: DATE1 has zero harness coverage (untriggered × 1)

`date_ratio` = 0.00 across the entire harness. No dataset includes temporal
QIs. The threshold (0.80) requires a very specific dataset shape: ≥80% of
QIs are date-type.

**Evidence:** Phase 1b metric range scan.

**Follow-up:** Add a temporal fixture if the rule is worth keeping. Consider
lowering threshold from 0.80 to 0.50. Scoped for Spec 18.

### Finding 5: DP4 has a narrow untriggered window (untriggered × 1)

integer_coded_qis exist in the harness (testdata: 7, free1: 5) but always
co-occur with reid_95 ≥ 0.50 (infeasible QI space). DP4 requires
reid_95 ≤ 0.30. Additionally, DP4 is priority 6 — higher-priority rules
preempt it for most reid_95 values.

**Evidence:** Phase 1b metric range scan.

**Follow-up:** Low priority. Could add fixture but the useful firing window
is very narrow. Candidate for deletion in Spec 18.

### Finding 6: HR1-HR5 are untriggered due to has_reid=False gate

All uniqueness_risk_rules require has_reid=False. The harness always computes
ReID. These rules are defensive fallbacks for when ReID computation fails or
is skipped. The uniqueness thresholds themselves are inside range.

**Evidence:** Phase 1b analysis. Spec 15 Item 3 (HR1-HR5 dormant).

**Follow-up:** Accept as defensive code. No deletion needed — they serve
a real purpose when ReID is unavailable. Could exercise with a
has_reid=False test path in Phase 2.

### Finding 7: Fix 0 unblocked three rules (config-blocked → live)

GEO1, RC4, and QR0 were silently blocked before Fix 0 added GENERALIZE /
GENERALIZE_FIRST to `METRIC_ALLOWED_METHODS`. All three now fire correctly
on their respective fixtures.

**Evidence:** Fixture verification (9/9 PASS post-Fix 0). G2 fixture
history documenting the discovery.

**Follow-up:** Complete. Prevention test added
(`test_method_metric_coverage.py`).

---

## Recommended Follow-Up (Spec 18 Scope)

Based on this audit, Spec 18 should address:

| Priority | Action | Rules affected | Rationale |
|---|---|---|---|
| 1 | Delete RC2, RC3, RC4 | 3 rules | preempted-always. No production value |
| 2 | Fix or delete DYN_CAT_Pipeline | 1 pipeline | unreachable. NOISE/l_diversity contradiction |
| 3 | Investigate CAT2 preemption | 1 rule | Likely preempted by pipeline rules. Delete if confirmed |
| 4 | Add temporal fixture for DATE1 | 1 rule | untriggered. Decide keep (lower threshold?) or delete |
| 5 | Decide on DP4 | 1 rule | untriggered, narrow window. Candidate for deletion |
| 6 | Accept HR1-HR5 as defensive | 5 rules | untriggered but valid. Document as fallback-only |
| 7 | Add fixture for SR3 | 1 rule | untriggered but valid gate. Low priority |

Items 1-3 are deletions/fixes (reduce dead code). Items 4-7 are
keep-or-delete decisions that require less urgency.

---

## Phases Not Run — Impact Assessment

### Phase 2 (harness firing matrix)

Would have provided fire counts for each rule across ~500-800 runs. Without
it, the 24 live-unverified rules remain unranked by importance. The most
valuable Phase 2 output would be identifying which of the 24 are **redundant**
(fire but produce the same outcome as the next-best rule).

**Risk of skipping:** Low. The calibration table already identifies the
structurally dead rules (findings 1-5). Redundancy detection is an
optimization — it saves maintenance effort but doesn't prevent silent bugs.

### Phase 3 (counterfactual outcome measurement)

Would have measured outcome delta for each firing rule. Without it, we can't
distinguish niche (fires rarely but helps) from redundant (fires but doesn't
help). This matters most for DP1-DP4 and LOW1-LOW3, which are low-priority
rules that may duplicate higher-priority rules' selections.

**Risk of skipping:** Low. The rules most likely to be redundant (DP1-DP4)
are already low-priority in the chain. Deleting them based on Phase 2 fire
counts alone (if they never fire) would be safe.

---

## Caveats

1. **Harness coverage.** The harness has 8 base datasets with 17 QI
   configurations + 9 fixtures. Production data may have distributions
   not represented (temporal data, geographic hierarchies, integer-coded
   categoricals). Rules verdicted as "untriggered" may fire on production
   data.

2. **Metric variation.** Phase 1b analyzed features under reid95 only. Rules
   gated to other metrics (l_diversity for CAT1/CAT2/LDIV1, k_anonymity for
   structural-only paths) were assessed by gate analysis, not by running the
   harness under those metrics.

3. **Tier variation.** Tier-gated rules (REG1, PUB1, SEC1) were assessed by
   gate analysis. No harness runs were performed with non-default tiers.

4. **live-unverified ≠ live.** 24 rules are verdicted live-unverified. Some
   may be redundant (fire but don't change the outcome vs the next-best rule)
   or niche (fire rarely). Phase 2 would resolve this.
