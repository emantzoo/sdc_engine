# RC-Family Preemption Investigation

**Date:** 2026-04-20
**Context:** Spec 16a fixture expansion. Attempting to build G9 fixture for
RC4_Single_Bottleneck revealed that the backward elimination contribution
metric's algebraic properties make RC1 the only RC rule that can fire in
practice. RC2, RC3, and RC4 are all structurally preempted.

---

## The contribution metric and its 50% floor

Backward elimination contribution for QI_i:

```
contrib_i = (reid_full - reid_without_i) / reid_full * 100
```

Where `reid_without_i` is reid_95 computed on all QIs except QI_i.

For a QI with cardinality k_i and approximately uniform distribution across
independent categoricals:

```
eq_without_i = eq_full * k_i
reid_without_i ≈ reid_full / k_i
contrib_i ≈ (1 - 1/k_i) * 100
```

| Cardinality k | Contribution | Label |
|---|---|---|
| 2 | 50.0% | HIGH |
| 3 | 66.7% | HIGH |
| 4 | 75.0% | HIGH |
| 5 | 80.0% | HIGH |
| 10 | 90.0% | HIGH |

**Every QI with k >= 2 produces contribution >= 50%.** The minimum possible
contribution for any QI with 2+ distinct values is 50%. To achieve LOW
(< 3%): need `1/k > 0.97`, i.e., `k < 1.03` — impossible for any real QI.

This is the 50% floor. It is a property of the leave-one-out contribution
metric, not of any particular dataset.

---

## Why `dominated` always matches first

`classify_risk_concentration()` in `features.py` evaluates patterns in
priority order:

```python
if top_pct >= 40:
    pattern = 'dominated'
elif top2_pct >= 60:
    pattern = 'concentrated'
elif n_high >= 3:
    pattern = 'spread_high'
else:
    pattern = 'balanced'
```

Since the 50% floor guarantees that the top contributor always has
contribution >= 50%, and `dominated` requires top_pct >= 40%, the
`dominated` branch always matches first.

The downstream consequence: `risk_concentration_rules()` checks
`pattern == 'dominated'` first (the RC1 branch), and it always matches
when var_priority exists and reid_95 > 0.15. RC2, RC3, and RC4 each
require a pattern that is never assigned.

---

## Consequences for RC2, RC3, and RC4

### RC2 (`concentrated`: top2_pct >= 60%, top_pct < 40%)

RC2 requires top_pct < 40%. The 50% floor means top_pct >= 50% for any
QI with k >= 2. RC2's gate is unreachable under the contribution metric.

### RC3 (`spread_high`: n_high >= 3, top_pct < 40%)

Same blocker: requires top_pct < 40%. Additionally, since every QI with
k >= 2 produces contribution >= 50% (all HIGH), n_high typically equals
n_qis — the `n_high >= 3` condition is trivially met for 3+ QIs, but
the pattern is always `dominated` first.

### RC4 (`balanced` + n_high == 1 + n_other >= 3)

RC4 has the preemption problem *and* an additional constraint problem.

**Preemption:** RC4 requires `balanced` pattern (none of dominated/
concentrated/spread_high). Since `dominated` always matches, RC4's
pattern condition is never met through genuine backward elimination.

**Bottleneck shape:** Even setting aside the pattern, RC4 requires
exactly 1 HIGH QI (15-39% contribution) and 3+ non-HIGH QIs. Under the
50% floor, every QI with k >= 2 is HIGH. The only way to achieve
n_high == 1 is with correlated categoricals where removing a QI doesn't
change equivalence class structure (contribution drops to 0%). But in
that scenario, the single HIGH QI's contribution is typically dominant
(>= 40%), producing `dominated` — not `balanced`.

**Tested empirically** on adult dataset (8K sample, 5 QIs):

```
age, sex, marital_status, race, education:
  age         : HIGH  75.0%
  sex         : LOW    0.0%
  marital     : LOW    0.0%
  race        : LOW    0.0%
  education   : LOW    0.0%
  pattern = dominated (top=75%)
```

This achieves n_high=1, n_low=4 — the RC4 contribution shape. But the
single HIGH QI has contribution 75%, which is `dominated` (>= 40%).
RC1 fires first.

The opposing constraints: RC4 needs the bottleneck QI at 15-39%
(moderate contribution) while other QIs are at < 3% (nearly redundant).
If others are redundant (correlated), the bottleneck drives most of the
risk — contribution goes high (>40%) — `dominated`. If others are
independent, they all have contribution >= 50% — all HIGH. No synthetic
dataset could hit the 15-39% window under the `max_n_records = 10,000`
backward elimination limit.

### Summary table

| Rule | Pattern condition | Why unreachable | Preempted? |
|---|---|---|---|
| RC1 | `dominated` (top >= 40%) | Always achievable — 50% floor exceeds 40% | No — fires first |
| RC2 | `concentrated` (top2 >= 60%, top < 40%) | Requires top < 40%; min is 50% | Yes — always |
| RC3 | `spread_high` (n_high >= 3, top < 40%) | Requires top < 40%; min is 50% | Yes — always |
| RC4 | `balanced` + n_high==1 + n_other>=3 | `balanced` never assigned; also needs 15-39% window | Yes — always |

---

## Empirical verification: RC1 fires on real datasets

RC1 fires on adult and greek datasets in the existing harness (both have
var_priority computed via the full backward elimination pipeline for
datasets <= 10K rows with <= 8 QIs).

On adult (8K sample, 5 QIs): `pattern = dominated`, `top_pct = 75.0%`,
`top_qi = age`. RC1 selects LOCSUPR k=5.

On greek (35K sample, 4 QIs): backward elimination runs on 10K subsample.
RC1 fires with `dominated` pattern.

---

## G9 fixture: rule-selection layer testing with injected var_priority

G9 uses injected var_priority to test RC4's rule-selection logic
independently of the contribution metric:

```python
var_priority = {
    'postcode': ('HIGH', 25.0),   # bottleneck in the 15-39% window
    'sex': ('LOW', 1.2),
    'marital': ('LOW', 2.1),
    'education': ('LOW', 1.8),
}
```

This confirms:
- Fix 0 (GENERALIZE added to METRIC_ALLOWED_METHODS) unblocked RC4's
  `['GENERALIZE', 'kANON']` pipeline at the rule-selection layer.
- RC4's rule logic is correct — it selects the right pipeline when given
  the right inputs. The issue is that the contribution metric never
  produces those inputs.

The fixture is labeled as injected in the verification README.

---

## Class-of-bug hypothesis

The RC-family preemption is a specific instance of a broader pattern:
**rule gates designed against intuitions about metric ranges that the
underlying metric cannot actually produce.**

The RC rules assumed that backward elimination contribution could produce
values across the full 0-100% range, with meaningful variation in
distribution patterns (dominated, concentrated, spread_high, balanced).
In practice, the leave-one-out metric has a 50% floor for k >= 2,
collapsing the entire pattern space to `dominated`.

Spec 16 should look for this pattern in other rule families:

- Do any gates consume `uniqueness_rate` with thresholds that
  `compute_uniqueness()` can't produce on real data?
- Do any gates consume `cat_ratio` with thresholds outside the range
  that typical datasets produce?
- Do any risk-pattern gates (tail/uniform_high/widespread) have threshold
  combinations that `compute_risk_pattern()` can't produce simultaneously?

The gate-metric calibration check (Spec 16 methodology step 1b) is
designed to catch this systematically.

---

## Production caveat

Real-world datasets may have correlation structures not testable with
synthetic data or UCI datasets. The finding is "structurally preempted
under the algebraic properties of the contribution metric" — which is
a stronger claim than "not observed on test data" but weaker than
"provably impossible for all possible data distributions." Highly
correlated production datasets might produce edge-case contribution
values, but even on the adult dataset (which has strong correlations),
the top contributor was 75% — still dominated.
