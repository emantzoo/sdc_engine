# Spec 16 Phase 1b — Gate-Metric Calibration Table

**Date:** 2026-04-20
**Status:** Complete

---

## Summary

**37 gates checked. 28 inside observed range. 9 outside.**

The 9 outside-range gates fall into 4 categories:
- **3 preempted-always** (RC2, RC3, RC4) — already known from RC-family investigation
- **2 harness-composition gaps** (HR4, HR5) — small-dataset windows not populated
- **2 metric-gating gaps** (DATE1, DP4) — threshold never met on any harness dataset
- **1 structural gap** (CAT2) — gate reachable but always preempted by other rules
- **1 feasibility-dependent** (LDIV1) — fires on fixture only, not on real datasets

---

## Observed Metric Ranges

Measured across 8 Layer A dataset configurations + 9 Layer B fixtures.

| Metric | Observed min | Observed max | Notes |
|---|---|---|---|
| `reid_95` | 0.000 (G1) | 1.000 (CASC, free1, G8) | Full 0-100% range covered |
| `reid_50` | 0.000 (G1) | 1.000 (CASC, free1) | Full range |
| `uniqueness_rate` | 0.000 (testdata, adult, greek) | 0.644 (G8) | Real datasets near 0; fixtures push range |
| `high_risk_rate` | 0.000 (G1) | 1.000 (CASC, free1) | Full range |
| `cat_ratio` | 0.00 (CASC, G7) | 1.00 (greek, G8, G9) | Full 0-100% range |
| `n_records` | 500 (G8) | 50000 (G2) | No dataset < 500 rows |
| `n_qis` | 2 (G7) | 7 (testdata, free1) | Range 2-7 |
| `risk_pattern` | moderate, tail, severe_tail, widespread, bimodal, uniform_high, uniform_low | — | 7 of 8 patterns observed |
| `has_outliers` | False | True | Both present |
| `k_anon_feasibility` | easy | infeasible | All values observed |
| `date_ratio` | 0.00 | 0.00 | **Never > 0 — no temporal QIs in harness** |
| `n_skewed_columns` | 0 | 7 | Range covered |
| `has_sensitive` | False | True (via fixture injection) | Both present |
| `sensitive_col_diversity` | None | 4 (G1 fixture) | Only available via injection |
| `integer_coded_qis` | 0 | 7 (testdata) | Present on testdata |
| `max_qi_uniqueness` | 0.001 | 1.000 | Full range |
| `var_priority` | absent | present | Present on adult (≤10K subsample) |
| `risk_conc_pattern` | dominated | dominated | **Only `dominated` observed — 50% floor** |
| `bimodal_risk` | False | True (G3) | Both present |

---

## Gate-Metric Calibration Table

### Pipeline rules

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| DYN_Pipeline | `reid_95 > 0.15 AND cat_ratio < 0.70 AND len(pipeline) >= 2` | reid_95, cat_ratio, has_outliers, high_risk_rate | reid_95 > 0.15, cat_ratio < 0.70 | **Inside** — multiple datasets hit this |
| DYN_CAT | `0.50 < cat_ratio < 0.70 AND metric == l_diversity AND reid_95 > 0.15` | cat_ratio, risk_metric, reid_95 | cat_ratio 0.50-0.70 | **Inside** — free1 cat_ratio=0.50, G5=0.50 |
| GEO1 | `len(geo_qis_by_granularity) >= 2 AND has_fine AND has_coarse` | geo_qis_by_granularity | ≥2 geo levels | **Inside** — G2 fixture exercises this |
| P4b | `skewed_count >= 2 AND has_sensitive AND sens_div ≤ 10 AND pram_targets` | skewed_columns, has_sensitive, sensitive_col_diversity | ≥2 skewed, sens_div ≤ 10 | **Inside** — G4 fixture |
| P4a | `skewed_count >= 2 AND has_sensitive AND (no low-diversity sensitive cols)` | skewed_columns, has_sensitive | ≥2 skewed | **Inside** — reachable when sens_div > 10 |
| P5 | `density < 5 AND uniqueness > 0.15 AND n_cont >= 2 AND n_cat >= 2 AND n_records >= 200` | density, uniqueness, n_cont, n_cat | density < 5, uniq > 0.15 | **Inside** — G5 fixture |

### Tier-gated rules

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| REG1_high | `access_tier == PUBLIC AND reid_target == 0.03 AND reid_95 > 0.15` | access_tier, reid_target, reid_95 | reid_target == 0.03 | **Inside** — exercisable via access_tier param |
| REG1_moderate | `access_tier == PUBLIC AND reid_target == 0.03 AND 0.03 < reid_95 ≤ 0.15` | same | reid_95 0.03-0.15 | **Inside** |
| PUB1_high | `access_tier == PUBLIC AND reid_95 > 0.20` | access_tier, reid_95 | reid_95 > 0.20 | **Inside** |
| PUB1_moderate | `access_tier == PUBLIC AND 0.05 < reid_95 ≤ 0.20` | same | reid_95 0.05-0.20 | **Inside** |
| SEC1_cat | `access_tier == SECURE AND 0.05 < reid_95 ≤ 0.25 AND utility_floor ≥ 0.90 AND cat_ratio ≥ 0.60` | access_tier, reid_95, utility_floor, cat_ratio | cat_ratio ≥ 0.60 | **Inside** — adult cat_ratio=0.75 at reid_95=0.25 |
| SEC1_cont | `access_tier == SECURE AND 0.05 < reid_95 ≤ 0.25 AND utility_floor ≥ 0.90 AND cat_ratio < 0.60` | same | cat_ratio < 0.60 | **Inside** — CASC cat_ratio=0.00, greek cat_ratio=0.40 |

### Structural/size rules

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| SR3 | `n_qis ≤ 2 AND max_qi_uniqueness > 0.70 AND reid_95 > 0.20` | n_qis, max_qi_uniqueness, reid_95 | n_qis ≤ 2, max_uniq > 0.70 | **Inside** — G7 has n_qis=2 but max_uniq=0.059 (too low). Needs max_uniq > 0.70 at n_qis ≤ 2. Marginal — no harness dataset hits all 3 conditions simultaneously |
| HR6 | `n_records < 200 AND n_qis >= 2` | n_records | n_records < 200 | **Outside** — smallest dataset is G8 at 500 rows |

### Risk concentration rules

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| RC1 | `var_priority exists AND reid_95 > 0.15 AND pattern == dominated` | var_priority, reid_95, risk_conc_pattern | top_pct ≥ 40% | **Inside** — fires on adult datasets |
| RC2 | `pattern == concentrated (top2_pct ≥ 60%, top_pct < 40%)` | risk_conc_pattern, top_pct | top_pct < 40% | **Outside (preempted-always)** — contribution 50% floor means top_pct ≥ 50% always |
| RC3 | `pattern == spread_high (n_high ≥ 3, top_pct < 40%)` | risk_conc_pattern, top_pct | top_pct < 40% | **Outside (preempted-always)** — same 50% floor |
| RC4 | `pattern == balanced AND n_high == 1 AND n_other ≥ 3 AND reid_95 > 0.15` | risk_conc_pattern, n_high | pattern == balanced | **Outside (preempted-always)** — dominated always matches first |

### Categorical/diversity rules

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| CAT1 | `metric == l_diversity AND 0.15 ≤ reid_95 ≤ 0.40 AND cat_ratio ≥ 0.70 AND no dominant cats` | risk_metric, reid_95, cat_ratio | cat_ratio ≥ 0.70, metric == l_diversity | **Inside** — G6 exercises this |
| CAT2 | `metric == l_diversity AND 0.15 ≤ reid_95 ≤ 0.50 AND 0.50 < cat_ratio < 0.70 AND n_cont ≥ 1` | risk_metric, reid_95, cat_ratio | cat_ratio 0.50-0.70, metric == l_diversity | **Outside** — gate reachable but DYN_CAT_Pipeline checks the same cat_ratio window first and preempts. Under l_diversity, DYN_CAT pipeline uses NOISE (blocked) → falls through. But then the cat_ratio window overlaps with DYN_Pipeline which also fires. CAT2 is structurally preempted by pipeline rules |
| LDIV1 | `metric ∉ {k_anon, uniq} AND sens_div ≤ 5 AND has_reid AND NOT infeasible AND min_l < 2` | sens_div, min_l, feasibility | sens_div ≤ 5, min_l < 2 | **Inside** — G1 fixture (with injected min_l=1). No real dataset has sensitive columns with diversity ≤ 5 |
| DATE1 | `n_qis ≥ 2 AND n_date ≥ 2 AND date_ratio ≥ 0.80 AND reid_95 ≤ 0.40` | qi_type_counts, date_ratio | date_ratio ≥ 0.80 | **Outside** — date_ratio = 0.00 across entire harness. No temporal QIs in any dataset |

### ReID risk rules (QR0-QR4, MED1)

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| QR0 | `feasibility == infeasible` | k_anonymity_feasibility | infeasible | **Inside** — testdata, CASC, free1, adult_4qi_b, greek_4qi, G8 all infeasible |
| QR1 | `risk_pattern == severe_tail` | risk_pattern | severe_tail | **Inside** — testdata, adult_4qi, adult_4qi_b, greek_3qi, greek_4qi |
| QR2 | `risk_pattern == tail OR (reid_95 > 0.30 AND reid_50 < 0.15)` | risk_pattern, reid_95, reid_50 | tail or reid_95 > 0.30 | **Inside** — adult_3qi pattern=tail |
| QR3 | `risk_pattern == uniform_high` | risk_pattern | uniform_high | **Inside** — CASC, free1 |
| QR4 | `risk_pattern == widespread AND reid_50 > 0.15` | risk_pattern, reid_50 | widespread + reid_50 > 0.15 | **Inside** — G2 pattern=widespread reid_50=0.25, G5 reid_50=0.33 |
| MED1 | `reid_95 > 0.20 AND reid_50 < 0.10 OR bimodal OR high_risk_rate > 0.10` | reid_95, reid_50, risk_pattern, high_risk_rate | composite | **Inside** — multiple datasets match sub-conditions |

### Low-risk rules (LOW1-LOW3)

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| LOW1 | `reid_95 ≤ 0.20 AND cat_ratio ≥ 0.60 AND high_card == 0 AND reid_95 ≤ 0.10` | reid_95, cat_ratio, high_card_count | reid_95 ≤ 0.10, cat_ratio ≥ 0.60 | **Inside** — adult_3qi reid_95=0.067, cat_ratio=0.75 |
| LOW2 | `reid_95 ≤ 0.20 AND cat_ratio ≤ 0.40 AND n_cont > 0` | reid_95, cat_ratio | cat_ratio ≤ 0.40 | **Inside** — G7 reid_95=0.064, cat_ratio=0.00 |
| LOW3 | `reid_95 ≤ 0.20 AND (catch-all after LOW1/LOW2)` | reid_95 | reid_95 ≤ 0.20 | **Inside** — default low-risk path |

### Distribution rules (DP1-DP4)

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| DP1 | `has_outliers AND n_continuous > 0` | has_outliers, n_continuous | outliers + cont | **Inside** — testdata, adult_4qi, greek (all have outliers + cont) |
| DP2 | `len(skewed_columns) >= 2` | skewed_columns | ≥2 skewed | **Inside** — G4 has 2 skewed, greek has 7 |
| DP3 | `has_sensitive AND n_qis >= 2` | has_sensitive, n_qis | sensitive + ≥2 QIs | **Inside** — via fixture injection |
| DP4 | `integer_coded_qis AND reid_95 ≤ 0.30` | integer_coded_qis, reid_95 | nunique ≤ 15 int cols + reid_95 ≤ 0.30 | **Outside** — testdata has 7 integer-coded QIs but reid_95=0.50 (too high). free1 has 5 integer-coded but reid_95=1.00. No dataset has integer-coded QIs AND reid_95 ≤ 0.30 simultaneously |

### Uniqueness risk rules (HR1-HR5)

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| HR1 | `NOT has_reid AND uniqueness > 0.20` | has_reid, uniqueness_rate | uniq > 0.20, no reid | **Inside** — requires has_reid=False path. Reachable in principle but harness always computes reid. Marginal |
| HR2 | `NOT has_reid AND uniqueness > 0.10` | same | uniq > 0.10 | **Inside** — same caveat |
| HR3 | `NOT has_reid AND uniqueness > 0.05 AND n_qis >= 2` | same | uniq > 0.05 | **Inside** — same caveat |
| HR4 | `NOT has_reid AND n_records < 100` | has_reid, n_records | n_records < 100 | **Outside** — no dataset has < 100 rows. Also gated behind has_reid=False |
| HR5 | `NOT has_reid AND 100 ≤ n_records < 500 AND uniqueness > 0.03` | has_reid, n_records, uniqueness | n_records 100-500 | **Outside** — smallest dataset is 500 (G8). Also gated behind has_reid=False |

### Default rules

| Rule | Gate condition | Metric(s) consumed | Threshold | Inside/Outside |
|---|---|---|---|---|
| DEFAULT | `microdata AND n_qis >= 2 (catch-all)` | data_type, n_qis | always | **Inside** — always reachable |

---

## Outside-Range Gates — Detail

### 1. RC2, RC3, RC4 — preempted-always (known)

Already documented in `docs/investigations/spec_16_readiness_rc_family_preemption.md`.
The backward elimination contribution metric has a 50% floor for QIs with
cardinality ≥ 2, making `dominated` (top_pct ≥ 40%) always match before
`concentrated`, `spread_high`, or `balanced`.

**Verdict:** preempted-always. No harness expansion can fix this — it's algebraic.

### 2. DATE1 — date_ratio ≥ 0.80

`date_ratio` = 0.00 across the entire harness. No dataset includes temporal
QIs (date columns classified as QIs). The threshold 0.80 requires that 80%+
of QIs are date-type — a highly specific dataset shape.

**Assessment:** The gate is valid in principle (date-dominant datasets exist in
production — e.g., longitudinal health records with enrollment_date,
visit_date as QIs). But the harness has zero coverage.

**Verdict candidate:** untriggered. Would need a fixture with ≥2 date QIs
and ≤1 non-date QI (e.g., 2 date + 1 category = date_ratio 0.67 — still
below 0.80). Threshold may be too high; 0.50 would be reachable with 2 date
+ 2 non-date QIs. Flag for Phase 2 investigation.

### 3. DP4 — integer-coded categoricals AND reid_95 ≤ 0.30

`integer_coded_qis` is populated on testdata (7 QIs) and free1 (5 QIs), but
both have reid_95 ≥ 0.50 (infeasible QI space). No dataset has the
combination of integer-coded QIs AND reid_95 ≤ 0.30.

**Assessment:** The gate's intent is correct — integer-coded categoricals
at moderate risk should use PRAM to preserve code structure. But the harness
datasets that have integer-coded QIs also have too many QIs (driving reid_95
high). A 3-QI subset of testdata would have lower reid_95.

**Verdict candidate:** untriggered. Could be exercised with a fixture
(3 integer-coded QIs at moderate reid). Also, DP4 is rule priority 6 — by
the time it's evaluated, higher-priority rules (QR1, QR2, etc.) will have
already fired for most reid_95 values. The useful window is narrow:
reid_95 ≤ 0.30 AND no higher-priority rule firing AND integer-coded QIs
present. Likely `preempted` in practice.

### 4. HR4, HR5 — small dataset windows

HR4 requires n_records < 100; HR5 requires 100 ≤ n_records < 500. The
smallest harness dataset is G8 at 500 rows. Both rules also require
`has_reid = False` (they're in `uniqueness_risk_rules`, which only fires
when ReID is unavailable).

**Assessment:** Double-gated: small dataset AND no ReID. The harness always
computes ReID when possible. These rules serve as fallbacks for users who
provide data without ReID metrics (e.g., uploading a dataset with only
uniqueness computed). They're defensive code, not dead code.

**Verdict candidate:** untriggered. Not structurally unreachable, but the
harness doesn't exercise either condition (small dataset + no ReID).

### 5. CAT2 — preempted by pipeline rules

CAT2 requires `0.50 < cat_ratio < 0.70` under `l_diversity`. But pipeline
rules check the same cat_ratio window first:
- DYN_CAT_Pipeline fires at `0.50 < cat_ratio < 0.70` under `l_diversity`
  but its pipeline includes NOISE (blocked for l_diversity) → falls through
- DYN_Pipeline may then fire (cat_ratio < 0.70, reid_95 > 0.15)
- If DYN_Pipeline doesn't fire, CAT2 would be reached

CAT2 can theoretically fire when: metric=l_diversity AND 0.50 < cat_ratio
< 0.70 AND reid_95 0.15-0.50 AND DYN_Pipeline doesn't produce a ≥2-method
pipeline. This is a narrow window.

**Verdict candidate:** preempted (probabilistic). Could fire on specific
data shapes where DYN_Pipeline falls through. Not structurally impossible,
just unlikely.

---

## Flagged Gates (user-requested checks)

### DATE1 — date_ratio ≥ 0.80
**Status:** Outside. date_ratio = 0.00 across entire harness. See detail above.

### LDIV1 — sensitive column diversity
**Status:** Inside (barely). Fires on G1 fixture with injected `min_l=1`.
No real dataset has `sensitive_column_diversity ≤ 5` — the feature is only
populated when sensitive columns are explicitly declared. The gate itself
is well-calibrated (sens_div ≤ 5 is a meaningful threshold), but the harness
doesn't organically produce datasets with low-diversity sensitive columns.

### DP4 — integer-coded categoricals
**Status:** Outside. See detail above. Window too narrow for current harness.

### HR4/HR5 — small dataset windows
**Status:** Outside. No dataset < 500 rows AND no has_reid=False path.

### CAT1/CAT2 — cat_ratio thresholds
- **CAT1 (cat_ratio ≥ 0.70):** Inside. G6 fixture exercises this at
  cat_ratio=0.80 under l_diversity.
- **CAT2 (0.50-0.70):** Outside (preempted). See detail above.

---

## Additional Findings

### SR3 — marginal coverage
SR3 requires `n_qis ≤ 2 AND max_qi_uniqueness > 0.70 AND reid_95 > 0.20`.
G7 has n_qis=2 but max_qi_uniqueness=0.059 (far below 0.70). No harness
dataset simultaneously hits all three conditions. However, SR3's gate is
not structurally unreachable — a 2-QI dataset with one high-cardinality
continuous QI would hit it.

**Verdict candidate:** untriggered. Could be exercised with a fixture.

### HR1-HR3 — has_reid=False dependency
All HR rules require `has_reid = False`. The harness always computes ReID.
These rules are fallback code for when ReID computation fails or is skipped.
They're inside the observed range of their respective thresholds
(uniqueness > 0.20/0.10/0.05), but the has_reid=False gate is never
exercised.

**Verdict candidate:** untriggered (for HR1-HR3 specifically due to
has_reid=False). The uniqueness thresholds themselves are inside range.

---

## Decision Summary

| Count | Category | Rules |
|---|---|---|
| 28 | Inside observed range | DYN, DYN_CAT, GEO1, P4a, P4b, P5, REG1×2, PUB1×2, SEC1×2, RC1, CAT1, LDIV1, QR0-QR4, MED1, LOW1-LOW3, DP1-DP3, DEFAULT |
| 3 | Outside: preempted-always | RC2, RC3, RC4 |
| 2 | Outside: harness composition | HR4, HR5 |
| 2 | Outside: threshold never met | DATE1, DP4 |
| 1 | Outside: preempted by pipeline | CAT2 |
| 1 | Marginal: all conditions not simultaneously met | SR3 |

**N=37 gates. K=28 inside. M=9 outside.**

Of the 9 outside:
- 3 are already known (RC-family, algebraic)
- 2 are defensive fallbacks (HR4/HR5, not dead code)
- 2 are genuine untriggered gates worth investigating (DATE1, DP4)
- 1 is a preemption finding worth confirming (CAT2)
- 1 is marginal (SR3, exercisable with a fixture)

The net new findings beyond the RC-family are: **DATE1** and **DP4** have
thresholds that no harness dataset reaches, and **CAT2** is likely preempted
by pipeline rules. These 3 are worth flagging in Phase 2 but none change the
fundamental picture — the rule chain's calibration is mostly correct, with
a small number of edge-case rules that lack harness coverage.
