# Spec 19 Phase 2 — Closeout Report

## Summary

Phase 2 applied wire-or-delete decisions to all rules whose feature
dependencies were connected during Phase 1.  Three groups of deletions
and one gate adjustment were committed:

| Commit | Change | Files |
|---|---|---|
| `6050a85` | Delete RC2/RC3/RC4 (structurally unreachable) | 13 files, -105 net |
| `73a8f6d` | Delete DYN_CAT/CAT2 (self-contradictory) | 13 files, -109 net |
| `d7bcbc9` | Gate LDIV1 under reid95 > 0.10 | 2 files, +67 |

All deletions committed.

## Per-rule verdicts

| Rule | Verdict | Rationale |
|---|---|---|
| RC2/RC3/RC4 | **Deleted** | Structurally preempted by RC1 (backward elimination contribution has ~50% floor) |
| DYN_CAT | **Deleted** | Self-contradictory: gated to l_diversity, pipeline used NOISE (blocked for l_diversity) |
| CAT2 | **Deleted** | Same contradiction as DYN_CAT |
| DP4 | **Deleted** | Dead code: LOW3 unconditional catch-all always preempts DP4 |
| P4a | **Deleted** | Crash bug (`KeyError` at `pipelines.py:478`) + no unique value over RC1/DEFAULT |
| P4b | **Deleted** | |skew| > 1.5 threshold too narrow; no harness dataset reaches it; G4 fixture proves logic but niche is impractical |
| LDIV1 | **Keep with gate** | Reid95 gate added: only fires when reid_95 ≤ 0.10 under reid95 metric. Fires freely under l_diversity |
| SR3 | **Keep** | Covers large datasets with ≤2 near-unique QIs where HR6 and RC1 both miss |
| GEO1 | **Keep** | GENERALIZE→kANON is correct SDC for multi-level geo. G2 fixture proves it works |
| DATE1 | **Keep** | Gate is narrow but rule embodies genuine domain logic for temporal data |
| DP3 | **Keep** | Bumps k from 3→5 when sensitive columns present and perturbative rules are metric-blocked |

## Findings

### P4a latent crash (Case A — no test coverage)

P4a returned `{'applies': True, 'use_pipeline': False}` without a `pipeline`
key.  `_all_allowed(pipeline_result.get('pipeline', []))` evaluated to `True`
(vacuous truth on empty list), routing execution to line 478 which accessed
`pipeline_result['pipeline']` → `KeyError`.

The crash never surfaced because:
- No test exercised P4a's `applies=True` path
- No harness/fixture dataset reached the gate (|skew| > 1.5 on 2+ continuous
  QIs + high-diversity sensitive columns)
- The highest skewness in any harness dataset is 0.530 (UCI Adult `age`)

**Implication**: Rules with no `applies=True` test path may harbour similar
latent bugs.  A systematic grep for untested positive branches is recommended
for Spec 20 cleanup.

### LDIV1 priority position under reid95

LDIV1's action (PRAM on sensitive columns) doesn't reduce QI-based
re-identification risk.  On `adult_mid_reid` (reid_95=0.250), LDIV1 fired as
primary under reid95, causing the retry engine to waste iterations escalating
PRAM before falling back to kANON k=5 — which is what MED1 would have
selected immediately.

Fix: reid95 gate (only fire when reid_95 ≤ 0.10).  This is a targeted fix for
the reid95 case, not a full priority rework.  LDIV1's position in the rule
chain relative to QR/MED/LOW is unchanged.  A broader priority restructuring
(e.g., making LDIV1 supplementary rather than primary) is a design question
deferred beyond Spec 19.

### DP4 structural unreachability

DP4 requires `integer_coded_qis` non-empty AND `reid_95 ≤ 0.20`.  But
`low_risk_rules()` fires on the same `has_reid=True` + `reid_95 ≤ 0.20`
condition, and LOW3 is an unconditional catch-all at the bottom of that
function.  Since `low_risk_rules` precedes `distribution_rules` in the
factory chain, LOW3 always preempts DP4.  Provably dead code.

### Stale injection workarounds

Two fixture verification scripts contain redundant feature injections:
- G10 (`verify_fixtures.py`): injects `max_qi_uniqueness=0.80`, but
  `build_data_features()` already computes the same value natively
- G2 (`verify_fixtures.py`): `inject_geo=True` flag is redundant, same reason

Both are cleanup items, not bugs.  The injected values match the computed
values, so test outcomes are correct.

## Harness coverage gap

### Context

The engine is a generic SDC tool.  It is intended to handle arbitrary dataset
shapes — temporal, geographic, sensitive-attribute-dominant, demographic,
financial, administrative, etc.

### Current harness composition

The empirical harness consists of:

- **8 core datasets**: UCI Adult (3 boundary configs), Greek real estate
  (2 configs), sdcMicro testdata, CASCrefmicrodata, free1
- **10 G-series fixtures**: Synthetic datasets targeting specific rules
  (G1–G10, with G8 testing floor regime and G9 removed after RC4 deletion)

All core datasets are **cross-sectional demographic or financial data**.
The G-series fixtures are synthetic constructions, not representative of
real-world data shapes.

### Coverage gaps by data shape

| Data shape | Harness coverage | Rules affected |
|---|---|---|
| **Temporal-dominant** (≥2 date QIs, ≥50% date ratio) | Zero | DATE1 has never fired on any run in project history |
| **Geographic multi-level** (fine + coarse geo hierarchy) | G2 fixture only (synthetic) | GEO1 fires on G2 but no real dataset has ≥2 geo QIs |
| **Sensitive-attribute-dominant with skew** (|skew| > 1.5 + sensitive cols) | G4 fixture only (synthetic) | P4b fires on G4 but no real dataset has |skew| > 1.5 |
| **Integer-coded categoricals** (numeric QIs with ≤15 unique ints) | Detected in testdata/free1/greek but DP4 is dead code | DP4 being deleted; LOW1/LOW3 cover the cases |

### Implication for "keep as niche" verdicts

Several rules verdicted as "keep" have **no real-data coverage**:

- **DATE1**: Fires on zero datasets.  Gate was widened (0.80→0.50) and still
  unreachable.  Name-based date detection is fragile (misses `DOB`,
  `contract_start`, etc.).
- **GEO1**: Fires only on synthetic G2 fixture.  Real geographic datasets
  (e.g., Greek real estate with "Nomarchía") are missed because the geo hint
  list lacks the derived noun form.  The `cat_ratio < 0.70` guard blocks
  most real geographic datasets (which are categorical-dominant).
- **P4b**: Fires only on synthetic G4 fixture.  The |skew| > 1.5 threshold
  on 2+ continuous QIs is strict; no real dataset in the harness reaches it.

"Keep as niche" currently means: **the rule logic is correct and the code
path is exercised by a synthetic fixture, but the rule has never been
validated on real data matching its intended use case.**

### Recommendation

A future harness expansion spec should add fixtures or real datasets that
exercise these niche rules on data shapes consistent with the product's
generic scope:

1. **Temporal-dominant**: Survey panel data, event logs, longitudinal studies.
   QIs like `birth_year`, `enrollment_quarter`, `report_month`.
2. **Geographic multi-level**: Census data with postcode + region + country
   structure.  Administrative data with municipality + prefecture hierarchy.
3. **Sensitive-attribute-dominant with skew**: Medical data with rare
   diagnosis codes, financial data with lognormal income/wealth distributions.
4. **Integer-coded categoricals**: Administrative codes (occupation, industry,
   education level) with ≤15 unique integer values.

**Out of Spec 19 scope.**  Documented here for the next harness expansion
spec.

## Deferred items (Spec 20 cleanup)

- 8 pre-existing test failures: 5 PRAM-gating (aa1f943 regression) +
  3 GENERALIZE_FIRST test-expectation mismatch.  Two separate issues.
- Systematic check for rules with no `applies=True` test path
- Stale G10/G2 injection cleanup in `verify_fixtures.py`
- LDIV1 priority restructuring (supplementary vs primary) — design question
- HR1–HR5 formal close-out (depend on `uniqueness_rate` which is not
  populated in the feature pipeline)
- RC1-infeasibility fix
