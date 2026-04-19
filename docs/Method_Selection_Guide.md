# Method Selection Guide

This guide covers ReID-based method selection, the reactive pipeline escalation system, and how SDC Engine results inform automatic method recommendation.

## Overview

The SDC Methods library uses a multi-rule system to automatically select the best protection method based on data characteristics and risk distribution patterns. The rules engine is integrated into the Configure view, providing automatic method recommendations after ReID calculation.

**Key Insight**: kANON is selected for most microdata cases because it's the most versatile structural method that reduces ReID. LOCSUPR is preferred for severe tail risk (targeted suppression) and benefits from SDC Engine-derived importance weights. Perturbation methods (PRAM, NOISE, RANKSWAP, RECSWAP) are used when ReID is already low and statistical utility is the priority. RANKSWAP preserves rank correlations for numeric data; RECSWAP preserves marginal distributions for mixed/categorical data.

## The Central Role of ReID

**ReID (Re-identification Risk)** is the single most important metric driving all method selection decisions. It measures per-record re-identification risk based on equivalence class sizes:

```
record_risk(i) = 1 / group_size(i)
```

where `group_size(i)` = number of records sharing identical QI values with record i. A unique record (group size = 1) has risk = 1.0 (certainly identifiable).

The three ReID percentiles play distinct roles in the decision tree:

| Metric | Formula | Decision Role |
|--------|---------|---------------|
| **ReID50** | 50th percentile of all record risks | **Spread indicator** — if >20%, risk is widespread (not just outliers) |
| **ReID95** | 95th percentile | **PRIMARY decision metric** — determines the risk severity tier and method category |
| **ReID99** | 99th percentile | **Tail indicator** — ReID99/ReID50 ratio reveals severe tail patterns requiring LOCSUPR |

### Why ReID Determines Everything

The fundamental insight is that **only structural methods (kANON, LOCSUPR) can reduce ReID**. Perturbation methods (PRAM, NOISE) modify values but don't change equivalence class structure — a unique record remains unique after perturbation.

This creates a hard decision boundary:

```
IF ReID95 > 5%:
    → MUST use structural method (kANON or LOCSUPR)
    → Perturbation CANNOT fix the problem
    → Rules QR1-QR8 select the specific structural method

IF ReID95 ≤ 5%:
    → Data is already structurally safe
    → CAN use perturbation methods for added deniability + utility preservation
    → Rules LR1-LR4 select based on variable types
```

### Without ReID

When ReID is unavailable (e.g., no QIs selected yet), the system falls back to **uniqueness-based rules** (HR1-HR5) that use `uniqueness_rate = unique_combinations / n_records` as a proxy. These are less accurate because uniqueness doesn't capture risk distribution shape.

## Column Types in SDC

Before selecting a protection method, understand the three column types:

| Type | Examples | Action |
|------|----------|--------|
| **Direct Identifiers** | name, email, SSN, phone | **EXCLUDE** - auto-detected with warning banner |
| **Quasi-Identifiers (QIs)** | age, gender, zipcode, occupation | **PROTECT** - apply SDC methods |
| **Sensitive Attributes** | income, diagnosis, salary | **KEEP** - protected via QI anonymization |

```python
from sdc_engine.sdc.detection import auto_detect_direct_identifiers
from sdc_engine.sdc.sdc_utils import detect_quasi_identifiers_enhanced

# Step 1: Check for direct identifiers (auto-detected in Configure view)
direct_ids = auto_detect_direct_identifiers(data)
if direct_ids:
    print(f"WARNING: Remove these columns: {list(direct_ids.keys())}")
    data = data.drop(columns=list(direct_ids.keys()))

# Step 2: Detect quasi-identifiers
qis = detect_quasi_identifiers_enhanced(data)
```

## Quick Start — UI Workflow

In the Panel application, method selection happens automatically:

1. **Configure tab** → Select QIs → Click "Calculate ReID Risk"
2. **Method Recommendation** panel appears with primary method, confidence, and reasoning
3. **Protect tab** → Method is auto-selected, parameters pre-filled
4. Click **Apply Protection** to execute

## Quick Start — Programmatic

```python
from sdc_engine.sdc.selection.pipelines import select_method_suite

# Build features dict (same as Configure view does internally)
features = {
    'data_type': 'microdata',
    'n_continuous': 3, 'n_categorical': 5,
    'continuous_vars': ['age', 'income', 'hours'],
    'categorical_vars': ['gender', 'region', 'education', 'occupation', 'marital'],
    'quasi_identifiers': ['age', 'gender', 'region', 'education'],
    'n_qis': 4, 'n_records': 5000,
    'has_reid': True, 'reid_50': 0.08, 'reid_95': 0.25, 'reid_99': 0.45,
    'risk_pattern': 'tail',
    'high_risk_rate': 0.05, 'high_risk_count': 250,
    'uniqueness_rate': 0.12,
    'has_outliers': False, 'skewed_columns': [],
    'has_sensitive_attributes': False,
    'k_anonymity_feasibility': 'feasible',
    'expected_eq_size': 8.5,
}

suite = select_method_suite(features, access_tier='SCIENTIFIC', verbose=True)
print(f"Primary: {suite['primary']}")        # e.g., 'kANON'
print(f"Rule: {suite['rule_applied']}")       # e.g., 'QR5_Moderate_Spread'
print(f"Confidence: {suite['confidence']}")   # 'HIGH', 'MEDIUM', 'LOW'
print(f"Fallbacks: {[f[0] for f in suite['fallbacks']]}")
```

## Protection Context → Access Tier

The protection context selected in Configure maps to rules engine access tier:

| Context | Access Tier | k_min | reid_95_max | info_loss_max |
|---------|-------------|-------|------------|---------------|
| Public release | PUBLIC | 10 | 1% | 15% |
| Scientific use | SCIENTIFIC | 5 | 5% | 10% |
| Secure environment | SECURE | 3 | 10% | 8% |
| Regulatory compliance | PUBLIC | 5 | 3% | 12% |

> **Note:** The Streamlit UI exposes three contexts (Public Release, Scientific Use, Secure Environment). Regulatory Compliance is available programmatically via `config.py`.

Access tier affects fallback selection: PUBLIC tier adds aggressive kANON fallbacks (k=7, k=10), SECURE tier adds utility-preserving fallbacks (PRAM, NOISE).

## ReID-Based Selection

When no specific goal is provided, the system uses **ReID (Re-identification Risk)** metrics to choose the optimal method based on risk distribution patterns.

### Risk Patterns Detected

| Pattern | Description | Typical Method |
|---------|-------------|----------------|
| `severe_tail` | Few records dominate risk (ReID_99 >> ReID_50) | LOCSUPR |
| `tail` | Moderate tail risk | LOCSUPR or kANON |
| `uniform_high` | Most records have high risk | kANON (k=10) |
| `widespread` | Risk spread across many records | kANON (structural protection) |
| `bimodal` | Two distinct risk groups | kANON |
| `moderate` | Moderate overall risk | kANON |
| `uniform_low` | Uniformly low risk | Default rules |

### ReID Rules (QR0-QR10)

The system uses 12 named rules for method selection based on risk patterns. Rules are evaluated in order — **first match wins**.

| Rule | Pattern | ReID Characteristics | Method |
|------|---------|---------------------|--------|
| **QR0_K_Anonymity_Infeasible** | Infeasible | QI combination space > dataset size | **GENERALIZE_FIRST** |
| **QR1_Severe_Tail_Risk** | Severe tail | ReID_99/ReID_50 > 10x, ReID_99 > 30% | **LOCSUPR (k=5)** |
| **QR2_Heavy_Tail_Risk** | Tail | ReID_95 > 40% | **kANON (k=7)** |
| **QR2_Moderate_Tail_Risk** | Tail | ReID_95 30-40%, ReID_50 < 15% | **LOCSUPR (k=3)** |
| **QR3_Uniform_High_Risk** | Uniform high | ReID_50 > 20% | **kANON (k=10)** |
| **QR4_Widespread_High** | Widespread | ReID_50 > 15%, ReID_95 > 50% | **kANON (k=10)** |
| **QR4_Widespread_Moderate** | Widespread | ReID_50 > 15%, ReID_95 ≤ 50% | **kANON (k=7)** |
| **QR5_Moderate_Spread** | Moderate | ReID_95 > 20%, ReID_50 < 10% | **kANON (k=5)** |
| **QR6_Bimodal_Risk** | Bimodal | Mean >> Median | **kANON (k=5)** |
| **QR7_Many_High_Risk** | Many high-risk | >10% records at risk >20% | **kANON (k=5)** |
| **QR8_Mild_Categorical** | Mild categorical | 5% < ReID_95 ≤ 10%, categorical-only | **PRAM (p=0.20)** |
| **QR8_Mild_Risk** | Mild | 5% < ReID_95 ≤ 20% | **kANON (k=3 or k=5)** |
| **QR9_Moderate_Risk_Continuous** | Moderate + continuous | 10% < ReID_95 ≤ 20%, continuous ≥ categorical | **RANKSWAP (p=10, R0=0.95)** |
| **QR10_Low_Risk_Mixed** | Low risk + mixed | ReID_95 ≤ 10%, mixed variable types | **RECSWAP (swap_rate=0.05)** |

### K-Anonymity Feasibility Check (QR0)

Before applying risk rules, the system checks if k-anonymity is feasible:

```
expected_eq_size = n_records / product(qi_cardinalities)

If expected_eq_size < 3:
  → QR0: INFEASIBLE — recommend preprocessing (generalize/bin) first
  → Suggest removing highest-cardinality QI
```

### Structural vs Perturbation Methods

**Structural Methods** (kANON, LOCSUPR):
- Create equivalence classes where multiple records share the same QI values
- **DO reduce ReID** by ensuring no record is uniquely identifiable
- Use when ReID reduction is the goal

**Perturbation Methods** (PRAM, NOISE, RANKSWAP, RECSWAP):
- Modify values but don't change the uniqueness structure
- **DO NOT reduce ReID** because each record may still be unique
- Useful for preserving statistical properties (means, distributions, correlations)
- Best when data already has low ReID and utility is priority
- RANKSWAP: rank-based value swapping for numeric data — preserves rank correlations
- RECSWAP: record swapping between similar records — preserves marginal distributions

**Important**: Perturbation methods (PRAM, NOISE, RANKSWAP, RECSWAP) do NOT reduce ReID because they don't create equivalence classes. Only structural methods (kANON, LOCSUPR) reduce re-identification risk.

### Rule Priority (First Match Wins)

Method selection follows this priority order:

1. **Pipeline Rules (P1-P6)** — Multi-method pipelines for complex scenarios
2. **Data Structure Rules (DS1-DS3)** — Tabular format detection
3. **ReID Risk Rules (QR0-QR9)** — Risk distribution patterns
4. **Low-Risk Structure Rules (LR1-LR4)** — Variable types for low-risk data
5. **Distribution Rules (DP1-DP3)** — Outliers, skewness
6. **Default Rules** — Final fallbacks

## Pipeline Rules (P1-P6)

Multi-method pipelines trigger when single methods are demonstrably insufficient:

| Rule | Condition | Pipeline | Reason |
|------|-----------|----------|--------|
| **P1** | ≥2 continuous + ≥2 categorical + outliers + high cardinality | NOISE → kANON | Mixed variables with dual risk |
| **P2a** | ReID_50 > 15% + ReID_99 > 70% + categorical-dominant | PRAM → LOCSUPR | Widespread + extreme tail |
| **P2b** | 10% < ReID_95 < 25% + high_risk_rate > 15% | kANON → LOCSUPR | Moderate + high-risk subgroup |
| **P3** | ≥3 high-cardinality QIs | kANON → LOCSUPR | Vast combination space |
| **P4** | ≥2 skewed + sensitive attributes | kANON → PRAM | Skewed + rare + sensitive |
| **P5** | Sparse data (density < 5), mixed types | NOISE → PRAM | Sparse mixed dataset |

## Low-Risk Structure Rules (LR1-LR4)

Applied **only** when ReID_95 ≤ 5% (data is already low risk). These select perturbation methods for utility preservation:

| Rule | Condition | Method |
|------|-----------|--------|
| **LR1** | Continuous-only + outliers | NOISE |
| **LR2** | Continuous-only | NOISE |
| **LR3** | Categorical-only | PRAM |
| **LR4** | Categorical-dominant mixed | PRAM |

## SDC Engine Integration: Importance Weights for LOCSUPR

When SDC Engine risk analysis has been performed, the backward elimination order is converted to importance weights for LOCSUPR:

```
SDC Engine elimination order → LOCSUPR importance weights:
  Variable eliminated 1st → weight 1 (suppress FIRST, highest risk)
  Variable eliminated 2nd → weight 2
  Variable eliminated 3rd → weight 3
  Variable eliminated last → weight N (suppress LAST, preserve)
```

**Why this works**: Variables eliminated first during SDC Engine backward elimination contribute the most to re-identification risk across all variable combinations. In LOCSUPR, lower importance weight means the variable is suppressed first when trying to achieve k-anonymity. This concentrates information loss on the most dangerous variables while preserving safer ones.

**Data flow**: Configure view extracts weights from `riskCalcInteractor.steps_df` → passes to Protect view → injected into LOCSUPR params at apply time.

Both Python and R implementations of LOCSUPR accept `importance_weights`:
- **Python**: inverts weights to create suppression priority ranking
- **R**: sorts weights and passes to sdcMicro's `localSuppression()`

## Method Suite with Automatic Fallbacks

The `select_method_suite()` function returns a complete suite with fallbacks:

```python
from sdc_engine.sdc.selection.pipelines import select_method_suite

suite = select_method_suite(features, access_tier='PUBLIC')

# Suite contains:
# - primary: Best method to try first
# - primary_params: Parameters for primary method
# - reid_fallback: Stronger method if ReID target not met
# - utility_fallback: Weaker method if utility target not met
# - fallbacks: List of [(method, params), ...] ordered by priority
# - pipeline: Multi-method pipeline if needed
# - rule_applied: Which rule triggered (e.g., 'QR5_Moderate_Spread')
# - confidence: 'HIGH', 'MEDIUM', 'LOW'
# - reason: Human-readable explanation
# - use_pipeline: bool — whether multi-method pipeline is recommended
```

### Fallback Logic

When the primary method doesn't meet the ReID target:

1. **Try PRIMARY method** with recommended parameters
2. **If ReID fails but utility OK** → use `reid_fallback` (stronger method)
3. **If utility fails but ReID OK** → use `utility_fallback` (weaker method)
4. **If both fail** → try `reid_fallback` first, then `utility_fallback`
5. **If all fail** → escalate to manual review

**Override rule**: If the recommended method is a perturbation method (PRAM, NOISE) but a ReID target is set, the system automatically overrides to kANON, since perturbation methods cannot reduce ReID.

## Decision Matrix

Quick reference for method selection:

| Data Type | Risk Pattern | Variable Types | Recommended |
|-----------|--------------|----------------|-------------|
| Microdata | K-anonymity infeasible | Any | GENERALIZE first |
| Microdata | Severe tail | Any | LOCSUPR (k=5) |
| Microdata | Heavy tail (>40%) | Any | kANON (k=7) |
| Microdata | Uniform high | Any | kANON (k=10) |
| Microdata | Widespread high | Any | kANON (k=7-10) |
| Microdata | Moderate spread | Any | kANON (k=5) |
| Microdata | Bimodal | Any | kANON (k=5) |
| Microdata | Many high-risk | Any | kANON (k=5) |
| Microdata | Mild (categorical) | Categorical only | PRAM (p=0.20) |
| Microdata | Mild | Mixed/continuous | kANON (k=3-5) |
| Microdata | Low risk | Continuous + outliers | NOISE |
| Microdata | Low risk | Continuous only | NOISE |
| Microdata | Low risk | Categorical only | PRAM |
| Microdata | Low risk | Mixed (cat dominant) | PRAM |
| Microdata | Moderate | Continuous + correlated | RANKSWAP (p=10, R0=0.95) |
| Microdata | Low risk | Mixed types | RECSWAP (swap_rate=0.05) |

## Risk-Based Preprocessing Aggressiveness

When SDC Engine risk score is available, preprocessing aggressiveness is auto-tuned:

| SDC Engine Risk | Level | Default Utility Threshold | Behavior |
|---|---|---|---|
| > 0.30 | Aggressive | 60% | Enable all preprocessing steps |
| 0.10 - 0.30 | Moderate | 70% | Balanced preprocessing |
| < 0.10 | Light | 85% | Minimal preprocessing, preserve utility |

## Configuration

Method defaults and protection thresholds are in `sdc_engine/sdc/config.py`:

```python
from sdc_engine.sdc.config import (
    PROTECTION_THRESHOLDS,
    METHOD_INFO,
    DIRECT_IDENTIFIER_KEYWORDS,
    DIRECT_IDENTIFIER_PATTERNS,
)

# Protection thresholds by context
PROTECTION_THRESHOLDS['scientific_use']
# {'k_min': 5, 'reid_95_max': 0.10, 'reid_99_max': 0.20, 'info_loss_max': 0.25, ...}

# Method display info
METHOD_INFO['LOCSUPR']
# {'name': 'Local Suppression', ...}
```

## Module Reference

| Module | Key Functions | Role |
|--------|---------------|------|
| `sdc/selection/pipelines.py` | `pipeline_rules()`, `select_method_suite()` | Pipeline + orchestration |
| `sdc/selection/rules.py` | `reid_risk_rules()`, `data_structure_rules()`, `low_risk_structure_rules()`, `distribution_rules()`, `default_rules()` | Individual rule sets |
| `sdc/selection/features.py` | `extract_data_features_with_reid()` | Feature extraction (legacy imports) |
| `sdc/config.py` | `PROTECTION_THRESHOLDS`, `METHOD_INFO` | Configuration constants |
| `sdc/metrics/reid.py` | `calculate_reid()`, `classify_risk_pattern()` | ReID metrics |
| `sdc/LOCSUPR.py` | `apply_locsupr(importance_weights=...)` | Local suppression with weights |

## See Also

- [Pipeline Architecture](sdc_pipeline_architecture.md) — Full system architecture + SDC Engine integration
- [User Guide](user_guide.md) — End-to-end user workflow documentation
