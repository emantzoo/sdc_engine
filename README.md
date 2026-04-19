# SDC Engine

Statistical Disclosure Control (SDC) toolkit for microdata anonymization. Provides a Streamlit web application and a Python library for automated privacy protection of individual-level datasets.

## What It Does

SDC Engine takes a raw microdata file (CSV/Excel), automatically classifies columns, assesses re-identification risk, and applies protection methods to produce an anonymized dataset safe for release or analysis.

The workflow follows five phases:

1. **Upload** -- load your dataset
2. **Classify** -- auto-detect identifiers, quasi-identifiers (QIs), and sensitive columns
3. **Configure** -- set protection context (public/scientific/secure), risk metric, per-QI treatment levels
4. **Protect** -- run preprocessing + protection with automatic method selection or manual override
5. **Download** -- export protected data + audit report

## Features

- **Smart Classification** -- dual-signal fusion of keyword matching and risk contribution analysis; supports English and Greek column names
- **Risk Assessment** -- ReID percentile analysis (50th/95th/99th), k-anonymity checks, uniqueness rates, structural risk scoring
- **4 Protection Methods** -- kANON (k-anonymity), LOCSUPR (local suppression), PRAM (post-randomization), NOISE (noise addition)
- **30+ Selection Rules** -- across 8 tiers (pipeline, data structure, ReID risk, low-risk, distribution, heuristic, default) with first-match-wins evaluation
- **AI-Assisted Mode** -- optional LLM review of rules engine recommendations (Cerebras or OpenAI), with safety constraints preventing protection reduction
- **Preprocessing Pipeline** -- adaptive generalization with tier escalation, top/bottom coding, rare category merging, type-aware routing (dates, ages, geographic codes, numerics)
- **Retry Engine** -- automatic parameter escalation and method fallback when targets are not met
- **Scenario Comparison** -- compare up to 4 protection configurations side-by-side
- **sdcMicro Toggle** -- optional R/sdcMicro backend for LOCSUPR and NOISE (sidebar toggle shows availability)

## Protection Methods

| Method | Type | Best For | Zero Suppression |
|--------|------|----------|:---:|
| **kANON** | Structural | Mixed QIs, high risk, regulatory compliance | No |
| **LOCSUPR** | Structural | Tail risk concentrated in 1-2 QIs | No (cells only) |
| **PRAM** | Perturbative | Categorical QIs, low-moderate risk | Yes |
| **NOISE** | Perturbative | Numeric QIs, distribution preservation | Yes |

**Structural methods** (kANON, LOCSUPR) create equivalence classes and reduce ReID. **Perturbation methods** (PRAM, NOISE) modify values for deniability but do not change uniqueness structure. See the [User Guide, Section 7](docs/user_guide.md#7-available-protection-methods) for parameter tables and detailed descriptions.

## Quick Start

```bash
# Install dependencies
pip install poetry
poetry install

# Run the Streamlit app
streamlit run streamlit_app/app.py
```

The app runs at `http://localhost:8501`. Enter your Cerebras API key in the sidebar to enable AI-assisted modes.

## Programmatic Usage

```python
from sdc_engine.sdc import apply_kanon, apply_pram, apply_noise, apply_locsupr
from sdc_engine.sdc.metrics import calculate_reid, calculate_utility_metrics

# Apply k-anonymity
result = apply_kanon(data, quasi_identifiers=['age', 'gender', 'region'], k=5)

# Assess risk
reid = calculate_reid(data, quasi_identifiers=['age', 'gender', 'region'])
print(f"ReID95: {reid['reid_95']:.2%}")

# Check utility
utility = calculate_utility_metrics(original_data, protected_data, quasi_identifiers)
print(f"Utility: {utility['overall_utility']:.0%}")
```

## Project Structure

```
sdc_engine/                  # Core library
  entities/                  # Domain entities
    dataset/                 #   Dataset abstraction (BaseDataset, PdDataset)
    algorithms/              #   Algorithm interfaces (ReID risk)
  interactors/               # Use-case orchestration
    sdc_protection.py        #   SDCProtection class (apply_method, smart protection)
    sdc_preprocessing.py     #   Preprocessing orchestrator
    sdc_detection.py         #   Column detection orchestrator
    risk_calculation.py      #   Risk calculation orchestrator
    load_dataset.py          #   Dataset loading
  sdc/                       # SDC methods and modules
    kANON.py                 #   k-Anonymity (generalization + suppression)
    LOCSUPR.py               #   Local suppression
    PRAM.py                  #   Post-randomization method
    NOISE.py                 #   Noise addition
    GENERALIZE.py            #   Adaptive generalization with tier escalation
    config.py                #   Rules, thresholds, method defaults
    sdc_preprocessing.py     #   Type-aware preprocessing pipeline
    llm_method_config.py     #   AI-assisted method selection prompt + validation
    llm_classify.py          #   AI-assisted column classification
    llm_assistant.py         #   LLM API wrapper
    detection/               #   Column classification (identifiers, QIs, sensitive)
      column_types.py        #     Structural + semantic type detection
      qi_detection.py        #     Quasi-identifier detection
    metrics/                 #   ReID, risk, utility, ML utility
      reid.py                #     ReID percentile calculations
      risk.py                #     Disclosure risk metrics
      utility.py             #     Information loss + utility scoring
      risk_metric.py         #     Pluggable risk metric (ReID95/k-anon/uniqueness)
      ml_utility.py          #     ML-based utility evaluation
    preprocessing/           #   QI-level preprocessing
      diagnose.py            #     Data diagnostics
      qi_handler.py          #     Per-QI treatment handler
    selection/               #   Rules engine, pipelines, fallback logic
      rules.py               #     30+ named selection rules
      pipelines.py           #     Multi-method pipeline orchestration
      features.py            #     Feature extraction for rule matching
streamlit_app/               # Streamlit web UI
  app.py                     #   Main entry point + sidebar (API key, sdcMicro toggle)
  state.py                   #   Session state defaults
  components.py              #   Shared UI components
  pages/
    1_Upload.py              #   Dataset upload
    2_Configure.py           #   Classification + risk + settings
    3_Protect.py             #   Protection execution + results
    4_Download.py            #   Export protected data
docs/                        # Documentation
data/                        # Local test datasets (gitignored)
tests/                       # Test suite
  data/                      #   Sample datasets (adult, ESS, declarations)
tools/
  classify_columns.py        # Standalone column classifier CLI
```

## Protection Modes

The Protect page offers four modes:

| Mode | Description |
|------|-------------|
| **Auto** | Rules engine selects method + parameters automatically based on data features and risk |
| **Smart Combo** | Adaptive multi-method pipeline with automatic escalation and fallback |
| **AI** | LLM reviews rules engine recommendation; can override with high confidence (requires API key) |
| **Manual** | User selects method and parameters directly |

## Method Selection Logic

The rules engine evaluates features in priority order (first match wins):

1. **Pipeline rules** (P1-P6) -- multi-method combos for complex scenarios
2. **Small dataset / structural risk rules** (HR6, SR3) -- special cases
3. **Risk concentration rules** (RC1-RC4) -- based on backward elimination priorities
4. **Categorical-aware rules** (CAT1-CAT2) -- PRAM for categorical-dominant data
5. **ReID risk rules** (QR1-QR8) -- risk pattern + severity tier
6. **Low-risk rules** (LOW1-LOW3) -- perturbation for already-safe data
7. **Distribution rules** (DP1-DP3) -- outliers, skewness
8. **Default rules** -- catch-all fallbacks

Full specification: [Smart Rules Reference](docs/smart_rules_complete.md)

## Requirements

- Python 3.11+
- Core: pandas, numpy, scikit-learn, streamlit, plotly, openpyxl
- Optional: rpy2 + sdcMicro (R backend for LOCSUPR/NOISE)
- Optional: Cerebras or OpenAI API key (AI-assisted mode)

## Documentation

| Document | Contents |
|----------|----------|
| [User Guide](docs/user_guide.md) | End-to-end workflow, method descriptions, parameter tables, decision tree |
| [Method Selection Guide](docs/Method_Selection_Guide.md) | ReID-based selection, rules reference, fallback logic, decision matrix |
| [Smart Rules Reference](docs/smart_rules_complete.md) | Complete rule specification across all 5 pipeline phases |
| [Pipeline Architecture](docs/sdc_pipeline_architecture.md) | System architecture, data flow, component interactions |
| [Empirical Validation](docs/empirical_validation_checklist.md) | Heuristic thresholds and validation checklist |
| [Build Guide](docs/build_guide.md) | Packaging for standalone distribution |

## Author

Eirini Mantzouni
