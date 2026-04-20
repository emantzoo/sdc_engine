# SDC Engine: User Guide

## Streamlit Application — Statistical Disclosure Control

---

> **How to read this guide:** Sections 1–6 cover everyday use (upload, configure, protect, download, interpret results). Sections 7–8 explain method selection for advanced users. Section 9 covers auto-classification logic. Section 10 is troubleshooting. Section 11 is the technical reference.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
   - [2.1 Installation](#21-installation)
   - [2.2 Launching the Application](#22-launching-the-application)
   - [2.3 Uploading Data](#23-uploading-data)
3. [Configure Protection](#3-configure-protection)
   - [3.1 Classification Modes](#31-classification-modes)
   - [3.2 Suggestion Details](#32-suggestion-details)
   - [3.3 Role Assignment Table](#33-role-assignment-table)
   - [3.4 Protection Settings](#34-protection-settings)
   - [3.5 Risk Preview](#35-risk-preview)
4. [Preprocess & Protect](#4-preprocess--protect)
   - [4.1 Preprocessing Plan](#41-preprocessing-plan)
   - [4.2 Protection Mode](#42-protection-mode)
   - [4.3 Results Display](#43-results-display)
   - [4.4 Privacy Metrics](#44-privacy-metrics)
   - [4.5 Utility Report](#45-utility-report)
5. [Download & Export](#5-download--export)
6. [Understanding the Results](#6-understanding-the-results)
   - [6.1 Three Risk Metrics](#61-three-risk-metrics)
   - [6.2 Privacy Metrics Explained](#62-privacy-metrics-explained)
   - [6.3 Interpreting Utility](#63-interpreting-utility)
   - [6.4 QI Suppression Warnings](#64-qi-suppression-warnings)
7. [Available Protection Methods](#7-available-protection-methods)
8. [Method Selection Decision Tree](#8-method-selection-decision-tree)
   - [8.1 The Role of ReID](#81-the-role-of-reid-in-method-selection)
   - [8.2 Structural vs Perturbation Methods](#82-structural-vs-perturbation-methods-why-reid-matters)
   - [8.3 The Decision Tree](#83-the-decision-tree)
   - [8.4 Pipeline Rules](#84-pipeline-rules-multi-method-sequences)
   - [8.5 Fallback Logic](#85-fallback-logic)
   - [8.6 Protection Context](#86-how-protection-context-affects-selection)
   - [8.7 SDC Engine Integration](#87-how-sdc-engine-informs-method-selection)
   - [8.8 Quick Reference](#88-quick-reference-when-to-use-each-method)
9. [Auto-Classification & QI Detection](#9-auto-classification--qi-detection)
   - [9.1 Backward Elimination](#91-backward-elimination)
   - [9.2 Keyword-Based Scoring](#92-keyword-based-scoring)
   - [9.3 QI Fusion](#93-qi-fusion)
   - [9.4 Low-Cardinality Guard](#94-low-cardinality-guard)
   - [9.5 Concentrated-Risk Fallback](#95-concentrated-risk-fallback)
   - [9.6 Sensitive Column Detection](#96-sensitive-column-detection)
   - [9.7 LLM Classification](#97-llm-classification-cerebrasqwen)
10. [Troubleshooting](#10-troubleshooting)
11. [Technical Reference](#11-technical-reference)
- [Appendix: SDC Protection — End-to-End Reference](#appendix-sdc-protection--end-to-end-reference)

---

## 1. Introduction

This guide covers the SDC Engine Streamlit application, which provides a complete statistical disclosure control workflow:

- **Upload** your dataset (CSV or Excel)
- **Configure** column roles — quasi-identifiers, sensitive, unassigned — using rules-based auto-classification, AI (Cerebras LLM), or manual assignment
- **Protect** data using 4 core methods: kANON, LOCSUPR, PRAM, NOISE — with type-aware preprocessing and automated method selection
- **Download** the protected dataset and a comprehensive HTML report

> **Key Concept: Three Risk Metrics**
>
> | Metric | Scope | Answers |
> |--------|-------|---------|
> | **Overall Risk** | All variables | Which variables contribute most to re-identification risk? |
> | **Structural Risk** | Selected QIs only | How much does the QI combination as a group enable re-identification? |
> | **ReID** (50/95/99) | Per-record, QI-scoped | Which individual records are at risk? |
>
> All three are needed for effective data protection.

---

## 2. Getting Started

### 2.1 Installation

The application uses Python 3.11+ with Poetry for dependency management.

1. Install Python 3.11+
2. Clone the repository: `git clone https://github.com/emantzoo/SDC_EM.git`
3. Install dependencies: `poetry install`
4. For full SDC support (optional): `poetry install --extras sdc-full`

> **Optional Dependencies**
> - **R + sdcMicro**: enables enhanced Local Suppression and Noise methods with fewer distortions
> - **Polars**: accelerates risk scanning on large datasets (5-10x speedup)
> - These are optional — the tool works fully without them

### 2.2 Launching the Application

```bash
cd streamlit_app
streamlit run app.py
```

Open your browser at `http://localhost:8501` to access the application.

**For AI classification mode** (optional): set the `CEREBRAS_API_KEY` environment variable before launching. Without it, the AI mode button will be hidden and only rules-based and manual classification are available.

### 2.3 Uploading Data

The **Upload** page is the entry point:

1. Click the file uploader or drag a file (`.csv`, `.xlsx`, `.xls`)
2. The page displays basic statistics: row count, column count, file size
3. A preview of the first 100 rows is shown below
4. Once uploaded, navigate to **Configure** using the sidebar

---

## 3. Configure Protection

The **Configure** page is where you assign column roles and set protection parameters.

### 3.1 Classification Modes

Three modes are available via radio buttons:

| Mode | How It Works | When to Use |
|------|-------------|-------------|
| **Manual** | You assign roles directly in the editable table | Full control over every column |
| **Auto (rules)** | Backward elimination + keyword-based auto-classification | Quick start — let the engine decide |
| **AI (Cerebras)** | LLM classification merged with rule-based results | Best quality, needs API key |

**Auto mode** runs two steps:
1. **Backward elimination** — computes per-variable risk contribution by measuring how much ReID drops when each column is removed
2. **Auto-classify** — combines keyword matching (column name patterns for date, geo, age, demographic) with risk contribution scores to assign roles

**AI mode** additionally calls `llm_classify_columns()` via the Cerebras Qwen 235B model, then merges LLM suggestions with rule-based results. The merge keeps the higher-confidence suggestion when they disagree.

### 3.2 Suggestion Details

After running Auto or AI classification, an expander shows the recommendation table:

| Column | Suggested Role | Advice | Confidence | Reason |
|--------|---------------|--------|------------|--------|

Two charts are also available:

- **Variable Importance** — horizontal bar chart showing backward elimination risk contribution per column, color-coded: red (HIGH ≥15%), orange (MED-HIGH ≥8%), amber (MODERATE ≥3%), grey (LOW <3%)
- **Backward Elimination Curve** — line chart showing how risk changes as variables are removed one at a time

### 3.3 Role Assignment Table

An editable table lets you view and modify column roles:

| Column | Type | Unique | Missing % | Role |
|--------|------|--------|-----------|------|

The **Role** column is a dropdown with three options:
- **QI** (quasi-identifier) — columns that could help identify individuals (age, region, date)
- **Sensitive** — analysis variables to preserve (income, health status)
- **Unassigned** — excluded from protection

### 3.4 Protection Settings

Below the role table:

- **Protection Context** — dropdown: Public Release / Scientific Use / Secure Environment. Sets default targets and method aggressiveness.
- **Risk Metric** — dropdown: ReID 95th percentile / k-Anonymity / Uniqueness / l-Diversity. Determines what the protection engine optimizes for.
- **Target** — numeric input for the selected metric (e.g., k=5 for k-Anonymity, 0.05 for ReID95)

| Context | Default k | Default ReID95 Target | Utility Floor |
|---------|-----------|----------------------|---------------|
| **Public Release** | 10 | 1% | 85% |
| **Scientific Use** | 5 | 5% | 90% |
| **Secure Environment** | 3 | 10% | 92% |

### 3.5 Risk Preview

- **Preview Risk** button — runs `calculate_reid()` on selected QIs and shows:
  - Risk badge (color-coded: LOW, MODERATE, HIGH, VERY HIGH)
  - Metric cards: ReID 50th/95th/99th percentile, high-risk rate

- **Confirm Configuration** button — finalizes column roles, recovers numeric/datetime types from text, and unlocks the Protect page

---

## 4. Preprocess & Protect

The **Protect** page handles both preprocessing and protection in a single workflow.

### 4.1 Preprocessing Plan

An auto-generated editable table shows the recommended preprocessing for each QI:

| Column | Action | Param | Reason | Cardinality Before | Cardinality After |
|--------|--------|-------|--------|-------------------|------------------|

Actions vary by detected column type:

| Column Type | Available Actions | Example |
|-------------|------------------|---------|
| **Date** | Truncate to year/quarter/month, None | Birth date (500 unique) → year (45 unique) |
| **Age** | Bin (5-year ranges), Round, None | Age (80 unique) → 5-year bins (17 groups) |
| **Geographic** | Coarsen (top-N grouping), None | Municipality (886 unique) → top 5 regions |
| **Numeric** | Bin (quantile), Round, None | Income (9433 unique) → 10 quantile bins |
| **Categorical** | Generalize (top-K), None | Occupation (50 unique) → top 15 + Other |

Two buttons:
- **Apply Preprocessing** — executes the plan, shows before/after cardinality per column
- **Skip Preprocessing** — bypasses preprocessing, uses raw data for protection

### 4.2 Protection Mode

Four modes via radio buttons:

| Mode | Description |
|------|-------------|
| **Auto (rules engine)** | Calls `run_rules_engine_protection()` — automatic method selection with escalation and fallbacks |
| **Smart Combo (adaptive)** | Wraps the rules engine in a GENERALIZE tier escalation (light → moderate → aggressive → very aggressive) |
| **AI (Cerebras)** | Gets LLM recommendation via `llm_select_method()`, shows reasoning and confidence, user can accept or override |
| **Manual** | User picks method (kANON/LOCSUPR/PRAM/NOISE) and parameters (k, p_change, magnitude) with pre-populated recommendations |

**Manual mode parameters:**

| Method | Parameter | Range | Default |
|--------|-----------|-------|---------|
| kANON | k | 2–50 | 5 |
| LOCSUPR | k | 2–50 | 3 |
| PRAM | p_change | 0.01–0.50 | 0.20 |
| NOISE | magnitude | 0.01–0.50 | 0.10 |

Click **Run Protection** to execute.

### 4.3 Results Display

After protection completes:

1. **Result badge** — Success/Fail with method name (e.g., "kANON succeeded")
2. **Metric cards with deltas** — ReID 95th (before → after), Utility score, Method name
3. **Risk badge** — color-coded risk level after protection
4. **Before/After risk histogram** — Plotly overlay showing per-record risk distribution shift
5. **Sample data comparison** — first 10 rows of original vs protected QI columns
6. **Preprocessing applied** — per-column action summary (if preprocessing ran)

### 4.4 Privacy Metrics

A collapsible **Privacy Metrics** expander shows before/after comparison:

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **k-Anonymity** (min group) | Smallest equivalence class size | k ≥ 5 (Scientific Use) |
| **Uniqueness rate** | % of records with unique QI combination | < 5% |
| **Records at risk** | % of records in groups < 5 | < 10% |
| **Avg group size** | Mean equivalence class size | Higher = better |
| **l-Diversity** (min) | Min distinct sensitive values per group | l ≥ 2 |
| **l-Diversity violations** | Groups failing l-diversity target | 0 = ideal |
| **t-Closeness** (max distance) | Max distributional distance of sensitive values | t ≤ 0.30 |

l-Diversity and t-Closeness only appear when sensitive columns are assigned.

### 4.5 Utility Report

A collapsible **Utility Report** expander provides detailed analysis:

- **Per-Variable Utility** — table showing unique values before/after, category overlap, row preservation per column
- **Cross-Tab Benchmark** — means, variances, frequency TVD, correlation preservation
- **Distributional Metrics** — KL divergence, Hellinger distance per variable
- **Information Loss (IL1s)** — per-record distortion measure
- **Auto-Diagnostics** — QI utility comparison, method quality assessment

**QI Over-Suppression Warnings** appear as a yellow warning box when any QI loses >20% of its values during protection, listing affected columns and suppression percentages.

**Protection Details** expander shows: rule applied, reasoning, parameters (JSON), and the last 20 engine log entries.

---

## 5. Download & Export

The **Download** page provides:

1. **Summary cards** — method used, utility score, ReID 95th (after), row count, risk badge
2. **Download Protected CSV** — exports the anonymized dataset as CSV
3. **Download Summary Report (HTML)** — self-contained HTML report with:
   - Executive summary (PASS/FAIL, method, before/after metrics)
   - Dataset overview (file name, date, rows, columns)
   - Before/after comparison table (ReID 95th/99th, high-risk rate, utility)
   - QIs and sensitive columns list
   - Privacy metrics table (k-anonymity, uniqueness, records at risk, l-diversity, t-closeness — before and after)
   - Per-variable utility table (when available)
   - Collapsible appendices: preprocessing metadata, variable importance, AI recommendation (if used), engine log
4. **Start Over** — resets all session state and returns to Upload

---

## 6. Understanding the Results

### 6.1 Three Risk Metrics

| | Overall Risk (all variables) | Structural Risk (QI-scoped) | ReID (per-record) |
|---|---|---|---|
| **Measures** | Per-variable risk contribution via leave-one-out analysis | Dataset-level risk from the QI combination | Individual record re-identification probability |
| **Scope** | All variables (dataset-wide) | Selected QIs only | Per record, QI-scoped |
| **Used For** | QI selection, preprocessing aggressiveness | Context indicator alongside ReID | Method selection, protection verification |

**ReID95** is the primary protection metric — it measures the re-identification probability that 95% of records fall below.

| ReID95 Range | Risk Level | Meaning |
|-------------|-----------|---------|
| < 5% | **LOW** | Most records well-protected. Perturbation may suffice. |
| 5–30% | **MODERATE** | Significant risk. Structural methods (kANON, LOCSUPR) recommended. |
| > 30% | **HIGH** | Many records identifiable. Aggressive protection required. |

**Key decision boundary:** ReID95 > 5% → structural methods required (kANON, LOCSUPR). ReID95 ≤ 5% → perturbation may suffice (PRAM, NOISE).

**Reading them together:**

| Scenario | Overall Risk | ReID95 | Structural Risk | Interpretation |
|----------|-------------|-------|-----------------|----------------|
| Easy dataset | Low (< 20%) | Low (< 5%) | Low (< 30%) | Minimal intervention needed |
| Risky columns, good QIs | High (50%+) | Moderate (20–30%) | Low–Moderate | QI selection is manageable |
| Hard anonymization | High (50%+) | High (50%+) | High (60%+) | Aggressive methods and utility loss expected |

### 6.2 Privacy Metrics Explained

> **The Privacy Trio:** k-Anonymity, l-Diversity, and t-Closeness form the "privacy trio" from disclosure control literature.

| Metric | Protects Against | How It Works |
|--------|-----------------|-------------|
| **k-Anonymity** | Identity disclosure | Every record shares its QI combination with ≥k−1 others. k=5 means groups of at least 5. |
| **l-Diversity** | Attribute disclosure | Each equivalence class has ≥l distinct sensitive values. Prevents "everyone in group X has disease Y" attacks. |
| **t-Closeness** | Skewness attacks | Sensitive value distribution within each group must be close to the overall distribution (distance ≤ t). |

**Additional metrics:**
- **Uniqueness rate** — % of records with a one-of-a-kind QI combination (group size = 1). Lower is safer.
- **Records at risk** — % of records in groups smaller than k. Should be 0% after successful protection.
- **Avg group size** — mean equivalence class size. Higher means stronger anonymity.
- **Disclosure risk** — composite metric combining uniqueness and small-group exposure.

### 6.3 Interpreting Utility

The composite utility score blends three components:

| Component | Weight | What to check if low |
|---|---|---|
| **Sensitive utility** (50%) | Are analysis variables preserved? | Check per-variable utility for sensitive columns |
| **QI utility** (20%) | How much resolution did QIs lose? | Low QI utility is normal and expected |
| **Relationship preservation** (30%) | Can cross-tabulations still be done? | Check relationship check results |

| Composite | Meaning | Action |
|---|---|---|
| ≥ 85% | Excellent — protected data is analytically similar | None needed |
| 70–85% | Good — minor utility loss | Review specific weak relationships |
| 60–70% | Moderate — some analytical quality lost | Consider less aggressive method |
| < 60% | Significant distortion | Reduce QI count or accept lower k |

### 6.4 QI Suppression Warnings

When a protection method suppresses (replaces with NaN) too many values in a QI column, a warning appears. The protection engine has multi-layered guards:

| Guard | Threshold | Action |
|-------|-----------|--------|
| **LOCSUPR per-column cap** | 60% | Restores column, raises error |
| **Engine post-method check** | 40% | Sets `qi_over_suppressed` flag |
| **Engine retry guard** (×4 checkpoints) | 40% flag | Rejects method, tries fallback |
| **UI warning** | >20% | Yellow warning box with affected columns |

If you see suppression warnings, consider:
- Removing the affected column from QIs (it may not contribute much to re-identification)
- Preprocessing it more aggressively before protection
- Using a perturbation method (PRAM/NOISE) instead of a structural method

---
## 7. Available Protection Methods

The application provides four microdata protection methods. Tabular methods are not supported — the tool works exclusively with individual-level records.

#### kANON — k-Anonymity

**Simple:** Groups records so that every person shares their identifying characteristics with at least k−1 others. If k = 5, every combination of quasi-identifier values appears in at least 5 records, making it impossible to single out any individual.

**How it works:** Generalizes values (e.g., age 37 → "30–39", city "Athens" → "Attica") and suppresses outlier cells that cannot be grouped. Uses ARX-inspired smart hierarchy builders that auto-select the right generalization strategy per column type (interval binning for numerics, date truncation for temporals, character masking for postal codes, frequency-based grouping for categoricals). The greedy generalization loop uses info-loss-aware scoring to prefer cheap generalizations first. When an l-diversity target is set, the loop enforces both k-anonymity and l-diversity simultaneously.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| k | 3 | 2–10 | Higher k = stronger privacy, more generalization |
| bin_size | 10 | 1–100 | Numeric binning width (e.g., 10 creates 0–9, 10–19, ...) |
| max_suppression_rate | 0.10 | 0–0.50 | Maximum fraction of records that can be suppressed |
| strategy | hybrid | generalization / suppression / hybrid / beam / recursive | Which technique to apply (see below) |

**Strategies:**
- **generalization** — greedy iterative generalization, then suppression fallback if needed
- **suppression** — suppression only (no generalization)
- **hybrid** (default) — generalization first, then suppression for residual violations
- **beam** — beam search over the generalization lattice (explores multiple paths simultaneously, finds lower info-loss solutions — §11.26)
- **recursive** — generalization + suppression + recursive local recoding to recover suppressed records (§11.27)

**Hybrid suppression (default):** kANON uses a two-phase suppression strategy that preserves more QI data than traditional full-row suppression:
1. **Phase 1 — Targeted:** Iteratively suppresses only the highest-cardinality QI for violation groups. After each round, equivalence classes are re-computed — previous suppressions may have merged groups. This touches more rows but blanks only 1 QI per row per round.
2. **Phase 2 — Full-row cleanup:** For residual violations, suppresses all QIs for the smallest groups (highest risk) within the remaining budget.

This preserves significantly more analytical value: partial suppression (1 QI per row) keeps 5/6 of the QI data for affected rows vs full-row suppression which loses all QI data.

**Best for:** Categorical data, regulatory compliance requiring proven k-anonymity guarantees.

---

#### LOCSUPR — Local Suppression

**Simple:** Removes only the specific cells (not entire rows) that prevent k-anonymity. Keeps as much data as possible while achieving the same privacy guarantee as kANON.

**How it works:** Identifies records violating k-anonymity and suppresses (replaces with NA) the minimum number of QI cells needed. Uses importance weights to protect high-value columns — columns with higher weight are suppressed last.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| k | 3 | 2–10 | Minimum group size |
| importance_weights | equal | per-column | Higher weight = less likely to be suppressed |

**R integration:** If R + sdcMicro is installed, uses optimal suppression (fewer cells suppressed). Otherwise falls back to a Python heuristic that may suppress more cells.

**Best for:** When you need k-anonymity but want to minimize data loss, especially when some columns are more important than others.

---

#### PRAM — Post-Randomization Method

**Simple:** Randomly changes categorical values according to controlled probabilities. A person's "Athens" might become "Thessaloniki" with some probability, making it impossible to know if any specific value is real.

**How it works:** For each categorical QI, a transition matrix defines the probability that value A becomes value B. In invariant mode (default), the matrices are designed so that marginal distributions are preserved on average — the overall proportions of each category remain approximately unchanged.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| p_change | 0.20 | 0–1.0 | Base probability that any value changes (20% default) |
| pd_min | 0.05 | 0–1.0 | Minimum probability of staying unchanged |
| alpha | 0.50 | 0–1.0 | Balance between global and local distribution preservation |
| invariant | True | True/False | If True, preserves marginal distributions |

**Best for:** Categorical data where you want to preserve statistical distributions for analysis while protecting individual records.

> **Supports attribute disclosure toggle** — can also perturb sensitive columns at lighter strength.

---

#### NOISE — Noise Addition

**Simple:** Adds random "fuzz" to numeric values. An income of €45,000 might become €44,200 or €45,800. The overall statistics (means, distributions) are approximately preserved, but individual values cannot be trusted.

**How it works:** For each numeric QI, random noise drawn from a chosen distribution is added. When R + sdcMicro is available, correlated noise is used, which preserves inter-variable correlations better.

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| magnitude | 0.10 | 0–1.0+ | Noise level (10% of std deviation by default) |
| noise_type | gaussian | gaussian/laplace/uniform/proportional | Distribution shape |
| relative | True | True/False | If True, magnitude is relative to column std dev |
| preserve_sign | True | True/False | Prevents noise from flipping positive to negative |

**Formula (Gaussian):** `protected_value = original + N(0, magnitude × σ)`

**Integer preservation:** Columns that contain only whole numbers (years, counts, prices) are automatically detected before noise is added. After perturbation, these columns are rounded back to integers so that values like year 1957 remain integer (e.g., 1955) rather than producing floating-point artifacts (e.g., 1955.123).

**Best for:** Continuous numeric data (income, measurements, amounts) where statistical distributions matter more than exact values.

> **Supports attribute disclosure toggle** — can also perturb sensitive columns at lighter strength.

---

## 8. Method Selection Decision Tree

This section explains exactly how the system decides which protection method to recommend. The decision tree is driven by ReID metrics — the per-record risk percentiles that measure how exposed individual records are.

### 8.1 The Role of ReID in Method Selection

**ReID (Re-identification Risk)** is the central metric that drives every protection decision. After grouping records by their quasi-identifier values, each record's risk is `1 / group_size`. The system then computes three percentiles across all records:

| Metric | What It Means | Role in Decision |
|--------|---------------|------------------|
| **ReID50** | Median risk — half of all records have risk at or below this | Detects whether risk is widespread (>20%) or concentrated in a tail |
| **ReID95** | 95th percentile — the **primary decision metric** | Drives method selection: determines risk severity tier |
| **ReID99** | 99th percentile — worst-case tail | Detects extreme outliers; ReID99/ReID50 ratio reveals severe tail patterns |

**Without ReID, the system cannot select an optimal method.** It falls back to heuristic rules based on uniqueness rate and dataset size, which are less accurate.

### 8.2 Structural vs Perturbation Methods: Why ReID Matters

This is the most important distinction in SDC:

**Structural methods** (kANON, LOCSUPR) create equivalence classes — groups of records that share identical QI values. They **reduce ReID** because no record in a group of k can be uniquely identified.

**Perturbation methods** (PRAM, NOISE) modify values but **do NOT reduce ReID**. Each record may still be unique after perturbation. They add plausible deniability but not structural anonymity.

**Consequence for the decision tree:**
- If ReID95 > 5% → the system must use a **structural method** (kANON or LOCSUPR) because perturbation cannot fix the risk
- If ReID95 ≤ 5% → the system can use **perturbation methods** for utility preservation, since risk is already acceptably low

### 8.3 The Decision Tree

The rules engine evaluates rules in priority order — **first match wins**. At a high level:

1. **Pipelines** checked first (dynamic builder or legacy CAT2/P4/P5)
2. **Small dataset guard** (HR6: <200 rows → LOCSUPR k=3)
3. **Structural rules** (SR3: near-unique QI with few QIs)
4. **Risk concentration rules** (RC1–RC4) — when per-QI risk data is available and ReID95 >15%
5. **Categorical / temporal / diversity rules** (CAT1, LDIV1, DATE1)
6. **ReID-based rules** (QR0–QR4, MED1) — risk pattern drives method and k/p selection
7. **Low-risk rules** (LOW1–LOW3) — type-based when ReID95 ≤20%
8. **Distribution rules** (DP1–DP4) — outliers, skew, sensitive attributes, integer-coded categoricals
9. **Uniqueness fallback** (HR1–HR5) — when no ReID is available
10. **Defaults** — microdata/categorical/continuous/emergency fallbacks

The key principle: **ReID95 determines whether a structural method (kANON, LOCSUPR) is required** — perturbation methods (PRAM, NOISE) cannot reduce equivalence-class-based risk and are only selected when risk is already low or for categorical/continuous-specific scenarios.

**Three safeguards refine method selection:**

- **Suppression-gated switching**: Rules QR3, QR4, and MED1 check estimated suppression *before* selecting kANON. If k-anonymity at the proposed k would suppress > 25% of records, the rule switches to PRAM (categorical-dominant data) or LOCSUPR (continuous-present data) directly — eliminating the confusing post-selection smart-switch pattern.
- **Dataset-size k modifier**: For datasets in the 200–5,000 row range, the proposed k is reduced before suppression clamping (<500 rows: cap k=3; <1,000: k-2; <2,000: k-2; <5,000: step down one schedule notch). Datasets >= 5,000 are unaffected. This prevents catastrophic suppression on small datasets where structural methods are destructive.
- **Post-preprocessing feature tagging**: After GENERALIZE bins a continuous column (e.g., income → 8 range bins), the column is still counted as *continuous* for method selection — preventing PRAM from being incorrectly selected for binned numeric ranges.

> For the complete rule tables with exact conditions, k values, p_change ranges, and design rationale, see [Appendix §3 — Method Selection Rules](#3-method-selection-rules).

### 8.4 Pipeline Rules (Multi-Method Sequences)

When data characteristics indicate that no single method is sufficient, the system recommends a **pipeline** — applying methods in sequence. Pipeline rules are checked before single-method rules.

Most pipelines are now assembled dynamically based on data features (e.g., continuous outliers → add NOISE, high ReID → add kANON, categorical-dominant → defer to PRAM rules, multi-level geographic → GENERALIZE + kANON). The dynamic builder includes a categorical guard (≥70% categorical defers to CAT1, 50–70% builds DYN_CAT with NOISE+PRAM) and a geographic guard (GEO1 for multi-level geo QIs). Legacy pipelines P4a/P4b and P5 handle skewed+sensitive and sparse mixed dataset edge cases.

> For the full pipeline rule table, dynamic builder logic, triggers, and method parameters, see [Appendix §3 — Method Selection Rules](#3-method-selection-rules).

### 8.5 Fallback Logic

Both Auto-Protect and Smart Combo use the same shared retry engine (`run_rules_engine_protection()`). When the primary method doesn't meet targets, the engine escalates parameters, then tries fallback methods — each with its own escalation schedule, safeguards (over-suppression guard, plateau detection, time budget), and intelligent cross-method starting parameters.

**Smart Combo** wraps this retry loop in a GENERALIZE tier escalation (light → aggressive), rebuilding data features at each tier so method selection adapts to reduced cardinalities.

**Metric-method compatibility filter**: When the risk metric is reid95 and a target is set, perturbative method selections (PRAM, NOISE) are converted to kANON via the retry engine's fallback logic. This is because perturbation reduces uniqueness through value-level changes but does not create equivalence classes, so it cannot meet structural risk targets. Empirical validation under both reid95 and k_anonymity metrics (see `tests/empirical/reports/COMBINED_SUMMARY.md`) confirms this: under k_anonymity, PRAM/NOISE selections never achieve min_k >= target, and the retry engine's natural fallback converges to kANON. The filter is therefore not a special-case override -- it's enforcing a mathematical constraint that would otherwise be discovered one iteration later.

> For the complete retry loop phases, safeguard details, escalation schedules, and fallback order, see [Appendix §4 — Engine Orchestration](#4-engine-orchestration).

### 8.6 How Protection Context Affects Selection

The protection context selected in Configure maps to an access tier via the centralized `CONTEXT_TO_TIER` mapping in `sdc/config.py`. All views (Configure, Protect, Combo) use the same mapping through `get_access_tier()`:

| Context | Access Tier | Effect on Selection |
|---------|-------------|---------------------|
| **Public release** | PUBLIC | Adds aggressive fallbacks (kANON k=7, k=10). Strictest ReID targets (1%). |
| **Scientific use** | SCIENTIFIC | Balanced fallbacks. Moderate ReID targets (5%). |
| **Secure Environment** | SECURE | Utility-preserving fallbacks (PRAM, NOISE). Relaxed thresholds (10%). |
| **Regulatory compliance** | PUBLIC | Same as public release — regulatory standards require strong protection (3%). |

> **Note:** The Streamlit UI exposes only the first three contexts (Public Release, Scientific Use, Secure Environment). `regulatory_compliance` is defined in `sdc/config.py` but is not currently selectable in the UI.

All targets for a context are retrievable via `get_context_targets(context)`, which returns `reid_target`, `utility_floor`, `k_min`, `access_tier`, and `description` in a single call.

### 8.7 How SDC Engine Informs Method Selection

When SDC Engine analysis is available, it enriches the decision tree:

| SDC Engine Output | How It's Used |
|----------------|---------------|
| **Dataset risk score** | Preprocess view auto-tunes aggressiveness (>0.30 = aggressive, 0.10-0.30 = moderate, <0.10 = light) |
| **Elimination order** | Converted to LOCSUPR importance weights — variables eliminated first are suppressed first |
| **Variable importance** | Displayed in Configure Column Analysis table (Risk Freq, Entropy, Med/Max Risk columns), guides QI selection |
| **Exposure score** | Shown in summary banner, provides structural risk context |

### 8.8 Quick Reference: When to Use Each Method

| Method | Use When | ReID Effect | Utility |
|--------|----------|------------|---------|
| **kANON** | ReID95 > 5%, mixed or categorical data | Reduces ReID | Moderate — generalizes values |
| **LOCSUPR** | Severe tail risk, few outlier records | Reduces ReID | High — only suppresses problem cells |
| **PRAM** | ReID95 ≤ 5%, categorical-only or categorical-dominant | No ReID reduction | High — preserves distributions |
| **NOISE** | ReID95 ≤ 5%, continuous-only or continuous + outliers | No ReID reduction | High — preserves means/distributions |

---


## 9. Auto-Classification & QI Detection

The auto-classification engine (`auto_classify()` in `sdc/auto_classify.py`) assigns column roles using a multi-signal fusion approach. This section explains how it works.

### 9.1 Backward Elimination

The first step runs backward elimination to rank variables by re-identification risk contribution:

1. Compute per-record ReID across all columns
2. For each column, measure how much ReID drops when it's removed
3. The column whose removal causes the largest drop contributes the most risk
4. Repeat until 10 columns are ranked (greedy, capped)

Output: `var_priority = {col: (label, contribution_pct)}` — e.g., `{'age': ('HIGH', 22.5)}`.

### 9.2 Keyword-Based Scoring

Column names are matched against keyword dictionaries in both English and Greek:

| Tier | Keywords (examples) | Score |
|------|-------------------|-------|
| **DEFINITE** | age, gender, φύλο, ηλικία, zip, postal, ΤΚ | 0.68+ |
| **PROBABLE** | date, year, ημερομηνία, region, περιφέρεια, municipality | 0.50–0.67 |
| **POSSIBLE** | type, category, τύπος, status, κατάσταση | 0.30–0.49 |

Domain boosters add +0.15 for date/geo matches and +0.10 for demographic matches.

### 9.3 QI Fusion

For each column, the final QI score combines keyword and risk signals:

```
fusion_score = 0.30 × keyword_score + 0.70 × risk_contribution
```

A column becomes a QI candidate when `fusion_score > threshold` (default 0.25). DEFINITE keywords have a floor that ensures they're always classified as QI regardless of risk contribution.

**Non-integer numeric guard**: Continuous float columns (prices, areas, measurements) are excluded from QI classification since they act as analysis variables, not identifiers.

### 9.4 Low-Cardinality Guard

Columns with very few unique values relative to dataset size provide negligible re-identification power and get destroyed by structural protection methods (e.g., 94% suppression in LOCSUPR). A proportional two-tier guard demotes them from QI:

| Tier | Condition | Example |
|------|-----------|---------|
| **Tier 1** | ≤3 unique AND ratio < 1% | 2 unique / 5000 rows → demote |
| **Tier 2** | 4–20 unique AND ratio < 0.05% | 7 unique / 41K rows → demote |

**DEFINITE keywords are never demoted** — gender with 2 values in a 50K dataset stays as QI because it's a textbook quasi-identifier.

Examples:
- 2 unique / 120 rows = 1.7% → **keep** (meaningful in small data)
- 7 unique / 41K rows = 0.017% → **demote** (avg 6K rows per value, no re-id power)
- 7 unique / 1K rows = 0.7% → **keep** (above 0.05%)

### 9.5 Concentrated-Risk Fallback

When backward elimination risk is dominated by 1-2 columns (e.g., an identifier absorbing 100% of risk, or a high-cardinality numeric absorbing 98%), all other columns show ~0% contribution. This makes risk-based QI fusion useless.

**Detection**: When the top-2 columns absorb ≥95% of total risk contribution AND all remaining columns have <1% contribution, the system flags `_risk_concentrated = True`.

**Effect**: Risk contribution is nullified for all columns — QI classification falls back to **keyword-only scoring**. This ensures that obvious QIs (age, region, date) are still detected even when risk is concentrated in an unrelated column.

### 9.6 Sensitive Column Detection

Columns not classified as QI or Identifier are scored for sensitive-column status using structural signals:

| Signal | Weight | When It Fires |
|--------|--------|--------------|
| Continuous numeric + low risk | +0.40 | Float columns that aren't re-identification drivers |
| High entropy (>5 bits) | +0.25 | Many distinct values, likely measurement data |
| Skewness | +0.15 | Skewed distribution (income, prices) |
| Value ratio pattern | +0.25 | Values between 0-1 or 0-100 (percentages, rates) |
| Keyword match | +0.20 | Column name matches sensitive patterns (income, salary, price) |

Negative signals: high-cardinality non-numeric (−0.30), admin keyword (−0.15), high risk contribution (−0.20).

### 9.7 LLM Classification (Cerebras/Qwen)

When `CEREBRAS_API_KEY` is set, AI mode calls `llm_classify_columns()`:

1. Builds column metadata (name, type, unique count, sample values)
2. Sends to Cerebras Qwen 235B with SDC-specific prompt
3. LLM returns role assignments with confidence and reasoning
4. `merge_llm_with_rules()` combines LLM and rule-based results:
   - When they agree → use the shared classification with boosted confidence
   - When they disagree → use the higher-confidence suggestion
   - Falls back to rules-only if LLM fails or is unavailable

---

## 10. Troubleshooting

| Issue | Solution |
|-------|---------|
| R/rpy2 warnings on startup | Non-fatal. LOCSUPR and NOISE fall back to Python implementations. Install R + sdcMicro for enhanced versions. |
| ReID95 still high after protection | Try increasing k (for kANON/LOCSUPR), or switch to Smart Combo mode which escalates automatically. |
| Too many QIs detected | Review the variable importance chart — remove columns with <3% risk contribution from QIs. |
| Too few QIs detected | Check if risk is concentrated in 1-2 columns (concentrated-risk fallback). Try AI mode for better detection. |
| High utility loss (< 60%) | Try perturbation methods (PRAM, NOISE) instead of structural (kANON). Or preprocess more aggressively to reduce cardinality before protection. |
| QI suppression warnings | The affected column has too few unique values for the chosen method. Remove it from QIs, preprocess it, or switch to PRAM/NOISE. |
| AI mode not available | Set `CEREBRAS_API_KEY` environment variable before launching. Without it, only Auto (rules) and Manual modes work. |
| Preprocessing shows no actions | QIs already have low cardinality — no preprocessing needed. Protection should work directly. |
| Protection returns "No method succeeded" | The QI combination is too complex. Try: (1) reduce QI count, (2) preprocess more aggressively, (3) lower k target, (4) use Smart Combo mode. |
| Slow risk computation | Install Polars (`pip install polars`) for 5-10x speedup on large datasets. Or reduce the number of active columns. |
| Date columns not detected | Ensure dates are in recognizable formats (ISO 8601, DD/MM/YYYY, or contain year/month patterns). Greek patterns (ημερομηνία, έτος) are supported. |

---
## 11. Technical Reference

This section provides the mathematical foundations and detailed algorithm descriptions for users who need to understand, validate, or cite the methods used in the SDC Engine.

### 11.1 Per-Combination Re-identification Risk

The SDC Engine measures structural disclosure risk at the **combination level**. Given a dataset with L rows and N columns, the system evaluates all C(N, p) subsets of columns (where p = tuple size, default 3).

For each combination of p columns, the risk is:

```
risk(c₁, c₂, ..., cₚ) = D(c₁, c₂, ..., cₚ) / L
```

where D is the number of distinct value tuples across the p columns. A risk of 1.0 means every row has a unique combination — perfect identifiability.

**Example:** If a dataset has 1,000 rows and the combination (age, zipcode, gender) produces 950 distinct tuples, the risk = 950/1000 = 0.95.

Values of exactly 1.0 are replaced with `1 - ε` (machine epsilon ≈ 2.22 × 10⁻¹⁶) to avoid numerical issues in the logit transformation.

### 11.2 Leave-One-Out Re-identification Risk

The SDC Engine estimates risk using leave-one-out re-identification analysis rather than tail-fitting methods.

#### Per-Record Risk

For each record, the re-identification risk is computed as:

```
risk(record) = 1 / equivalence_class_size
```

where the equivalence class size is the number of records sharing the same combination of values across the active columns. A record in a group of 1 has risk = 1.0 (unique); a record in a group of 10 has risk = 0.10.

#### Summary Statistic

The overall risk is reported as the **95th percentile** of per-record risks (reid_95). This captures the risk level of the most vulnerable 5% of records while being robust to outliers.

#### Variable Importance (Leave-One-Out)

For each column c in the active set:

1. Temporarily remove column c from the analysis
2. Recompute per-record reid_95 on the remaining columns
3. The risk contribution of c is the difference: `reid_95(all) - reid_95(all \ {c})`

Columns whose removal causes the largest drop in reid_95 are the strongest re-identification drivers. This forms the basis of the backward elimination ranking used throughout the application.

### 11.3 Shannon Entropy

Entropy measures the information content (or "unpredictability") of a column. For a column X with values {v₁, v₂, ..., vₖ} occurring with frequencies {f₁, f₂, ..., fₖ}:

```
H(X) = −Σᵢ pᵢ × log₂(pᵢ)    where pᵢ = fᵢ / L
```

| Scenario | Entropy | Interpretation |
|----------|---------|----------------|
| All values identical | 0 bits | No information — cannot distinguish anyone |
| Two equally frequent values | 1 bit | Minimal distinguishing power |
| 1000 equally frequent values | ~9.97 bits | Very high distinguishing power |
| 1000 values, 95% in one category | ~0.38 bits | Low distinguishing power despite high cardinality |

In the context of risk analysis, entropy indicates the **effective** cardinality of a column. High raw cardinality with low entropy (skewed distribution) is less dangerous than moderate cardinality with high entropy (uniform distribution).

### 11.4 Backward Elimination Algorithm

The sensitivity analysis uses a greedy backward elimination procedure to rank variables by their contribution to re-identification risk.

```
ALGORITHM: SDC Engine Backward Elimination

INPUT:  Dataset D with columns {c₁, ..., cₙ}, tuple size p, alpha threshold α
OUTPUT: Elimination order with per-step risk (reid_95) and mean entropy

1. Compute all C(N, p) combination risks
2. Identify extremes: combinations with risk > quantile(risks, 1 − α/100)
3. Set remaining = {c₁, ..., cₙ}, step = 0

4. FOR step = 1 TO min(N − 1, 10):
     FOR each column c in remaining:
       a. Filter extremes to combinations NOT containing c
       b. Compute reid_95_c on remaining columns (without c)
     END FOR

     c* = argmin_c reid_95_c                // column whose removal reduces risk most

     Record: (step, c*, reid_95_c*, mean_entropy(remaining \ {c*}))
     remaining = remaining \ {c*}
   END FOR
```

**Key properties:**
- **Greedy**: Each step removes the single best variable, not the globally optimal subset
- **Capped at 10 steps**: Sufficient for QI identification without excessive computation
- **Entropy tracking**: Mean entropy of remaining columns is recorded at each step, enabling privacy-utility trade-off analysis

### 11.5 Per-Record Risk (SDC Perspective)

While the combination analysis measures per-combination risk, the SDC module measures **per-record risk** for selected quasi-identifiers:

```
record_risk(i) = 1 / group_size(i)
```

where `group_size(i)` is the number of records sharing the same QI values as record i. A unique record (group size = 1) has risk = 1.0.

The ReID percentiles aggregate these per-record risks:
- **ReID50**: Median record risk (50% of records have risk at or below this)
- **ReID95**: 95th percentile (the primary metric for protection decisions)
- **ReID99**: 99th percentile (worst-case tail)

### 11.6 Utility Score Formula

#### Shared Utility Module (`sdc/utility.py`)

All utility calculations across the application (Preprocess, Protect, Smart Combo) use a single shared module (`sdc_engine/sdc/utility.py`). This ensures consistent measurement regardless of which tab or workflow produced the result.

**Key design decisions:**
- **Sensitive-column scope** — utility is measured primarily on **sensitive columns** (analysis variables) when they are assigned in Configure. These columns are what downstream analysts care about — preserving their statistical properties is the goal of SDC. When no sensitive columns are assigned, the system falls back to all non-QI columns, then all common columns as a last resort. This column resolution is handled by `resolve_utility_columns()`.
- **Indirect distortion awareness** — even though sensitive columns are not directly modified by SDC methods, they are indirectly affected through: (1) **record suppression** — deleted records lose their sensitive values; (2) **record swapping** — swapping QI values between records breaks the QI × sensitive relationship; (3) **QI generalization** — grouping QI values loses subgroup granularity for cross-tabulations with sensitive variables. Benchmark analyses (running the same regression or cross-tab before and after) are the most meaningful way to detect this indirect distortion.
- **Row-level preservation for categoricals** — after GENERALIZE bins categories (e.g., "Athens" becomes "0-10"), the original category labels no longer exist in the output. Category overlap (label intersection) would report 0%, even though most data relationships are preserved. Row preservation correctly measures what fraction of actual values changed.
- **String-data awareness** — the application loads all data as strings (`dtype=str`). Utility functions use the centralised column-type labels from `sdc/column_types.py` (not pandas dtype) to detect numeric columns stored as strings (e.g., `"Char (numeric) — Continuous"`). Empty strings are treated as missing values in all association measures (η², Cramér's V, categorical preservation). The `_coerce_pair()` helper converts string-stored numerics to float for correlation and mean-based metrics. When kANON generalises numeric columns to range strings (e.g., "14848-30000"), range midpoints are automatically parsed to compute meaningful correlation instead of returning 0%.
- **Centralised column types** — the same `classify_column_type()` function from `sdc/column_types.py` is used everywhere: Configure table, utility metrics, auto-classify, and relationship checks. This ensures consistent type labels (e.g., a column classified as "Char (numeric) — Continuous" in Configure is treated as numeric throughout the pipeline). The `numeric=False` parameter from Configure's Data Type labels is now propagated through the entire preprocessing pipeline, ensuring string-stored numeric columns are correctly detected during cross-tab utility computation.

#### Column Resolution (`resolve_utility_columns`)

Before computing any utility metric, the system determines which columns to measure:

1. **Sensitive columns** — if assigned in Configure, use only these. This is the primary SDC framing: measure what analysts need to preserve.
2. **Non-QI columns** — if no sensitive columns are assigned, use all columns that are not quasi-identifiers. A UI nudge suggests tagging analysis columns as Sensitive for more meaningful results.
3. **All common columns** — last-resort fallback if the above yield no columns (e.g., when all columns are QIs).

This resolution is centralized in `resolve_utility_columns()` and used by all metric functions.

#### Overall Utility (`compute_utility`)

```
utility = mean(score(col) for col in utility_columns)
```

Where `utility_columns` are resolved as described above (sensitive → non-QI → all), and `score(col)` depends on the column type:

| Column Type | Metric | Formula | Interpretation |
|-------------|--------|---------|----------------|
| **Numeric** | Pearson correlation (abs) | `\|corr(original[col], processed[col])\|` | 1.0 = values perfectly preserved |
| **Categorical** | Row preservation | `sum(original[i] == processed[i]) / n` | Fraction of rows unchanged |

The overall score is the unweighted mean across the resolved utility columns. When utility priority weights are provided, a weighted mean is used instead (see Utility Priority Weighting below). This score drives the per-step acceptance threshold in the preprocessing loop (fast, per-variable). The final displayed utility score is the **composite** (see Composite Utility below), which blends this score with cross-tab relationship preservation. A composite score above 0.85 is good; below 0.60 suggests trying a different method.

> **Why not category overlap?** After GENERALIZE, original labels are replaced by bin labels (e.g., "Athens" → "Population: 0-10k"). Label intersection = 0%, but the actual information loss may be moderate. Row preservation avoids this artifact by comparing values row-by-row.

#### Overall Utility (Protection — Weighted)

When protection methods compute their own utility (e.g., in the Compare Methods tab), a weighted formula is used:

```
utility_score = 0.25 × (1 − information_loss)
             + 0.15 × correlation_preserved
             + 0.20 × mean_preserved
             + 0.15 × distribution_similarity
             + 0.15 × (1 − suppression_rate)
             + 0.10 × row_retention
```

Where:
- **information_loss** = `0.4 × suppression_rate + 0.3 × relative_deviation + 0.3 × (1 − sum_preserved)` — penalizes missing values, shifted means, and changed totals
- **correlation_preserved** — Pearson correlation between original and protected numeric columns (averaged)
- **mean_preserved** — 1 minus the normalized mean absolute difference per column
- **distribution_similarity** — statistical test-based comparison of value distributions
- **suppression_rate** — proportion of cells replaced with NA
- **row_retention** — proportion of rows kept (after any row-level suppression)

A utility score above 0.80 indicates good data preservation. Below 0.60 suggests trying a different method or adjusting parameters.

> **Note: Method ranking vs displayed utility.** The weighted 6-component metric is used internally by `select_method.py` to rank candidate protection methods. It is now scoped to quasi-identifier columns (matching the UI's focus on the variables being transformed). The composite score displayed in the UI blends per-variable utility (30%) with cross-tab relationship preservation (70%) — relationship-focused by default. When sensitive columns are directly modified (per-variable utility < 90%, e.g. by PRAM or noise), weights dynamically shift to 50/50 so direct distortion isn't masked. In rare cases, the auto-selected "best" method may show a slightly lower composite score than expected — this means the method preserved per-column data well but damaged cross-tab analytical relationships.

#### Per-Variable Utility Metrics (`compute_per_variable_utility`)

For each quasi-identifier, detailed metrics are computed:

**Numeric columns:**
- **Correlation** — Pearson correlation with original column (1.0 = perfect preservation). Primary metric shown in the UI.
- **Range ratio** — `processed_range / original_range` (1.0 = range preserved)
- **Mean shift** — `|proc_mean − orig_mean| / |orig_mean|` (0 = no shift)
- **Cardinality change** — unique values before → after

**Categorical columns:**
- **Row preservation** — fraction of rows whose value is unchanged (primary metric shown in the UI)
- **Category overlap** — fraction of original category labels still present (secondary; unreliable after GENERALIZE)
- **Categories removed** — count of original categories merged or dropped
- **Categories added** — count of new categories (e.g., "Other" bucket, bin labels)
- **Cardinality change** — unique values before → after

#### IL1s Information Loss (`compute_il1s`)

sdcMicro-style per-record information loss, measured on the resolved utility columns (sensitive → non-QI → all):

| Column Type | Formula | Interpretation |
|---|---|---|
| **Numeric** | `mean(|orig_i - prot_i| / range(orig))` per record | 0 = no distortion, 1 = maximum distortion |
| **Categorical** | `proportion of changed values` | 0 = all values preserved, 1 = all changed |

The overall IL1 is the (optionally weighted) average across the measured variables. This is the standard information loss measure used by sdcMicro and is more sensitive to per-record distortion than correlation. When measured on sensitive columns, near-zero IL1 is expected (since SDC does not directly modify them); nonzero values indicate indirect distortion from record suppression or swapping.

**Interpretation guide:**

| IL1 Value | Verdict | Meaning |
|---|---|---|
| 0.000 | Unchanged | Variable was not modified at all |
| < 0.05 | Excellent | Negligible information loss |
| 0.05 – 0.10 | Good | Minor distortion, safe for most analyses |
| 0.10 – 0.25 | Moderate | Some variables affected — check which ones |
| 0.25 – 0.50 | Significant | Noticeable distortion, review affected variables |
| > 0.50 | Severe | Variable substantially altered, may not be usable |
| 1.000 | Total loss | All values changed (categorical) or maximum shift (numeric) |

**Why 0% on sensitive variables?** Protection only modifies QIs (quasi-identifiers). Sensitive/analysis variables are preserved as-is. An IL1 of 0 on sensitive columns confirms they were correctly left untouched — this is the desired outcome. Non-zero values on sensitive columns indicate indirect effects (e.g., record suppression removed some rows).

#### Benchmark Analysis (`compute_benchmark_analysis`)

Runs standard statistical analyses on both original and protected data, measured on the resolved utility columns, and compares:

- **Column means** — original vs protected mean + relative difference
- **Column variances** — original vs protected variance + relative difference
- **Pairwise correlations** — correlation matrix difference (mean and max deviation)
- **Frequency tables** — for categorical variables, Total Variation Distance (TVD) between original and protected frequency distributions
- **Relationship check (cross-tab preservation)** — measures whether the relationships between sensitive (analysis) variables and QIs are preserved. Computed when records are suppressed, QIs are generalised, or sensitive values are perturbed. Contributes **30% of the composite utility score**.
  - **Numeric sensitive × QI** — for each qualifying pair (original η² > 0.02, or r² for high-cardinality numeric/date QIs), preservation is computed as 50% group-mean correlation + 50% effect-size ratio (η²_protected / η²_original). The group-mean correlation groups both original and protected sensitive values by the **protected QI bins**, computes group means, and correlates the two vectors. The effect-size ratio measures how well the overall association strength is maintained. Values near 1.0 mean the analyst would get similar cross-tabulation results before and after protection.
  - **Categorical sensitive × QI** — for each qualifying pair (original Cramér's V > 0.05), preservation is computed as 50% group-mean correlation + 50% effect-size ratio (V_protected / V_original). Cramér's V is computed using a vectorized numpy implementation with a cardinality cap of 500 categories to prevent performance issues on high-cardinality columns. Values near 1.0 mean the conditional distributions are well preserved.
  - **Preservation labels**: Strong (≥85%), Moderate (≥65%), Weak (<65%) — shown per-column in the Results column table for both QI and Sensitive columns.
  - Pairs with negligible original relationships (η² < 0.02 or V < 0.05) are filtered out — these are noise, not analytically meaningful.
  - **Range-binned numeric handling**: when kANON generalises numeric columns to range strings (e.g., "14848-30000"), the utility engine parses range midpoints to compute meaningful correlation instead of returning 0%. This affects columns that undergo quantile-binning followed by generalisation.

This approach is recommended by statistical offices: if standard analyses produce similar results on original and protected data, the protection method preserved analytical utility. The relationship check detects indirect distortion that per-column metrics miss.

**Interpretation guide for Rel Diff (means/variances):**

| Rel Diff | Color | Meaning |
|---|---|---|
| 0.00% | Green | No change — variable preserved exactly |
| < 1% | Green | Negligible shift, safe for all analyses |
| 1% – 5% | Yellow | Minor shift, acceptable for most uses |
| > 5% | Red | Notable shift, verify downstream impact |

**Interpretation guide for Freq TVD (categorical):**

| TVD | Meaning |
|---|---|
| 0.000 | Identical frequency distribution |
| < 0.10 | Minor shifts in category proportions |
| 0.10 – 0.25 | Moderate redistribution of categories |
| 0.25 – 0.50 | Significant category changes |
| 1.000 | Complete distribution change (e.g., generalization replaced all category names) |

**Note on TVD = 1.0:** This occurs when category labels in the protected data are entirely new (e.g., kANON generalized "Athens" and "Thessaloniki" to "Region A"). The original labels have zero frequency in the protected data. This does not necessarily mean the information is lost — the generalized labels may still carry analytical value, just at a coarser level.

#### Distributional Metrics (`compute_distributional_metrics`)

Per-variable distributional comparison using information-theoretic measures, on the resolved utility columns:

- **KL divergence** — `D_KL(orig || prot)` (0 = identical distributions; higher = more divergent)
- **Hellinger distance** — bounded [0, 1] measure of distribution similarity (0 = identical, 1 = no overlap)

For numeric variables, distributions are binned into 20 bins. For categorical variables, exact frequency distributions are compared. A small epsilon is added to avoid log(0).

**Interpretation guide for Hellinger distance:**

| Hellinger | Verdict | Meaning |
|---|---|---|
| 0.0000 | Identical | Distributions match exactly |
| < 0.05 | Negligible | Virtually no distributional shift |
| 0.05 – 0.15 | Minor | Small shape changes, safe for analysis |
| 0.15 – 0.30 | Moderate | Noticeable distributional shift |
| 0.30 – 0.70 | Significant | Distribution shape substantially altered |
| > 0.70 | Severe | Distributions are very different |
| 1.0000 | Complete | No overlap between distributions |

KL divergence is unbounded (can be very large for extreme changes) and is harder to interpret intuitively. Use Hellinger as the primary interpretive measure.

#### Composite Utility Score (`compute_composite_utility`)

The composite utility score blends three components that answer different questions about data quality after protection:

| Component | Weight | What It Answers | Source |
|---|---|---|---|
| **Sensitive utility** | 50% | Are analysis variables still statistically valid? | `compute_utility()` — Pearson correlation (numeric) and row-match (categorical) on sensitive columns |
| **QI utility** | 20% | How much resolution did QIs lose? | `compute_per_variable_utility()` — correlation, row preservation, or category overlap per QI |
| **Relationship preservation** | 30% | Can I still do the same cross-tabulations? | `compute_benchmark_analysis()` — 50% group-mean correlation + 50% effect-size ratio (η²_protected/η²_original for numeric sensitive, V_protected/V_original for categorical sensitive). Weak pairs skipped. |

**Fallback**: when no meaningful relationships exist between sensitive and QI columns (all pairs below association threshold), the score falls back to a two-component blend: 70% sensitive + 30% QI.

**Association measures by type**:
- **Numeric sensitive × any QI**: eta-squared (η²) measures original association strength. Pairs with η² < 0.02 are skipped (negligible relationship). High-cardinality numeric and date QIs use Pearson r² instead of raw η² to prevent artificial inflation. Preservation = 50% group-mean correlation + 50% η²_protected/η²_original.
- **Categorical sensitive × any QI**: Cramér's V (bias-corrected) measures original association strength. Pairs with V < 0.05 are skipped. Preservation = 50% group-mean correlation + 50% V_protected/V_original. Cramér's V is computed using a vectorized numpy implementation with a cardinality cap of 500 categories, preventing O(n^2) performance issues on high-cardinality columns.

The Relationship check banner in the utility report shows the relationship preservation score and its contribution to the composite (30%), formatted as "Relationship check (30% of composite): Relationship: X% | Frequency: Y%". Below the banner, a per-pair detail table lists each QI × Sensitive pair with: metric type (η² or Cramér's V), original association strength, protected strength, and preservation %. Skipped pairs (weak original relationships) are shown in grey with the skip reason. Preservation labels in the Results column table: Strong (≥85%), Moderate (≥65%), Weak (<65%).

#### Utility Priority Weighting

Utility priority weights are automatically derived from the **Protection** column in the Configure table:

| Protection | Weight | Meaning |
|---|---|---|
| **Protect +++** (Heavy QI) | 0.5 | Aggressive modification acceptable |
| **Protect ++** (Standard QI) | 1.0 | Standard importance (default) |
| **Protect +** (Light QI) | 2.0 | Preserve as much as possible |
| **Preserve** (Sensitive) | 2.0 | Critical for downstream analysis |
| **Exclude** | 0.0 | Removed from dataset |

When weights are provided, the overall utility score uses weighted mean:
```
utility = sum(weight_i * score_i) / sum(weight_i)
```

This steers method selection: methods that distort high-priority columns score worse, naturally preserving what matters most for analysis.

> **Advanced: Where utility is computed.** All three result tabs (Preprocess, Protect, Combo) use the same utility functions from `sdc/utility.py`:
> - `sdc_preprocessing.py` interactor — at each step (per-variable utility gate), then full metrics at the end. All calls pass `sensitive_columns` when available.
> - `sdc_protection.py` interactor — after applying any protection method. Computes `compute_utility`, `compute_per_variable_utility`, `compute_il1s`, `compute_benchmark_analysis`, `compute_distributional_metrics` with `sensitive_columns` when available. The displayed utility score is the per-variable score.
> - `smart_defaults.py` — during adaptive retry tier evaluation (uses fast `compute_utility()` to decide whether to stop escalating)
> - Smart Combo — computes the same metrics on the final best result directly in the view
> - The shared `build_utility_report()` function in `sdc_result_helpers.py` renders the identical two-section Comprehensive Utility Report across all three tabs

### 11.7 Risk-Weighted GENERALIZE

When backward elimination risk data is available from the Configure tab, GENERALIZE applies **per-QI cardinality limits** instead of a single global `max_categories` value. This means high-risk QIs are binned more aggressively while low-risk QIs retain more granularity.

#### How Per-QI Limits Are Computed

The function `compute_risk_weighted_limits(var_priority, global_max_categories)` maps each QI's risk tier to a specific limit:

| Risk Tier | Condition | max_categories | Rationale |
|---|---|---|---|
| **HIGH** | Risk contribution ≥ 15% | `max(5, global // 2)` | Aggressive reduction — this QI drives most of the risk (floor: 5) |
| **MED-HIGH** | Risk contribution ≥ 8% | `max(5, global × 0.8)` | Somewhat tighter than baseline |
| **MODERATE** | Risk contribution ≥ 3% | `global` (unchanged) | Standard treatment — proportional risk |
| **LOW** | Risk contribution < 3% | `min(20, global × 1.5)` | More generous — preserve utility on low-risk QIs |

**Example:** With `global_max_categories = 10` (moderate tier):
- A QI contributing 42% risk (HIGH) gets `max_categories = 5`
- A QI contributing 12% risk (MED-HIGH) gets `max_categories = 8`
- A QI contributing 4% risk (MODERATE) gets `max_categories = 10`
- A QI contributing 1% risk (LOW) gets `max_categories = 15`

#### Where Risk-Weighted Limits Apply

Risk-weighted limits are automatically injected into:

1. **Preprocess tab** (Auto and Manual mode) — when building the GENERALIZE config, per-QI limits override the global max_categories
2. **Smart Combo tab** — each tier in the escalation loop computes per-QI limits from its tier's global max_categories
3. **Adaptive retry** (`apply_smart_workflow_with_adaptive_retry`) — per-QI limits flow through each retry tier

When no risk data is available (empty `var_priority`), the system falls back to global max_categories for all QIs — no behavior change from the non-risk-weighted pipeline.

> **Numeric bins floor:** Regardless of tier or risk-weighted limits, numeric QIs always get at least 5 bins. This prevents single-bin collapse (e.g., a range of 17-9853 collapsing to "0-9853") which destroys all granularity while adding no privacy benefit beyond what 5 bins already provide.

#### Risk Data Source

The `var_priority` dict comes from backward elimination in the Configure tab. Each entry maps `{column_name: (priority_label, pct_score)}`. The risk contribution percentages are computed by measuring ReID reduction when each variable is removed — the variable whose removal causes the largest ReID drop contributes the most risk.

### 11.8 Adaptive Binning

When `adaptive_binning` is enabled (automatic in both Auto and Manual modes, Smart Combo, and adaptive retry), GENERALIZE searches for the best bin count per numeric QI and the best category count per categorical QI, instead of using a single deterministic formula.

#### How It Works

For each numeric QI, the system tries **3 candidate bin counts** anchored around the risk-weighted `effective_max_categories`:

| Candidate | Formula | Purpose |
|-----------|---------|---------|
| **Tighter** | `max(3, effective_max // 2)` | More aggressive binning for high-risk QIs (subject to numeric floor of 5) |
| **Baseline** | `effective_max` | The risk-weighted default |
| **Looser** | `min(cardinality, effective_max × 1.5)` | Preserve more granularity if utility allows |

For each candidate, the system computes `bin_size = value_range / n_bins` and applies `pd.cut()` to generalize the column.

**Without utility gate** (no threshold set): all candidates are evaluated and the one with the **highest Pearson correlation** (best utility preservation) is selected. Metadata is recorded under `adaptive_search`.

**With utility gate** (threshold > 0): candidates are tried from **gentlest first** (most bins / most categories kept) to most aggressive (fewest bins / fewest categories). The first candidate that passes the per-QI fast utility proxy is accepted — preserving maximum data quality while still reducing cardinality. If all candidates fail, the QI is skipped entirely. Metadata is recorded under `adaptive_retry` with the number of attempts, accepted option, and utility score.

**Categorical QIs** follow the same adaptive pattern: 3 candidates are generated — `effective_max × 1.5`, `effective_max`, and `effective_max // 2` categories to keep (rest merged into "Other"). Tried gentlest-first (most categories preserved).

#### Interaction with Risk-Weighted Limits

Adaptive binning and risk-weighted limits work together: risk-weighted limits set the `effective_max_categories` anchor, then adaptive binning searches around it. This means a HIGH-risk QI (which gets `effective_max = global // 2`) will search candidates `[global // 4, global // 2, global // 2 × 1.5]` — all tighter than a LOW-risk QI.

#### When Adaptive Binning Is Disabled

When `adaptive_binning=False` (default for direct `apply_generalize()` calls), the system uses the deterministic formula: `bin_size = value_range / effective_max_categories` for numeric and a single top-N grouping for categorical.

### 11.9 Sequential Early-Exit GENERALIZE

When both `var_priority` (risk data) and `reid_target` are available, GENERALIZE processes QIs **one at a time in priority order** (HIGH → MED-HIGH → MODERATE → LOW) and checks ReID95 after each. If the target is already met, remaining QIs are **skipped entirely**, preserving their original cardinality.

#### Why This Matters

Without early exit, GENERALIZE bins every high-cardinality QI regardless of whether risk has already been sufficiently reduced. This over-generalizes: if binning the top 2 HIGH-risk QIs already achieves ReID95 < 5%, the remaining LOW-risk QIs lose granularity for no benefit.

#### How It Works

1. QIs needing generalization are **sorted by risk priority** (HIGH first, then MED-HIGH, MODERATE, LOW; ties broken by contribution %)
2. Each QI is generalized as normal (binning, top-N, hierarchy)
3. After each QI, `calculate_reid()` measures the current ReID95 on the partially-generalized data
4. If `reid_95 <= reid_target`: **break** — remaining QIs are left untouched

#### Metadata

The generalization metadata includes:

| Field | Description |
|-------|-------------|
| `generalized_qis` | List of QIs that were actually generalized |
| `skipped_qis` | List of QIs that were preserved (early exit) |
| `early_exit` | Boolean — whether early exit was triggered |
| `reid_after_each` | Dict mapping each processed QI to the ReID95 measured after it |

#### When It Activates

- **Auto mode**: always active — `var_priority` from Configure + `reid_target` from protection context
- **Manual mode**: active when risk data is available from Configure
- **Smart Combo**: active via `apply_smart_workflow_with_adaptive_retry()`

When `var_priority` or `reid_target` is not provided, GENERALIZE processes all QIs in the original order (backward-compatible).

### 11.10 Risk-Informed Method Selection

When per-QI risk data (`var_priority`) is available from backward elimination AND ReID95 exceeds 15%, the method selection engine uses **risk concentration rules** (RC1–RC4) before the generic ReID-based rules. These classify how risk is distributed across QIs (dominated, concentrated, spread-high, single bottleneck) and select a targeted method accordingly. When risk concentration rules fire, they include a `preprocessing_hint` identifying the dominant QI, which is logged in the protection engine output.

**Automatic var_priority population (Spec 07, 2026-04):** Prior to this, `var_priority` was populated only via the Configure tab's backward elimination view. It is now computed automatically by the protection engine in `build_data_features()` for datasets up to 10,000 rows with ≤8 QIs. This means RC rules are reachable without a manual backward-elimination step. Performance guard: computation is skipped for larger datasets (configurable via `VAR_PRIORITY_COMPUTATION` in `config.py` -- see `max_n_records`, `max_n_qis`, `timeout_seconds`).

> For the complete risk concentration rule table, conditions, thresholds, and the full rule priority chain, see [Appendix §3 — Method Selection Rules](#3-method-selection-rules).

### 11.11 Smart Defaults Decision Logic

The smart defaults system in `sdc/smart_defaults.py` operates in **three layers**, each with its own set of rules. The layers run in sequence: Layer 1 diagnoses problems, Layer 2 applies type-specific fixes, and Layer 3 handles generic generalization with adaptive escalation.

---

#### Layer 1: Data Characteristics Diagnosis (`_detect_data_characteristics()`)

This layer scans each QI and produces actionable warnings. It does not modify data — it classifies problems for display in the Configure tab's Data Quality Warnings panel (Section 4.1) and feeds into the type-aware routing in Layer 2.

**Type detection** uses a two-tier approach: first checks the semantic `column_types` label from the Configure table (e.g. `"Integer — Age (demographic)"`), then falls back to data-driven detection (dtype checks, column name pattern matching in both English and Greek).

| # | Rule | Severity | Trigger Condition | Warning / Action |
|---|------|----------|-------------------|------------------|
| 1 | **Direct identifiers** | 🔴 CRITICAL | Column matches known ID patterns (SSN, passport, email, etc.) | "Drop these columns before processing" |
| 2 | **ID-like / Free-text QI** | 🔴 CRITICAL | Semantic type label is "identifier", "id-like", "direct id", or "free text" | "Should not be a QI — remove or change role" |
| 3 | **High-precision dates** | 🟠 HIGH | Date column (by semantic label or dtype) with >30 unique values | "Enable GENERALIZE to bin dates by month/year" |
| 4 | **Fine-grained geography** | 🟠 HIGH | Geographic column (by semantic label or name pattern) with >20 unique values | "Enable GENERALIZE to group into regions" |
| 5a | **Skewed numeric** | 🟠 HIGH | Numeric with >50 unique values AND \|skewness\| > 2 | "Enable Bin numerics + Top/Bottom Coding" |
| 5b | **High-cardinality numeric** | 🟠 HIGH (>200 unique) or 🟡 MEDIUM | Numeric/coded with >50 unique values, not skewed | "Enable Bin numerics to reduce cardinality" |
| 6 | **High-cardinality categorical** | 🟠 HIGH | Non-numeric column with >50 unique categories | "Enable GENERALIZE (Top-K) to keep top categories" |
| 7 | **Rare categories** | 🟡 MEDIUM | Categorical with 10-50 unique values, some values <1% frequency | "Enable Merge Rare Categories" |
| 8 | **Small dataset** | 🟡 MEDIUM | Dataset has <1000 rows | "Consider synthetic data release as alternative" |
| 9 | **Sensitive numeric data** | 🟠 HIGH | Column name matches income/salary/wage/revenue/profit/credit/debt/balance patterns | "Consider Differential Privacy for mathematical privacy guarantees" |

**Skipped columns:** Binary columns and low-cardinality ordinal columns (≤10 unique) are silently skipped — no warning needed.

**Column name patterns** for fallback detection (used when no semantic type label is available):
- **Date patterns** (EN/GR): `date`, `ημερομ`, `ημερ`, `time`, `year`, `month`
- **Geographic patterns** (EN/GR): `zip`, `postal`, `δήμος`, `δημοτ`, `κοινοτ`, `νομός`, `περιφέρ`, `region`, `city`, `municipality`

---

#### Layer 2: Type-Aware Preprocessing (`build_type_aware_preprocessing()`)

Before the GENERALIZE tier loop runs, this layer applies a **one-time preprocessing pass** that routes each QI to a specialized transformation function based on its data type. This runs in both Smart Combo and adaptive retry workflows, before any GENERALIZE steps.

Rules are evaluated in **priority order** (first match wins). QIs that don't match any rule fall through to the generic GENERALIZE step in Layer 3.

| Priority | Rule | Detection Criteria | Transformation | Function | Example |
|----------|------|--------------------|----------------|----------|---------|
| **1** | **Date truncation** | `is_date` (by semantic label, dtype, or name pattern) AND `nunique > 30` | Truncate to month; or to year if >365 unique | `apply_date_truncation()` | "birth_date" (500 unique) → truncate to year |
| **2** | **Age binning** | Column name matches age pattern (EN: `age`, GR: `ηλικ`, `ηλικία`) AND `is_numeric` AND `nunique > 10` AND `_is_human_age()` passes | Bin into 5-year ranges (0-4, 5-9, 10-14, …) | `apply_age_binning(bin_size=5)` | "age" (80 unique) → 5-year bins |
| **3** | **Geographic routing** | `is_geo` (by semantic label or name pattern) AND `nunique > 20` | **Numeric codes:** keep first 3 digits. **Categorical names:** top-K generalization | Numeric: `apply_geographic_coarsening(keep_digits=3)`. Categorical: `apply_generalize(max_categories=N)` | "zip_code" (300 unique) → 3-digit prefix; "Δήμος" (100 names) → top-33 |
| **4** | **Top/bottom coding** | Column flagged as skewed in Layer 1 warnings (warning method contains "Top/Bottom Coding") AND `is_numeric` | Cap outliers at 2nd and 98th percentiles | `apply_top_bottom_coding(method='percentile')` | "income" (skew=3.5) → capped at p2/p98 |
| **5** | **Quantile binning** | `is_numeric` AND `nunique > 20` AND not in skewed set AND not already routed above | Bin into quantile-based ranges using `pd.qcut` (equal-frequency bins); falls back to `pd.cut` (equal-width) if `qcut` produces non-unique edges. **Context-aware target:** the number of bins accounts for other QIs' cardinalities so the total combination space stays feasible for k=5 (range: 5–30 bins). | `apply_quantile_binning(n_bins=target)` | "income" (9433 unique, 3 other QIs with card 2×5×16) → 5 quantile bins; "weight_kg" (150 unique, only QI) → 30 bins |
| **6** | **Default** | None of the above | No type-specific action — falls through to GENERALIZE tier loop | — | "education_level" (8 categories) |

**The age detection guard** (`_is_human_age()`) prevents false positives. It requires ALL of:
- `median < 100` — typical human age distribution
- `min >= 0` — no negative ages
- `max <= 130` — no impossibly old ages

This blocks columns like `building_age` (median 150), `account_age_days` (median 500), or `years_since_founding` (median 40 but max 200) from being binned as demographic age.

**Context-aware bin budget:** For quantile binning (Priority 5), the target number of bins is not fixed — it is computed by estimating how many categories the other QIs contribute to the total combination space. The formula: `budget = n_rows / (other_QI_cardinality_product × k)`, clamped to 5–30. This means a numeric QI paired with many high-cardinality QIs gets fewer bins (e.g., 5) to keep the space feasible, while a numeric QI paired with low-cardinality categoricals gets more bins (e.g., 30) to preserve analytical value. Rounding is no longer used — quantile binning always produces more predictable cardinality reduction.

**Geographic routing — automatic type detection:** The geographic rule now detects whether the column contains numeric codes or categorical names:
- **Numeric postal codes** (>80% digit-only values): `apply_geographic_coarsening(keep_digits=3)` — truncates to regional prefix (e.g. `12345` → `123**`).
- **Categorical place names** (ΜΑΡΟΥΣΙ, ΧΑΛΑΝΔΡΙ, etc.): routes to `apply_generalize(max_categories=N)` instead — groups infrequent locations into "Other" while preserving the most common ones. The `max_categories` is set to `min(10, max(5, nunique // 3))`.

This prevents meaningless digit truncation on string geography (e.g. `"Δήμος Αθηναίων"` → `"Δήμ***"` is useless). A future enhancement could add hierarchical geographic mapping (Δήμος → Νομαρχία → Περιφέρεια).

**Why type-aware preprocessing matters:** Generic GENERALIZE bins all columns the same way (equal-width or top-K). Type-specific transformations preserve domain structure that generic binning destroys. For example:
- Truncating dates to month preserves temporal ordering and seasonality
- 5-year age bins (0-4, 5-9, …) produce clinically and demographically meaningful categories
- Top/bottom coding addresses skewness without destroying the bulk of the distribution

---

#### Layer 3: Smart Defaults Calculation + Adaptive Retry

After type-aware preprocessing, remaining QIs go through the GENERALIZE tier loop. The `calculate_smart_defaults()` function computes parameters based on dataset characteristics.

##### Complexity Score

```
complexity = n_qis × log₁₀(avg_cardinality + 1)
```

Higher complexity = more QIs with more unique values = needs more aggressive treatment.

**Relationship to risk-weighted per-QI limits:** The complexity score and the per-QI risk-weighted limits from `compute_risk_weighted_limits()` (Section 11.7) serve **complementary, not redundant** roles:

| Mechanism | Scope | What it decides | Input |
|-----------|-------|----------------|-------|
| **Complexity score** | Global (all QIs) | The **base** `max_categories` value for the entire dataset | `n_qis`, `avg_cardinality` |
| **Risk-weighted limits** | Per-QI | **Individual overrides** that tighten or loosen the base value per QI | `var_priority` (per-QI risk labels from backward elimination) |

The global base is necessary as an anchor — `compute_risk_weighted_limits()` applies multipliers to it (HIGH: ÷2, MED-HIGH: ×0.8, MODERATE: ×1.0, LOW: ×1.5). Without a data-aware base, the per-QI multipliers would have nothing to scale. For example, with `complexity > 15` the base is capped at 3, so a LOW-risk QI gets `min(20, 3 × 1.5) = 4` — the complexity cap still constrains even the most lenient per-QI override.

##### QI Reduction Recommendations

| Condition | Action |
|-----------|--------|
| `n_qis > 8` | Recommend dropping least-important QIs to reach 7. Importance = inverse cardinality, boosted 2x for demographics (age, gender, education, location). |
| `n_qis > 7` | Advisory: "Consider dropping 1-2 QIs if anonymization fails" |

**Treatment override protection:** QIs that the user manually set to **Heavy** treatment in the Configure table are never recommended for dropping. Their importance score is set to infinity, so they always sort to the "keep" end of the ranking. This prevents the system from contradicting explicit user decisions — if a user flagged a high-cardinality QI as important enough for Heavy treatment, the recommendation respects that. The `drop_recommendations` output includes a `protected_qis` field listing which QIs were excluded from drop candidates.

##### max_categories Rules (How Many Categories to Keep per QI)

The base value is computed from dataset dimensions to ensure k=5 anonymity is feasible:

```
max_groups = n_records / 5
data_aware_cats = max(2, floor(max_groups^(1/n_qis)))
data_aware_cats = min(data_aware_cats, 10)
```

Then constrained by complexity and risk:

| Condition | max_categories | Rationale |
|-----------|---------------|-----------|
| `n_qis > 8` OR `complexity > 15` | min(data_aware, **3**) | Very many QIs — aggressive reduction needed |
| `complexity > 10` OR `ReID₉₅ > 90%` OR `structural_risk > 50%` | min(data_aware, **4**) | High risk needs tighter control |
| Otherwise | data_aware value | Standard complexity — use data-driven value |

##### Numeric Bin Size

| Condition | bin_size |
|-----------|---------|
| Max numeric cardinality > 50 | 10 |
| Max numeric cardinality ≤ 50 | 5 |

##### Preprocessing Strategy

| Condition | Strategy | Meaning |
|-----------|----------|---------|
| `n_qis > 7` | `'all'` | Force GENERALIZE on every QI column |
| `n_qis ≤ 7` | `'auto'` | Only generalize high-cardinality QIs |

##### Method Selection (Preprocessing Quick-Test)

This is a **simplified** method picker used for the preprocessing tier loop's quick protection test — it checks whether the target ReID is achievable after each preprocessing tier. It is **not** the final method selection. The full rules engine (`select_method.py` — RC1-RC3, CAT1-CAT2, QR0-QR4, etc.) is used for the actual Protection phase and considers risk patterns, data type composition, structural risk, and treatment levels. The two are complementary: the preprocessing retry varies preprocessing intensity while keeping the method constant; the rules engine picks the optimal method after preprocessing is complete.

| Condition | Method | Parameters | Rationale |
|-----------|--------|------------|-----------|
| `n_qis > 7` | PRAM | p_change=0.15, pd_min=0.60 | Many QIs → PRAM handles categoricals without massive suppression |
| `4 < n_qis ≤ 7` | kANON | k=3 (if >10K rows) or k=5 | Medium complexity → standard k-Anonymity |
| `n_qis ≤ 4` | kANON | k=3 (if >10K rows) or k=5 | Few QIs → standard k-Anonymity |

##### Adaptive Retry Tiers

In Auto mode, GENERALIZE is not a fixed step — it uses an adaptive retry mechanism that progressively increases aggressiveness until the target ReID is met:

| Tier | Max Categories | When Used |
|------|---------------|-----------|
| **Light** | 15 | First attempt if risk is moderate |
| **Moderate** | 10 | Default starting tier |
| **Aggressive** | 5 | Escalation when moderate fails |
| **Very Aggressive** | 3 | Final escalation — maximum cardinality reduction |

Each tier starts from the original data (never re-generalizes already-generalized data).

**Two-level utility thresholds:** The adaptive retry system uses two distinct utility thresholds:

| Threshold | Default | Source | Purpose |
|-----------|---------|--------|---------|
| **Quality target** | 85–92% | Protection context's `1 - info_loss_max` (e.g. Scientific Use: 90%, Public Release: 85%) | The desired utility level. Tiers that produce utility above this are considered good results. The Smart Combo "Min Utility" slider overrides this value. |
| **Abort floor** | 50% | `min_utility` parameter in `apply_smart_workflow_with_adaptive_retry()` | The absolute minimum before giving up entirely. If any tier drops below this, escalation stops — further tiers will only destroy more data. Values below 50% trigger a warning (data with <50% utility is typically unusable). Callers can override via protection context. |

In practice: the quality target (85%+) is what the system aims for, and escalation continues if the target isn't met but utility remains above 50%. If even the abort floor is breached, the system returns the best result it found so far rather than producing unusable output.

**Per-QI utility gating:** Inside each tier, `apply_generalize()` is called with a utility function and threshold (60% or the abort floor, whichever is higher). GENERALIZE uses this to roll back individual QI binnings that drop per-variable utility below the threshold, preserving analytical structure even within aggressive tiers.

**Structural Risk overrides:**

| SR Range | Override | Effect |
|----------|----------|--------|
| `SR > 50%` and start_tier is light or moderate | Start at **aggressive** | Skips hopeless light tiers |
| `SR > 30%` and start_tier is light | Start at **moderate** | Avoids wasting time on light tier |
| `SR > 80%` | Quality target **−20pp** (min 50%) | Accommodates heavy generalization needed for high-SR datasets |
| `SR > 50%` | Quality target **−10pp** (min 55%) | Sets realistic expectations |

**Best result selection:** The retry loop tracks the best result across all tiers:
1. If any tier meets the ReID target → keep the one with highest utility
2. If no tier meets the ReID target → keep the one with lowest ReID
3. If utility drops below the abort floor (35%) → stop escalating immediately (further tiers will only make it worse)

---

#### Layer Execution Order in Practice

When the user clicks **Run** in Smart Combo or Preprocess Auto mode, the three layers execute in this order:

```
1. _detect_data_characteristics()        → warnings + skewed column set
2. build_type_aware_preprocessing()       → per-QI routing plan
3. apply_type_aware_preprocessing()       → one-time pass (dates, ages, geo, etc.)
4. For each GENERALIZE tier:
   a. calculate_smart_defaults()          → max_categories, strategy, method
   b. apply_generalize()                  → GENERALIZE with tier's max_categories
   c. apply protection method             → kANON or PRAM
   d. check risk → if target met, stop
   e. check utility → if below floor, stop
```

Step 3 modifies the data once. Steps 4a-4e repeat up to 4 times (one per tier), always starting from the type-preprocessed data (not the original raw data, but also not the previous tier's output).

---

#### Type-Aware GENERALIZE (Internal)

Within each GENERALIZE call, the function detects column storage types before applying generalization:

| Storage Type | Detection | Generalization Strategy |
|-------------|-----------|------------------------|
| **Numeric (native)** | `pd.api.types.is_numeric_dtype` | Bin into equal-width ranges |
| **Numeric strings** | >80% parseable as numbers | Convert to numeric, then bin |
| **Date (native)** | `is_datetime64` dtype | Bin by Y/Q/M (auto-selects coarsest unit within max_categories) |
| **Date strings** | >80% parseable as dates (dayfirst=True) | Parse to datetime, then bin by Y/Q/M |
| **Categorical** | Everything else | Keep top (N-1) categories, group rest as "Other" |

Date binning granularity is automatic: tries Year first, then Quarter, then Month, picking the coarsest unit that fits within `max_categories`.

**Pre-binned interval detection:** When a numeric column has already been binned (e.g., by quantile binning in Layer 2), its values are stored as pandas `Interval` objects or strings like `"(10.0, 20.0]"`. GENERALIZE detects both formats — native `pd.Interval` via dtype check and string-encoded intervals via the `(lo, hi]` pattern — and uses the numeric bounds directly for range-aware merging instead of treating the values as arbitrary categories.

#### Configure Table as Source of Truth

The Configure tab's **Data Type** column (e.g. "Char (numeric) — Continuous") is the single source of truth for column type classification across the entire pipeline. When a column is stored as `object` (string) dtype but the Configure table classifies it as numeric, both preprocessing and protection methods convert it automatically before processing.

This applies to:
- **Layer 1 & 2**: `_detect_data_characteristics()` and `build_type_aware_preprocessing()` both check `column_types` first, falling back to data-driven detection only when no label exists
- **GENERALIZE**: checks `column_types` before inline probing
- **Protection methods**: NOISE, kANON, and all other methods receive `column_types` and coerce object-dtype columns to numeric before applying their logic
- **Rules engine**: `build_data_features()` uses `column_types` for accurate continuous/categorical classification

The coercion uses `pd.to_numeric(col, errors='coerce')`, so values that cannot be parsed become NaN and are handled by each method's existing missing-value logic.

### 11.12 Combination Analysis & k-Anonymity

The interactive combination frequency analysis in the Configure view provides a direct assessment of k-anonymity feasibility.

#### Equivalence Classes

An **equivalence class** is a group of records that share identical values for all selected quasi-identifiers. k-Anonymity requires every equivalence class to have at least k records.

```
k = min(group_size for all equivalence classes)
```

If k = 1, there exist records with a unique combination of QI values — these are directly identifiable.

#### Uniqueness Rate

```
uniqueness_rate = (number of equivalence classes of size 1) / (total records)
```

A high uniqueness rate (> 20%) indicates that many records can be uniquely identified by their QI combination. This means:
- k-Anonymity will require significant generalization or suppression
- Preprocessing (binning, merging rare categories) is recommended first
- Consider reducing the number of QIs

#### Group Size Distribution

The histogram shows how equivalence classes are distributed across sizes:

| Group Size | Meaning | Risk Level |
|-----------|---------|------------|
| 1 | Uniquely identifiable record | HIGH — must be generalized or suppressed |
| 2–3 | Weakly protected (small crowd) | MODERATE — attacker has 33–50% chance |
| 4–9 | Reasonably protected | LOW — attacker has < 25% chance |
| 10+ | Well protected (large crowd) | VERY LOW — strong k-anonymity |

#### Practical Thresholds

| k-Value | Suitability |
|---------|-------------|
| k ≥ 10 | Public release (strictest) |
| k ≥ 5 | Scientific use / regulatory |
| k ≥ 3 | Secure environment |
| k = 1–2 | Insufficient for any release |

#### Relationship to ReID

The per-record re-identification risk (ReID) is directly related to equivalence class sizes:

```
risk(record) = 1 / (size of its equivalence class)
```

A record in a group of size 5 has risk = 0.20 (20%). ReID95 is the 95th percentile of these per-record risks — it captures the risk level of the most vulnerable 5% of records.

### 11.13 Three Risk Metrics — Architecture

The system uses three complementary risk metrics:

#### Overall Risk (all variables, dataset-wide)

- **What:** Dataset-level risk from leave-one-out re-identification analysis across all variables
- **How:** Runs backward elimination using per-record reid_95 (95th percentile of 1/equivalence_class_size)
- **Speed:** O(C(n,k)²) — expensive, runs once during "Compute Risk"
- **Used for:** UI-level aggressiveness tuning, preprocessing defaults, variable importance ranking
- **Value range:** 0.0–1.0 (0 = no risk, 1 = maximum risk)

#### Structural Risk (QI-scoped)

- **What:** Backward elimination re-computed on selected QIs only (not all dataset variables)
- **How:** Same backward elimination as Overall Risk, but scoped to the QI subset. The full elimination data (`steps_df`, `top_cols`) is retained and used to rebuild per-QI priorities.
- **Speed:** Moderate — computed once during Save Configuration
- **QI-scoped priority rebuild:** After Save Configuration, the per-QI priority ranking (`_var_priority`) is rebuilt from the QI-scoped elimination data instead of the all-columns analysis. This affects all downstream consumers: treatment auto-fill, LOCSUPR importance weights, GENERALIZE ordering, risk concentration rules, and risk contribution charts. The initial column table retains the all-columns ranking to inform role assignment.
- **Used for:**
  - **Adaptive retry starting tier:** SR > 50% → start at aggressive (skip light/moderate); SR > 30% → start at moderate (skip light)
  - **Smart defaults max_categories:** SR > 50% triggers tighter category cap (same as high complexity)
  - **Method selection:** SR > 50% + ReID95 > 5% → kANON generalization preferred; SR < 20% + tail risk → LOCSUPR preferred
  - **Utility floor adjustment:** SR > 80% → utility floor lowered by 20pp; SR > 50% → lowered by 10pp (prevents engine from rejecting every step as too costly)
  - **Feasibility early warning:** SR > 50% triggers a warning panel with `check_feasibility()` analysis (combination space, % groups below k, recommended tier)
  - **Per-QI risk contribution:** QI-scoped backward elimination data shown per-QI with risk contribution bars (replaces all-columns filtered view)
  - **QI-scoped sensitivity curve:** elimination curve displayed in Configure showing risk change as QIs are removed one at a time, with knee-point annotation
  - **Per-QI before/after table:** risk contribution recomputed on preprocessed data to show how the risk landscape changed
  - **Display:** before/after in Preprocess results
- **Fast fallback:** When the full backward elimination cannot run (large dataset, exception), only a scalar approximation is available (0.6 × uniqueness + 0.4 × reid_95). In this case, per-QI priorities fall back to the all-columns ranking filtered to QIs, and a note is displayed instead of the sensitivity curve.
- **Differs from Overall Risk** because it is scoped to your QI selection only

#### ReID (per-record)

- **What:** Individual record re-identification probability from QI combinations
- **How:** Groups records by QI values, computes 1/group_size for each record
- **Speed:** O(n) — fast, runs during each preprocessing step and protection iteration
- **Used for:** Preprocessing iteration target, protection method selection, before/after comparison
- **Percentiles:** ReID50 (median), ReID95 (main decision metric), ReID99 (worst-case)

#### Data Flow

```
Overall Risk (all variables) — Compute Risk button
├── Drives: preprocessing aggressiveness, utility threshold auto-tuning
├── Feeds: initial _var_priority (all-columns ranking for role assignment)
└── Display: before/after in Preprocess results

Structural Risk (QI-scoped) — Save Configuration button
├── Retains: steps_df + top_cols from QI-only backward elimination
├── Rebuilds: _var_priority from QI-scoped data (replaces all-columns version)
│   └── Downstream: treatment auto-fill, LOCSUPR weights, GENERALIZE ordering,
│       risk concentration rules, risk contribution chart — all automatically upgraded
├── Drives: adaptive retry starting tier (SR>50% → aggressive)
├── Drives: smart defaults max_categories cap (SR>50% → tighter)
├── Drives: method selection (SR>50% → kANON generalization; SR<20%+tail → LOCSUPR)
├── Drives: utility floor adjustment (SR>80% → −20pp; SR>50% → −10pp)
├── Drives: feasibility early warning (SR>50% → check_feasibility panel)
├── Feeds: QI-scoped sensitivity curve (Configure, with knee annotation)
├── Feeds: QI-scoped risk contribution bars (Configure, with "consider removing" hints)
├── Feeds: per-QI before/after table (risk contrib recomputed on preprocessed data)
├── Fallback: fast approximation when full elimination fails → all-columns filtered
└── Display: before/after in Preprocess results

ReID (per-record) — Calculate ReID button + pipeline iterations
├── Drives: method selection rules (QR0-QR4, MED1, LOW1-3, etc.), adaptive parameter tuning
├── Used in: preprocessing step-by-step iteration (target metric)
└── Display: before/after in Preprocess + Protect results
```

### 11.14 Utility Module Architecture (Advanced)

The utility calculation is centralized in a single shared module (`sdc_engine/sdc/utility.py`) to ensure consistency across all SDC workflows. This section documents the architecture for developers and advanced users.

#### Module API

```python
from sdc_engine.sdc.utility import compute_utility, compute_per_variable_utility

# Overall utility on QI columns only
score = compute_utility(original_df, processed_df, quasi_identifiers=["age", "region", "education"])
# Returns: float in [0, 1]

# Per-variable breakdown
metrics = compute_per_variable_utility(original_df, processed_df, quasi_identifiers=["age", "region"])
# Returns: {"age": {"correlation": 0.95, "dtype": "numeric", ...},
#           "region": {"row_preservation": 0.72, "dtype": "categorical", ...}}
```

#### Callers and Data Flow

```
compute_utility() called by:

sdc_preprocessing.py (Preprocess interactor)
  ├── Step-by-step gate (per-variable on sensitive columns)
  ├── Final result score
  ├── Smart workflow utility
  └── Adaptive retry utility

smart_defaults.py (Combo / adaptive retry engine)
  └── Tier utility check (preprocessed vs original, decides stop/escalate)

Both modules pass quasi_identifiers to scope the measurement.
```

#### Why QI-Only Scope Matters

Consider a dataset with 20 columns, 5 of which are QIs. After GENERALIZE:
- 5 QI columns have significantly changed values (utility ~ 0.60)
- 15 non-QI columns are completely unchanged (utility = 1.0)

**Old behavior (all columns):** `utility = (5 × 0.60 + 15 × 1.0) / 20 = 0.90` — misleadingly high

**New behavior (QI-only):** `utility = (5 × 0.60) / 5 = 0.60` — accurately reflects QI distortion

The QI-only approach gives users an honest picture of how much their quasi-identifiers were modified.

#### Why Row Preservation for Categoricals

After GENERALIZE bins a categorical column:
- Original values: `["Athens", "Thessaloniki", "Patras", "Athens", "Heraklion"]`
- Generalized: `["Pop: 1M+", "Pop: 100k-1M", "Pop: 100k-1M", "Pop: 1M+", "Pop: 100k-1M"]`

**Category overlap:** `|{"Athens","Thessaloniki","Patras","Heraklion"} & {"Pop: 1M+","Pop: 100k-1M"}| = 0` → 0%

**Row preservation:** 0 of 5 values are identical → 0%, but this is correct — every value was actually changed.

In practice, when GENERALIZE only affects a subset of categories (e.g., merging 3 rare cities into "Other" while keeping Athens, Thessaloniki unchanged), row preservation correctly reports the fraction of data that was modified (e.g., 85% unchanged = 0.85).

### 11.15 Protection Context Thresholds (Advanced)

The protection context selected in Configure drives default parameters throughout the workflow. Thresholds are defined in `sdc/config.py`:

| Context | `k_min` | `reid_95_max` | `uniqueness_max` | `l_min` | `t_max` | `info_loss_max` | `suppression_max` | Utility Floor |
|---------|---------|-------------|------------------|---------|---------|-----------------|-------------------|----|
| **Public Release** | 10 | 0.01 (1%) | 0.01 (1%) | 3 | 0.15 | 0.15 (15%) | 0.10 (10%) | 85% |
| **Scientific Use** | 5 | 0.05 (5%) | 0.05 (5%) | 2 | 0.25 | 0.10 (10%) | 0.15 (15%) | 90% |
| **Secure Environment** | 3 | 0.10 (10%) | 0.10 (10%) | 2 | 0.30 | 0.08 (8%) | 0.20 (20%) | 92% |
| **Regulatory (HIPAA/GDPR)** | 5 | 0.03 (3%) | 0.03 (3%) | 3 | 0.20 | 0.12 (12%) | 0.12 (12%) | 88% |

**Utility Floor** = `1 - info_loss_max`. Used by the adaptive retry workflow: if a GENERALIZE tier drops utility below the floor, escalation stops and the best result so far is returned.

**l_min** = Minimum distinct l-diversity per equivalence class. When active, the escalation loop continues until both the risk target and l-diversity targets are met. Set to 0 in Configure to disable.

**t_max** = Maximum t-closeness distance (EMD for numeric, TVD for categorical). When active, the escalation loop continues until t-closeness is satisfied. Set to 0 in Configure to disable.

#### How Thresholds Flow Through the System

```
Protection Context + Risk Metric + Privacy Targets (Configure tab)
│
│  Risk Metric: ReID95 (default), k-Anonymity, or Uniqueness Rate
│  → Normalized to 0-1 score: ReID95 passthrough, k→1/k, Uniqueness passthrough
│  → Target also normalized, so all decision rules work unchanged
│
│  Privacy Targets: l-diversity (l_min), t-closeness (t_max)
│  → Auto-filled from context, editable by user, 0 = disabled
│  → Drive the escalation gate alongside the risk target
│
├─→ Smart Combo tab
│     ├── Risk Target ← reid_95_max / k_min / uniqueness_max (per metric)
│     ├── Min Utility ← 1 - info_loss_max
│     └── Both are editable by the user
│
├─→ Apply Protection tab
│     ├── Target banner compares preprocess risk against metric target
│     ├── Auto-Protect maps context → access_tier for select_method_suite()
│     ├── Retry loop uses normalized risk target + l/t gates, stops at utility floor
│     ├── Parameter escalation via PARAMETER_TUNING_SCHEDULES
│     ├── Fallback methods from METHOD_FALLBACK_ORDER
│     └── Method params auto-tuned (k ← k_min, max_suppression ← suppression_max)
│
├─→ Adaptive Retry (smart_defaults.py)
│     ├── target_reid ← reid_95_max
│     └── min_utility ← 1 - info_loss_max
│
└─→ Method Selection Rules Engine
      ├── k values scaled by k_min
      └── Fallback aggressiveness tuned by access tier
```

### 11.16 Column Roles and State Flow (Advanced)

The SDC workflow maintains column role state that flows between tabs:

#### Role Types

| Role | Set In | Effect on Preprocessing | Effect on Protection | Utility Role |
|------|--------|------------------------|---------------------|-------------|
| **QI** | Configure | Modified (generalized, binned, merged) | Modified (anonymized) | Expected loss — shown for transparency |
| **Sensitive** | Configure | Not modified directly | Not modified directly — but indirectly affected by record suppression/swapping | **Primary** — utility measured here |
| **—** (unassigned) | Configure | Not modified | Not modified | Fallback if no Sensitive assigned |

#### State Flow Between Tabs

```
Configure tab
├── quasi_identifiers: List[str]        ← columns with Role = QI
├── sensitive_columns: List[str]        ← columns with Role = Sensitive
├── protection_context: str             ← selected release scenario
├── reid_result: Dict                   ← ReID50/95/99 from Calculate ReID
└── importance_weights: Dict            ← elimination order → LOCSUPR weights

    ↓ (passed to Preprocess, Protect, Combo via project.py)

Preprocess tab
├── Uses quasi_identifiers for all preprocessing operations
├── Uses sensitive_columns for utility measurement (primary scope)
├── Produces: preprocessed_data, reid_after, utility_score
└── get_preprocessing_result() → PreprocessingResult

    ↓ (passed to Protect via project.py)

Protect tab
├── Receives: preprocessed_data, preprocess_reid_after, preprocess_utility
├── Input Data toggle: Raw vs Preprocessed
├── Displays: sensitive_columns, quasi_identifiers
└── Produces: ProtectionResult (protected_data, reid_before/after, utility)

Smart Combo tab
├── Receives: sensitive_columns, protection_context
├── Always starts from raw data (independent of Preprocess)
└── Produces: ProtectionResult (same structure as Protect)
```

### 11.17 Auto-Protect Retry Architecture (Advanced)

Both **Auto-Protect** and **Smart Combo** use the shared protection engine (`run_rules_engine_protection()` in `sdc/protection_engine.py`). The full retry loop — phases, escalation schedules, fallback order, safeguards, and cross-method starts — is documented in [Appendix §4 — Engine Orchestration](#4-engine-orchestration).

This section covers the architectural details specific to how the views invoke the engine.

#### Smart Combo Tier Wrapper

Smart Combo wraps the retry engine in a GENERALIZE tier escalation:

```
for each GENERALIZE tier (light → moderate → aggressive → very_aggressive):
    preprocess_result = apply_preprocessing(config with max_categories=tier)
      (remove IDs → top/bottom coding → merge rare → GENERALIZE)
    utility_check → stop if below floor
    features = build_data_features(preprocessed)   ← FRESH per tier
    result, log = run_rules_engine_protection(preprocessed, features, ...)
    if targets met → stop
```

#### Fresh Data Features

Auto-Protect builds features from the *actual input data* using `build_data_features()` from `protection_engine.py`, rather than reading stale cached features from the Configure tab. This ensures the rules engine correctly adapts when data has been preprocessed, GENERALIZE has been applied at different tiers, or ReID has changed from original levels.

#### Centralized Configuration

All protection context targets and tier mappings are centralized in `sdc/config.py`:

- `CONTEXT_TO_TIER` — maps protection context to access tier
- `get_access_tier(context)` — returns access tier string
- `get_context_targets(context)` — returns `{reid_target, utility_floor, k_min, access_tier, description}`

| protection_context | access_tier | reid_95_max | utility_floor |
|-------------------|-------------|------------|---------------|
| `public_release` | PUBLIC | 0.01 | 85% |
| `scientific_use` | SCIENTIFIC | 0.05 | 90% |
| `secure_environment` | SECURE | 0.10 | 92% |
| `regulatory_compliance` | PUBLIC | 0.03 | 88% |

#### Composite Utility in Retry Loop

The composite utility score blends 50% sensitive utility + 20% QI utility + 30% relationship preservation (when cross-tab data is available; falls back to 70% sensitive + 30% QI otherwise). The utility floor check during escalation uses this composite score to ensure consistent quality assessment.

### 11.18 Shared Module Architecture (Advanced)

The SDC protection system uses shared modules to avoid code duplication across the five SDC views. Here is the module dependency map:

#### Core Shared Modules

| Module | Purpose | Used By |
|--------|---------|---------|
| `sdc/config.py` | Thresholds, tuning schedules, fallback chains, `CONTEXT_TO_TIER`, `get_context_targets()` | All SDC views |
| `sdc/protection_engine.py` | `build_data_features()`, `run_rules_engine_protection()`, `run_pipeline()` | Protect Auto, Smart Combo |
| `sdc/post_protection_diagnostics.py` | `compare_qi_utility()`, `check_l_diversity()`, `check_entropy_l_diversity()`, `check_t_closeness()`, `assess_method_quality()`, `build_failure_guidance()` | protection_engine.py (Auto-Protect) |
| `sdc/utility.py` | `compute_utility()`, `compute_per_variable_utility()`, `compute_composite_utility()`, `compute_benchmark_analysis()`, `_eta_squared()`, `_cramers_v()`, `_categorical_preservation()` | Preprocess, Smart Combo, smart_defaults.py, Results |
| `sdc/column_types.py` | `classify_column_type()`, `classify_columns()`, `is_continuous_type()` — centralised column-type classification | Configure, utility.py, auto_classify.py |
| `tools/classify_columns.py` | Standalone CLI tool — classifies CSV/Excel columns by structural type, detects direct identifiers (including Greek national IDs), flags SDC patterns. Usage: `python tools/classify_columns.py <file.csv>` | Standalone (no app dependencies) |
| `_archive/panel_ui/views/sdc_result_helpers.py` | `build_metrics_cards()`, `build_column_changes_table()` (archived Panel-era helper; Streamlit pages now build metrics inline) | Protect, Smart Combo |
| `sdc/selection/pipelines.py` | `select_method_suite()`, dynamic + legacy pipeline rules | protection_engine.py |
| `interactors/sdc_protection.py` | `SDCProtection.apply_method()` — applies any protection method | Protect, Smart Combo |

#### Data Flow

```
Configure ──→ data_features (cached for recommendation display)
                 │
Preprocess ──→ preprocessed_data ──→ Protect (input_data toggle)
                                          │
                                          ├── build_data_features(input_data) ← FRESH
                                          └── run_rules_engine_protection(...)

Smart Combo:
  raw_data ──→ GENERALIZE(tier) ──→ build_data_features(preprocessed) ← FRESH
                                        └── run_rules_engine_protection(...)
```

#### Key Design Decisions

1. **Configure keeps its own `_build_data_features()`** — it reads from `self.dataset` and caches results for the recommendation display. The standalone `build_data_features()` in `protection_engine.py` is used by Protect and Combo for fresh features.
2. **`smart_defaults.py` is preserved** for backward compatibility but is no longer called by Smart Combo (which now uses the rules engine directly).
3. **All views use `get_context_targets()`** instead of directly accessing `PROTECTION_THRESHOLDS` — this ensures consistent target derivation everywhere.

### 11.19 Per-QI Treatment Architecture (Advanced)

The per-QI treatment system is centralized in `sdc/qi_treatment.py` to avoid code duplication across methods and views.

#### Module: `sdc/qi_treatment.py`

| Constant / Function | Purpose |
|---------------------|---------|
| `TREATMENT_MULTIPLIERS` | `{'Heavy': 1.5, 'Standard': 1.0, 'Light': 0.5}` |
| `PRIORITY_TO_TREATMENT` | Maps risk priority labels → treatment levels |
| `METHOD_PARAM_INFO` | Per-method param name, min/max clamp range |
| `build_per_variable_params()` | Creates `{col: scaled_value}` dict from base param + treatment map. Returns `None` if all Standard (zero overhead). |
| `build_locsupr_weights()` | Creates `importance_weights` dict for LOCSUPR (Heavy=1, Standard=3, Light=5) |
| `build_per_qi_percentiles()` | Creates `{col: (bottom, top)}` for treatment-scaled top/bottom coding percentiles |
| `build_per_qi_min_frequency()` | Creates `{col: min_freq}` for treatment-scaled rare category merging |
| `get_adaptive_binning_candidates()` | Returns 3 candidate bin counts shifted by treatment (Heavy→aggressive, Light→gentle) |
| `TREATMENT_GATE_MULT` | GENERALIZE utility gate multipliers: Heavy=0.75 (lenient), Standard=1.0, Light=1.25 (strict) |

#### Data Flow

```
Configure tab
│   Protection (editable — derives Role + Treatment from single gesture)
│   get_qi_treatment() → {col: 'Heavy'|'Standard'|'Light'}
│
├── project.py → Preprocess view (qi_treatment param)
│   └── sdc_preprocess.py
│       ├── _on_apply() → config['qi_treatment'] → preprocess_for_sdc()
│       │   ├── Top/bottom coding → per_qi_percentiles (Heavy=tighter, Light=wider)
│       │   ├── Merge rare → per_qi_min_frequency (Heavy=higher, Light=lower)
│       │   └── GENERALIZE → shifted adaptive binning + treatment-aware gate
│       └── _run_adaptive_generalize() → apply_generalize(qi_treatment=...)
│
├── project.py → Protect view (qi_treatment param)
│   └── sdc_protect.py
│       ├── Manual: apply_method(..., qi_treatment=self.qi_treatment)
│       └── Auto: closure captures qi_treatment → apply_method(...)
│           └── sdc_protection.py → _build_method_kwargs(qi_treatment=...)
│               ├── NOISE → per_variable_magnitude dict
│               ├── PRAM → per_variable_p_change dict
│               ├── kANON → per_qi_bin_size dict
│               └── LOCSUPR → importance_weights dict
```

#### Clamping Ranges

| Method | Parameter | Min | Max |
|--------|-----------|-----|-----|
| NOISE | magnitude | 0.01 | 0.25 |
| PRAM | p_change | 0.05 | 0.50 |
| kANON | bin_size | 2 | 50 |

Example: PRAM with base `p_change=0.40` and Heavy treatment → `0.40 × 1.5 = 0.60` → clamped to `0.50` (max valid range).

#### Preprocessing Treatment Effects

| Step | Parameter | Heavy (1.5×) | Standard (1.0×) | Light (0.5×) |
|------|-----------|-------------|-----------------|-------------|
| Top/bottom coding | percentiles | Tighter (2nd/98th) | Default (1st/99th) | Wider (0.5th/99.5th) |
| Merge rare | min_frequency | Higher (e.g. 5 if base=3) | Default (3) | Lower (2) |
| GENERALIZE | adaptive binning | Aggressive candidates | Default 3 candidates | Gentle candidates |
| GENERALIZE | utility gate | Lenient (threshold × 0.75) | Default threshold | Strict (threshold × 1.25) |

Percentile clamping: bottom ∈ [0.1, 10], top ∈ [90, 99.9]. Min-frequency clamping: [1, 50].

Safe-skip guard: if all categories in a QI already exceed the effective min_frequency, that QI is skipped in merge rare — no unnecessary category loss.

### 11.20 Feasibility Checking and Per-QI Protection Tuning (Advanced)

The protection engine integrates `diagnose_qis()` to obtain `max_achievable_k` for the QI combination, which informs several protection decisions.

#### Cardinality-Aware k Pruning

Escalation schedules (e.g., kANON k = 3 -> 5 -> 7 -> 10 -> ...) now skip k values that exceed `max_achievable_k`. If the QI combination space makes k=10 impossible to achieve, the schedule skips directly to feasible values. This avoids wasting time on doomed attempts and reaches the best achievable result faster.

#### Feasibility-Aware Cardinality Budgets

Before iterative generalization, kANON computes per-QI cardinality targets based on the combination space constraint:

- **Budget calculation**: `max_combinations = n_rows / k`, then each QI gets `max_combinations^(1/n_qis)` as its target cardinality
- **Global minimum floor**: 5 categories, applied uniformly to all QI types. The floor is the same regardless of whether the QI is numeric, categorical, or a pre-binned range column. This resolves a prior inconsistency where different QI types received different floors, causing the floor to exceed the computed target in some cases and blocking generalization entirely.
- **Pre-binned range detection**: Columns that contain numeric range patterns (e.g., `"10000-19999"` from type-aware preprocessing, or `pd.cut`/`qcut` interval strings like `"(10.0, 20.0]"`) are detected automatically and handled with range-aware merging instead of text truncation
- **Range merging**: Adjacent ranges are merged into broader ranges (e.g., `"1950-1959"` + `"1960-1969"` → `"1950-1969"`) — preserving ordered numeric structure instead of collapsing to categories like `"1*"` or `"Other"`
- **Dynamic floor cap**: The floor is capped at half the current bin count to always allow at least one level of merging

This prevents the common failure mode where kANON crushes all QIs to 3 categories when the combination space is infeasible.

#### Per-QI kANON bin_size

When kANON is selected, the `bin_size` parameter is adjusted per QI based on risk priority:

| Risk Priority | bin_size Adjustment | Effect |
|---|---|---|
| **HIGH** | Smaller bins (more generalization) | Aggressive reduction of the highest-risk QIs |
| **MODERATE** | Default bin_size | Standard treatment |
| **LOW** | Larger bins (less generalization) | Preserves more detail in low-risk QIs |

This means a HIGH-risk QI (e.g., exact date of birth) gets binned into wider ranges (more generalization), while a LOW-risk QI (e.g., broad region) retains finer granularity.

#### Per-QI PRAM p_change

When PRAM is selected, the per-QI swap probability is scaled by risk priority:

| Risk Priority | p_change Multiplier | Example (base=0.20) |
|---|---|---|
| **HIGH** | 1.5x | 0.30 |
| **MODERATE** | 1.0x (default) | 0.20 |
| **LOW** | 0.6x | 0.12 |

HIGH-risk QIs receive more aggressive perturbation, while LOW-risk QIs are barely touched.

#### Cross-Method Fallback Start

When the primary method fails at strength X, the fallback method starts at an equivalent strength level rather than at its minimum. For example:

- kANON fails at k=5 -> LOCSUPR starts at k=5 (not k=3)
- kANON fails at k=10 -> PRAM starts at p_change=0.15 (equivalent aggressiveness, not 0.10)

This avoids re-trying weak settings that are guaranteed to fail, and reaches the target faster.

#### Ensure Feasibility (Advisory)

When all methods fail to meet targets, `ensure_feasibility()` runs as an advisory check. It analyzes the QI combination and suggests which QI(s) to remove to make protection feasible. The suggestion is displayed in the failure guidance panel but is NOT auto-applied — the user must confirm any QI removal. This preserves user control over the privacy/utility trade-off.

### 11.21 Post-Protection Diagnostics (Advanced)

After protection completes (via Auto-Protect or Smart Combo), a collapsible **Protection Diagnostics** section appears in the results panel. It is collapsed by default to minimize UI impact while providing detailed diagnostic information on demand.

The diagnostics are implemented in `sdc/post_protection_diagnostics.py` as pure functions with no UI dependencies.

#### Diagnostic Components

| Component | What It Shows |
|---|---|
| **Per-QI Utility Comparison** | Table comparing per-QI utility before protection (from Preprocess) vs after. Shows delta and impact verdict (minimal / moderate / significant). Highlights QIs where protection caused disproportionate utility loss. |
| **l-Diversity (distinct)** | Whether equivalence classes have diverse sensitive values. Checks small classes (≤50 records) where violations concentrate. Reports per-sensitive-column min-l and violation counts. |
| **Entropy l-Diversity** | Stronger variant: requires Shannon entropy H ≥ log(l_target) per class. Reports effective l = exp(min H), catches skewed distributions where distinct count passes but one value dominates. Per-sensitive breakdown with min entropy and effective l. |
| **t-Closeness** | Whether sensitive attribute distribution within each equivalence class is close to the overall distribution. Uses EMD (Earth Mover's Distance) for numeric columns and TVD (Total Variation Distance) for categorical. Reports max distance, target threshold, and per-sensitive breakdown with metric type (EMD/TVD). |
| **Method Quality Checks** | Method-specific quality metrics from protection metadata: kANON suppression rate, LOCSUPR suppression concentration, PRAM distribution preservation, NOISE correlation preservation. |
| **Timing and Escalation Info** | Duration of each phase (pipeline, primary+escalation, fallbacks, total), number of escalation steps tried, and which methods were attempted. |
| **Treatment Alignment** | Verifies that Heavy QIs actually received heavier treatment than Light QIs (e.g., more suppressions, more value changes, lower correlation). Flags misalignments. |
| **Feasibility Suggestions** | When all methods fail: identifies the bottleneck QI (highest cardinality), lists per-method failure reasons, and suggests specific remediation steps (e.g., "Remove column X to reduce combination space by 95%"). |

#### When Diagnostics Appear

- **Auto-Protect**: always generated after the retry loop completes
- **Smart Combo**: always generated for the best tier result
- **Manual Apply**: not generated (user has full control)

The diagnostics section uses the same collapsible panel pattern as other advanced sections in the application, keeping the default view clean while making detailed information available with a single click.

### 11.22 Smart Method Configuration (Advanced)

Before calling any protection method, the engine runs a **pre-estimation** pass that analyses the data to adapt method parameters and detect situations where the selected method will be ineffective. This is implemented in `sdc/smart_method_config.py`.

#### Suppression Pre-Estimation (kANON / LOCSUPR)

For kANON, the engine groups records by QI values to compute equivalence classes, then counts how many fall below the target k. This yields:

- **Estimated suppression rate** — the fraction of records that would be suppressed. If this exceeds 25%, the engine logs a warning and may automatically switch to PRAM or NOISE before wasting a full protection run.
- **Strategy selection** — based on the percentage of violating equivalence classes:
  - <10% violating → *suppression-dominant* (suppress outliers, don't generalize)
  - 10–40% violating → *hybrid* (generalize high-risk QIs, suppress remaining)
  - ≥40% violating → *generalization-dominant* (structural changes needed)
- **Smart starting k** — if most equivalence classes already satisfy the target, start at target k directly; if violations are widespread, start lower.

For LOCSUPR, the engine estimates per-QI violation concentration (which QIs appear most in violating groups). When a single QI absorbs >60% of all suppressions, a warning is issued recommending heavier preprocessing on that column. When estimated cell suppression exceeds 10%, the engine may switch to kANON instead.

#### PRAM Category Dominance Detection

For each QI, the engine checks whether a single category holds >80% of records. When this occurs, PRAM perturbation is ineffective because the transition matrix keeps most records in the dominant category regardless of the p_change parameter. Affected QIs are flagged with "low effectiveness", and if more than half of all QIs are dominated, the engine switches to kANON before trying PRAM.

#### IQR-Proportional Noise (NOISE)

Instead of applying a uniform magnitude to all numeric QIs (which treats a column ranging 0–100 the same as one ranging 0–1,000,000), the engine computes per-variable magnitude proportional to each column's interquartile range (IQR = Q3 − Q1):

    effective_magnitude = base_magnitude × treatment_multiplier × (IQR / column_std)

This ensures columns with larger spread receive proportionally larger noise, and columns with tight distributions receive less. The engine also checks:

- **Distribution risk** — if noise std exceeds 50% of the column std, the distribution will be noticeably altered (flagged as "high" risk).
- **Correlation preservation** — for pairs of numeric QIs with strong correlation (r > 0.30), estimates post-noise correlation attenuation. Warns when a strong correlation (r > 0.70) would drop below 0.50.

#### Pre-Application Method Switching

When smart configuration detects that the selected primary method will be ineffective (e.g., kANON with >25% suppression, PRAM with dominated categories), the engine automatically switches to a more appropriate method **before** the first application. The original primary is prepended to the fallback chain so it is still attempted if the switched method also fails.

### 11.23 Smart Hierarchy Builders (Advanced)

The system uses ARX-inspired multi-level generalization hierarchies instead of ad-hoc binning. Each hierarchy defines multiple generalization levels (0 = original, N = most general) and is auto-selected based on the column's detected type.

#### Hierarchy Types

| Builder | Data Type | Example Levels | Auto-Detected When |
|---------|-----------|----------------|-------------------|
| **IntervalHierarchyBuilder** | Numeric (age, income) | 37 → "35-39" → "30-39" → "20-39" → "*" | Column is numeric or age-like |
| **DateHierarchyBuilder** | Temporal (dates) | 2024-03-15 → "2024-03" → "2024-Q1" → "2024" → "*" | Column is date/datetime |
| **MaskingHierarchyBuilder** | Alphanumeric codes (postal) | "12345" → "1234*" → "123**" → "*" | Geographic + numeric string pattern |
| **CategoricalHierarchyBuilder** | Nominal categories | "Athens" → "Athens" → "Other" → "*" | Default for categoricals (>2 unique) |

#### Info-Loss-Aware Generalization

The kANON greedy loop scores each QI by **violations per unit info loss** rather than raw violation count. This means:
- High-cardinality columns (postal codes, dates) are generalized first — each level produces small info loss but resolves many violations
- Low-cardinality columns (gender, binary flags) are generalized last — each level produces large info loss

Info loss at each level is defined as `1 - (cardinality_at_level / cardinality_at_original)`, matching ARX's "Loss" metric.

#### User-Provided Hierarchies

You can override any auto-generated hierarchy by providing a dict mapping in the Configure table's hierarchy field. The system wraps it as a 2-level hierarchy (original → mapped values) via `Hierarchy.from_legacy_dict()`.

---

### 11.24 l-Diversity Enforcement During k-Anonymity (Advanced)

When an l-diversity target is set (via the protection context or explicitly), the kANON generalization loop enforces **both** k-anonymity and l-diversity simultaneously:

1. The loop continues until all equivalence classes satisfy **both** k ≥ target AND l ≥ target
2. After each generalization step, l-diversity is checked (performance guard: only when close to k-anonymity, i.e., < 20% violation rate)
3. If a generalization step significantly worsens l-diversity (>50% more violations), it is **reverted** and that QI is marked as exhausted

This prevents the common problem where aggressive generalization merges equivalence classes and destroys sensitive-value diversity.

**Distinct-l-diversity:** Each equivalence class must contain at least *l* distinct values of the sensitive attribute.

The l-diversity check reuses the existing `check_l_diversity()` function from `post_protection_diagnostics.py` — no separate computation is needed.

---

### 11.25 t-Closeness Enforcement During k-Anonymity (Advanced)

t-Closeness is enforced alongside k-anonymity and l-diversity in the kANON generalization loop, giving **simultaneous k + l + t** — matching ARX's key advantage.

**t-Closeness criterion:** The distribution of each sensitive attribute within every equivalence class must be "close" to its overall distribution. Distance is measured using:
- **Earth Mover's Distance (EMD)** for numeric sensitive columns (normalized to [0, 1])
- **Total Variation Distance (TVD)** for categorical sensitive columns

**Enforcement in the generalization loop:**

1. After each iteration achieves k-anonymity, t-closeness is checked (same performance guard as l-diversity — only when < 20% violation rate)
2. If `t_achieved > t_target`, the loop continues generalizing
3. The convergence condition requires **all three**: `k_satisfied AND l_satisfied AND t_satisfied`
4. If a generalization step increases t-closeness distance by > 50%, the step is **reverted** (same revert pattern as l-diversity)

**Configuration:** Set t-closeness target in the Configure tab's privacy targets section. Default: 0.30 (moderate protection). Set to 0 to disable. Typical ranges:
- 0.15–0.20: Strong protection (hard to achieve, may require aggressive generalization)
- 0.25–0.35: Moderate protection (good balance)
- 0.40–0.50: Light protection (easy to achieve)

**Note:** t-Closeness is harder to satisfy than l-diversity, especially with categorical sensitive columns that have skewed distributions. If the target cannot be met, the engine logs a warning and returns the best result achieved.

---

### 11.26 Beam Search Strategy (Advanced)

The `beam` strategy replaces the greedy single-QI-per-step approach with a lightweight lattice search inspired by ARX's Flash algorithm.

**Algorithm:**
1. **State space:** Each state is a tuple of generalization levels, one per QI. E.g., `(0, 0, 0)` = no generalization, `(1, 0, 2)` = QI₁ at level 1, QI₃ at level 2.
2. **Beam search:** Maintains top-B candidate states (default B = 5). At each iteration:
   - Expands each state by incrementing each QI by one level → generates up to B × n_QI successors
   - Scores each successor: primary = k-anonymity violations (lower is better), secondary = total info loss via hierarchy objects (lower is better)
   - Keeps only the top-B successors for the next iteration
3. **Termination:** Stops when a state achieves all privacy targets, or time budget is exceeded, or all states are fully generalized.

**Advantages over greedy:**
- Explores fundamentally different generalization paths (e.g., "generalize age heavily + keep region" vs "keep age + generalize region heavily")
- Finds solutions with lower total information loss
- In testing, beam search often preserves more unique values in high-cardinality columns

**Performance guards:**
- If `beam_width × n_QIs > 50`, beam width is automatically reduced
- If dataset has > 50K rows, beam width is capped at 2
- Same time budget as greedy: 8s + 0.5s per 10K rows, max 20s
- State scores are cached to avoid recomputation

**Usage:** Set `strategy='beam'` in kANON parameters. Falls back to greedy if no solution is found.

---

### 11.27 Recursive Local Recoding (Advanced)

The `recursive` strategy implements ARX's approach to recovering suppressed records. After standard generalization + suppression, some records are lost (QI cells set to NaN). Instead of discarding them, recursive local recoding re-anonymizes the suppressed subset with more aggressive parameters.

**Algorithm:**
1. **Phase 1:** Standard generalization (same as `'generalization'` strategy)
2. **Phase 2:** Standard suppression on remaining violations
3. **Phase 3:** Recursive recovery (up to `max_depth=2` levels):
   - Identify suppressed rows (any QI column is NaN)
   - Restore their original QI values from pre-anonymization data
   - Re-run generalization with more aggressive parameters (bin_size doubled, prefix_length reduced, higher starting generalization level per depth)
   - Re-run suppression with a generous 20% budget on the subset
   - Merge recovered rows back into the main result
   - Verify k-anonymity on the combined dataset; apply cleanup suppression if needed
   - Recurse on remaining suppressions

**Benefits:**
- Recovers data that would otherwise be lost to suppression
- Each recursion level uses progressively more aggressive generalization — appropriate for "outlier" records that couldn't fit normal equivalence classes
- Preserves more records than pure suppression while maintaining k-anonymity

**Usage:** Set `strategy='recursive'` in kANON parameters.

---

### 11.28 ML Utility Validation (Advanced)

Measures how well analytical relationships are preserved after anonymization by comparing classifier performance on original vs anonymized data.

**Methodology:**
1. For each eligible sensitive column (categorical, 2–10 unique values):
   - Features = QI columns (numeric + low-cardinality categoricals, one-hot encoded)
   - Target = sensitive column
   - Model = Logistic Regression with stratified 3-fold cross-validation
2. Train on original data → mean accuracy₁
3. Train on anonymized data → mean accuracy₂
4. **Accuracy ratio** = accuracy₂ / accuracy₁ (capped at 1.0)

**Interpretation:**
- **≥ 0.90** (green): Excellent utility — anonymization barely affects analytical value
- **0.70–0.89** (yellow): Acceptable utility — some information loss but relationships mostly preserved
- **< 0.70** (red): Significant utility loss — consider less aggressive protection

**Where it appears:** In the Post-Protection Diagnostics panel (Protect and Combo tabs), as an "ML Utility Validation" section showing a table with target column, original accuracy, anonymized accuracy, and color-coded ratio.

**Note:** Requires scikit-learn. If not installed, ML utility is silently skipped. The metric is computed as a non-blocking diagnostic — it never affects the protection decision.

---

## Appendix: SDC Protection — End-to-End Reference

This appendix covers the complete SDC protection flow from raw data to protected output: preprocessing, method selection, method internals, and engine orchestration. It serves as a single reference for the full protection landscape.

```
Raw data
  → 1. QI Preprocessing: type-aware routing + adaptive tier loop
  → 2. Method Selection: rules engine picks protection method
  → 3. Method Execution: kANON / LOCSUPR / PRAM / NOISE / pipeline
  → 4. Engine Orchestration: retry loop with escalation + fallbacks
  → Protected output
```

---

### 1. QI Preprocessing

#### 1.1 Type-Aware Preprocessing (one-time pass)

Each QI is routed to a specific function based on its semantic type. The routing priority is:

1. **Dates** (>30 unique) → `apply_date_truncation`
   - 365+ unique → truncate to year
   - 100+ unique → truncate to quarter
   - otherwise → month
   - Guard: falls back to finer granularity if fewer than 3 periods would result

2. **Age-like numeric** (name matches "age/ηλικ", >10 unique, and passes human-age guard: median <100, min ≥0, max ≤135) → `apply_age_binning` into 5-year ranges

3. **Geographic** (>20 unique):
   - Numeric/digit postal codes → `apply_geographic_coarsening` (truncate leading digits, context-aware budget)
   - Categorical place names → `apply_generalize` (top-K frequency grouping)

4. **Skewed numeric** (detected as skewed with |skew| > 2.0):
   - Moderate cardinality → `apply_top_bottom_coding` at p2/p98
   - Near-unique skewed (>n_rows/5) → falls through to step 5

5. **High-cardinality numeric** (>20 unique) → `apply_quantile_binning`
   - Budget is context-aware: accounts for the cardinality product of all other QIs so the combined combination space stays feasible for k=5

6. **High-cardinality categorical** (>20 unique) → `apply_generalize` (top-K frequency grouping)

7. **Default** → falls through to GENERALIZE in the tier loop

**Key design principle:** every cardinality decision (date granularity, geo digits, quantile bins) uses a context-aware budget — it accounts for the combined cardinality of all other QIs so the total combination space stays feasible for k=5, rather than treating each QI in isolation.

#### 1.2 Adaptive Tier Loop

After type-aware preprocessing, GENERALIZE runs in escalating tiers until the risk target (default ReID95 ≤ 5%) is met:

- **Tiers:** light → moderate → aggressive → very aggressive (with decreasing `max_categories`)
- `max_categories` is data-aware: computed as `(n_records / 5) ^ (1 / n_qis)`, capped at 10
- Each tier starts fresh from the type-preprocessed data (not compounding)
- Stops early if utility drops below the minimum floor (default 60%)
- If `structural_risk > 50%`, skips straight to aggressive tier

**Quick-test method selection** (simplified, for the tier loop only — NOT the final method selection; the full rules engine in §3 runs after preprocessing to pick the actual protection method):
- 7+ QIs → PRAM (p_change=0.15) — avoids massive suppression from k-ANON with many QIs
- 4–7 QIs → k-ANON with k=3 (datasets >10K rows) or k=5
- <4 QIs → k-ANON with k=3 (datasets >10K rows) or k=5

#### 1.3 QI Reduction Recommendations

If >8 QIs are detected, the system recommends dropping the least important ones (by inverse cardinality) to reach 7 QIs, boosting standard demographics (age, gender, sex, location, region, education). QIs with Heavy treatment set by the user are never recommended for dropping and get `importance_score = inf`. At 8 QIs exactly, a softer advisory ("consider dropping 1–2 if anonymization fails") is issued instead.

---

### 2. Method Execution Details

#### 2.1 GENERALIZE — Cardinality Reduction

Operates on each QI in priority order (HIGH-risk QIs first if `var_priority` is provided). For each:

- **Type detection:** uses Configure table's `column_types` as source of truth; falls back to data probing (>80% numeric values, EU number format detection, date string detection)
- **Dates:** bins to year/quarter/month with a hard floor of 3 distinct periods
- **Numeric:** equal-width binning by default; switches to quantile binning when skew > 3.0; adaptive mode tries multiple bin counts and picks the one with best correlation or highest utility
- **Categorical with custom hierarchy:** applies the provided mapping
- **Categorical without hierarchy:** keeps top-N + "Other", but skips if values already look like numeric ranges (e.g. "20-24") — those are handled by kANON's range-merge
- **Floors:** categoricals never collapse below 3; numerics have cardinality-proportional floors (5 for <1000 unique, 10 for <5000, 15 above)
- **Utility gate:** if `utility_fn` + `utility_threshold` provided, each generalization is checked and rolled back if utility drops too far; with `qi_treatment`, Heavy QIs get a more lenient threshold, Light QIs stricter
- **Early exit:** stops generalizing once risk target is met, preserving cardinality on remaining QIs

**Risk-weighted limits** (`compute_risk_weighted_limits`): HIGH-priority QIs get `max_categories // 2`, MED-HIGH get 80% of global, LOW get up to 150%.

#### 2.2 kANON (k-Anonymity) — The Workhorse

Takes the generalized data and enforces the k-anonymity guarantee through an iterative greedy loop.

**Generalization phase** (`_achieve_kanon_generalization`):
- Computes a feasibility check: combination space vs. `n_rows / k`
- **Budget-aware cardinality floors**: each QI's floor is `max(5, min(target_per_qi, current_cardinality))` where `target_per_qi` is the nth-root of `n_rows / k` distributed across QIs. High-cardinality QIs get higher floors (more room to generalize); low-cardinality QIs stay at the minimum floor of 5
- **Info-loss-aware QI scoring**: each iteration scores QIs by `fragmentation_ratio / marginal_info_loss` — prioritizing cheap generalizations (high cardinality reduction, low info loss) over expensive ones. Uses hierarchy objects' `info_loss_at()` when available, falls back to plain fragmentation ratio. Treatment boost and generalization headroom serve as secondary tiebreakers
- **Simultaneous k + l + t enforcement**: when l-diversity and/or t-closeness targets are set, the loop continues until ALL privacy targets are satisfied. Privacy checks use performance guards (only when < 20% violation rate). Generalizations that worsen l-diversity or t-closeness by >50% are reverted
- Detects pre-binned range columns from type-aware preprocessing and merges them by merging adjacent numeric ranges (not prefix truncation)
- For strings: tries prefix truncation to find the length closest to target cardinality; falls back to frequency-based top-K + "Other"
- For very high-range numeric columns (>100 unique): targets ~50 bins at level 0, halving each level; uses quantile binning when skew > 3.0
- Date-aware: detects datetime columns (native dtype, column_types hint, or string probing with ISO-first/dayfirst fallback) and routes to `_generalize_date_column` for proper quarter/month/year binning instead of prefix truncation
- Stall detection: exits after 2 consecutive iterations with no progress on large datasets (>10K rows), 3 on small datasets

**Suppression phase** (`_achieve_kanon_suppression`) — two phases:
- **Phase 1** (targeted): suppresses only the single highest-cardinality QI for each violation group, iterates, checks if suppressions from one round resolved other violations
- **Phase 2** (full-row): nulls out all QIs for still-violating groups
- If k-anonymity can't be achieved within budget, logs a warning recommending LOCSUPR instead

**Beam search** (`strategy='beam'`): Explores the generalization lattice with beam width B (default 5). Each state is a tuple of per-QI levels. At each step, expands top-B states by incrementing each QI → B×n_QI successors, scored by violations + total info loss. Finds lower-loss solutions than greedy by exploring alternative paths. Falls back to greedy if no solution found.

**Recursive local recoding** (`strategy='recursive'`): After standard generalization + suppression, identifies suppressed rows (NaN in QI columns), restores original values, re-runs generalization with more aggressive params (doubled bin_size per depth), merges recovered rows back. Post-merge k-anonymity check + cleanup suppression ensures validity. Up to 2 recursion levels.

**Date-aware generalization:** kANON's `_generalize_column` detects datetime columns through three probes: (1) native `datetime64` dtype, (2) `column_types` hint from the protection engine, (3) string date parsing (ISO format first, then `dayfirst=True` fallback for European dates). Detected date columns are routed to `_generalize_date_column()` from GENERALIZE.py, which bins into quarter/month/year periods instead of applying prefix truncation. This prevents dates from being truncated into meaningless strings like `"2023-0*"`.

**Escalation schedule:** k = 3 → 5 → 7 → 10 → 15 → 20 → 25 → 30. Subject to cardinality-aware k-pruning (values exceeding `max_achievable_k` removed) and smart start (skips early values based on ReID gap).

**Smart config** (pre-estimation): Estimates suppression rate before applying kANON. If estimated suppression >25%, may auto-switch to PRAM/NOISE. Also selects strategy (suppression-dominant / hybrid / generalization-dominant) and computes smart starting k. See §11.22.

**Per-QI tuning** (when `var_priority` available): `per_qi_bin_size` injected via `compute_risk_weighted_limits()` — HIGH-risk QIs get smaller bins (÷2, more generalization), LOW-risk QIs get larger bins (×1.5, less generalization).

**In pipelines:** Dynamic pipeline (1st step, k=5–7 hybrid), P4a/P4b (1st, k=5–7 generalization), GEO1 (2nd, k=5 hybrid).

**Fallback order:** LOCSUPR → PRAM → NOISE.

#### 2.3 LOCSUPR (Local Suppression) — The Surgeon

Suppression-only method. Replaces individual cells with NaN rather than generalizing values into ranges. The record stays in the dataset; only the problematic QI value disappears. Preserves more records at the cost of missing values.

**Execution path — R first, Python fallback:**

When called, LOCSUPR always tries R/sdcMicro first (if available and `use_r=True`). R's optimal algorithm produces roughly 61% fewer suppressions than the Python heuristic for the same k-anonymity level.

**R path** (`_apply_r_locsupr`):
- Converts the dataframe to R, runs `localSuppression(sdc, k=k_val)` from sdcMicro
- Converts NAs back carefully (R uses `-2147483648` as integer NA sentinel, and `NACharacterType` objects for string columns — both need special handling)
- Restores all non-QI columns from original data (sdcMicro may modify them)
- Has a hard per-column cap of 60%: if R suppressed more than 60% of any single QI column, that column is restored and a RuntimeError is raised — signalling the caller to try a different method

**Python fallback path** — four strategies for choosing which QI to suppress first:
- `minimum`: equal priority — suppress whichever QI resolves the most violations
- `weighted`: inverse of importance weights — less important QIs suppressed first
- `entropy`: higher cardinality = higher suppression priority
- `random`: random baseline

The loop is group-based (not record-by-record, for speed):
1. Find all violating equivalence classes (groups with fewer than k records)
2. Pick the highest-priority QI that still has values and hasn't hit its column cap
3. Bulk-suppress that QI for all violating records at once
4. Re-check k-anonymity and repeat

**Suppression budgets (Python):**
- Total cell budget: 20% of all cells (`n_records × n_qis`)
- Per-column cap: 60% of that column's non-null values — prevents any single QI from being wiped out
- Optional per-record cap (`max_suppressions_per_record`): limits how many QIs can be suppressed in a single record

**Escalation schedule:** k = 3 → 5 → 7 → 10 → 15 → 20. Subject to k-pruning and smart start.

**In pipelines:** Dynamic pipeline (tail cleanup — only when kANON absent, k=3; or k=7 for very high risk with kANON present), DYN_CAT (tail cleanup, k=3).

**Fallback order:** kANON → PRAM → NOISE.

**Cross-method start** (from kANON): kANON k=3→LOCSUPR k=3, kANON k=5→LOCSUPR k=5, kANON k=7–10→LOCSUPR k=7, kANON k=15+→LOCSUPR k=10.

**Cross-method start** (from LOCSUPR to others): LOCSUPR k=5→kANON k=5 / PRAM p=0.15 / NOISE mag=0.10, LOCSUPR k=7→kANON k=7 / PRAM p=0.20 / NOISE mag=0.15, LOCSUPR k=10→kANON k=10 / PRAM p=0.25 / NOISE mag=0.20.

#### 2.4 PRAM (Post-Randomization Method) — The Marginal Preserver

Perturbation method for categorical data. Swaps category values with controlled probability while preserving marginal distributions. **Does NOT reduce ReID** — used when risk is already low, for categorical-only data, or when categorical variables dominate at moderate risk.

**Smart config** (dominance detection): Before applying PRAM, checks each QI for category dominance (>80% single category). Dominated QIs are flagged as "low effectiveness". If ≥50% of QIs are dominated, auto-switches to kANON. See §11.22.

**Per-QI tuning** (when `var_priority` available): `per_variable_p_change` — HIGH-risk QIs get p_change × 1.5 (capped at 0.50), LOW-risk QIs get p_change × 0.6.

**Escalation schedule:** p_change = 0.10 → 0.15 → 0.20 → 0.25 → 0.30 → 0.35 → 0.40 → 0.50.

**In pipelines:** DYN_CAT (2nd, after NOISE, p_change=0.25–0.30), CAT2 (2nd, after NOISE, p_change=0.25–0.30), P4b (2nd, after kANON, p_change=0.20 — targets **sensitive columns**), P5 (2nd, after NOISE, p_change=0.30).

**Fallback order:** kANON → LOCSUPR → NOISE.

**Cross-method start** (when falling back from kANON): kANON k=3→PRAM p=0.10, kANON k=5→PRAM p=0.15, kANON k=7–10→PRAM p=0.20, kANON k=15→PRAM p=0.25.

#### 2.5 NOISE (Noise Addition) — The Continuous Protector

Perturbation method for numeric data. Adds calibrated random noise to continuous variables. Like PRAM, **does NOT reduce ReID** — used when risk is already low or to handle outliers.

**Smart config** (IQR-proportional): Before applying NOISE, computes per-variable magnitude proportional to each column's IQR (interquartile range). Columns with larger spread receive proportionally larger noise. Also checks for distribution distortion risk and pairwise correlation attenuation. See §11.22.

**Per-value 25% cap with distributional correction**: Each value's perturbation is capped at 25% of its absolute value — prevents small values in high-std columns from being destroyed (e.g., a price of 14K in a column with std 100K would otherwise get noise of ±50K). After capping, a mean-correction step adjusts uncapped values to restore the original column mean, preventing the systematic bias that capping introduces (capped values are pulled toward originals, shifting the overall mean).

**Escalation schedule:** magnitude = 0.05 → 0.10 → 0.15 → 0.20 → 0.25 → 0.30 → 0.40 → 0.50.

**In pipelines:** Dynamic pipeline (1st step when kANON not selected, magnitude=0.15–0.20), DYN_CAT (1st, magnitude=0.15–0.20), CAT2 (1st, magnitude=0.15–0.20), P5 (1st, magnitude=0.10–0.25 scaled by uniqueness).

**Fallback order:** kANON → PRAM → LOCSUPR.

**Cross-method start** (when falling back from kANON): kANON k=3→NOISE mag=0.05, kANON k=5→NOISE mag=0.10, kANON k=7–10→NOISE mag=0.15, kANON k=15→NOISE mag=0.20.

---

### 3. Method Selection Rules

The protection engine uses **4 methods** with a retry loop: Pipeline → Primary + Escalation → Fallbacks. Rules are evaluated in priority order (first match wins): Pipeline → HR6 → SR3 → RC → CAT → LDIV1 → DATE1 → QR (incl. QR0, MED1) → LOW → DP → HR → Default.

#### 3.0 Metric-Based Method Filtering

Before the rule chain executes, a **metric compatibility filter** checks every candidate method (primary, fallback, pipeline step) against the user's chosen risk metric. Incompatible methods are skipped — the rule engine falls through to the next matching rule.

| Target Metric | Allowed | Blocked | Rationale |
|--------------|---------|---------|-----------|
| ReID95 | kANON, LOCSUPR, PRAM, NOISE | — | ReID is universal; all methods reduce re-identification probability |
| k-Anonymity | kANON, LOCSUPR | PRAM, NOISE | Perturbation cannot guarantee minimum equivalence class size |
| Uniqueness | kANON, LOCSUPR | PRAM, NOISE | Perturbation may increase uniqueness (each perturbed value becomes distinct) |
| l-Diversity | kANON, LOCSUPR, PRAM | NOISE | PRAM on sensitive columns increases diversity; NOISE on QIs doesn't affect sensitive value diversity |

**Where filtering applies:**
- **Pipeline rules**: If any step in a pipeline is blocked, the entire pipeline is skipped
- **Single-method rules**: If the primary method is blocked, the rule is skipped
- **Fallback chains**: Blocked methods are stripped from `reid_fallback`, `utility_fallback`, alternatives, and `METHOD_FALLBACK_ORDER`
- **LDIV1 rule**: Gated — does not fire when metric is k-anonymity or uniqueness (it recommends PRAM)
- **Emergency fallback**: Uses kANON/LOCSUPR only — always passes all filters

#### 3.1 kANON — When Selected as Primary

| Rule | Condition | k | Strategy |
|------|-----------|---|----------|
| RC2 | Top 2 QIs ≥60% risk, ReID95 >15% | 5† | hybrid |
| RC3 | 3+ HIGH-risk QIs, ReID95 >15% | 7 or 10† | generalization |
| QR2 heavy | ReID95 >40%, heavy tail, est. suppression at k=7 ≤25% | 7† | hybrid |
| QR3 | Uniform high (ReID50 >20%) | 10† | generalization |
| QR4 high | Widespread, ReID95 >50% | 10† | generalization |
| QR4 moderate | Widespread, ReID50 >15%, ReID95 30–50% | 7† | hybrid |
| MED1 | Moderate spread, bimodal, or >10% high-risk | 5† | hybrid |
| LOW2 kANON | ReID95 ≤20%, continuous-dominant (excl. very-low or low+outliers → NOISE) | 3 or 5 | generalization |
| LOW3 | ReID95 ≤20%, mixed or high-cardinality | 3 or 5 | generalization |
| HR2 | Uniqueness >10%, no ReID available | 7† | hybrid |
| HR3 | Uniqueness >5% + ≥2 QIs, no ReID | 5† | generalization |
| DP3 | Sensitive attributes + ≥2 QIs | 5 | generalization |
| DEFAULT | Microdata with ≥2 QIs, no other match | 3 | generalization |
| EMERGENCY | Nothing matched at all | 5 | — |

†k values marked with † are passed through `_clamp_k_by_suppression()`, which may reduce k based on estimated suppression at that level. This prevents selecting a k that would cause excessive record loss.

**RC4 (pipeline):** 1 HIGH QI + 3+ LOW QIs, ReID95 >15% → GENERALIZE (bottleneck QI only, `max_categories=5`) → kANON (`k=3, strategy='hybrid'`). Listed separately because RC4 produces a two-step pipeline, not a single-method selection.

**QR0 — K-Anonymity Infeasible:** When the QI combination space far exceeds the number of records (expected equivalence class size < 3), k-anonymity is structurally infeasible. QR0 triggers `GENERALIZE_FIRST` — aggressive generalization (`max_categories=5`) applied as a preprocessing step before re-running method selection on the reduced data. If still infeasible after generalization, falls back to LOCSUPR k=3. Risk fallback: PRAM p=0.25. Also recommends specific QIs for removal based on cardinality.

**Suppression estimation:** Before selecting kANON at QR2 heavy tail (k=7), the engine estimates the fraction of records in equivalence classes smaller than k=7. If estimated suppression exceeds 25%, LOCSUPR is selected instead (QR2_Heavy_Tail_Low_Suppression) as it preserves more records. The same check now applies to MED1 (moderate risk): if estimated suppression at k=5 exceeds 25%, MED1_Moderate_High_Suppression selects LOCSUPR k=5 instead of kANON.

#### 3.2 LOCSUPR — When Selected as Primary

| Rule | Condition | k |
|------|-----------|---|
| SR3 | ≤2 QIs + max QI uniqueness >70% + ReID95 >20% (no var_priority needed) | 3 |
| RC1 | One QI dominates ≥40% of risk, ReID95 >15% | 5 |
| QR1 | Severe tail (ReID99/ReID50 ratio >10) | 5 |
| QR2 moderate | Tail risk, ReID95 30–40% | 3 |
| QR2 low-supp. | Heavy tail (ReID95 >40%) + est. suppression at k=7 >25% | 5 |
| MED1 high-supp. | Moderate risk (spread/bimodal/tail) + est. suppression at k=5 >25% | 5 |
| HR1 | Extreme uniqueness >20%, no ReID available | 5 |
| HR6 | Dataset <200 rows + ≥2 QIs (structural constraint, max 1 suppression/record) | 3 |

**As utility fallback** (weaker alternative when primary kANON hurts utility too much): MED1 (k=3), RC2 concentrated (k=5).

#### 3.3 PRAM — When Selected as Primary

| Rule | Condition | p_change |
|------|-----------|----------|
| CAT1 | ≥70% categorical, ReID95 15–40%, no near-constant QIs | 0.25–0.35 |
| LDIV1 | Sensitive column n_unique ≤5 + estimated min_l <2 (attribute disclosure risk) | 0.15 |
| LDIV1+DATE1 | LDIV1 conditions + ≥80% temporal QIs — merged PRAM on sensitive + date cols | 0.20–0.25 |
| DATE1 | ≥80% of QIs are temporal + ReID95 ≤40% (preserves temporal distributions) | 0.20–0.25 |
| DP4 | Integer-coded categorical QIs (≤15 unique ints) + ReID95 ≤30% | 0.20–0.30 |
| LOW1 | ReID95 ≤10%, categorical-dominant (≥60%), low cardinality | 0.15–0.20 |
| HR4 | Very small dataset (<100 records), no ReID | 0.30 |
| HR5 small (no cont.) | Small dataset (100–500), no continuous QIs, no ReID | 0.25 |
| DP2 | ≥2 skewed columns | 0.20 |
| DEFAULT categorical | Mostly categorical, no other match | 0.20 |
| DEFAULT fallback | Nothing else matched | 0.20 |

#### 3.4 NOISE — When Selected as Primary

| Rule | Condition | magnitude |
|------|-----------|-----------|
| LOW2 noise | ReID95 ≤5% (very low) or ≤10% with outliers, continuous-dominant | 0.15–0.20 |
| HR5 small (cont.) | Small dataset (100–500), has continuous QIs, no ReID | 0.15 |
| DP1 | Outliers present + continuous data | 0.20 |
| DEFAULT continuous | Mostly continuous, no other match | 0.15 |

#### 3.5 Rule Design Rationale

**CAT1 — Categorical-aware selection at moderate risk:** When ≥70% of variables are categorical and risk is moderate (ReID95 15–40%), PRAM is selected instead of kANON. PRAM preserves all records while kANON would typically suppress 15–25%. The rule checks that no QI has a near-constant category (≥80% frequency), since PRAM is ineffective on concentrated distributions. Risk dependency: CAT1 requires `has_reid=True` — if risk computation was skipped, CAT1 does not fire and the dataset falls through to HR or default rules. Users who skip risk computation with categorical-dominant data should expect kANON or PRAM via a less-targeted route.

**SR3 — Near-unique QI with few QIs:** When only 1–2 QIs exist and one has >70% uniqueness, kANON would collapse the entire column. LOCSUPR k=3 with targeted importance weighting is more surgical. Fires without var_priority, catching cases RC1 misses when backward elimination hasn't run.

**RC4 — Single bottleneck QI:** When exactly 1 HIGH-risk QI coexists with 3+ LOW-risk QIs, targeted generalization of only the bottleneck QI followed by light kANON avoids unnecessary modification of the low-risk QIs.

**LDIV1 — L-diversity gap:** When a sensitive column has ≤5 distinct values, most k-anonymous equivalence classes will be homogeneous (l≈1), offering no protection against attribute disclosure. LDIV1 recommends light PRAM on the sensitive column itself (p=0.15). **Co-fire merge:** When DATE1 conditions are also met (≥80% temporal QIs), LDIV1 and DATE1 are merged into a single `LDIV1_DATE1_Merged` rule — PRAM is applied to the union of sensitive columns and date QIs with the higher p_change (0.20–0.25), avoiding the first-match-wins problem where LDIV1 would block DATE1.

**DATE1 — Temporal-dominant QIs:** When ≥80% of QIs are date/temporal, kANON generalization produces overlapping date ranges that are hard to interpret. PRAM on binned date columns preserves the temporal distribution shape. Note: if LDIV1 also applies, the two rules are merged (see LDIV1 above).

**DP4 — Integer-coded categoricals:** Numeric columns with ≤15 unique integer values (e.g. municipality_code, education_level) are categorical codes. kANON range-binning ("1–5") destroys the coding structure. PRAM preserves it with p_change scaled to the number of categories.

**HR6 — Very small dataset:** Datasets under 200 rows cannot support k≥5 without catastrophic suppression. HR6 fires early (before all other risk rules) and uses LOCSUPR k=3 with max 1 suppression per record. Issues a strong warning recommending synthetic data release.

#### 3.6 Pipeline Rules (Multi-Method Combinations)

Pipelines run two or more methods sequentially when single methods are demonstrably insufficient. Checked before any single-method rules.

**Dynamic Pipeline Builder** — Assembles pipelines from data features rather than matching hardcoded patterns. Triggers when ReID95 >15% and mixed types benefit from multi-method treatment:

1. **Categorical guard** — If ≥70% categorical, defers to CAT1 (PRAM). If 50–70% categorical with ≥1 continuous, builds DYN_CAT variant (NOISE + PRAM + optional LOCSUPR tail).
2. **GEO1** — If 2+ geographic QIs with both fine-grained (>50 unique) and coarse (≤50 unique), uses GENERALIZE (fine geos) → kANON.
3. **kANON** added when ReID95 >20% (k=7 if ReID95 >40%, else k=5; strategy='hybrid')
4. **NOISE** added when continuous QIs have outliers **and** kANON was not already added (magnitude scaled by risk level)
5. **LOCSUPR** added only when kANON is **not** in the pipeline and >15% of records at high risk (k=3). Exception: added at k=7 if kANON is present but high_risk_rate >30% and estimated suppression at k=7 is <40%.

Note: kANON and NOISE are mutually exclusive in the dynamic builder — if kANON is selected (step 3), the NOISE step is skipped. LOCSUPR k=3 is no longer added after kANON k≥5 (redundant — kANON already satisfies k=3). Only fires when 2+ methods are assembled. Replaces the previous P1, P2a, P2b, P3, P6 hardcoded patterns.

**Legacy Pipelines** (edge cases not covered by dynamic builder):

| Pipeline | Trigger | Method Sequence | Rationale |
|----------|---------|-----------------|-----------|
| **CAT2** | 50–70% categorical + ReID95 15–50% + ≥1 continuous | NOISE → PRAM | Split: numerics get noise, categoricals get perturbation. Parameters scale with risk (p_change 0.25–0.30, magnitude 0.15–0.20). |
| **P4b** | ≥2 skewed columns + sensitive attributes (diversity ≤10) + ≥2 QIs | kANON → PRAM | Structure for QIs, PRAM targets **sensitive columns** (not skewed QIs) |
| **P4a** | ≥2 skewed columns + sensitive attributes (diversity >10) + ≥2 QIs | kANON only | High-diversity sensitive cols — no PRAM (would target wrong columns) |
| **P5** | Density <5 (records/QI-space) + ≥200 rows + uniqueness >15% + ≥2 continuous + ≥2 categorical | NOISE → PRAM | Sparse mixed dataset — NOISE magnitude scaled by uniqueness (0.10–0.25). Datasets <200 rows deferred to HR6. |

---

### 4. Engine Orchestration

#### 4.1 Treatment Balance

When users assign treatment levels (Heavy/Standard/Light) to QIs via the Configure tab, the engine adjusts method parameters post-selection:

- **≥60% Heavy** → k bumped +2 (or p_change/magnitude +0.05) for stronger protection
- **≥60% Light** → k reduced -1 (or p_change -0.05, magnitude -0.03) for more utility preservation
- **Mixed** → no adjustment; rule defaults apply

Treatment balance is applied to both `select_method_by_features()` and `select_method_suite()` output before the engine starts its retry loop.

#### 4.2 Bidirectional Cross-Method Starts

When a primary method fails and the engine falls back to a different method type, it uses intelligent starting parameters rather than defaults:

- **Structural → Perturbative**: kANON k=5 → PRAM p=0.15, LOCSUPR k=7 → NOISE mag=0.15, etc.
- **Perturbative → Structural**: Uses the current ReID risk gap to pick structural starting point. E.g., if PRAM reduced ReID from 25% to 12% and target is 5%, the gap is 7% → kANON starts at k=5.

Gap thresholds: ≤5% → k=3, ≤15% → k=5, ≤30% → k=7, >30% → k=10.

#### 4.3 Retry Loop

The retry loop in `run_rules_engine_protection()`:

0. **Feasibility diagnosis** — `diagnose_qis()` runs upfront to compute `max_achievable_k`. All k-based escalation schedules are pruned accordingly.
1. **Pipeline** — If `select_method_suite()` returns a pipeline (dynamic or legacy), run methods sequentially. Mid-pipeline risk check runs after each structural/GENERALIZE step: if risk drops below target × multiplier (1.1× for kANON/LOCSUPR, 1.2× for GENERALIZE), remaining steps are skipped. The check is skipped entirely after perturbative steps (PRAM/NOISE) since they don't reduce structural risk and the computation would be wasted. If targets met → done.
1b. **GENERALIZE_FIRST** (conditional) — If QR0 triggered (k-anonymity infeasible and ReID95 >15%), restore from pre-pipeline snapshot (ensuring generalization starts from clean original data, not pipeline-modified data), apply aggressive generalization (`max_categories=5`), rebuild features on the generalized data, and re-select method. If still infeasible → fall back to LOCSUPR k=3.
2. **Primary + Escalation** — Try primary method at initial params. If fails, escalate via tuning schedule (smart start skips early values, k-pruning removes impossible values). Guards: QI over-suppression (any QI >40% suppressed → skip to fallbacks), plateau detection (stop if ReID doesn't improve after N attempts), time budget (30s max per phase), utility floor.
3. **Fallbacks** — Try each fallback method with bidirectional cross-method start params, per-QI injection, and its own escalation schedule. Up to 5 fallback methods tried. Perturbation-only methods (PRAM/NOISE) are filtered out if ReID is far above target (they cannot close a structural gap). Current ReID is tracked across fallbacks for accurate gap calculation.
4. **Feasibility suggestion** — If all methods fail, `ensure_feasibility()` suggests QI removal or binning (advisory only — user must confirm).

#### 4.4 Suppression Estimation

Both `extract_data_features_with_reid()` and `build_data_features()` compute `estimated_suppression` — a dict mapping k values (3, 5, 7) to the fraction of records in equivalence classes smaller than that k. Computed from a single groupby pass. QR2 uses the k=7 estimate when deciding whether to switch from kANON k=7 to LOCSUPR (threshold: 25% suppression). CAT1 reports the k=5 estimate in its reasoning. The legacy key `estimated_suppression_k5` is preserved for backward compatibility.

#### 4.5 Post-Protection

`_attach_diagnostics()` adds per-QI utility comparison, l-diversity check, entropy l-diversity, t-closeness, ML utility validation (§11.28), method quality assessment, timing, feasibility suggestions, and failure guidance — all surfaced in the UI as metric cards and collapsible diagnostics HTML (Section 11.21).

---

*For technical architecture details, see [docs/sdc_pipeline_architecture.md](sdc_pipeline_architecture.md)*
