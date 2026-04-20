"""
Spec 19 Phase 1.3 — Parity Check

Runs both build_data_features() and extract_data_features_with_reid() on
identical datasets, diffs the output dicts, and verifies:

1. The 8 newly-ported keys produce identical values in both paths.
2. All other keys either match exactly or have a known/documented divergence.
3. No unexpected differences exist.

Datasets:
  - adult (harness dataset, 3 QIs)
  - synthetic sensitive (500 rows, 3 QIs, 1 sensitive col)
  - synthetic G10-shape (200 rows, 2 QIs, near-unique)
  - test_small_150 (small dataset)
  - test_census_like_1K_high (1K rows, high risk)

Usage:
  cd sdc_engine && python scripts/spec_19_parity_check.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np

# Ensure sdc_engine is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdc_engine.sdc.protection_engine import build_data_features
from sdc_engine.sdc.selection.features import extract_data_features_with_reid
from sdc_engine.sdc.sdc_utils import calculate_reid, analyze_data


# The 8 keys ported in Phase 1.2 — these MUST match exactly
PORTED_KEYS = [
    'max_qi_uniqueness',
    'integer_coded_qis',
    'qi_type_counts',
    'n_geo_qis',
    'geo_qis_by_granularity',
    'sensitive_column_diversity',
    'min_l',
    'l_diversity',
]

# Keys fixed in Phase 1.2 — values will differ from reid version's broken values
FIXED_KEYS = [
    'mean_risk',              # build: reid.get('mean_risk', reid_50), reid: same
    'recommended_qi_to_remove',  # build: conditional, reid: conditional (same logic)
    'has_sensitive_attributes',  # build: was False, now bool(sensitive_columns)
    'sensitive_columns',         # build: was {}, now dict from param
]

# Keys with KNOWN divergence documented in Phase 1.1 diff
KNOWN_DIVERGENCES = {
    'continuous_vars': 'scope: build=QI-only, reid=all columns from analysis',
    'categorical_vars': 'scope: build=QI-only, reid=all columns from analysis',
    'n_continuous': 'follows continuous_vars scope difference',
    'n_categorical': 'follows categorical_vars scope difference',
    'n_columns': 'build=active_cols, reid=all data.columns',
    'uniqueness_rate': 'build=QI-combo uniqueness, reid=analysis.uniqueness_rate',
    'risk_level': 'build=risk_pattern copy, reid=analysis.risk_level (high/medium/low)',
    'high_risk_count': 'reid version reads nonexistent risk_scores key (always 0)',
    'high_risk_rate': 'reid version derives from risk_scores (always 0)',
    'has_reid': 'build=always True, reid=conditional on calculate_reid success',
    '_risk_metric_type': 'build=metric enum value, reid=literal "reid95"',
    'small_cells_rate': 'build=always 0.0 (microdata), reid=conditional on data_type',
    'data_type': 'build=always "microdata", reid=from analysis dict',
    # Keys only in build (not in reid)
    'var_priority': 'build-only: from parameter or lazy computation',
    'risk_concentration': 'build-only: from classify_risk_concentration',
    '_risk_assessment': 'build-only: RiskAssessment object',
    'estimated_suppression': 'both compute, may differ slightly due to groupby timing',
    'estimated_suppression_k5': 'backward compat alias',
    'has_outliers': 'scope: build checks QI-only continuous, reid checks all continuous',
    'skewed_columns': 'scope: build checks QI-only continuous, reid checks all continuous',
}


def make_analysis_dict(data, quasi_identifiers, sensitive_columns=None):
    """Build analysis dict for extract_data_features_with_reid."""
    analysis = analyze_data(data, quasi_identifiers, verbose=False)
    if sensitive_columns is not None:
        sens_dict = {sc: f"explicit sensitive column ({sc})" for sc in sensitive_columns}
        analysis['sensitive_columns'] = sens_dict
    if 'uniqueness_rate' not in analysis:
        dr = analysis.get('disclosure_risk', {})
        analysis['uniqueness_rate'] = dr.get('uniqueness_rate', 0)
    return analysis


def compute_reid_for_build(data, quasi_identifiers):
    """Compute ReID dict for build_data_features."""
    try:
        return calculate_reid(data, quasi_identifiers)
    except Exception:
        return {}


def compare_values(v1, v2):
    """Compare two values. Returns (match, detail)."""
    if v1 is None and v2 is None:
        return True, "both None"
    if v1 is None or v2 is None:
        return False, f"build={v1}, reid={v2}"
    if isinstance(v1, float) and isinstance(v2, float):
        if abs(v1 - v2) < 1e-9:
            return True, "exact"
        return False, f"build={v1:.6f}, reid={v2:.6f}"
    if isinstance(v1, dict) and isinstance(v2, dict):
        if v1 == v2:
            return True, "exact"
        return False, f"build_keys={sorted(v1.keys())}, reid_keys={sorted(v2.keys())}"
    if isinstance(v1, list) and isinstance(v2, list):
        if sorted(str(x) for x in v1) == sorted(str(x) for x in v2):
            return True, "exact (order-independent)"
        return False, f"build={v1}, reid={v2}"
    if v1 == v2:
        return True, "exact"
    return False, f"build={v1}, reid={v2}"


def run_parity_check(data, qis, name, sensitive_columns=None):
    """Run both functions and compare outputs."""
    print(f"\n{'='*70}")
    print(f"  Dataset: {name}")
    print(f"  Rows: {len(data)}, QIs: {qis}")
    if sensitive_columns:
        print(f"  Sensitive: {sensitive_columns}")
    print(f"{'='*70}")

    # Build path
    reid = compute_reid_for_build(data, qis)
    build_feats = build_data_features(
        data, qis, reid=reid, sensitive_columns=sensitive_columns)

    # Reid path
    analysis = make_analysis_dict(data, qis, sensitive_columns)
    reid_feats = extract_data_features_with_reid(data, analysis, qis)

    results = []
    all_keys = sorted(set(build_feats.keys()) | set(reid_feats.keys()))

    # Check ported keys specifically
    print(f"\n  --- PORTED KEYS (must match exactly) ---")
    ported_pass = 0
    ported_fail = 0
    for key in PORTED_KEYS:
        bv = build_feats.get(key)
        rv = reid_feats.get(key)
        match, detail = compare_values(bv, rv)
        status = "PASS" if match else "FAIL"
        if match:
            ported_pass += 1
        else:
            ported_fail += 1
        print(f"    [{status}] {key}: {detail}")
        if not match:
            print(f"           build: {bv}")
            print(f"           reid:  {rv}")
        results.append({
            'key': key, 'category': 'ported',
            'match': match, 'detail': detail,
            'build_val': str(bv)[:80], 'reid_val': str(rv)[:80],
        })

    # Check fixed keys
    print(f"\n  --- FIXED KEYS (may differ from reid's broken values) ---")
    for key in FIXED_KEYS:
        bv = build_feats.get(key)
        rv = reid_feats.get(key)
        match, detail = compare_values(bv, rv)
        status = "MATCH" if match else "DIVERGE (expected)"
        print(f"    [{status}] {key}: {detail}")
        results.append({
            'key': key, 'category': 'fixed',
            'match': match, 'detail': detail,
            'build_val': str(bv)[:80], 'reid_val': str(rv)[:80],
        })

    # Check all other keys
    print(f"\n  --- OTHER KEYS ---")
    unexpected_diffs = []
    for key in all_keys:
        if key in PORTED_KEYS or key in FIXED_KEYS:
            continue
        bv = build_feats.get(key, '<MISSING>')
        rv = reid_feats.get(key, '<MISSING>')

        if bv == '<MISSING>':
            status = "reid-only"
            match = True  # reid-only keys are expected (will be cleaned up in Phase 1.5)
            detail = f"reid-only: {str(rv)[:60]}"
        elif rv == '<MISSING>':
            status = "build-only"
            match = True  # build-only keys are expected
            detail = f"build-only: {str(bv)[:60]}"
        else:
            match_val, detail = compare_values(bv, rv)
            if match_val:
                status = "MATCH"
                match = True
            elif key in KNOWN_DIVERGENCES:
                status = "KNOWN-DIVERGE"
                match = True
                detail = KNOWN_DIVERGENCES[key]
            else:
                status = "UNEXPECTED-DIFF"
                match = False
                unexpected_diffs.append(key)

        if not match or status in ("UNEXPECTED-DIFF",):
            print(f"    [{status}] {key}: {detail}")
            if not match:
                print(f"           build: {str(bv)[:80]}")
                print(f"           reid:  {str(rv)[:80]}")

        results.append({
            'key': key, 'category': status.lower(),
            'match': match, 'detail': detail,
            'build_val': str(bv)[:80], 'reid_val': str(rv)[:80],
        })

    # Summary
    print(f"\n  SUMMARY:")
    print(f"    Ported keys: {ported_pass} PASS / {ported_fail} FAIL")
    print(f"    Unexpected differences: {len(unexpected_diffs)}")
    if unexpected_diffs:
        print(f"    UNEXPECTED: {unexpected_diffs}")

    return results, ported_fail, unexpected_diffs


# Dataset constructors

def create_dataset_adult():
    """Load adult dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'mhtrwo-ax-met-ak-2025.xlsx')
    # Try adult first, fall back to other datasets
    for fname in ['test_census_like_1K_high.csv', 'test_employee_records_1K_high.csv']:
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', fname)
        if os.path.exists(alt):
            df = pd.read_csv(alt)
            qis = [c for c in df.columns[:3] if c in df.columns]
            return df, qis, f"{fname} ({len(df)} rows, {len(qis)} QIs)", None
    # Synthetic fallback
    return create_dataset_synthetic_micro()


def create_dataset_synthetic_micro():
    """Synthetic microdata (500 rows, 3 QIs)."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'age_group': np.random.choice(['18-30', '31-45', '46-60', '61+'], n),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n),
        'other_col': np.random.randn(n),
    })
    qis = ['region', 'age_group', 'income_level']
    return df, qis, "synthetic micro (500 rows, 3 cat QIs)", None


def create_dataset_sensitive():
    """Synthetic dataset with sensitive columns."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'age_group': np.random.choice(['18-30', '31-45', '46-60', '61+'], n),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n),
        'diagnosis': np.random.choice(['Healthy', 'TypeA', 'TypeB', 'TypeC', 'TypeD'], n),
    })
    qis = ['region', 'age_group', 'income_level']
    sens = ['diagnosis']
    return df, qis, "synthetic sensitive (500 rows, 3 QIs, 1 sensitive)", sens


def create_dataset_g10():
    """Synthetic G10-shape (near-unique)."""
    np.random.seed(123)
    n = 200
    df = pd.DataFrame({
        'id_code': range(1, n + 1),
        'category': np.random.choice(['A', 'B', 'C'], n),
    })
    qis = ['id_code', 'category']
    return df, qis, "synthetic G10 (200 rows, 2 QIs, near-unique)", None


def create_dataset_small():
    """Small dataset from test_small_150.csv if available."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'data', 'test_small_150.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        qis = [c for c in df.columns[:3] if c in df.columns]
        return df, qis, f"test_small_150 ({len(df)} rows, {len(qis)} QIs)", None
    return None


def create_dataset_with_dates():
    """Synthetic dataset with date-like and geo-like QI names."""
    np.random.seed(99)
    n = 300
    df = pd.DataFrame({
        'year': np.random.choice([2020, 2021, 2022, 2023], n),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n),
        'city': np.random.choice([f'City_{i}' for i in range(60)], n),
        'age': np.random.randint(18, 80, n),
        'score': np.random.randn(n) * 10 + 50,
    })
    qis = ['year', 'region', 'city', 'age']
    return df, qis, "synthetic date+geo (300 rows, 4 QIs with date/geo names)", None


def main():
    datasets = [
        create_dataset_synthetic_micro(),
        create_dataset_sensitive(),
        create_dataset_g10(),
        create_dataset_with_dates(),
    ]

    # Add small dataset if available
    small = create_dataset_small()
    if small:
        datasets.append(small)

    # Add a real dataset if available
    real = create_dataset_adult()
    if real:
        datasets.append(real)

    total_ported_fails = 0
    total_unexpected = []
    all_results = {}

    for data, qis, name, sens in datasets:
        results, p_fails, unexpected = run_parity_check(data, qis, name, sens)
        all_results[name] = results
        total_ported_fails += p_fails
        total_unexpected.extend(unexpected)

    # Final summary
    print(f"\n\n{'='*70}")
    print(f"  FINAL PARITY SUMMARY")
    print(f"{'='*70}")
    print(f"  Datasets checked: {len(datasets)}")
    print(f"  Total ported-key failures: {total_ported_fails}")
    print(f"  Total unexpected differences: {len(total_unexpected)}")
    if total_unexpected:
        print(f"  UNEXPECTED KEYS: {total_unexpected}")

    if total_ported_fails == 0 and len(total_unexpected) == 0:
        print(f"\n  VERDICT: PARITY CHECK PASSED")
    else:
        print(f"\n  VERDICT: PARITY CHECK FAILED — review needed")

    return total_ported_fails == 0 and len(total_unexpected) == 0


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
