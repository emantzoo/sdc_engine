"""
Spec 19 Phase 1.1 — Feature Extractor Diff

Runs both build_data_features() and extract_data_features_with_reid() on the
same datasets and produces a structured diff of their output feature dicts.

Datasets:
  1. Adult (harness dataset) — 3 QIs: age, workclass, education
  2. Synthetic sensitive-column dataset (G1/G4 shape)
  3. Synthetic G10-shape dataset (2 QIs, high uniqueness)

Usage:
  cd sdc_engine && python scripts/spec_19_feature_diff.py
"""

import sys
import os
import json
import pandas as pd
import numpy as np

# Ensure sdc_engine is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdc.protection_engine import build_data_features
from sdc.selection.features import extract_data_features_with_reid
from sdc.sdc_utils import calculate_reid, analyze_data


def make_analysis_dict(data, quasi_identifiers, sensitive_columns=None):
    """Build an analysis dict compatible with extract_data_features_with_reid.

    Calls analyze_data() and patches in sensitive_columns if provided.
    """
    analysis = analyze_data(data, quasi_identifiers, verbose=False)

    # analyze_data uses auto_detect_sensitive_columns; override if explicit
    if sensitive_columns is not None:
        sens_dict = {}
        for sc in sensitive_columns:
            sens_dict[sc] = f"explicit sensitive column ({sc})"
        analysis['sensitive_columns'] = sens_dict

    # Ensure uniqueness_rate is top-level (extract_data_features reads it)
    if 'uniqueness_rate' not in analysis:
        dr = analysis.get('disclosure_risk', {})
        analysis['uniqueness_rate'] = dr.get('uniqueness_rate', 0)

    return analysis


def compute_reid_for_build(data, quasi_identifiers):
    """Compute ReID dict for build_data_features (which expects pre-computed reid)."""
    try:
        return calculate_reid(data, quasi_identifiers)
    except Exception:
        return {}


def type_str(val):
    """Short type description."""
    if val is None:
        return "None"
    t = type(val).__name__
    if isinstance(val, list):
        return f"list[{len(val)}]"
    elif isinstance(val, dict):
        return f"dict[{len(val)}]"
    return t


def compare_values(v1, v2, key):
    """Compare two values. Returns (match_status, detail)."""
    if v1 is None and v2 is None:
        return "exact", ""

    # Both missing from their respective function
    # (handled at caller level)

    t1, t2 = type(v1).__name__, type(v2).__name__

    if isinstance(v1, float) and isinstance(v2, float):
        if abs(v1 - v2) < 1e-9:
            return "exact", ""
        elif abs(v1 - v2) < 0.01:
            return "close", f"build={v1:.6f}, reid={v2:.6f}"
        else:
            return "value_mismatch", f"build={v1:.4f}, reid={v2:.4f}"
    elif isinstance(v1, (int, np.integer)) and isinstance(v2, (int, np.integer)):
        if int(v1) == int(v2):
            return "exact", ""
        else:
            return "value_mismatch", f"build={v1}, reid={v2}"
    elif isinstance(v1, str) and isinstance(v2, str):
        if v1 == v2:
            return "exact", ""
        else:
            return "value_mismatch", f"build='{v1}', reid='{v2}'"
    elif isinstance(v1, bool) and isinstance(v2, bool):
        if v1 == v2:
            return "exact", ""
        else:
            return "value_mismatch", f"build={v1}, reid={v2}"
    elif isinstance(v1, list) and isinstance(v2, list):
        if sorted(str(x) for x in v1) == sorted(str(x) for x in v2):
            return "exact", ""
        else:
            return "value_mismatch", f"build={v1}, reid={v2}"
    elif isinstance(v1, dict) and isinstance(v2, dict):
        if v1 == v2:
            return "exact", ""
        # Check subset differences
        diff_keys = set(v1.keys()) ^ set(v2.keys())
        val_diffs = {k: (v1.get(k), v2.get(k)) for k in set(v1) & set(v2) if v1[k] != v2[k]}
        detail_parts = []
        if diff_keys:
            detail_parts.append(f"key_diff={diff_keys}")
        if val_diffs:
            detail_parts.append(f"val_diffs={val_diffs}")
        return "value_mismatch", "; ".join(detail_parts)
    elif type(v1) != type(v2):
        return "type_mismatch", f"build={t1}({v1}), reid={t2}({v2})"
    else:
        if v1 == v2:
            return "exact", ""
        return "value_mismatch", f"build={v1}, reid={v2}"


def diff_features(build_feats, reid_feats, dataset_name):
    """Produce a diff table between two feature dicts."""
    all_keys = sorted(set(build_feats.keys()) | set(reid_feats.keys()))

    rows = []
    for key in all_keys:
        in_build = key in build_feats
        in_reid = key in reid_feats

        if in_build and in_reid:
            bv, rv = build_feats[key], reid_feats[key]
            match, detail = compare_values(bv, rv, key)
            rows.append({
                'key': key,
                'type_build': type_str(bv),
                'type_reid': type_str(rv),
                'in_build': 'yes',
                'in_reid': 'yes',
                'match': match,
                'detail': detail,
            })
        elif in_build and not in_reid:
            bv = build_feats[key]
            rows.append({
                'key': key,
                'type_build': type_str(bv),
                'type_reid': '-',
                'in_build': 'yes',
                'in_reid': 'no',
                'match': 'build_only',
                'detail': f"value={bv}" if not isinstance(bv, (dict, list)) or len(str(bv)) < 60 else f"type={type_str(bv)}",
            })
        else:
            rv = reid_feats[key]
            rows.append({
                'key': key,
                'type_build': '-',
                'type_reid': type_str(rv),
                'in_build': 'no',
                'in_reid': 'yes',
                'match': 'reid_only',
                'detail': f"value={rv}" if not isinstance(rv, (dict, list)) or len(str(rv)) < 60 else f"type={type_str(rv)}",
            })

    return rows


def create_dataset_adult():
    """Load adult dataset with 3 QIs."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             '..', 'tests', 'data', 'adult.xlsx')
    data_path = os.path.normpath(data_path)
    df = pd.read_excel(data_path)
    qis = ['age', 'workclass', 'education']
    # Verify QIs exist
    for qi in qis:
        assert qi in df.columns, f"QI '{qi}' not found in adult. Columns: {list(df.columns)}"
    return df, qis, "adult (3 QIs: age, workclass, education)", None


def create_dataset_sensitive():
    """Synthetic dataset with sensitive columns (G1/G4 shape).

    - 500 rows
    - 3 QIs: region (cat), age_group (cat), income_level (cat)
    - 1 sensitive column: diagnosis
    """
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
    """Synthetic G10-shape dataset (2 QIs, high uniqueness).

    - 200 rows
    - 2 QIs: id_code (near-unique numeric), category (low-card categorical)
    - max_qi_uniqueness should be very high (~1.0)
    """
    np.random.seed(123)
    n = 200
    df = pd.DataFrame({
        'id_code': range(1, n + 1),  # near-unique
        'category': np.random.choice(['A', 'B', 'C'], n),
    })
    qis = ['id_code', 'category']
    return df, qis, "synthetic G10-shape (200 rows, 2 QIs, near-unique)", None


def run_one_dataset(data, qis, name, sensitive_columns=None):
    """Run both functions and return diff rows."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {name}")
    print(f"  Rows: {len(data)}, QIs: {qis}")
    if sensitive_columns:
        print(f"  Sensitive columns: {sensitive_columns}")
    print(f"{'='*60}")

    # 1. build_data_features
    reid = compute_reid_for_build(data, qis)
    build_feats = build_data_features(
        data, qis,
        reid=reid,
        sensitive_columns=sensitive_columns,
    )

    # 2. extract_data_features_with_reid
    analysis = make_analysis_dict(data, qis, sensitive_columns)
    reid_feats = extract_data_features_with_reid(data, analysis, qis)

    # Diff
    rows = diff_features(build_feats, reid_feats, name)

    # Print summary
    exact = sum(1 for r in rows if r['match'] == 'exact')
    close = sum(1 for r in rows if r['match'] == 'close')
    val_mm = sum(1 for r in rows if r['match'] == 'value_mismatch')
    type_mm = sum(1 for r in rows if r['match'] == 'type_mismatch')
    build_only = sum(1 for r in rows if r['match'] == 'build_only')
    reid_only = sum(1 for r in rows if r['match'] == 'reid_only')

    print(f"\n  Total keys: {len(rows)}")
    print(f"  Exact match: {exact}")
    print(f"  Close match: {close}")
    print(f"  Value mismatch: {val_mm}")
    print(f"  Type mismatch: {type_mm}")
    print(f"  build_data_features only: {build_only}")
    print(f"  extract_data_features_with_reid only: {reid_only}")

    # Print non-exact details
    for r in rows:
        if r['match'] != 'exact':
            print(f"    [{r['match']}] {r['key']}: {r['detail']}")

    return rows, build_feats, reid_feats


def main():
    datasets = [
        create_dataset_adult(),
        create_dataset_sensitive(),
        create_dataset_g10(),
    ]

    all_results = {}
    all_build = {}
    all_reid = {}
    for data, qis, name, sens in datasets:
        rows, bf, rf = run_one_dataset(data, qis, name, sens)
        all_results[name] = rows
        all_build[name] = bf
        all_reid[name] = rf

    # Dump raw dicts for inspection
    print("\n\n" + "="*60)
    print("  RAW FEATURE DICTS (JSON)")
    print("="*60)

    for name in all_results:
        print(f"\n--- {name} ---")
        print(f"\nbuild_data_features keys ({len(all_build[name])}):")
        for k in sorted(all_build[name].keys()):
            v = all_build[name][k]
            if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
                print(f"  {k}: <{type(v).__name__} shape={getattr(v, 'shape', '?')}>")
            else:
                vstr = str(v)
                if len(vstr) > 100:
                    vstr = vstr[:100] + "..."
                print(f"  {k}: {vstr}")

        print(f"\nextract_data_features_with_reid keys ({len(all_reid[name])}):")
        for k in sorted(all_reid[name].keys()):
            v = all_reid[name][k]
            if isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
                print(f"  {k}: <{type(v).__name__} shape={getattr(v, 'shape', '?')}>")
            else:
                vstr = str(v)
                if len(vstr) > 100:
                    vstr = vstr[:100] + "..."
                print(f"  {k}: {vstr}")


if __name__ == '__main__':
    main()
