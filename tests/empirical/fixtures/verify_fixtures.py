"""
Spec 16a -- Verify each fixture triggers its intended rule.

Runs feature extraction + select_method_suite() for each fixture
and reports whether the expected rule fires.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from sdc_engine.sdc.protection_engine import build_data_features
from sdc_engine.sdc.selection.pipelines import select_method_suite

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Geo hints used by features.py for GEO1 detection
_GEO_HINTS = {'region', 'city', 'state', 'country', 'zip', 'postal',
              'county', 'district', 'municipality', 'prefecture'}


def _inject_geo_features(features, df, quasi_identifiers):
    """Inject geo_qis_by_granularity (computed by features.py but not protection_engine)."""
    geo_qis_by_granularity = {}
    for qi in quasi_identifiers:
        if qi not in df.columns:
            continue
        if any(h in qi.lower() for h in _GEO_HINTS):
            card = df[qi].nunique()
            geo_qis_by_granularity[qi] = 'fine' if card > 50 else 'coarse'
    features['geo_qis_by_granularity'] = geo_qis_by_granularity


def verify(name, csv_name, quasi_identifiers, expected_rule, *,
           risk_metric='reid95', sensitive_columns=None,
           feature_overrides=None, inject_geo=False):
    """Run feature extraction + method selection, return (actual_rule, match)."""
    df = pd.read_csv(DATA_DIR / csv_name)
    features = build_data_features(df, quasi_identifiers)
    features['_risk_metric_type'] = risk_metric

    # Inject geo features (build_data_features doesn't compute these)
    if inject_geo:
        _inject_geo_features(features, df, quasi_identifiers)

    # Inject sensitive column info if provided
    if sensitive_columns:
        features['has_sensitive_attributes'] = True
        features['sensitive_columns'] = {c: {} for c in sensitive_columns}
        # Compute sensitive_column_diversity
        min_div = None
        for sc in sensitive_columns:
            if sc in df.columns:
                nu = df[sc].nunique()
                if min_div is None or nu < min_div:
                    min_div = nu
        features['sensitive_column_diversity'] = min_div

    if feature_overrides:
        features.update(feature_overrides)

    suite = select_method_suite(features, access_tier='SCIENTIFIC', verbose=False)
    actual_rule = suite.get('rule_applied', 'NONE')
    pipeline = suite.get('pipeline')
    method = suite.get('primary_method') or (pipeline[0] if pipeline else 'NONE')
    use_pipe = suite.get('use_pipeline', False)

    match = expected_rule in actual_rule
    status = "PASS" if match else "FAIL"

    print(f"  [{status}] {name:20s} expected={expected_rule:30s} actual={actual_rule:35s} "
          f"{'pipeline=' + str(pipeline) if use_pipe else 'method=' + method}")

    if not match:
        # Debug info
        total = features['n_categorical'] + features['n_continuous']
        cat_ratio = features['n_categorical'] / total if total > 0 else 0
        print(f"         reid_95={features.get('reid_95', 0):.4f} "
              f"n_cat={features['n_categorical']} n_cont={features['n_continuous']} "
              f"cat_ratio={cat_ratio:.2f} "
              f"has_outliers={features.get('has_outliers')} "
              f"high_risk_rate={features.get('high_risk_rate', 0):.3f} "
              f"uniqueness={features.get('uniqueness_rate', 0):.3f} "
              f"feasibility={features.get('k_anonymity_feasibility')}")

    return actual_rule, match


def main():
    print("=" * 100)
    print("Spec 16a Fixture Verification")
    print("=" * 100)
    print()

    results = []

    # G1: LDIV1 (l-diversity sensitive column warning)
    # Trigger: sensitive_column_diversity <= 5, has_reid, NOT infeasible, min_l < 2
    # min_l is not computed by build_data_features (only by production path),
    # so we inject min_l=1 to simulate real l-diversity computation.
    r = verify("G1_LDIV1", "fixture_g1_ldiv1.csv",
               ["occupation", "education", "marital", "income", "age"],
               "LDIV1_Low_Sensitive_Diversity",
               risk_metric='reid95',
               sensitive_columns=["disease"],
               feature_overrides={'min_l': 1})
    results.append(("G1", "LDIV1_Low_Sensitive_Diversity", r))

    # G2: GEO1 pipeline (GENERALIZE + kANON for multi-level geographic QIs)
    # QIs: city(60 unique → fine), region(10 → coarse), income(22 cont).
    # Pre-Fix 0: GENERALIZE was missing from METRIC_ALLOWED_METHODS, so GEO1
    # was silently rejected and QR4_Widespread fired instead. Fixed 2026-04-20.
    r = verify("G2_GEO1", "fixture_g2_geo1.csv",
               ["city", "region", "income"],
               "GEO1_Multi_Level_Geographic",
               inject_geo=True)
    results.append(("G2", "GEO1_Multi_Level_Geographic", r))

    # G3: DYN pipeline (kANON + LOCSUPR)
    r = verify("G3_DYN", "fixture_g3_dyn.csv",
               ["var1", "var2", "category"],
               "DYN_Pipeline")
    results.append(("G3", "DYN_Pipeline", r))

    # G4: P4b pipeline
    r = verify("G4_P4b", "fixture_g4_p4.csv",
               ["income", "wealth", "sex"],
               "P4b_Skewed_Sensitive_Targeted",
               sensitive_columns=["disease"])
    results.append(("G4", "P4b_Skewed_Sensitive_Targeted", r))

    # G5: P5 pipeline
    r = verify("G5_P5", "fixture_g5_p5.csv",
               ["district", "sector", "salary", "experience"],
               "P5_Small_Dataset_Mixed")
    results.append(("G5", "P5_Small_Dataset_Mixed_Risks", r))

    # G6: CAT1 under l_diversity
    r = verify("G6_CAT1", "fixture_g6_cat1.csv",
               ["education", "marital", "employment", "housing", "age"],
               "CAT1_Categorical_Dominant",
               risk_metric='l_diversity',
               sensitive_columns=["health_status"])
    results.append(("G6", "CAT1_Categorical_Dominant", r))

    # G7: LOW2_Continuous_Noise (low reid, continuous-dominant, outliers)
    # QIs: var1/var2 (cont only). cat_ratio=0.0. has_outliers from injected extremes.
    r = verify("G7_LOW2", "fixture_g7_low2.csv",
               ["var1", "var2"],
               "LOW2_Continuous_Noise")
    results.append(("G7", "LOW2_Continuous_Noise", r))

    # G10: SR3_Near_Unique_Few_QIs (2 QIs, one high-uniqueness, moderate reid)
    # max_qi_uniqueness INJECTED — build_data_features doesn't compute it.
    # Spec 18 Item 5: fixture proves SR3 fires on realistic 2-QI data.
    r = verify("G10_SR3", "fixture_sr3_few_qis.csv",
               ["id_code", "sex"],
               "SR3_Near_Unique_Few_QIs",
               feature_overrides={
                   'max_qi_uniqueness': 0.80,
               })
    results.append(("G10", "SR3_Near_Unique_Few_QIs", r))

    # G8: Floor regime (just verify features, not a specific rule)
    df = pd.read_csv(DATA_DIR / "fixture_g8_floor.csv")
    features = build_data_features(df, ["dept", "grade", "shift"])
    eq = features.get('expected_eq_size', 999)
    feasibility = features.get('k_anonymity_feasibility', '?')
    g8_pass = eq <= 4
    status = "PASS" if g8_pass else "FAIL"
    print(f"  [{status}] {'G8_floor':20s} expected_eq={eq:.2f} feasibility={feasibility} "
          f"(need expected_eq<=4)")
    results.append(("G8", "floor_regime", ("", g8_pass)))

    # G9: RC4_Single_Bottleneck (GENERALIZE → kANON pipeline)
    # var_priority INJECTED — backward elimination cannot produce the RC4
    # bottleneck pattern (1 HIGH 15-39%, 3+ LOW) on synthetic data under 10K rows.
    # See docs/investigations/spec_16_readiness_rc_family_preemption.md
    # Tests that Fix 0 (GENERALIZE added to METRIC_ALLOWED_METHODS) unblocked
    # RC4's pipeline at the rule-selection layer.
    r = verify("G9_RC4", "fixture_g9_rc4.csv",
               ["postcode", "sex", "marital", "education"],
               "RC4_Single_Bottleneck",
               sensitive_columns=["disease"],
               feature_overrides={
                   'var_priority': {
                       'postcode': ('HIGH', 25.0),
                       'sex': ('LOW', 1.2),
                       'marital': ('LOW', 2.1),
                       'education': ('LOW', 1.8),
                   },
                   'risk_concentration': {
                       'pattern': 'balanced',
                       'top_qi': 'postcode',
                       'top_pct': 25.0,
                       'top2_pct': 27.1,
                       'n_high_risk': 1,
                   },
               })
    results.append(("G9", "RC4_Single_Bottleneck", r))

    # Summary
    print()
    print("-" * 100)
    passes = sum(1 for _, _, (_, m) in results if m)
    total = len(results)
    print(f"Results: {passes}/{total} PASS")
    if passes < total:
        print("\nFailed fixtures need tuning:")
        for gap, expected, (actual, match) in results:
            if not match:
                print(f"  {gap}: expected {expected}, got {actual}")

    return passes == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
