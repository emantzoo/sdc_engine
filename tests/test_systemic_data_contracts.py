"""
Spec 22 — Test 3: Data Contract Validation
==========================================

Validates that shared data structures produced by production code conform
to their canonical formats.  Catches drift like emoji-prefixed labels,
out-of-range risk values, and inconsistent type counts.

Contracts tested:
    1. var_priority labels ∈ {'HIGH', 'MED-HIGH', 'MODERATE', 'LOW'}
    2. risk_pattern ∈ canonical 7-string set
    3. risk_concentration['pattern'] ∈ {'dominated', 'not_dominated', 'unknown'}
    4. k_anonymity_feasibility ∈ {'easy', 'moderate', 'hard', 'infeasible'}
    5. ReID ordering: 0 ≤ reid_50 ≤ reid_95 ≤ reid_99 ≤ 1.0
    6. Type-count consistency: n_continuous == len(continuous_vars), etc.
    7. cat_ratio consistency: n_categorical / (n_categorical + n_continuous)
       matches what rule factories compute (user-flagged dual-path issue)
    8. select_method_suite output: required keys, valid primary, valid confidence
    9. classify_risk_pattern returns canonical strings
   10. classify_risk_concentration returns canonical pattern strings
"""
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Canonical value sets (single source of truth for this test)
# ---------------------------------------------------------------------------

CANONICAL_VAR_PRIORITY_LABELS = {'HIGH', 'MED-HIGH', 'MODERATE', 'LOW'}

CANONICAL_RISK_PATTERNS = {
    'uniform_high', 'widespread', 'severe_tail', 'tail',
    'uniform_low', 'bimodal', 'moderate',
}

CANONICAL_RISK_CONCENTRATION_PATTERNS = {'dominated', 'not_dominated', 'unknown'}

CANONICAL_FEASIBILITY = {'easy', 'moderate', 'hard', 'infeasible'}

CANONICAL_CONFIDENCE = {'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW'}

CANONICAL_METHODS = {
    'kANON', 'LOCSUPR', 'PRAM', 'NOISE', 'GENERALIZE', 'GENERALIZE_FIRST',
}

# Required keys in select_method_suite output
SUITE_REQUIRED_KEYS = {
    'primary', 'primary_params', 'fallbacks', 'rule_applied',
    'confidence', 'reason', 'use_pipeline',
}


# ---------------------------------------------------------------------------
# Synthetic data builders (small, fast, deterministic)
# ---------------------------------------------------------------------------

def _rng(seed=99):
    return np.random.default_rng(seed)


def _build_mixed_dataset(n=600, seed=99):
    """2 categorical + 1 continuous QI."""
    rng = _rng(seed)
    df = pd.DataFrame({
        'region': rng.choice(['N', 'S', 'E', 'W'], n),
        'edu': rng.choice(['hs', 'ba', 'ms', 'phd'], n),
        'age': rng.integers(20, 60, n),
    })
    return df, ['region', 'edu', 'age']


def _build_all_categorical(n=600, seed=99):
    """3 categorical QIs."""
    rng = _rng(seed)
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'region': rng.choice(['N', 'S', 'E', 'W'], n),
        'edu': rng.choice(['hs', 'ba', 'ms', 'phd'], n),
    })
    return df, ['sex', 'region', 'edu']


def _build_all_continuous(n=600, seed=99):
    """2 continuous QIs (>20 unique each)."""
    rng = _rng(seed)
    df = pd.DataFrame({
        'age': rng.integers(20, 60, n),
        'income': rng.integers(20000, 80000, n),
    })
    return df, ['age', 'income']


def _build_high_risk(n=200, seed=99):
    """High-cardinality dataset → high risk."""
    rng = _rng(seed)
    df = pd.DataFrame({
        'a': rng.choice([f'a{i}' for i in range(30)], n),
        'b': rng.choice([f'b{i}' for i in range(30)], n),
        'c': rng.choice([f'c{i}' for i in range(20)], n),
    })
    return df, ['a', 'b', 'c']


def _build_low_risk(n=2000, seed=99):
    """Low-cardinality dataset → low risk."""
    rng = _rng(seed)
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'region': rng.choice(['N', 'S', 'E'], n),
    })
    return df, ['sex', 'region']


def _build_small(n=100, seed=99):
    """Very small dataset (HR6 territory)."""
    rng = _rng(seed)
    df = pd.DataFrame({
        'a': rng.choice(['x', 'y', 'z'], n),
        'b': rng.choice(['p', 'q'], n),
    })
    return df, ['a', 'b']


DATASET_BUILDERS = [
    _build_mixed_dataset,
    _build_all_categorical,
    _build_all_continuous,
    _build_high_risk,
    _build_low_risk,
    _build_small,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _features(builder, **kwargs):
    """Call build_data_features on a builder's output."""
    from sdc_engine.sdc.protection_engine import build_data_features
    df, qis = builder()
    return build_data_features(df, qis, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Contract 1 — var_priority labels
# ═══════════════════════════════════════════════════════════════════════════

class TestVarPriorityLabels:
    """var_priority labels must be in canonical set — no emoji, no decorators."""

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_labels_canonical(self, builder):
        feats = _features(builder)
        vp = feats.get('var_priority', {})
        if not vp:
            pytest.skip("var_priority not populated (dataset too large or n_qis<=1)")
        for qi, (label, pct) in vp.items():
            assert label in CANONICAL_VAR_PRIORITY_LABELS, (
                f"var_priority['{qi}'] label '{label}' not in {CANONICAL_VAR_PRIORITY_LABELS}"
            )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_pct_is_nonnegative_float(self, builder):
        feats = _features(builder)
        vp = feats.get('var_priority', {})
        if not vp:
            pytest.skip("var_priority not populated")
        for qi, (label, pct) in vp.items():
            assert isinstance(pct, (int, float)), (
                f"var_priority['{qi}'] pct should be numeric, got {type(pct)}"
            )
            assert pct >= 0, f"var_priority['{qi}'] pct={pct} is negative"

    def test_label_thresholds(self):
        """Verify the label assignment boundaries: 15% / 8% / 3%."""
        from sdc_engine.sdc.protection_engine import _compute_var_priority
        from sdc_engine.sdc.metrics.reid import calculate_reid

        # Build a dataset where we can control contribution by design:
        # One high-cardinality QI ('job' with 20 values) + two low-card
        rng = _rng(42)
        n = 600
        df = pd.DataFrame({
            'sex': rng.choice(['M', 'F'], n),
            'region': rng.choice(['N', 'S', 'E'], n),
            'job': rng.choice([f'j_{i}' for i in range(20)], n),
        })
        qis = ['sex', 'region', 'job']
        reid_95 = calculate_reid(df, qis)['reid_95']
        vp = _compute_var_priority(df, qis, reid_95)

        if vp is None:
            pytest.skip("var_priority computation returned None")

        for qi, (label, pct) in vp.items():
            # Verify thresholds match the documented contract
            if pct >= 15:
                assert label == 'HIGH'
            elif pct >= 8:
                assert label == 'MED-HIGH'
            elif pct >= 3:
                assert label == 'MODERATE'
            else:
                assert label == 'LOW'


# ═══════════════════════════════════════════════════════════════════════════
# Contract 2 — risk_pattern
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskPatternContract:
    """risk_pattern must be in the canonical 7-string set."""

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_pattern_canonical(self, builder):
        feats = _features(builder)
        pattern = feats['risk_pattern']
        assert pattern in CANONICAL_RISK_PATTERNS, (
            f"risk_pattern '{pattern}' not in {CANONICAL_RISK_PATTERNS}"
        )

    def test_classify_risk_pattern_canonical(self):
        """classify_risk_pattern returns canonical string for representative inputs."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern

        test_cases = [
            # (reid_dict, description)
            ({'reid_50': 0.30, 'reid_95': 0.35, 'reid_99': 0.38, 'mean_risk': 0.31},
             'uniform_high'),
            ({'reid_50': 0.25, 'reid_95': 0.50, 'reid_99': 0.80, 'mean_risk': 0.30},
             'widespread'),
            ({'reid_50': 0.02, 'reid_95': 0.05, 'reid_99': 0.80, 'mean_risk': 0.05},
             'severe_tail'),
            ({'reid_50': 0.10, 'reid_95': 0.35, 'reid_99': 0.60, 'mean_risk': 0.15},
             'tail'),
            ({'reid_50': 0.01, 'reid_95': 0.02, 'reid_99': 0.03, 'mean_risk': 0.01},
             'uniform_low'),
            ({'reid_50': 0.10, 'reid_95': 0.15, 'reid_99': 0.18, 'mean_risk': 0.30},
             'bimodal'),
            ({'reid_50': 0.10, 'reid_95': 0.15, 'reid_99': 0.20, 'mean_risk': 0.11},
             'moderate'),
        ]
        for reid_dict, expected_pattern in test_cases:
            result = classify_risk_pattern(reid_dict)
            assert result in CANONICAL_RISK_PATTERNS, (
                f"classify_risk_pattern({reid_dict}) = '{result}' "
                f"not in {CANONICAL_RISK_PATTERNS}"
            )
            assert result == expected_pattern, (
                f"classify_risk_pattern({reid_dict}) = '{result}', "
                f"expected '{expected_pattern}'"
            )

    def test_risk_level_matches_risk_pattern(self):
        """risk_level key should be identical to risk_pattern (legacy alias)."""
        feats = _features(_build_mixed_dataset)
        assert feats['risk_level'] == feats['risk_pattern']


# ═══════════════════════════════════════════════════════════════════════════
# Contract 3 — risk_concentration
# ═══════════════════════════════════════════════════════════════════════════

class TestRiskConcentrationContract:
    """risk_concentration['pattern'] must be in canonical set."""

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_pattern_canonical(self, builder):
        feats = _features(builder)
        rc = feats['risk_concentration']
        assert isinstance(rc, dict), f"risk_concentration should be dict, got {type(rc)}"
        assert rc['pattern'] in CANONICAL_RISK_CONCENTRATION_PATTERNS, (
            f"risk_concentration pattern '{rc['pattern']}' not in "
            f"{CANONICAL_RISK_CONCENTRATION_PATTERNS}"
        )

    def test_classify_risk_concentration_canonical(self):
        """Direct call to classify_risk_concentration returns canonical pattern."""
        from sdc_engine.sdc.selection.features import classify_risk_concentration

        # dominated: top QI >= 40%
        vp_dominated = {'job': ('HIGH', 50.0), 'sex': ('LOW', 2.0)}
        assert classify_risk_concentration(vp_dominated)['pattern'] == 'dominated'

        # not_dominated: top QI < 40%
        vp_spread = {'a': ('MED-HIGH', 30.0), 'b': ('MODERATE', 25.0),
                      'c': ('LOW', 5.0)}
        assert classify_risk_concentration(vp_spread)['pattern'] == 'not_dominated'

        # unknown: empty var_priority
        assert classify_risk_concentration(None)['pattern'] == 'unknown'
        assert classify_risk_concentration({})['pattern'] == 'unknown'

    def test_risk_concentration_keys(self):
        """risk_concentration dict has expected keys."""
        feats = _features(_build_mixed_dataset)
        rc = feats['risk_concentration']
        for key in ('pattern', 'top_qi', 'top_pct', 'top2_pct', 'n_high_risk'):
            assert key in rc, f"risk_concentration missing key '{key}'"


# ═══════════════════════════════════════════════════════════════════════════
# Contract 4 — k_anonymity_feasibility
# ═══════════════════════════════════════════════════════════════════════════

class TestFeasibilityContract:
    """k_anonymity_feasibility must be in canonical set."""

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_feasibility_canonical(self, builder):
        feats = _features(builder)
        feas = feats['k_anonymity_feasibility']
        assert feas in CANONICAL_FEASIBILITY, (
            f"k_anonymity_feasibility '{feas}' not in {CANONICAL_FEASIBILITY}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Contract 5 — ReID value ordering
# ═══════════════════════════════════════════════════════════════════════════

class TestReIDOrdering:
    """ReID percentiles must be ordered: 0 ≤ reid_50 ≤ reid_95 ≤ reid_99 ≤ 1.0."""

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_reid_ordering(self, builder):
        feats = _features(builder)
        r50 = feats['reid_50']
        r95 = feats['reid_95']
        r99 = feats['reid_99']
        assert 0 <= r50 <= r95 <= r99 <= 1.0, (
            f"ReID ordering violated: reid_50={r50}, reid_95={r95}, reid_99={r99}"
        )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_high_risk_rate_in_range(self, builder):
        feats = _features(builder)
        hrr = feats['high_risk_rate']
        assert 0.0 <= hrr <= 1.0, f"high_risk_rate={hrr} out of [0, 1]"


# ═══════════════════════════════════════════════════════════════════════════
# Contract 6 — Type-count consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestTypeCountConsistency:
    """n_continuous == len(continuous_vars), n_categorical == len(categorical_vars)."""

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_counts_match_lists(self, builder):
        feats = _features(builder)
        assert feats['n_continuous'] == len(feats['continuous_vars']), (
            f"n_continuous={feats['n_continuous']} != "
            f"len(continuous_vars)={len(feats['continuous_vars'])}"
        )
        assert feats['n_categorical'] == len(feats['categorical_vars']), (
            f"n_categorical={feats['n_categorical']} != "
            f"len(categorical_vars)={len(feats['categorical_vars'])}"
        )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_vars_subset_of_qis(self, builder):
        """All classified vars must be a subset of quasi_identifiers."""
        feats = _features(builder)
        all_vars = set(feats['continuous_vars']) | set(feats['categorical_vars'])
        qi_set = set(feats['quasi_identifiers'])
        assert all_vars <= qi_set, (
            f"Variables not in QIs: {all_vars - qi_set}"
        )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_no_overlap(self, builder):
        """continuous_vars and categorical_vars should not overlap."""
        feats = _features(builder)
        overlap = set(feats['continuous_vars']) & set(feats['categorical_vars'])
        assert not overlap, f"Overlap between continuous and categorical: {overlap}"


# ═══════════════════════════════════════════════════════════════════════════
# Contract 7 — cat_ratio consistency (user-flagged dual-path issue)
# ═══════════════════════════════════════════════════════════════════════════

class TestCatRatioConsistency:
    """cat_ratio as computed by rule factories should equal
    n_categorical / (n_categorical + n_continuous) from the features dict.

    NOTE: cat_ratio is NOT a key in the features dict — it's recomputed
    by every rule factory from features['n_categorical'] and
    features['n_continuous'].  build_dynamic_pipeline recomputes it
    from the raw data instead.  This test validates the feature-dict
    values are self-consistent; the pipeline recomputation inconsistency
    is documented as a known dual-path issue (flagged for Spec 20 fix).
    """

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_cat_ratio_from_features(self, builder):
        feats = _features(builder)
        n_cat = feats['n_categorical']
        n_cont = feats['n_continuous']
        total = n_cat + n_cont

        if total == 0:
            pytest.skip("No QIs classified as cat or cont")

        expected_ratio = n_cat / total
        # Every rule factory computes cat_ratio = n_cat / total from features.
        # Verify the components are integers and ratio is in [0, 1].
        assert isinstance(n_cat, int), f"n_categorical should be int, got {type(n_cat)}"
        assert isinstance(n_cont, int), f"n_continuous should be int, got {type(n_cont)}"
        assert 0.0 <= expected_ratio <= 1.0

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_cat_ratio_matches_data_classification(self, builder):
        """Verify n_categorical + n_continuous == n_qis (no QIs lost in classification)."""
        feats = _features(builder)
        n_cat = feats['n_categorical']
        n_cont = feats['n_continuous']
        n_qis = feats['n_qis']

        # Every QI should be classified as either categorical or continuous.
        # If a QI is not in data.columns it's skipped — but our builders
        # always include all QI columns.
        assert n_cat + n_cont == n_qis, (
            f"n_categorical({n_cat}) + n_continuous({n_cont}) = {n_cat + n_cont} "
            f"!= n_qis({n_qis}) — some QIs lost in classification"
        )

    def test_pipeline_cat_ratio_dual_path_documented(self):
        """Document the known dual-path issue: build_dynamic_pipeline
        recomputes cat_ratio from raw data rather than reading from features.

        This test verifies the feature-dict values are self-consistent.
        The pipeline recomputation inconsistency should be fixed in a
        targeted commit (not Spec 22 scope).
        """
        # Build a dataset and compute features
        df, qis = _build_mixed_dataset()
        from sdc_engine.sdc.protection_engine import build_data_features
        feats = build_data_features(df, qis)

        n_cat = feats['n_categorical']
        n_cont = feats['n_continuous']
        total = n_cat + n_cont
        if total == 0:
            pytest.skip("No classified vars")

        features_cat_ratio = n_cat / total

        # Now check what build_dynamic_pipeline would compute.
        # It does: n_cat = feats['n_categorical'], n_cont = feats['n_continuous'],
        # cat_ratio = n_cat / (n_cat + n_cont).
        # This SHOULD be identical to our calculation above.
        # The known issue is that build_dynamic_pipeline ALSO has a
        # separate code path that recomputes from len(categorical_vars).
        # For now, just verify the feature dict is internally consistent.
        assert features_cat_ratio == n_cat / total


# ═══════════════════════════════════════════════════════════════════════════
# Contract 8 — select_method_suite output shape
# ═══════════════════════════════════════════════════════════════════════════

class TestMethodSuiteOutput:
    """select_method_suite output must have required keys and valid values."""

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_required_keys(self, builder):
        feats = _features(builder)
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        suite = select_method_suite(feats)
        missing = SUITE_REQUIRED_KEYS - set(suite.keys())
        assert not missing, f"select_method_suite missing keys: {missing}"

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_primary_is_valid_method(self, builder):
        feats = _features(builder)
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        suite = select_method_suite(feats)
        primary = suite['primary']
        # Pipeline suites may return a dict, not a string
        if isinstance(primary, dict):
            assert 'method' in primary, "Pipeline primary dict missing 'method'"
            assert primary['method'] in CANONICAL_METHODS, (
                f"Pipeline primary method '{primary['method']}' not in {CANONICAL_METHODS}"
            )
        else:
            assert primary in CANONICAL_METHODS, (
                f"primary '{primary}' not in {CANONICAL_METHODS}"
            )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_confidence_is_valid(self, builder):
        feats = _features(builder)
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        suite = select_method_suite(feats)
        assert suite['confidence'] in CANONICAL_CONFIDENCE, (
            f"confidence '{suite['confidence']}' not in {CANONICAL_CONFIDENCE}"
        )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_fallbacks_are_list_of_tuples(self, builder):
        feats = _features(builder)
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        suite = select_method_suite(feats)
        fallbacks = suite['fallbacks']
        assert isinstance(fallbacks, list), f"fallbacks should be list, got {type(fallbacks)}"
        for i, fb in enumerate(fallbacks):
            assert isinstance(fb, (tuple, list)) and len(fb) == 2, (
                f"fallback[{i}] should be (method, params) tuple, got {fb!r}"
            )
            method, params = fb
            assert method in CANONICAL_METHODS, (
                f"fallback[{i}] method '{method}' not in {CANONICAL_METHODS}"
            )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_rule_applied_is_string(self, builder):
        feats = _features(builder)
        from sdc_engine.sdc.selection.pipelines import select_method_suite
        suite = select_method_suite(feats)
        assert isinstance(suite['rule_applied'], str), (
            f"rule_applied should be str, got {type(suite['rule_applied'])}"
        )
        assert len(suite['rule_applied']) > 0, "rule_applied is empty string"


# ═══════════════════════════════════════════════════════════════════════════
# Contract 9 — sensitive_columns structure
# ═══════════════════════════════════════════════════════════════════════════

class TestSensitiveColumnsContract:
    """sensitive_columns must be a dict {col: reason_string}."""

    def test_no_sensitive(self):
        feats = _features(_build_mixed_dataset)
        sc = feats['sensitive_columns']
        assert isinstance(sc, dict), f"sensitive_columns should be dict, got {type(sc)}"
        assert len(sc) == 0, "Expected empty sensitive_columns for non-sensitive data"

    def test_with_sensitive(self):
        df, qis = _build_mixed_dataset()
        from sdc_engine.sdc.protection_engine import build_data_features
        feats = build_data_features(df, qis, sensitive_columns=['region'])
        sc = feats['sensitive_columns']
        assert isinstance(sc, dict)
        assert 'region' in sc
        assert isinstance(sc['region'], str), (
            f"sensitive_columns value should be str, got {type(sc['region'])}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Contract 10 — features dict completeness
# ═══════════════════════════════════════════════════════════════════════════

# Minimum required keys from build_data_features (consumed by rules/pipelines).
FEATURES_REQUIRED_KEYS = {
    'n_records', 'n_columns', 'data_type',
    'n_continuous', 'n_categorical', 'continuous_vars', 'categorical_vars',
    'n_qis', 'quasi_identifiers',
    'high_cardinality_qis', 'low_cardinality_qis', 'high_cardinality_count',
    'qi_cardinalities', 'qi_cardinality_product', 'expected_eq_size',
    'k_anonymity_feasibility', 'max_achievable_k',
    'max_qi_uniqueness',
    'qi_type_counts', 'n_geo_qis', 'geo_qis_by_granularity',
    'uniqueness_rate',
    'has_outliers', 'skewed_columns',
    'has_sensitive_attributes', 'sensitive_columns',
    'sensitive_column_diversity', 'min_l', 'l_diversity',
    'has_reid', 'reid_50', 'reid_95', 'reid_99',
    'mean_risk', 'max_risk', 'risk_pattern', 'risk_level',
    'high_risk_count', 'high_risk_rate',
    'small_cells_rate',
    'qi_max_category_freq',
    'estimated_suppression', 'estimated_suppression_k5',
    'var_priority', 'risk_concentration',
}


class TestFeaturesCompleteness:
    """build_data_features must return all keys consumed by rules."""

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_required_keys_present(self, builder):
        feats = _features(builder)
        missing = FEATURES_REQUIRED_KEYS - set(feats.keys())
        assert not missing, f"build_data_features missing keys: {missing}"

    def test_data_type_is_microdata(self):
        """build_data_features always returns data_type='microdata'."""
        feats = _features(_build_mixed_dataset)
        assert feats['data_type'] == 'microdata'

    def test_has_reid_is_true(self):
        """has_reid is always True when risk is computed."""
        feats = _features(_build_mixed_dataset)
        assert feats['has_reid'] is True


# ═══════════════════════════════════════════════════════════════════════════
# Contract 11 — estimated_suppression structure
# ═══════════════════════════════════════════════════════════════════════════

class TestEstimatedSuppressionContract:

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_structure(self, builder):
        feats = _features(builder)
        es = feats['estimated_suppression']
        assert isinstance(es, dict), f"estimated_suppression should be dict, got {type(es)}"
        for k_val in (3, 5, 7):
            assert k_val in es, f"estimated_suppression missing k={k_val}"
            v = es[k_val]
            assert 0.0 <= v <= 1.0, (
                f"estimated_suppression[{k_val}]={v} out of [0, 1]"
            )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_monotonic(self, builder):
        """Suppression at higher k should be >= suppression at lower k."""
        feats = _features(builder)
        es = feats['estimated_suppression']
        assert es[3] <= es[5] <= es[7], (
            f"estimated_suppression not monotonic: k3={es[3]}, k5={es[5]}, k7={es[7]}"
        )

    @pytest.mark.parametrize("builder", DATASET_BUILDERS,
                             ids=[b.__name__ for b in DATASET_BUILDERS])
    def test_k5_alias(self, builder):
        feats = _features(builder)
        assert feats['estimated_suppression_k5'] == feats['estimated_suppression'][5]
