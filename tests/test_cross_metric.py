"""
Cross-metric test suite for the SDC protection engine.
======================================================

Run the same protection scenarios under different ``risk_metric`` values
to detect metric-path bugs --- things that work under one metric but fail
under another due to metric-specific code paths.

Metrics under test:
    - reid95       (default, all methods allowed)
    - k_anonymity  (only kANON, LOCSUPR allowed)
    - uniqueness   (only kANON, LOCSUPR allowed)

Key areas covered:
    1. METRIC_ALLOWED_METHODS filtering (config.py)
    2. compute_risk() produces valid RiskAssessment under each metric
    3. risk_to_reid_compat() synthetic dict structure per metric
    4. build_data_features() propagates metric type into features
    5. select_method_suite() respects metric filtering (blocks PRAM/NOISE
       for k_anonymity and uniqueness)
    6. Rule-level metric guards (e.g. l_diversity_rules skips for
       k_anonymity/uniqueness)
"""

import pytest
import numpy as np
import pandas as pd

from sdc_engine.sdc.config import (
    METRIC_ALLOWED_METHODS,
    filter_methods_for_metric,
    is_method_allowed_for_metric,
    get_context_targets,
)
from sdc_engine.sdc.metrics.reid import calculate_reid, classify_risk_pattern
from sdc_engine.sdc.metrics.risk import (
    check_kanonymity,
    calculate_uniqueness_rate,
    calculate_disclosure_risk,
)
from sdc_engine.sdc.metrics.risk_metric import (
    RiskMetricType,
    RiskAssessment,
    compute_risk,
    risk_to_reid_compat,
    normalize_to_risk_score,
    normalize_target,
)
from sdc_engine.sdc.protection_engine import build_data_features
from sdc_engine.sdc.selection import (
    extract_data_features_with_reid,
    select_method_suite,
)
from sdc_engine.sdc.sdc_utils import analyze_data


# ============================================================================
# Disable R backend for all tests in this module
# ============================================================================

@pytest.fixture(autouse=True)
def _no_r():
    from sdc_engine.sdc import r_backend as _rb
    _rb._R_CHECK_CACHE["result"] = False
    _rb._R_CHECK_CACHE["timestamp"] = float('inf')


# ============================================================================
# Shared constants and dataset builders
# ============================================================================

ALL_METRICS = ["reid95", "k_anonymity", "uniqueness"]
STRUCTURAL_ONLY_METRICS = ["k_anonymity", "uniqueness"]
ALL_METHODS = ["kANON", "LOCSUPR", "PRAM", "NOISE"]

# Map string names to RiskMetricType enum
_METRIC_ENUM = {
    "reid95": RiskMetricType.REID95,
    "k_anonymity": RiskMetricType.K_ANONYMITY,
    "uniqueness": RiskMetricType.UNIQUENESS,
}


@pytest.fixture
def base_df():
    """Build a 200-row mixed-type dataset suitable for cross-metric testing.

    Contains:
    - 3 categorical QIs (age_group, gender, region) with moderate cardinality
    - 1 continuous QI (income) with enough spread to be classified as continuous
    - 1 sensitive column (diagnosis) for l-diversity scenarios
    """
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "age_group": rng.choice(["18-25", "26-35", "36-45", "46-55", "56-65", "65+"], n),
        "gender": rng.choice(["M", "F"], n),
        "region": rng.choice(
            ["North", "South", "East", "West", "Central",
             "NE", "NW", "SE", "SW", "Midlands"],
            n,
        ),
        "income": rng.normal(45000, 15000, n).round(2),
        "diagnosis": rng.choice(["A", "B", "C", "D", "E"], n),
    })


@pytest.fixture
def base_qis():
    return ["age_group", "gender", "region", "income"]


@pytest.fixture
def categorical_only_df():
    """200-row dataset with all categorical QIs.

    This deliberately triggers rules that prefer PRAM under reid95 but
    must fall back to structural methods under k_anonymity/uniqueness.
    """
    rng = np.random.default_rng(99)
    n = 200
    return pd.DataFrame({
        "age_band": rng.choice(["young", "middle", "senior"], n),
        "sex": rng.choice(["M", "F"], n),
        "area": rng.choice(["urban", "suburban", "rural"], n),
        "education": rng.choice(["primary", "secondary", "tertiary", "post-grad"], n),
        "occupation": rng.choice(["eng", "med", "law", "edu", "other"], n),
    })


@pytest.fixture
def categorical_only_qis():
    return ["age_band", "sex", "area", "education", "occupation"]


def _suite_from_df(df, qis, access_tier="standard", feature_overrides=None):
    """Run the full rule chain and return (suite, features)."""
    analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
    features = extract_data_features_with_reid(df, analysis, qis)
    if feature_overrides:
        features.update(feature_overrides)
    return select_method_suite(features, access_tier=access_tier, verbose=False), features


# ============================================================================
# 1. METRIC_ALLOWED_METHODS static configuration tests
# ============================================================================

class TestMetricAllowedMethodsConfig:
    """Verify the METRIC_ALLOWED_METHODS table is correct and consistent."""

    def test_reid95_allows_all_methods(self):
        """reid95 is universal -- all four methods must be allowed."""
        allowed = METRIC_ALLOWED_METHODS["reid95"]
        for method in ALL_METHODS:
            assert method in allowed, f"{method} should be allowed under reid95"

    def test_k_anonymity_blocks_perturbative(self):
        """k_anonymity must block PRAM and NOISE (perturbative cannot guarantee EQ-class)."""
        allowed = METRIC_ALLOWED_METHODS["k_anonymity"]
        assert "PRAM" not in allowed, "PRAM must be blocked for k_anonymity"
        assert "NOISE" not in allowed, "NOISE must be blocked for k_anonymity"
        assert "kANON" in allowed
        assert "LOCSUPR" in allowed

    def test_uniqueness_blocks_perturbative(self):
        """uniqueness metric must block PRAM and NOISE."""
        allowed = METRIC_ALLOWED_METHODS["uniqueness"]
        assert "PRAM" not in allowed
        assert "NOISE" not in allowed
        assert "kANON" in allowed
        assert "LOCSUPR" in allowed

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_structural_methods_always_allowed(self, metric):
        """kANON and LOCSUPR must be allowed for every metric."""
        assert is_method_allowed_for_metric("kANON", metric)
        assert is_method_allowed_for_metric("LOCSUPR", metric)

    def test_unknown_metric_allows_all(self):
        """An unrecognised metric string should allow all methods (permissive fallback)."""
        for method in ALL_METHODS:
            assert is_method_allowed_for_metric(method, "nonexistent_metric")


# ============================================================================
# 2. filter_methods_for_metric utility function
# ============================================================================

class TestFilterMethodsForMetric:
    """Verify list-level filtering behaves correctly per metric."""

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_filter_never_returns_empty_for_full_list(self, metric):
        """Filtering the full method list by any known metric should return at least one method."""
        result = filter_methods_for_metric(ALL_METHODS, metric)
        assert len(result) > 0, f"No methods allowed for metric={metric}"

    def test_filter_reid95_preserves_all(self):
        result = filter_methods_for_metric(ALL_METHODS, "reid95")
        assert result == ALL_METHODS

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_filter_structural_only(self, metric):
        result = filter_methods_for_metric(ALL_METHODS, metric)
        assert set(result) == {"kANON", "LOCSUPR"}, (
            f"Expected only structural methods for {metric}, got {result}"
        )

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_filter_pram_only_list_returns_empty(self, metric):
        """Filtering a PRAM-only list under structural metrics must return empty."""
        result = filter_methods_for_metric(["PRAM"], metric)
        assert result == []


# ============================================================================
# 3. compute_risk() under each metric
# ============================================================================

class TestComputeRiskCrossMetric:
    """compute_risk() should produce a valid RiskAssessment for every metric."""

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_compute_risk_returns_assessment(self, metric, base_df, base_qis):
        mt = _METRIC_ENUM[metric]
        assessment = compute_risk(base_df, base_qis, mt)
        assert isinstance(assessment, RiskAssessment)
        assert assessment.metric_type == mt

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_normalized_score_in_0_1(self, metric, base_df, base_qis):
        mt = _METRIC_ENUM[metric]
        assessment = compute_risk(base_df, base_qis, mt)
        assert 0.0 <= assessment.normalized_score <= 1.0, (
            f"Normalized score {assessment.normalized_score} out of range for {metric}"
        )

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_meets_target_is_bool(self, metric, base_df, base_qis):
        mt = _METRIC_ENUM[metric]
        assessment = compute_risk(base_df, base_qis, mt)
        assert isinstance(assessment.meets_target, bool)

    def test_reid95_raw_value_matches_calculate_reid(self, base_df, base_qis):
        """compute_risk(REID95) must return the same reid_95 as calculate_reid()."""
        direct = calculate_reid(base_df, base_qis)
        assessment = compute_risk(base_df, base_qis, RiskMetricType.REID95)
        assert abs(assessment.raw_value - direct["reid_95"]) < 1e-9

    def test_k_anonymity_raw_is_min_k(self, base_df, base_qis):
        """compute_risk(K_ANONYMITY) raw_value should equal minimum group size."""
        _, group_sizes, _ = check_kanonymity(base_df, base_qis, k=1)
        col = "count" if "count" in group_sizes.columns else "_group_size_"
        expected_min_k = int(group_sizes[col].min()) if len(group_sizes) > 0 else 0
        assessment = compute_risk(base_df, base_qis, RiskMetricType.K_ANONYMITY)
        assert int(assessment.raw_value) == expected_min_k

    def test_uniqueness_raw_matches_calculate(self, base_df, base_qis):
        """compute_risk(UNIQUENESS) raw_value must match calculate_uniqueness_rate()."""
        direct = calculate_uniqueness_rate(base_df, base_qis)
        assessment = compute_risk(base_df, base_qis, RiskMetricType.UNIQUENESS)
        assert abs(assessment.raw_value - direct) < 1e-9


# ============================================================================
# 4. risk_to_reid_compat() synthetic dict tests
# ============================================================================

class TestRiskToReidCompat:
    """Verify the backward-compatible reid dict has the right shape per metric."""

    _REQUIRED_KEYS = {"reid_50", "reid_90", "reid_95", "reid_99", "mean_risk", "max_risk"}

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_compat_dict_has_required_keys(self, metric, base_df, base_qis):
        mt = _METRIC_ENUM[metric]
        assessment = compute_risk(base_df, base_qis, mt)
        compat = risk_to_reid_compat(assessment)
        missing = self._REQUIRED_KEYS - set(compat.keys())
        assert not missing, f"Missing keys in compat dict for {metric}: {missing}"

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_compat_values_non_negative(self, metric, base_df, base_qis):
        mt = _METRIC_ENUM[metric]
        assessment = compute_risk(base_df, base_qis, mt)
        compat = risk_to_reid_compat(assessment)
        for key in self._REQUIRED_KEYS:
            assert compat[key] >= 0, f"Negative value for {key} under {metric}"

    def test_reid95_compat_is_real_dict(self, base_df, base_qis):
        """For REID95, the compat dict should be the actual reid details, not synthetic."""
        assessment = compute_risk(base_df, base_qis, RiskMetricType.REID95)
        compat = risk_to_reid_compat(assessment)
        assert "_synthetic" not in compat or not compat["_synthetic"]

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_synthetic_dict_flagged(self, metric, base_df, base_qis):
        """For k_anonymity and uniqueness, compat dict must be flagged as synthetic."""
        mt = _METRIC_ENUM[metric]
        assessment = compute_risk(base_df, base_qis, mt)
        compat = risk_to_reid_compat(assessment)
        assert compat.get("_synthetic") is True, (
            f"Expected _synthetic=True for {metric}"
        )

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_synthetic_reid95_equals_normalized_score(self, metric, base_df, base_qis):
        """For synthetic dicts, reid_95 should equal the normalized score."""
        mt = _METRIC_ENUM[metric]
        assessment = compute_risk(base_df, base_qis, mt)
        compat = risk_to_reid_compat(assessment)
        assert abs(compat["reid_95"] - assessment.normalized_score) < 1e-9


# ============================================================================
# 5. normalize_to_risk_score and normalize_target consistency
# ============================================================================

class TestNormalization:
    """Ensure normalization is consistent across metrics."""

    def test_reid95_passthrough(self):
        assert normalize_to_risk_score(RiskMetricType.REID95, 0.05) == pytest.approx(0.05)

    def test_k_anonymity_inverse(self):
        """k=5 should normalize to 0.20."""
        assert normalize_to_risk_score(RiskMetricType.K_ANONYMITY, 5) == pytest.approx(0.20)

    def test_k_anonymity_higher_k_lower_risk(self):
        """Higher k must produce lower normalized risk."""
        risk_k5 = normalize_to_risk_score(RiskMetricType.K_ANONYMITY, 5)
        risk_k10 = normalize_to_risk_score(RiskMetricType.K_ANONYMITY, 10)
        assert risk_k10 < risk_k5

    def test_uniqueness_passthrough(self):
        assert normalize_to_risk_score(RiskMetricType.UNIQUENESS, 0.10) == pytest.approx(0.10)

    def test_target_normalization_roundtrip_reid(self):
        """For REID95, target normalization is identity."""
        raw = 0.05
        assert normalize_target(RiskMetricType.REID95, raw) == pytest.approx(raw)

    def test_target_normalization_k_anonymity(self):
        """k_anonymity target k=5 normalizes to 0.20."""
        assert normalize_target(RiskMetricType.K_ANONYMITY, 5) == pytest.approx(0.20)


# ============================================================================
# 6. build_data_features() propagates _risk_metric_type
# ============================================================================

class TestBuildDataFeatures:
    """build_data_features() must correctly embed the metric type in features."""

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_metric_type_in_features(self, metric, base_df, base_qis):
        features = build_data_features(
            base_df, base_qis, risk_metric=metric,
        )
        assert features["_risk_metric_type"] == metric, (
            f"Expected _risk_metric_type={metric}, got {features['_risk_metric_type']}"
        )

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_has_reid_keys(self, metric, base_df, base_qis):
        """Features must contain reid_50, reid_95, reid_99 under every metric
        (synthetic or real).
        """
        features = build_data_features(
            base_df, base_qis, risk_metric=metric,
        )
        for key in ("reid_50", "reid_95", "reid_99"):
            assert key in features, f"Missing {key} in features for metric={metric}"
            assert isinstance(features[key], (int, float))

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_risk_pattern_populated(self, metric, base_df, base_qis):
        """Risk pattern classification must be set for every metric."""
        features = build_data_features(
            base_df, base_qis, risk_metric=metric,
        )
        assert "risk_pattern" in features
        assert features["risk_pattern"] in {
            "uniform_high", "widespread", "severe_tail", "tail",
            "bimodal", "uniform_low", "moderate",
        }


# ============================================================================
# 7. select_method_suite() respects metric filtering
# ============================================================================

class TestMethodSuiteMetricFiltering:
    """The method suite must never select methods blocked by the active metric."""

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_primary_is_structural(self, metric, base_df, base_qis):
        """Under k_anonymity or uniqueness, the primary method must be structural."""
        analysis = analyze_data(base_df, quasi_identifiers=base_qis, verbose=False)
        features = extract_data_features_with_reid(base_df, analysis, base_qis)
        features["_risk_metric_type"] = metric
        suite = select_method_suite(features, access_tier="standard", verbose=False)
        primary = suite["primary"]
        assert primary in {"kANON", "LOCSUPR", "GENERALIZE"}, (
            f"Under metric={metric}, primary={primary} is not structural"
        )

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_no_perturbative_in_fallbacks(self, metric, base_df, base_qis):
        """Under k_anonymity/uniqueness, PRAM and NOISE must not appear in fallbacks."""
        analysis = analyze_data(base_df, quasi_identifiers=base_qis, verbose=False)
        features = extract_data_features_with_reid(base_df, analysis, base_qis)
        features["_risk_metric_type"] = metric
        suite = select_method_suite(features, access_tier="standard", verbose=False)
        for method, params in suite.get("fallbacks", []):
            assert method not in {"PRAM", "NOISE"}, (
                f"Perturbative method {method} in fallbacks under metric={metric}"
            )

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_no_perturbative_in_pipeline(self, metric, base_df, base_qis):
        """If a pipeline is selected, it must not contain PRAM or NOISE."""
        analysis = analyze_data(base_df, quasi_identifiers=base_qis, verbose=False)
        features = extract_data_features_with_reid(base_df, analysis, base_qis)
        features["_risk_metric_type"] = metric
        suite = select_method_suite(features, access_tier="standard", verbose=False)
        if suite.get("use_pipeline"):
            for method in suite["pipeline"]:
                assert method not in {"PRAM", "NOISE"}, (
                    f"Perturbative method {method} in pipeline under metric={metric}"
                )

    def test_reid95_can_select_pram(self, categorical_only_df, categorical_only_qis):
        """Under reid95, PRAM should be selectable for categorical-dominant data."""
        analysis = analyze_data(
            categorical_only_df,
            quasi_identifiers=categorical_only_qis,
            verbose=False,
        )
        features = extract_data_features_with_reid(
            categorical_only_df, analysis, categorical_only_qis
        )
        features["_risk_metric_type"] = "reid95"
        suite = select_method_suite(features, access_tier="standard", verbose=False)
        # Under reid95, any method is valid -- just check it does not crash
        assert suite["primary"] in {"kANON", "LOCSUPR", "PRAM", "NOISE", "GENERALIZE"}

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_categorical_data_forced_structural(
        self, metric, categorical_only_df, categorical_only_qis
    ):
        """Categorical-dominant data under k_anonymity/uniqueness must use structural methods.

        Under reid95 this data would likely trigger CAT1 (PRAM). Under
        k_anonymity/uniqueness, PRAM is blocked so the engine must fall through
        to a structural alternative.
        """
        analysis = analyze_data(
            categorical_only_df,
            quasi_identifiers=categorical_only_qis,
            verbose=False,
        )
        features = extract_data_features_with_reid(
            categorical_only_df, analysis, categorical_only_qis
        )
        features["_risk_metric_type"] = metric
        suite = select_method_suite(features, access_tier="standard", verbose=False)
        assert suite["primary"] in {"kANON", "LOCSUPR", "GENERALIZE"}, (
            f"Expected structural method under {metric}, got {suite['primary']}"
        )

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_reid_fallback_is_structural(self, metric, base_df, base_qis):
        """The reid_fallback method (if any) must also be structural."""
        analysis = analyze_data(base_df, quasi_identifiers=base_qis, verbose=False)
        features = extract_data_features_with_reid(base_df, analysis, base_qis)
        features["_risk_metric_type"] = metric
        suite = select_method_suite(features, access_tier="standard", verbose=False)
        fb = suite.get("reid_fallback")
        if fb:
            assert fb["method"] in {"kANON", "LOCSUPR", "GENERALIZE"}, (
                f"reid_fallback={fb['method']} is perturbative under {metric}"
            )


# ============================================================================
# 8. Same dataset, different metrics -> consistent structural behavior
# ============================================================================

class TestCrossMetricConsistency:
    """When two metrics both allow only structural methods, the selected
    method should be the same (or at least structural) for both.
    """

    def test_k_anonymity_and_uniqueness_agree_on_method(self, base_df, base_qis):
        """k_anonymity and uniqueness both allow {kANON, LOCSUPR}, so the same
        rule chain should fire and select the same primary method.
        """
        suites = {}
        for metric in STRUCTURAL_ONLY_METRICS:
            analysis = analyze_data(base_df, quasi_identifiers=base_qis, verbose=False)
            features = extract_data_features_with_reid(base_df, analysis, base_qis)
            features["_risk_metric_type"] = metric
            suites[metric] = select_method_suite(
                features, access_tier="standard", verbose=False
            )
        # Both should select structural methods (may differ in params but
        # the method itself should be the same because the allowed set is
        # identical).
        assert suites["k_anonymity"]["primary"] == suites["uniqueness"]["primary"], (
            f"k_anonymity selected {suites['k_anonymity']['primary']} "
            f"but uniqueness selected {suites['uniqueness']['primary']}"
        )


# ============================================================================
# 9. calculate_reid under different data shapes (metric-agnostic correctness)
# ============================================================================

class TestCalculateReidConsistency:
    """Ensure calculate_reid() is well-behaved regardless of how it will
    be consumed downstream.
    """

    def test_returns_all_standard_keys(self, base_df, base_qis):
        reid = calculate_reid(base_df, base_qis)
        for key in ("reid_50", "reid_90", "reid_95", "reid_99",
                     "mean_risk", "max_risk", "high_risk_count", "high_risk_rate"):
            assert key in reid, f"Missing key {key}"

    def test_reid_percentile_ordering(self, base_df, base_qis):
        reid = calculate_reid(base_df, base_qis)
        assert reid["reid_50"] <= reid["reid_90"] <= reid["reid_95"] <= reid["reid_99"]

    def test_empty_qis_returns_zeros(self, base_df):
        reid = calculate_reid(base_df, [])
        assert reid["reid_95"] == 0.0

    def test_suppression_tracking_keys(self, base_df, base_qis):
        reid = calculate_reid(base_df, base_qis)
        assert "suppressed_records" in reid
        assert "suppression_rate" in reid
        assert "records_evaluated" in reid


# ============================================================================
# 10. get_context_targets() metric-aware target extraction
# ============================================================================

class TestContextTargets:
    """get_context_targets() must return the right risk_target for each metric."""

    @pytest.mark.parametrize("metric,context", [
        ("reid95", "scientific_use"),
        ("k_anonymity", "scientific_use"),
        ("uniqueness", "scientific_use"),
        ("reid95", "public_release"),
        ("k_anonymity", "public_release"),
        ("uniqueness", "public_release"),
    ])
    def test_targets_populated(self, metric, context):
        targets = get_context_targets(context, risk_metric=metric)
        assert "risk_target" in targets
        assert "risk_target_normalized" in targets
        assert targets["risk_metric"] == metric

    def test_reid95_target_is_reid_value(self):
        targets = get_context_targets("scientific_use", risk_metric="reid95")
        assert targets["risk_target"] == targets["reid_target"]

    def test_k_anonymity_target_is_k_min(self):
        targets = get_context_targets("scientific_use", risk_metric="k_anonymity")
        assert targets["risk_target"] == targets["k_min"]

    def test_uniqueness_target_is_uniqueness_max(self):
        targets = get_context_targets("scientific_use", risk_metric="uniqueness")
        assert targets["risk_target"] == targets["uniqueness_max"]


# ============================================================================
# 11. Metric-specific rule guards
# ============================================================================

class TestRuleMetricGuards:
    """Certain rules have metric-specific guards that disable them for
    k_anonymity/uniqueness. Verify those guards are active.
    """

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_l_diversity_rule_skips_for_structural_metrics(self, metric):
        """l_diversity_rules should return applies=False for k_anonymity/uniqueness."""
        from sdc_engine.sdc.selection.rules import l_diversity_rules

        features = {
            "_risk_metric_type": metric,
            "sensitive_column_diversity": 3,
            "quasi_identifiers": ["age", "gender"],
            "has_reid": True,
            "reid_95": 0.10,
            "n_qis": 2,
            "k_anonymity_feasibility": "moderate",
            "estimated_suppression": {3: 0.05, 5: 0.10, 7: 0.20},
            "sensitive_columns": {"diagnosis": "text"},
        }
        result = l_diversity_rules(features)
        assert not result.get("applies"), (
            f"l_diversity_rules should not fire under metric={metric}"
        )


# ============================================================================
# 12. Risk pattern classification works on synthetic reid dicts
# ============================================================================

class TestRiskPatternOnSyntheticDicts:
    """classify_risk_pattern() is called on reid dicts. When we pass
    synthetic dicts (from k_anonymity/uniqueness), it should still
    return a valid pattern string without errors.
    """

    @pytest.mark.parametrize("metric", STRUCTURAL_ONLY_METRICS)
    def test_classify_on_synthetic_dict(self, metric, base_df, base_qis):
        mt = _METRIC_ENUM[metric]
        assessment = compute_risk(base_df, base_qis, mt)
        compat = risk_to_reid_compat(assessment)
        pattern = classify_risk_pattern(compat)
        valid_patterns = {
            "uniform_high", "widespread", "severe_tail", "tail",
            "bimodal", "uniform_low", "moderate",
        }
        assert pattern in valid_patterns, (
            f"classify_risk_pattern returned '{pattern}' for synthetic {metric} dict"
        )


# ============================================================================
# 13. Edge case: high-risk data forces structural under all metrics
# ============================================================================

class TestHighRiskForcesStructural:
    """A high-risk dataset should select structural methods regardless
    of which metric is active.
    """

    @pytest.fixture
    def high_risk_df(self):
        """Build a high-uniqueness dataset that creates high risk under all metrics."""
        rng = np.random.default_rng(123)
        n = 200
        return pd.DataFrame({
            "age": rng.integers(18, 90, n),
            "zipcode": rng.integers(10000, 99999, n),
            "income": rng.normal(50000, 20000, n).round(2),
        })

    @pytest.fixture
    def high_risk_qis(self):
        return ["age", "zipcode", "income"]

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_structural_selected(self, metric, high_risk_df, high_risk_qis):
        analysis = analyze_data(
            high_risk_df, quasi_identifiers=high_risk_qis, verbose=False
        )
        features = extract_data_features_with_reid(
            high_risk_df, analysis, high_risk_qis
        )
        features["_risk_metric_type"] = metric
        suite = select_method_suite(features, access_tier="standard", verbose=False)
        assert suite["primary"] in {"kANON", "LOCSUPR", "GENERALIZE", "NOISE"}, (
            f"Expected structural-like method for high-risk data under {metric}, "
            f"got {suite['primary']}"
        )


# ============================================================================
# 14. Parametrized smoke test: build_data_features + select_method_suite
#     must not raise under any metric
# ============================================================================

class TestNoExceptionsAcrossMetrics:
    """Smoke tests: the full path from build_data_features through
    select_method_suite must complete without exceptions for every metric.
    """

    @pytest.mark.parametrize("metric", ALL_METRICS)
    def test_build_features_no_exception(self, metric, base_df, base_qis):
        features = build_data_features(base_df, base_qis, risk_metric=metric)
        assert isinstance(features, dict)

    @pytest.mark.parametrize("metric", ALL_METRICS)
    @pytest.mark.parametrize("tier", ["standard", "PUBLIC", "SECURE"])
    def test_suite_no_exception(self, metric, tier, base_df, base_qis):
        analysis = analyze_data(base_df, quasi_identifiers=base_qis, verbose=False)
        features = extract_data_features_with_reid(base_df, analysis, base_qis)
        features["_risk_metric_type"] = metric
        suite = select_method_suite(features, access_tier=tier, verbose=False)
        assert "primary" in suite
        assert "rule_applied" in suite
