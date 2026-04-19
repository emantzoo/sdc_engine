"""
Known-case regression tests for the rule selection engine.

Each test constructs a minimal synthetic dataset designed to trigger exactly
one rule (or deliberately blocked by a specific guard). These tests guard
against regressions in rule priority ordering, gate conditions, and feature
detection.

They do NOT assert on parameter values or method outcomes — those are
calibration questions handled by tests/empirical/ and method unit tests.
"""
import pytest
import numpy as np

from sdc_engine.sdc.selection import (
    extract_data_features_with_reid,
    select_method_suite,
)
from sdc_engine.sdc.sdc_utils import analyze_data
from sdc_engine.sdc.selection.rules import (
    _has_dominant_categories,
    uniqueness_risk_rules,
    risk_concentration_rules,
    public_release_rules,
    secure_environment_rules,
    regulatory_compliance_rules,
)
from tests.fixtures.rule_test_builders import (
    build_severe_tail_dataset,
    build_qr2_moderate_tail_dataset,
    build_qr2_heavy_tail_dataset,
    build_qr4_widespread_dataset,
    build_med1_dataset,
    build_dominated_risk_dataset,
    build_cat1_dataset,
    build_cat1_blocked_by_dominance,
    build_cat2_dataset,
    build_low1_dataset,
    build_low2_dataset,
    build_low3_dataset,
    build_sr3_dataset,
    build_hr6_dataset,
    build_extreme_uniqueness_dataset,
    build_default_fallback_dataset,
    build_pub1_high_risk_dataset,
    build_pub1_moderate_risk_dataset,
    build_sec1_categorical_dataset,
    build_sec1_continuous_dataset,
    build_reg1_high_risk_dataset,
    build_reg1_moderate_risk_dataset,
)


# ════════════════════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════════════════════

def get_suite(df, qis, access_tier='standard', feature_overrides=None):
    """Run the full rule chain and return the suite dict."""
    analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
    features = extract_data_features_with_reid(df, analysis, qis)
    if feature_overrides:
        features.update(feature_overrides)
    return select_method_suite(features, access_tier=access_tier, verbose=False), features


# ════════════════════════════════════════════════════════════════════════
# High-risk structural rules (QR1, QR2, QR4, MED1)
# ════════════════════════════════════════════════════════════════════════

class TestHighRiskRules:
    """QR-series and MED1 rules for moderate-to-high reid scenarios."""

    def test_qr1_fires_on_severe_tail(self):
        """Severe tail (reid_50 low, reid_99 extreme) → QR1 LOCSUPR."""
        df, qis, _ = build_severe_tail_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'QR1_Severe_Tail_Risk', \
            f"Expected QR1_Severe_Tail_Risk, got {suite['rule_applied']}"
        assert suite['primary'] == 'LOCSUPR'

    def test_qr2_moderate_tail(self):
        """Tail pattern with reid_95 in [0.30, 0.40] → QR2 Moderate Tail."""
        df, qis, _ = build_qr2_moderate_tail_dataset()
        suite, features = get_suite(df, qis)
        assert 'QR2' in suite['rule_applied'], \
            f"Expected QR2, got {suite['rule_applied']}"
        assert suite['primary'] in ('LOCSUPR', 'kANON')

    def test_qr2_heavy_tail(self):
        """Heavy tail (reid_95 > 40%) → QR2 Heavy Tail (kANON or LOCSUPR)."""
        df, qis, _ = build_qr2_heavy_tail_dataset()
        suite, features = get_suite(df, qis)
        assert 'QR2' in suite['rule_applied'], \
            f"Expected QR2, got {suite['rule_applied']}"
        assert suite['primary'] in ('kANON', 'LOCSUPR')

    def test_qr4_widespread(self):
        """Widespread risk (reid_50 > 0.15, widespread pattern) → QR4."""
        df, qis, _ = build_qr4_widespread_dataset()
        suite, features = get_suite(df, qis)
        assert 'QR4' in suite['rule_applied'], \
            f"Expected QR4, got {suite['rule_applied']}"

    def test_med1_moderate_structural(self):
        """Moderate risk with significant tail → MED1 kANON."""
        df, qis, _ = build_med1_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'MED1_Moderate_Structural', \
            f"Expected MED1_Moderate_Structural, got {suite['rule_applied']}"
        assert suite['primary'] == 'kANON'


# ════════════════════════════════════════════════════════════════════════
# Risk concentration rules (RC1)
# ════════════════════════════════════════════════════════════════════════

class TestRiskConcentrationRules:
    """RC-series rules requiring per-QI risk data (var_priority).

    RC rules depend on features['var_priority'] from backward elimination,
    which is NOT populated by extract_data_features_with_reid(). Tests
    inject var_priority manually to verify the rule logic.
    """

    def test_rc1_fires_with_injected_var_priority(self):
        """Dominated risk (one QI >= 40% of risk) → RC1 LOCSUPR.

        var_priority and risk_concentration are injected manually because
        extract_data_features_with_reid() does not compute them.
        var_priority format: {qi: (label, percentage)}.
        """
        df, qis, _ = build_dominated_risk_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)

        # Inject var_priority with (label, percentage) format
        features['var_priority'] = {
            'job': ('HIGH', 65), 'sex': ('LOW', 15),
            'race': ('LOW', 12), 'region': ('LOW', 8),
        }
        # Inject risk_concentration derived from var_priority
        from sdc_engine.sdc.selection.features import classify_risk_concentration
        features['risk_concentration'] = classify_risk_concentration(
            features['var_priority'])

        result = risk_concentration_rules(features)
        assert result.get('applies'), \
            "RC1 should fire with var_priority showing dominated risk"
        assert 'RC1' in result.get('rule', ''), \
            f"Expected RC1, got {result.get('rule', '?')}"

    def test_rc_rules_dormant_without_var_priority(self):
        """Without var_priority, RC rules should NOT fire."""
        df, qis, _ = build_dominated_risk_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)

        result = risk_concentration_rules(features)
        assert not result.get('applies'), \
            "RC rules should not fire without var_priority"


# ════════════════════════════════════════════════════════════════════════
# Categorical-aware rules (CAT1, DYN_CAT)
# ════════════════════════════════════════════════════════════════════════

class TestCategoricalAwareRules:
    """CAT1, DYN_CAT pipeline, and the dominance guard."""

    def test_cat1_fires_on_categorical_dominant(self):
        """cat_ratio >= 0.70, reid_95 in [0.15, 0.40], no dominance → CAT1 PRAM."""
        df, qis, _ = build_cat1_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'CAT1_Categorical_Dominant', \
            f"Expected CAT1_Categorical_Dominant, got {suite['rule_applied']}"
        assert suite['primary'] == 'PRAM'

    def test_cat1_blocked_by_dominance_guard(self):
        """One category >= 80% → CAT1 must NOT fire."""
        df, qis, _ = build_cat1_blocked_by_dominance()
        suite, features = get_suite(df, qis)
        assert _has_dominant_categories(features), \
            "Dataset should have dominant categories (max freq >= 0.80)"
        assert 'CAT1' not in suite['rule_applied'], \
            f"CAT1 should be blocked by dominance guard; got {suite['rule_applied']}"

    def test_dyn_cat_pipeline_fires_on_mixed_categorical(self):
        """cat_ratio in (0.50, 0.70), reid_95 > 0.15 → DYN_CAT_Pipeline."""
        df, qis, _ = build_cat2_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'DYN_CAT_Pipeline', \
            f"Expected DYN_CAT_Pipeline, got {suite['rule_applied']}"
        # DYN_CAT produces a pipeline
        assert suite.get('use_pipeline'), \
            "DYN_CAT should set use_pipeline=True"


# ════════════════════════════════════════════════════════════════════════
# Low-risk rules (LOW1-LOW3)
# ════════════════════════════════════════════════════════════════════════

class TestLowRiskRules:
    """LOW-series rules for already low-risk data."""

    def test_low1_categorical(self):
        """cat_ratio >= 0.60, reid_95 <= 0.10 → LOW1 PRAM."""
        df, qis, _ = build_low1_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'LOW1_Categorical', \
            f"Expected LOW1_Categorical, got {suite['rule_applied']}"
        assert suite['primary'] == 'PRAM'

    def test_low2_continuous_noise(self):
        """cat_ratio <= 0.40, reid_95 <= 0.05, continuous-dominant → LOW2 NOISE."""
        df, qis, _ = build_low2_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'LOW2_Continuous_Noise', \
            f"Expected LOW2_Continuous_Noise, got {suite['rule_applied']}"
        assert suite['primary'] == 'NOISE'

    def test_low3_mixed(self):
        """cat_ratio ~0.50, reid_95 <= 0.10 → LOW3 kANON."""
        df, qis, _ = build_low3_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'LOW3_Mixed', \
            f"Expected LOW3_Mixed, got {suite['rule_applied']}"
        assert suite['primary'] == 'kANON'


# ════════════════════════════════════════════════════════════════════════
# Special structural rules (SR3, HR6)
# ════════════════════════════════════════════════════════════════════════

class TestStructuralRules:
    """SR3 (few QIs + near-unique) and HR6 (tiny dataset)."""

    def test_sr3_near_unique_few_qis(self):
        """<= 2 QIs, max_qi_uniqueness > 0.70, reid_95 > 0.20 → SR3 LOCSUPR."""
        df, qis, _ = build_sr3_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'SR3_Near_Unique_Few_QIs', \
            f"Expected SR3_Near_Unique_Few_QIs, got {suite['rule_applied']}"
        assert suite['primary'] == 'LOCSUPR'

    def test_hr6_very_small_dataset(self):
        """n < 200 rows, n_qis >= 2 → HR6 LOCSUPR."""
        df, qis, _ = build_hr6_dataset()
        suite, features = get_suite(df, qis)
        assert suite['rule_applied'] == 'HR6_Very_Small_Dataset', \
            f"Expected HR6_Very_Small_Dataset, got {suite['rule_applied']}"
        assert suite['primary'] == 'LOCSUPR'


# ════════════════════════════════════════════════════════════════════════
# Uniqueness risk rules (HR1-HR5) — dormant in current pipeline
# ════════════════════════════════════════════════════════════════════════

class TestUniquenessRiskRules:
    """HR1-HR5 uniqueness rules.

    These rules check features['uniqueness_rate'], which is always 0 in
    the current pipeline because analyze_data() puts uniqueness_rate in
    disclosure_risk sub-dict, not at the top level. Tests inject
    uniqueness_rate manually to verify the rule logic works.
    """

    def test_hr1_fires_with_injected_uniqueness(self):
        """uniqueness_rate > 0.20 → HR1 LOCSUPR."""
        df, qis, _ = build_extreme_uniqueness_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)

        # Inject uniqueness_rate (normally dormant)
        features['uniqueness_rate'] = 0.25

        result = uniqueness_risk_rules(features)
        assert result.get('applies'), \
            "HR1 should fire with uniqueness_rate=0.25"
        assert 'HR1' in result.get('rule', ''), \
            f"Expected HR1, got {result.get('rule', '?')}"
        assert result.get('method') == 'LOCSUPR'

    def test_hr3_fires_with_moderate_uniqueness(self):
        """uniqueness_rate > 0.05, n_qis >= 2 → HR3 kANON."""
        df, qis, _ = build_extreme_uniqueness_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)

        features['uniqueness_rate'] = 0.08  # between HR2 (0.10) and HR3 (0.05)

        result = uniqueness_risk_rules(features)
        assert result.get('applies'), \
            "HR3 should fire with uniqueness_rate=0.08 and n_qis=3"
        assert 'HR3' in result.get('rule', ''), \
            f"Expected HR3, got {result.get('rule', '?')}"

    def test_hr_rules_dormant_without_uniqueness(self):
        """Without injected uniqueness_rate, HR rules should NOT fire."""
        df, qis, _ = build_extreme_uniqueness_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)

        result = uniqueness_risk_rules(features)
        assert not result.get('applies'), \
            f"HR rules should not fire without uniqueness; got {result.get('rule', '?')}"


# ════════════════════════════════════════════════════════════════════════
# Rule priority and ordering
# ════════════════════════════════════════════════════════════════════════

class TestRulePriority:
    """Verify higher-priority rules win when multiple could apply."""

    def test_hr6_beats_qr_rules_on_tiny_data(self):
        """HR6 (priority 1b) fires before QR rules (priority 4) on small data."""
        df, qis, _ = build_hr6_dataset()
        suite, features = get_suite(df, qis)
        assert 'HR6' in suite['rule_applied'], \
            f"HR6 should fire on tiny dataset, got {suite['rule_applied']}"

    def test_sr3_fires_before_reid_risk_rules(self):
        """SR3 (priority 1c) fires before QR/MED1 (priority 4)."""
        df, qis, _ = build_sr3_dataset()
        suite, features = get_suite(df, qis)
        assert 'SR3' in suite['rule_applied'], \
            f"SR3 should fire before QR rules, got {suite['rule_applied']}"

    def test_cat1_fires_before_reid_risk_rules(self):
        """CAT1 (priority 3) fires before QR/MED1 (priority 4)."""
        df, qis, _ = build_cat1_dataset()
        suite, features = get_suite(df, qis)
        # CAT1 should fire — if QR2 or MED1 fired instead, priority is wrong
        assert 'CAT1' in suite['rule_applied'], \
            f"CAT1 should fire before reid_risk rules, got {suite['rule_applied']}"

    def test_pipeline_rules_fire_before_rule_factories(self):
        """DYN_CAT pipeline (checked first) beats CAT2 in rule_factories."""
        df, qis, _ = build_cat2_dataset()
        suite, features = get_suite(df, qis)
        # DYN_CAT_Pipeline fires from pipeline_rules (before CAT2 in rule_factories)
        assert suite['rule_applied'] == 'DYN_CAT_Pipeline', \
            f"Pipeline should fire before rule_factories CAT2, got {suite['rule_applied']}"


# ════════════════════════════════════════════════════════════════════════
# Context-aware rules (PUB1, SEC1, REG1)
# ════════════════════════════════════════════════════════════════════════

class TestPublicReleaseRules:
    """PUB1 rules — PUBLIC tier structural preference."""

    def test_pub1_high_risk_fires(self):
        """PUBLIC + reid_95 > 0.20 → PUB1_Public_Release_High_Risk kANON k=10."""
        df, qis, _ = build_pub1_high_risk_dataset()
        suite, features = get_suite(
            df, qis, access_tier='PUBLIC',
            feature_overrides={'_reid_target_raw': 0.01, '_utility_floor': 0.85},
        )
        assert suite['rule_applied'] == 'PUB1_Public_Release_High_Risk', \
            f"Expected PUB1_Public_Release_High_Risk, got {suite['rule_applied']}"
        assert suite['primary'] == 'kANON'

    def test_pub1_moderate_risk_fires(self):
        """PUBLIC + 0.05 < reid_95 <= 0.20 → PUB1_Public_Release_Moderate_Risk."""
        df, qis, _ = build_pub1_moderate_risk_dataset()
        suite, features = get_suite(
            df, qis, access_tier='PUBLIC',
            feature_overrides={'_reid_target_raw': 0.01, '_utility_floor': 0.85},
        )
        assert suite['rule_applied'] == 'PUB1_Public_Release_Moderate_Risk', \
            f"Expected PUB1_Public_Release_Moderate_Risk, got {suite['rule_applied']}"
        assert suite['primary'] == 'kANON'

    def test_pub1_skips_scientific_tier(self):
        """PUB1 should NOT fire under SCIENTIFIC tier."""
        df, qis, _ = build_pub1_high_risk_dataset()
        suite, features = get_suite(df, qis, access_tier='SCIENTIFIC')
        assert 'PUB1' not in suite['rule_applied'], \
            f"PUB1 should not fire for SCIENTIFIC, got {suite['rule_applied']}"

    def test_pub1_defers_to_reg1(self):
        """PUB1 should NOT fire when reid_target_raw=0.03 (regulatory)."""
        df, qis, _ = build_pub1_high_risk_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)
        features['_access_tier'] = 'PUBLIC'
        features['_reid_target_raw'] = 0.03
        result = public_release_rules(features)
        assert not result.get('applies'), \
            "PUB1 should defer to REG1 when reid_target_raw=0.03"


class TestSecureEnvironmentRules:
    """SEC1 rules — SECURE tier perturbation preference."""

    def test_sec1_categorical_fires(self):
        """SECURE + cat_ratio >= 0.60 + reid_95 in [0.05, 0.25] → SEC1 PRAM."""
        df, qis, _ = build_sec1_categorical_dataset()
        suite, features = get_suite(
            df, qis, access_tier='SECURE',
            feature_overrides={'_reid_target_raw': 0.10, '_utility_floor': 0.92},
        )
        assert suite['rule_applied'] == 'SEC1_Secure_Categorical', \
            f"Expected SEC1_Secure_Categorical, got {suite['rule_applied']}"
        assert suite['primary'] == 'PRAM'

    def test_sec1_continuous_fires(self):
        """SECURE + cat_ratio < 0.60 + continuous → SEC1 NOISE."""
        df, qis, _ = build_sec1_continuous_dataset()
        suite, features = get_suite(
            df, qis, access_tier='SECURE',
            feature_overrides={'_reid_target_raw': 0.10, '_utility_floor': 0.92},
        )
        assert suite['rule_applied'] == 'SEC1_Secure_Continuous', \
            f"Expected SEC1_Secure_Continuous, got {suite['rule_applied']}"
        assert suite['primary'] == 'NOISE'

    def test_sec1_skips_scientific_tier(self):
        """SEC1 should NOT fire under SCIENTIFIC tier."""
        df, qis, _ = build_sec1_categorical_dataset()
        suite, features = get_suite(df, qis, access_tier='SCIENTIFIC')
        assert 'SEC1' not in suite['rule_applied'], \
            f"SEC1 should not fire for SCIENTIFIC, got {suite['rule_applied']}"

    def test_sec1_skips_low_utility_floor(self):
        """SEC1 should NOT fire when utility_floor < 0.90."""
        df, qis, _ = build_sec1_categorical_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)
        features['_access_tier'] = 'SECURE'
        features['_utility_floor'] = 0.80
        result = secure_environment_rules(features)
        assert not result.get('applies'), \
            "SEC1 should not fire when utility_floor < 0.90"


class TestRegulatoryComplianceRules:
    """REG1 rules — regulatory compliance (PUBLIC + target=3%)."""

    def test_reg1_high_risk_fires(self):
        """PUBLIC + target=0.03 + reid_95 > 0.15 → REG1_Regulatory_High_Risk."""
        df, qis, _ = build_reg1_high_risk_dataset()
        suite, features = get_suite(
            df, qis, access_tier='PUBLIC',
            feature_overrides={'_reid_target_raw': 0.03, '_utility_floor': 0.88},
        )
        assert suite['rule_applied'] == 'REG1_Regulatory_High_Risk', \
            f"Expected REG1_Regulatory_High_Risk, got {suite['rule_applied']}"
        assert suite['primary'] == 'kANON'

    def test_reg1_moderate_risk_fires(self):
        """PUBLIC + target=0.03 + 0.03 < reid_95 <= 0.15 → REG1_Regulatory_Moderate_Risk."""
        df, qis, _ = build_reg1_moderate_risk_dataset()
        suite, features = get_suite(
            df, qis, access_tier='PUBLIC',
            feature_overrides={'_reid_target_raw': 0.03, '_utility_floor': 0.88},
        )
        assert suite['rule_applied'] == 'REG1_Regulatory_Moderate_Risk', \
            f"Expected REG1_Regulatory_Moderate_Risk, got {suite['rule_applied']}"
        assert suite['primary'] == 'kANON'

    def test_reg1_skips_public_release_target(self):
        """REG1 should NOT fire when target=0.01 (public_release)."""
        df, qis, _ = build_reg1_high_risk_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)
        features['_access_tier'] = 'PUBLIC'
        features['_reid_target_raw'] = 0.01
        result = regulatory_compliance_rules(features)
        assert not result.get('applies'), \
            "REG1 should not fire when reid_target_raw=0.01"

    def test_reg1_skips_scientific_tier(self):
        """REG1 should NOT fire under SCIENTIFIC tier."""
        df, qis, _ = build_reg1_high_risk_dataset()
        analysis = analyze_data(df, quasi_identifiers=qis, verbose=False)
        features = extract_data_features_with_reid(df, analysis, qis)
        features['_access_tier'] = 'SCIENTIFIC'
        features['_reid_target_raw'] = 0.03
        result = regulatory_compliance_rules(features)
        assert not result.get('applies'), \
            "REG1 should not fire under SCIENTIFIC tier"


class TestContextAwarePriority:
    """Verify context-aware rules fire at the correct priority."""

    def test_pub1_fires_before_cat1(self):
        """PUB1 (before CAT) should fire instead of CAT1 under PUBLIC tier."""
        df, qis, _ = build_cat1_dataset()
        # Under SCIENTIFIC, CAT1 fires. Under PUBLIC, PUB1 should preempt it.
        suite_sci, _ = get_suite(df, qis, access_tier='SCIENTIFIC')
        suite_pub, _ = get_suite(
            df, qis, access_tier='PUBLIC',
            feature_overrides={'_reid_target_raw': 0.01, '_utility_floor': 0.85},
        )
        assert 'CAT1' in suite_sci['rule_applied'], \
            f"Under SCIENTIFIC, CAT1 should fire; got {suite_sci['rule_applied']}"
        assert 'PUB1' in suite_pub['rule_applied'], \
            f"Under PUBLIC, PUB1 should preempt CAT1; got {suite_pub['rule_applied']}"

    def test_reg1_fires_before_everything(self):
        """REG1 fires first in chain — preempts all other rules."""
        df, qis, _ = build_reg1_high_risk_dataset()
        suite, features = get_suite(
            df, qis, access_tier='PUBLIC',
            feature_overrides={'_reid_target_raw': 0.03, '_utility_floor': 0.88},
        )
        assert 'REG1' in suite['rule_applied'], \
            f"REG1 should fire first; got {suite['rule_applied']}"

    def test_existing_rules_unchanged_under_scientific(self):
        """SCIENTIFIC tier should not trigger any context-aware rules."""
        df, qis, _ = build_cat1_dataset()
        suite, features = get_suite(df, qis, access_tier='SCIENTIFIC')
        for prefix in ('PUB1', 'SEC1', 'REG1'):
            assert prefix not in suite['rule_applied'], \
                f"Context rule {prefix} should not fire under SCIENTIFIC"


# ════════════════════════════════════════════════════════════════════════
# Default fallback
# ════════════════════════════════════════════════════════════════════════

class TestDefaultFallback:
    """When no specific high-risk rule matches, valid defaults should fire."""

    def test_default_produces_valid_result(self):
        """A simple low-risk dataset should produce a valid result, not crash."""
        df, qis, _ = build_default_fallback_dataset()
        suite, features = get_suite(df, qis)
        assert suite['primary'] in ('kANON', 'LOCSUPR', 'PRAM', 'NOISE'), \
            f"Expected valid method, got {suite['primary']}"
        assert suite['rule_applied'], "rule_applied should be non-empty"

    def test_suite_always_has_required_keys(self):
        """Every suite result must contain rule_applied, primary, and confidence."""
        df, qis, _ = build_default_fallback_dataset()
        suite, features = get_suite(df, qis)
        for key in ('rule_applied', 'primary', 'confidence'):
            assert key in suite, f"Suite missing required key: {key}"
