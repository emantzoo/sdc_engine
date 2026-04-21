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

from sdc_engine.sdc.protection_engine import build_data_features
from sdc_engine.sdc.selection import (
    select_method_suite,
)

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
    build_small_dominated_dataset,
)


# ════════════════════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════════════════════

def get_suite(df, qis, access_tier='standard', feature_overrides=None):
    """Run the full rule chain and return the suite dict.

    Strips var_priority/risk_concentration so RC rules stay dormant
    by default — tests that need RC behaviour inject them via overrides.
    """
    features = build_data_features(df, qis)
    features.pop('var_priority', None)
    features.pop('risk_concentration', None)
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

    RC rules depend on features['var_priority'] from backward elimination.
    Tests strip the auto-computed var_priority and inject it manually to
    control the concentration pattern.
    """

    def test_rc1_fires_with_injected_var_priority(self):
        """Dominated risk (one QI >= 40% of risk) → RC1 LOCSUPR.

        var_priority and risk_concentration are injected manually to
        control the concentration pattern.
        var_priority format: {qi: (label, percentage)}.
        """
        df, qis, _ = build_dominated_risk_dataset()
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)

        # Inject var_priority with (label, percentage) format
        features['var_priority'] = {
            'job': ('HIGH', 65), 'sex': ('LOW', 15),
            'race': ('LOW', 12), 'region': ('LOW', 8),
        }
        # Inject risk_concentration derived from var_priority
        from sdc_engine.sdc.selection.features import classify_risk_concentration
        features['risk_concentration'] = classify_risk_concentration(
            features['var_priority'])
        # Mark feasible — this test targets RC1's dominated-pattern logic,
        # not the infeasibility gate (tested separately in
        # test_rc1_defers_when_infeasible).
        features['k_anonymity_feasibility'] = 'feasible'

        result = risk_concentration_rules(features)
        assert result.get('applies'), \
            "RC1 should fire with var_priority showing dominated risk"
        assert 'RC1' in result.get('rule', ''), \
            f"Expected RC1, got {result.get('rule', '?')}"

    def test_rc_rules_dormant_without_var_priority(self):
        """Without var_priority, RC rules should NOT fire."""
        df, qis, _ = build_dominated_risk_dataset()
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)

        result = risk_concentration_rules(features)
        assert not result.get('applies'), \
            "RC rules should not fire without var_priority"

    def test_rc1_defers_when_infeasible(self):
        """RC1 must NOT fire on infeasible data — QR0 should handle it.

        When k_anonymity_feasibility == 'infeasible', RC1's LOCSUPR k=5
        produces 20%+ suppression on raw high-cardinality data. QR0's
        GENERALIZE_FIRST addresses the root cause (~1% suppression).
        Regression discovered in Spec 19 Phase 1.4.
        """
        df, qis, _ = build_dominated_risk_dataset()
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)

        # Inject var_priority with dominated pattern (would normally fire RC1)
        features['var_priority'] = {
            'job': ('HIGH', 65), 'sex': ('LOW', 15),
            'race': ('LOW', 12), 'region': ('LOW', 8),
        }
        from sdc_engine.sdc.selection.features import classify_risk_concentration
        features['risk_concentration'] = classify_risk_concentration(
            features['var_priority'])

        # Mark data as infeasible
        features['k_anonymity_feasibility'] = 'infeasible'

        # RC1 directly should defer
        result = risk_concentration_rules(features)
        assert not result.get('applies'), \
            "RC1 should NOT fire when k_anonymity_feasibility == 'infeasible'"

        # Full chain: QR0 should fire instead with GENERALIZE_FIRST
        suite = select_method_suite(features, access_tier='standard',
                                    verbose=False)
        assert 'QR0' in suite['rule_applied'], \
            f"Expected QR0 on infeasible data, got {suite['rule_applied']}"
        assert suite['primary'] == 'GENERALIZE_FIRST', \
            f"Expected GENERALIZE_FIRST, got {suite['primary']}"


# ════════════════════════════════════════════════════════════════════════
# Categorical-aware rules (CAT1)
# ════════════════════════════════════════════════════════════════════════

class TestCategoricalAwareRules:
    """CAT1 and the dominance guard.

    DYN_CAT and CAT2 deleted in Spec 19 Phase 2 — self-contradictory
    (gated to l_diversity but used NOISE, blocked for l_diversity).
    """

    def test_cat1_gated_for_reid95(self):
        """CAT1 must NOT fire when risk_metric is reid95 (default).

        PRAM invalidates frequency-count-based risk metrics like reid_95.
        sdcMicro: "Risk measures based on frequency counts of keys are
        no longer valid after perturbative methods."
        """
        df, qis, _ = build_cat1_dataset()
        suite, features = get_suite(df, qis)
        assert 'CAT1' not in suite['rule_applied'], \
            f"CAT1 should be gated for reid95; got {suite['rule_applied']}"
        assert suite['primary'] != 'PRAM', \
            f"PRAM should not be selected as primary for reid95; got {suite['primary']}"

    def test_cat1_fires_for_l_diversity(self):
        """CAT1 fires when risk_metric is l_diversity (attribute disclosure)."""
        df, qis, _ = build_cat1_dataset()
        suite, features = get_suite(
            df, qis, feature_overrides={'_risk_metric_type': 'l_diversity'})
        assert suite['rule_applied'] == 'CAT1_Categorical_Dominant', \
            f"Expected CAT1_Categorical_Dominant for l_diversity, got {suite['rule_applied']}"
        assert suite['primary'] == 'PRAM'

    def test_cat1_blocked_by_dominance_guard(self):
        """One category >= 80% → CAT1 must NOT fire (even for l_diversity)."""
        df, qis, _ = build_cat1_blocked_by_dominance()
        suite, features = get_suite(
            df, qis, feature_overrides={'_risk_metric_type': 'l_diversity'})
        assert _has_dominant_categories(features), \
            "Dataset should have dominant categories (max freq >= 0.80)"
        assert 'CAT1' not in suite['rule_applied'], \
            f"CAT1 should be blocked by dominance guard; got {suite['rule_applied']}"

    # DYN_CAT tests removed — rule deleted in Spec 19 Phase 2.
    # (Self-contradictory: gated to l_diversity, pipeline used NOISE.)


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
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)

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
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)

        features['uniqueness_rate'] = 0.08  # between HR2 (0.10) and HR3 (0.05)

        result = uniqueness_risk_rules(features)
        assert result.get('applies'), \
            "HR3 should fire with uniqueness_rate=0.08 and n_qis=3"
        assert 'HR3' in result.get('rule', ''), \
            f"Expected HR3, got {result.get('rule', '?')}"

    def test_hr_rules_preempted_in_chain(self):
        """HR rules never fire in full chain when has_reid=True.

        Production always computes uniqueness_rate (non-zero), so HR
        rules WOULD fire if reached.  But reid_risk_rules fires first
        when has_reid=True, preempting HR at priority 15.
        """
        df, qis, _ = build_extreme_uniqueness_dataset()
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)

        # Verify uniqueness_rate is non-zero under production semantics
        assert features['uniqueness_rate'] > 0, \
            "Production should compute non-zero uniqueness_rate"
        # HR rules would fire if called directly
        result = uniqueness_risk_rules(features)
        assert result.get('applies'), \
            "HR rules should fire when uniqueness_rate > 0.05"
        # But in the full chain, higher-priority rules preempt HR
        suite = select_method_suite(features, access_tier='standard',
                                    verbose=False)
        assert 'HR' not in suite['rule_applied'], \
            f"HR rules should be preempted in full chain, got {suite['rule_applied']}"


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

    def test_cat1_fires_before_reid_risk_rules_for_l_diversity(self):
        """CAT1 (priority 3) fires before QR/MED1 (priority 4) under l_diversity."""
        df, qis, _ = build_cat1_dataset()
        suite, features = get_suite(
            df, qis, feature_overrides={'_risk_metric_type': 'l_diversity'})
        # CAT1 should fire — if QR2 or MED1 fired instead, priority is wrong
        assert 'CAT1' in suite['rule_applied'], \
            f"CAT1 should fire before reid_risk rules, got {suite['rule_applied']}"

    # DYN_CAT/CAT2 combined test removed — both rules deleted in Spec 19 Phase 2.


# ════════════════════════════════════════════════════════════════════════
# L-diversity rules (LDIV1)
# ════════════════════════════════════════════════════════════════════════

# LDIV1 features injected via overrides because get_suite/build_data_features
# doesn't receive sensitive_columns.  The overrides simulate a dataset with
# binary sensitive column (diversity=2, min_l=1).
_LDIV1_OVERRIDES = {
    'sensitive_column_diversity': 2,
    'min_l': 1,
    'has_sensitive_attributes': True,
    'sensitive_columns': {'income': {'n_unique': 2}},
}


class TestLDiversityRules:
    """LDIV1 fires for low sensitive-column diversity when appropriate."""

    def test_ldiv1_fires_low_reid95(self):
        """LDIV1 fires under reid95 when reid_95 <= 0.10."""
        df, qis, _ = build_low1_dataset()  # reid_95 ~ 0.06
        suite, features = get_suite(df, qis, feature_overrides=_LDIV1_OVERRIDES)
        assert features['reid_95'] <= 0.10, \
            f"Precondition: need reid_95 <= 0.10, got {features['reid_95']:.3f}"
        assert suite['rule_applied'] == 'LDIV1_Low_Sensitive_Diversity', \
            f"Expected LDIV1 at low reid95, got {suite['rule_applied']}"
        assert suite['primary'] == 'PRAM'

    def test_ldiv1_gated_high_reid95(self):
        """LDIV1 must NOT fire under reid95 when reid_95 > 0.10.

        PRAM on sensitive columns doesn't reduce QI-based re-identification
        risk.  At elevated reid95, defer to QR/MED rules (Spec 19).
        """
        df, qis, _ = build_med1_dataset()  # reid_95 ~ 0.25
        suite, features = get_suite(df, qis, feature_overrides=_LDIV1_OVERRIDES)
        assert features['reid_95'] > 0.10, \
            f"Precondition: need reid_95 > 0.10, got {features['reid_95']:.3f}"
        assert 'LDIV1' not in suite['rule_applied'], \
            f"LDIV1 should be gated at reid_95={features['reid_95']:.3f}, got {suite['rule_applied']}"

    def test_ldiv1_fires_l_diversity_any_reid(self):
        """LDIV1 fires under l_diversity metric regardless of reid_95 level.

        Uses sec1_continuous dataset (cat_ratio=0.50, reid_95~0.14) to avoid
        triggering CAT1 (requires cat_ratio >= 0.70) while still having
        elevated reid_95.
        """
        df, qis, _ = build_sec1_continuous_dataset()  # reid_95 ~ 0.14
        overrides = {**_LDIV1_OVERRIDES, '_risk_metric_type': 'l_diversity'}
        suite, features = get_suite(df, qis, feature_overrides=overrides)
        assert features['reid_95'] > 0.10, \
            f"Precondition: need elevated reid_95, got {features['reid_95']:.3f}"
        assert suite['rule_applied'] == 'LDIV1_Low_Sensitive_Diversity', \
            f"Expected LDIV1 under l_diversity, got {suite['rule_applied']}"

    def test_ldiv1_gated_k_anonymity(self):
        """LDIV1 must NOT fire under k_anonymity (PRAM blocked)."""
        df, qis, _ = build_low1_dataset()
        overrides = {**_LDIV1_OVERRIDES, '_risk_metric_type': 'k_anonymity'}
        suite, features = get_suite(df, qis, feature_overrides=overrides)
        assert 'LDIV1' not in suite['rule_applied'], \
            f"LDIV1 should be gated for k_anonymity, got {suite['rule_applied']}"


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
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)
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
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)
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
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)
        features['_access_tier'] = 'PUBLIC'
        features['_reid_target_raw'] = 0.01
        result = regulatory_compliance_rules(features)
        assert not result.get('applies'), \
            "REG1 should not fire when reid_target_raw=0.01"

    def test_reg1_skips_scientific_tier(self):
        """REG1 should NOT fire under SCIENTIFIC tier."""
        df, qis, _ = build_reg1_high_risk_dataset()
        features = build_data_features(df, qis)
        features.pop('var_priority', None)
        features.pop('risk_concentration', None)
        features['_access_tier'] = 'SCIENTIFIC'
        features['_reid_target_raw'] = 0.03
        result = regulatory_compliance_rules(features)
        assert not result.get('applies'), \
            "REG1 should not fire under SCIENTIFIC tier"


class TestContextAwarePriority:
    """Verify context-aware rules fire at the correct priority."""

    def test_pub1_fires_before_cat1(self):
        """PUB1 (before CAT) should fire instead of CAT1 under PUBLIC tier.

        Both tested under l_diversity where CAT1 is active.
        """
        df, qis, _ = build_cat1_dataset()
        # Under SCIENTIFIC + l_diversity, CAT1 fires.
        # Under PUBLIC + l_diversity, PUB1 should preempt it.
        suite_sci, _ = get_suite(
            df, qis, access_tier='SCIENTIFIC',
            feature_overrides={'_risk_metric_type': 'l_diversity'})
        suite_pub, _ = get_suite(
            df, qis, access_tier='PUBLIC',
            feature_overrides={'_reid_target_raw': 0.01, '_utility_floor': 0.85,
                               '_risk_metric_type': 'l_diversity'},
        )
        assert 'CAT1' in suite_sci['rule_applied'], \
            f"Under SCIENTIFIC + l_diversity, CAT1 should fire; got {suite_sci['rule_applied']}"
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


# ════════════════════════════════════════════════════════════════════════
# Dormant-rule activation (Spec 07): RC rules via organic var_priority
# ════════════════════════════════════════════════════════════════════════

def _get_suite_with_var_priority(df, qis):
    """Build features via build_data_features (includes lazy var_priority)."""
    from sdc_engine.sdc.protection_engine import build_data_features
    features = build_data_features(df, qis)
    return select_method_suite(features, access_tier='standard', verbose=False), features


class TestDormantRulesNowActive:
    """Verify RC rules fire organically after var_priority is populated.

    Before Spec 07: these tests would fire DEFAULT/QR.
    After Spec 07: RC rules fire naturally on small datasets.
    """

    def test_rc1_fires_organically_on_dominated_data(self):
        """Small dataset with one dominant QI -> RC1 via lazy computation."""
        df, qis, _ = build_small_dominated_dataset()
        suite, features = _get_suite_with_var_priority(df, qis)
        assert 'RC1' in suite['rule_applied'], \
            f"Expected RC1, got {suite['rule_applied']}"
        assert suite['primary'] == 'LOCSUPR'

    def test_rc_rules_skipped_on_large_dataset(self):
        """Large dataset should NOT compute var_priority (perf guard)."""
        import pandas as pd
        df, qis, _ = build_small_dominated_dataset()
        df_large = pd.concat([df] * 20, ignore_index=True)  # 12000 rows
        suite, features = _get_suite_with_var_priority(df_large, qis)
        assert 'RC' not in suite['rule_applied'], \
            f"RC rule fired on large data — perf guard failed: {suite['rule_applied']}"

    def test_var_priority_populated_for_small_data(self):
        """Directly verify var_priority is in features dict for small data."""
        from sdc_engine.sdc.protection_engine import build_data_features
        df, qis, _ = build_small_dominated_dataset()
        features = build_data_features(df, qis)
        assert features.get('var_priority'), \
            "var_priority should be populated for small dataset"
        assert 'risk_concentration' in features, \
            "risk_concentration should be classified"
