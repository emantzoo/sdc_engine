"""
Spec 22 — Test 5: Rule-Metric Appropriateness
==============================================

Validates that every rule's primary method is compatible with at least one
metric, and that the ``select_method_suite`` global ``_is_allowed()`` gate
correctly blocks rules whose method is incompatible with the active metric.

Two layers:
    1. **Static**: For each rule in CANONICAL_RULES, assert its method
       appears in METRIC_ALLOWED_METHODS for at least one metric.
    2. **Runtime gating**: For (rule, metric) pairs where the method is
       blocked, verify select_method_suite skips the rule or the rule's
       own metric gate prevents it from firing.

Known limits:
    - This test catches method-metric mismatches, NOT data-regime mismatches
      (e.g., RC1 firing in a regime where LOCSUPR is infeasible). That's
      a subtler invariant beyond Spec 22 scope.
    - Pipeline rules (GEO1, DYN, P5) recommend lists of methods. Static
      check validates each list element; runtime check exercises the full
      chain.
"""
import pytest
from tests.fixtures.canonical_rules import CANONICAL_RULES

from sdc_engine.sdc.config import METRIC_ALLOWED_METHODS


# ---------------------------------------------------------------------------
# Canonical data for parameterization
# ---------------------------------------------------------------------------

ALL_METRICS = list(METRIC_ALLOWED_METHODS.keys())
# ['reid95', 'k_anonymity', 'uniqueness', 'l_diversity']

# Rules whose method is a list (pipeline rules)
PIPELINE_RULES = {
    name: info for name, info in CANONICAL_RULES.items()
    if isinstance(info.get('method'), list)
}

# Rules with a single method string
SINGLE_METHOD_RULES = {
    name: info for name, info in CANONICAL_RULES.items()
    if isinstance(info.get('method'), str)
}

# Rules with dynamic/None method (DYN_Pipeline)
DYNAMIC_RULES = {
    name: info for name, info in CANONICAL_RULES.items()
    if info.get('method') is None
}


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — Static: method-metric compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestStaticMethodMetricCompatibility:
    """Every rule's primary method must be allowed for at least one metric."""

    @pytest.mark.parametrize("rule_name,info", list(SINGLE_METHOD_RULES.items()),
                             ids=list(SINGLE_METHOD_RULES.keys()))
    def test_single_method_allowed_somewhere(self, rule_name, info):
        """Rule's primary method appears in METRIC_ALLOWED_METHODS for ≥1 metric."""
        method = info['method']
        allowed_metrics = [
            m for m, methods in METRIC_ALLOWED_METHODS.items()
            if method in methods
        ]
        assert allowed_metrics, (
            f"Rule {rule_name} recommends '{method}' which is not in "
            f"METRIC_ALLOWED_METHODS for any metric. "
            f"Available: {METRIC_ALLOWED_METHODS}"
        )

    @pytest.mark.parametrize("rule_name,info", list(PIPELINE_RULES.items()),
                             ids=list(PIPELINE_RULES.keys()))
    def test_pipeline_methods_allowed_somewhere(self, rule_name, info):
        """Each method in a pipeline rule's list must be allowed for ≥1 metric."""
        for method in info['method']:
            allowed_metrics = [
                m for m, methods in METRIC_ALLOWED_METHODS.items()
                if method in methods
            ]
            assert allowed_metrics, (
                f"Pipeline rule {rule_name} includes '{method}' which is not "
                f"in METRIC_ALLOWED_METHODS for any metric"
            )

    def test_dynamic_rules_acknowledged(self):
        """DYN_Pipeline has method=None — dynamic method. Just document."""
        for name, info in DYNAMIC_RULES.items():
            assert info['method'] is None, (
                f"Expected method=None for dynamic rule {name}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Static: identify blocked (rule, metric) pairs
# ═══════════════════════════════════════════════════════════════════════════

def _blocked_pairs():
    """Yield (rule_name, method, metric) tuples where the method is blocked."""
    for name, info in SINGLE_METHOD_RULES.items():
        method = info['method']
        for metric, allowed in METRIC_ALLOWED_METHODS.items():
            if method not in allowed:
                yield name, method, metric


def _allowed_pairs():
    """Yield (rule_name, method, metric) tuples where the method is allowed."""
    for name, info in SINGLE_METHOD_RULES.items():
        method = info['method']
        for metric, allowed in METRIC_ALLOWED_METHODS.items():
            if method in allowed:
                yield name, method, metric


class TestBlockedPairsDocumented:
    """Document which (rule, metric) pairs are blocked by _is_allowed()."""

    def test_pram_blocked_for_k_anon_and_uniqueness(self):
        """PRAM is not in METRIC_ALLOWED_METHODS for k_anonymity or uniqueness."""
        assert 'PRAM' not in METRIC_ALLOWED_METHODS['k_anonymity']
        assert 'PRAM' not in METRIC_ALLOWED_METHODS['uniqueness']

    def test_noise_blocked_for_k_anon_and_uniqueness(self):
        """NOISE is not in METRIC_ALLOWED_METHODS for k_anonymity or uniqueness."""
        assert 'NOISE' not in METRIC_ALLOWED_METHODS['k_anonymity']
        assert 'NOISE' not in METRIC_ALLOWED_METHODS['uniqueness']

    def test_pram_allowed_for_reid95_and_l_diversity(self):
        """PRAM is allowed for reid95 and l_diversity."""
        assert 'PRAM' in METRIC_ALLOWED_METHODS['reid95']
        assert 'PRAM' in METRIC_ALLOWED_METHODS['l_diversity']

    def test_structural_methods_universal(self):
        """kANON, LOCSUPR, GENERALIZE, GENERALIZE_FIRST allowed for all metrics."""
        universal = {'kANON', 'LOCSUPR', 'GENERALIZE', 'GENERALIZE_FIRST'}
        for method in universal:
            for metric, allowed in METRIC_ALLOWED_METHODS.items():
                assert method in allowed, (
                    f"{method} not in METRIC_ALLOWED_METHODS['{metric}']"
                )

    def test_blocked_pairs_are_non_empty(self):
        """There should be blocked pairs — otherwise the gating system is vacuous."""
        pairs = list(_blocked_pairs())
        assert len(pairs) > 0, "No blocked (rule, metric) pairs found"

    def test_enumerate_blocked_rules_by_metric(self):
        """Document which rules are blocked for each metric."""
        by_metric = {}
        for name, method, metric in _blocked_pairs():
            by_metric.setdefault(metric, []).append((name, method))

        # k_anonymity and uniqueness should block PRAM/NOISE rules
        assert 'k_anonymity' in by_metric, "Expected blocked rules for k_anonymity"
        assert 'uniqueness' in by_metric, "Expected blocked rules for uniqueness"

        # Verify specific known blocked rules
        k_anon_blocked = {name for name, method in by_metric['k_anonymity']}
        # These rules recommend PRAM — should be blocked for k_anonymity
        pram_rules = {name for name, info in SINGLE_METHOD_RULES.items()
                      if info['method'] == 'PRAM'}
        noise_rules = {name for name, info in SINGLE_METHOD_RULES.items()
                       if info['method'] == 'NOISE'}
        assert pram_rules <= k_anon_blocked, (
            f"PRAM rules not blocked for k_anonymity: {pram_rules - k_anon_blocked}"
        )
        assert noise_rules <= k_anon_blocked, (
            f"NOISE rules not blocked for k_anonymity: {noise_rules - k_anon_blocked}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — Explicit metric gate validation
# ═══════════════════════════════════════════════════════════════════════════

class TestExplicitMetricGates:
    """Validate that rules with explicit metric gates fire/block correctly."""

    def test_cat1_gated_to_l_diversity_only(self):
        """CAT1_Categorical_Dominant has an explicit gate: l_diversity only."""
        from sdc_engine.sdc.selection.rules import categorical_aware_rules
        from tests.test_systemic_rule_coverage import _base_features, _RULE_FEATURES

        base = _base_features(**_RULE_FEATURES['CAT1_Categorical_Dominant'])

        # Should fire under l_diversity
        base_ldiv = {**base, '_risk_metric_type': 'l_diversity'}
        result = categorical_aware_rules(base_ldiv)
        assert result['applies'] is True, "CAT1 should fire under l_diversity"
        assert result['rule'] == 'CAT1_Categorical_Dominant'

        # Should NOT fire under reid95, k_anonymity, uniqueness
        for metric in ('reid95', 'k_anonymity', 'uniqueness'):
            base_other = {**base, '_risk_metric_type': metric}
            result = categorical_aware_rules(base_other)
            assert result['applies'] is False, (
                f"CAT1 should NOT fire under {metric}"
            )

    def test_ldiv1_gated_reid95_threshold(self):
        """LDIV1 blocked for k_anonymity/uniqueness; under reid95 blocked if reid_95 > 0.10."""
        from sdc_engine.sdc.selection.rules import l_diversity_rules
        from tests.test_systemic_rule_coverage import _base_features, _RULE_FEATURES

        base = _base_features(**_RULE_FEATURES['LDIV1_Low_Sensitive_Diversity'])

        # Should fire under l_diversity
        base_ldiv = {**base, '_risk_metric_type': 'l_diversity'}
        result = l_diversity_rules(base_ldiv)
        assert result['applies'] is True, "LDIV1 should fire under l_diversity"

        # Should NOT fire under k_anonymity or uniqueness
        for metric in ('k_anonymity', 'uniqueness'):
            base_other = {**base, '_risk_metric_type': metric}
            result = l_diversity_rules(base_other)
            assert result['applies'] is False, (
                f"LDIV1 should NOT fire under {metric}"
            )

        # Under reid95 with reid_95 <= 0.10 → fires
        base_reid_low = {**base, '_risk_metric_type': 'reid95', 'reid_95': 0.08}
        result = l_diversity_rules(base_reid_low)
        assert result['applies'] is True, "LDIV1 should fire under reid95 if reid_95 <= 0.10"

        # Under reid95 with reid_95 > 0.10 → blocked
        base_reid_high = {**base, '_risk_metric_type': 'reid95', 'reid_95': 0.15}
        result = l_diversity_rules(base_reid_high)
        assert result['applies'] is False, (
            "LDIV1 should NOT fire under reid95 if reid_95 > 0.10"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4 — Runtime: full-chain sweep per metric
# ═══════════════════════════════════════════════════════════════════════════

class TestFullChainMetricSweep:
    """For each metric, run select_method_suite on multiple profiles and
    assert primary method is always allowed for that metric.

    Known issue: 8 pre-existing test_cross_metric.py failures share the
    same root cause (PRAM-gating under k_anonymity/uniqueness). If Test 5
    encounters similar issues, document as 'known from Spec 20 Item 1'.
    """

    @staticmethod
    def _make_features(metric, profile):
        """Build features dict with specific metric and profile."""
        from tests.test_systemic_rule_coverage import _base_features

        profiles = {
            'low_risk_categorical': {
                'n_categorical': 3, 'n_continuous': 0,
                'categorical_vars': ['a', 'b', 'c'],
                'continuous_vars': [],
                'reid_95': 0.03, 'reid_50': 0.01, 'reid_99': 0.05,
                'risk_pattern': 'uniform_low',
                'high_risk_rate': 0.0,
                'n_records': 2000,
                '_risk_metric_type': metric,
            },
            'low_risk_continuous': {
                'n_categorical': 0, 'n_continuous': 2,
                'categorical_vars': [],
                'continuous_vars': ['x', 'y'],
                'reid_95': 0.03, 'reid_50': 0.01, 'reid_99': 0.05,
                'risk_pattern': 'uniform_low',
                'high_risk_rate': 0.0,
                'n_records': 2000,
                '_risk_metric_type': metric,
            },
            'low_risk_mixed': {
                'n_categorical': 1, 'n_continuous': 1,
                'categorical_vars': ['a'],
                'continuous_vars': ['x'],
                'reid_95': 0.03, 'reid_50': 0.01, 'reid_99': 0.05,
                'risk_pattern': 'uniform_low',
                'high_risk_rate': 0.0,
                'n_records': 2000,
                '_risk_metric_type': metric,
            },
            'high_risk': {
                'n_categorical': 2, 'n_continuous': 1,
                'categorical_vars': ['a', 'b'],
                'continuous_vars': ['x'],
                'reid_95': 0.40, 'reid_50': 0.20, 'reid_99': 0.80,
                'risk_pattern': 'tail',
                'high_risk_rate': 0.25,
                'n_records': 500,
                '_risk_metric_type': metric,
            },
            'severe_risk': {
                'n_categorical': 3, 'n_continuous': 0,
                'categorical_vars': ['a', 'b', 'c'],
                'continuous_vars': [],
                'reid_95': 0.60, 'reid_50': 0.30, 'reid_99': 1.0,
                'risk_pattern': 'widespread',
                'high_risk_rate': 0.40,
                'n_records': 300,
                '_risk_metric_type': metric,
            },
        }
        return _base_features(**profiles[profile])

    @pytest.mark.parametrize("metric", ALL_METRICS)
    @pytest.mark.parametrize("profile", [
        'low_risk_categorical', 'low_risk_continuous', 'low_risk_mixed',
        'high_risk', 'severe_risk',
    ])
    def test_primary_allowed_for_metric(self, metric, profile):
        """select_method_suite primary is always in METRIC_ALLOWED_METHODS[metric]."""
        from sdc_engine.sdc.selection.pipelines import select_method_suite

        features = self._make_features(metric, profile)
        suite = select_method_suite(features, verbose=False)

        primary = suite['primary']
        allowed = METRIC_ALLOWED_METHODS[metric]

        # Pipeline primary may be a dict
        if isinstance(primary, dict):
            method = primary.get('method', primary)
        else:
            method = primary

        assert method in allowed, (
            f"metric={metric}, profile={profile}: "
            f"primary='{method}' not in allowed={allowed}. "
            f"Rule: {suite.get('rule_applied', '?')}"
        )

    @pytest.mark.parametrize("metric", ALL_METRICS)
    @pytest.mark.parametrize("profile", [
        'low_risk_categorical', 'low_risk_continuous', 'low_risk_mixed',
        'high_risk', 'severe_risk',
    ])
    def test_fallbacks_allowed_for_metric(self, metric, profile):
        """All fallback methods must also be allowed for the active metric."""
        from sdc_engine.sdc.selection.pipelines import select_method_suite

        features = self._make_features(metric, profile)
        suite = select_method_suite(features, verbose=False)

        allowed = METRIC_ALLOWED_METHODS[metric]
        for i, (fb_method, fb_params) in enumerate(suite.get('fallbacks', [])):
            assert fb_method in allowed, (
                f"metric={metric}, profile={profile}: "
                f"fallback[{i}]='{fb_method}' not in allowed={allowed}. "
                f"Rule: {suite.get('rule_applied', '?')}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 5 — Specific known-issue validation
# ═══════════════════════════════════════════════════════════════════════════

class TestKnownMetricIssues:
    """Validate specific fixes from Spec 19 and earlier."""

    def test_ldiv1_not_firing_at_high_reid95(self):
        """Spec 19 added reid95 gate to LDIV1. Verify it doesn't fire at reid_95=0.20."""
        from sdc_engine.sdc.selection.rules import l_diversity_rules
        from tests.test_systemic_rule_coverage import _base_features, _RULE_FEATURES

        base = _base_features(**_RULE_FEATURES['LDIV1_Low_Sensitive_Diversity'])
        # Override with high reid and reid95 metric
        features = {**base, '_risk_metric_type': 'reid95', 'reid_95': 0.20}
        result = l_diversity_rules(features)
        assert result['applies'] is False, (
            "LDIV1 should not fire under reid95 with reid_95=0.20 (> 0.10 gate). "
            "If this fails, the Spec 19 reid95 gate may have regressed."
        )

    def test_generalize_first_in_all_metrics(self):
        """GENERALIZE_FIRST was added to all metrics in Fix 0. Verify still present."""
        for metric, methods in METRIC_ALLOWED_METHODS.items():
            assert 'GENERALIZE_FIRST' in methods, (
                f"GENERALIZE_FIRST missing from METRIC_ALLOWED_METHODS['{metric}']. "
                f"Fix 0 regression."
            )

    def test_generalize_in_all_metrics(self):
        """GENERALIZE was added in Fix 0. Verify still present."""
        for metric, methods in METRIC_ALLOWED_METHODS.items():
            assert 'GENERALIZE' in methods, (
                f"GENERALIZE missing from METRIC_ALLOWED_METHODS['{metric}']"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6 — Smart Combo path: calculate_smart_defaults metric gate
# ═══════════════════════════════════════════════════════════════════════════

class TestSmartDefaultsMetricGate:
    """calculate_smart_defaults is a parallel method-selection path used by
    the Smart Combo UI mode.  Its method recommendation must also respect
    METRIC_ALLOWED_METHODS — otherwise the Smart Combo path can produce
    PRAM output under k_anonymity/uniqueness metrics.

    Added after retry engine read (Spec 20 follow-up, RF-1).
    """

    @staticmethod
    def _make_dataframe(n_qis):
        """Build a minimal DataFrame with the requested number of QI columns."""
        import pandas as pd
        n = 1000
        cols = {f'qi_{i}': range(n) for i in range(n_qis)}
        return pd.DataFrame(cols), list(cols.keys())

    @pytest.mark.parametrize("metric", ALL_METRICS)
    @pytest.mark.parametrize("n_qis", [3, 5, 10])
    def test_method_allowed_for_metric(self, metric, n_qis):
        """calculate_smart_defaults must never recommend a method blocked
        by METRIC_ALLOWED_METHODS for the active metric."""
        from sdc_engine.sdc.smart_defaults import calculate_smart_defaults

        df, qis = self._make_dataframe(n_qis)
        defaults = calculate_smart_defaults(
            df, qis, initial_reid_95=0.30, risk_metric=metric)

        method = defaults['method']
        allowed = METRIC_ALLOWED_METHODS[metric]
        assert method in allowed, (
            f"metric={metric}, n_qis={n_qis}: "
            f"calculate_smart_defaults recommended '{method}' "
            f"which is not in METRIC_ALLOWED_METHODS['{metric}']={allowed}"
        )

    def test_pram_blocked_for_k_anonymity_many_qis(self):
        """Regression guard: n_qis > 7 used to select PRAM unconditionally.
        Under k_anonymity, PRAM is blocked — must fall back to kANON."""
        from sdc_engine.sdc.smart_defaults import calculate_smart_defaults

        df, qis = self._make_dataframe(10)
        defaults = calculate_smart_defaults(
            df, qis, initial_reid_95=0.30, risk_metric='k_anonymity')

        assert defaults['method'] != 'PRAM', (
            "calculate_smart_defaults recommended PRAM under k_anonymity metric"
        )
        assert defaults['method'] == 'kANON'
        # Check reasoning trail includes the fallback explanation
        reasons = ' '.join(defaults.get('reasoning', []))
        assert 'blocked' in reasons.lower(), (
            f"Reasoning should mention metric blocking: {defaults['reasoning']}"
        )
