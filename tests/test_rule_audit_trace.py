"""Tests for Spec 16 rule audit instrumentation.

Verifies that:
1. Flag-off: _rule_trace is NOT present in results.
2. Flag-on: _rule_trace IS present and contains expected structure.
3. Flag-on: the winning rule (rule_applied) is the same as flag-off.
4. Flag-on: trace includes entries for rules that didn't fire.
5. Flag-on: config-blocked rules are marked with blocked=True.
"""

import os
import pytest

from sdc_engine.sdc.selection import pipelines


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_features(**overrides):
    """Build a complete features dict that exercises select_method_suite."""
    base = {
        'quasi_identifiers': ['age', 'sex', 'education'],
        'n_qis': 3,
        'reid_95': 0.30,
        'reid_50': 0.15,
        'reid_99': 0.35,
        'n_continuous': 1,
        'n_categorical': 2,
        'continuous_vars': ['age'],
        'categorical_vars': ['sex', 'education'],
        'has_outliers': False,
        'has_reid': True,
        'uniqueness_rate': 0.05,
        'high_risk_rate': 0.08,
        'n_records': 5000,
        'cat_ratio': 0.67,
        'high_card_count': 0,
        'risk_pattern': 'widespread',
        'bimodal_risk': False,
        '_risk_metric_type': 'reid95',
        # Keys required by rule factories that access features directly:
        'data_type': 'microdata',
        'skewed_columns': [],
        'has_sensitive_attributes': False,
        'sensitive_columns': {},
        'sensitive_column_diversity': None,
        'k_anonymity_feasibility': 'feasible',
        'estimated_suppression': {},
        'qi_cardinality_product': 1000,
        'date_columns': [],
    }
    base.update(overrides)
    return base


def _suite_without_trace(suite):
    """Return a copy of suite without _rule_trace for comparison."""
    return {k: v for k, v in suite.items() if k != '_rule_trace'}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAuditFlagOff:
    """With SDC_RULE_AUDIT unset (default), no trace is present."""

    def test_no_trace_key(self):
        # Ensure flag is off
        old = os.environ.pop('SDC_RULE_AUDIT', None)
        # Reload the module-level flag
        pipelines._RULE_AUDIT = False
        try:
            features = _minimal_features()
            suite = pipelines.select_method_suite(features, verbose=False)
            assert '_rule_trace' not in suite
        finally:
            pipelines._RULE_AUDIT = False
            if old is not None:
                os.environ['SDC_RULE_AUDIT'] = old


class TestAuditFlagOn:
    """With SDC_RULE_AUDIT=1, trace is present and structured correctly."""

    @pytest.fixture(autouse=True)
    def _enable_audit(self):
        old = os.environ.get('SDC_RULE_AUDIT')
        os.environ['SDC_RULE_AUDIT'] = '1'
        pipelines._RULE_AUDIT = True
        yield
        pipelines._RULE_AUDIT = False
        if old is None:
            os.environ.pop('SDC_RULE_AUDIT', None)
        else:
            os.environ['SDC_RULE_AUDIT'] = old

    def test_trace_present(self):
        features = _minimal_features()
        suite = pipelines.select_method_suite(features, verbose=False)
        assert '_rule_trace' in suite
        assert isinstance(suite['_rule_trace'], list)
        assert len(suite['_rule_trace']) > 0

    def test_trace_entry_structure(self):
        features = _minimal_features()
        suite = pipelines.select_method_suite(features, verbose=False)
        for entry in suite['_rule_trace']:
            assert 'rule' in entry
            assert 'applies' in entry
            assert 'method' in entry

    def test_winner_unchanged(self):
        """The winning rule is the same with audit on vs off."""
        features = _minimal_features()

        # Run with audit off
        pipelines._RULE_AUDIT = False
        suite_off = pipelines.select_method_suite(features, verbose=False)

        # Run with audit on
        pipelines._RULE_AUDIT = True
        suite_on = pipelines.select_method_suite(features, verbose=False)

        assert suite_on['rule_applied'] == suite_off['rule_applied']
        assert suite_on['primary'] == suite_off['primary']
        assert suite_on['use_pipeline'] == suite_off['use_pipeline']

    def test_trace_includes_non_firing_rules(self):
        """Trace should include entries for rules that didn't fire."""
        features = _minimal_features()
        suite = pipelines.select_method_suite(features, verbose=False)
        trace = suite['_rule_trace']

        # There must be entries where applies=False (not all rules fire)
        non_firing = [e for e in trace if not e['applies']]
        assert len(non_firing) > 0, "Expected some non-firing rules in trace"

    def test_trace_has_pipeline_entry(self):
        """Trace should include the pipeline rules check."""
        features = _minimal_features()
        suite = pipelines.select_method_suite(features, verbose=False)
        trace = suite['_rule_trace']

        # First entry should be the pipeline check
        assert trace[0]['rule'] in ('PIPELINE', 'DYN_Pipeline',
                                     'DYN_CAT_Pipeline',
                                     'GEO1_Multi_Level_Geographic',
                                     'P4b_Skewed_Sensitive_Targeted',
                                     'P4a_Skewed_Structural',
                                     'P5_Small_Dataset_Mixed_Risks')

    def test_trace_covers_all_rule_factories(self):
        """Trace should have entries for all 15 rule factories + pipeline."""
        features = _minimal_features()
        suite = pipelines.select_method_suite(features, verbose=False)
        trace = suite['_rule_trace']

        # 1 pipeline entry + up to 15 rule factory entries
        # (some rule factories may produce named sub-rules)
        # At minimum: pipeline + default_rules = 2 entries
        assert len(trace) >= 2

        # The total should be 1 (pipeline) + 15 (factories) = 16
        # Some factories might produce applies=True with a named rule
        assert len(trace) == 16, (
            f"Expected 16 trace entries (1 pipeline + 15 factories), "
            f"got {len(trace)}: {[e['rule'] for e in trace]}"
        )

    def test_config_blocked_marked(self):
        """Rules blocked by METRIC_ALLOWED_METHODS should have blocked=True.

        SEC1_Secure_Categorical selects PRAM, which is blocked under
        k_anonymity.  Features are crafted so SEC1's gate matches
        (SECURE tier, reid_95 in 5-25%, cat_ratio >= 60%, utility_floor >= 90%),
        but the method is rejected by the metric filter.
        """
        features = _minimal_features(
            _risk_metric_type='k_anonymity',
            _utility_floor=0.92,
            reid_95=0.15,
            n_categorical=3,
            n_continuous=1,
            categorical_vars=['sex', 'education', 'marital'],
            continuous_vars=['age'],
            cat_ratio=0.75,
        )
        suite = pipelines.select_method_suite(
            features, access_tier='SECURE', verbose=False)
        trace = suite['_rule_trace']

        blocked = [e for e in trace if e.get('blocked')]
        assert len(blocked) > 0, (
            f"Expected at least one blocked entry under k_anonymity, "
            f"got none. Trace rules: {[e['rule'] for e in trace]}"
        )
        # Find the SEC1 blocked entry specifically
        sec1_blocked = [e for e in blocked
                        if 'SEC1' in str(e.get('rule', ''))]
        assert len(sec1_blocked) > 0, (
            f"Expected SEC1 to be blocked under k_anonymity. "
            f"Blocked entries: {blocked}"
        )
        for entry in blocked:
            assert 'blocked_reason' in entry
            assert 'metric' in entry['blocked_reason']

    def test_rc_sub_rule_trace(self):
        """RC rules should include sub_rules with per-sub-rule gate details."""
        features = _minimal_features(
            reid_95=0.30,
            var_priority={
                'age': ('HIGH', 75.0),
                'sex': ('LOW', 1.2),
                'education': ('LOW', 2.1),
            },
            risk_concentration={
                'pattern': 'dominated',
                'top_qi': 'age',
                'top_pct': 75.0,
                'top2_pct': 76.2,
                'n_high_risk': 1,
            },
        )
        suite = pipelines.select_method_suite(features, verbose=False)
        trace = suite['_rule_trace']

        # Find the risk_concentration entry
        rc_entries = [e for e in trace
                      if 'RC' in str(e.get('rule', ''))
                      or e.get('rule', '') == 'risk_concentration_rules']
        assert len(rc_entries) > 0, (
            f"Expected RC entry in trace. Rules: {[e['rule'] for e in trace]}"
        )

        rc_entry = rc_entries[0]
        assert 'sub_rules' in rc_entry, (
            f"Expected sub_rules in RC entry: {rc_entry}"
        )
        sub_rules = rc_entry['sub_rules']
        assert len(sub_rules) == 4, f"Expected 4 RC sub-rules, got {len(sub_rules)}"

        # RC1 should apply (dominated), RC2-RC4 should not
        names = {s['rule']: s['applies'] for s in sub_rules}
        assert names['RC1_Risk_Dominated'] is True
        assert names['RC2_Risk_Concentrated'] is False
        assert names['RC3_Risk_Spread_High'] is False
        assert names['RC4_Single_Bottleneck'] is False

        # Each sub-rule should have a gate description
        for s in sub_rules:
            assert 'gate' in s, f"Missing gate in sub-rule: {s}"

    def test_pipeline_winner_still_traces_all_factories(self):
        """When pipeline wins, trace should still include all rule factories."""
        # Build features that trigger DYN_Pipeline
        features = _minimal_features(
            reid_95=0.35,
            n_continuous=2,
            n_categorical=2,
            continuous_vars=['age', 'income'],
            categorical_vars=['sex', 'education'],
            has_outliers=True,
            high_risk_rate=0.20,
            cat_ratio=0.50,
        )
        suite = pipelines.select_method_suite(features, verbose=False)

        if suite['rule_applied'].startswith('DYN'):
            trace = suite['_rule_trace']
            # Even though pipeline won, all 15 rule factories should be traced
            # Total: 1 pipeline + 15 factories = 16
            assert len(trace) == 16, (
                f"Expected 16 entries when pipeline wins, got {len(trace)}"
            )
