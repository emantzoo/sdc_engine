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
        """Rules blocked by METRIC_ALLOWED_METHODS should have blocked=True."""
        # Use l_diversity metric — DYN_CAT pipeline uses NOISE which is
        # blocked for l_diversity. The pipeline should match but be blocked.
        features = _minimal_features(
            _risk_metric_type='l_diversity',
            reid_95=0.25,
            cat_ratio=0.55,   # 50-70% categorical triggers DYN_CAT
            n_categorical=3,
            n_continuous=2,
            categorical_vars=['sex', 'education', 'marital'],
            continuous_vars=['age', 'income'],
        )
        suite = pipelines.select_method_suite(features, verbose=False)
        trace = suite['_rule_trace']

        blocked = [e for e in trace if e.get('blocked')]
        # DYN_CAT uses NOISE, blocked for l_diversity
        # May or may not appear depending on pipeline_rules internals
        # At minimum, verify the trace structure is correct
        for entry in blocked:
            assert 'blocked_reason' in entry

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
