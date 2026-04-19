"""
Self-tests for the empirical validation harness.

These verify the harness machinery itself (patchers, crossover detection),
NOT the threshold values.
"""
import pytest
import pandas as pd

from .thresholds import (
    _patch_rc1_dominated, _patch_cat1_ratio,
    _patch_qr2_suppression_gate, _patch_low1_reid_gate,
)
from .harness import find_crossovers, run_matrix


class TestPatcherIsolation:
    """Verify patchers restore original functions on exit."""

    def test_rc1_restores(self):
        from sdc_engine.sdc.selection import features as feat_mod
        original = feat_mod.classify_risk_concentration
        with _patch_rc1_dominated(0.30):
            assert feat_mod.classify_risk_concentration is not original
        assert feat_mod.classify_risk_concentration is original

    def test_cat1_restores(self):
        from sdc_engine.sdc.selection import rules as rules_mod
        original = rules_mod.categorical_aware_rules
        with _patch_cat1_ratio(0.60):
            assert rules_mod.categorical_aware_rules is not original
        assert rules_mod.categorical_aware_rules is original

    def test_qr2_restores(self):
        from sdc_engine.sdc.selection import rules as rules_mod
        original = rules_mod.reid_risk_rules
        with _patch_qr2_suppression_gate(0.20):
            assert rules_mod.reid_risk_rules is not original
        assert rules_mod.reid_risk_rules is original

    def test_low1_restores(self):
        from sdc_engine.sdc.selection import rules as rules_mod
        original = rules_mod.low_risk_rules
        with _patch_low1_reid_gate(0.05):
            assert rules_mod.low_risk_rules is not original
        assert rules_mod.low_risk_rules is original

    def test_rc1_restores_after_exception(self):
        """Patcher must restore even if body raises."""
        from sdc_engine.sdc.selection import features as feat_mod
        original = feat_mod.classify_risk_concentration
        with pytest.raises(ValueError):
            with _patch_rc1_dominated(0.30):
                raise ValueError("test")
        assert feat_mod.classify_risk_concentration is original


class TestEmptyDatasets:
    """Verify clear error when no datasets registered."""

    def test_run_matrix_raises(self):
        # Use a dataset filter that matches nothing
        with pytest.raises(RuntimeError, match="No datasets registered"):
            run_matrix(dataset_names=["__nonexistent__"])


def _make_row(tid, ds, value, rule, initial_method, method, error=None):
    """Helper to create a test row with all required columns."""
    return {
        "threshold_id": tid, "dataset": ds, "threshold_value": value,
        "selected_rule": rule, "initial_method": initial_method,
        "selected_method": method, "error": error,
    }


class TestCrossoverDetection:
    """Verify crossover detection on synthetic data."""

    def test_detects_rule_crossover(self):
        """Rule changes from LOW3 at low value to LOW1 at high value."""
        rows = [
            _make_row("T4", "test", 0.05, "LOW3_Mixed", "kANON", "kANON"),
            _make_row("T4", "test", 0.075, "LOW3_Mixed", "kANON", "kANON"),
            _make_row("T4", "test", 0.10, "LOW1_Categorical", "PRAM", "kANON"),
            _make_row("T4", "test", 0.125, "LOW1_Categorical", "PRAM", "kANON"),
        ]
        df = pd.DataFrame(rows)
        xovers = find_crossovers(df)

        assert len(xovers) == 1
        assert xovers.iloc[0]['observed_crossover'] == 0.10
        assert xovers.iloc[0]['threshold_id'] == 'T4'
        assert xovers.iloc[0]['low_rule'] == 'LOW3_Mixed'
        assert xovers.iloc[0]['high_rule'] == 'LOW1_Categorical'
        assert not xovers.iloc[0]['method_changed']  # both are kANON output

    def test_detects_method_crossover(self):
        """Method changes from kANON to LOCSUPR (also rule change)."""
        rows = [
            _make_row("T1", "test", 0.30, "QR2_kANON", "kANON", "kANON"),
            _make_row("T1", "test", 0.35, "QR2_kANON", "kANON", "kANON"),
            _make_row("T1", "test", 0.40, "RC1_LOCSUPR", "LOCSUPR", "LOCSUPR"),
            _make_row("T1", "test", 0.45, "RC1_LOCSUPR", "LOCSUPR", "LOCSUPR"),
        ]
        df = pd.DataFrame(rows)
        xovers = find_crossovers(df)

        assert len(xovers) == 1
        assert xovers.iloc[0]['observed_crossover'] == 0.40
        assert xovers.iloc[0]['method_changed']

    def test_no_crossover(self):
        """Same rule and method at all values -> no crossover."""
        rows = [
            _make_row("T2", "test", v, "MED1", "kANON", "kANON")
            for v in [0.60, 0.65, 0.70, 0.75, 0.80]
        ]
        df = pd.DataFrame(rows)
        xovers = find_crossovers(df)
        assert len(xovers) == 0

    def test_flags_large_shift(self):
        """Crossover far from current value -> flagged."""
        rows = [
            _make_row("T1", "test", 0.30, "RuleA", "LOCSUPR", "LOCSUPR"),
            _make_row("T1", "test", 0.35, "RuleA", "LOCSUPR", "LOCSUPR"),
            _make_row("T1", "test", 0.50, "RuleB", "kANON", "kANON"),
        ]
        df = pd.DataFrame(rows)
        xovers = find_crossovers(df)
        assert len(xovers) == 1
        assert abs(xovers.iloc[0]['shift_pp'] - 10) < 0.01  # (0.50 - 0.40) * 100
        assert "CONSIDER" in xovers.iloc[0]['recommendation']
