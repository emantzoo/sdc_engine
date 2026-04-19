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
        with pytest.raises(RuntimeError, match="No datasets registered"):
            run_matrix()


class TestCrossoverDetection:
    """Verify crossover detection on synthetic data."""

    def test_detects_crossover(self):
        """Method changes from kANON at low value to LOCSUPR at high value."""
        rows = [
            {"threshold_id": "T1", "dataset": "test", "threshold_value": 0.30,
             "selected_method": "kANON", "error": None},
            {"threshold_id": "T1", "dataset": "test", "threshold_value": 0.35,
             "selected_method": "kANON", "error": None},
            {"threshold_id": "T1", "dataset": "test", "threshold_value": 0.40,
             "selected_method": "LOCSUPR", "error": None},
            {"threshold_id": "T1", "dataset": "test", "threshold_value": 0.45,
             "selected_method": "LOCSUPR", "error": None},
        ]
        df = pd.DataFrame(rows)
        xovers = find_crossovers(df)

        assert len(xovers) == 1
        assert xovers.iloc[0]['observed_crossover'] == 0.40
        assert xovers.iloc[0]['threshold_id'] == 'T1'
        assert xovers.iloc[0]['shift_pp'] == 0  # 0.40 is current value

    def test_no_crossover(self):
        """Same method at all values -> no crossover."""
        rows = [
            {"threshold_id": "T2", "dataset": "test", "threshold_value": v,
             "selected_method": "PRAM", "error": None}
            for v in [0.60, 0.65, 0.70, 0.75, 0.80]
        ]
        df = pd.DataFrame(rows)
        xovers = find_crossovers(df)
        assert len(xovers) == 0

    def test_flags_large_shift(self):
        """Crossover at 0.30 when current is 0.40 -> shift -10pp -> flagged."""
        rows = [
            {"threshold_id": "T1", "dataset": "test", "threshold_value": 0.30,
             "selected_method": "LOCSUPR", "error": None},
            {"threshold_id": "T1", "dataset": "test", "threshold_value": 0.35,
             "selected_method": "LOCSUPR", "error": None},
            {"threshold_id": "T1", "dataset": "test", "threshold_value": 0.40,
             "selected_method": "kANON", "error": None},
        ]
        df = pd.DataFrame(rows)
        # Crossover at 0.40 (first value where method != low_method "LOCSUPR")
        xovers = find_crossovers(df)
        assert len(xovers) == 1
        assert xovers.iloc[0]['observed_crossover'] == 0.40
        # shift_pp = (0.40 - 0.40) * 100 = 0, not flagged
        # Let's make a real shift
        rows[2]['threshold_value'] = 0.50
        df2 = pd.DataFrame(rows)
        xovers2 = find_crossovers(df2)
        assert len(xovers2) == 1
        assert abs(xovers2.iloc[0]['shift_pp'] - 10) < 0.01  # (0.50 - 0.40) * 100
        assert "CONSIDER" in xovers2.iloc[0]['recommendation']
