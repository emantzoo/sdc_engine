"""
Tests for the feasibility badge utility (Spec 17 — preflight floor estimator).

Exercises all four badge states using synthetic expected_eq_size values.
"""

import sys
import os

# Add streamlit_app to path so we can import classify_feasibility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'streamlit_app'))

from components import classify_feasibility


class TestClassifyFeasibility:
    """All four badge states from classify_feasibility()."""

    def test_comfortable_green(self):
        """expected_eq_size >= 10 → Comfortable (green)."""
        color, label, message, suggestion = classify_feasibility(15.0)
        assert label == "Comfortable"
        assert color == "#27ae60"
        assert suggestion == ""

    def test_comfortable_boundary(self):
        """expected_eq_size == 10 → Comfortable (green, boundary)."""
        color, label, _, suggestion = classify_feasibility(10.0)
        assert label == "Comfortable"
        assert suggestion == ""

    def test_tight_amber(self):
        """expected_eq_size in [5, 10) → Tight (amber)."""
        color, label, message, suggestion = classify_feasibility(7.5)
        assert label == "Tight"
        assert color == "#f39c12"
        assert "aggressive" in suggestion.lower()

    def test_tight_boundary(self):
        """expected_eq_size == 5 → Tight (amber, boundary)."""
        color, label, _, _ = classify_feasibility(5.0)
        assert label == "Tight"

    def test_infeasible_low_targets_red(self):
        """expected_eq_size in [2, 5) → Infeasible for low targets (red)."""
        qi_cards = {"age": 50, "region": 10, "job": 200}
        color, label, message, suggestion = classify_feasibility(3.5, qi_cards)
        assert label == "Infeasible for low targets"
        assert color == "#e74c3c"
        assert "job" in suggestion  # highest cardinality QI

    def test_infeasible_red(self):
        """expected_eq_size < 2 → Infeasible (red)."""
        qi_cards = {"income": 500, "sex": 2, "edu": 15}
        color, label, message, suggestion = classify_feasibility(0.8, qi_cards)
        assert label == "Infeasible"
        assert color == "#e74c3c"
        assert "income" in suggestion  # highest cardinality QI

    def test_infeasible_without_qi_cardinalities(self):
        """Red band without qi_cardinalities still works."""
        color, label, _, suggestion = classify_feasibility(1.0)
        assert label == "Infeasible"
        assert "combination space" in suggestion.lower()

    def test_return_type(self):
        """All states return 4-tuple of strings."""
        for eq_size in [0.5, 3.0, 7.0, 15.0]:
            result = classify_feasibility(eq_size)
            assert isinstance(result, tuple)
            assert len(result) == 4
            assert all(isinstance(s, str) for s in result)
