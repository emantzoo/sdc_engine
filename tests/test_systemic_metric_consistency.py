"""
Spec 22 — Test 4: Metric Consistency
=====================================

Validates that named concepts with multiple implementations or consumers
produce consistent results.  Catches divergence like the dual risk_pattern
classifier (fixed in Spec 19) from recurring.

Tests:
    1. classify_risk_pattern determinism — same input always → same output
    2. classify_risk_pattern boundary behavior — codify observed tie-breaking
       at threshold values (0.20, 0.30, 0.05, etc.)
    3. var_priority label thresholds — verify 15%/8%/3% boundaries
    4. build_data_features determinism — two calls on identical input → same output
    5. risk_drop_pct dual-path documentation — codify known architectural debt

Philosophy:
    Boundary tests codify *observed* behavior, not *designed-from-spec* behavior.
    If the classifier changes its boundary tie-breaking in a future commit,
    these tests surface the change as intentional (update the test) rather
    than accidental (regression).
"""
import pytest
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Test 1 — classify_risk_pattern determinism
# ---------------------------------------------------------------------------

# Representative inputs spanning all 7 patterns
_PATTERN_INPUTS = {
    'uniform_high': {'reid_50': 0.30, 'reid_95': 0.35, 'reid_99': 0.38, 'mean_risk': 0.31},
    'widespread': {'reid_50': 0.25, 'reid_95': 0.50, 'reid_99': 0.80, 'mean_risk': 0.30},
    'severe_tail': {'reid_50': 0.02, 'reid_95': 0.05, 'reid_99': 0.80, 'mean_risk': 0.05},
    'tail': {'reid_50': 0.10, 'reid_95': 0.35, 'reid_99': 0.60, 'mean_risk': 0.15},
    'uniform_low': {'reid_50': 0.01, 'reid_95': 0.02, 'reid_99': 0.03, 'mean_risk': 0.01},
    'bimodal': {'reid_50': 0.10, 'reid_95': 0.15, 'reid_99': 0.18, 'mean_risk': 0.30},
    'moderate': {'reid_50': 0.10, 'reid_95': 0.15, 'reid_99': 0.20, 'mean_risk': 0.11},
}


class TestClassifyRiskPatternDeterminism:
    """Same input always → same output (10 repeated calls per input)."""

    @pytest.mark.parametrize("pattern,reid_dict",
                             list(_PATTERN_INPUTS.items()),
                             ids=list(_PATTERN_INPUTS.keys()))
    def test_repeated_calls_stable(self, pattern, reid_dict):
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        results = [classify_risk_pattern(reid_dict) for _ in range(10)]
        assert all(r == pattern for r in results), (
            f"Non-deterministic: expected all '{pattern}', got {set(results)}"
        )

    def test_all_seven_patterns_reachable(self):
        """Verify the representative inputs actually produce all 7 patterns."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        observed = {classify_risk_pattern(d) for d in _PATTERN_INPUTS.values()}
        expected = set(_PATTERN_INPUTS.keys())
        assert observed == expected, (
            f"Missing patterns: {expected - observed}, "
            f"unexpected: {observed - expected}"
        )


# ---------------------------------------------------------------------------
# Test 2 — classify_risk_pattern boundary behavior
# ---------------------------------------------------------------------------

class TestClassifyRiskPatternBoundaries:
    """Codify observed behavior at exact threshold values.

    These tests document what the classifier does at boundaries.
    If a future change alters boundary behavior, the test surfaces
    the change as intentional (update) rather than accidental (regression).

    Key thresholds from the classifier:
        reid_50 > 0.20  → uniform_high or widespread branch
        reid_50 == 0.20 → NOT in that branch (strict >)
        reid_99 - reid_50 < 0.10 → uniform_high (vs widespread)
        tail_ratio > 10, reid_99 > 0.30 → severe_tail
        reid_95 > 0.30 → tail
        reid_50 < 0.05 → uniform_low branch (or tail if reid_99 high)
        abs(mean - median) > 0.15 → bimodal
    """

    def test_reid50_exactly_020(self):
        """reid_50=0.20 exactly: NOT in uniform_high branch (strict >)."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        # reid_50=0.20 exactly, not > 0.20, so falls through to later checks
        result = classify_risk_pattern({
            'reid_50': 0.20, 'reid_95': 0.25, 'reid_99': 0.30, 'mean_risk': 0.21,
        })
        # reid_95=0.25 not > 0.30, reid_50 not < 0.05, |mean-median|=0.01 not > 0.15
        assert result == 'moderate', (
            f"reid_50=0.20 exactly: observed '{result}', codified as 'moderate'"
        )

    def test_reid50_just_above_020(self):
        """reid_50=0.201: enters uniform_high/widespread branch."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0.201, 'reid_95': 0.25, 'reid_99': 0.28, 'mean_risk': 0.21,
        })
        # reid_99 - reid_50 = 0.079 < 0.10 → uniform_high
        assert result == 'uniform_high', (
            f"reid_50=0.201: observed '{result}', codified as 'uniform_high'"
        )

    def test_reid95_exactly_030(self):
        """reid_95=0.30 exactly: NOT in tail branch (strict >)."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0.10, 'reid_95': 0.30, 'reid_99': 0.40, 'mean_risk': 0.12,
        })
        # reid_50 not > 0.20, tail_ratio=4 not > 10 so not severe_tail
        # reid_95=0.30 not > 0.30, so not tail (first check)
        # but reid_99=0.40 not > 0.50, so not tail (second check)
        # reid_50=0.10 not < 0.05
        # |mean - median| = 0.02 not > 0.15
        assert result == 'moderate', (
            f"reid_95=0.30 exactly: observed '{result}', codified as 'moderate'"
        )

    def test_reid95_just_above_030(self):
        """reid_95=0.301: enters tail branch."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0.10, 'reid_95': 0.301, 'reid_99': 0.40, 'mean_risk': 0.12,
        })
        assert result == 'tail', (
            f"reid_95=0.301: observed '{result}', codified as 'tail'"
        )

    def test_reid50_exactly_005(self):
        """reid_50=0.05 exactly: NOT in uniform_low branch (strict <)."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0.05, 'reid_95': 0.10, 'reid_99': 0.15, 'mean_risk': 0.06,
        })
        # reid_50 not > 0.20, not in tail checks, not < 0.05
        # |mean-median|=0.01 not > 0.15 → moderate
        assert result == 'moderate', (
            f"reid_50=0.05 exactly: observed '{result}', codified as 'moderate'"
        )

    def test_reid50_just_below_005(self):
        """reid_50=0.049: enters uniform_low branch."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0.049, 'reid_95': 0.10, 'reid_99': 0.15, 'mean_risk': 0.05,
        })
        # reid_50 < 0.05, reid_99=0.15 not > 0.20 → uniform_low
        assert result == 'uniform_low', (
            f"reid_50=0.049: observed '{result}', codified as 'uniform_low'"
        )

    def test_uniform_low_with_significant_tail(self):
        """Low median but high 99th → severe_tail (tail_ratio check fires first).

        Observed-and-codified: reid_50=0.01, reid_99=0.50 gives tail_ratio=50.
        The severe_tail check (tail_ratio > 10 AND reid_99 > 0.30) fires BEFORE
        the uniform_low branch (reid_50 < 0.05) is reached. So this is
        classified as 'severe_tail', not 'tail' or 'uniform_low'.
        """
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0.01, 'reid_95': 0.05, 'reid_99': 0.50, 'mean_risk': 0.03,
        })
        # tail_ratio = 0.50/0.01 = 50 > 10, reid_99=0.50 > 0.30 → severe_tail
        assert result == 'severe_tail', (
            f"Low median + high tail: observed '{result}', codified as 'severe_tail'"
        )

    def test_bimodal_threshold(self):
        """abs(mean - median) > 0.15 → bimodal."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        # Need: reid_50 not > 0.20, not in tail, not < 0.05, then |gap| > 0.15
        result = classify_risk_pattern({
            'reid_50': 0.10, 'reid_95': 0.15, 'reid_99': 0.20, 'mean_risk': 0.26,
        })
        # |0.26 - 0.10| = 0.16 > 0.15 → bimodal
        assert result == 'bimodal', (
            f"Bimodal threshold: observed '{result}', codified as 'bimodal'"
        )

    def test_bimodal_at_exactly_015_gap(self):
        """abs(mean - median) = 0.15 exactly: NOT bimodal (strict >)."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0.10, 'reid_95': 0.15, 'reid_99': 0.20, 'mean_risk': 0.25,
        })
        # |0.25 - 0.10| = 0.15 not > 0.15 → moderate
        assert result == 'moderate', (
            f"Gap=0.15 exactly: observed '{result}', codified as 'moderate'"
        )

    def test_widespread_vs_uniform_high(self):
        """reid_99 - reid_50 exactly 0.10: NOT uniform_high (strict <), → widespread."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0.30, 'reid_95': 0.35, 'reid_99': 0.40, 'mean_risk': 0.32,
        })
        # reid_50 > 0.20, reid_99 - reid_50 = 0.10, not < 0.10 → widespread
        assert result == 'widespread', (
            f"Spread=0.10 exactly: observed '{result}', codified as 'widespread'"
        )

    def test_severe_tail_boundary(self):
        """tail_ratio > 10 AND reid_99 > 0.30 → severe_tail."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        # tail_ratio = 0.31 / 0.02 = 15.5 > 10, reid_99 = 0.31 > 0.30
        result = classify_risk_pattern({
            'reid_50': 0.02, 'reid_95': 0.10, 'reid_99': 0.31, 'mean_risk': 0.05,
        })
        assert result == 'severe_tail'

    def test_not_severe_tail_low_ratio(self):
        """tail_ratio <= 10 blocks severe_tail even if reid_99 is high."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        # reid_50=0.05 exactly, not < 0.05 so not in uniform_low branch first
        # Wait — reid_50=0.05 not > 0.20 (falls through).
        # tail_ratio = 0.40/0.05 = 8 ≤ 10, so not severe_tail
        # reid_95=0.20 not > 0.30, reid_99=0.40 not > 0.50 with tail_ratio 8 > 5
        # → actually it IS tail via second check (reid_99 > 0.50 and tail_ratio > 5)
        # reid_99=0.40 NOT > 0.50, so not tail either
        # reid_50=0.05, not < 0.05 → not uniform_low
        # |mean - median| = |0.07-0.05| = 0.02, not > 0.15 → moderate
        result = classify_risk_pattern({
            'reid_50': 0.05, 'reid_95': 0.20, 'reid_99': 0.40, 'mean_risk': 0.07,
        })
        assert result == 'moderate', (
            f"Low tail_ratio: observed '{result}', codified as 'moderate'"
        )

    def test_zero_risk(self):
        """All zeros → uniform_low (reid_50=0 < 0.05 branch)."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({
            'reid_50': 0, 'reid_95': 0, 'reid_99': 0, 'mean_risk': 0,
        })
        assert result == 'uniform_low'

    def test_empty_dict(self):
        """Empty dict (defaults to 0) → uniform_low."""
        from sdc_engine.sdc.metrics.reid import classify_risk_pattern
        result = classify_risk_pattern({})
        assert result == 'uniform_low'


# ---------------------------------------------------------------------------
# Test 3 — var_priority label thresholds
# ---------------------------------------------------------------------------

class TestVarPriorityLabelThresholds:
    """Verify the 15%/8%/3% boundaries produce correct labels."""

    def _compute_labels(self, contributions):
        """Simulate _compute_var_priority's label assignment logic."""
        result = {}
        for qi, pct in contributions.items():
            if pct >= 15:
                result[qi] = 'HIGH'
            elif pct >= 8:
                result[qi] = 'MED-HIGH'
            elif pct >= 3:
                result[qi] = 'MODERATE'
            else:
                result[qi] = 'LOW'
        return result

    def test_boundary_at_15(self):
        """pct=15.0 → HIGH; pct=14.9 → MED-HIGH."""
        labels = self._compute_labels({'a': 15.0, 'b': 14.9})
        assert labels['a'] == 'HIGH'
        assert labels['b'] == 'MED-HIGH'

    def test_boundary_at_8(self):
        """pct=8.0 → MED-HIGH; pct=7.9 → MODERATE."""
        labels = self._compute_labels({'a': 8.0, 'b': 7.9})
        assert labels['a'] == 'MED-HIGH'
        assert labels['b'] == 'MODERATE'

    def test_boundary_at_3(self):
        """pct=3.0 → MODERATE; pct=2.9 → LOW."""
        labels = self._compute_labels({'a': 3.0, 'b': 2.9})
        assert labels['a'] == 'MODERATE'
        assert labels['b'] == 'LOW'

    def test_real_computation(self):
        """Run _compute_var_priority on real data, verify labels match thresholds."""
        from sdc_engine.sdc.protection_engine import _compute_var_priority
        from sdc_engine.sdc.metrics.reid import calculate_reid

        rng = np.random.default_rng(42)
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
            expected = self._compute_labels({qi: pct})[qi]
            assert label == expected, (
                f"var_priority['{qi}']: pct={pct}, expected label '{expected}', "
                f"got '{label}'"
            )


# ---------------------------------------------------------------------------
# Test 4 — build_data_features determinism
# ---------------------------------------------------------------------------

class TestBuildDataFeaturesDeterminism:
    """Two calls on identical input → identical output."""

    def _build_df(self):
        rng = np.random.default_rng(99)
        n = 500
        df = pd.DataFrame({
            'region': rng.choice(['N', 'S', 'E', 'W'], n),
            'edu': rng.choice(['hs', 'ba', 'ms', 'phd'], n),
            'age': rng.integers(20, 60, n),
        })
        return df, ['region', 'edu', 'age']

    def test_identical_output(self):
        from sdc_engine.sdc.protection_engine import build_data_features
        df, qis = self._build_df()

        f1 = build_data_features(df, qis)
        f2 = build_data_features(df, qis)

        # Compare all keys
        assert set(f1.keys()) == set(f2.keys()), (
            f"Key mismatch: only in f1={set(f1) - set(f2)}, "
            f"only in f2={set(f2) - set(f1)}"
        )

        for key in f1:
            # Skip internal objects that may differ by reference
            if key in ('_risk_assessment',):
                continue
            v1, v2 = f1[key], f2[key]
            # Handle floating point comparison
            if isinstance(v1, float) and isinstance(v2, float):
                assert abs(v1 - v2) < 1e-10, (
                    f"features['{key}']: {v1} != {v2}"
                )
            elif isinstance(v1, dict) and isinstance(v2, dict):
                assert v1 == v2, f"features['{key}']: {v1} != {v2}"
            else:
                assert v1 == v2, f"features['{key}']: {v1!r} != {v2!r}"

    def test_with_sensitive_columns(self):
        from sdc_engine.sdc.protection_engine import build_data_features
        df, qis = self._build_df()

        f1 = build_data_features(df, qis, sensitive_columns=['region'])
        f2 = build_data_features(df, qis, sensitive_columns=['region'])

        for key in f1:
            if key in ('_risk_assessment',):
                continue
            v1, v2 = f1[key], f2[key]
            if isinstance(v1, float):
                assert abs(v1 - v2) < 1e-10, f"features['{key}']: {v1} != {v2}"
            else:
                assert v1 == v2, f"features['{key}']: {v1!r} != {v2!r}"


# ---------------------------------------------------------------------------
# Test 5 — risk_drop_pct dual-path documentation
# ---------------------------------------------------------------------------

class TestRiskDropPctDualPath:
    """Codify the known architectural debt: two paths compute risk contribution.

    Path A (ReidentificationRisk): Normalized to 100% total. Surfaces in
    Configure page Variable Importance chart, Backward Elimination Curve,
    HTML report export. Uses 'relative contribution' semantics.

    Path B (_compute_var_priority in protection_engine.py): NOT normalized.
    Uses leave-one-out reid_95 drop. Internal only — feeds rules engine
    var_priority. Sum of contributions can exceed 100% or be much less.

    This test codifies the difference as observed-and-documented rather
    than a bug to fix (both paths serve different purposes).
    """

    def test_var_priority_not_normalized(self):
        """_compute_var_priority contributions don't sum to 100%."""
        from sdc_engine.sdc.protection_engine import _compute_var_priority
        from sdc_engine.sdc.metrics.reid import calculate_reid

        rng = np.random.default_rng(42)
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

        total_pct = sum(pct for _, (_, pct) in vp.items())

        # Codified: the sum is NOT constrained to 100%.
        # Leave-one-out contributions are independent measurements, not shares.
        # Total can be < 100% (when removing QIs individually doesn't account
        # for interaction effects) or > 100% (when QIs have overlapping contributions).
        assert total_pct != pytest.approx(100.0, abs=1.0), (
            f"var_priority contributions sum to {total_pct:.1f}% ≈ 100% — "
            f"this would indicate normalization was added (Path A/B merged). "
            f"If intentional, update this test."
        )

    def test_var_priority_contributions_nonnegative(self):
        """All contributions should be >= 0 (risk can't increase by removing a QI)."""
        from sdc_engine.sdc.protection_engine import _compute_var_priority
        from sdc_engine.sdc.metrics.reid import calculate_reid

        rng = np.random.default_rng(42)
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
            assert pct >= 0, (
                f"var_priority['{qi}'] contribution={pct}% is negative. "
                f"_compute_var_priority uses max(0, drop) so this should never happen."
            )


# ---------------------------------------------------------------------------
# Test 6 — classify_risk_concentration consistency
# ---------------------------------------------------------------------------

class TestRiskConcentrationConsistency:
    """classify_risk_concentration uses the same var_priority dict that
    _compute_var_priority produces. Verify the chain works end-to-end."""

    def test_end_to_end_consistency(self):
        """build_data_features → var_priority → risk_concentration is consistent."""
        from sdc_engine.sdc.protection_engine import build_data_features
        from sdc_engine.sdc.selection.features import classify_risk_concentration

        rng = np.random.default_rng(42)
        n = 600
        df = pd.DataFrame({
            'sex': rng.choice(['M', 'F'], n),
            'region': rng.choice(['N', 'S', 'E'], n),
            'job': rng.choice([f'j_{i}' for i in range(20)], n),
        })
        qis = ['sex', 'region', 'job']
        features = build_data_features(df, qis)

        vp = features['var_priority']
        rc_from_features = features['risk_concentration']

        if not vp:
            pytest.skip("var_priority not populated")

        # Recompute risk_concentration from var_priority directly
        rc_recomputed = classify_risk_concentration(vp)

        # Should be identical
        assert rc_from_features == rc_recomputed, (
            f"Mismatch: features['risk_concentration']={rc_from_features} "
            f"vs classify_risk_concentration(vp)={rc_recomputed}"
        )

    def test_dominated_threshold_at_40pct(self):
        """Top QI at exactly 40% → 'dominated'; at 39.9% → 'not_dominated'."""
        from sdc_engine.sdc.selection.features import classify_risk_concentration

        # Exactly 40%
        vp_40 = {'a': ('HIGH', 40.0), 'b': ('MODERATE', 10.0)}
        assert classify_risk_concentration(vp_40)['pattern'] == 'dominated'

        # Just below
        vp_39 = {'a': ('HIGH', 39.9), 'b': ('MODERATE', 10.0)}
        assert classify_risk_concentration(vp_39)['pattern'] == 'not_dominated'
