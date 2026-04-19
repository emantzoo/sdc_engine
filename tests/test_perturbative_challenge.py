"""
Tests for the perturbative challenge feature.

Tests _try_perturbative_challenge() — the post-structural-success PRAM
challenge in sdc_engine/sdc/protection_engine.py.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# Lightweight ProtectionResult stand-in (matches the real dataclass shape)
@dataclass
class FakeResult:
    method: str = ''
    success: bool = False
    protected_data: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    utility_score: Optional[float] = None
    qi_suppression_detail: Optional[Dict[str, float]] = None
    reid_before: Optional[Dict[str, float]] = None
    reid_after: Optional[Dict[str, float]] = None


def _make_data_features(*, n_categorical=3, n_continuous=1, reid_95=0.15,
                        categorical_vars=None):
    """Build a minimal data_features dict."""
    if categorical_vars is None:
        categorical_vars = [f'cat_{i}' for i in range(n_categorical)]
    return {
        'n_categorical': n_categorical,
        'n_continuous': n_continuous,
        'reid_95': reid_95,
        'categorical_vars': categorical_vars,
        'n_qis': n_categorical + n_continuous,
        'qi_cardinalities': {v: 5 for v in categorical_vars},
    }


def _make_structural_result(*, method='kANON', utility=0.75, suppression=0.10):
    """Build a fake structural result with suppression."""
    return FakeResult(
        method=method,
        success=True,
        utility_score=utility,
        qi_suppression_detail={'cat_0': suppression, 'cat_1': suppression * 0.5},
        metadata={
            'statistics': {'suppression_rate': suppression},
        },
        reid_after={'reid_95': 0.04, 'min_k': 5},
    )


def _make_pram_result(*, utility=0.92, reid_95=0.04):
    """Build a fake PRAM result (zero suppression, higher utility)."""
    return FakeResult(
        method='PRAM',
        success=True,
        utility_score=utility,
        qi_suppression_detail={},
        metadata={},
        reid_before={'reid_95': 0.15},
        reid_after={'reid_95': reid_95, 'min_k': 3},
    )


from sdc_engine.sdc.protection_engine import _try_perturbative_challenge


class TestPerturbativeChallengePositive:
    """Test 1: Categorical-dominant data, kANON suppresses 10%, PRAM wins."""

    def test_challenge_wins(self):
        structural = _make_structural_result(method='kANON', utility=0.75, suppression=0.10)
        pram = _make_pram_result(utility=0.92, reid_95=0.04)
        data_features = _make_data_features(n_categorical=3, n_continuous=1, reid_95=0.15)
        log_entries = []

        def mock_apply(method, qis, params, input_data=None):
            if method == 'PRAM':
                return pram
            return None

        result = _try_perturbative_challenge(
            structural_result=structural,
            structural_method='kANON',
            input_data=pd.DataFrame({'cat_0': [1, 2, 3], 'cat_1': [4, 5, 6]}),
            quasi_identifiers=['cat_0', 'cat_1', 'cat_2', 'cont_0'],
            data_features=data_features,
            apply_method_fn=mock_apply,
            risk_target_raw=0.05,
            log_entries=log_entries,
        )

        assert result is not None
        assert result is pram
        assert result.metadata is not None
        assert 'perturbative_challenge' in result.metadata
        info = result.metadata['perturbative_challenge']
        assert info['replaced_method'] == 'kANON'
        assert info['utility_gain'] > 0
        assert any('WON' in entry or 'challenge' in entry.lower() for entry in log_entries)


class TestPerturbativeChallengeLowCatRatio:
    """Test 2: Continuous-heavy data → no challenge attempted."""

    def test_skipped_low_cat_ratio(self):
        structural = _make_structural_result(method='kANON', utility=0.75, suppression=0.10)
        # 1 categorical, 4 continuous → ratio = 0.20, below 0.50 threshold
        data_features = _make_data_features(n_categorical=1, n_continuous=4, reid_95=0.15)
        log_entries = []

        mock_apply = MagicMock()  # Should never be called

        result = _try_perturbative_challenge(
            structural_result=structural,
            structural_method='kANON',
            input_data=pd.DataFrame({'x': [1]}),
            quasi_identifiers=['cat_0', 'cont_0', 'cont_1', 'cont_2', 'cont_3'],
            data_features=data_features,
            apply_method_fn=mock_apply,
            risk_target_raw=0.05,
            log_entries=log_entries,
        )

        assert result is None
        mock_apply.assert_not_called()


class TestPerturbativeChallengeLowSuppression:
    """Test 3: kANON succeeds with <1% suppression → no challenge."""

    def test_skipped_low_suppression(self):
        structural = _make_structural_result(method='kANON', utility=0.90, suppression=0.005)
        data_features = _make_data_features(n_categorical=3, n_continuous=1, reid_95=0.15)
        log_entries = []

        mock_apply = MagicMock()

        result = _try_perturbative_challenge(
            structural_result=structural,
            structural_method='kANON',
            input_data=pd.DataFrame({'x': [1]}),
            quasi_identifiers=['cat_0', 'cat_1', 'cat_2', 'cont_0'],
            data_features=data_features,
            apply_method_fn=mock_apply,
            risk_target_raw=0.05,
            log_entries=log_entries,
        )

        assert result is None
        mock_apply.assert_not_called()


class TestPerturbativeChallengePramMissesTarget:
    """Test 4: PRAM runs but ReID above target → structural kept."""

    def test_pram_misses_target(self):
        structural = _make_structural_result(method='LOCSUPR', utility=0.70, suppression=0.15)
        # PRAM returns with ReID above risk target
        pram = _make_pram_result(utility=0.90, reid_95=0.08)
        data_features = _make_data_features(n_categorical=3, n_continuous=1, reid_95=0.15)
        log_entries = []

        def mock_apply(method, qis, params, input_data=None):
            if method == 'PRAM':
                return pram
            return None

        result = _try_perturbative_challenge(
            structural_result=structural,
            structural_method='LOCSUPR',
            input_data=pd.DataFrame({'x': [1]}),
            quasi_identifiers=['cat_0', 'cat_1', 'cat_2', 'cont_0'],
            data_features=data_features,
            apply_method_fn=mock_apply,
            risk_target_raw=0.05,  # PRAM's 0.08 > 0.05 target
            log_entries=log_entries,
        )

        assert result is None
        assert any('missed target' in entry.lower() for entry in log_entries)


class TestPerturbativeChallengeInsufficientUtilityGain:
    """Test 5: PRAM utility only +1% (below min_utility_gain) → structural kept."""

    def test_insufficient_utility_gain(self):
        structural = _make_structural_result(method='kANON', utility=0.85, suppression=0.10)
        # PRAM utility is 0.86 — only +1%, below the 3% threshold
        pram = _make_pram_result(utility=0.86, reid_95=0.04)
        data_features = _make_data_features(n_categorical=3, n_continuous=1, reid_95=0.15)
        log_entries = []

        def mock_apply(method, qis, params, input_data=None):
            if method == 'PRAM':
                return pram
            return None

        result = _try_perturbative_challenge(
            structural_result=structural,
            structural_method='kANON',
            input_data=pd.DataFrame({'x': [1]}),
            quasi_identifiers=['cat_0', 'cat_1', 'cat_2', 'cont_0'],
            data_features=data_features,
            apply_method_fn=mock_apply,
            risk_target_raw=0.05,
            log_entries=log_entries,
        )

        assert result is None
        assert any('insufficient' in entry.lower() or 'gain' in entry.lower()
                    for entry in log_entries)
