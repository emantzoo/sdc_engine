"""
Dataset builders for rule selection known-case tests.

Each builder returns (DataFrame, expected_qis, expected_sensitive). The
builders are constructed so that the rule-engine features computed from
their output fall into well-defined ranges.

Builders use a fixed seed for determinism. Verified profiles annotated
in docstrings (2026-04-20).
"""
from typing import Tuple, List
import numpy as np
import pandas as pd


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ════════════════════════════════════════════════════════════════════════
# QR-series: high-risk rules
# ════════════════════════════════════════════════════════════════════════

def build_severe_tail_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers QR1_Severe_Tail_Risk → LOCSUPR.
    Verified: reid_50=0.019, reid_95=0.024, reid_99=1.000, pattern=severe_tail.
    """
    rng = _rng()
    n_common = 980
    n_outliers = 20
    df = pd.DataFrame({
        'age': np.concatenate([
            rng.choice([30, 35, 40], n_common),
            rng.integers(85, 100, n_outliers),
        ]),
        'zip': np.concatenate([
            rng.choice([10001, 10002, 10003], n_common),
            rng.integers(99990, 100000, n_outliers),
        ]),
        'sex': rng.choice(['M', 'F'], n_common + n_outliers),
    })
    return df, ['age', 'zip', 'sex'], []


def build_qr2_moderate_tail_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers QR2_Moderate_Tail_Risk → LOCSUPR.
    Verified: cat_ratio=0.33, reid_50=0.167, reid_95=0.333, pattern=tail.
    """
    rng = _rng()
    n = 300
    df = pd.DataFrame({
        'region': rng.choice([f'r_{i}' for i in range(5)], n),
        'age': rng.choice([20, 40, 60], n),
        'score': rng.choice([1, 3, 5, 7], n),
    })
    return df, ['region', 'age', 'score'], []


def build_qr2_heavy_tail_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers QR2_Heavy_Tail (kANON or LOCSUPR via suppression gate).
    Verified: cat_ratio=1.00, reid_50=0.200, reid_95=0.500, pattern=tail.
    """
    rng = _rng()
    n = 600
    df = pd.DataFrame({
        'a': rng.choice([f'a{i}' for i in range(5)], n),
        'b': rng.choice([f'b{i}' for i in range(5)], n),
        'c': rng.choice([f'c{i}' for i in range(6)], n),
    })
    return df, ['a', 'b', 'c'], []


def build_qr4_widespread_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers QR4_Widespread (high or moderate).
    Verified: reid_50=0.250, reid_95=1.000, pattern=widespread.
    """
    rng = _rng()
    n = 300
    df = pd.DataFrame({
        'a': rng.choice([f'a{i}' for i in range(4)], n),
        'b': rng.choice([f'b{i}' for i in range(5)], n),
        'c': rng.choice([f'c{i}' for i in range(5)], n),
    })
    return df, ['a', 'b', 'c'], []


def build_med1_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers MED1_Moderate_Structural → kANON.
    Verified: pattern=moderate, hrr=0.138 (>0.10 trigger), cat_ratio=0.33.

    MED1 fires via high_risk_rate > 0.10 (13.8% of records at risk >20%).
    4 regions × 5 ages × 4 scores = 80 combos at n=500 → some small cells.
    cat_ratio=0.33 blocks CAT1 (needs >=0.70).
    """
    rng = _rng()
    n = 500
    df = pd.DataFrame({
        'region': rng.choice(['N', 'S', 'E', 'W'], n),
        'age': rng.choice([20, 30, 40, 50, 60], n),
        'score': rng.choice([1, 2, 3, 4], n),
    })
    return df, ['region', 'age', 'score'], []


# ════════════════════════════════════════════════════════════════════════
# RC-series: risk concentration rules
# ════════════════════════════════════════════════════════════════════════

def build_dominated_risk_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Would trigger RC1 if var_priority were populated.

    RC rules require var_priority from backward elimination. The test
    injects var_priority manually to control concentration patterns.
    """
    rng = _rng()
    n = 800
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'race': rng.choice(['A', 'B', 'C'], n),
        'region': rng.choice(['N', 'S'], n),
        'job': rng.choice([f'j_{i}' for i in range(80)], n),
    })
    return df, ['sex', 'race', 'region', 'job'], []


# ════════════════════════════════════════════════════════════════════════
# CAT-series: categorical-aware rules
# ════════════════════════════════════════════════════════════════════════

def build_cat1_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers CAT1_Categorical_Dominant → PRAM.
    Verified: cat_ratio=1.00, reid_95=0.200, no dominance.
    """
    rng = _rng()
    n = 600
    df = pd.DataFrame({
        'region': rng.choice(['N', 'S', 'E', 'W'], n, p=[0.3, 0.25, 0.25, 0.2]),
        'edu': rng.choice(['hs', 'ba', 'ms', 'phd'], n, p=[0.35, 0.35, 0.2, 0.1]),
        'marital': rng.choice(['single', 'married', 'div'], n, p=[0.4, 0.4, 0.2]),
    })
    return df, ['region', 'edu', 'marital'], []


def build_cat1_blocked_by_dominance() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """CAT1 range but blocked by dominance guard (one category >= 80%).
    Verified: has_dominant=True, race max_freq=0.825, rule != CAT1.

    CAT1 conditions met (cat_ratio=1.00, reid_95 in [0.15, 0.40]) but
    _has_dominant_categories fires because 'race' has max freq ~83%.
    Falls through to QR1 (severe_tail from the dominance skew).
    Test verifies negative case: CAT1 does NOT fire.
    """
    rng = _rng()
    n = 800
    df = pd.DataFrame({
        'race': np.concatenate([['White'] * 660,
                                rng.choice(['Black', 'Asian', 'Hispanic'], 140)]),
        'region': rng.choice(['N', 'S', 'E', 'W'], n),
        'edu': rng.choice(['hs', 'ba', 'ms', 'phd'], n),
    })
    return df, ['race', 'region', 'edu'], []


def build_cat2_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers DYN_CAT_Pipeline (dynamic categorical pipeline).
    Verified: cat_ratio=0.67, reid_95=0.250, rule=DYN_CAT_Pipeline.

    DYN_CAT_Pipeline fires from pipeline_rules before CAT2 in rule_factories.
    Same condition: 0.50 < cat_ratio < 0.70, reid_95 > 0.15, n_cont >= 1.
    Strategy: 2 cat + 1 cont (cat_ratio=0.67), n=800 for reid_95 ~0.25.
    """
    rng = _rng()
    n = 800
    df = pd.DataFrame({
        'region': rng.choice(['N', 'S', 'E', 'W', 'NE', 'NW'], n),
        'edu': rng.choice(['hs', 'ba', 'ms', 'phd'], n),
        'age': rng.choice([20, 30, 40, 50, 60], n),
    })
    return df, ['region', 'edu', 'age'], []


# ════════════════════════════════════════════════════════════════════════
# LOW-series: low-risk rules
# ════════════════════════════════════════════════════════════════════════

def build_low1_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers LOW1_Categorical → PRAM.
    Verified: cat_ratio=1.00, reid_95=0.020.
    """
    rng = _rng()
    n = 2000
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'region': rng.choice(['N', 'S', 'E', 'W'], n),
        'age_band': rng.choice(['20-30', '30-40', '40-50', '50-60'], n),
    })
    return df, ['sex', 'region', 'age_band'], []


def build_low2_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers LOW2_Continuous_Noise → NOISE.
    Verified: cat_ratio=0.25, reid_95=0.010.
    """
    rng = _rng()
    n = 8000
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'age_group': rng.choice([20, 40, 60], n),
        'income_level': rng.choice([30, 50, 70], n),
        'score': rng.choice([1, 3, 5, 7], n),
    })
    return df, ['sex', 'age_group', 'income_level', 'score'], []


def build_low3_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers LOW3_Mixed → kANON.
    Verified: cat_ratio=0.50, reid_95=0.040.
    """
    rng = _rng()
    n = 5000
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'region': rng.choice(['N', 'S', 'E'], n),
        'age': rng.choice([20, 30, 40, 50, 60], n),
        'score': rng.choice([1, 2, 3, 4, 5], n),
    })
    return df, ['sex', 'region', 'age', 'score'], []


# ════════════════════════════════════════════════════════════════════════
# Structural rules: SR3, HR6
# ════════════════════════════════════════════════════════════════════════

def build_sr3_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers SR3_Near_Unique_Few_QIs → LOCSUPR.
    Verified: max_qi_uniqueness=1.0, n_qis=2, reid_95=1.0.

    Needs: n_qis <= 2, max_qi_uniqueness > 0.70, reid_95 > 0.20.
    One continuous column (income) with high cardinality (200 unique in 200 rows)
    gives max_qi_uniqueness=1.0. n >= 200 avoids HR6.
    """
    rng = _rng()
    n = 200
    df = pd.DataFrame({
        'income': rng.integers(20000, 100000, n),
        'sex': rng.choice(['M', 'F'], n),
    })
    return df, ['income', 'sex'], []


def build_hr6_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers HR6_Very_Small_Dataset → LOCSUPR.
    Verified: n=150, rule=HR6_Very_Small_Dataset.
    """
    rng = _rng()
    n = 150
    df = pd.DataFrame({
        'age_band': rng.choice(['20s', '30s', '40s', '50s', '60s'], n),
        'sex': rng.choice(['M', 'F'], n),
        'region': rng.choice(['N', 'S', 'E'], n),
    })
    return df, ['age_band', 'sex', 'region'], []


# ════════════════════════════════════════════════════════════════════════
# Uniqueness risk rules: HR1-HR5
# ════════════════════════════════════════════════════════════════════════

def build_extreme_uniqueness_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Dataset for HR1/HR3 uniqueness risk rules.

    HR rules require features['uniqueness_rate'] > threshold.
    Tests must inject uniqueness_rate manually into features.
    """
    rng = _rng()
    n = 300
    df = pd.DataFrame({
        'cat_a': rng.choice([f'a{i}' for i in range(20)], n),
        'cat_b': rng.choice([f'b{i}' for i in range(20)], n),
        'cat_c': rng.choice([f'c{i}' for i in range(15)], n),
    })
    return df, ['cat_a', 'cat_b', 'cat_c'], []


# ════════════════════════════════════════════════════════════════════════
# Context-aware rules: PUB1, SEC1, REG1
# ════════════════════════════════════════════════════════════════════════

def build_pub1_high_risk_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers PUB1_Public_Release_High_Risk under PUBLIC tier.
    Verified: reid_95=0.333, cat_ratio=0.33.

    Needs: access_tier='PUBLIC', reid_target_raw=0.01, reid_95 > 0.20.
    Uses qr2_moderate_tail shape (reid_95=0.333 > 0.20 threshold).
    """
    rng = _rng()
    n = 300
    df = pd.DataFrame({
        'region': rng.choice([f'r_{i}' for i in range(5)], n),
        'age': rng.choice([20, 40, 60], n),
        'score': rng.choice([1, 3, 5, 7], n),
    })
    return df, ['region', 'age', 'score'], []


def build_pub1_moderate_risk_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers PUB1_Public_Release_Moderate_Risk under PUBLIC tier.
    Verified: reid_95 in (0.05, 0.20].

    Uses low1 shape (cat_ratio=1.00, low risk) but with fewer rows → moderate reid.
    """
    rng = _rng()
    n = 500
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'region': rng.choice(['N', 'S', 'E', 'W', 'NE'], n),
        'age_band': rng.choice(['20-30', '30-40', '40-50', '50-60'], n),
    })
    return df, ['sex', 'region', 'age_band'], []


def build_sec1_categorical_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers SEC1_Secure_Categorical under SECURE tier.
    Verified: cat_ratio >= 0.60, reid_95 in [0.05, 0.25].

    Needs: access_tier='SECURE', utility_floor >= 0.90, reid_95 in [0.05, 0.25].
    Reuses cat1 builder's shape (all categorical, reid_95 ~0.20).
    """
    rng = _rng()
    n = 600
    df = pd.DataFrame({
        'region': rng.choice(['N', 'S', 'E', 'W'], n, p=[0.3, 0.25, 0.25, 0.2]),
        'edu': rng.choice(['hs', 'ba', 'ms', 'phd'], n, p=[0.35, 0.35, 0.2, 0.1]),
        'marital': rng.choice(['single', 'married', 'div'], n, p=[0.4, 0.4, 0.2]),
    })
    return df, ['region', 'edu', 'marital'], []


def build_sec1_continuous_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers SEC1_Secure_Continuous under SECURE tier.
    Verified: cat_ratio < 0.60, n_continuous > 0, reid_95 in [0.05, 0.25].

    Uses low2 shape but with fewer rows to push reid_95 into range.
    """
    rng = _rng()
    n = 500
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'age_group': rng.choice([20, 40, 60], n),
        'income_level': rng.choice([30, 50, 70], n),
        'score': rng.choice([1, 3, 5, 7], n),
    })
    return df, ['sex', 'age_group', 'income_level', 'score'], []


def build_reg1_high_risk_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers REG1_Regulatory_High_Risk under PUBLIC tier with target=0.03.
    Verified: reid_95 > 0.15.

    Reuses cat1 builder's shape (reid_95 ~0.20).
    """
    rng = _rng()
    n = 600
    df = pd.DataFrame({
        'region': rng.choice(['N', 'S', 'E', 'W'], n, p=[0.3, 0.25, 0.25, 0.2]),
        'edu': rng.choice(['hs', 'ba', 'ms', 'phd'], n, p=[0.35, 0.35, 0.2, 0.1]),
        'marital': rng.choice(['single', 'married', 'div'], n, p=[0.4, 0.4, 0.2]),
    })
    return df, ['region', 'edu', 'marital'], []


def build_reg1_moderate_risk_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Triggers REG1_Regulatory_Moderate_Risk under PUBLIC tier with target=0.03.
    Verified: reid_95 in (0.03, 0.15].

    Needs moderate reid_95 so MED1/QR rules don't preempt — but REG1
    fires FIRST in the chain, so it preempts everything.
    """
    rng = _rng()
    n = 500
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'region': rng.choice(['N', 'S', 'E', 'W', 'NE'], n),
        'age_band': rng.choice(['20-30', '30-40', '40-50', '50-60'], n),
    })
    return df, ['sex', 'region', 'age_band'], []


# ════════════════════════════════════════════════════════════════════════
# Dormant-rule activation (Spec 07): RC rules via organic var_priority
# ════════════════════════════════════════════════════════════════════════

def build_small_dominated_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Small dataset where one QI clearly dominates risk contribution.

    Designed for Spec 07 integration tests: n < max_n_records,
    n_qis <= max_n_qis, and one QI's cardinality is 10x+ the others
    so it dominates backward-elimination risk.

    Verified: var_priority populated, risk_concentration
    pattern='dominated', top_qi='job', top_pct > 40%.
    """
    rng = _rng()
    n = 600  # well under max_n_records=10000
    df = pd.DataFrame({
        'sex': rng.choice(['M', 'F'], n),
        'region': rng.choice(['N', 'S', 'E'], n),
        'job': rng.choice([f'j_{i}' for i in range(60)], n),  # dominant
    })
    return df, ['sex', 'region', 'job'], []


# ════════════════════════════════════════════════════════════════════════
# Edge cases
# ════════════════════════════════════════════════════════════════════════

def build_default_fallback_dataset() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Simple dataset — should produce a valid result, not crash."""
    rng = _rng()
    df = pd.DataFrame({
        'a': rng.choice(['x', 'y'], 500),
        'b': rng.choice(['p', 'q', 'r'], 500),
    })
    return df, ['a', 'b'], []
