"""
SDC Method Selection Helper
===========================

Analyzes your data and recommends the most appropriate SDC method(s).

Usage:
    # Interactive mode
    python select_method.py data.csv

    # Programmatic mode
    from select_method import recommend_method
    recommendation = recommend_method(data, goal='k-anonymity')

Author: SDC Methods Implementation
Date: December 2025
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Optional, Tuple, Union

from sdc_engine.sdc.sdc_utils import (
    analyze_data,
    calculate_disclosure_risk,
    calculate_reid,
    METHOD_INFO,
)

# Import active SDC methods (microdata only: kANON, PRAM, NOISE, LOCSUPR)
from sdc_engine.sdc import (
    apply_kanon, apply_pram, apply_noise, apply_locsupr
)

# Import QI preprocessing
try:
    from sdc_engine.sdc.preprocessing import QIHandler, preprocess_for_anonymization, TIER_CONSTRAINTS
    HAS_PREPROCESSING = True
except ImportError:
    HAS_PREPROCESSING = False

# Import utility metrics
try:
    from sdc_engine.sdc.metrics import calculate_utility_metrics
    HAS_UTILITY_METRICS = True
except ImportError:
    HAS_UTILITY_METRICS = False


# =============================================================================
# ReID-BASED METHOD SELECTION (Data-Driven)
# =============================================================================

def extract_data_features_with_reid(
    data: pd.DataFrame,
    analysis: Dict,
    quasi_identifiers: Optional[List[str]],
    structural_risk: float = 0.0,
    risk_metric: Optional[str] = None,
    risk_target_raw: Optional[float] = None,
) -> Dict:
    """
    Extract data features including risk metrics for method selection.

    Parameters
    ----------
    structural_risk : float
        QI-scoped backward elimination risk (0-1).  When high, signals
        large combination space driven by QI structure rather than
        individual-level outliers.
    risk_metric : str, optional
        'reid95', 'k_anonymity', or 'uniqueness'. Default 'reid95'.
    risk_target_raw : float, optional
        Raw target for the chosen metric.
    """
    if quasi_identifiers is None:
        quasi_identifiers = analysis.get('quasi_identifiers', [])

    features = {
        'n_records': len(data),
        'n_columns': len(data.columns),
        'data_type': analysis.get('data_type', 'microdata'),
        'n_continuous': len(analysis.get('continuous_variables', [])),
        'n_categorical': len(analysis.get('categorical_variables', [])),
        'n_qis': len(quasi_identifiers),
        'continuous_vars': analysis.get('continuous_variables', []),
        'categorical_vars': analysis.get('categorical_variables', []),
        'quasi_identifiers': quasi_identifiers,
        'uniqueness_rate': analysis.get('uniqueness_rate', 0),
        'risk_level': analysis.get('risk_level', 'medium'),
        'high_cardinality_qis': [],
        'low_cardinality_qis': [],
        'has_sensitive_attributes': bool(analysis.get('sensitive_columns')),
        'sensitive_columns': analysis.get('sensitive_columns', {}),
        'small_cells_rate': 0,
        'has_reid': False,
        'has_outliers': False,
        'skewed_columns': [],
        'structural_risk': structural_risk,
    }

    # Calculate cardinality for QIs
    if quasi_identifiers:
        for qi in quasi_identifiers:
            if qi in data.columns:
                card = data[qi].nunique() / len(data)
                if card > 0.5:
                    features['high_cardinality_qis'].append(qi)
                elif card < 0.1:
                    features['low_cardinality_qis'].append(qi)

    # Calculate small cells rate for tabular data
    if features['data_type'] == 'tabular':
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.size > 0:
                small_cells = (numeric_data < 3).sum().sum()
                features['small_cells_rate'] = small_cells / numeric_data.size
        except Exception:
            pass

    # Detect outliers
    for col in features['continuous_vars']:
        if col in data.columns:
            try:
                Q1, Q3 = data[col].quantile(0.25), data[col].quantile(0.75)
                IQR = Q3 - Q1
                if ((data[col] < Q1 - 3 * IQR) | (data[col] > Q3 + 3 * IQR)).any():
                    features['has_outliers'] = True
                    break
            except Exception:
                pass

    # Detect skewed categorical distributions
    for col in features['categorical_vars']:
        if col in data.columns:
            try:
                value_counts = data[col].value_counts()
                if len(value_counts) > 0 and value_counts.iloc[0] / len(data) > 0.7:
                    features['skewed_columns'].append(col)
            except Exception:
                pass

    # Calculate risk metric if microdata with QIs
    if quasi_identifiers and features['data_type'] == 'microdata':
        try:
            from sdc_engine.sdc.metrics.risk_metric import (
                RiskMetricType, compute_risk, risk_to_reid_compat,
            )
            _mt_map = {
                'reid95': RiskMetricType.REID95,
                'k_anonymity': RiskMetricType.K_ANONYMITY,
                'uniqueness': RiskMetricType.UNIQUENESS,
            }
            mt = _mt_map.get(risk_metric or 'reid95', RiskMetricType.REID95)
            assessment = compute_risk(data, quasi_identifiers, mt, risk_target_raw)
            reid_metrics = risk_to_reid_compat(assessment)

            features['reid'] = reid_metrics
            features['reid_50'] = reid_metrics.get('reid_50', 0)
            features['reid_90'] = reid_metrics.get('reid_90', 0)
            features['reid_95'] = reid_metrics.get('reid_95', 0)
            features['reid_99'] = reid_metrics.get('reid_99', 0)
            features['max_risk'] = reid_metrics.get('max_risk', 0)
            features['mean_risk'] = reid_metrics.get('mean_risk', 0)
            features['high_risk_count'] = reid_metrics.get('high_risk_count', 0)
            features['high_risk_rate'] = reid_metrics.get('high_risk_rate', 0)
            features['risk_pattern'] = _classify_risk_pattern(reid_metrics)
            features['has_reid'] = True
            features['_risk_metric_type'] = mt.value
            features['_risk_assessment'] = assessment
        except Exception as e:
            features['has_reid'] = False
            features['reid_error'] = str(e)

    return features


def _classify_risk_pattern(reid_metrics: Dict) -> str:
    """Classify the risk distribution pattern."""
    reid_50 = reid_metrics.get('reid_50', 0)
    reid_95 = reid_metrics.get('reid_95', 0)
    reid_99 = reid_metrics.get('reid_99', 0)
    mean_risk = reid_metrics.get('mean_risk', 0)

    if reid_50 > 0.20:
        return 'uniform_high' if reid_99 - reid_50 < 0.10 else 'widespread'
    elif reid_50 < 0.05:
        if reid_95 > 0.30:
            return 'severe_tail' if reid_99 > 0.50 else 'tail'
        return 'uniform_low'
    elif abs(mean_risk - reid_50) > 0.15:
        return 'bimodal'
    return 'moderate'


def _data_structure_rules(features: Dict) -> Dict:
    """Rules based on data structure (tabular only - microdata handled by ReID/distribution rules)."""
    data_type = features['data_type']

    # Tabular-specific methods (THRES, TABSUPR, CROUND, CTA) are not available.
    # Only microdata methods (kANON, PRAM, NOISE, LOCSUPR) are supported.
    if data_type == 'tabular':
        import logging
        logging.getLogger(__name__).info(
            "Tabular data detected. Tabular-specific methods not available "
            "— microdata methods will be used."
        )
        return {'applies': False}

    return {'applies': False}


def _low_risk_structure_rules(features: Dict) -> Dict:
    """Low-Risk Structure Rules (LR1-LR4).

    Applied ONLY when ReID_95 <= 5% (data is already low risk).
    These rules select perturbation methods for UTILITY preservation.

    IMPORTANT: Perturbation methods (NOISE, PRAM) do NOT reduce ReID.
    Only use them when:
    1. ReID is already LOW (< 5%)
    2. QIs don't have high cardinality (NOISE on high-card continuous QIs makes it worse)
    3. Utility preservation is the goal
    For moderate/high risk, always use structural methods (kANON, LOCSUPR).
    """
    if features['data_type'] != 'microdata':
        return {'applies': False}

    # Get ReID_95 if available
    reid_95 = features.get('reid_95', 1.0)
    qis = features.get('quasi_identifiers', [])
    high_card_count = features.get('high_cardinality_count', 0)
    uniqueness = features.get('uniqueness_rate', 0)

    # CRITICAL: Only use perturbation methods if ReID is already LOW (< 5%)
    # AND there's no high cardinality issue
    if reid_95 > 0.05 or high_card_count > 0 or uniqueness > 0.05:
        # Use kANON for structural protection
        reason = []
        if reid_95 > 0.05:
            reason.append(f"ReID_95={reid_95:.1%} > 5%")
        if high_card_count > 0:
            reason.append(f"{high_card_count} high-cardinality QIs")
        if uniqueness > 0.05:
            reason.append(f"uniqueness={uniqueness:.1%}")
        return {'applies': True, 'rule': 'LR_Structural_Required', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
                'reason': f"{', '.join(reason)} - structural protection required",
                'confidence': 'HIGH', 'priority': 'REQUIRED'}

    # Check if QIs contain continuous variables
    # NOISE applied to continuous QIs INCREASES ReID (creates new unique values)
    # So we must use kANON even for low ReID if QIs are continuous
    continuous_qis = [q for q in qis if q in features.get('continuous_vars', [])]
    if continuous_qis:
        # QIs include continuous variables - must use kANON to bin/generalize
        return {'applies': True, 'rule': 'LR_Continuous_QI', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 3, 'strategy': 'generalization'},
                'reason': f"Continuous QIs {continuous_qis} require k-anonymity (perturbation increases uniqueness)",
                'confidence': 'HIGH', 'priority': 'REQUIRED'}

    # LR3: ReID is low AND QIs are all categorical - PRAM for utility
    if features['n_categorical'] > 0 and features['n_continuous'] == 0:
        return {'applies': True, 'rule': 'LR3_Categorical_Only', 'method': 'PRAM',
                'parameters': {'variables': features['categorical_vars'][:5], 'p_change': 0.2},
                'reason': f"Low risk (ReID_95={reid_95:.1%}) + categorical-only - PRAM for utility preservation",
                'confidence': 'HIGH', 'priority': 'RECOMMENDED'}

    return {'applies': False}


# Alias for backward compatibility
_microdata_structure_rules = _low_risk_structure_rules


def _reid_risk_rules(features: Dict) -> Dict:
    """ReID-based risk rules - uses risk distribution patterns.

    Structural Risk (SR) modifies method selection:
    - High SR + moderate ReID: large combination space with common patterns
      → favour kANON with generalization strategy (reduce combinations)
    - Low SR + high ReID: few QIs with unlucky distributions
      → targeted suppression (LOCSUPR) may be more efficient
    """
    if not features.get('has_reid'):
        return _uniqueness_risk_rules(features)

    reid_50, reid_95, reid_99 = features['reid_50'], features['reid_95'], features['reid_99']
    risk_pattern = features['risk_pattern']
    high_risk_rate = features['high_risk_rate']
    qis = features['quasi_identifiers']
    sr = features.get('structural_risk', 0.0)

    # SR-aware overrides: when SR is very high the combination space is the
    # problem, so generalization-based kANON is preferred over suppression.
    if sr > 0.50 and reid_95 > 0.05:
        # High structural risk — prefer generalization to shrink combo space
        k = 10 if sr > 0.70 else 7
        strategy = 'generalization'
        return {
            'applies': True,
            'rule': 'SR1_High_Structural_Risk',
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': strategy},
            'reason': (f"High structural risk ({sr:.0%}) with ReID_95={reid_95:.1%} — "
                       f"generalization-based k-anonymity to reduce combination space"),
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'alternatives': ['LOCSUPR'],
        }

    # Low SR + high ReID: targeted suppression is more efficient than
    # heavy generalization (few QIs, problem is individual-level outliers)
    if sr < 0.20 and reid_95 > 0.30 and risk_pattern in ('tail', 'severe_tail'):
        return {
            'applies': True,
            'rule': 'SR2_Low_Structural_Tail_Risk',
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 5},
            'reason': (f"Low structural risk ({sr:.0%}) with tail ReID_95={reid_95:.1%} — "
                       f"targeted suppression preferred over heavy generalization"),
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'alternatives': ['kANON'],
        }

    if risk_pattern == 'severe_tail':
        return {'applies': True, 'rule': 'QR1_Severe_Tail_Risk', 'method': 'LOCSUPR',
                'parameters': {'quasi_identifiers': qis, 'k': 5},
                'reason': f"Severe tail: ReID_50={reid_50:.1%}, ReID_99={reid_99:.1%} - few records dominate risk",
                'confidence': 'HIGH', 'priority': 'REQUIRED', 'alternatives': ['kANON']}

    # QR2: Tail risk - high ReID_95 needs structural protection
    if risk_pattern == 'tail' or (reid_95 > 0.30 and reid_50 < 0.15):
        if reid_95 > 0.40:
            return {'applies': True, 'rule': 'QR2_High_Tail_Risk', 'method': 'kANON',
                    'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'},
                    'reason': f"High tail risk: ReID_95={reid_95:.1%} - k-anonymity for structural protection",
                    'confidence': 'HIGH', 'priority': 'REQUIRED', 'alternatives': ['LOCSUPR']}
        return {'applies': True, 'rule': 'QR2_Moderate_Tail_Risk', 'method': 'LOCSUPR',
                'parameters': {'quasi_identifiers': qis, 'k': 3},
                'reason': f"Tail risk: ReID_50={reid_50:.1%}, ReID_95={reid_95:.1%} - targeted suppression",
                'confidence': 'HIGH', 'priority': 'REQUIRED', 'alternatives': ['kANON']}

    if risk_pattern == 'uniform_high':
        return {'applies': True, 'rule': 'QR3_Uniform_High_Risk', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 10, 'strategy': 'generalization'},
                'reason': f"Uniform high risk: ReID_50={reid_50:.1%} - widespread protection needed",
                'confidence': 'HIGH', 'priority': 'REQUIRED'}

    # QR4: Widespread risk - need structural protection (PRAM/NOISE don't reduce ReID)
    if risk_pattern == 'widespread' and reid_50 > 0.15:
        if reid_95 > 0.50:
            return {'applies': True, 'rule': 'QR4_Widespread_High', 'method': 'kANON',
                    'parameters': {'quasi_identifiers': qis, 'k': 10, 'strategy': 'generalization'},
                    'reason': f"Widespread high risk: ReID_50={reid_50:.1%}, ReID_95={reid_95:.1%} - aggressive k-anonymity",
                    'confidence': 'HIGH', 'priority': 'REQUIRED', 'alternatives': ['LOCSUPR']}
        return {'applies': True, 'rule': 'QR4_Widespread_Moderate', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'},
                'reason': f"Widespread: ReID_50={reid_50:.1%} - k-anonymity for structural protection",
                'confidence': 'HIGH', 'priority': 'REQUIRED', 'alternatives': ['LOCSUPR', 'PRAM']}

    if reid_95 > 0.20 and reid_50 < 0.10:
        return {'applies': True, 'rule': 'QR5_High_95th_Percentile', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'},
                'reason': f"High tail: ReID_95={reid_95:.1%} - k-anonymity with hybrid strategy",
                'confidence': 'HIGH', 'priority': 'REQUIRED'}

    if risk_pattern == 'bimodal':
        return {'applies': True, 'rule': 'QR6_Bimodal_Risk', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
                'reason': "Bimodal distribution - k-anonymity handles both groups",
                'confidence': 'MEDIUM', 'priority': 'RECOMMENDED'}

    # QR7: Many high-risk records - need structural protection
    if high_risk_rate > 0.10:
        return {'applies': True, 'rule': 'QR7_Many_High_Risk', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
                'reason': f"{high_risk_rate:.1%} at risk >20% - k-anonymity for structural protection",
                'confidence': 'MEDIUM', 'priority': 'RECOMMENDED', 'alternatives': ['LOCSUPR']}

    # QR8: Moderate-to-low ReID (5-20%) - still use kANON for structural protection
    # k-value selection:
    #   - ReID_95 > 10%: k=5 (higher risk needs more records per group)
    #   - ReID_95 5-10%: k=3 (lower risk, smaller groups sufficient)
    if 0.05 < reid_95 <= 0.20:
        k = 5 if reid_95 > 0.10 else 3
        return {'applies': True, 'rule': 'QR8_Moderate_Risk', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': 'generalization'},
                'reason': f"Moderate risk: ReID_95={reid_95:.1%} - kANON(k={k}) for structural protection",
                'confidence': 'MEDIUM', 'priority': 'RECOMMENDED'}

    # ReID_95 <= 5% - data is already low risk, perturbation methods acceptable
    # Let _microdata_structure_rules handle this case
    return {'applies': False}


def _uniqueness_risk_rules(features: Dict) -> Dict:
    """Fallback risk rules when ReID unavailable."""
    uniqueness = features['uniqueness_rate']
    n_qis, n_records = features['n_qis'], features['n_records']
    qis = features['quasi_identifiers']

    if uniqueness > 0.20:
        return {'applies': True, 'rule': 'HR1_Extreme_Uniqueness', 'method': 'LOCSUPR',
                'parameters': {'quasi_identifiers': qis, 'k': 5},
                'reason': f"Extreme uniqueness ({uniqueness:.1%})", 'confidence': 'HIGH', 'priority': 'REQUIRED'}

    if uniqueness > 0.10:
        return {'applies': True, 'rule': 'HR2_Very_High_Uniqueness', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'},
                'reason': f"Very high uniqueness ({uniqueness:.1%})", 'confidence': 'HIGH', 'priority': 'REQUIRED'}

    if uniqueness > 0.05 and n_qis >= 2:
        return {'applies': True, 'rule': 'HR3_High_Uniqueness_QIs', 'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'generalization'},
                'reason': f"Uniqueness {uniqueness:.1%} with {n_qis} QIs", 'confidence': 'HIGH', 'priority': 'REQUIRED'}

    if n_records < 100:
        return {'applies': True, 'rule': 'HR4_Very_Small_Dataset', 'method': 'PRAM',
                'parameters': {'variables': features['categorical_vars'][:5], 'p_change': 0.3},
                'reason': f"Small dataset ({n_records} records)", 'confidence': 'MEDIUM', 'priority': 'REQUIRED'}

    if 100 <= n_records < 500 and uniqueness > 0.03:
        if features['n_continuous'] > 0:
            return {'applies': True, 'rule': 'HR5_Small_Dataset', 'method': 'NOISE',
                    'parameters': {'variables': features['continuous_vars'], 'magnitude': 0.15},
                    'reason': f"Small dataset with {uniqueness:.1%} uniqueness", 'confidence': 'MEDIUM', 'priority': 'REQUIRED'}
        return {'applies': True, 'rule': 'HR5_Small_Dataset', 'method': 'PRAM',
                'parameters': {'variables': features['categorical_vars'][:5], 'p_change': 0.25},
                'reason': f"Small dataset with {uniqueness:.1%} uniqueness", 'confidence': 'MEDIUM', 'priority': 'REQUIRED'}

    return {'applies': False}


def _distribution_rules(features: Dict) -> Dict:
    """Rules based on distribution characteristics."""
    if features['has_outliers'] and features['n_continuous'] > 0:
        return {'applies': True, 'rule': 'DP1_Outliers', 'method': 'NOISE',
                'parameters': {'variables': features['continuous_vars'], 'magnitude': 0.20},
                'reason': "Outliers present - noise addition preserves distributions",
                'confidence': 'HIGH', 'priority': 'RECOMMENDED'}

    if len(features['skewed_columns']) >= 2:
        return {'applies': True, 'rule': 'DP2_Skewed', 'method': 'PRAM',
                'parameters': {'variables': features['skewed_columns'][:5], 'p_change': 0.2},
                'reason': f"{len(features['skewed_columns'])} skewed columns", 'confidence': 'MEDIUM', 'priority': 'RECOMMENDED'}

    if features['has_sensitive_attributes'] and features['n_qis'] >= 2:
        return {'applies': True, 'rule': 'DP3_Sensitive', 'method': 'kANON',
                'parameters': {'quasi_identifiers': features['quasi_identifiers'], 'k': 5, 'strategy': 'generalization'},
                'reason': "Sensitive attributes present", 'confidence': 'MEDIUM', 'priority': 'RECOMMENDED'}

    return {'applies': False}


def _default_rules(features: Dict) -> Dict:
    """Default fallback rules."""
    if features['data_type'] == 'microdata' and features['n_qis'] >= 2:
        return {'applies': True, 'rule': 'DEFAULT_Microdata_QIs', 'method': 'kANON',
                'parameters': {'quasi_identifiers': features['quasi_identifiers'], 'k': 3, 'strategy': 'generalization'},
                'reason': "Default: k-anonymity for microdata with QIs", 'confidence': 'LOW', 'priority': 'DEFAULT'}

    if features['n_categorical'] > features['n_continuous']:
        return {'applies': True, 'rule': 'DEFAULT_Categorical', 'method': 'PRAM',
                'parameters': {'variables': features['categorical_vars'][:5], 'p_change': 0.2},
                'reason': "Default: mostly categorical", 'confidence': 'LOW', 'priority': 'DEFAULT'}

    if features['n_continuous'] > features['n_categorical']:
        return {'applies': True, 'rule': 'DEFAULT_Continuous', 'method': 'NOISE',
                'parameters': {'variables': features['continuous_vars'], 'magnitude': 0.15},
                'reason': "Default: mostly continuous", 'confidence': 'LOW', 'priority': 'DEFAULT'}

    return {'applies': True, 'rule': 'DEFAULT_Fallback', 'method': 'PRAM',
            'parameters': {'p_change': 0.2}, 'reason': "Fallback", 'confidence': 'VERY_LOW', 'priority': 'FALLBACK'}


def _pipeline_rules(features: Dict) -> Dict:
    """
    Check if data requires multi-method pipeline.
    Pure data-driven - triggers when single method demonstrably insufficient.
    """
    n_continuous = features['n_continuous']
    n_categorical = features['n_categorical']
    has_reid = features.get('has_reid', False)
    qis = features['quasi_identifiers']

    # P1: Mixed variables with high risk in both types
    if n_continuous >= 2 and n_categorical >= 2:
        high_card_cats = len(features.get('high_cardinality_qis', []))
        has_outliers = features.get('has_outliers', False)
        uniqueness = features.get('uniqueness_rate', 0)

        if (high_card_cats >= 2 or uniqueness > 0.10) and has_outliers:
            return {
                'applies': True,
                'rule': 'P1_Mixed_Variables_Dual_Risk',
                'use_pipeline': True,
                'pipeline': ['NOISE', 'kANON'],
                'parameters': {
                    'NOISE': {'variables': features['continuous_vars'], 'magnitude': 0.15},
                    'kANON': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'generalization'}
                },
                'reason': f"Mixed data: {n_continuous} continuous with outliers + {n_categorical} categorical with high cardinality",
                'confidence': 'HIGH',
                'priority': 'REQUIRED'
            }

    # P2: ReID shows dual pattern (widespread + extreme tail)
    if has_reid:
        reid_50, reid_99 = features['reid_50'], features['reid_99']
        high_risk_rate = features['high_risk_rate']

        if reid_50 > 0.15 and reid_99 > 0.70 and (reid_99 - reid_50) > 0.40:
            if n_categorical > n_continuous:
                pipeline = ['PRAM', 'LOCSUPR']
                params = {
                    'PRAM': {'variables': features['categorical_vars'][:5], 'p_change': 0.25},
                    'LOCSUPR': {'quasi_identifiers': qis, 'k': 3}
                }
            else:
                pipeline = ['NOISE', 'LOCSUPR']
                params = {
                    'NOISE': {'variables': features['continuous_vars'], 'magnitude': 0.20},
                    'LOCSUPR': {'quasi_identifiers': qis, 'k': 3}
                }
            return {
                'applies': True,
                'rule': 'P2_Dual_Risk_Widespread_Plus_Tail',
                'use_pipeline': True,
                'pipeline': pipeline,
                'parameters': params,
                'reason': f"Dual risk: ReID_50={reid_50:.1%} (widespread) + ReID_99={reid_99:.1%} (extreme tail)",
                'confidence': 'HIGH',
                'priority': 'REQUIRED'
            }

        # P2b: Moderate overall + significant high-risk subgroup
        if 0.10 < features['reid_95'] < 0.25 and high_risk_rate > 0.15:
            return {
                'applies': True,
                'rule': 'P2_Moderate_With_High_Risk_Subgroup',
                'use_pipeline': True,
                'pipeline': ['kANON', 'LOCSUPR'],
                'parameters': {
                    'kANON': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
                    'LOCSUPR': {'quasi_identifiers': qis, 'k': 3}
                },
                'reason': f"Moderate risk (ReID_95={features['reid_95']:.1%}) but {high_risk_rate:.1%} at extreme risk",
                'confidence': 'HIGH',
                'priority': 'RECOMMENDED'
            }

    # P3: Multiple high-cardinality QIs
    high_card_qis = features.get('high_cardinality_qis', [])
    if len(high_card_qis) >= 3:
        return {
            'applies': True,
            'rule': 'P3_Multiple_High_Cardinality_QIs',
            'use_pipeline': True,
            'pipeline': ['kANON', 'LOCSUPR'],
            'parameters': {
                'kANON': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'generalization'},
                'LOCSUPR': {'quasi_identifiers': qis, 'k': 3}
            },
            'reason': f"{len(high_card_qis)} high-cardinality QIs create vast combination space",
            'confidence': 'HIGH',
            'priority': 'REQUIRED'
        }

    # P4: Skewed + rare values + sensitive attributes
    skewed_count = len(features.get('skewed_columns', []))
    has_sensitive = features.get('has_sensitive_attributes', False)
    if skewed_count >= 2 and has_sensitive and features.get('n_qis', 0) >= 2:
        return {
            'applies': True,
            'rule': 'P4_Skewed_Rare_Sensitive',
            'use_pipeline': True,
            'pipeline': ['kANON', 'PRAM'],
            'parameters': {
                'kANON': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'generalization'},
                'PRAM': {'variables': features['skewed_columns'][:5], 'p_change': 0.2}
            },
            'reason': f"{skewed_count} skewed distributions + sensitive attributes",
            'confidence': 'HIGH',
            'priority': 'RECOMMENDED'
        }

    # P5: Small dataset with mixed variables
    n_records = features.get('n_records', 0)
    uniqueness = features.get('uniqueness_rate', 0)
    if n_records < 500 and uniqueness > 0.15 and n_continuous >= 2 and n_categorical >= 2:
        return {
            'applies': True,
            'rule': 'P5_Small_Dataset_Mixed_Risks',
            'use_pipeline': True,
            'pipeline': ['NOISE', 'PRAM'],
            'parameters': {
                'NOISE': {'variables': features['continuous_vars'], 'magnitude': 0.15},
                'PRAM': {'variables': features['categorical_vars'][:5], 'p_change': 0.3}
            },
            'reason': f"Small dataset ({n_records}) with {uniqueness:.1%} uniqueness and mixed types",
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED'
        }

    # P6: Outliers + high categorical cardinality
    if features.get('has_outliers') and len(high_card_qis) >= 2:
        return {
            'applies': True,
            'rule': 'P6_Outliers_Plus_High_Cardinality',
            'use_pipeline': True,
            'pipeline': ['NOISE', 'kANON'],
            'parameters': {
                'NOISE': {'variables': features['continuous_vars'], 'magnitude': 0.20},
                'kANON': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'}
            },
            'reason': f"Outliers in continuous + {len(high_card_qis)} high-cardinality categorical QIs",
            'confidence': 'HIGH',
            'priority': 'RECOMMENDED'
        }

    return {'applies': False}


def select_method_by_features(*args, **kwargs):
    """Removed in favor of sdc_engine.sdc.selection.select_method_by_features.

    This legacy stub raises ImportError to catch any accidental imports.
    If you reached this error, update your import:

        # Old:
        from sdc_engine.sdc.select_method import select_method_by_features

        # New:
        from sdc_engine.sdc.selection import select_method_by_features
    """
    raise ImportError(
        "select_method_by_features has moved to sdc_engine.sdc.selection. "
        "Update your import."
    )


# =============================================================================
# METHOD RECOMMENDATION
# =============================================================================

def recommend_method(
    data: pd.DataFrame,
    goal: Optional[str] = None,
    quasi_identifiers: Optional[List[str]] = None,
    preserve_utility: Optional[List[str]] = None,
    allow_suppression: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Recommend SDC method(s) based on data analysis and requirements.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    goal : str, optional
        Privacy goal: 'k-anonymity', 'perturbation', 'formal_privacy'
    quasi_identifiers : list, optional
        QI columns (auto-detected if not provided)
    preserve_utility : list, optional
        What to preserve: 'mean', 'distribution', 'additivity', 'correlation'
    allow_suppression : bool
        Whether suppression/NaN values are acceptable (default: True)
    verbose : bool
        Print recommendations

    Returns:
    --------
    dict : Recommendation including primary method, alternatives, and reasoning
    """
    # Analyze data
    analysis = analyze_data(data, quasi_identifiers, verbose=verbose)

    recommendation = {
        'analysis': analysis,
        'primary': None,
        'alternatives': [],
        'parameters': {},
        'reasoning': [],
        'warnings': []
    }

    # Add warnings for sensitive columns
    if analysis.get('sensitive_columns'):
        recommendation['warnings'].append(
            f"Consider removing sensitive columns before processing: {list(analysis['sensitive_columns'].keys())}"
        )

    data_type = analysis['data_type']
    risk_level = analysis.get('risk_level', 'medium')

    if verbose:
        print("\n" + "=" * 60)
        print("  METHOD RECOMMENDATION")
        print("=" * 60)

    # MICRODATA RECOMMENDATIONS
    if data_type == 'microdata':
        continuous = analysis.get('continuous_variables', [])
        categorical = analysis.get('categorical_variables', [])
        high_card = analysis.get('high_cardinality_columns', [])

        # Goal-based recommendations
        if goal == 'k-anonymity':
            recommendation['primary'] = 'kANON'
            recommendation['reasoning'].append('Goal is k-anonymity')
            recommendation['parameters'] = {'k': 5 if risk_level == 'high' else 3}

        elif goal == 'formal_privacy' or goal == 'differential_privacy':
            recommendation['primary'] = 'NOISE'
            recommendation['reasoning'].append('Goal is formal/differential privacy')
            recommendation['parameters'] = {'noise_type': 'laplace', 'epsilon': 1.0}

        elif goal == 'add_noise' or goal == 'noise':
            recommendation['primary'] = 'NOISE'
            recommendation['reasoning'].append('Goal is noise addition')
            recommendation['parameters'] = {'magnitude': 0.1}

        elif goal == 'perturb_categories' or goal == 'pram':
            recommendation['primary'] = 'PRAM'
            recommendation['reasoning'].append('Goal is categorical perturbation')
            recommendation['parameters'] = {'p_change': 0.2}

        elif goal in ('aggregate', 'maggr', 'microaggregation'):
            # MAGGR removed — remap to NOISE for continuous data
            recommendation['primary'] = 'NOISE'
            recommendation['reasoning'].append('Microaggregation not available — using NOISE instead')
            recommendation['parameters'] = {'magnitude': 0.15}

        elif goal in ('record_swap', 'recswap'):
            # RECSWAP removed — remap to PRAM for categorical data
            recommendation['primary'] = 'PRAM'
            recommendation['reasoning'].append('Record swapping not available — using PRAM instead')
            recommendation['parameters'] = {'p_change': 0.2}

        elif goal in ('rank_swap', 'rankswap'):
            # RANKSWAP removed — remap to NOISE for continuous data
            recommendation['primary'] = 'NOISE'
            recommendation['reasoning'].append('Rank swapping not available — using NOISE instead')
            recommendation['parameters'] = {'magnitude': 0.15}

        elif goal == 'local_suppression' or goal == 'locsupr':
            recommendation['primary'] = 'LOCSUPR'
            recommendation['reasoning'].append('Goal is local suppression for k-anonymity')
            recommendation['parameters'] = {'k': 3, 'strategy': 'minimum'}

        elif goal == 'perturbation':
            if continuous and categorical:
                recommendation['primary'] = 'PRAM'
                recommendation['alternatives'].append('NOISE')
                recommendation['reasoning'].append('Mixed data - PRAM for categorical, NOISE for continuous')
            elif continuous:
                recommendation['primary'] = 'NOISE'
                recommendation['reasoning'].append('Continuous data - use NOISE')
            else:
                recommendation['primary'] = 'PRAM'
                recommendation['reasoning'].append('Categorical data - use PRAM')

        else:
            # No goal specified - use ReID-based data-driven selection
            if quasi_identifiers is None:
                quasi_identifiers = analysis.get('quasi_identifiers', [])

            from sdc_engine.sdc.selection.rules import select_method_by_features as _select
            reid_result = _select(data, analysis, quasi_identifiers, verbose=False)

            # Single method recommended (pipeline escalation handled by smart_protect if needed)
            recommendation['primary'] = reid_result['method']
            recommendation['parameters'] = reid_result['parameters']
            recommendation['reasoning'].append(reid_result['reason'])
            recommendation['rule'] = reid_result['rule']
            recommendation['confidence'] = reid_result.get('confidence', 'MEDIUM')
            recommendation['alternatives'] = reid_result.get('alternatives', [])

            # Check if pipeline would be appropriate for reactive escalation
            features = reid_result.get('features', {})
            if features:
                pipeline_result = _pipeline_rules(features)
                if pipeline_result.get('applies'):
                    # Store pipeline hint for smart_protect to use if escalation is needed
                    recommendation['pipeline_hint'] = {
                        'pipeline': pipeline_result['pipeline'],
                        'parameters': pipeline_result['parameters'],
                        'reason': pipeline_result['reason'],
                        'priority': pipeline_result.get('priority', 'RECOMMENDED')
                    }
                    recommendation['reasoning'].append(
                        f"Pipeline available if escalation needed: {' -> '.join(pipeline_result['pipeline'])}"
                    )

        # Add alternatives (only the 4 active methods)
        if recommendation['primary'] == 'kANON':
            recommendation['alternatives'].extend(['LOCSUPR', 'PRAM', 'NOISE'])
        elif recommendation['primary'] == 'LOCSUPR':
            recommendation['alternatives'].extend(['kANON', 'PRAM', 'NOISE'])
        elif recommendation['primary'] == 'NOISE':
            recommendation['alternatives'].extend(['PRAM', 'kANON', 'LOCSUPR'])
        elif recommendation['primary'] == 'PRAM':
            recommendation['alternatives'].extend(['kANON', 'NOISE', 'LOCSUPR'])

    # TABULAR RECOMMENDATIONS
    # Tabular-specific methods (THRES, TABSUPR, CROUND, CTA) are not available.
    # Fall back to microdata methods for any tabular data.
    else:
        import logging
        logging.getLogger(__name__).info(
            "Tabular data detected. Tabular-specific methods not available "
            "— microdata methods will be used."
        )
        recommendation['primary'] = 'PRAM'
        recommendation['reasoning'].append('Tabular methods not available — using PRAM as fallback')
        recommendation['alternatives'].extend(['kANON', 'NOISE', 'LOCSUPR'])

    # Remove duplicates from alternatives
    recommendation['alternatives'] = list(dict.fromkeys(
        [a for a in recommendation['alternatives'] if a != recommendation['primary']]
    ))

    # Print recommendation
    if verbose:
        primary = recommendation['primary']
        print(f"\n[*] PRIMARY RECOMMENDATION: {primary}")
        print(f"  {METHOD_INFO[primary]['description']}")

        if recommendation['parameters']:
            print(f"\n  Suggested parameters:")
            for param, value in recommendation['parameters'].items():
                print(f"    - {param}: {value}")

        print(f"\n  Reasoning:")
        for reason in recommendation['reasoning']:
            print(f"    - {reason}")

        if recommendation['alternatives']:
            print(f"\n  Alternatives: {', '.join(recommendation['alternatives'])}")

        if recommendation['warnings']:
            print(f"\n  [!] Warnings:")
            for warning in recommendation['warnings']:
                print(f"    - {warning}")

        # Usage example
        print(f"\n  Example usage:")
        print(f"    from src import apply_{primary.lower()}")
        if primary == 'kANON':
            k = recommendation['parameters'].get('k', 5)
            qis = analysis.get('quasi_identifiers', ['col1', 'col2'])
            print(f"    result = apply_kanon(data, k={k}, quasi_identifiers={qis[:3]})")
        elif primary == 'PRAM':
            cats = analysis.get('categorical_variables', ['category'])[:2]
            print(f"    result = apply_pram(data, variables={cats})")
        elif primary == 'NOISE':
            conts = analysis.get('continuous_variables', ['value'])[:2]
            print(f"    result = apply_noise(data, variables={conts})")
        elif primary == 'LOCSUPR':
            qis = analysis.get('quasi_identifiers', ['col1', 'col2'])
            print(f"    result = apply_locsupr(data, quasi_identifiers={qis[:3]}, k=3)")

    return recommendation


# =============================================================================
# APPLY AND VALIDATE
# =============================================================================

# Map method names to functions (only the 4 active methods)
METHOD_FUNCTIONS = {
    'kANON': apply_kanon,
    'PRAM': apply_pram,
    'NOISE': apply_noise,
    'LOCSUPR': apply_locsupr,
}


def apply_and_validate(
    data: pd.DataFrame,
    method: Optional[str] = None,
    quasi_identifiers: Optional[List[str]] = None,
    goal: Optional[str] = None,
    auto_fix: bool = True,
    verbose: bool = True,
    **method_params
) -> Dict:
    """
    Recommend, apply SDC method, and validate the results.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to protect
    method : str, optional
        SDC method to use. If None, auto-recommends based on data.
    quasi_identifiers : list, optional
        QI columns for microdata (auto-detected if not provided)
    goal : str, optional
        Privacy goal for recommendation
    auto_fix : bool
        If True, automatically re-applies method if validation fails
    verbose : bool
        Print progress and results
    **method_params : dict
        Additional parameters for the chosen method

    Returns:
    --------
    dict : Results including:
        - 'original_data': Original input
        - 'protected_data': Anonymized output
        - 'method': Method used
        - 'metadata': Method metadata
        - 'validation': Validation results
        - 'risk_before': Risk metrics before
        - 'risk_after': Risk metrics after (for microdata)
    """
    results = {
        'original_data': data,
        'protected_data': None,
        'method': None,
        'parameters': {},
        'metadata': None,
        'validation': {},
        'risk_before': None,
        'risk_after': None,
        'success': False
    }

    # Step 1: Analyze data
    if verbose:
        print("\n" + "=" * 60)
        print("  APPLY AND VALIDATE SDC METHOD")
        print("=" * 60)

    analysis = analyze_data(data, quasi_identifiers, verbose=verbose)
    data_type = analysis['data_type']

    # Get QIs for microdata
    if data_type == 'microdata':
        if quasi_identifiers is None:
            quasi_identifiers = analysis.get('quasi_identifiers', [])
        results['risk_before'] = analysis.get('disclosure_risk')

    # Step 2: Determine method
    if method is None:
        if verbose:
            print("\n--- Auto-selecting method ---")
        rec = recommend_method(data, goal=goal, quasi_identifiers=quasi_identifiers, verbose=verbose)
        method = rec['primary']
        # Use recommended parameters if not overridden
        for param, value in rec.get('parameters', {}).items():
            if param not in method_params:
                method_params[param] = value

    results['method'] = method
    results['parameters'] = method_params.copy()

    if verbose:
        print(f"\n--- Applying {method} ---")

    # Step 3: Build method parameters
    apply_func = METHOD_FUNCTIONS.get(method)
    if apply_func is None:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHOD_FUNCTIONS.keys())}")

    # Add method-specific defaults
    if method == 'kANON':
        if 'quasi_identifiers' not in method_params and quasi_identifiers:
            method_params['quasi_identifiers'] = quasi_identifiers
        if 'k' not in method_params:
            method_params['k'] = 5
    elif method == 'PRAM':
        if 'variables' not in method_params:
            method_params['variables'] = analysis.get('categorical_variables', [])[:5]
    elif method == 'NOISE':
        if 'variables' not in method_params:
            method_params['variables'] = analysis.get('continuous_variables', [])[:5]
    elif method == 'LOCSUPR':
        if 'quasi_identifiers' not in method_params and quasi_identifiers:
            method_params['quasi_identifiers'] = quasi_identifiers
        if 'k' not in method_params:
            method_params['k'] = 3

    # Step 4: Apply method
    # Filter out parameters that aren't supported by the method
    # These are parameters used by apply_and_validate/recommend_method
    non_method_params = {'goal', 'auto_fix', 'allow_suppression', 'preserve_utility'}

    # Method-specific parameter filtering
    # Each method only accepts certain parameters
    method_valid_params = {
        'kANON': {'k', 'quasi_identifiers', 'strategy', 'importance'},
        'LOCSUPR': {'k', 'quasi_identifiers'},
        'PRAM': {'variables', 'p_change', 'transition_matrix', 'invariant'},
        'NOISE': {'variables', 'noise_type', 'magnitude', 'epsilon'},
    }

    # Get valid params for this method (or allow all if not specified)
    valid_params = method_valid_params.get(method, set())

    # Filter: remove non-method params AND params not valid for this specific method
    filtered_params = {
        k: v for k, v in method_params.items()
        if k not in non_method_params and (not valid_params or k in valid_params)
    }

    try:
        protected, metadata = apply_func(data, return_metadata=True, verbose=verbose, **filtered_params)
        results['protected_data'] = protected
        results['metadata'] = metadata
    except Exception as e:
        if verbose:
            print(f"\n[ERROR] Method failed: {e}")
        results['validation']['error'] = str(e)
        return results

    # Step 5: Validate results
    if verbose:
        print("\n--- Validating Results ---")

    validation = validate_protection(
        original=data,
        protected=protected,
        method=method,
        quasi_identifiers=quasi_identifiers,
        metadata=metadata,
        verbose=verbose
    )
    results['validation'] = validation

    # Step 6: Calculate risk after (for microdata)
    if data_type == 'microdata' and quasi_identifiers:
        try:
            # For kANON, use the protected data's QIs
            protected_qis = [qi for qi in quasi_identifiers if qi in protected.columns]
            if protected_qis:
                risk_after = calculate_disclosure_risk(protected, protected_qis, k=3)
                results['risk_after'] = risk_after

                # Calculate ReID before and after
                try:
                    reid_before = calculate_reid(data, quasi_identifiers)
                    reid_after = calculate_reid(protected, protected_qis)
                    results['reid_before'] = reid_before
                    results['reid_after'] = reid_after
                except Exception:
                    pass  # ReID is optional

                if verbose:
                    print(f"\n--- Risk Comparison ---")
                    print(f"Before: {results['risk_before']['uniqueness_rate']:.1%} uniqueness, "
                          f"{results['risk_before']['risk_rate']:.1%} at risk")
                    print(f"After:  {risk_after['uniqueness_rate']:.1%} uniqueness, "
                          f"{risk_after['risk_rate']:.1%} at risk")

                    risk_reduction = results['risk_before']['risk_rate'] - risk_after['risk_rate']
                    print(f"Risk reduction: {risk_reduction:.1%}")

                    # Show ReID if available
                    if results.get('reid_before') and results.get('reid_after'):
                        print(f"\n--- ReID (Re-Identification Risk) ---")
                        print(f"ReID95 Before: {results['reid_before']['reid_95']:.1%}")
                        print(f"ReID95 After:  {results['reid_after']['reid_95']:.1%}")
        except Exception as e:
            if verbose:
                print(f"Could not calculate risk after: {e}")

    # Step 7: Check if validation passed
    results['success'] = validation.get('passed', False)

    if verbose:
        if results['success']:
            print(f"\n[OK] Validation PASSED")
        else:
            print(f"\n[!] Validation FAILED")
            for issue in validation.get('issues', []):
                print(f"    - {issue}")

    # Step 8: Auto-fix if needed
    if not results['success'] and auto_fix and validation.get('can_retry', False):
        if verbose:
            print(f"\n--- Auto-fixing with stronger parameters ---")

        # Adjust parameters for retry
        retry_params = method_params.copy()
        if method == 'kANON':
            # Increase k
            retry_params['k'] = retry_params.get('k', 5) + 2
        elif method == 'NOISE':
            # Increase noise magnitude
            retry_params['magnitude'] = retry_params.get('magnitude', 0.1) * 2
        elif method == 'LOCSUPR':
            # Increase k
            retry_params['k'] = retry_params.get('k', 3) + 2

        if retry_params != method_params:
            if verbose:
                print(f"Retrying with: {retry_params}")

            try:
                protected2, metadata2 = apply_func(data, return_metadata=True, verbose=False, **retry_params)

                # Re-validate
                validation2 = validate_protection(
                    original=data,
                    protected=protected2,
                    method=method,
                    quasi_identifiers=quasi_identifiers,
                    metadata=metadata2,
                    verbose=verbose
                )

                if validation2.get('passed', False):
                    results['protected_data'] = protected2
                    results['metadata'] = metadata2
                    results['parameters'] = retry_params
                    results['validation'] = validation2
                    results['success'] = True

                    if verbose:
                        print(f"[OK] Auto-fix successful with adjusted parameters")
            except Exception as e:
                if verbose:
                    print(f"Auto-fix failed: {e}")

    # Final summary
    if verbose:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        print(f"Method: {results['method']}")
        print(f"Parameters: {results['parameters']}")
        print(f"Success: {results['success']}")

        if results['metadata']:
            stats = results['metadata'].get('statistics', {})
            if stats:
                print(f"Key statistics:")
                for key, value in list(stats.items())[:5]:
                    if isinstance(value, float):
                        print(f"  - {key}: {value:.4f}")
                    else:
                        print(f"  - {key}: {value}")

    return results


def validate_protection(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    method: str,
    quasi_identifiers: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    verbose: bool = True
) -> Dict:
    """
    Validate that the SDC method was applied correctly.

    Returns:
    --------
    dict : Validation results with 'passed', 'issues', 'checks'
    """
    validation = {
        'passed': True,
        'issues': [],
        'checks': {},
        'can_retry': False
    }

    if verbose:
        print(f"Validating {method} results...")

    # Common checks
    # 1. Data shape preserved (except for suppression)
    if method not in ['kANON']:  # kANON may suppress records
        if len(protected) != len(original):
            validation['checks']['shape_preserved'] = False
            validation['issues'].append(f"Record count changed: {len(original)} -> {len(protected)}")
        else:
            validation['checks']['shape_preserved'] = True

    # Method-specific validation
    if method == 'kANON':
        # Check k-anonymity achieved
        if quasi_identifiers and metadata:
            k_check = metadata.get('k_anonymity_check', {})
            is_k_anon = k_check.get('is_k_anonymous', False)
            validation['checks']['k_anonymous'] = is_k_anon

            if not is_k_anon:
                validation['passed'] = False
                validation['issues'].append(
                    f"k-anonymity not achieved (min group: {k_check.get('min_group_size', '?')})"
                )
                validation['can_retry'] = True

            # Check suppression rate is reasonable
            supp_rate = metadata.get('statistics', {}).get('suppression_rate', 0)
            if supp_rate > 0.5:
                validation['issues'].append(f"High suppression rate: {supp_rate:.1%}")

    elif method == 'PRAM':
        # Check changes were made
        if metadata:
            changes = metadata.get('statistics', {}).get('total_changes', 0)
            validation['checks']['changes_made'] = changes > 0

            if changes == 0:
                validation['issues'].append("No changes made by PRAM")

    elif method == 'NOISE':
        # Check correlation preserved
        if metadata:
            for var, stats in metadata.get('statistics', {}).get('value_changes', {}).items():
                corr = stats.get('correlation', 0)
                validation['checks'][f'{var}_correlation'] = corr

                if corr < 0.7:
                    validation['issues'].append(f"Low correlation for {var}: {corr:.2f}")

    elif method == 'LOCSUPR':
        # Check k-anonymity achieved
        if metadata:
            k_check = metadata.get('k_anonymity_check', {})
            is_k_anon = k_check.get('is_k_anonymous', False)
            validation['checks']['k_anonymous'] = is_k_anon

            if not is_k_anon:
                validation['issues'].append(
                    f"k-anonymity may not be achieved (min group: {k_check.get('min_group_size', '?')})"
                )

            # Check suppression rate is reasonable
            supp_rate = metadata.get('statistics', {}).get('suppression_rate', 0)
            if supp_rate > 0.3:
                validation['issues'].append(f"High suppression rate: {supp_rate:.1%}")

    # Overall pass/fail
    if validation['issues']:
        # Only fail for critical issues
        critical_issues = [i for i in validation['issues'] if 'not achieved' in i or 'remain' in i]
        if critical_issues:
            validation['passed'] = False

    if verbose:
        for check, result in validation['checks'].items():
            status = "[OK]" if result else "[FAIL]"
            print(f"  {status} {check}: {result}")

    return validation


# =============================================================================
# PIPELINE RECOMMENDATION
# =============================================================================

# Pipeline goals with corresponding method sequences
PIPELINE_GOALS = {
    # Privacy-focused goals
    'k_anonymity': {
        'description': 'Achieve k-anonymity through generalization or suppression',
        'methods': ['kANON'],
        'alternatives': ['LOCSUPR']
    },
    'strong_privacy': {
        'description': 'Maximum privacy protection for sensitive data',
        'methods': ['kANON', 'NOISE'],
        'alternatives': ['LOCSUPR']
    },
    'perturbation': {
        'description': 'Perturb data while preserving statistical properties',
        'methods': ['PRAM', 'NOISE'],
        'alternatives': ['kANON']
    },

    # Utility-focused goals
    'preserve_statistics': {
        'description': 'Protect data while preserving means and distributions',
        'methods': ['NOISE'],
        'alternatives': ['PRAM']
    },
    'preserve_distributions': {
        'description': 'Protect categorical data preserving marginal distributions',
        'methods': ['PRAM'],
        'alternatives': ['NOISE']
    },
    'preserve_rankings': {
        'description': 'Protect continuous data preserving approximate rankings',
        'methods': ['NOISE'],
        'alternatives': ['PRAM']
    },

    # Combined/comprehensive goals
    'full_protection': {
        'description': 'Apply multiple methods for comprehensive protection',
        'methods': ['kANON', 'PRAM', 'NOISE'],
        'alternatives': []
    },
}


def _auto_detect_goals(analysis: Dict) -> List[str]:
    """
    Auto-detect appropriate pipeline goals based on data analysis.

    Parameters:
    -----------
    analysis : dict
        Result from analyze_data()

    Returns:
    --------
    list : Detected goals in order of priority
    """
    goals = []
    data_type = analysis.get('data_type', 'microdata')
    risk_level = analysis.get('risk_level', 'medium')

    if data_type == 'microdata':
        continuous = analysis.get('continuous_variables', [])
        categorical = analysis.get('categorical_variables', [])

        # Has both variable types -> perturbation pipeline (most comprehensive)
        if continuous and categorical:
            if risk_level == 'high':
                goals.append('k_anonymity')
            goals.append('perturbation')

        # Mostly continuous -> statistics preservation
        elif continuous and len(continuous) > len(categorical):
            if risk_level in ['high', 'medium']:
                goals.append('preserve_statistics')
            else:
                goals.append('preserve_rankings')

        # Mostly categorical -> distribution preservation + k-anonymity if high risk
        elif categorical:
            if risk_level == 'high':
                goals.append('k_anonymity')
            goals.append('preserve_distributions')

    else:  # tabular — tabular-specific methods not available, use perturbation
        goals.append('perturbation')

    return goals if goals else ['perturbation']  # Default fallback


def recommend_pipeline(
    data: pd.DataFrame,
    goals: Optional[List[str]] = None,
    auto_detect: bool = True,
    quasi_identifiers: Optional[List[str]] = None,
    preserve_utility: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Recommend a pipeline of SDC methods based on goals (hybrid: user-defined + auto-detect).

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to protect
    goals : list of str, optional
        User-defined goals. Options:
        - 'k_anonymity': Achieve formal k-anonymity
        - 'strong_privacy': Maximum privacy protection
        - 'perturbation': Perturb while preserving stats
        - 'preserve_statistics': Protect preserving means
        - 'preserve_distributions': Protect preserving marginals
        - 'preserve_rankings': Protect preserving rankings
        - 'full_protection': Comprehensive multi-method protection
    auto_detect : bool, default=True
        If True and goals is None, auto-detect goals from data
        If True and goals is provided, append detected goals
    quasi_identifiers : list, optional
        QI columns for k-anonymity methods
    preserve_utility : list, optional
        What to preserve: 'mean', 'distribution', 'correlation'
    verbose : bool
        Print pipeline recommendation

    Returns:
    --------
    dict : Pipeline recommendation with:
        - 'goals': Final list of goals
        - 'pipeline': Ordered list of methods to apply
        - 'parameters': Suggested parameters per method
        - 'alternatives': Alternative methods per step
        - 'analysis': Data analysis results
        - 'reasoning': Explanation for choices

    Examples:
    ---------
    >>> # Auto-detect goals
    >>> pipeline = recommend_pipeline(data)

    >>> # User-defined goal
    >>> pipeline = recommend_pipeline(data, goals=['k_anonymity'])

    >>> # Hybrid: user goal + auto-detect additional
    >>> pipeline = recommend_pipeline(data, goals=['k_anonymity'], auto_detect=True)
    """
    # Step 1: Analyze data
    analysis = analyze_data(data, quasi_identifiers, verbose=False)
    data_type = analysis.get('data_type', 'microdata')

    # Get QIs if needed
    if quasi_identifiers is None and data_type == 'microdata':
        quasi_identifiers = analysis.get('quasi_identifiers', [])

    recommendation = {
        'goals': [],
        'pipeline': [],
        'parameters': {},
        'alternatives': {},
        'analysis': analysis,
        'reasoning': [],
        'warnings': []
    }

    # Step 2: Determine goals (hybrid approach)
    final_goals = []

    if goals:
        # Validate user-provided goals
        for g in goals:
            if g in PIPELINE_GOALS:
                final_goals.append(g)
            else:
                recommendation['warnings'].append(f"Unknown goal '{g}' - ignoring")

    if auto_detect and (not goals or len(final_goals) < len(goals)):
        # Auto-detect additional goals
        detected = _auto_detect_goals(analysis)
        for g in detected:
            if g not in final_goals:
                final_goals.append(g)
                recommendation['reasoning'].append(f"Auto-detected goal: {g}")

    if not final_goals:
        # Fallback to default
        final_goals = ['perturbation']
        recommendation['reasoning'].append(f"Using default goal: {final_goals[0]}")

    recommendation['goals'] = final_goals

    # Step 3: Build pipeline from goals
    pipeline_methods = []
    seen_methods = set()

    for goal in final_goals:
        goal_info = PIPELINE_GOALS.get(goal, {})
        methods = goal_info.get('methods', [])

        for method in methods:
            # Filter by data type compatibility
            method_info = METHOD_INFO.get(method, {})
            method_data_type = method_info.get('data_type', 'microdata')

            if method_data_type == data_type and method not in seen_methods:
                pipeline_methods.append(method)
                seen_methods.add(method)

                # Get alternatives
                recommendation['alternatives'][method] = [
                    alt for alt in goal_info.get('alternatives', [])
                    if alt not in seen_methods and METHOD_INFO.get(alt, {}).get('data_type') == data_type
                ]

    recommendation['pipeline'] = pipeline_methods

    # Step 4: Determine parameters for each method
    continuous = analysis.get('continuous_variables', [])
    categorical = analysis.get('categorical_variables', [])
    risk_level = analysis.get('risk_level', 'medium')

    for method in pipeline_methods:
        params = {}

        if method == 'kANON':
            params['k'] = 5 if risk_level == 'high' else 3
            params['quasi_identifiers'] = quasi_identifiers
            params['strategy'] = 'hybrid'
            recommendation['reasoning'].append(f"kANON: k={params['k']} based on {risk_level} risk")

        elif method == 'LOCSUPR':
            params['k'] = 3
            params['quasi_identifiers'] = quasi_identifiers
            params['strategy'] = 'minimum'

        elif method == 'PRAM':
            params['variables'] = categorical[:5] if categorical else []
            params['p_change'] = 0.2
            params['invariant'] = True
            recommendation['reasoning'].append(f"PRAM: targeting {len(params['variables'])} categorical vars")

        elif method == 'NOISE':
            params['variables'] = continuous[:5] if continuous else []
            params['noise_type'] = 'gaussian'
            params['magnitude'] = 0.1 if risk_level == 'low' else 0.15
            recommendation['reasoning'].append(f"NOISE: targeting {len(params['variables'])} continuous vars")

        recommendation['parameters'][method] = params

    # Step 5: Add warnings for sensitive columns
    if analysis.get('sensitive_columns'):
        recommendation['warnings'].append(
            f"Consider removing sensitive columns before pipeline: {list(analysis['sensitive_columns'].keys())}"
        )

    # Print recommendation
    if verbose:
        print("\n" + "=" * 60)
        print("  PIPELINE RECOMMENDATION")
        print("=" * 60)

        print(f"\nData Type: {data_type}")
        print(f"Risk Level: {risk_level}")
        print(f"\nGoals: {', '.join(final_goals)}")

        print(f"\n--- Pipeline ({len(pipeline_methods)} methods) ---")
        for i, method in enumerate(pipeline_methods, 1):
            method_info = METHOD_INFO.get(method, {})
            print(f"\n  Step {i}: {method}")
            print(f"    {method_info.get('description', '')}")

            params = recommendation['parameters'].get(method, {})
            if params:
                print(f"    Parameters:")
                for k, v in list(params.items())[:3]:
                    if isinstance(v, list):
                        print(f"      - {k}: {v[:3]}{'...' if len(v) > 3 else ''}")
                    else:
                        print(f"      - {k}: {v}")

            alts = recommendation['alternatives'].get(method, [])
            if alts:
                print(f"    Alternatives: {', '.join(alts)}")

        print("\n--- Reasoning ---")
        for reason in recommendation['reasoning']:
            print(f"  - {reason}")

        if recommendation['warnings']:
            print("\n--- Warnings ---")
            for warning in recommendation['warnings']:
                print(f"  [!] {warning}")

        # Usage example
        print("\n--- Usage ---")
        print("  from select_method import apply_pipeline")
        print(f"  result = apply_pipeline(data, goals={final_goals})")

    return recommendation


def apply_pipeline(
    data: pd.DataFrame,
    goals: Optional[List[str]] = None,
    pipeline: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
    auto_detect: bool = True,
    verbose: bool = True,
    **override_params
) -> Dict:
    """
    Apply a pipeline of SDC methods sequentially.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to protect
    goals : list of str, optional
        Goals to achieve (see recommend_pipeline for options)
    pipeline : list of str, optional
        Explicit list of methods to apply. Overrides goals.
    quasi_identifiers : list, optional
        QI columns for relevant methods
    auto_detect : bool, default=True
        Auto-detect goals if not provided
    verbose : bool
        Print progress
    **override_params : dict
        Override parameters for specific methods
        Format: method_param (e.g., kANON_k=5, NOISE_magnitude=0.2)

    Returns:
    --------
    dict : Pipeline results with:
        - 'original_data': Original input
        - 'protected_data': Final protected output
        - 'pipeline': Methods applied
        - 'steps': Results per step
        - 'success': Overall success
        - 'risk_before': Initial risk metrics
        - 'risk_after': Final risk metrics

    Examples:
    ---------
    >>> # Apply based on goals
    >>> result = apply_pipeline(data, goals=['k_anonymity'])

    >>> # Apply explicit pipeline
    >>> result = apply_pipeline(data, pipeline=['kANON', 'NOISE'])

    >>> # Override parameters
    >>> result = apply_pipeline(data, goals=['k_anonymity'], kANON_k=7)
    """
    import time

    results = {
        'original_data': data,
        'protected_data': None,
        'pipeline': [],
        'steps': [],
        'success': False,
        'risk_before': None,
        'risk_after': None,
        'total_time': 0
    }

    start_time = time.time()

    # Step 1: Determine pipeline
    if pipeline:
        # Use explicit pipeline
        methods_to_apply = pipeline
        params_per_method = {}

        # Get default parameters for each method
        analysis = analyze_data(data, quasi_identifiers, verbose=False)
        for method in methods_to_apply:
            # Use recommend_method to get default params
            rec = recommend_method(data, verbose=False)
            params_per_method[method] = rec.get('parameters', {})
    else:
        # Use goals to determine pipeline
        rec = recommend_pipeline(
            data,
            goals=goals,
            auto_detect=auto_detect,
            quasi_identifiers=quasi_identifiers,
            verbose=verbose
        )
        methods_to_apply = rec['pipeline']
        params_per_method = rec['parameters']
        results['risk_before'] = rec['analysis'].get('disclosure_risk')

    if not methods_to_apply:
        if verbose:
            print("[!] No methods in pipeline")
        return results

    results['pipeline'] = methods_to_apply

    # Apply override parameters
    for key, value in override_params.items():
        # Parse method_param format (e.g., kANON_k=7, NOISE_magnitude=0.2)
        parts = key.split('_', 1)
        if len(parts) == 2:
            method_key, param = parts
            # Try to find matching method (case-insensitive)
            matched_method = None
            for m in params_per_method:
                if m.upper() == method_key.upper():
                    matched_method = m
                    break
            if matched_method:
                params_per_method[matched_method][param] = value

    if verbose:
        print("\n" + "=" * 60)
        print("  APPLYING PIPELINE")
        print("=" * 60)
        print(f"\nMethods: {' -> '.join(methods_to_apply)}")

    # Step 2: Apply methods sequentially
    current_data = data.copy()
    all_success = True

    for i, method in enumerate(methods_to_apply, 1):
        step_start = time.time()

        if verbose:
            print(f"\n--- Step {i}/{len(methods_to_apply)}: {method} ---")

        # Get parameters
        method_params = params_per_method.get(method, {}).copy()

        # Apply method
        apply_func = METHOD_FUNCTIONS.get(method)
        if apply_func is None:
            if verbose:
                print(f"  [!] Unknown method: {method}")
            all_success = False
            continue

        try:
            # Filter parameters for method signature
            non_method_params = {'goal', 'auto_fix', 'allow_suppression', 'preserve_utility'}
            filtered_params = {k: v for k, v in method_params.items() if k not in non_method_params}

            protected, metadata = apply_func(
                current_data,
                return_metadata=True,
                verbose=False,
                **filtered_params
            )

            step_time = time.time() - step_start

            step_result = {
                'method': method,
                'success': True,
                'time': round(step_time, 3),
                'parameters': filtered_params,
                'metadata': metadata
            }

            if verbose:
                print(f"  [OK] Applied in {step_time:.3f}s")

                # Print key stats
                stats = metadata.get('statistics', {})
                if method == 'kANON':
                    print(f"      Suppression rate: {stats.get('suppression_rate', 0):.1%}")
                elif method == 'PRAM':
                    print(f"      Changes: {stats.get('total_changes', 0)}")
                elif method == 'NOISE':
                    print(f"      Variables perturbed: {len(filtered_params.get('variables', []))}")

            current_data = protected
            results['steps'].append(step_result)

        except Exception as e:
            step_time = time.time() - step_start
            step_result = {
                'method': method,
                'success': False,
                'time': round(step_time, 3),
                'error': str(e)
            }
            results['steps'].append(step_result)
            all_success = False

            if verbose:
                print(f"  [FAIL] {str(e)[:50]}")

    # Step 3: Final results
    results['protected_data'] = current_data
    results['success'] = all_success
    results['total_time'] = round(time.time() - start_time, 3)

    # Calculate final risk (for microdata)
    analysis = analyze_data(data, quasi_identifiers, verbose=False)
    if analysis.get('data_type') == 'microdata' and quasi_identifiers:
        try:
            final_qis = [qi for qi in quasi_identifiers if qi in current_data.columns]
            if final_qis:
                risk_after = calculate_disclosure_risk(current_data, final_qis, k=3)
                results['risk_after'] = risk_after

                # Calculate ReID before and after
                try:
                    reid_before = calculate_reid(data, quasi_identifiers)
                    reid_after = calculate_reid(current_data, final_qis)
                    results['reid_before'] = reid_before
                    results['reid_after'] = reid_after
                except Exception:
                    pass  # ReID is optional
        except Exception:
            pass

    if verbose:
        print("\n" + "=" * 60)
        print("  PIPELINE SUMMARY")
        print("=" * 60)
        print(f"\nTotal time: {results['total_time']}s")
        print(f"Success: {results['success']}")
        print(f"Steps completed: {sum(1 for s in results['steps'] if s.get('success'))}/{len(methods_to_apply)}")

        if results.get('risk_before') and results.get('risk_after'):
            before = results['risk_before'].get('risk_rate', 0)
            after = results['risk_after'].get('risk_rate', 0)
            print(f"\nRisk reduction: {before:.1%} -> {after:.1%}")

        # Show ReID if available
        if results.get('reid_before') and results.get('reid_after'):
            print(f"ReID95: {results['reid_before']['reid_95']:.1%} -> {results['reid_after']['reid_95']:.1%}")

    return results


def _check_reid_targets(reid: Dict, targets: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if ReID metrics meet target thresholds.

    Parameters:
    -----------
    reid : dict
        ReID metrics from calculate_reid()
    targets : dict
        Target thresholds (e.g., {'reid_95': 0.05})

    Returns:
    --------
    tuple : (all_met: bool, per_target: dict)
    """
    results = {}
    for metric, threshold in targets.items():
        actual = reid.get(metric)
        if actual is not None:
            results[metric] = actual <= threshold
        else:
            results[metric] = False  # Unknown metric not met
    return all(results.values()), results


def _get_method_fallback_order(initial_method: str, data_has_categorical: bool, data_has_continuous: bool, has_reid_target: bool = False) -> List[str]:
    """
    Get ordered list of methods to try if initial method fails to meet ReID target.

    When has_reid_target is True, structural methods (kANON, LOCSUPR) that can
    actually reduce re-identification risk are prioritised.  Perturbation methods
    (PRAM, NOISE) cannot reduce ReID because they don't change the uniqueness
    structure — they only add noise to values.

    When has_reid_target is False (no ReID constraint), fast perturbation methods
    are tried first for speed.
    """
    if has_reid_target:
        # ReID-reducing methods first: only structural methods lower uniqueness
        if initial_method == 'kANON':
            all_methods = ['LOCSUPR', 'PRAM', 'NOISE']
        elif initial_method == 'LOCSUPR':
            all_methods = ['kANON', 'PRAM', 'NOISE']
        elif initial_method == 'PRAM':
            all_methods = ['kANON', 'LOCSUPR', 'NOISE']
        elif initial_method == 'NOISE':
            all_methods = ['kANON', 'LOCSUPR', 'PRAM']
        else:
            all_methods = ['kANON', 'LOCSUPR', 'PRAM', 'NOISE']
    else:
        # No ReID target — speed first
        if initial_method == 'PRAM':
            all_methods = ['NOISE', 'kANON', 'LOCSUPR']
        elif initial_method == 'NOISE':
            all_methods = ['PRAM', 'kANON', 'LOCSUPR']
        elif initial_method == 'kANON':
            all_methods = ['PRAM', 'NOISE', 'LOCSUPR']
        elif initial_method == 'LOCSUPR':
            all_methods = ['PRAM', 'NOISE', 'kANON']
        else:
            all_methods = ['PRAM', 'NOISE', 'kANON', 'LOCSUPR']

    # Remove initial method and filter by data type
    result = []
    for m in all_methods:
        if m == initial_method:
            continue
        # PRAM needs categorical data
        if m == 'PRAM' and not data_has_categorical:
            continue
        # NOISE needs continuous data
        if m == 'NOISE' and not data_has_continuous:
            continue
        result.append(m)

    return result


def _try_method_with_tuning(
    data: pd.DataFrame,
    method: str,
    quasi_identifiers: List[str],
    reid_target: Dict[str, float],
    max_iterations: int,
    verbose: bool,
    min_utility: float = 0.50,  # Minimum acceptable utility score
    max_info_loss: Optional[float] = None,  # Reject iterations exceeding this
    start_k: int = 5,  # Starting k from Configure tab target_k
) -> Tuple[Optional[Dict], bool, int, float]:
    """
    Try a single method with parameter tuning to meet ReID target.

    Returns:
    --------
    tuple: (best_result, target_met, iterations_used, utility_score)
    """
    best_result = None
    target_met = False
    iterations_used = 0
    best_utility = 0.0

    for iteration in range(max_iterations):
        iterations_used = iteration + 1
        tuned_params = _get_tuning_params(method, iteration, {}, start_k=start_k)

        if verbose:
            param_str = ', '.join(f"{k}={v}" for k, v in tuned_params.items()
                                  if k not in ['variables', 'quasi_identifiers'])
            print(f"    Iteration {iteration + 1}: {param_str}")

        try:
            iter_result = apply_and_validate(
                data,
                method=method,
                quasi_identifiers=quasi_identifiers,
                verbose=False,
                **tuned_params

            )

            if iter_result.get('success'):
                protected = iter_result.get('protected_data')
                if protected is not None:
                    try:
                        protected_qis = [qi for qi in quasi_identifiers if qi in protected.columns]
                        reid_after = calculate_reid(protected, protected_qis)
                        iter_result['reid_after'] = reid_after

                        # Calculate utility for this result
                        try:
                            utility = calculate_utility_metrics(data, protected, columns=quasi_identifiers)
                            iter_result['utility_metrics'] = utility
                            utility_score = utility.get('utility_score', 0)
                        except Exception:
                            utility_score = 0

                        targets_met, target_results = _check_reid_targets(reid_after, reid_target)

                        if verbose:
                            for metric, met in target_results.items():
                                actual = reid_after.get(metric, 0)
                                target_val = reid_target.get(metric, 0)
                                status = "[OK]" if met else "[MISS]"
                                print(f"      {status} {metric}: {actual:.1%} (target: {target_val:.1%})")
                            if utility_score > 0:
                                print(f"      Utility: {utility_score:.1%}")

                        # Check info loss constraint per iteration
                        iter_info_loss = utility.get('information_loss', 1.0 - utility_score) if isinstance(utility, dict) else (1.0 - utility_score)
                        if max_info_loss is not None and iter_info_loss > max_info_loss:
                            if verbose:
                                print(f"      Iteration rejected: info loss {iter_info_loss:.1%} > max {max_info_loss:.1%}")
                            continue  # Try next iteration with stronger params

                        if targets_met:
                            if verbose:
                                print(f"      ReID targets MET!")
                            return iter_result, True, iterations_used, utility_score

                        # Keep best result so far (prioritize by ReID, then utility)
                        if best_result is None or reid_after.get('reid_95', 1) < best_result.get('reid_after', {}).get('reid_95', 1):
                            best_result = iter_result
                            best_utility = utility_score

                    except Exception as e:
                        if verbose:
                            print(f"      Could not calculate ReID: {e}")
                        if best_result is None:
                            best_result = iter_result
            else:
                if verbose:
                    print(f"      Method failed")

        except Exception as e:
            if verbose:
                print(f"      Error: {e}")

    return best_result, False, iterations_used, best_utility


def _get_tuning_params(method: str, iteration: int, base_params: Dict, start_k: int = 5) -> Dict:
    """
    Get tuned parameters for a method based on iteration.

    Higher iterations = stronger protection.
    start_k controls the initial k value for k-based methods (from Configure tab).
    """
    params = base_params.copy()

    if method == 'kANON':
        # Build escalation sequence starting from start_k
        all_k = [3, 5, 7, 10, 15, 20, 25, 30]
        k_values = [k for k in all_k if k >= start_k] or [start_k]
        params['k'] = k_values[min(iteration, len(k_values) - 1)]

    elif method == 'LOCSUPR':
        # Build escalation sequence starting from start_k
        all_k = [3, 5, 7, 10, 15]
        k_values = [k for k in all_k if k >= start_k] or [start_k]
        params['k'] = k_values[min(iteration, len(k_values) - 1)]

    elif method == 'NOISE':
        # Increase noise magnitude: 0.1 -> 0.15 -> 0.2 -> 0.3 -> 0.5
        mag_values = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
        params['magnitude'] = mag_values[min(iteration, len(mag_values) - 1)]

    elif method == 'PRAM':
        # Increase change probability
        p_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        params['p_change'] = p_values[min(iteration, len(p_values) - 1)]

    return params


def _select_optimal_starting_method(
    initial_reid: Dict,
    analysis: Dict,
    reid_target: Dict[str, float]
) -> str:
    """
    Proactively select the optimal starting method based on ReID distribution.

    Key insight: For very high uniqueness (ReID_95 >= 80%), PRAM and NOISE
    cannot reduce ReID because they perturb values but don't group records.
    Only suppression/grouping methods (kANON, LOCSUPR) can help.

    Parameters:
    -----------
    initial_reid : dict
        Initial ReID metrics (reid_50, reid_95, reid_99, etc.)
    analysis : dict
        Data analysis results
    reid_target : dict
        Target ReID thresholds

    Returns:
    --------
    str : Recommended starting method
    """
    if initial_reid is None:
        return 'kANON'  # Default fallback

    reid_95 = initial_reid.get('reid_95', 1)
    reid_50 = initial_reid.get('reid_50', 0.5)
    uniqueness_rate = initial_reid.get('uniqueness_rate', 0)

    target_95 = reid_target.get('reid_95', 0.10)

    has_continuous = bool(analysis.get('continuous_variables'))
    has_categorical = bool(analysis.get('categorical_variables'))
    n_records = analysis.get('n_records', 0)

    # KEY OPTIMIZATION: For very high uniqueness, skip PRAM/NOISE
    # These methods perturb values but don't reduce uniqueness
    # Only grouping methods (kANON, LOCSUPR) can help
    if reid_95 >= 0.80 or uniqueness_rate >= 0.80:
        # High uniqueness - go straight to kANON (fastest grouping method)
        return 'kANON'

    # For moderate risk, we can try perturbation methods first
    # They're faster and preserve more utility when they work

    # Case 1: Low-moderate risk with categorical - try PRAM
    if has_categorical and reid_95 < 0.50:
        return 'PRAM'

    # Case 2: Low-moderate risk with continuous - try NOISE
    if has_continuous and not has_categorical and reid_95 < 0.50:
        return 'NOISE'

    # Case 3: Moderate risk (50-80%) - kANON is safer
    if reid_95 >= 0.50:
        return 'kANON'

    # Case 4: Mixed data with low risk - PRAM first
    if has_categorical:
        return 'PRAM'

    # Default - kANON is the most reliable
    return 'kANON'


def _calculate_tabular_utility_metrics(
    original: pd.DataFrame,
    protected: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate utility metrics for tabular (frequency table) data.

    Compares cell values directly between original and protected tables.
    Handles suppressed cells (NaN) appropriately.

    Parameters:
    -----------
    original : pd.DataFrame
        Original frequency table
    protected : pd.DataFrame
        Protected frequency table (with suppressions/rounding)

    Returns:
    --------
    dict : Utility metrics including:
        - cells_preserved: Proportion of cells unchanged
        - total_preserved: Proportion of total sum preserved
        - suppression_rate: Proportion of cells suppressed (NaN)
        - mean_cell_change: Average relative change in non-suppressed cells
        - utility_score: Combined score (0-1, higher is better)
    """
    metrics = {}

    # Ensure same shape and columns
    common_cols = [c for c in original.columns if c in protected.columns]
    common_idx = original.index.intersection(protected.index)

    if len(common_cols) == 0 or len(common_idx) == 0:
        return {
            'information_loss': 1.0,
            'correlation_preserved': 0.0,
            'mean_preserved': 0.0,
            'distribution_similarity': 0.0,
            'records_suppressed': 1.0,
            'row_retention': 0.0,
            'utility_score': 0.0
        }

    orig_subset = original.loc[common_idx, common_cols]
    prot_subset = protected.loc[common_idx, common_cols]

    total_cells = orig_subset.size

    # 1. Suppression rate (NaN cells in protected)
    suppressed_cells = prot_subset.isna().sum().sum()
    suppression_rate = suppressed_cells / total_cells if total_cells > 0 else 0
    metrics['records_suppressed'] = suppression_rate

    # 2. Cells preserved exactly (excluding suppressed)
    # For rounding methods, cells may change but not be suppressed
    non_suppressed_mask = ~prot_subset.isna()
    if non_suppressed_mask.sum().sum() > 0:
        orig_non_supp = orig_subset.where(non_suppressed_mask)
        prot_non_supp = prot_subset.where(non_suppressed_mask)
        cells_exact_match = (orig_non_supp == prot_non_supp).sum().sum()
        cells_preserved = cells_exact_match / total_cells
    else:
        cells_preserved = 0
    metrics['cells_preserved'] = cells_preserved

    # 3. Total sum preservation (important for marginal totals)
    orig_total = orig_subset.sum().sum()
    prot_total = prot_subset.sum().sum()  # NaN treated as 0
    if orig_total > 0:
        total_preserved = 1 - abs(orig_total - prot_total) / orig_total
        total_preserved = max(0, total_preserved)
    else:
        total_preserved = 1.0
    metrics['total_preserved'] = total_preserved

    # 4. Mean cell change (relative) for non-suppressed cells
    cell_changes = []
    for col in common_cols:
        for idx in common_idx:
            orig_val = orig_subset.loc[idx, col]
            prot_val = prot_subset.loc[idx, col]
            if pd.notna(prot_val) and pd.notna(orig_val):
                if orig_val != 0:
                    rel_change = abs(orig_val - prot_val) / abs(orig_val)
                    cell_changes.append(min(rel_change, 1.0))
                elif prot_val == 0:
                    cell_changes.append(0)  # Both zero = perfect
                else:
                    cell_changes.append(1.0)  # orig=0, prot!=0

    mean_cell_change = np.mean(cell_changes) if cell_changes else 1.0
    metrics['mean_cell_change'] = mean_cell_change

    # 5. Information loss based on cell changes and suppressions
    # Suppression = full loss for that cell, change = partial loss
    info_loss_per_cell = suppression_rate + (1 - suppression_rate) * mean_cell_change
    metrics['information_loss'] = info_loss_per_cell

    # 6. Distribution/correlation preservation
    # For tabular data, measure how well row/column proportions are preserved
    try:
        orig_row_totals = orig_subset.sum(axis=1)
        prot_row_totals = prot_subset.sum(axis=1)
        if orig_row_totals.sum() > 0 and prot_row_totals.sum() > 0:
            orig_row_props = orig_row_totals / orig_row_totals.sum()
            prot_row_props = prot_row_totals / prot_row_totals.sum()
            row_prop_preserved = 1 - np.abs(orig_row_props - prot_row_props).mean()
        else:
            row_prop_preserved = 0
    except Exception:
        row_prop_preserved = 0.5

    metrics['correlation_preserved'] = max(0, row_prop_preserved)
    metrics['mean_preserved'] = total_preserved
    metrics['distribution_similarity'] = metrics['correlation_preserved']
    metrics['row_retention'] = 1.0  # Tables don't lose rows

    # 7. Overall utility score
    # Weight: preservation of data (non-suppressed) matters most
    metrics['utility_score'] = max(0, min(1, (
        0.30 * (1 - suppression_rate) +          # Fewer suppressions = better
        0.25 * total_preserved +                  # Preserve totals
        0.20 * (1 - mean_cell_change) +          # Small changes in values
        0.15 * metrics['correlation_preserved'] + # Preserve proportions
        0.10 * cells_preserved                    # Exact matches
    )))

    return metrics


# NOTE: calculate_utility_metrics is imported from src.metrics (line 51).
# All callers in this file use that canonical implementation.



# NOTE: calculate_utility_metrics was previously duplicated here.
# It is now imported from src.metrics (line 51) — single source of truth.


# =============================================================================
# METHOD COMPARISON
# =============================================================================

def compare_methods(
    data: pd.DataFrame,
    methods: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
    method_params: Optional[Dict[str, Dict]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple SDC methods on the same data.

    Runs each method and compares risk reduction vs utility preservation.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to protect
    methods : list, optional
        Methods to compare. If None, auto-selects based on data type.
        Options: 'kANON', 'PRAM', 'NOISE', 'LOCSUPR'
    quasi_identifiers : list, optional
        QI columns for microdata (auto-detected if not provided)
    method_params : dict, optional
        Custom parameters per method. Keys are method names, values are dicts
        of parameters to pass. E.g. {'kANON': {'k': 3}, 'NOISE': {'magnitude': 0.2}}.
        Methods not listed use defaults. A method name can appear multiple times
        in the `methods` list with different params by using suffixed keys like
        'kANON (k=3)' — the base method is extracted automatically.
    verbose : bool
        Print progress and results

    Returns:
    --------
    pd.DataFrame : Comparison table with columns:
        - method: Method name (may include param suffix)
        - success: Whether method succeeded
        - time: Execution time (seconds)
        - risk_before: Initial risk rate
        - risk_after: Final risk rate
        - risk_reduction: Percentage risk reduced
        - utility_score: Overall utility (0-1)
        - info_loss: Information loss (0-1)
        - mean_preserved: Mean preservation (0-1)
        - correlation_preserved: Correlation preservation (0-1)

    Example:
    --------
    >>> comparison = compare_methods(data, methods=['kANON', 'PRAM', 'NOISE'])
    >>> print(comparison.to_string())
    >>> best = comparison.loc[comparison['utility_score'].idxmax()]
    """
    import time

    # Auto-detect data type and select methods
    analysis = analyze_data(data, quasi_identifiers, verbose=False)
    data_type = analysis['data_type']

    if methods is None:
        if data_type == 'microdata':
            methods = ['kANON', 'PRAM', 'LOCSUPR', 'NOISE']
        elif data_type == 'continuous':
            methods = ['NOISE', 'kANON', 'LOCSUPR']
        else:
            methods = ['NOISE', 'PRAM']

    # Get QIs
    qis = quasi_identifiers or analysis.get('quasi_identifiers', [])

    if verbose:
        print(f"\n{'='*60}")
        print(f"  COMPARING {len(methods)} METHODS")
        print(f"{'='*60}")
        print(f"Data: {len(data)} rows, {len(data.columns)} columns")
        print(f"Methods: {', '.join(methods)}")

    results = []

    # Calculate initial ReID once for microdata
    reid_before = None
    if data_type == 'microdata' and qis:
        try:
            reid_before = calculate_reid(data, qis)
        except Exception:
            pass

    # Direct method function mapping (faster than apply_and_validate)
    from sdc_engine.sdc import (
        apply_kanon, apply_pram, apply_noise, apply_locsupr,
    )

    # Default parameters per method (used when method_params not specified)
    default_params = {
        'kANON': {'k': 5},
        'PRAM': {},
        'NOISE': {},
        'LOCSUPR': {'k': 5},
    }

    # Method function mapping — accepts custom params
    method_func_map = {
        'kANON': lambda d, q, p: apply_kanon(d, quasi_identifiers=q, k=p.get('k', 5), return_metadata=True, verbose=False),
        'PRAM': lambda d, q, p: apply_pram(d, p_change=p.get('p_change', 0.2), return_metadata=True, verbose=False),
        'NOISE': lambda d, q, p: apply_noise(d, magnitude=p.get('magnitude', 0.10), return_metadata=True, verbose=False),
        'LOCSUPR': lambda d, q, p: apply_locsupr(d, quasi_identifiers=q, k=p.get('k', 5), max_suppressions_per_record=p.get('max_suppressions_per_record'), return_metadata=True, verbose=False),
    }

    if method_params is None:
        method_params = {}

    # Check if LOCSUPR will be slow (large data + no R)
    from sdc_engine.sdc.LOCSUPR import _check_r_available
    if 'LOCSUPR' in [m.upper() for m in methods] and not _check_r_available() and len(data) > 1000:
        if verbose:
            print(f"\n  WARNING: LOCSUPR on {len(data)} rows without R/sdcMicro may be slow.")
            print(f"           Consider installing R + sdcMicro for 60x faster execution.")

    for method in methods:
        # Extract base method name (handle suffixed names like 'kANON (k=3)')
        import re
        base_match = re.match(r'^(\w+)', method)
        base_method = base_match.group(1).upper() if base_match else method.upper()

        # Get custom params for this method entry (check full name first, then base)
        custom_params = method_params.get(method, method_params.get(base_method, {}))

        if verbose:
            param_note = f" (params: {custom_params})" if custom_params else ""
            print(f"\n  Testing {method}{param_note}...", end=" ")

        start_time = time.time()

        try:
            protected = None
            metadata = {}

            # Special handling for GENERALIZE
            if base_method == 'GENERALIZE':
                from sdc_engine.sdc.GENERALIZE import apply_generalize
                gen_params = {**{'max_categories': 5, 'strategy': 'auto'}, **custom_params}
                protected, metadata = apply_generalize(
                    data,
                    quasi_identifiers=qis,
                    max_categories=gen_params.get('max_categories', 5),
                    strategy=gen_params.get('strategy', 'auto'),
                    return_metadata=True,
                    verbose=False
                )
            elif base_method in method_func_map:
                # Call method with custom or default params
                result_tuple = method_func_map[base_method](data.copy(), qis, custom_params)
                if isinstance(result_tuple, tuple):
                    protected, metadata = result_tuple
                else:
                    protected = result_tuple
                    metadata = {}
            else:
                # Fallback to apply_and_validate for unknown methods
                result = apply_and_validate(
                    data,
                    method=method,
                    quasi_identifiers=qis,
                    verbose=False
                )
                protected = result.get('protected_data')
                metadata = result.get('metadata', {})

            elapsed = time.time() - start_time

            if protected is not None:
                # Calculate utility metrics scoped to QI columns
                utility = calculate_utility_metrics(data, protected, columns=qis)

                # Calculate ReID after for microdata
                reid_95_before = reid_before.get('reid_95', 0) if reid_before else 0
                reid_95_after = 0
                risk_reduction = 0

                if data_type == 'microdata' and qis:
                    try:
                        reid_after = calculate_reid(protected, qis)
                        reid_95_after = reid_after.get('reid_95', 0)
                        if reid_95_before > 0:
                            risk_reduction = (reid_95_before - reid_95_after) / reid_95_before
                    except Exception:
                        pass

                # Get tabular-specific metrics from metadata
                stats = metadata.get('statistics', {}) if isinstance(metadata, dict) else {}
                total_cells = stats.get('total_cells', 0)
                total_suppressions = stats.get('total_suppressions', stats.get('sensitive_cells', 0))
                tab_suppression_rate = stats.get('suppression_rate', stats.get('sensitive_percentage', 0) / 100 if stats.get('sensitive_percentage') else 0)

                results.append({
                    'method': method,
                    'success': True,
                    'time': round(elapsed, 2),
                    # Microdata risk metrics
                    'risk_before': round(reid_95_before, 3),
                    'risk_after': round(reid_95_after, 3),
                    'risk_reduction': round(risk_reduction, 3),
                    'reid_95_before': round(reid_95_before, 3),
                    'reid_95_after': round(reid_95_after, 3),
                    # Tabular metrics
                    'total_cells': total_cells,
                    'cells_suppressed': total_suppressions,
                    'suppression_rate': round(tab_suppression_rate, 3),
                    # Utility metrics
                    'utility_score': round(utility.get('utility_score', 0), 3),
                    'info_loss': round(utility.get('information_loss', 0), 3),
                    'mean_preserved': round(utility.get('mean_preserved', 0), 3),
                    'correlation_preserved': round(utility.get('correlation_preserved', 0), 3),
                    # Store protected data for reuse when user selects this method
                    'protected_data': protected,
                    'utility_metrics': utility,
                })

                if verbose:
                    print(f"OK ({elapsed:.1f}s) - Utility: {utility.get('utility_score', 0):.1%}")
            else:
                results.append({
                    'method': method,
                    'success': False,
                    'time': round(elapsed, 2),
                    'risk_before': None, 'risk_after': None, 'risk_reduction': None,
                    'reid_95_before': None, 'reid_95_after': None,
                    'total_cells': None, 'cells_suppressed': None, 'suppression_rate': None,
                    'utility_score': None, 'info_loss': None,
                    'mean_preserved': None, 'correlation_preserved': None
                })
                if verbose:
                    print(f"FAILED")

        except Exception as e:
            elapsed = time.time() - start_time
            results.append({
                'method': method,
                'success': False,
                'time': round(elapsed, 2),
                'risk_before': None, 'risk_after': None, 'risk_reduction': None,
                'reid_95_before': None, 'reid_95_after': None,
                'total_cells': None, 'cells_suppressed': None, 'suppression_rate': None,
                'utility_score': None, 'info_loss': None,
                'mean_preserved': None, 'correlation_preserved': None
            })
            if verbose:
                print(f"ERROR: {str(e)[:50]}")

    comparison_df = pd.DataFrame(results)

    if verbose:
        print(f"\n{'='*60}")
        print("  COMPARISON RESULTS")
        print(f"{'='*60}")

        # Show successful methods sorted by utility
        successful = comparison_df[comparison_df['success'] == True].copy()
        if len(successful) > 0:
            successful = successful.sort_values('utility_score', ascending=False)

            # Check if this is tabular data (all risk_reduction are 0 or suppression_rate > 0)
            is_tabular = (
                successful['risk_reduction'].fillna(0).sum() == 0 and
                successful['suppression_rate'].fillna(0).sum() > 0
            )

            print("\nRanked by Utility Score:")
            if is_tabular:
                # Show tabular-specific columns
                print(successful[['method', 'utility_score', 'info_loss', 'suppression_rate', 'time']].to_string(index=False))
            else:
                # Show microdata columns
                print(successful[['method', 'utility_score', 'info_loss', 'risk_reduction', 'time']].to_string(index=False))

            best = successful.iloc[0]
            print(f"\n  Best Method: {best['method']}")
            print(f"    Utility Score: {best['utility_score']:.1%}")
            print(f"    Information Loss: {best['info_loss']:.1%}")
            if is_tabular:
                print(f"    Suppression Rate: {best['suppression_rate']:.1%}")
                if best.get('cells_suppressed'):
                    print(f"    Cells Suppressed: {int(best['cells_suppressed'])}")
            else:
                print(f"    Risk Reduction: {best['risk_reduction']:.1%}")

    return comparison_df


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_result(
    result: Dict,
    output_path: str,
    include_original: bool = False,
    format: str = 'auto'
) -> Dict[str, str]:
    """
    Export protection result to file(s).

    Saves protected data and metadata to specified format.

    Parameters:
    -----------
    result : dict
        Result from smart_protect() or apply_and_validate()
    output_path : str
        Output file path. Extension determines format unless 'format' is specified.
        Supported: .csv, .xlsx, .json
    include_original : bool
        If True, also save original data (for .xlsx creates separate sheet)
    format : str
        Force output format: 'csv', 'xlsx', 'json', or 'auto' (detect from extension)

    Returns:
    --------
    dict : Paths to created files
        - 'data': Path to protected data file
        - 'metadata': Path to metadata file (JSON)
        - 'original': Path to original data (if include_original=True)

    Example:
    --------
    >>> result = smart_protect(data, reid_target={'reid_95': 0.10})
    >>> files = export_result(result, 'protected_output.xlsx')
    >>> print(f"Saved to: {files['data']}")
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    base_name = output_path.stem
    parent = output_path.parent

    # Determine format
    if format == 'auto':
        ext = output_path.suffix.lower()
        if ext == '.xlsx':
            format = 'xlsx'
        elif ext == '.json':
            format = 'json'
        else:
            format = 'csv'

    protected_data = result.get('protected_data')
    if protected_data is None:
        raise ValueError("No protected data in result")

    created_files = {}

    # Build metadata
    metadata = {
        'success': result.get('success'),
        'method': result.get('method'),
        'approach': result.get('approach'),
        'total_time': result.get('total_time'),
        'protected_shape': list(result.get('protected_data').shape) if result.get('protected_data') is not None else None,
    }

    # Add risk metrics
    if result.get('risk_before'):
        metadata['risk_before'] = result['risk_before']
    if result.get('risk_after'):
        metadata['risk_after'] = result['risk_after']

    # Add ReID metrics
    if result.get('reid_before'):
        metadata['reid_before'] = result['reid_before']
    if result.get('reid_after'):
        metadata['reid_after'] = result['reid_after']

    # Add utility metrics
    if result.get('utility_metrics'):
        metadata['utility_metrics'] = result['utility_metrics']

    # Add ReID workflow info
    if result.get('reid_target_met') is not None:
        metadata['reid_target_met'] = result['reid_target_met']
        metadata['methods_tried'] = result.get('methods_tried', [])
        metadata['final_method'] = result.get('final_method')
        metadata['reid_iterations'] = result.get('reid_iterations')

    # Export based on format
    if format == 'xlsx':
        xlsx_path = parent / f"{base_name}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            protected_data.to_excel(writer, sheet_name='Protected Data', index=False)

            if include_original and 'original_data' in result:
                result['original_data'].to_excel(writer, sheet_name='Original Data', index=False)

            # Metadata sheet
            meta_df = pd.DataFrame([
                {'Key': k, 'Value': str(v)} for k, v in metadata.items()
            ])
            meta_df.to_excel(writer, sheet_name='Metadata', index=False)

        created_files['data'] = str(xlsx_path)

    elif format == 'json':
        json_path = parent / f"{base_name}.json"
        output = {
            'metadata': metadata,
            'protected_data': protected_data.to_dict(orient='records')
        }
        if include_original and 'original_data' in result:
            output['original_data'] = result['original_data'].to_dict(orient='records')

        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        created_files['data'] = str(json_path)

    else:  # CSV
        csv_path = parent / f"{base_name}.csv"
        protected_data.to_csv(csv_path, index=False)
        created_files['data'] = str(csv_path)

        # Save metadata separately
        meta_path = parent / f"{base_name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        created_files['metadata'] = str(meta_path)

        if include_original and 'original_data' in result:
            orig_path = parent / f"{base_name}_original.csv"
            result['original_data'].to_csv(orig_path, index=False)
            created_files['original'] = str(orig_path)

    return created_files


def smart_protect(
    data: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
    goal: Optional[str] = None,
    prefer_pipeline: bool = False,
    reid_target: Optional[Dict[str, float]] = None,
    max_iterations: int = 5,
    max_info_loss: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    start_k: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Smart data protection: automatically choose single method vs pipeline.

    .. deprecated::
        Use ``run_rules_engine_protection()`` from ``protection_engine.py``
        instead.  This legacy function has its own retry/escalation logic
        that diverges from the rules engine path.  It is kept for CLI
        ``--protect`` backward compatibility but should not be called from
        new code.

    This function analyzes your data and decides:
    - Use single method: For simple cases or when one method suffices
    - Use pipeline: For complex data requiring multiple protection layers

    ReID-Driven Workflow:
    When reid_target is provided, the function will auto-tune parameters
    (e.g., increase k) until ReID thresholds are met or max_iterations reached.

    Info Loss Constraint:
    When max_info_loss is provided, methods with info loss above this threshold
    are rejected even if they meet ReID targets.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to protect
    quasi_identifiers : list, optional
        QI columns for microdata
    goal : str, optional
        Specific goal (overrides auto-detection)
    prefer_pipeline : bool, default=False
        If True, prefer pipeline over single method when both are viable
    reid_target : dict, optional
        ReID thresholds to achieve. Keys can be:
        - 'reid_50': Maximum median risk (e.g., 0.10 for 10%)
        - 'reid_90': Maximum 90th percentile risk
        - 'reid_95': Maximum 95th percentile risk (most common)
        - 'reid_99': Maximum 99th percentile risk
        - 'max_risk': Maximum individual risk
        Example: {'reid_95': 0.05} means 95% of records should have risk <= 5%
    max_iterations : int, default=5
        Maximum iterations for ReID-driven tuning
    max_info_loss : float, optional
        Maximum acceptable information loss (0-1). Methods exceeding this
        are rejected even if they meet ReID targets. E.g., 0.20 means max 20% loss.
    weights : dict, optional
        Normalized weights for scoring methods when multiple candidates exist.
        Keys: 'risk_reduction', 'reid', 'utility', 'info_loss'. Values should
        sum to 1.0. If None, methods are compared by utility alone.
    verbose : bool
        Print progress and results

    Returns:
    --------
    dict : Protection results with:
        - 'approach': 'single' or 'pipeline'
        - 'protected_data': Final protected output
        - 'method' or 'pipeline': Method(s) used
        - 'success': Overall success
        - 'reid_target_met': Whether ReID targets were achieved (if reid_target provided)
        - 'reid_iterations': Number of iterations used for ReID tuning
        - Other relevant metadata

    Examples:
    ---------
    >>> # Auto-select best approach (no ReID tuning)
    >>> result = smart_protect(data)

    >>> # Force pipeline for complex data
    >>> result = smart_protect(data, prefer_pipeline=True)

    >>> # With specific goal
    >>> result = smart_protect(data, goal='k-anonymity')

    >>> # ReID-driven: auto-tune until 95% of records have risk <= 5%
    >>> result = smart_protect(data, reid_target={'reid_95': 0.05})

    >>> # Stricter: 99% of records with risk <= 3%
    >>> result = smart_protect(data, reid_target={'reid_99': 0.03})
    """
    import warnings
    warnings.warn(
        "smart_protect() is deprecated. Use run_rules_engine_protection() "
        "from protection_engine.py instead — this legacy function has its "
        "own retry/escalation logic that diverges from the rules engine.",
        DeprecationWarning, stacklevel=2,
    )

    # Step 1: Analyze data and get recommendation with pipeline hint
    analysis = analyze_data(data, quasi_identifiers, verbose=False)
    data_type = analysis.get('data_type', 'microdata')
    risk_level = analysis.get('risk_level', 'medium')

    # Track if QIs were auto-detected (for potential reduction later)
    qis_auto_detected = quasi_identifiers is None
    original_qis = None  # Will store original QIs if we reduce them

    if quasi_identifiers is None and data_type == 'microdata':
        quasi_identifiers = analysis.get('quasi_identifiers', [])

    # Step 1b: Check for very high uniqueness (>50%) - may need generalization or aggregation
    high_uniqueness_warning = None
    generalized_data = None
    generalization_metadata = None

    if data_type == 'microdata' and quasi_identifiers:
        try:
            from sdc_engine.sdc.metrics import calculate_reid
            reid_check = calculate_reid(data, quasi_identifiers)
            initial_reid_95 = reid_check.get('reid_95', 0)

            if initial_reid_95 >= 0.5:
                # Calculate QI cardinality to recommend generalization parameters
                qi_cardinality = {qi: data[qi].nunique() for qi in quasi_identifiers if qi in data.columns}
                total_combos = 1
                for card in qi_cardinality.values():
                    total_combos *= card
                max_groups = len(data) // 5  # For k=5

                # Determine if generalization can help
                needs_generalization = total_combos > max_groups * 2
                suggested_max_cats = max(3, int((max_groups ** (1/len(quasi_identifiers)))))

                high_uniqueness_warning = {
                    'reid_95': initial_reid_95,
                    'qi_cardinality': qi_cardinality,
                    'total_combinations': total_combos,
                    'max_groups_for_k5': max_groups,
                    'needs_generalization': needs_generalization,
                    'suggested_max_categories': suggested_max_cats,
                    'message': (
                        f"Very high uniqueness detected (ReID_95={initial_reid_95:.0%}). "
                        f"QI combinations ({total_combos:,}) >> max groups ({max_groups}). "
                        f"Recommend: 1) GENERALIZE to reduce cardinality, then kANON, or "
                        f"2) Use auto_aggregate() for summary statistics."
                    ),
                    'recommendation': 'generalize_then_kanon' if needs_generalization else 'aggregation'
                }

                if verbose:
                    print(f"\n[!] WARNING: {high_uniqueness_warning['message']}")
                    print(f"    QI cardinality: {qi_cardinality}")
                    print(f"    Suggested max_categories for GENERALIZE: {suggested_max_cats}")
                    print(f"    RECOMMENDATION: Use Preprocess tab to apply GENERALIZE before protection")

        except Exception:
            pass  # Continue with normal flow if ReID calculation fails

    # Get ReID-based recommendation (includes pipeline_hint for escalation)
    recommendation = recommend_method(data, goal=goal, quasi_identifiers=quasi_identifiers, verbose=False)
    pipeline_hint = recommendation.get('pipeline_hint')  # For reactive escalation

    if verbose:
        print("\n" + "=" * 60)
        print("  SMART PROTECTION")
        print("=" * 60)
        print(f"\nData Type: {data_type}")
        print(f"Risk Level: {risk_level}")
        print(f"Recommended: {recommendation.get('primary')} ({recommendation.get('rule', 'N/A')})")
        if pipeline_hint:
            print(f"Pipeline hint: {' -> '.join(pipeline_hint['pipeline'])}")
        if reid_target:
            print(f"ReID Target: {reid_target}")
            print(f"Mode: ReID-driven auto-tuning (max {max_iterations} iterations)")

    # Step 2: Decide approach based on data complexity
    use_pipeline = False
    reason = ""

    if prefer_pipeline:
        use_pipeline = True
        reason = "Pipeline preferred by user"
    elif data_type == 'microdata':
        continuous = analysis.get('continuous_variables', [])
        categorical = analysis.get('categorical_variables', [])

        # Use pipeline when:
        # 1. High risk + both variable types (need k-anonymity + perturbation)
        # 2. Both continuous AND categorical variables present
        # 3. Very high uniqueness + mixed types
        if continuous and categorical:
            # Mixed data benefits from pipeline (PRAM + NOISE or kANON + perturbation)
            if risk_level == 'high':
                use_pipeline = True
                reason = "High risk with mixed variable types"
            elif len(continuous) >= 2 and len(categorical) >= 2:
                use_pipeline = True
                reason = "Multiple continuous and categorical variables"
            else:
                reason = "Single method sufficient for data complexity"
        elif continuous and not categorical:
            # Continuous only - single method (NOISE or kANON) is sufficient
            reason = "Continuous-only data - single method sufficient"
        elif categorical and not continuous:
            # Categorical only - single method (PRAM or kANON) is sufficient
            reason = "Categorical-only data - single method sufficient"
        else:
            reason = "Single method sufficient for data complexity"
    else:
        # Tabular data - usually single method is enough
        reason = "Tabular data typically needs single method"

    if verbose:
        approach_str = "Pipeline (multi-method)" if use_pipeline else "Single method"
        print(f"Approach: {approach_str}")
        print(f"Reason: {reason}")

    # Step 3: Apply chosen approach (with optional ReID-driven tuning)
    reid_target_met = None
    reid_iterations = 0

    if reid_target and data_type == 'microdata' and quasi_identifiers:
        # ReID-driven workflow: iterate until targets met, with method fallback
        if verbose:
            print("\n--- ReID-Driven Auto-Tuning (with method fallback) ---")

        # Calculate initial ReID
        try:
            initial_reid = calculate_reid(data, quasi_identifiers)
            if verbose:
                print(f"Initial ReID95: {initial_reid.get('reid_95', 0):.1%}")

            # Check if targets already met
            targets_met, _ = _check_reid_targets(initial_reid, reid_target)
            if targets_met:
                if verbose:
                    print("ReID targets already met - minimal protection needed")
                reid_target_met = True
        except Exception as e:
            if verbose:
                print(f"Could not calculate initial ReID: {e}")
            initial_reid = None

        # Determine initial method based on goal or rule-based recommendation
        if goal in ['k-anonymity', 'k_anonymity']:
            initial_method = 'kANON'
        elif goal in ['local_suppression', 'locsupr']:
            initial_method = 'LOCSUPR'
        elif goal in ['noise', 'add_noise']:
            initial_method = 'NOISE'
        else:
            # USE RULE-BASED RECOMMENDATION from recommend_method
            # This respects the ReID-based rules in src/selection/rules.py
            recommended = recommendation.get('primary', 'kANON')
            rule_name = recommendation.get('rule', 'N/A')

            # When ReID target is set, override perturbation methods to kANON
            # because PRAM/NOISE cannot reduce uniqueness
            if recommended in ['PRAM', 'NOISE']:
                initial_method = 'kANON'
                if verbose:
                    print(f"Rule selected {recommended}, but ReID target requires structural method → kANON")
            else:
                initial_method = recommended
                if verbose:
                    print(f"Using rule-based selection: {initial_method} (rule: {rule_name})")

        # Get fallback methods
        has_categorical = bool(analysis.get('categorical_variables'))
        has_continuous = bool(analysis.get('continuous_variables'))
        fallback_methods = _get_method_fallback_order(initial_method, has_categorical, has_continuous, has_reid_target=True)

        # Build list of methods to try: initial + fallbacks
        methods_to_try = [initial_method] + fallback_methods

        if verbose:
            print(f"Methods to try: {' -> '.join(methods_to_try)}")

        # Try each method until target met
        # OPTIMIZATION: Stop if a method achieves target with good utility (>70%)
        # Don't try expensive methods (LOCSUPR) if cheaper ones work well
        MIN_GOOD_UTILITY = 0.70  # Threshold for "good enough" utility

        # Build scoring function based on weights
        def _score_method(method_result, utility_val):
            """Score a method result using user weights or pure utility."""
            if not weights:
                return utility_val
            # Weighted composite score
            w_utility = weights.get('utility', 0.25)
            w_info_loss = weights.get('info_loss', 0.25)
            w_reid = weights.get('reid', 0.25)
            w_risk_red = weights.get('risk_reduction', 0.25)

            info_loss = 0.0
            if method_result and 'utility_metrics' in method_result:
                info_loss = method_result['utility_metrics'].get('information_loss', 1.0 - utility_val)

            method_reid_val = 1.0
            if method_result:
                method_reid_val = (method_result.get('reid_after') or {}).get('reid_95', 1.0)

            risk_red_val = 0.0
            if initial_reid and method_result:
                init_q = initial_reid.get('reid_95', 0)
                if init_q > 0:
                    risk_red_val = max(0, (init_q - method_reid_val) / init_q)

            return (
                w_utility * utility_val +
                w_info_loss * (1 - info_loss) +
                w_reid * ((1 - method_reid_val) ** 1.5) +
                w_risk_red * risk_red_val
            )

        # Track best result among methods that MET the target (not failed methods)
        best_result = None
        best_method = None
        best_utility = 0.0
        best_score = -1.0
        total_iterations = 0
        methods_tried = []

        # Track best "near-miss" result (within 10% of target) for utility tradeoff
        # e.g., if target is 5%, consider methods achieving up to 5.5%
        NEAR_MISS_TOLERANCE = 0.10  # Allow 10% above target if utility is much better
        MIN_UTILITY_GAIN = 0.20  # Require at least 20% utility gain to accept near-miss
        target_reid = reid_target.get('reid_95', 0.10)
        near_miss_threshold = target_reid * (1 + NEAR_MISS_TOLERANCE)

        best_near_miss_result = None
        best_near_miss_method = None
        best_near_miss_utility = 0.0
        best_near_miss_reid = 1.0

        # Track best failed result (for fallback if no method meets target)
        best_failed_result = None
        best_failed_method = None
        best_failed_reid = 1.0

        # Use generalized data if available (for high-uniqueness cases)
        working_data = generalized_data if generalized_data is not None else data

        for method in methods_to_try:
            if verbose:
                if generalized_data is not None:
                    print(f"\n  Trying {method} on GENERALIZED data...")
                else:
                    print(f"\n  Trying {method}...")

            method_result, target_met, iters, utility = _try_method_with_tuning(
                working_data, method, quasi_identifiers, reid_target, max_iterations, verbose,
                max_info_loss=max_info_loss,
                start_k=start_k,
            )
            total_iterations += iters
            methods_tried.append(method)

            # Check if method returned no result (all iterations exceeded info loss)
            if method_result is None:
                if verbose:
                    print(f"  {method} produced no valid result (all iterations exceeded constraints)")
                continue

            # Calculate info loss from utility metrics
            method_info_loss = 0.0
            if method_result and 'utility_metrics' in method_result:
                method_info_loss = method_result['utility_metrics'].get('information_loss', 1.0 - utility)

            # Check max_info_loss constraint (double-check — tuning loop may have allowed it)
            if max_info_loss is not None and method_info_loss > max_info_loss:
                if verbose:
                    print(f"  {method} rejected: info loss {method_info_loss:.1%} > max {max_info_loss:.1%}")
                continue  # Skip this method, try next

            if target_met:
                reid_target_met = True

                # Update best if this is first success OR has better weighted score
                method_score = _score_method(method_result, utility)
                if best_result is None or method_score > best_score:
                    best_result = method_result
                    best_method = method
                    best_utility = utility
                    best_score = method_score

                if utility >= MIN_GOOD_UTILITY:
                    # Good utility - stop searching
                    if verbose:
                        print(f"\n  SUCCESS: {method} met ReID targets with {utility:.1%} utility!")
                        print(f"  Skipping remaining methods (utility is acceptable)")
                    break
                else:
                    # Low utility - try more methods to find better option
                    if verbose:
                        print(f"\n  {method} met ReID targets but utility is low ({utility:.1%})")
                        print(f"  Checking if other methods can achieve target with better utility...")
                    # Continue to next method
            else:
                # Method didn't meet target - check if it's a near-miss with good utility
                if method_result:
                    method_reid = method_result.get('reid_after', {}).get('reid_95', 1)

                    # Track as near-miss if within tolerance and better utility
                    if method_reid <= near_miss_threshold and utility > best_near_miss_utility:
                        best_near_miss_result = method_result
                        best_near_miss_method = method
                        best_near_miss_utility = utility
                        best_near_miss_reid = method_reid

                        # EARLY EXIT: If we have a near-miss with good utility, stop trying
                        # This prevents wasting time on PRAM/NOISE which can't reduce ReID
                        if utility >= MIN_GOOD_UTILITY and method in ['kANON', 'LOCSUPR']:
                            if verbose:
                                print(f"\n  NEAR-MISS: {method} achieved {method_reid:.1%} (target: {target_reid:.1%})")
                                print(f"  Utility: {utility:.1%} - stopping early (close enough)")
                            break

                    # Track as fallback if best ReID so far AND actually improved
                    initial_reid_95 = initial_reid.get('reid_95', 0) if initial_reid else 0
                    actually_improved = method_reid < initial_reid_95 * 0.95  # At least 5% reduction
                    if method_reid < best_failed_reid and actually_improved:
                        best_failed_result = method_result
                        best_failed_method = method
                        best_failed_reid = method_reid
                    elif not actually_improved and verbose:
                        print(f"  {method} skipped as fallback: ReID {method_reid:.1%} ~ initial {initial_reid_95:.1%} (no meaningful reduction)")

                if verbose:
                    print(f"  {method} did not meet target, trying next method...")

        # Decision logic for final result:
        # 1. If target was met, use best method that met target
        # 2. If target not met but near-miss has much better utility, consider using it
        # 3. Otherwise use best failed result
        if reid_target_met:
            # Check if near-miss has significantly better utility (20%+ gain)
            if best_near_miss_result and best_near_miss_utility > best_utility + MIN_UTILITY_GAIN:
                if verbose:
                    print(f"\n  Note: {best_near_miss_method} achieves {best_near_miss_reid:.1%} ReID (near target {target_reid:.1%})")
                    print(f"  with {best_near_miss_utility:.1%} utility vs {best_utility:.1%} from {best_method}")
                    print(f"  Using {best_method} to meet target exactly (use --target {best_near_miss_reid:.2f} for higher utility)")
        else:
            # No method met target
            # Check if near-miss has significantly better utility than best failed
            best_failed_utility = 0  # Failed results don't track utility well
            if best_near_miss_result and best_near_miss_utility >= MIN_GOOD_UTILITY:
                if verbose:
                    print(f"\n  No method met exact target. Using {best_near_miss_method} (near-miss)")
                    print(f"  ReID: {best_near_miss_reid:.1%} (target: {target_reid:.1%}, tolerance: {near_miss_threshold:.1%})")
                    print(f"  Utility: {best_near_miss_utility:.1%}")
                best_result = best_near_miss_result
                best_method = best_near_miss_method
                reid_target_met = True  # Mark as "effectively met" with tolerance
            elif best_failed_result is not None:
                best_result = best_failed_result
                best_method = best_failed_method

        reid_iterations = total_iterations

        # If single methods failed, try pipeline as last resort
        # Use recommended pipeline if available, otherwise fall back to generic
        if reid_target_met is None or reid_target_met is False:
            if pipeline_hint:
                pipeline_steps = pipeline_hint['pipeline']
                pipeline_params = pipeline_hint.get('parameters', {})
                if verbose:
                    print(f"\n  Single methods failed. Trying recommended pipeline: {' -> '.join(pipeline_steps)}")
                    print(f"  Reason: {pipeline_hint.get('reason', 'Data-driven recommendation')}")
            else:
                pipeline_steps = None  # Will use generic approach
                pipeline_params = {}
                if verbose:
                    print(f"\n  Single methods failed to meet target. Trying generic PIPELINE...")

            try:
                # Apply pipeline with early stopping after each step
                # Use generalized data if available
                current_data = generalized_data.copy() if generalized_data is not None else data.copy()
                pipeline_methods_applied = []
                pipeline_target_met = False

                if pipeline_steps:
                    # Use recommended pipeline with early stopping
                    for step_idx, method in enumerate(pipeline_steps):
                        if verbose:
                            print(f"\n    Step {step_idx + 1}/{len(pipeline_steps)}: {method}")

                        # Get parameters for this method
                        method_params = pipeline_params.get(method, {})
                        if 'quasi_identifiers' not in method_params and method in ['kANON', 'LOCSUPR']:
                            method_params['quasi_identifiers'] = quasi_identifiers

                        # Apply the method
                        try:
                            step_result = apply_and_validate(current_data, method=method, verbose=False, **method_params)
                            if step_result.get('success') and step_result.get('protected_data') is not None:
                                current_data = step_result['protected_data']
                                pipeline_methods_applied.append(method)

                                # Check if target is met after this step (early stopping)
                                try:
                                    step_qis = [qi for qi in quasi_identifiers if qi in current_data.columns]
                                    if step_qis:
                                        step_reid = calculate_reid(current_data, step_qis)
                                        targets_met, _ = _check_reid_targets(step_reid, reid_target)

                                        if verbose:
                                            print(f"      ReID_95 after {method}: {step_reid.get('reid_95', 0):.1%}")

                                        if targets_met:
                                            if verbose:
                                                print(f"      Target met! Stopping pipeline early.")
                                            pipeline_target_met = True
                                            break
                                except Exception:
                                    pass
                        except Exception as e:
                            if verbose:
                                print(f"      {method} failed: {e}")

                    # Build pipeline result
                    if pipeline_methods_applied:
                        pipeline_result = {
                            'success': True,
                            'protected_data': current_data,
                            'pipeline': pipeline_methods_applied,
                            'early_stopped': pipeline_target_met and len(pipeline_methods_applied) < len(pipeline_steps)
                        }
                    else:
                        pipeline_result = {'success': False}

                else:
                    # Fallback to generic pipeline
                    pipeline_result = apply_pipeline(
                        data.copy(),
                        goals=['strong_privacy'],
                        quasi_identifiers=quasi_identifiers,
                        verbose=False
                    )

                if pipeline_result.get('success'):
                    protected = pipeline_result.get('protected_data')
                    if protected is not None:
                        try:
                            protected_qis = [qi for qi in quasi_identifiers if qi in protected.columns]
                            pipeline_reid = calculate_reid(protected, protected_qis)
                            pipeline_result['reid_after'] = pipeline_reid

                            targets_met, target_results = _check_reid_targets(pipeline_reid, reid_target)

                            if verbose:
                                for metric, met in target_results.items():
                                    actual = pipeline_reid.get(metric, 0)
                                    target_val = reid_target.get(metric, 0)
                                    status = "[OK]" if met else "[MISS]"
                                    print(f"      {status} {metric}: {actual:.1%} (target: {target_val:.1%})")

                            if targets_met:
                                # Check info loss constraint on pipeline result
                                pipeline_utility = calculate_utility_metrics(data, protected, columns=quasi_identifiers)
                                pipeline_info_loss = pipeline_utility.get('information_loss', 1.0 - pipeline_utility.get('utility_score', 0)) if isinstance(pipeline_utility, dict) else 1.0

                                if max_info_loss is not None and pipeline_info_loss > max_info_loss:
                                    if verbose:
                                        print(f"\n  PIPELINE met ReID but info loss {pipeline_info_loss:.1%} > max {max_info_loss:.1%} -- rejected")
                                        print(f"  Preprocessing (GENERALIZE) is needed to reduce QI cardinality first")
                                    methods_tried.append('PIPELINE (info_loss)')
                                else:
                                    reid_target_met = True
                                    best_result = pipeline_result
                                    best_method = 'PIPELINE'
                                    methods_tried.append('PIPELINE')
                                    if verbose:
                                        early_note = " (early stop)" if pipeline_result.get('early_stopped') else ""
                                        print(f"\n  SUCCESS: PIPELINE met ReID targets!{early_note}")
                            else:
                                # Check if pipeline result is better than single method results
                                pipeline_reid_95 = pipeline_reid.get('reid_95', 1)
                                _reid_val = best_result.get('reid_after', 1) if best_result else 1
                                best_reid_95 = _reid_val.get('reid_95', 1) if isinstance(_reid_val, dict) else _reid_val
                                if pipeline_reid_95 < best_reid_95:
                                    best_result = pipeline_result
                                    best_method = 'PIPELINE'
                                methods_tried.append('PIPELINE')
                                if verbose:
                                    print(f"  PIPELINE improved ReID but did not meet target")

                        except Exception as e:
                            if verbose:
                                print(f"      Could not calculate pipeline ReID: {e}")
                            methods_tried.append('PIPELINE')

            except Exception as e:
                if verbose:
                    print(f"  Pipeline failed: {e}")
                methods_tried.append('PIPELINE (failed)')

        if reid_target_met is None:
            reid_target_met = False
            if verbose:
                print(f"\nReID targets NOT met after trying {len(methods_tried)} methods (including pipeline)")
                print(f"Methods tried: {' -> '.join(methods_tried)}")
                print("Using best result achieved")

        # Use best_result if available, otherwise create a failure result
        if best_result:
            result = best_result
        else:
            result = {
                'original_data': data,
                'protected_data': data,
                'method': initial_method,
                'success': False,
                'validation': {'error': 'All ReID tuning methods failed'}
            }

        result['approach'] = 'reid_tuned'
        result['reason'] = f"ReID-driven with method selection (pipeline fallback)"
        result['reid_target'] = reid_target
        result['reid_target_met'] = reid_target_met
        result['reid_iterations'] = reid_iterations
        result['methods_tried'] = methods_tried
        result['final_method'] = best_method or initial_method

        # Calculate utility metrics
        if result.get('protected_data') is not None:
            try:
                result['utility_metrics'] = calculate_utility_metrics(
                    data, result['protected_data'], columns=quasi_identifiers
                )
            except Exception as e:
                result['utility_metrics'] = {'error': str(e)}

        # Set info_loss flags for UI (auto-fix button, warnings)
        if max_info_loss is not None and isinstance(result.get('utility_metrics'), dict):
            actual_loss = result['utility_metrics'].get('information_loss', 0)
            result['actual_info_loss'] = actual_loss
            if actual_loss > max_info_loss:
                result['info_loss_exceeded'] = True
                result['max_info_loss'] = max_info_loss

        # Ensure final ReID95 is present for downstream reporting and UI
        final_reid_95 = result.get('final_reid_95')
        if final_reid_95 is None:
            # try candidate locations
            if isinstance(result.get('final_reid'), dict):
                final_reid_95 = result['final_reid'].get('reid_95')
            if final_reid_95 is None and isinstance(result.get('reid_after'), dict):
                final_reid_95 = result['reid_after'].get('reid_95')
            # last resort: calculate from protected_data if possible
            if final_reid_95 is None and result.get('protected_data') is not None and quasi_identifiers:
                try:
                    from sdc_engine.sdc.sdc_utils import calculate_reid
                    reid_calc = calculate_reid(result['protected_data'], quasi_identifiers)
                    final_reid_95 = reid_calc.get('reid_95')
                except Exception:
                    final_reid_95 = None

        result['final_reid_95'] = final_reid_95

    elif use_pipeline:
        # Standard pipeline workflow (no ReID tuning)
        # Use generalized data if available
        working_data = generalized_data if generalized_data is not None else data

        pipeline_goals = None
        if goal:
            goal_mapping = {
                'k-anonymity': ['k_anonymity'],
                'k_anonymity': ['k_anonymity'],
                'privacy': ['strong_privacy'],
                'strong_privacy': ['strong_privacy'],
                'perturbation': ['perturbation'],
            }
            pipeline_goals = goal_mapping.get(goal, [goal])

        result = apply_pipeline(
            working_data,
            goals=pipeline_goals,
            quasi_identifiers=quasi_identifiers,
            verbose=verbose
        )
        result['approach'] = 'pipeline'
        result['reason'] = reason
        if generalized_data is not None:
            result['pre_generalized'] = True
            result['generalization_metadata'] = generalization_metadata

        # Calculate utility metrics
        if result.get('protected_data') is not None:
            try:
                result['utility_metrics'] = calculate_utility_metrics(
                    data, result['protected_data'], columns=quasi_identifiers
                )
            except Exception as e:
                result['utility_metrics'] = {'error': str(e)}

        # Check max_info_loss constraint for pipeline
        if max_info_loss is not None:
            actual_loss = result.get('utility_metrics', {}).get('information_loss', 0)
            if actual_loss > max_info_loss:
                result['info_loss_exceeded'] = True
                result['actual_info_loss'] = actual_loss
                result['max_info_loss'] = max_info_loss
                if verbose:
                    print(f"\nWARNING: Pipeline info loss {actual_loss:.1%} exceeds max {max_info_loss:.1%}")

    else:
        # Standard single method workflow (no ReID tuning)
        # Use generalized data if available
        working_data = generalized_data if generalized_data is not None else data

        result = apply_and_validate(
            working_data,
            goal=goal,
            quasi_identifiers=quasi_identifiers,
            verbose=verbose
        )
        result['approach'] = 'single'
        result['reason'] = reason
        if generalized_data is not None:
            result['pre_generalized'] = True
            result['generalization_metadata'] = generalization_metadata

        # Calculate utility metrics
        if result.get('protected_data') is not None:
            try:
                result['utility_metrics'] = calculate_utility_metrics(
                    data, result['protected_data'], columns=quasi_identifiers
                )
            except Exception as e:
                result['utility_metrics'] = {'error': str(e)}

        # Check max_info_loss constraint for single method
        if max_info_loss is not None:
            actual_loss = result.get('utility_metrics', {}).get('information_loss', 0)
            if actual_loss > max_info_loss:
                result['info_loss_exceeded'] = True
                result['actual_info_loss'] = actual_loss
                result['max_info_loss'] = max_info_loss
                if verbose:
                    print(f"\nWARNING: Info loss {actual_loss:.1%} exceeds max {max_info_loss:.1%}")
                    print(f"Consider using a different method or relaxing the constraint.")

    # =========================================================================
    # QI REDUCTION: If suppression is too high and QIs were auto-detected,
    # try again with fewer QIs to improve utility
    # =========================================================================
    MAX_ACCEPTABLE_SUPPRESSION = 0.50  # 50% suppression threshold
    MIN_QIS_FOR_REDUCTION = 3  # Only reduce if we have at least 3 QIs

    um = result.get('utility_metrics', {})
    suppression_rate = um.get('records_suppressed', 0)

    if (qis_auto_detected and
        suppression_rate > MAX_ACCEPTABLE_SUPPRESSION and
        len(quasi_identifiers) >= MIN_QIS_FOR_REDUCTION and
        result.get('reid_target_met', False)):  # Only if target was met (otherwise no point)

        if verbose:
            print(f"\n--- QI Reduction (suppression {suppression_rate:.1%} > {MAX_ACCEPTABLE_SUPPRESSION:.0%}) ---")
            print(f"Original QIs ({len(quasi_identifiers)}): {quasi_identifiers}")

        # Save original result in case reduction fails
        original_result = result
        original_qis = quasi_identifiers.copy()

        # Try progressively fewer QIs until suppression is acceptable or we have 2 QIs left
        # Track best fallback in case we can't meet target with fewer QIs
        best_fallback = None
        best_fallback_qis = None
        best_fallback_suppression = 1.0

        for num_qis in range(len(quasi_identifiers) - 1, 1, -1):
            # Select subset of QIs (prioritize by uniqueness contribution)
            # For simplicity, just take the first N QIs (could be smarter later)
            reduced_qis = quasi_identifiers[:num_qis]

            if verbose:
                print(f"  Trying {num_qis} QIs: {reduced_qis}")

            # Retry protection with fewer QIs
            try:
                retry_result = smart_protect(
                    data,
                    quasi_identifiers=reduced_qis,
                    goal=goal,
                    prefer_pipeline=prefer_pipeline,
                    reid_target=reid_target,
                    max_iterations=max_iterations,
                    verbose=False  # Don't print nested output
                )

                retry_um = retry_result.get('utility_metrics', {})
                retry_suppression = retry_um.get('records_suppressed', 0)
                retry_target_met = retry_result.get('reid_target_met', False)

                if verbose:
                    print(f"    Suppression: {retry_suppression:.1%}, ReID target met: {retry_target_met}")

                # Accept if suppression is acceptable AND target is still met
                if retry_suppression <= MAX_ACCEPTABLE_SUPPRESSION and retry_target_met:
                    if verbose:
                        print(f"  SUCCESS: Using {num_qis} QIs with {retry_suppression:.1%} suppression")
                    result = retry_result
                    result['qi_reduced'] = True
                    result['original_qis'] = original_qis
                    result['reduced_qis'] = reduced_qis
                    quasi_identifiers = reduced_qis
                    break

                # Track best fallback (lowest suppression even if target not met)
                if retry_suppression < best_fallback_suppression:
                    best_fallback = retry_result
                    best_fallback_qis = reduced_qis
                    best_fallback_suppression = retry_suppression

                # If target not met anymore, keep looking for better suppression
                if not retry_target_met:
                    if verbose:
                        print(f"    ReID target not met with {num_qis} QIs, continuing search...")
                    continue

            except Exception as e:
                if verbose:
                    print(f"    Error with {num_qis} QIs: {e}")
                continue

        else:
            # Loop completed without finding acceptable solution that meets target
            # If original has excessive suppression (>80%), use best fallback instead
            EXCESSIVE_SUPPRESSION = 0.80
            if suppression_rate > EXCESSIVE_SUPPRESSION and best_fallback is not None:
                if verbose:
                    print(f"  WARNING: ReID target cannot be met with fewer QIs")
                    print(f"  Original suppression ({suppression_rate:.1%}) is excessive (>{EXCESSIVE_SUPPRESSION:.0%})")
                    print(f"  Using fallback with {len(best_fallback_qis)} QIs and {best_fallback_suppression:.1%} suppression")
                    print(f"  Note: ReID target NOT fully met, but data is usable")
                result = best_fallback
                result['qi_reduced'] = True
                result['original_qis'] = original_qis
                result['reduced_qis'] = best_fallback_qis
                result['reid_target_met'] = False  # Mark that target wasn't met
                result['fallback_used'] = True
                quasi_identifiers = best_fallback_qis
            else:
                if verbose:
                    print(f"  Could not reduce suppression below {MAX_ACCEPTABLE_SUPPRESSION:.0%} while meeting ReID target")
                    print(f"  Using original result with {len(original_qis)} QIs")

    if verbose:
        print("\n" + "=" * 60)
        print("  SMART PROTECTION COMPLETE")
        print("=" * 60)
        print(f"Approach used: {result['approach']}")
        print(f"Success: {result.get('success', False)}")
        if result.get('qi_reduced'):
            print(f"QI Reduction: {len(result.get('original_qis', []))} -> {len(result.get('reduced_qis', []))} QIs")
        if reid_target:
            print(f"ReID target met: {result.get('reid_target_met', 'N/A')}")
            print(f"Iterations used: {result.get('reid_iterations', 0)}")

        # Display utility metrics if available
        um = result.get('utility_metrics', {})
        if um and 'error' not in um:
            print(f"\nUtility Metrics:")
            print(f"  Utility Score: {um.get('utility_score', 0):.1%}")
            print(f"  Information Loss: {um.get('information_loss', 0):.1%}")
            print(f"  Mean Preserved: {um.get('mean_preserved', 0):.1%}")
            print(f"  Correlation Preserved: {um.get('correlation_preserved', 0):.1%}")
            if um.get('generalization_rate', 0) > 0:
                print(f"  Generalization Rate: {um.get('generalization_rate', 0):.1%}")
            if um.get('records_suppressed', 0) > 0:
                print(f"  Records Suppressed: {um.get('records_suppressed', 0):.1%}")

    # Add high-uniqueness warning to result if detected
    if high_uniqueness_warning:
        result['high_uniqueness_warning'] = high_uniqueness_warning
        result['preprocessing_recommended'] = True
        result['preprocessing_suggestion'] = (
            f"Data has very high uniqueness (ReID_95={high_uniqueness_warning['reid_95']:.0%}). "
            f"Consider using the Preprocess tab to apply GENERALIZE (max_categories={high_uniqueness_warning['suggested_max_categories']}) "
            f"before protection to reduce QI cardinality."
        )

    return result


# =============================================================================
# SMART ANONYMIZE - INTEGRATED QI PREPROCESSING + ITERATIVE OPTIMIZATION
# =============================================================================

def smart_anonymize(
    data: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
    sensitive_vars: Optional[List[str]] = None,
    exclude_vars: Optional[List[str]] = None,
    access_tier: str = 'SCIENTIFIC',
    hierarchies: Optional[Dict] = None,
    reid_target: Optional[float] = None,
    utility_target: Optional[float] = None,
    max_iterations: int = 5,
    auto_detect: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Smart anonymization with QI preprocessing and iterative method optimization.

    This function:
    1. Auto-detects or uses user-defined column classifications
    2. Removes direct identifiers and sensitive vars from QIs
    3. Preprocesses high-cardinality QIs (aggregation/binning)
    4. Uses rule-based selection as starting point
    5. Iteratively tries methods until ReID + utility targets are met

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to anonymize
    quasi_identifiers : list, optional
        User-defined list of quasi-identifier columns.
        If None and auto_detect=True, will auto-detect.
    sensitive_vars : list, optional
        User-defined sensitive columns (kept in output but not used for k-anonymity)
    exclude_vars : list, optional
        User-defined columns to exclude from output entirely
    access_tier : str
        Target access tier: 'PUBLIC', 'SCIENTIFIC', or 'SECURE'
        - PUBLIC: ReID_95 <= 5%, utility >= 70%
        - SCIENTIFIC: ReID_95 <= 10%, utility >= 80%
        - SECURE: ReID_95 <= 20%, utility >= 90%
    hierarchies : dict, optional
        User-defined preprocessing rules for high-cardinality QIs:
        {
            'column_name': {
                'strategy': 'hierarchy' | 'top_k' | 'binning' | 'exclude' | 'reclassify',
                'mapping': dict or callable,  # for hierarchy
                'target': str,  # new column name
                'k': int,  # for top_k
                'bins': int,  # for binning
            }
        }
    reid_target : float, optional
        Target ReID_95 (overrides access_tier default)
    utility_target : float, optional
        Target utility score (overrides access_tier default)
    max_iterations : int
        Maximum method iterations to try
    auto_detect : bool
        If True and quasi_identifiers not provided, auto-detect columns
    verbose : bool
        Print progress

    Returns:
    --------
    dict : {
        'success': bool,
        'protected_data': pd.DataFrame,
        'preprocessing': PreprocessingPlan,
        'method_used': str,
        'reid_before': float,
        'reid_after': float,
        'utility': float,
        'iterations_tried': list,
        'warnings': list
    }

    Example:
    --------
    >>> result = smart_anonymize(
    ...     data,
    ...     quasi_identifiers=['age', 'municipality', 'occupation'],
    ...     access_tier='PUBLIC',
    ...     hierarchies={
    ...         'municipality': {
    ...             'strategy': 'hierarchy',
    ...             'mapping': muni_to_region,
    ...             'target': 'region'
    ...         },
    ...         'occupation': {
    ...             'strategy': 'top_k',
    ...             'k': 30
    ...         }
    ...     }
    ... )
    >>> if result['success']:
    ...     protected = result['protected_data']
    """
    if not HAS_PREPROCESSING:
        raise ImportError("QI preprocessing module not available. Check src/preprocessing/")

    # Set targets based on access tier
    tier_defaults = {
        'PUBLIC': {'reid': 0.05, 'utility': 0.70},
        'SCIENTIFIC': {'reid': 0.10, 'utility': 0.80},
        'SECURE': {'reid': 0.20, 'utility': 0.90},
    }

    tier_config = tier_defaults.get(access_tier.upper(), tier_defaults['SCIENTIFIC'])
    reid_target = reid_target or tier_config['reid']
    utility_target = utility_target or tier_config['utility']

    result = {
        'success': False,
        'protected_data': None,
        'preprocessing': None,
        'method_used': None,
        'parameters_used': None,
        'reid_before': None,
        'reid_after': None,
        'utility': None,
        'iterations_tried': [],
        'warnings': [],
        'access_tier': access_tier,
        'reid_target': reid_target,
        'utility_target': utility_target,
    }

    # ==========================================================================
    # STEP 0: Handle column classifications (user-defined or auto-detect)
    # ==========================================================================
    working_data = data.copy()
    warnings_list = []

    # Remove excluded columns first
    if exclude_vars:
        cols_to_remove = [c for c in exclude_vars if c in working_data.columns]
        if cols_to_remove:
            working_data = working_data.drop(columns=cols_to_remove)
            if verbose:
                print(f"\n  Excluded columns: {cols_to_remove}")

    # Auto-detect or use user-defined classifications
    if quasi_identifiers is None:
        if auto_detect:
            # Use QIHandler to auto-detect column classifications
            handler = QIHandler(access_tier=access_tier)
            classification = handler.suggest_column_classification(working_data, verbose=verbose)

            quasi_identifiers = classification.get('quasi_identifiers', [])

            # Merge auto-detected sensitive with user-defined
            auto_sensitive = classification.get('sensitive', [])
            if sensitive_vars:
                sensitive_vars = list(set(sensitive_vars + auto_sensitive))
            else:
                sensitive_vars = auto_sensitive

            # Warn about direct identifiers
            direct_ids = classification.get('direct_identifiers', [])
            if direct_ids:
                warnings_list.append(f"Direct identifiers detected (should remove): {direct_ids}")
                # Auto-remove direct identifiers from working data
                working_data = working_data.drop(columns=[c for c in direct_ids if c in working_data.columns])
                if verbose:
                    print(f"\n  [!] Auto-removed direct identifiers: {direct_ids}")

            if verbose:
                print(f"\n  Auto-detected QIs: {quasi_identifiers}")
                print(f"  Auto-detected sensitive: {sensitive_vars}")
        else:
            # No QIs provided and no auto-detect - use all columns except sensitive
            all_cols = list(working_data.columns)
            if sensitive_vars:
                quasi_identifiers = [c for c in all_cols if c not in sensitive_vars]
            else:
                quasi_identifiers = all_cols
            warnings_list.append("No QIs specified - using all non-sensitive columns as QIs")

    # Remove sensitive vars from QIs (they should be protected separately)
    if sensitive_vars:
        quasi_identifiers = [qi for qi in quasi_identifiers if qi not in sensitive_vars]

    # Validate QIs exist in data
    quasi_identifiers = [qi for qi in quasi_identifiers if qi in working_data.columns]

    if not quasi_identifiers:
        result['warnings'].append("No valid quasi-identifiers found")
        result['warnings'].extend(warnings_list)
        return result

    if verbose:
        print("\n" + "=" * 70)
        print(f"  SMART ANONYMIZE - {access_tier} ACCESS TIER")
        print("=" * 70)
        print(f"  Targets: ReID_95 <= {reid_target:.0%}, Utility >= {utility_target:.0%}")
        print(f"  Dataset: {len(working_data)} records, {len(quasi_identifiers)} QIs")
        if sensitive_vars:
            print(f"  Sensitive vars (preserved): {sensitive_vars}")

    # Step 1: Preprocess high-cardinality QIs
    if verbose:
        print("\n" + "-" * 70)
        print("  STEP 1: QI PREPROCESSING")
        print("-" * 70)

    processed_data, processed_qis, preprocessing_plan = preprocess_for_anonymization(
        working_data,
        quasi_identifiers,
        access_tier=access_tier,
        hierarchies=hierarchies,
        verbose=verbose
    )

    # Store classification info in result
    result['column_classification'] = {
        'quasi_identifiers': quasi_identifiers,
        'sensitive_vars': sensitive_vars,
        'exclude_vars': exclude_vars,
        'auto_detected': auto_detect and quasi_identifiers is not None,
    }
    result['warnings'].extend(warnings_list)

    result['preprocessing'] = preprocessing_plan
    result['processed_qis'] = processed_qis

    # Check if preprocessing made anonymization feasible
    if preprocessing_plan.feasibility_after['feasibility'] == 'infeasible':
        result['warnings'].append(
            f"K-anonymity still infeasible after preprocessing. "
            f"Expected EQ size: {preprocessing_plan.feasibility_after['expected_eq_size']:.2f}"
        )
        if verbose:
            print(f"\n  [!] WARNING: Still infeasible after preprocessing!")

    # Step 2: Calculate baseline ReID
    try:
        reid_before = calculate_reid(processed_data, processed_qis)
        result['reid_before'] = reid_before['reid_95']
        if verbose:
            print(f"\n  Baseline ReID_95: {reid_before['reid_95']:.1%}")
    except Exception as e:
        result['warnings'].append(f"Could not calculate baseline ReID: {e}")
        reid_before = {'reid_95': 1.0}

    # Step 3: Get initial method recommendation
    if verbose:
        print("\n" + "-" * 70)
        print("  STEP 2: METHOD SELECTION & OPTIMIZATION")
        print("-" * 70)

    analysis = analyze_data(processed_data, processed_qis, verbose=False)
    recommendation = recommend_method(
        processed_data,
        quasi_identifiers=processed_qis,
        verbose=False
    )

    # Build method priority list
    methods_to_try = _build_method_priority_list(
        recommendation,
        preprocessing_plan,
        processed_data,
        processed_qis
    )

    if verbose:
        print(f"\n  Initial recommendation: {recommendation['primary']}")
        print(f"  Methods to try: {[m[0] for m in methods_to_try[:5]]}")

    # Step 4: Iterate through methods until targets met
    best_result = None
    best_score = -float('inf')

    for iteration, (method, params) in enumerate(methods_to_try[:max_iterations]):
        if verbose:
            print(f"\n  Iteration {iteration + 1}: Trying {method}...")

        try:
            # Apply method
            apply_func = METHOD_FUNCTIONS.get(method)
            if not apply_func:
                continue

            # Prepare parameters
            method_params = params.copy()
            if method in ['kANON', 'LOCSUPR']:
                method_params['quasi_identifiers'] = processed_qis

            protected = apply_func(processed_data, **method_params)

            # Evaluate results
            reid_after = calculate_reid(protected, processed_qis)

            # Calculate utility
            if HAS_UTILITY_METRICS:
                utility_metrics = calculate_utility_metrics(processed_data, protected, columns=processed_qis)
                utility_score = utility_metrics.get('utility_score', utility_metrics.get('overall_utility', 0))
            else:
                # Simple utility estimate
                suppressed = protected[processed_qis].isna().sum().sum()
                total_cells = len(protected) * len(processed_qis)
                utility_score = 1 - (suppressed / total_cells) if total_cells > 0 else 0

            iteration_result = {
                'method': method,
                'params': method_params,
                'reid_95': reid_after['reid_95'],
                'utility': utility_score,
                'meets_reid': reid_after['reid_95'] <= reid_target,
                'meets_utility': utility_score >= utility_target,
                'protected_data': protected,
            }

            result['iterations_tried'].append({
                'method': method,
                'reid_95': reid_after['reid_95'],
                'utility': utility_score,
                'meets_targets': iteration_result['meets_reid'] and iteration_result['meets_utility']
            })

            if verbose:
                status = "[OK]" if iteration_result['meets_reid'] and iteration_result['meets_utility'] else "[X]"
                print(f"    {status} ReID: {reid_after['reid_95']:.1%}, Utility: {utility_score:.1%}")

            # Check if targets met
            if iteration_result['meets_reid'] and iteration_result['meets_utility']:
                result['success'] = True
                result['protected_data'] = protected
                result['method_used'] = method
                result['parameters_used'] = method_params
                result['reid_after'] = reid_after['reid_95']
                result['utility'] = utility_score

                if verbose:
                    print(f"\n  [OK] SUCCESS! Targets met with {method}")

                return result

            # Track best result even if targets not fully met
            # Score: weighted combination favoring ReID compliance
            score = (1 - reid_after['reid_95']) * 0.6 + utility_score * 0.4
            if iteration_result['meets_reid']:
                score += 0.5  # Bonus for meeting ReID
            if iteration_result['meets_utility']:
                score += 0.3  # Bonus for meeting utility

            if score > best_score:
                best_score = score
                best_result = iteration_result

        except Exception as e:
            if verbose:
                print(f"    [X] Error: {e}")
            result['iterations_tried'].append({
                'method': method,
                'error': str(e)
            })

    # If no method met all targets, return best result
    if best_result:
        result['protected_data'] = best_result['protected_data']
        result['method_used'] = best_result['method']
        result['parameters_used'] = best_result['params']
        result['reid_after'] = best_result['reid_95']
        result['utility'] = best_result['utility']

        if best_result['meets_reid']:
            result['warnings'].append(
                f"ReID target met but utility ({best_result['utility']:.1%}) below target ({utility_target:.0%})"
            )
        elif best_result['meets_utility']:
            result['warnings'].append(
                f"Utility target met but ReID ({best_result['reid_95']:.1%}) above target ({reid_target:.0%})"
            )
        else:
            result['warnings'].append(
                f"Neither target fully met. Best: ReID={best_result['reid_95']:.1%}, Utility={best_result['utility']:.1%}"
            )

    if verbose:
        print("\n" + "-" * 70)
        print("  RESULT SUMMARY")
        print("-" * 70)
        if result['success']:
            print(f"  [OK] SUCCESS with {result['method_used']}")
        else:
            print(f"  [X] Targets not fully met. Best: {result['method_used']}")
        print(f"  ReID: {result['reid_before']:.1%} -> {result['reid_after']:.1%} (target: {reid_target:.0%})")
        print(f"  Utility: {result['utility']:.1%} (target: {utility_target:.0%})")
        if result['warnings']:
            print(f"\n  Warnings:")
            for w in result['warnings']:
                print(f"    - {w}")

    return result


def _build_method_priority_list(
    recommendation: Dict,
    preprocessing_plan,
    data: pd.DataFrame,
    qis: List[str]
) -> List[Tuple[str, Dict]]:
    """Build prioritized list of methods to try based on recommendation and data characteristics."""

    methods = []

    # Start with primary recommendation
    primary = recommendation.get('primary', 'kANON')
    primary_params = recommendation.get('parameters', {})
    methods.append((primary, primary_params))

    # Check feasibility
    feasibility = preprocessing_plan.feasibility_after['feasibility']
    expected_eq = preprocessing_plan.feasibility_after.get('expected_eq_size', 0)

    # Add structural methods if needed
    if feasibility in ['easy', 'moderate']:
        if primary != 'kANON':
            methods.append(('kANON', {'k': 5}))
        methods.append(('kANON', {'k': 3}))
        methods.append(('kANON', {'k': 7}))
        methods.append(('kANON', {'k': 10}))
        methods.append(('LOCSUPR', {'k': 3}))
        methods.append(('LOCSUPR', {'k': 5}))

    elif feasibility == 'hard':
        # Aggressive structural methods for hard cases
        methods.append(('kANON', {'k': 3}))
        methods.append(('LOCSUPR', {'k': 3}))
        methods.append(('kANON', {'k': 5}))

    # Add alternatives from recommendation
    for alt in recommendation.get('alternatives', []):
        if alt not in [m[0] for m in methods]:
            methods.append((alt, {}))

    # Add perturbation methods for already-low-risk data
    baseline_reid = recommendation.get('analysis', {}).get('reid_95', 1.0)
    if baseline_reid < 0.10:
        methods.extend([
            ('PRAM', {'p_change': 0.15}),
            ('NOISE', {'magnitude': 0.10}),
        ])

    return methods


def smart_anonymize_with_classification(
    data: pd.DataFrame,
    column_classification: Dict,
    access_tier: str = 'SCIENTIFIC',
    hierarchies: Optional[Dict] = None,
    reid_target: Optional[float] = None,
    utility_target: Optional[float] = None,
    max_iterations: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Smart anonymization using a user-provided column classification dictionary.

    This is a convenience wrapper around smart_anonymize() that accepts column
    classifications in a single dictionary format.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to anonymize
    column_classification : dict
        Dictionary specifying column types:
        {
            'quasi_identifiers': ['age', 'region', 'occupation'],  # Required
            'sensitive': ['salary', 'health_status'],              # Optional
            'exclude': ['id', 'name', 'ssn'],                      # Optional
            'direct_identifiers': ['name', 'email'],               # Optional (will be removed)
        }
    access_tier : str
        Target access tier: 'PUBLIC', 'SCIENTIFIC', or 'SECURE'
    hierarchies : dict, optional
        Preprocessing rules for high-cardinality QIs
    reid_target : float, optional
        Target ReID_95 (overrides access_tier default)
    utility_target : float, optional
        Target utility score (overrides access_tier default)
    max_iterations : int
        Maximum method iterations to try
    verbose : bool
        Print progress

    Returns:
    --------
    dict : Same as smart_anonymize()

    Example:
    --------
    >>> classification = {
    ...     'quasi_identifiers': ['age', 'region', 'occupation'],
    ...     'sensitive': ['salary'],
    ...     'exclude': ['id', 'full_name']
    ... }
    >>> result = smart_anonymize_with_classification(
    ...     data,
    ...     column_classification=classification,
    ...     access_tier='PUBLIC'
    ... )
    """
    # Extract from classification dict
    quasi_identifiers = column_classification.get('quasi_identifiers', [])
    sensitive_vars = column_classification.get('sensitive', column_classification.get('sensitive_vars', []))
    exclude_vars = column_classification.get('exclude', column_classification.get('exclude_vars', []))
    direct_ids = column_classification.get('direct_identifiers', [])

    # Combine exclude and direct_identifiers
    all_exclude = list(set(exclude_vars + direct_ids))

    return smart_anonymize(
        data=data,
        quasi_identifiers=quasi_identifiers if quasi_identifiers else None,
        sensitive_vars=sensitive_vars if sensitive_vars else None,
        exclude_vars=all_exclude if all_exclude else None,
        access_tier=access_tier,
        hierarchies=hierarchies,
        reid_target=reid_target,
        utility_target=utility_target,
        max_iterations=max_iterations,
        auto_detect=not quasi_identifiers,  # Auto-detect if no QIs provided
        verbose=verbose
    )


def print_method_comparison():
    """Print comparison table of all SDC methods."""
    print("\n" + "=" * 80)
    print("  SDC METHODS COMPARISON")
    print("=" * 80)

    print("\n--- Available Methods ---\n")
    print("MICRODATA METHODS:")
    print("-" * 60)
    for method in ['kANON', 'PRAM', 'NOISE', 'LOCSUPR']:
        info = METHOD_INFO.get(method, {})
        print(f"  {method}: {info.get('description', '')}")
        print(f"         Best for: {info.get('best_for', '')}")
        print()

    print("\n--- Decision Guide ---\n")
    print("If you need to...                          Use...")
    print("-" * 60)
    print("Achieve formal k-anonymity                 -> kANON or LOCSUPR")
    print("Perturb categorical variables              -> PRAM")
    print("Add noise to continuous variables          -> NOISE")
    print("Suppress specific cells for k-anonymity    -> LOCSUPR")


def interactive_selection(data: pd.DataFrame):
    """Interactive method selection wizard."""
    print("\n" + "=" * 60)
    print("  SDC METHOD SELECTION WIZARD")
    print("=" * 60)

    # Analyze data first
    analysis = analyze_data(data, verbose=True)

    print("\n" + "-" * 60)
    print("Answer the following questions to get a recommendation:")
    print("-" * 60)

    # Question 1: Goal
    print("\n1. What is your primary privacy goal?")
    print("   [1] Achieve k-anonymity (formal guarantee)")
    print("   [2] Perturb data while preserving statistics")
    print("   [3] Hide/suppress small values")
    print("   [4] Round values for uncertainty")
    print("   [5] Not sure - recommend based on data")

    try:
        goal_input = input("\n   Enter choice (1-5): ").strip()
        goal_map = {
            '1': 'k-anonymity',
            '2': 'perturbation',
            '3': 'threshold',
            '4': 'rounding',
            '5': None
        }
        goal = goal_map.get(goal_input)
    except:
        goal = None

    # Question 2: Suppression
    print("\n2. Is suppression (hiding values with NaN) acceptable?")
    print("   [1] Yes, suppression is fine")
    print("   [2] No, all values must be visible")

    try:
        supp_input = input("\n   Enter choice (1-2): ").strip()
        allow_suppression = supp_input != '2'
    except:
        allow_suppression = True

    # Question 3: Utility preservation
    print("\n3. What statistics need to be preserved? (comma-separated)")
    print("   Options: mean, distribution, additivity, correlation")
    print("   Or press Enter to skip")

    try:
        util_input = input("\n   Enter choices: ").strip()
        if util_input:
            preserve_utility = [x.strip() for x in util_input.split(',')]
        else:
            preserve_utility = None
    except:
        preserve_utility = None

    # Get recommendation
    print("\n" + "=" * 60)
    recommendation = recommend_method(
        data,
        goal=goal,
        quasi_identifiers=analysis.get('quasi_identifiers'),
        preserve_utility=preserve_utility,
        allow_suppression=allow_suppression,
        verbose=True
    )

    return recommendation


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='SDC Method Selection & Protection Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python select_method.py data.csv                     # Analyze and recommend
  python select_method.py data.csv --protect           # Auto-protect data
  python select_method.py data.csv --protect --output protected.csv
  python select_method.py data.csv --pipeline          # Show pipeline recommendation
  python select_method.py data.csv --interactive       # Interactive mode
  python select_method.py                              # Show help and demo

Programmatic usage:
  from select_method import smart_protect, apply_pipeline
  result = smart_protect(data)                         # Auto-select approach
  result = apply_pipeline(data, goals=['k_anonymity']) # Specific pipeline
        """
    )
    parser.add_argument('file', nargs='?', help='Input CSV file')
    parser.add_argument('--protect', '-p', action='store_true',
                        help='Apply smart protection to data')
    parser.add_argument('--pipeline', action='store_true',
                        help='Show pipeline recommendation')
    parser.add_argument('--output', '-o', help='Output file for protected data')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--goal', '-g', help='Specific goal (k-anonymity, privacy, etc.)')
    parser.add_argument('--qi', nargs='+', help='Quasi-identifier columns')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    if args.file:
        # Load data from file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)

        print(f"Loading data from: {args.file}")

        # Try to detect if it's a table (first column is index)
        data = pd.read_csv(args.file)

        # If first column looks like an index, reload with index
        first_col = data.columns[0]
        if first_col == 'Unnamed: 0' or first_col == '':
            data = pd.read_csv(args.file, index_col=0)
            print("  (Detected tabular format with row index)")

        verbose = not args.quiet

        if args.interactive:
            interactive_selection(data)

        elif args.protect:
            # Smart protect
            print("\n" + "=" * 60)
            print("  SMART PROTECTION")
            print("=" * 60)

            result = smart_protect(
                data,
                quasi_identifiers=args.qi,
                goal=args.goal,
                verbose=verbose
            )

            if result['success']:
                if args.output:
                    result['protected_data'].to_csv(args.output, index=False)
                    print(f"\nProtected data saved to: {args.output}")
                else:
                    print("\nUse --output <file.csv> to save protected data")
            else:
                print("\nProtection failed!")
                sys.exit(1)

        elif args.pipeline:
            # Show pipeline recommendation
            rec = recommend_pipeline(
                data,
                goals=[args.goal] if args.goal else None,
                quasi_identifiers=args.qi,
                verbose=verbose
            )

        else:
            # Default: recommend method
            recommend_method(data, goal=args.goal, verbose=verbose)

    else:
        # Demo with sample data
        print("=" * 60)
        print("  SDC METHOD SELECTION & SMART PROTECTION TOOL")
        print("=" * 60)
        print("\nUsage:")
        print("  python select_method.py <data.csv>              # Analyze & recommend")
        print("  python select_method.py <data.csv> --protect    # Auto-protect")
        print("  python select_method.py <data.csv> --pipeline   # Pipeline recommendation")
        print("  python select_method.py <data.csv> --interactive")
        print("\nOptions:")
        print("  --protect, -p    Apply smart protection")
        print("  --output, -o     Save protected data to file")
        print("  --goal, -g       Specify goal (k-anonymity, privacy, etc.)")
        print("  --qi             Quasi-identifier columns")
        print("  --pipeline       Show pipeline recommendation")
        print("  --interactive    Interactive mode")
        print("\nProgrammatic usage:")
        print("  from select_method import smart_protect, apply_pipeline")
        print("  result = smart_protect(data)  # Auto-select best approach")
        print("  result = apply_pipeline(data, goals=['k_anonymity'])")

        # Show method comparison
        print_method_comparison()

        # Demo with sample data
        print("\n" + "=" * 60)
        print("  DEMO: SMART PROTECTION")
        print("=" * 60)

        # Create sample microdata
        np.random.seed(42)
        sample_micro = pd.DataFrame({
            'age': np.random.randint(18, 80, 50),
            'gender': np.random.choice(['M', 'F'], 50),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 50),
            'income': np.random.randint(20000, 100000, 50)
        })

        print("\nSample microdata (50 records, mixed variables):")
        result = smart_protect(sample_micro, verbose=True)
        print(f"\nResult: {result['approach']} approach, Success: {result['success']}")
