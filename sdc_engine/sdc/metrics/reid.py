"""
Re-identification Risk (ReID) Percentiles
==========================================

Calculates per-record re-identification risk as 1/equivalence_class_size,
then reports percentiles (reid_50, reid_95, reid_99) across all records.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_reid(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    quantiles: Optional[List[float]] = None
) -> Dict:
    """
    Calculate re-identification risk percentiles across records.

    Each record's individual risk = 1 / equivalence_class_size (the
    probability of re-identification given the quasi-identifier combination).
    Returns percentiles of that distribution.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : list of str
        Columns used to assess re-identification risk
    quantiles : list of float, optional
        Quantiles to calculate (default: [0.5, 0.9, 0.95, 0.99])

    Returns:
    --------
    dict : ReID metrics including:
        - reid_50: Median risk (50% of records have risk <= this)
        - reid_90: 90th percentile risk
        - reid_95: 95th percentile risk
        - reid_99: 99th percentile risk
        - max_risk: Maximum individual risk
        - mean_risk: Average risk across all records
        - high_risk_count: Number of records with risk > 0.2 (20%)
        - high_risk_rate: Proportion of high-risk records

    Example:
    --------
    >>> reid = calculate_reid(data, ['age', 'gender', 'region'])
    >>> print(f"95% of records have risk <= {reid['reid_95']:.1%}")
    95% of records have risk <= 10.0%
    """
    if quantiles is None:
        quantiles = [0.5, 0.9, 0.95, 0.99]

    if not quasi_identifiers:
        return {
            'reid_50': 0.0, 'reid_90': 0.0, 'reid_95': 0.0, 'reid_99': 0.0,
            'max_risk': 0.0, 'mean_risk': 0.0, 'high_risk_count': 0
        }

    # Filter to rows without NaN in quasi-identifiers (suppressed cells)
    total_records = len(data)
    valid_data = data.dropna(subset=quasi_identifiers)
    n_suppressed = total_records - len(valid_data)
    suppression_rate = n_suppressed / total_records if total_records > 0 else 0.0

    if len(valid_data) == 0:
        # All records have suppressed QIs - return zero risk (fully protected)
        return {
            'reid_50': 0.0, 'reid_90': 0.0, 'reid_95': 0.0, 'reid_99': 0.0,
            'max_risk': 0.0, 'mean_risk': 0.0, 'high_risk_count': 0,
            'high_risk_rate': 0.0,
            'suppressed_records': n_suppressed,
            'suppression_rate': 1.0,
            'records_evaluated': 0,
        }

    # Calculate group sizes for each record
    group_sizes = valid_data.groupby(quasi_identifiers, observed=True).transform('size')

    # Individual risk = 1 / group_size (probability of re-identification)
    individual_risk = 1 / group_sizes

    # Calculate quantiles
    result = {}
    for q in quantiles:
        key = f'reid_{int(q * 100)}'
        result[key] = float(individual_risk.quantile(q))

    # Additional summary statistics
    result['max_risk'] = float(individual_risk.max())
    result['mean_risk'] = float(individual_risk.mean())
    result['high_risk_count'] = int((individual_risk > 0.2).sum())
    result['high_risk_rate'] = float((individual_risk > 0.2).mean())

    # Suppression tracking — always report how many records were excluded
    result['suppressed_records'] = n_suppressed
    result['suppression_rate'] = suppression_rate
    result['records_evaluated'] = len(valid_data)

    # Include risk scores for further analysis
    result['risk_scores'] = individual_risk.tolist()

    return result


def assess_risk_with_reid(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    reid_thresholds: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Assess whether data meets ReID-based risk thresholds.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : list of str
        Quasi-identifier columns
    reid_thresholds : dict, optional
        Target thresholds, e.g., {'reid_95': 0.05, 'reid_99': 0.10}
        Default: {'reid_95': 0.05, 'reid_99': 0.10}

    Returns:
    --------
    dict : Assessment results including pass/fail for each threshold
    """
    if reid_thresholds is None:
        reid_thresholds = {'reid_95': 0.05, 'reid_99': 0.10}

    reid = calculate_reid(data, quasi_identifiers)

    assessment = {
        'reid_metrics': reid,
        'thresholds': reid_thresholds,
        'passes_all': True,
        'details': {}
    }

    for metric, threshold in reid_thresholds.items():
        if metric in reid:
            passes = reid[metric] <= threshold
            assessment['details'][metric] = {
                'value': reid[metric],
                'threshold': threshold,
                'passes': passes
            }
            if not passes:
                assessment['passes_all'] = False

    return assessment


def classify_risk_pattern(reid_metrics: Dict) -> str:
    """
    Classify the risk distribution pattern based on ReID percentiles.

    Patterns:
    - uniform_high: Most records have high risk
    - widespread: Risk spread across many records
    - severe_tail: Few records dominate risk (reid_99 >> reid_50)
    - tail: Moderate tail risk
    - uniform_low: Uniformly low risk
    - bimodal: Two distinct risk groups
    - moderate: Moderate overall risk

    Parameters:
    -----------
    reid_metrics : dict
        Output from calculate_reid()

    Returns:
    --------
    str : Risk pattern name
    """
    reid_50 = reid_metrics.get('reid_50', 0)
    reid_95 = reid_metrics.get('reid_95', 0)
    reid_99 = reid_metrics.get('reid_99', 0)
    mean_risk = reid_metrics.get('mean_risk', 0)

    # Calculate tail ratio (how much worse is worst-case vs median)
    tail_ratio = reid_99 / reid_50 if reid_50 > 0 else (100 if reid_99 > 0 else 1)

    # Uniform high risk: most records have high risk
    if reid_50 > 0.20:
        return 'uniform_high' if reid_99 - reid_50 < 0.10 else 'widespread'

    # Check for tail patterns: few records dominate risk
    # Severe tail: reid_99 >> reid_50 (ratio > 10x) AND reid_99 is high
    if tail_ratio > 10 and reid_99 > 0.30:
        return 'severe_tail'

    # Moderate tail: reid_99 >> reid_95 or high reid_95
    if reid_95 > 0.30 or (reid_99 > 0.50 and tail_ratio > 5):
        return 'tail'

    # Low median risk
    if reid_50 < 0.05:
        # But check if there's still a significant tail
        if reid_99 > 0.20 and tail_ratio > 5:
            return 'tail'
        return 'uniform_low'

    # Bimodal: large gap between mean and median
    if abs(mean_risk - reid_50) > 0.15:
        return 'bimodal'

    return 'moderate'
