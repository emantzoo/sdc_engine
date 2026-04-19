"""
Disclosure Risk Metrics
=======================

Calculate disclosure risk metrics including k-anonymity and uniqueness.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def check_kanonymity(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int = 3
) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
    """
    Check if data satisfies k-anonymity.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to check
    quasi_identifiers : list of str
        Columns to use as quasi-identifiers
    k : int, default=3
        Minimum group size required

    Returns:
    --------
    tuple : (is_k_anonymous, group_sizes, violations)
        - is_k_anonymous: bool, whether all groups have >= k records
        - group_sizes: DataFrame with all group sizes
        - violations: DataFrame with groups that have < k records
    """
    if not quasi_identifiers:
        return True, pd.DataFrame(), pd.DataFrame()

    # Count records per group
    # Always use '_group_size_' to avoid conflicts with any column named 'count'
    count_col = '_group_size_'
    group_sizes = data.groupby(quasi_identifiers, observed=True).size().reset_index(name=count_col)

    # Find violations
    violations = group_sizes[group_sizes[count_col] < k]

    is_k_anonymous = len(violations) == 0

    # Rename to 'count' for consistent output (only if no conflict)
    if 'count' not in quasi_identifiers:
        group_sizes = group_sizes.rename(columns={count_col: 'count'})
        violations = violations.rename(columns={count_col: 'count'})

    return is_k_anonymous, group_sizes, violations


def calculate_uniqueness_rate(
    data: pd.DataFrame,
    columns: List[str]
) -> float:
    """
    Calculate the proportion of unique records based on specified columns.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
    columns : list of str
        Columns to use for uniqueness calculation

    Returns:
    --------
    float : Proportion of records that are unique (0.0 to 1.0)
    """
    if not columns or len(data) == 0:
        return 0.0

    # Filter to valid columns
    valid_columns = [c for c in columns if c in data.columns]
    if not valid_columns:
        return 0.0

    # Count unique combinations
    grouped = data.groupby(valid_columns, observed=True).size()
    n_unique = (grouped == 1).sum()
    uniqueness_rate = n_unique / len(data)

    return uniqueness_rate


def calculate_disclosure_risk(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int = 3
) -> Dict[str, float]:
    """
    Calculate comprehensive disclosure risk metrics.

    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to analyze
    quasi_identifiers : list of str
        Quasi-identifier columns
    k : int, default=3
        Threshold for considering risk

    Returns:
    --------
    dict : Risk metrics including:
        - uniqueness_rate: Proportion of unique records
        - is_k_anonymous: Whether data satisfies k-anonymity
        - n_violations: Number of groups with < k records
        - records_at_risk: Number of records in small groups
        - risk_rate: Proportion of records at risk
        - min_group_size: Smallest equivalence class
        - avg_group_size: Average equivalence class size
    """
    # Calculate uniqueness
    uniqueness = calculate_uniqueness_rate(data, quasi_identifiers)

    # Check k-anonymity violations
    is_kanon, group_sizes, violations = check_kanonymity(data, quasi_identifiers, k)

    # Calculate proportion of records at risk
    if len(violations) > 0:
        records_at_risk = violations['count'].sum()
        risk_rate = records_at_risk / len(data)
    else:
        records_at_risk = 0
        risk_rate = 0.0

    metrics = {
        'uniqueness_rate': float(uniqueness),
        'is_k_anonymous': bool(is_kanon),
        'n_violations': int(len(violations)),
        'records_at_risk': int(records_at_risk),
        'risk_rate': float(risk_rate),
        'min_group_size': int(group_sizes['count'].min()) if len(group_sizes) > 0 else 0,
        'avg_group_size': float(group_sizes['count'].mean()) if len(group_sizes) > 0 else 0.0
    }

    return metrics


def find_rare_combinations(
    data: pd.DataFrame,
    columns: List[str],
    threshold: int
) -> pd.DataFrame:
    """
    Find combinations of columns that appear fewer than threshold times.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    columns : list of str
        Columns to analyze
    threshold : int
        Minimum frequency threshold

    Returns:
    --------
    pd.DataFrame : Rare combinations with their counts
    """
    # Count occurrences of each combination
    # Use unique column name to avoid conflicts
    count_col = '_grp_count_' if 'count' in columns else 'count'
    combo_counts = data.groupby(columns, observed=True).size().reset_index(name=count_col)

    # Filter for rare combinations
    rare = combo_counts[combo_counts[count_col] < threshold]

    # Rename back to 'count' for consistent output
    if count_col != 'count':
        rare = rare.rename(columns={count_col: 'count'})

    return rare.sort_values('count')


def assess_risk_level(risk: Dict) -> str:
    """
    Assess overall risk level based on risk metrics.

    Parameters:
    -----------
    risk : dict
        Output from calculate_disclosure_risk()

    Returns:
    --------
    str : Risk level ('low', 'medium', 'high', 'critical')
    """
    uniqueness = risk.get('uniqueness_rate', 0)
    risk_rate = risk.get('risk_rate', 0)
    is_kanon = risk.get('is_k_anonymous', True)
    min_group = risk.get('min_group_size', 3)

    # Critical: high uniqueness or many records at risk
    if uniqueness > 0.10 or risk_rate > 0.15:
        return 'critical'

    # High: moderate uniqueness or not k-anonymous
    if uniqueness > 0.05 or not is_kanon or min_group < 2:
        return 'high'

    # Medium: some uniqueness or small groups
    if uniqueness > 0.02 or risk_rate > 0.05:
        return 'medium'

    # Low: good protection
    return 'low'
