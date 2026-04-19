"""
Rank Swapping (RANKSWAP)
========================

Description:
This method implements Rank Swapping for statistical disclosure control.
Rank swapping sorts records by a variable, then swaps values between
records that are within a specified rank distance. This ensures that
swapped values are similar to original values while providing protection.

Key features:
- Preserves approximate distributions and rankings
- Limits perturbation by restricting swap distance
- Better utility preservation than random swapping
- Can be applied to continuous or ordinal variables

Dependencies:
- sdc_utils: Uses shared utilities for validation and auto-detection

Input:
- data: pandas DataFrame with microdata records
- variables: list of column names to swap
- p: maximum rank distance as proportion (0.0 to 1.0)

Output:
- If return_metadata=False: DataFrame with swapped values
- If return_metadata=True: (DataFrame, metadata_dict)

Author: SDC Methods Implementation
Date: December 2025
"""

import logging
import pandas as pd
import numpy as np
import warnings
from typing import Union, List, Optional, Tuple, Dict, Any
from .sdc_utils import (
    validate_quasi_identifiers,
    auto_detect_continuous_variables,
    auto_detect_sensitive_columns
)

log = logging.getLogger(__name__)


def apply_rankswap(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    p: Optional[float] = None,
    R0: Optional[float] = None,
    top_percent: float = 5.0,
    bottom_percent: float = 5.0,
    multivariate: bool = False,
    seed: Optional[int] = None,
    return_metadata: bool = False,
    verbose: bool = True,
    column_types: Optional[Dict[str, str]] = None,
    per_variable_p: Optional[Dict[str, float]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Apply Rank Swapping to continuous variables.

    Rank swapping sorts records by a variable, then swaps values between
    records that are within a specified rank distance (p * n). This ensures
    swapped values are similar to originals while providing protection.

    Parameters:
    -----------
    data : pd.DataFrame
        Input microdata with numeric variables
    variables : list of str, optional
        Column names to swap. If None, auto-detects continuous variables.
    p : float, optional
        Maximum rank distance as proportion of total records (0.0 to 1.0).
        - 0.05: Can swap with records within 5% rank distance
        - 0.10: Can swap with records within 10% rank distance
        Higher p = more perturbation but potentially less utility.
        Either p or R0 must be provided (not both). If neither provided,
        defaults to R0=0.95 (like R sdcMicro).
    R0 : float, optional
        Correlation preservation factor (0.0 to 1.0).
        Alternative to p parameter. Higher values = more correlation preserved.
        - 0.95: Preserve 95% correlation (5% rank distance)
        - 0.90: Preserve 90% correlation (10% rank distance)
        Internally converts to: p = (1 - R0) * 100%
        Either p or R0 must be provided (not both).
    top_percent : float, default=5.0
        Percentage of highest values to protect as separate group.
        Top values only swap within top group, preventing mixing with
        middle values. Useful for protecting high earners, rare cases.
    bottom_percent : float, default=5.0
        Percentage of lowest values to protect as separate group.
        Bottom values only swap within bottom group, preventing mixing
        with middle values. Useful for protecting zero values, rare cases.
    multivariate : bool, default=False
        If True, use same swap pairs across all variables.
        If False, swap each variable independently.
    seed : int, optional
        Random seed for reproducibility
    return_metadata : bool, default=False
        If True, returns (anonymized_data, metadata_dict)
    verbose : bool, default=True
        If True, prints progress messages

    Returns:
    --------
    If return_metadata=False:
        pd.DataFrame : Data with rank-swapped values

    If return_metadata=True:
        anonymized_data : pd.DataFrame
        metadata : dict containing 'parameters', 'statistics'

    Algorithm:
    ----------
    1. For each variable:
       a. Sort records by variable value
       b. For each record (in order), find swap partner:
          - Partner must be within p*n ranks
          - Partner must not already be swapped
       c. Swap values between record and partner
    2. Result: Each value moves to a record with similar rank

    Examples:
    ---------
    # Example 1: Simple rank swapping
    >>> protected = apply_rankswap(data, variables=['income'], p=0.05)

    # Example 2: Multiple variables independently
    >>> protected = apply_rankswap(data, variables=['income', 'age'], p=0.10)

    # Example 3: Multivariate (same swaps for all variables)
    >>> protected = apply_rankswap(data, variables=['income', 'age'],
    ...                            multivariate=True, p=0.05)
    """

    if seed is not None:
        np.random.seed(seed)

    # Step 0: Parameter validation - only one of p or R0
    if p is not None and R0 is not None:
        raise ValueError("Only one of 'p' or 'R0' can be provided, not both")
    
    if p is None and R0 is None:
        # Default to R0=0.95 like R sdcMicro
        R0 = 0.95
        if verbose:
            print("RANKSWAP: Neither p nor R0 specified, defaulting to R0=0.95")
    
    # Convert R0 to p if needed
    if R0 is not None:
        if not 0.0 <= R0 <= 1.0:
            raise ValueError(f"R0 must be between 0.0 and 1.0, got {R0}")
        p = (1.0 - R0)
        if verbose:
            print(f"RANKSWAP: Using R0={R0} → p={p:.3f}")
    
    # Validate p
    if not 0.0 < p <= 0.5:
        raise ValueError(f"p must be between 0.0 and 0.5, got {p}")
    
    # Validate top/bottom percent
    if not 0.0 <= top_percent <= 50.0:
        raise ValueError(f"top_percent must be between 0 and 50, got {top_percent}")
    if not 0.0 <= bottom_percent <= 50.0:
        raise ValueError(f"bottom_percent must be between 0 and 50, got {bottom_percent}")
    if top_percent + bottom_percent >= 100.0:
        raise ValueError(f"top_percent + bottom_percent must be < 100, got {top_percent + bottom_percent}")

    # Step 1: Check for sensitive columns
    sensitive_cols = auto_detect_sensitive_columns(data, check_patterns=True)
    if sensitive_cols:
        warnings.warn(
            f"Data contains potentially sensitive columns: {list(sensitive_cols.keys())}. "
            f"Consider removing these columns before anonymization.",
            UserWarning
        )

    # Step 2: Determine variables to swap
    if variables is not None:
        is_valid, missing = validate_quasi_identifiers(data, variables)
        if not is_valid:
            raise ValueError(f"Variables not found in data: {missing}")
        swap_vars = variables
    else:
        try:
            swap_vars = auto_detect_continuous_variables(data)
            if not swap_vars:
                raise ValueError("No continuous variables found for rank swapping")
            warnings.warn(
                f"No variables specified. Auto-detected continuous: {swap_vars}. "
                f"Specify 'variables' parameter explicitly for better control.",
                UserWarning
            )
        except ValueError as e:
            raise ValueError(f"Cannot auto-detect variables: {e}")

    # Validate p parameter (duplicate guard - already checked above)
    if not 0.0 < p <= 0.5:
        raise ValueError(f"p must be between 0.0 and 0.5, got {p}")

    # Coerce object-dtype columns that Configure classified as numeric
    from sdc_engine.sdc.sdc_utils import coerce_columns_by_types
    data = data.copy()
    coerce_columns_by_types(data, column_types, swap_vars)

    # Filter to numeric-only variables; skip string/categorical columns
    numeric_vars = []
    skipped_vars = []
    for var in swap_vars:
        if pd.api.types.is_numeric_dtype(data[var]):
            numeric_vars.append(var)
        else:
            skipped_vars.append(var)
    if skipped_vars:
        log.warning(f"[RANKSWAP] Skipping non-numeric variables: {skipped_vars}")
        if verbose:
            print(f"  Skipping non-numeric: {skipped_vars}")
    if not numeric_vars:
        raise ValueError(f"No numeric variables to rank-swap after type filtering. "
                         f"Skipped: {skipped_vars}")
    swap_vars = numeric_vars

    log.info(f"[RANKSWAP] vars={swap_vars}, p={p:.3f}, R0={R0}, "
             f"top%={top_percent}, bottom%={bottom_percent}, n={len(data)}")

    # Step 2: Create copy of data
    protected_data = data.copy()
    n_records = len(data)
    max_rank_distance = int(n_records * p)

    if max_rank_distance < 1:
        max_rank_distance = 1
        warnings.warn(f"max_rank_distance was less than 1, setting to 1")

    if verbose:
        print(f"RANKSWAP: Processing {len(swap_vars)} variables with p={p} (max rank distance: {max_rank_distance})")

    statistics = {
        'total_records': n_records,
        'max_rank_distance': max_rank_distance,
        'variables_processed': [],
        'swaps_per_variable': {},
        'correlation_per_variable': {},
        'per_variable_p': per_variable_p if per_variable_p else None,
    }

    if multivariate:
        # Multivariate: Use same swap pairs for all variables
        # Per-QI treatment: use max p across variables (joint space can't differentiate)
        mv_p = p
        if per_variable_p:
            mv_p = max(per_variable_p.get(v, p) for v in swap_vars)
            if len(set(per_variable_p.get(v, p) for v in swap_vars)) > 1:
                log.info("[RANKSWAP] Multivariate mode: per-QI treatment uses "
                         f"max(p)={mv_p:.3f} for shared swap pairs")
        mv_max_rank = max(1, int(n_records * mv_p))

        # Sort by first variable to determine pairs (exclude NaN rows)
        first_var = swap_vars[0]
        non_nan_mask = data[first_var].notna()
        sorted_indices = data.loc[non_nan_mask, first_var].sort_values().index.tolist()
        n_nan = int((~non_nan_mask).sum())
        if n_nan > 0:
            log.info(f"[RANKSWAP] Multivariate: excluded {n_nan} NaN rows from '{first_var}'")

        swap_pairs = _compute_rank_swap_pairs(sorted_indices, mv_max_rank, top_percent, bottom_percent)

        for var in swap_vars:
            # Apply the same swap pairs
            original_values = protected_data[var].copy()

            for idx1, idx2 in swap_pairs:
                val1 = original_values[idx1]
                val2 = original_values[idx2]
                protected_data.loc[idx1, var] = val2
                protected_data.loc[idx2, var] = val1

            # Calculate correlation (only on non-NaN values)
            valid = data[var].notna() & protected_data[var].notna()
            if valid.sum() > 1:
                corr = np.corrcoef(data.loc[valid, var].values,
                                   protected_data.loc[valid, var].values)[0, 1]
            else:
                corr = 1.0

            statistics['variables_processed'].append(var)
            statistics['swaps_per_variable'][var] = len(swap_pairs)
            statistics['correlation_per_variable'][var] = float(corr) if not np.isnan(corr) else 1.0

            log.info(f"[RANKSWAP] '{var}': {len(swap_pairs)} swaps, corr={corr:.4f}")
            if verbose:
                print(f"  {var}: {len(swap_pairs)} swaps, correlation={corr:.4f}")
    else:
        # Univariate: Swap each variable independently
        for var in swap_vars:
            # Per-QI treatment scaling: use per-variable p if available
            effective_p = (per_variable_p.get(var, p)
                           if per_variable_p else p)
            var_max_rank = max(1, int(n_records * effective_p))

            # Sort by this variable (exclude NaN values — they stay unchanged)
            non_nan_mask = data[var].notna()
            sorted_indices = data.loc[non_nan_mask, var].sort_values().index.tolist()
            n_nan = int((~non_nan_mask).sum())
            if n_nan > 0:
                log.info(f"[RANKSWAP] '{var}': excluded {n_nan} NaN rows from swapping")

            swap_pairs = _compute_rank_swap_pairs(sorted_indices, var_max_rank, top_percent, bottom_percent)

            # Apply swaps
            original_values = protected_data[var].copy()

            for idx1, idx2 in swap_pairs:
                val1 = original_values[idx1]
                val2 = original_values[idx2]
                protected_data.loc[idx1, var] = val2
                protected_data.loc[idx2, var] = val1

            # Calculate correlation (only on non-NaN values)
            valid = data[var].notna() & protected_data[var].notna()
            if valid.sum() > 1:
                corr = np.corrcoef(data.loc[valid, var].values,
                                   protected_data.loc[valid, var].values)[0, 1]
            else:
                corr = 1.0

            statistics['variables_processed'].append(var)
            statistics['swaps_per_variable'][var] = len(swap_pairs)
            statistics['correlation_per_variable'][var] = float(corr) if not np.isnan(corr) else 1.0

            log.info(f"[RANKSWAP] '{var}': {len(swap_pairs)} swaps, corr={corr:.4f}")
            if verbose:
                print(f"  {var}: {len(swap_pairs)} swaps, correlation={corr:.4f}")

    log.info(f"[RANKSWAP] Done: processed {len(statistics['variables_processed'])} variables")

    # Step 3: Prepare return
    if return_metadata:
        metadata = {
            'method': 'RANKSWAP',
            'parameters': {
                'variables': swap_vars,
                'p': p,
                'R0': R0,
                'top_percent': top_percent,
                'bottom_percent': bottom_percent,
                'multivariate': multivariate,
                'seed': seed
            },
            'statistics': statistics
        }
        return protected_data, metadata
    else:
        return protected_data


def _compute_rank_swap_pairs(sorted_indices: List, max_distance: int, 
                            top_percent: float = 0.0, bottom_percent: float = 0.0) -> List[Tuple]:
    """
    Compute swap pairs for rank swapping with top/bottom protection.

    Parameters:
    -----------
    sorted_indices : list
        Record indices sorted by variable value
    max_distance : int
        Maximum rank distance for swapping
    top_percent : float, default=0.0
        Percentage of highest values to protect (only swap within top block)
    bottom_percent : float, default=0.0
        Percentage of lowest values to protect (only swap within bottom block)

    Returns:
    --------
    list of tuples : Swap pairs (idx1, idx2)
    """
    n = len(sorted_indices)
    swapped = set()
    pairs = []
    
    # Calculate block boundaries
    n_bottom = int(n * bottom_percent / 100.0)
    n_top = int(n * top_percent / 100.0)
    
    # Bottom block: [0, n_bottom)
    # Middle block: [n_bottom, n-n_top)
    # Top block: [n-n_top, n)

    for i in range(n):
        if sorted_indices[i] in swapped:
            continue

        # Determine which block current index belongs to
        if i < n_bottom:
            # Bottom block - can only swap within bottom
            block_min = 0
            block_max = n_bottom
        elif i >= n - n_top:
            # Top block - can only swap within top
            block_min = n - n_top
            block_max = n
        else:
            # Middle block - can only swap within middle
            block_min = n_bottom
            block_max = n - n_top

        # Find swap partner within rank distance AND within same block
        min_j = max(block_min, i - max_distance)
        max_j = min(block_max, i + max_distance + 1)

        candidates = []
        for j in range(min_j, max_j):
            if j != i and sorted_indices[j] not in swapped:
                candidates.append(j)

        if candidates:
            # Pick random candidate
            j = np.random.choice(candidates)
            idx1 = sorted_indices[i]
            idx2 = sorted_indices[j]
            pairs.append((idx1, idx2))
            swapped.add(idx1)
            swapped.add(idx2)

    return pairs


def get_rankswap_report(
    original_data: pd.DataFrame,
    protected_data: pd.DataFrame,
    variables: List[str]
) -> Dict:
    """
    Generate a report comparing original and rank-swapped data.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original dataset
    protected_data : pd.DataFrame
        Rank-swapped dataset
    variables : list of str
        Variables that were swapped

    Returns:
    --------
    report : dict
        Contains utility metrics and correlation statistics
    """
    report = {
        'variables': {},
        'summary': {
            'total_records': len(original_data),
            'avg_correlation': 0.0,
            'utility_preserved': True
        }
    }

    correlations = []

    for var in variables:
        if var not in original_data.columns or var not in protected_data.columns:
            continue

        orig = original_data[var]
        prot = protected_data[var]

        # Correlation
        corr = np.corrcoef(orig.values, prot.values)[0, 1]
        correlations.append(corr)

        # Mean and std comparison
        orig_mean = orig.mean()
        prot_mean = prot.mean()
        mean_diff = abs(orig_mean - prot_mean)

        orig_std = orig.std()
        prot_std = prot.std()
        std_diff = abs(orig_std - prot_std)

        # Rank correlation (Spearman)
        rank_corr = orig.rank().corr(prot.rank())

        report['variables'][var] = {
            'correlation': float(corr),
            'rank_correlation': float(rank_corr),
            'original_mean': float(orig_mean),
            'protected_mean': float(prot_mean),
            'mean_difference': float(mean_diff),
            'original_std': float(orig_std),
            'protected_std': float(prot_std),
            'std_difference': float(std_diff)
        }

        # Check if utility significantly degraded
        if corr < 0.8 or rank_corr < 0.8:
            report['summary']['utility_preserved'] = False

    report['summary']['avg_correlation'] = float(np.mean(correlations)) if correlations else 0.0

    return report


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("RANKSWAP (Rank Swapping) Examples")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 200
    sample_data = pd.DataFrame({
        'id': range(1, n + 1),
        'income': np.random.exponential(50000, n),
        'age': np.random.normal(45, 15, n).clip(18, 90).astype(int),
        'expenses': np.random.exponential(30000, n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n)
    })

    print("\nOriginal data sample:")
    print(sample_data.head(10))

    # Example 1: Simple rank swapping
    print("\n" + "=" * 60)
    print("Example 1: Simple rank swap (p=0.05)")
    print("=" * 60)

    result1, meta1 = apply_rankswap(
        sample_data,
        variables=['income', 'age'],
        p=0.05,
        seed=123,
        return_metadata=True
    )

    print(f"\nStatistics:")
    for var in meta1['statistics']['variables_processed']:
        swaps = meta1['statistics']['swaps_per_variable'][var]
        corr = meta1['statistics']['correlation_per_variable'][var]
        print(f"  {var}: {swaps} swaps, correlation={corr:.4f}")

    # Example 2: Higher perturbation
    print("\n" + "=" * 60)
    print("Example 2: Higher perturbation (p=0.15)")
    print("=" * 60)

    result2, meta2 = apply_rankswap(
        sample_data,
        variables=['income'],
        p=0.15,
        seed=456,
        return_metadata=True
    )

    print(f"\nIncome comparison:")
    print(f"  Original mean: {sample_data['income'].mean():.2f}")
    print(f"  Swapped mean: {result2['income'].mean():.2f}")
    print(f"  Correlation: {meta2['statistics']['correlation_per_variable']['income']:.4f}")

    # Example 3: Multivariate
    print("\n" + "=" * 60)
    print("Example 3: Multivariate (same swaps for all)")
    print("=" * 60)

    result3, meta3 = apply_rankswap(
        sample_data,
        variables=['income', 'expenses'],
        p=0.10,
        multivariate=True,
        seed=789,
        return_metadata=True
    )

    # Generate report
    print("\n" + "=" * 60)
    print("Rank Swap Report")
    print("=" * 60)

    report = get_rankswap_report(sample_data, result1, ['income', 'age'])
    print(f"\nSummary:")
    print(f"  Average correlation: {report['summary']['avg_correlation']:.4f}")
    print(f"  Utility preserved: {report['summary']['utility_preserved']}")

    for var, stats in report['variables'].items():
        print(f"\n{var}:")
        print(f"  Correlation: {stats['correlation']:.4f}")
        print(f"  Rank correlation: {stats['rank_correlation']:.4f}")
        print(f"  Mean diff: {stats['mean_difference']:.2f}")
