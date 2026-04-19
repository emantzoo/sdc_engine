"""
Record Swapping (RECSWAP)
=========================

Description:
This method implements Record Swapping for statistical disclosure control.
Record swapping exchanges values between records that are similar on certain
key variables (matching variables), while preserving marginal distributions
and aggregate statistics.

Key features:
- Preserves marginal distributions exactly
- Maintains relationships within swapped records
- Configurable swap rate and matching criteria
- Supports geographic/hierarchical swapping

Dependencies:
- sdc_utils: Uses shared utilities for validation and auto-detection

Input:
- data: pandas DataFrame with microdata records
- variables: list of column names to swap
- swap_rate: proportion of records to swap (0.0 to 1.0)
- match_variables: variables that must match for swapping partners
- within_strata: optional stratification variable

Output:
- If return_metadata=False: DataFrame with swapped values
- If return_metadata=True: (DataFrame, metadata_dict)

Author: SDC Methods Implementation
Date: December 2025
"""

import pandas as pd
import numpy as np
import warnings
from typing import Union, List, Optional, Tuple, Dict, Any
from .sdc_utils import (
    validate_quasi_identifiers,
    auto_detect_categorical_variables,
    auto_detect_sensitive_columns
)


def apply_recswap(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    swap_rate: float = 0.05,
    match_variables: Optional[List[str]] = None,
    within_strata: Optional[str] = None,
    targeted: bool = False,
    target_variable: Optional[str] = None,
    seed: Optional[int] = None,
    return_metadata: bool = False,
    verbose: bool = True,
    column_types: Optional[Dict[str, str]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Apply Record Swapping to microdata.

    Record swapping exchanges values of specified variables between pairs
    of records that match on certain criteria. This provides protection
    while preserving marginal distributions and aggregate statistics.

    Parameters:
    -----------
    data : pd.DataFrame
        Input microdata
    variables : list of str, optional
        Column names to swap. If None, auto-detects categorical variables.
    swap_rate : float, default=0.05
        Proportion of records to swap (0.0 to 1.0).
        - 0.05: 5% of records swapped (low perturbation)
        - 0.10: 10% of records swapped (moderate)
        - 0.20: 20% of records swapped (high perturbation)
    match_variables : list of str, optional
        Variables that must match between swap partners.
        E.g., ['age_group'] ensures only records with same age_group are swapped.
    within_strata : str, optional
        Stratification variable - swapping only occurs within strata.
        E.g., 'region' means records only swap within same region.
    targeted : bool, default=False
        If True, prioritize swapping high-risk (unique) records.
    target_variable : str, optional
        If targeted=True, use this variable to identify high-risk records.
    seed : int, optional
        Random seed for reproducibility
    return_metadata : bool, default=False
        If True, returns (anonymized_data, metadata_dict)
    verbose : bool, default=True
        If True, prints progress messages

    Returns:
    --------
    If return_metadata=False:
        pd.DataFrame : Data with swapped values

    If return_metadata=True:
        anonymized_data : pd.DataFrame
        metadata : dict containing 'parameters', 'statistics'

    Examples:
    ---------
    # Example 1: Simple swapping
    >>> protected = apply_recswap(data, variables=['income'], swap_rate=0.10)

    # Example 2: Swap within same region
    >>> protected = apply_recswap(data, variables=['income', 'education'],
    ...                           within_strata='region', swap_rate=0.05)

    # Example 3: Match on age group
    >>> protected = apply_recswap(data, variables=['address'],
    ...                           match_variables=['age_group'], swap_rate=0.10)
    """

    if seed is not None:
        np.random.seed(seed)

    # Step 0: Check for sensitive columns
    sensitive_cols = auto_detect_sensitive_columns(data, check_patterns=True)
    if sensitive_cols:
        warnings.warn(
            f"Data contains potentially sensitive columns that could directly identify individuals: "
            f"{list(sensitive_cols.keys())}. Consider removing these columns before anonymization.",
            UserWarning
        )

    # Step 1: Determine variables to swap
    if variables is not None:
        is_valid, missing = validate_quasi_identifiers(data, variables)
        if not is_valid:
            raise ValueError(f"Variables not found in data: {missing}")
        swap_vars = variables
    else:
        try:
            swap_vars = auto_detect_categorical_variables(data)
            if not swap_vars:
                raise ValueError("No categorical variables found for swapping")
            warnings.warn(
                f"No variables specified. Auto-detected: {swap_vars}. "
                f"Specify 'variables' parameter explicitly for better control.",
                UserWarning
            )
        except ValueError as e:
            raise ValueError(f"Cannot auto-detect variables: {e}")

    # Validate match_variables
    if match_variables:
        is_valid, missing = validate_quasi_identifiers(data, match_variables)
        if not is_valid:
            raise ValueError(f"Match variables not found: {missing}")

    # Validate within_strata
    if within_strata and within_strata not in data.columns:
        raise ValueError(f"Strata variable '{within_strata}' not found in data")

    # Validate swap_rate
    if not 0.0 <= swap_rate <= 1.0:
        raise ValueError(f"swap_rate must be between 0.0 and 1.0, got {swap_rate}")

    # Step 2: Create copy of data
    protected_data = data.copy()

    # Step 3: Perform swapping
    n_records = len(data)
    n_to_swap = int(n_records * swap_rate)

    if verbose:
        print(f"RECSWAP: Swapping {n_to_swap} records ({swap_rate:.1%}) for variables: {swap_vars}")

    statistics = {
        'total_records': n_records,
        'target_swaps': n_to_swap,
        'actual_swaps': 0,
        'variables_swapped': swap_vars,
        'swaps_per_variable': {var: 0 for var in swap_vars}
    }

    # Get indices to potentially swap
    if targeted and target_variable:
        # Prioritize unique records
        if target_variable in data.columns:
            value_counts = data[target_variable].value_counts()
            # Records with rare values are higher priority
            priorities = data[target_variable].map(lambda x: 1.0 / max(value_counts.get(x, 1), 1))
            priorities = priorities / priorities.sum()
            swap_indices = np.random.choice(
                data.index, size=min(n_to_swap * 2, n_records),
                replace=False, p=priorities
            )
        else:
            swap_indices = np.random.choice(data.index, size=min(n_to_swap * 2, n_records), replace=False)
    else:
        swap_indices = np.random.choice(data.index, size=min(n_to_swap * 2, n_records), replace=False)

    # Track which records have been swapped
    swapped = set()
    swap_pairs = []

    # Group by strata if specified
    if within_strata:
        groups = data.groupby(within_strata).groups
    else:
        groups = {'all': data.index.tolist()}

    # Perform swapping within each stratum
    for stratum, stratum_indices in groups.items():
        stratum_indices = set(stratum_indices)
        available = list(stratum_indices - swapped)

        if len(available) < 2:
            continue

        # Number of swaps for this stratum (proportional)
        stratum_swaps = int(len(stratum_indices) / n_records * n_to_swap)

        np.random.shuffle(available)

        i = 0
        swaps_done = 0
        while i < len(available) - 1 and swaps_done < stratum_swaps:
            idx1 = available[i]

            # Find matching partner
            partner_found = False
            for j in range(i + 1, len(available)):
                idx2 = available[j]

                # Check match criteria
                if match_variables:
                    match = all(
                        data.loc[idx1, var] == data.loc[idx2, var]
                        for var in match_variables
                    )
                    if not match:
                        continue

                # Perform swap
                for var in swap_vars:
                    val1 = protected_data.loc[idx1, var]
                    val2 = protected_data.loc[idx2, var]
                    protected_data.loc[idx1, var] = val2
                    protected_data.loc[idx2, var] = val1
                    statistics['swaps_per_variable'][var] += 1

                swapped.add(idx1)
                swapped.add(idx2)
                swap_pairs.append((idx1, idx2))
                swaps_done += 1
                partner_found = True

                # Remove partner from available
                available.remove(idx2)
                break

            i += 1

        statistics['actual_swaps'] += swaps_done

    if verbose:
        print(f"  Actual swaps completed: {statistics['actual_swaps']}")
        print(f"  Records affected: {len(swapped)}")

    # Step 4: Prepare return
    if return_metadata:
        metadata = {
            'method': 'RECSWAP',
            'parameters': {
                'variables': swap_vars,
                'swap_rate': swap_rate,
                'match_variables': match_variables,
                'within_strata': within_strata,
                'targeted': targeted,
                'seed': seed
            },
            'statistics': statistics,
            'swap_pairs': swap_pairs[:100]  # Store first 100 pairs for reference
        }
        return protected_data, metadata
    else:
        return protected_data


def get_recswap_report(
    original_data: pd.DataFrame,
    protected_data: pd.DataFrame,
    variables: List[str]
) -> Dict:
    """
    Generate a report comparing original and swapped data.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original dataset
    protected_data : pd.DataFrame
        Swapped dataset
    variables : list of str
        Variables that were swapped

    Returns:
    --------
    report : dict
        Contains distribution comparisons and swap statistics
    """
    report = {
        'variables': {},
        'summary': {
            'total_records': len(original_data),
            'records_changed': 0,
            'distribution_preserved': True
        }
    }

    total_changed = 0

    for var in variables:
        if var not in original_data.columns or var not in protected_data.columns:
            continue

        orig = original_data[var]
        prot = protected_data[var]

        # Count changes
        changes = (orig != prot).sum()
        total_changed += changes

        # Compare distributions
        orig_dist = orig.value_counts(normalize=True).sort_index()
        prot_dist = prot.value_counts(normalize=True).sort_index()

        # Check if distributions match
        dist_diff = (orig_dist - prot_dist.reindex(orig_dist.index, fill_value=0)).abs().max()

        report['variables'][var] = {
            'records_changed': int(changes),
            'change_rate': float(changes / len(original_data)),
            'original_distribution': orig_dist.to_dict(),
            'protected_distribution': prot_dist.to_dict(),
            'max_distribution_difference': float(dist_diff)
        }

        if dist_diff > 0.01:  # More than 1% difference
            report['summary']['distribution_preserved'] = False

    report['summary']['records_changed'] = int(total_changed)

    return report


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("RECSWAP (Record Swapping) Examples")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 200
    sample_data = pd.DataFrame({
        'id': range(1, n + 1),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'age_group': np.random.choice(['18-30', '31-45', '46-60', '60+'], n),
        'income': np.random.randint(20000, 100000, n),
        'education': np.random.choice(['HS', 'Bachelor', 'Master', 'PhD'], n)
    })

    print("\nOriginal data sample:")
    print(sample_data.head(10))

    # Example 1: Simple swapping
    print("\n" + "=" * 60)
    print("Example 1: Simple 10% swap rate")
    print("=" * 60)

    result1, meta1 = apply_recswap(
        sample_data,
        variables=['income', 'education'],
        swap_rate=0.10,
        seed=123,
        return_metadata=True
    )

    print(f"\nStatistics:")
    print(f"  Target swaps: {meta1['statistics']['target_swaps']}")
    print(f"  Actual swaps: {meta1['statistics']['actual_swaps']}")

    # Example 2: Swap within strata
    print("\n" + "=" * 60)
    print("Example 2: Swap within same region")
    print("=" * 60)

    result2, meta2 = apply_recswap(
        sample_data,
        variables=['income'],
        within_strata='region',
        swap_rate=0.15,
        seed=456,
        return_metadata=True
    )

    print(f"\nStatistics:")
    print(f"  Actual swaps: {meta2['statistics']['actual_swaps']}")

    # Check distribution preservation
    print("\nIncome distribution comparison:")
    print(f"  Original mean: {sample_data['income'].mean():.2f}")
    print(f"  Swapped mean: {result2['income'].mean():.2f}")
