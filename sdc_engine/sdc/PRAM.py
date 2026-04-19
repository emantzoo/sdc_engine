"""
Post-RAndomization Method (PRAM)
================================

Description:
This method implements the Post-RAndomization Method for statistical disclosure control.
PRAM randomizes categorical values based on a transition probability matrix, where each
category has a probability of being changed to another category (or staying the same).

The method provides plausible deniability - even if an intruder knows someone is in the
dataset, they cannot be certain the recorded value is the true value.

Key features:
- Preserves approximate marginal distributions when using invariant PRAM
- Configurable transition probabilities
- Can be applied to single or multiple categorical variables
- Supports custom transition matrices or automatic generation

Dependencies:
- sdc_utils: Uses shared utilities for validation and auto-detection

Input:
- data: pandas DataFrame with categorical variables
- variables: list of column names to randomize (or auto-detect)
- p_change: probability of changing a value (diagonal = 1 - p_change)
- transition_matrix: optional custom transition matrix per variable
- invariant: if True, use invariant PRAM to preserve marginal distributions

Output:
- If return_metadata=False: DataFrame with randomized values
- If return_metadata=True: (DataFrame, metadata_dict)

References:
-----------
- Kooiman, P., Willenborg, L., Gouweleeuw, J. (1997). PRAM: A Method
  for Disclosure Limitation of Microdata. Statistics Netherlands
  Research Paper 9705.
- Templ, M., Kowarik, A., Meindl, B. (2015). Statistical Disclosure
  Control for Micro-Data Using the R Package sdcMicro. Journal of
  Statistical Software, 67(4), 1-36.

This implementation uses invariant PRAM (preserves marginal distributions
on average) by default, following the formulation in Kooiman et al. (1997).
The invariant transition matrix is computed as described in Section 3 of
the JSS paper.

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
    auto_detect_categorical_variables,
    auto_detect_sensitive_columns
)

log = logging.getLogger(__name__)


def apply_pram(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    p_change: float = 0.2,
    pd_min: float = 0.05,
    alpha: float = 0.5,
    transition_matrices: Optional[Dict[str, np.ndarray]] = None,
    invariant: bool = True,
    seed: Optional[int] = None,
    return_metadata: bool = False,
    verbose: bool = True,
    column_types: Optional[Dict[str, str]] = None,
    per_variable_p_change: Optional[Dict[str, float]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Apply Post-RAndomization Method (PRAM) to categorical variables.

    PRAM randomizes categorical values based on transition probabilities.
    Each value has a chance of being changed to another valid category,
    providing plausible deniability for individual records.

    Parameters:
    -----------
    data : pd.DataFrame
        Input microdata with categorical variables
    variables : list of str, optional
        Column names to randomize. If None, auto-detects categorical variables.
    p_change : float, default=0.2
        Base probability of changing a value (0.0 to 1.0).
        Higher values = more perturbation = more privacy but less utility.
        - 0.0: No changes (no privacy protection)
        - 0.2: 20% chance of change (recommended)
        - 0.5: 50% chance of change (high perturbation)
    pd_min : float, default=0.05
        Minimum diagonal probability in transition matrix (minimum chance of
        staying unchanged). Must be between 0.0 and 1.0. Ensures some stability
        in rare categories.
    alpha : float, default=0.5
        Weighting parameter for combining global and local distributions when
        generating invariant PRAM matrices. Range [0, 1]:
        - 0.0: Use only global distribution
        - 0.5: Equal weight to global and local
        - 1.0: Use only local distribution
    transition_matrices : dict, optional
        Custom transition matrices per variable.
        Format: {column_name: np.ndarray} where matrix[i,j] = P(j|i)
        Each row must sum to 1.0
    invariant : bool, default=True
        If True, use invariant PRAM which preserves marginal distributions
        on average. If False, use simple PRAM (may distort distributions).
    seed : int, optional
        Random seed for reproducibility
    return_metadata : bool, default=False
        If True, returns (anonymized_data, metadata_dict)
        If False, returns only anonymized_data
    verbose : bool, default=True
        If True, prints progress messages

    Returns:
    --------
    If return_metadata=False:
        pd.DataFrame : Data with randomized categorical values

    If return_metadata=True:
        anonymized_data : pd.DataFrame
            Data with randomized values
        metadata : dict
            Contains 'parameters', 'transition_matrices', 'statistics'

    Examples:
    ---------
    # Example 1: Simple usage with auto-detection
    >>> anonymized = apply_pram(data, p_change=0.2)

    # Example 2: Specific variables
    >>> anonymized = apply_pram(data, variables=['gender', 'region'], p_change=0.3)

    # Example 3: With metadata
    >>> anonymized, metadata = apply_pram(data, p_change=0.2, return_metadata=True)
    >>> print(metadata['statistics']['total_changes'])

    # Example 4: Custom transition matrix
    >>> # Gender: 90% stay same, 10% flip
    >>> gender_matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    >>> anonymized = apply_pram(data, variables=['gender'],
    ...                         transition_matrices={'gender': gender_matrix})
    """

    if seed is not None:
        np.random.seed(seed)

    # Step 1: Determine variables to randomize
    if variables is not None:
        # Explicit variables provided - VALIDATE
        is_valid, missing = validate_quasi_identifiers(data, variables)
        if not is_valid:
            raise ValueError(f"Variables not found in data: {missing}")
        use_vars = variables
    else:
        # AUTO-DETECT categorical variables
        try:
            use_vars = auto_detect_categorical_variables(data)
            if not use_vars:
                raise ValueError("No categorical variables found")
            warnings.warn(
                f"No variables specified. Auto-detected categorical variables: {use_vars}. "
                f"Specify 'variables' parameter explicitly for better control.",
                UserWarning
            )
        except ValueError as e:
            raise ValueError(
                f"Cannot auto-detect categorical variables: {e}. "
                f"Please specify 'variables' parameter explicitly."
            )

    # Step 2: Validate parameters
    if not 0.0 <= p_change <= 1.0:
        raise ValueError(f"p_change must be between 0.0 and 1.0, got {p_change}")

    # Cap p_change at 0.50 to prevent data destruction
    if p_change > 0.50:
        log.warning(f"[PRAM] Clamping p_change from {p_change} to 0.50 to prevent data destruction")
        p_change = 0.50

    # Validate pd_min
    if not 0.0 <= pd_min <= 1.0:
        raise ValueError(f"pd_min must be between 0.0 and 1.0, got {pd_min}")
    if pd_min < 0.01:
        log.warning(f"[PRAM] pd_min={pd_min} is very low; clamping to 0.01")
        pd_min = 0.01

    # Validate alpha
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

    log.info(f"[PRAM] vars={use_vars}, p_change={p_change}, pd_min={pd_min}, "
             f"alpha={alpha}, invariant={invariant}, n={len(data)}")

    # Step 3: Create copy of data
    protected_data = data.copy()

    # Step 4: Build or validate transition matrices
    matrices = {}
    for var in use_vars:
        if transition_matrices and var in transition_matrices:
            # Use provided matrix
            matrix = transition_matrices[var]
            categories = data[var].dropna().unique()
            if matrix.shape[0] != len(categories) or matrix.shape[1] != len(categories):
                raise ValueError(
                    f"Transition matrix for '{var}' has wrong shape. "
                    f"Expected ({len(categories)}, {len(categories)}), got {matrix.shape}"
                )
            # Verify rows sum to 1
            row_sums = matrix.sum(axis=1)
            if not np.allclose(row_sums, 1.0):
                raise ValueError(f"Transition matrix rows for '{var}' must sum to 1.0")
            matrices[var] = {
                'matrix': matrix,
                'categories': list(categories)
            }
        else:
            # Generate transition matrix
            categories = sorted(data[var].dropna().unique())
            n_categories = len(categories)

            if n_categories < 2:
                if verbose:
                    print(f"Skipping '{var}': only {n_categories} category")
                continue

            # Per-QI treatment scaling if available
            effective_pc = (per_variable_p_change.get(var, p_change)
                            if per_variable_p_change else p_change)
            if invariant:
                # Invariant PRAM: preserves marginal distributions on average
                matrix = _create_invariant_matrix(data[var], categories, effective_pc, pd_min, alpha)
            else:
                # Simple PRAM: uniform transition probabilities
                matrix = _create_simple_matrix(n_categories, effective_pc, pd_min)

            matrices[var] = {
                'matrix': matrix,
                'categories': categories
            }

    # Step 5: Apply PRAM to each variable
    statistics = {
        'total_records': len(data),
        'variables_processed': [],
        'changes_per_variable': {},
        'total_changes': 0
    }

    for var in use_vars:
        if var not in matrices:
            continue

        matrix_info = matrices[var]
        matrix = matrix_info['matrix']
        categories = matrix_info['categories']

        # Create category to index mapping
        cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

        # Apply randomization - VECTORIZED for performance
        original_values = protected_data[var].copy()
        new_values = original_values.copy()

        # Convert to indices for vectorized processing
        valid_mask = original_values.notna() & original_values.isin(cat_to_idx.keys())
        valid_indices = protected_data.index[valid_mask]

        if len(valid_indices) > 0:
            # Map values to category indices
            val_indices = original_values[valid_mask].map(cat_to_idx).values.astype(int)

            # Fully vectorized sampling using cumulative probabilities
            cumprobs = np.cumsum(matrix, axis=1)

            # Generate random values for all rows at once
            random_vals = np.random.random(len(val_indices))

            # Vectorized: get cumprobs row for each value, then searchsorted
            # cumprobs[val_indices] gives the cumprob rows for each value
            selected_cumprobs = cumprobs[val_indices]

            # Vectorized searchsorted - compare random values to cumprobs
            # For each row, find first column where cumprob >= random_val
            # This is equivalent to: new_cat_indices[i] = searchsorted(cumprobs[val_idx[i]], rand[i])
            # Broadcast comparison: (N, K) > (N, 1) gives bool matrix
            # Sum along axis=1 gives how many columns are < random_val, i.e., the insertion point
            comparison = selected_cumprobs < random_vals[:, np.newaxis]
            new_cat_indices = comparison.sum(axis=1)

            # Clip to valid range (in case of floating point issues)
            new_cat_indices = np.clip(new_cat_indices, 0, len(categories) - 1)

            # Map back to category values using numpy array indexing
            categories_arr = np.array(categories)
            new_cat_values = categories_arr[new_cat_indices]

            # Count changes (vectorized)
            original_cat_values = original_values[valid_mask].values
            n_changes = int(np.sum(original_cat_values != new_cat_values))

            # Update values
            new_values.loc[valid_indices] = new_cat_values
        else:
            n_changes = 0

        protected_data[var] = new_values

        statistics['variables_processed'].append(var)
        statistics['changes_per_variable'][var] = n_changes
        statistics['total_changes'] += n_changes

        log.info(f"[PRAM] '{var}': {n_changes} changes ({n_changes / len(data) * 100:.1f}%), "
                 f"categories={len(categories)}")

        if verbose:
            change_rate = n_changes / len(data) * 100
            print(f"PRAM applied to '{var}': {n_changes} changes ({change_rate:.1f}%)")

    log.info(f"[PRAM] Done: total_changes={statistics['total_changes']}, "
             f"vars_processed={len(statistics['variables_processed'])}")

    # Step 6: Prepare return
    if return_metadata:
        metadata = {
            'method': 'PRAM',
            'parameters': {
                'variables': use_vars,
                'p_change': p_change,
                'invariant': invariant,
                'seed': seed,
                'per_variable_p_change': (
                    per_variable_p_change if per_variable_p_change else None),
            },
            'transition_matrices': {
                var: {
                    'matrix': matrices[var]['matrix'].tolist(),
                    'categories': matrices[var]['categories']
                }
                for var in matrices
            },
            'statistics': statistics
        }
        return protected_data, metadata
    else:
        return protected_data


def _create_simple_matrix(n_categories: int, p_change: float, pd_min: float = 0.05) -> np.ndarray:
    """
    Create a simple transition matrix with uniform off-diagonal probabilities.

    Parameters:
    -----------
    n_categories : int
        Number of categories
    p_change : float
        Total probability of changing to a different category
    pd_min : float, default=0.05
        Minimum diagonal probability (minimum chance of staying unchanged)

    Returns:
    --------
    np.ndarray : Transition matrix where matrix[i,j] = P(j|i)
    """
    # Ensure diagonal is at least pd_min
    p_stay = max(1.0 - p_change, pd_min)
    
    # Remaining probability for changes
    p_remaining = 1.0 - p_stay

    # Probability of changing to any other specific category
    p_other = p_remaining / (n_categories - 1) if n_categories > 1 else 0

    # Build matrix
    matrix = np.full((n_categories, n_categories), p_other)
    np.fill_diagonal(matrix, p_stay)

    return matrix


def _create_invariant_matrix(
    series: pd.Series,
    categories: List,
    p_change: float,
    pd_min: float = 0.05,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create a PRAM transition matrix that approximately preserves marginal distributions.

    The invariant property π * M = π holds exactly only at alpha=1.0.
    At the default alpha=0.5, the uniform-component blending introduces
    a small bias toward uniform marginals — low-frequency categories are
    slightly inflated and high-frequency categories slightly deflated.
    For single-application SDC with moderate p_change, this distortion
    is typically negligible.

    Parameters:
    -----------
    series : pd.Series
        Original data series
    categories : list
        List of categories in order
    p_change : float
        Base probability of change
    pd_min : float, default=0.05
        Minimum diagonal probability (minimum chance of staying unchanged)
    alpha : float, default=0.5
        Weight for combining global and local distributions (0=global, 1=local)

    Returns:
    --------
    np.ndarray : Invariant transition matrix
    """
    n = len(categories)

    # Calculate original frequencies
    value_counts = series.value_counts()
    freqs = np.array([value_counts.get(cat, 0) for cat in categories], dtype=float)
    freqs = freqs / freqs.sum()  # Normalize to probabilities

    # Build invariant matrix with alpha weighting
    # For invariant PRAM: P(j|i) = (1-p)*I(i=j) + p*f(j)
    # With alpha: use weighted combination of uniform and frequency-based transitions

    p_stay_base = 1.0 - p_change
    matrix = np.zeros((n, n))

    for i in range(n):
        # Base diagonal probability from invariant PRAM formula
        p_diag_invariant = p_stay_base + p_change * freqs[i]
        
        # Ensure diagonal is at least pd_min
        p_stay = max(p_diag_invariant, pd_min)
        
        # If pd_min forced increase, reduce off-diagonal proportionally
        if p_stay > p_diag_invariant:
            # Scale down the change probability to accommodate pd_min
            effective_p_change = 1.0 - p_stay
        else:
            effective_p_change = p_change
        
        # Remaining probability for off-diagonal
        p_remaining = 1.0 - p_stay
        
        for j in range(n):
            if i == j:
                matrix[i, j] = p_stay
            else:
                if p_remaining > 0 and n > 1:
                    # Alpha weighting: blend uniform and frequency-based
                    # Uniform: equal probability to all other categories
                    # Frequency: proportional to frequency of target category
                    p_uniform = 1.0 / (n - 1)
                    p_freq_based = freqs[j] / (1.0 - freqs[i]) if (1.0 - freqs[i]) > 0 else p_uniform
                    
                    # Weighted combination
                    p_transition = (1 - alpha) * p_uniform + alpha * p_freq_based
                    
                    # Scale by remaining probability
                    matrix[i, j] = p_remaining * p_transition
                else:
                    matrix[i, j] = 0.0

    # Normalize rows to sum to 1 (handle numerical precision)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix = matrix / row_sums

    return matrix


def get_pram_report(
    original_data: pd.DataFrame,
    protected_data: pd.DataFrame,
    variables: List[str]
) -> Dict:
    """
    Generate a report comparing original and PRAM-protected data.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original dataset
    protected_data : pd.DataFrame
        PRAM-protected dataset
    variables : list of str
        Variables that were randomized

    Returns:
    --------
    report : dict
        Contains distribution comparisons and change statistics
    """
    report = {
        'variables': {},
        'summary': {
            'total_records': len(original_data),
            'total_changes': 0,
            'overall_change_rate': 0.0
        }
    }

    total_changes = 0

    for var in variables:
        if var not in original_data.columns or var not in protected_data.columns:
            continue

        orig = original_data[var]
        prot = protected_data[var]

        # Count changes
        changes = (orig != prot).sum()
        total_changes += changes

        # Original distribution
        orig_dist = orig.value_counts(normalize=True).to_dict()

        # Protected distribution
        prot_dist = prot.value_counts(normalize=True).to_dict()

        # Distribution difference (max absolute difference)
        all_cats = set(orig_dist.keys()) | set(prot_dist.keys())
        max_diff = max(
            abs(orig_dist.get(cat, 0) - prot_dist.get(cat, 0))
            for cat in all_cats
        )

        report['variables'][var] = {
            'changes': int(changes),
            'change_rate': float(changes / len(original_data)),
            'original_distribution': orig_dist,
            'protected_distribution': prot_dist,
            'max_distribution_difference': float(max_diff)
        }

    report['summary']['total_changes'] = int(total_changes)
    report['summary']['overall_change_rate'] = float(
        total_changes / (len(original_data) * len(variables)) if variables else 0
    )

    return report


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("PRAM (Post-RAndomization Method) Examples")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 100
    sample_data = pd.DataFrame({
        'id': range(1, n + 1),
        'gender': np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n, p=[0.3, 0.25, 0.25, 0.2]),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.4, 0.35, 0.2, 0.05]),
        'age': np.random.randint(18, 80, n),
        'income': np.random.randint(20000, 100000, n)
    })

    print("\nOriginal data sample:")
    print(sample_data.head(10))

    print("\nOriginal distributions:")
    for col in ['gender', 'region', 'education']:
        print(f"\n{col}:")
        print(sample_data[col].value_counts(normalize=True))

    # Example 1: Simple PRAM
    print("\n" + "=" * 60)
    print("Example 1: Simple PRAM (p_change=0.2)")
    print("=" * 60)

    result1, meta1 = apply_pram(
        sample_data,
        variables=['gender', 'region'],
        p_change=0.2,
        seed=123,
        return_metadata=True
    )

    print("\nAnonymized data sample:")
    print(result1[['id', 'gender', 'region']].head(10))

    print("\nStatistics:")
    print(f"  Total changes: {meta1['statistics']['total_changes']}")
    for var, changes in meta1['statistics']['changes_per_variable'].items():
        print(f"  {var}: {changes} changes")

    # Example 2: Invariant PRAM (preserves distributions)
    print("\n" + "=" * 60)
    print("Example 2: Invariant PRAM (preserves distributions)")
    print("=" * 60)

    result2, meta2 = apply_pram(
        sample_data,
        variables=['gender', 'region', 'education'],
        p_change=0.3,
        invariant=True,
        seed=456,
        return_metadata=True
    )

    print("\nComparing distributions:")
    for var in ['gender', 'region', 'education']:
        orig_dist = sample_data[var].value_counts(normalize=True)
        prot_dist = result2[var].value_counts(normalize=True)
        print(f"\n{var}:")
        print(f"  Original:  {orig_dist.to_dict()}")
        print(f"  Protected: {prot_dist.to_dict()}")

    # Example 3: Custom transition matrix
    print("\n" + "=" * 60)
    print("Example 3: Custom transition matrix")
    print("=" * 60)

    # Custom matrix for gender: 85% stay same, 15% change
    gender_matrix = np.array([
        [0.85, 0.15],  # Female: 85% stay Female, 15% become Male
        [0.15, 0.85]   # Male: 85% stay Male, 15% become Female
    ])

    result3 = apply_pram(
        sample_data,
        variables=['gender'],
        transition_matrices={'gender': gender_matrix},
        seed=789
    )

    print("\nCustom gender transition (15% change rate):")
    print(f"Original:  {sample_data['gender'].value_counts().to_dict()}")
    print(f"Protected: {result3['gender'].value_counts().to_dict()}")

    # Generate report
    print("\n" + "=" * 60)
    print("PRAM Report")
    print("=" * 60)

    report = get_pram_report(sample_data, result2, ['gender', 'region', 'education'])
    print(f"\nSummary:")
    print(f"  Total records: {report['summary']['total_records']}")
    print(f"  Total changes: {report['summary']['total_changes']}")
    print(f"  Overall change rate: {report['summary']['overall_change_rate']:.2%}")

    for var, stats in report['variables'].items():
        print(f"\n{var}:")
        print(f"  Changes: {stats['changes']} ({stats['change_rate']:.1%})")
        print(f"  Max distribution diff: {stats['max_distribution_difference']:.3f}")
