"""
Random Noise Addition (NOISE)
=============================

Description:
This method implements random noise addition for statistical disclosure control.
It adds random perturbation to continuous variables to protect individual values
while preserving statistical properties of the data.

The method supports multiple noise distributions:
- Gaussian (Normal): Symmetric noise, preserves mean
- Laplace: Heavier tails, better for differential privacy
- Uniform: Bounded noise within a range
- Proportional: Noise scaled by the value itself

Key features:
- Configurable noise magnitude (standard deviation or percentage)
- Multiple noise distributions
- Optional bounds to prevent unrealistic values
- Correlated noise option to preserve relationships between variables
- Preserves aggregate statistics when properly configured
- R integration via sdcMicro for correlated noise (67% less distortion)

Dependencies:
- numpy: For random number generation
- scipy: For Laplace distribution (optional fallback to numpy)
- rpy2 (optional): For R/sdcMicro integration

Input:
- data: pandas DataFrame with continuous variables
- variables: list of column names to perturb
- noise_type: 'gaussian', 'laplace', 'uniform', or 'proportional'
- magnitude: noise level (std dev, scale, or percentage)
- bounds: optional (min, max) to clip resulting values

Output:
- If return_metadata=False: DataFrame with perturbed values
- If return_metadata=True: (DataFrame, metadata_dict)

Author: SDC Methods Implementation
Date: December 2025
"""

import logging
import pandas as pd
import numpy as np
import warnings
from typing import Union, List, Optional, Tuple, Dict, Any

log = logging.getLogger(__name__)

# Suppress rpy2 thread warning (harmless)
warnings.filterwarnings('ignore', message='R is not initialized by the main thread')

# Lazy R availability check - don't load at import time
_R_AVAILABLE = None  # None = not checked yet, True/False = checked

def _check_r_available():
    """Lazily check if R/sdcMicro is available."""
    global _R_AVAILABLE
    if _R_AVAILABLE is not None:
        return _R_AVAILABLE

    # Skip R on Streamlit Cloud - use Python implementations
    import os
    if os.environ.get('STREAMLIT_SHARING_MODE') or os.path.exists('/mount/src'):
        _R_AVAILABLE = False
        return False

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        # Check if sdcMicro is installed
        try:
            ro.r('library(sdcMicro)')
            _R_AVAILABLE = True
        except Exception:
            _R_AVAILABLE = False
    except ImportError:
        _R_AVAILABLE = False

    return _R_AVAILABLE

from .sdc_utils import (
    validate_quasi_identifiers,
    auto_detect_continuous_variables,
    auto_detect_sensitive_columns
)


def _apply_r_noise(
    data: pd.DataFrame,
    variables: List[str],
    magnitude: float = 0.1,
    method: str = 'correlated',
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply noise using R's sdcMicro package.

    R's sdcMicro uses correlated noise that preserves relationships between
    variables, resulting in ~67% less distortion for the same privacy level.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    # Need at least one categorical for sdcMicro
    # Find or create a dummy key variable
    cat_cols = [c for c in data.columns if c not in variables and data[c].dtype == 'object']
    if not cat_cols:
        # Create a dummy categorical column
        data_r = data.copy()
        data_r['__dummy_key__'] = 'A'
        key_var = '__dummy_key__'
    else:
        data_r = data.copy()
        key_var = cat_cols[0]

    # Normalize data types for R conversion - avoid mixed-type issues
    # This is needed because GENERALIZE may create string bins like "20-29"
    for col in data_r.columns:
        # Use infer_objects() to normalize types
        data_r[col] = data_r[col].infer_objects()

        # If column has object dtype, convert to string (becomes R factor)
        if data_r[col].dtype == 'object':
            data_r[col] = data_r[col].astype(str)

    # Convert to R
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data_r)

    ro.globalenv['data'] = r_data
    ro.globalenv['keyVars'] = ro.StrVector([key_var])
    ro.globalenv['numVars'] = ro.StrVector(variables)
    ro.globalenv['noise_level'] = magnitude

    if seed is not None:
        ro.r(f'set.seed({seed})')

    if verbose:
        print(f"  Using R sdcMicro correlated noise...")

    # Run R noise addition
    ro.r(f'''
    library(sdcMicro)
    sdc <- createSdcObj(data, keyVars=keyVars, numVars=numVars)
    sdc <- addNoise(sdc, noise=noise_level, method='{method}')
    r_result <- extractManipData(sdc)
    ''')

    # Get results back
    with localconverter(ro.default_converter + pandas2ri.converter):
        result_df = ro.conversion.rpy2py(ro.r('r_result'))

    # Remove dummy column if we added it
    if '__dummy_key__' in result_df.columns:
        result_df = result_df.drop(columns=['__dummy_key__'])

    # Calculate statistics
    statistics = {}
    for var in variables:
        orig = data[var].values
        noisy = result_df[var].values
        diff = noisy - orig
        mae = np.abs(diff).mean()
        rmse = np.sqrt((diff ** 2).mean())
        corr = np.corrcoef(orig, noisy)[0, 1] if len(orig) > 1 else 1.0

        statistics[var] = {
            'mean_absolute_error': float(mae),
            'rmse': float(rmse),
            'correlation': float(corr),
            'noise_std': float(np.std(diff))
        }

    return result_df, statistics


def apply_noise(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    noise_type: str = 'gaussian',
    magnitude: float = 0.1,
    relative: bool = True,
    use_r: bool = True,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    preserve_sign: bool = True,
    seed: Optional[int] = None,
    return_metadata: bool = False,
    verbose: bool = True,
    column_types: Optional[Dict[str, str]] = None,
    per_variable_magnitude: Optional[Dict[str, float]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Apply random noise to continuous variables for privacy protection.

    When R/sdcMicro is available and use_r=True, uses correlated noise that
    preserves relationships between variables (~67% less distortion).

    Parameters:
    -----------
    data : pd.DataFrame
        Input microdata with continuous variables
    variables : list of str, optional
        Column names to perturb. If None, auto-detects continuous variables.
    noise_type : str, default='gaussian'
        Type of noise distribution (Python only):
        - 'gaussian': Normal distribution N(0, sigma)
        - 'laplace': Laplace distribution with heavier tails
        - 'uniform': Uniform distribution U(-a, a)
        - 'proportional': Noise proportional to value magnitude
        - 'additive0': Zero-preserving additive noise (only perturbs non-zero values)
    magnitude : float, default=0.1
        Noise magnitude:
        - If relative=True: percentage of std dev (0.1 = 10% of std)
        - If relative=False: absolute standard deviation
    relative : bool, default=True
        If True, magnitude is relative to variable's standard deviation
        If False, magnitude is absolute
    use_r : bool, default=True
        If True and R/sdcMicro is available, use R's correlated noise.
        R produces ~67% less distortion while maintaining privacy.
    bounds : dict, optional
        Per-variable bounds as {column: (min, max)} (Python only)
        Values outside bounds are clipped
    preserve_sign : bool, default=True
        If True, prevents sign changes (positive stays positive, Python only)
    seed : int, optional
        Random seed for reproducibility
    return_metadata : bool, default=False
        If True, returns (anonymized_data, metadata_dict)
    verbose : bool, default=True
        If True, prints progress messages

    Returns:
    --------
    If return_metadata=False:
        pd.DataFrame : Data with noisy values

    If return_metadata=True:
        anonymized_data : pd.DataFrame
            Data with perturbed values
        metadata : dict
            Contains 'parameters', 'noise_applied', 'statistics'

    Examples:
    ---------
    # Example 1: Simple usage with 10% Gaussian noise
    >>> anonymized = apply_noise(data, variables=['income', 'age'], magnitude=0.1)

    # Example 2: Laplace noise for differential privacy
    >>> anonymized = apply_noise(data, variables=['salary'],
    ...                          noise_type='laplace', magnitude=0.15)

    # Example 3: With bounds to ensure realistic values
    >>> bounds = {'age': (0, 120), 'income': (0, None)}
    >>> anonymized = apply_noise(data, variables=['age', 'income'], bounds=bounds)

    # Example 4: Absolute noise (not relative to std)
    >>> anonymized = apply_noise(data, variables=['score'],
    ...                          magnitude=5.0, relative=False)

    Notes:
    ------
    - Gaussian noise preserves mean and approximately preserves variance
    - Laplace noise provides better differential privacy guarantees
    - Higher magnitude = more privacy but less data utility
    - Use relative=True for variables with different scales
    """

    if seed is not None:
        np.random.seed(seed)

    # Step 1: Determine variables to perturb
    if variables is not None:
        # Explicit variables provided - VALIDATE
        is_valid, missing = validate_quasi_identifiers(data, variables)
        if not is_valid:
            raise ValueError(f"Variables not found in data: {missing}")
        use_vars = variables
    else:
        # AUTO-DETECT continuous variables
        try:
            use_vars = auto_detect_continuous_variables(data)
            if not use_vars:
                raise ValueError("No continuous variables found")
            warnings.warn(
                f"No variables specified. Auto-detected continuous variables: {use_vars}. "
                f"Specify 'variables' parameter explicitly for better control.",
                UserWarning
            )
        except ValueError as e:
            raise ValueError(
                f"Cannot auto-detect continuous variables: {e}. "
                f"Please specify 'variables' parameter explicitly."
            )

    # Step 2: Validate noise_type
    valid_noise_types = ['gaussian', 'laplace', 'uniform', 'proportional', 'additive0']
    if noise_type not in valid_noise_types:
        raise ValueError(f"noise_type must be one of {valid_noise_types}, got '{noise_type}'")

    # Step 3: Validate magnitude
    if magnitude <= 0:
        raise ValueError(f"magnitude must be positive, got {magnitude}")

    # Coerce object-dtype columns that Configure classified as numeric
    from sdc_engine.sdc.sdc_utils import coerce_columns_by_types
    data = data.copy()
    coerce_columns_by_types(data, column_types, use_vars)

    # Detect integer columns BEFORE noise addition so we can round afterward
    _integer_cols: set = set()
    for _v in use_vars:
        if _v not in data.columns:
            continue
        _s = data[_v].dropna()
        if len(_s) == 0:
            continue
        # Check actual dtype first (int64, Int64, etc.)
        if pd.api.types.is_integer_dtype(data[_v]):
            _integer_cols.add(_v)
        elif pd.api.types.is_float_dtype(data[_v]):
            # Float column but all values are whole numbers → treat as integer
            if (_s == _s.round(0)).all():
                _integer_cols.add(_v)
    if _integer_cols:
        log.info(f"[NOISE] Integer columns (will round after noise): {_integer_cols}")

    # Validate magnitude upper bound — prevent absurd noise levels
    if magnitude > 10.0:
        log.warning(f"[NOISE] Clamping magnitude from {magnitude} to 10.0")
        magnitude = 10.0

    log.info(f"[NOISE] vars={use_vars}, type={noise_type}, magnitude={magnitude}, "
             f"relative={relative}, preserve_sign={preserve_sign}, n={len(data)}")

    # Try R implementation first (67% less distortion with correlated noise)
    # R's addNoise requires all numVars to be numeric; skip R when data has
    # non-numeric columns in the noise variable list to avoid repeated failures
    _all_numeric = all(pd.api.types.is_numeric_dtype(data[v]) for v in use_vars)
    if use_r and _check_r_available() and noise_type == 'gaussian' and _all_numeric:
        try:
            if verbose:
                print(f"NOISE: Applying correlated noise to {len(use_vars)} variables")

            protected_data, var_statistics = _apply_r_noise(
                data, use_vars, magnitude, 'correlated', seed, verbose
            )

            # Apply caps, bounds, and preserve_sign AFTER R noise
            for var in use_vars:
                if not pd.api.types.is_numeric_dtype(protected_data[var]):
                    continue
                original = data[var]
                perturbed = protected_data[var].copy()

                # Cap relative change at 25% per value
                _abs_orig = original.abs()
                _max_change = _abs_orig * 0.25
                _nonzero = _abs_orig > 0
                _change = perturbed - original
                _capped_change = _change.clip(lower=-_max_change, upper=_max_change)
                _was_capped = _nonzero & ((_change.abs() - _max_change) > 1e-12)
                perturbed = perturbed.where(~_was_capped, original + _capped_change)

                # Distributional correction after capping — proportional
                # scaling preserves distribution shape better than uniform
                # shift for columns with wide value ranges.
                _valid = original.notna()
                _uncapped = _valid & ~_was_capped & _nonzero
                n_uncapped = _uncapped.sum()
                if n_uncapped > 5:
                    _pre_cap_mean = original[_valid].mean()
                    _post_cap_mean = perturbed[_valid].mean()
                    if abs(_post_cap_mean) > 1e-12:
                        _scale = _pre_cap_mean / _post_cap_mean
                        perturbed = perturbed.where(
                            ~_uncapped, perturbed * _scale)
                    else:
                        _mean_shift = _post_cap_mean - _pre_cap_mean
                        perturbed = perturbed.where(
                            ~_uncapped, perturbed - _mean_shift)

                # Apply bounds if specified
                var_bounds = bounds.get(var) if bounds else None
                if var_bounds:
                    lower, upper = var_bounds
                    if lower is not None:
                        perturbed = perturbed.clip(lower=lower)
                    if upper is not None:
                        perturbed = perturbed.clip(upper=upper)
                    log.info(f"[NOISE] Applied bounds to '{var}' after R noise: {var_bounds}")

                # Preserve sign if requested (zeros stay non-negative)
                if preserve_sign:
                    non_negative_mask = original >= 0
                    perturbed = perturbed.where(
                        ~non_negative_mask | (perturbed >= 0),
                        np.abs(perturbed))
                    negative_mask = original < 0
                    perturbed = perturbed.where(
                        ~negative_mask | (perturbed < 0),
                        -np.abs(perturbed))

                protected_data[var] = perturbed

            # Round integer columns back to whole numbers
            for var in use_vars:
                if var in _integer_cols and var in protected_data.columns:
                    protected_data[var] = protected_data[var].round(0).astype('Int64')

            log.info(f"[NOISE] R path done for {len(use_vars)} variables")

            if verbose:
                for var, stats in var_statistics.items():
                    print(f"  {var}: MAE={stats['mean_absolute_error']:.2f}, "
                          f"correlation={stats['correlation']:.4f}")

            if return_metadata:
                metadata = {
                    'method': 'NOISE',
                    'parameters': {
                        'variables': use_vars,
                        'noise_type': 'R_sdcMicro_correlated',
                        'magnitude': magnitude,
                        'use_r': True,
                        'bounds': {k: list(v) for k, v in bounds.items()} if bounds else None,
                        'preserve_sign': preserve_sign,
                        'seed': seed
                    },
                    'statistics': {
                        'total_records': len(data),
                        'variables_processed': use_vars,
                        'noise_per_variable': var_statistics
                    }
                }
                return protected_data, metadata
            return protected_data

        except Exception as e:
            log.warning(f"[NOISE] R implementation failed: {e}, falling back to Python")
            if verbose:
                print(f"  R implementation failed: {e}")
                print(f"  Falling back to Python implementation...")

    # Python implementation
    # Step 4: Create copy of data (already coerced above)
    protected_data = data.copy()

    # Step 5: Apply noise to each variable
    statistics = {
        'total_records': len(data),
        'variables_processed': [],
        'noise_per_variable': {},
        'value_changes': {}
    }

    for var in use_vars:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(data[var]):
            if verbose:
                print(f"Skipping '{var}': not numeric")
            continue

        original = data[var].copy()
        n = len(original)

        # Calculate noise scale — per-QI override from smart_method_config
        # already accounts for treatment level × IQR ratio (absolute value,
        # not a multiplier), so it replaces base magnitude entirely.
        effective_mag = (per_variable_magnitude.get(var, magnitude)
                         if per_variable_magnitude else magnitude)
        if relative:
            # Scale relative to variable's standard deviation
            var_std = original.std()
            if var_std == 0:
                if verbose:
                    print(f"Skipping '{var}': zero variance")
                continue
            noise_scale = effective_mag * var_std
        else:
            noise_scale = effective_mag

        # Generate noise based on distribution type
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_scale, n)
        elif noise_type == 'laplace':
            # Laplace scale parameter (b) relates to std as: std = sqrt(2) * b
            laplace_scale = noise_scale / np.sqrt(2)
            noise = np.random.laplace(0, laplace_scale, n)
        elif noise_type == 'uniform':
            # Uniform bounds chosen so std matches noise_scale
            # For U(-a, a): std = a / sqrt(3), so a = sqrt(3) * std
            uniform_bound = np.sqrt(3) * noise_scale
            noise = np.random.uniform(-uniform_bound, uniform_bound, n)
        elif noise_type == 'proportional':
            # Noise proportional to absolute value
            noise = np.random.normal(0, 1, n) * np.abs(original) * magnitude
        elif noise_type == 'additive0':
            # Zero-preserving additive noise: only add noise to non-zero values
            noise = np.random.normal(0, noise_scale, n)
            # Preserve zeros: only apply noise where original value is non-zero
            zero_mask = (original == 0) | original.isna()
            noise[zero_mask] = 0

        # Apply noise
        perturbed = original + noise

        # Cap relative change at 25% per value — prevents small values in
        # high-std columns from being destroyed (e.g., price 14k in a column
        # with std 100k would otherwise get noise of ±50k)
        _abs_orig = original.abs()
        _max_change = _abs_orig * 0.25
        _nonzero = _abs_orig > 0
        _change = perturbed - original
        _capped_change = _change.clip(lower=-_max_change, upper=_max_change)
        _was_capped = _nonzero & ((_change.abs() - _max_change) > 1e-12)
        perturbed = perturbed.where(~_was_capped, original + _capped_change)

        # Distributional correction after capping — proportional scaling
        _valid = original.notna()
        _uncapped = _valid & ~_was_capped & _nonzero
        n_uncapped = _uncapped.sum()
        if n_uncapped > 5:
            _pre_cap_mean = original[_valid].mean()
            _post_cap_mean = perturbed[_valid].mean()
            if abs(_post_cap_mean) > 1e-12:
                _scale = _pre_cap_mean / _post_cap_mean
                perturbed = perturbed.where(
                    ~_uncapped, perturbed * _scale)
            else:
                _mean_shift = _post_cap_mean - _pre_cap_mean
                perturbed = perturbed.where(
                    ~_uncapped, perturbed - _mean_shift)

        # Handle NaN values (preserve them)
        perturbed = perturbed.where(~original.isna(), np.nan)

        # Apply bounds if specified
        var_bounds = bounds.get(var) if bounds else None
        if var_bounds:
            lower, upper = var_bounds
            if lower is not None:
                perturbed = perturbed.clip(lower=lower)
            if upper is not None:
                perturbed = perturbed.clip(upper=upper)

        # Preserve sign if requested (zeros stay non-negative)
        if preserve_sign:
            non_negative_mask = original >= 0
            perturbed = perturbed.where(
                ~non_negative_mask | (perturbed >= 0),
                np.abs(perturbed))
            negative_mask = original < 0
            perturbed = perturbed.where(~negative_mask | (perturbed < 0),
                                        -np.abs(perturbed))

        # Round integer columns back to whole numbers
        if var in _integer_cols:
            perturbed = perturbed.round(0).astype('Int64')

        protected_data[var] = perturbed

        # Calculate statistics (cast to float for math with nullable Int64)
        actual_noise = perturbed.astype(float) - original.astype(float)
        mae = np.abs(actual_noise).mean()
        rmse = np.sqrt((actual_noise ** 2).mean())

        statistics['variables_processed'].append(var)
        statistics['noise_per_variable'][var] = {
            'noise_type': noise_type,
            'noise_scale': float(noise_scale),
            'mae': float(mae),
            'rmse': float(rmse),
            'relative_error': float(mae / np.abs(original).mean()) if np.abs(original).mean() > 0 else 0
        }
        statistics['value_changes'][var] = {
            'original_mean': float(original.mean()),
            'perturbed_mean': float(perturbed.mean()),
            'original_std': float(original.std()),
            'perturbed_std': float(perturbed.std()),
            'correlation': float(original.corr(perturbed)) if original.std() > 0 and perturbed.std() > 0 else 1.0
        }

        log.info(f"[NOISE] '{var}': scale={noise_scale:.4f}, MAE={mae:.2f}, "
                 f"corr={statistics['value_changes'][var]['correlation']:.4f}")

        if verbose:
            print(f"NOISE applied to '{var}': MAE={mae:.2f}, RMSE={rmse:.2f}, "
                  f"correlation={statistics['value_changes'][var]['correlation']:.4f}")

    log.info(f"[NOISE] Done: processed {len(statistics['variables_processed'])} variables")

    # Step 6: Prepare return
    if return_metadata:
        metadata = {
            'method': 'NOISE',
            'parameters': {
                'variables': use_vars,
                'noise_type': noise_type,
                'magnitude': magnitude,
                'relative': relative,
                'preserve_sign': preserve_sign,
                'seed': seed,
                'per_variable_magnitude': (
                    per_variable_magnitude if per_variable_magnitude else None),
            },
            'statistics': statistics
        }
        return protected_data, metadata
    else:
        return protected_data


def get_noise_report(
    original_data: pd.DataFrame,
    protected_data: pd.DataFrame,
    variables: List[str]
) -> Dict:
    """
    Generate a report comparing original and noise-protected data.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original dataset
    protected_data : pd.DataFrame
        Noise-protected dataset
    variables : list of str
        Variables that were perturbed

    Returns:
    --------
    report : dict
        Contains statistical comparisons and utility metrics
    """
    report = {
        'variables': {},
        'summary': {
            'total_records': len(original_data),
            'avg_correlation': 0.0,
            'avg_relative_error': 0.0
        }
    }

    correlations = []
    relative_errors = []

    for var in variables:
        if var not in original_data.columns or var not in protected_data.columns:
            continue

        orig = original_data[var]
        prot = protected_data[var]

        # Calculate metrics
        diff = prot - orig
        mae = np.abs(diff).mean()
        rmse = np.sqrt((diff ** 2).mean())
        correlation = orig.corr(prot) if orig.std() > 0 and prot.std() > 0 else 1.0
        relative_error = mae / np.abs(orig).mean() if np.abs(orig).mean() > 0 else 0

        correlations.append(correlation)
        relative_errors.append(relative_error)

        report['variables'][var] = {
            'original_mean': float(orig.mean()),
            'protected_mean': float(prot.mean()),
            'mean_difference': float(prot.mean() - orig.mean()),
            'original_std': float(orig.std()),
            'protected_std': float(prot.std()),
            'mae': float(mae),
            'rmse': float(rmse),
            'correlation': float(correlation),
            'relative_error': float(relative_error)
        }

    if correlations:
        report['summary']['avg_correlation'] = float(np.mean(correlations))
        report['summary']['avg_relative_error'] = float(np.mean(relative_errors))

    return report


def calculate_noise_for_epsilon(
    data: pd.DataFrame,
    variable: str,
    epsilon: float,
    sensitivity: Optional[float] = None
) -> float:
    """
    Calculate Laplace noise scale for differential privacy.

    For epsilon-differential privacy with Laplace mechanism:
    scale = sensitivity / epsilon

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    variable : str
        Column name
    epsilon : float
        Privacy parameter (smaller = more privacy)
    sensitivity : float, optional
        Query sensitivity. If None, uses range of data as estimate.

    Returns:
    --------
    scale : float
        Laplace scale parameter to achieve epsilon-DP
    """
    if sensitivity is None:
        # Estimate sensitivity as range of the data
        sensitivity = data[variable].max() - data[variable].min()

    scale = sensitivity / epsilon
    return scale


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("NOISE (Random Noise Addition) Examples")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n = 200
    sample_data = pd.DataFrame({
        'id': range(1, n + 1),
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 15000, n).clip(15000, 150000),
        'score': np.random.uniform(0, 100, n),
        'height': np.random.normal(170, 10, n),
        'gender': np.random.choice(['M', 'F'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n)
    })

    print("\nOriginal data sample:")
    print(sample_data[['age', 'income', 'score', 'height']].describe())

    # Example 1: Gaussian noise
    print("\n" + "=" * 60)
    print("Example 1: Gaussian Noise (10% of std)")
    print("=" * 60)

    result1, meta1 = apply_noise(
        sample_data,
        variables=['income', 'age'],
        noise_type='gaussian',
        magnitude=0.1,
        seed=123,
        return_metadata=True
    )

    print("\nPerturbed data sample:")
    print(result1[['age', 'income']].describe())

    print("\nNoise statistics:")
    for var, stats in meta1['statistics']['noise_per_variable'].items():
        print(f"  {var}: MAE={stats['mae']:.2f}, correlation={meta1['statistics']['value_changes'][var]['correlation']:.4f}")

    # Example 2: Laplace noise
    print("\n" + "=" * 60)
    print("Example 2: Laplace Noise (for differential privacy)")
    print("=" * 60)

    result2, meta2 = apply_noise(
        sample_data,
        variables=['income'],
        noise_type='laplace',
        magnitude=0.15,
        seed=456,
        return_metadata=True
    )

    print(f"\nLaplace noise on income:")
    print(f"  Original mean: {sample_data['income'].mean():.2f}")
    print(f"  Protected mean: {result2['income'].mean():.2f}")
    print(f"  Correlation: {meta2['statistics']['value_changes']['income']['correlation']:.4f}")

    # Example 3: With bounds
    print("\n" + "=" * 60)
    print("Example 3: With Bounds (age 0-120, income >= 0)")
    print("=" * 60)

    bounds = {
        'age': (0, 120),
        'income': (0, None)
    }

    result3 = apply_noise(
        sample_data,
        variables=['age', 'income'],
        noise_type='gaussian',
        magnitude=0.2,
        bounds=bounds,
        seed=789
    )

    print(f"\nAge range: {result3['age'].min():.0f} - {result3['age'].max():.0f}")
    print(f"Income min: {result3['income'].min():.2f}")

    # Example 4: Proportional noise
    print("\n" + "=" * 60)
    print("Example 4: Proportional Noise (larger values get more noise)")
    print("=" * 60)

    result4, meta4 = apply_noise(
        sample_data,
        variables=['income'],
        noise_type='proportional',
        magnitude=0.05,
        seed=111,
        return_metadata=True
    )

    # Compare noise for low vs high incomes
    low_income_mask = sample_data['income'] < 40000
    high_income_mask = sample_data['income'] > 60000

    low_income_noise = np.abs(result4.loc[low_income_mask, 'income'] - sample_data.loc[low_income_mask, 'income']).mean()
    high_income_noise = np.abs(result4.loc[high_income_mask, 'income'] - sample_data.loc[high_income_mask, 'income']).mean()

    print(f"\nProportional noise effect:")
    print(f"  Avg noise for low incomes (<40k): {low_income_noise:.2f}")
    print(f"  Avg noise for high incomes (>60k): {high_income_noise:.2f}")

    # Example 5: Generate report
    print("\n" + "=" * 60)
    print("Noise Report")
    print("=" * 60)

    report = get_noise_report(sample_data, result1, ['income', 'age'])
    print(f"\nSummary:")
    print(f"  Average correlation: {report['summary']['avg_correlation']:.4f}")
    print(f"  Average relative error: {report['summary']['avg_relative_error']:.2%}")

    for var, stats in report['variables'].items():
        print(f"\n{var}:")
        print(f"  Mean change: {stats['mean_difference']:.2f}")
        print(f"  MAE: {stats['mae']:.2f}")
        print(f"  Correlation: {stats['correlation']:.4f}")
