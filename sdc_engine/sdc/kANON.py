"""
k-Anonymity (kANON)
===================

Description:
k-Anonymity is a formal privacy criterion that requires each combination of
quasi-identifiers (QIs) to appear in at least k records. This ensures that each
individual is "hidden" within a group of at least k-1 other individuals with
identical QI values.

The method achieves k-anonymity through:
- Generalization: Replacing specific values with more general ones (e.g., age 25 -> 20-29)
- Suppression: Removing specific values that create unique combinations

Quasi-identifiers are attributes that, when combined, could be used for re-identification
(e.g., age, gender, zip code, occupation).

Dependencies:
- sdc_utils: Uses shared utilities for k-anonymity checking, generalization, and validation

Input:
- data: pandas DataFrame with microdata
- k: minimum group size (each QI combination must appear at least k times)
- quasi_identifiers: list of column names that are quasi-identifiers
- hierarchies: dict mapping QI columns to their generalization hierarchies
- max_suppression_rate: maximum percentage of records that can be suppressed

Output:
- k-anonymous DataFrame where all QI combinations appear at least k times

References:
-----------
- Samarati, P. (2001). Protecting respondents' identities in microdata
  release. IEEE Transactions on Knowledge and Data Engineering, 13(6),
  1010-1027.
- Sweeney, L. (2002). k-Anonymity: A Model for Protecting Privacy.
  International Journal of Uncertainty, Fuzziness and Knowledge-Based
  Systems, 10(5), 557-570.
- Templ, M., Kowarik, A., Meindl, B. (2015). Statistical Disclosure
  Control for Micro-Data Using the R Package sdcMicro. Journal of
  Statistical Software, 67(4), 1-36.

This implementation follows Samarati (2001) and Sweeney (2002) for the
core k-anonymity definition, with the generalization + suppression
hybrid strategy inspired by sdcMicro's approach (Templ et al. 2015).
The beam-search and recursive local recoding strategies are novel
extensions developed for this engine.

Author: SDC Methods Implementation
Date: December 2025
"""

import time as _time
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple, Union, Any
from copy import deepcopy
from .sdc_utils import (
    check_kanonymity,
    generalize_numeric,
    generalize_string_prefix,
    validate_quasi_identifiers,
    auto_detect_quasi_identifiers,
    auto_detect_sensitive_columns
)


def apply_kanon(
    data: pd.DataFrame,
    k: int = 3,
    quasi_identifiers: Optional[List[str]] = None,
    hierarchies: Optional[Dict[str, Any]] = None,
    max_suppression_rate: float = 0.1,
    strategy: str = 'generalization',
    bin_size: int = 10,
    string_method: str = 'prefix',
    prefix_length: int = 3,
    return_metadata: bool = False,
    verbose: bool = True,
    column_types: Optional[Dict[str, str]] = None,
    per_qi_bin_size: Optional[Dict[str, int]] = None,
    sensitive_columns: Optional[List[str]] = None,
    l_target: Optional[int] = None,
    t_target: Optional[float] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Apply k-anonymity to microdata using generalization and/or suppression.

    Parameters:
    -----------
    data : pd.DataFrame
        Input microdata (any DataFrame with columns)
    k : int, default=3
        Minimum group size - each QI combination must appear at least k times
    quasi_identifiers : list of str, optional
        Column names that are quasi-identifiers
        If None, will attempt to identify them automatically (not recommended)
    hierarchies : dict, optional
        Custom generalization hierarchies for each QI
        Format: {column_name: {original_value: generalized_value, ...}}
        If None, will create default hierarchies using bin_size and string_method
    max_suppression_rate : float, default=0.1
        Maximum proportion of records that can be suppressed (0.0 to 1.0)
    strategy : str, default='generalization'
        Strategy to achieve k-anonymity:
        - 'generalization': Use only generalization (greedy, one QI at a time)
        - 'suppression': Use only local suppression
        - 'hybrid': Use both (generalization first, then suppression)
        - 'beam': Beam search over generalization lattice (explores multiple
          paths simultaneously; finds lower-info-loss solutions than greedy)
        - 'recursive': ARX-inspired recursive local recoding. Runs
          generalization + suppression, then re-anonymizes suppressed records
          with progressively more aggressive parameters (up to 2 levels deep)
          to recover data that would otherwise be lost
    bin_size : int, default=10
        Size of bins for numeric generalization (e.g., 10 means 0-9, 10-19, etc.)
        Use smaller values (5) for more precision, larger (20, 50) for more privacy
    string_method : str, default='prefix'
        Method for generalizing string/categorical columns:
        - 'prefix': Keep first N characters + '*' (e.g., "North" -> "Nor*")
        - 'first_char': Keep only first character + '*' (e.g., "North" -> "N*")
        - 'suppress': Replace with '*' (full suppression)
    prefix_length : int, default=3
        Number of characters to keep when string_method='prefix'
    return_metadata : bool, default=False
        If True, returns (anonymized_data, metadata_dict)
        If False, returns only anonymized_data
    verbose : bool, default=True
        If True, prints progress messages during anonymization
        If False, runs silently (useful for batch processing)

    Returns:
    --------
    If return_metadata=False:
        pd.DataFrame : k-anonymous dataset

    If return_metadata=True:
        anonymized_data : pd.DataFrame
            k-anonymous dataset
        metadata : dict
            Contains 'method', 'parameters', 'k_anonymity_check', 'statistics'

    Examples:
    ---------
    # Basic usage with default settings
    >>> anonymized = apply_kanon(data, k=5, quasi_identifiers=['age', 'gender', 'region'])

    # Custom bin size for numeric columns (smaller bins = more precision)
    >>> anonymized = apply_kanon(data, k=5, quasi_identifiers=['age', 'income'], bin_size=5)

    # Custom bin size for numeric columns (larger bins = more privacy)
    >>> anonymized = apply_kanon(data, k=5, quasi_identifiers=['age', 'income'], bin_size=20)

    # Custom string generalization
    >>> anonymized = apply_kanon(
    ...     data, k=5,
    ...     quasi_identifiers=['region', 'occupation'],
    ...     string_method='first_char'  # "North" -> "N*"
    ... )

    # Custom hierarchies for specific columns
    >>> hierarchies = {
    ...     'region': {'North': 'Northern', 'South': 'Southern', 'East': 'Eastern', 'West': 'Western'},
    ...     'education': {'PhD': 'Higher', 'Masters': 'Higher', 'Bachelor': 'Basic', 'Secondary': 'Basic'}
    ... }
    >>> anonymized = apply_kanon(data, k=5, quasi_identifiers=['region', 'education'], hierarchies=hierarchies)

    # With metadata
    >>> anonymized, metadata = apply_kanon(data, k=5, quasi_identifiers=['age'], return_metadata=True)
    >>> print(metadata['k_anonymity_check'])
    """

    # Step 1: Validate or auto-detect quasi-identifiers
    if quasi_identifiers is not None:
        # Explicit QIs provided - VALIDATE
        is_valid, missing = validate_quasi_identifiers(data, quasi_identifiers)
        if not is_valid:
            raise ValueError(f"Quasi-identifiers not found in data: {missing}")

        # Use provided QIs
        use_qis = quasi_identifiers

    else:
        # No QIs provided - AUTO-DETECT with warning
        try:
            use_qis = auto_detect_quasi_identifiers(data, exclude_identifiers=True, max_qis=5)
            warnings.warn(
                f"No quasi-identifiers specified. Auto-detected QIs: {use_qis}. "
                f"Specify 'quasi_identifiers' parameter explicitly for better control.",
                UserWarning
            )
        except ValueError as e:
            raise ValueError(
                f"Cannot auto-detect quasi-identifiers: {e}. "
                f"Please specify 'quasi_identifiers' parameter explicitly."
            )

    import logging
    log = logging.getLogger(__name__)

    # Create a copy of the data
    protected_data = data.copy()

    log.info(
        f"[kANON] Starting: k={k}, strategy={strategy}, "
        f"QIs={use_qis}, shape={data.shape}, "
        f"max_suppression_rate={max_suppression_rate:.0%}")

    # Coerce object-dtype columns that Configure classified as numeric
    from sdc_engine.sdc.sdc_utils import coerce_columns_by_types
    coerce_columns_by_types(protected_data, column_types, use_qis)

    # Build generalization config
    gen_config = {
        'bin_size': bin_size,
        'string_method': string_method,
        'prefix_length': prefix_length,
        'hierarchies': hierarchies or {},
        'per_qi_bin_size': per_qi_bin_size or {},
        'column_types': column_types or {},
    }

    # Auto-build smart hierarchy objects for each QI (ARX-inspired)
    from sdc_engine.sdc.config import HIERARCHY_DEFAULTS
    if HIERARCHY_DEFAULTS.get('auto_build', True):
        from sdc_engine.sdc.hierarchies import build_hierarchy_for_column, Hierarchy
        hierarchy_objects = {}
        for qi in use_qis:
            user_h = (hierarchies or {}).get(qi)
            h = build_hierarchy_for_column(qi, protected_data, column_types, user_h)
            if h is not None:
                hierarchy_objects[qi] = h
                log.info("[kANON] Built hierarchy for '%s': %s", qi, h)
        gen_config['hierarchy_objects'] = hierarchy_objects
    else:
        gen_config['hierarchy_objects'] = {}

    # Apply k-anonymity based on strategy
    if strategy == 'generalization':
        protected_data = _achieve_kanon_generalization(
            protected_data, k, use_qis, gen_config, verbose=verbose,
            sensitive_columns=sensitive_columns, l_target=l_target,
            t_target=t_target,
        )
        # Check if k-anonymity was achieved, if not, fall back to suppression
        is_kanon, _, violations = check_kanonymity(protected_data, use_qis, k)
        if not is_kanon and len(violations) > 0:
            log.info(
                f"[kANON] Generalization left {len(violations)} violations — "
                f"falling back to suppression (budget {max_suppression_rate:.0%})")
            if verbose:
                print(f"Generalization alone didn't achieve k-anonymity ({len(violations)} violations). Applying suppression...")
            protected_data = _achieve_kanon_suppression(
                protected_data, k, use_qis, max_suppression_rate, verbose=verbose
            )
        else:
            log.info("[kANON] Generalization alone achieved k-anonymity")
    elif strategy == 'suppression':
        protected_data = _achieve_kanon_suppression(
            protected_data, k, use_qis, max_suppression_rate, verbose=verbose
        )
    elif strategy == 'hybrid':
        # First apply generalization
        protected_data = _achieve_kanon_generalization(
            protected_data, k, use_qis, gen_config, verbose=verbose,
            sensitive_columns=sensitive_columns, l_target=l_target,
            t_target=t_target,
        )
        # Then suppress any remaining violations
        protected_data = _achieve_kanon_suppression(
            protected_data, k, use_qis, max_suppression_rate, verbose=verbose
        )
    elif strategy == 'beam':
        protected_data = _beam_search_generalization(
            protected_data, k, use_qis, gen_config, verbose=verbose,
            sensitive_columns=sensitive_columns, l_target=l_target,
            t_target=t_target,
        )
        # Fall back to suppression if needed (same pattern as 'generalization')
        is_kanon, _, violations = check_kanonymity(protected_data, use_qis, k)
        if not is_kanon and len(violations) > 0:
            log.info(
                f"[kANON] Beam search left {len(violations)} violations — "
                f"falling back to suppression (budget {max_suppression_rate:.0%})")
            if verbose:
                print(f"Beam search didn't achieve k-anonymity ({len(violations)} violations). Applying suppression...")
            protected_data = _achieve_kanon_suppression(
                protected_data, k, use_qis, max_suppression_rate, verbose=verbose
            )
        else:
            log.info("[kANON] Beam search achieved k-anonymity")
    elif strategy == 'recursive':
        # Phase 1: Standard generalization
        protected_data = _achieve_kanon_generalization(
            protected_data, k, use_qis, gen_config, verbose=verbose,
            sensitive_columns=sensitive_columns, l_target=l_target,
            t_target=t_target,
        )
        # Phase 2: Standard suppression on remaining violations
        is_kanon, _, violations = check_kanonymity(protected_data, use_qis, k)
        if not is_kanon and len(violations) > 0:
            log.info(
                f"[kANON] Generalization left {len(violations)} violations — "
                f"applying suppression before recursive recoding "
                f"(budget {max_suppression_rate:.0%})")
            if verbose:
                print(f"Generalization left {len(violations)} violations. Applying suppression...")
            protected_data = _achieve_kanon_suppression(
                protected_data, k, use_qis, max_suppression_rate, verbose=verbose
            )
        # Phase 3: Recursive local recoding on suppressed records
        n_suppressed = int(
            protected_data[use_qis].isna().any(axis=1).sum())
        if n_suppressed > 0:
            log.info(
                f"[kANON] {n_suppressed:,} records suppressed — "
                f"starting recursive local recoding")
            protected_data = _recursive_local_recode(
                original_data=data,
                protected_data=protected_data,
                k=k,
                quasi_identifiers=use_qis,
                gen_config=gen_config,
                max_depth=2,
                verbose=verbose,
                sensitive_columns=sensitive_columns,
                l_target=l_target,
                t_target=t_target,
                max_suppression_rate=0.20,
            )
        else:
            log.info("[kANON] No suppressed records — recursive recoding skipped")
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Use 'generalization', 'suppression', 'hybrid', 'beam', or 'recursive'"
        )

    # Prepare return based on return_metadata parameter
    if return_metadata:
        # Verify k-anonymity and collect statistics
        is_kanon, group_sizes, violations = check_kanonymity(protected_data, use_qis, k)

        # Count suppressed values
        n_suppressed = protected_data[use_qis].isna().any(axis=1).sum()
        supp_rate = n_suppressed / len(protected_data) if len(protected_data) > 0 else 0

        log.info(
            f"[kANON] Done: k_anonymous={is_kanon}, "
            f"violations={len(violations)}, "
            f"suppressed={n_suppressed:,} ({supp_rate:.1%}), "
            f"equiv_classes={len(group_sizes)}")

        metadata = {
            'method': 'kANON',
            'parameters': {
                'k': k,
                'quasi_identifiers': use_qis,
                'strategy': strategy,
                'max_suppression_rate': max_suppression_rate,
                'bin_size': bin_size,
                'string_method': string_method,
                'prefix_length': prefix_length
            },
            'k_anonymity_check': {
                'is_k_anonymous': is_kanon,
                'k_value': k,
                'n_violations': len(violations)
            },
            'statistics': {
                'n_records': len(protected_data),
                'n_equivalence_classes': len(group_sizes),
                'min_group_size': int(group_sizes['count'].min()),
                'max_group_size': int(group_sizes['count'].max()),
                'avg_group_size': float(group_sizes['count'].mean()),
                'n_suppressed_records': int(n_suppressed),
                'suppression_rate': float(n_suppressed / len(protected_data))
            }
        }
        return protected_data, metadata
    else:
        return protected_data


def _generalize_column(series: pd.Series, col_name: str, gen_config: Dict, level: int = 0) -> pd.Series:
    """
    Generalize a single column based on its type and configuration.

    Parameters:
    -----------
    series : pd.Series
        The column data to generalize
    col_name : str
        Name of the column (for looking up custom hierarchies)
    gen_config : dict
        Generalization configuration containing bin_size, string_method, etc.
    level : int
        Generalization level (0 = first level, higher = more general)

    Returns:
    --------
    pd.Series : Generalized column
    """
    import logging
    _log = logging.getLogger(__name__)

    # Check for structured Hierarchy object first (ARX-inspired)
    hierarchy_objects = gen_config.get('hierarchy_objects', {})
    if col_name in hierarchy_objects:
        h = hierarchy_objects[col_name]
        effective_level = min(level + 1, h.max_level)  # level 0 in loop = hierarchy level 1
        if effective_level > 0:
            n_before = series.nunique()
            result = h.generalize(series, effective_level)
            n_after = result.nunique()
            _log.info(
                "[kANON-GEN] Hierarchy '%s' level %d: %d → %d (info_loss=%.3f)",
                col_name, effective_level, n_before, n_after,
                h.info_loss_at(effective_level))
            return result
        return series.copy()

    # Check for custom hierarchy (legacy dict format)
    if col_name in gen_config.get('hierarchies', {}):
        hierarchy = gen_config['hierarchies'][col_name]
        if isinstance(hierarchy, dict):
            # Apply custom mapping
            return series.map(hierarchy).fillna(series)

    # Determine if column is numeric or datetime — check pandas dtype first,
    # then fall back to column_types from Configure table
    _NUMERIC_KW = {'numeric', 'continuous', 'integer', 'float', 'coded'}
    _DATE_KW = {'date', 'datetime', 'temporal', 'time'}
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_datetime = pd.api.types.is_datetime64_any_dtype(series)

    if not is_numeric and not is_datetime:
        ct_label = gen_config.get('column_types', {}).get(col_name, '').lower()
        if ct_label:
            if any(kw in ct_label for kw in _NUMERIC_KW):
                # Configure says numeric but dtype is object — try coercion
                coerced = pd.to_numeric(series, errors='coerce')
                pct_valid = coerced.notna().mean()
                if pct_valid > 0.5:
                    _log.info(
                        "[kANON-GEN] Coerced '%s' to numeric (%.0f%% valid) — "
                        "Configure type: %s", col_name, pct_valid * 100, ct_label)
                    series = coerced
                    is_numeric = True
            elif any(kw in ct_label for kw in _DATE_KW):
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore", UserWarning)
                    # Try ISO first, then dayfirst
                    coerced = pd.to_datetime(series, errors='coerce')
                    if coerced.notna().mean() <= 0.5:
                        coerced = pd.to_datetime(series, errors='coerce', dayfirst=True)
                if coerced.notna().mean() > 0.5:
                    _log.info(
                        "[kANON-GEN] Coerced '%s' to datetime — "
                        "Configure type: %s", col_name, ct_label)
                    series = coerced
                    is_datetime = True

    # Also detect date strings by probing if not yet identified
    if not is_numeric and not is_datetime and series.dtype == object:
        _sample = series.dropna().head(50)
        if len(_sample) > 0:
            try:
                import warnings as _w
                # Try ISO format first (no dayfirst), then dayfirst for DD/MM/YYYY
                for _df_flag in (False, True):
                    with _w.catch_warnings():
                        _w.simplefilter("ignore", UserWarning)
                        parsed = pd.to_datetime(_sample, errors='coerce', dayfirst=_df_flag)
                    if parsed.notna().mean() > 0.8:
                        with _w.catch_warnings():
                            _w.simplefilter("ignore", UserWarning)
                            series = pd.to_datetime(series, errors='coerce', dayfirst=_df_flag)
                        is_datetime = True
                        _log.info("[kANON-GEN] Detected '%s' as date strings (dayfirst=%s)",
                                  col_name, _df_flag)
                        break
            except Exception:
                pass

    # Datetime column — bin by quarter/year to preserve temporal structure
    if is_datetime:
        from sdc_engine.sdc.GENERALIZE import _generalize_date_column
        # Scale max_categories by level: fewer periods at higher levels
        max_cats = max(3, 10 // (2 ** level))
        result = _generalize_date_column(series, max_categories=max_cats)
        n_before = series.nunique()
        n_after = result.nunique()
        _log.info(
            "[kANON-GEN] Date '%s': %d → %d periods (level %d, max_cats=%d)",
            col_name, n_before, n_after, level, max_cats)
        return result

    # Numeric column
    if is_numeric:
        # Per-QI treatment: use per_qi_bin_size if available, else global
        per_qi = gen_config.get('per_qi_bin_size', {})
        base_bin_size = per_qi.get(col_name, gen_config.get('bin_size', 10))
        # Scale bin_size to the column's actual value range
        # A base_bin_size of 10 is meaningless for a column spanning 0–1M
        vmin = series.min()
        vmax = series.max()
        value_range = vmax - vmin if pd.notna(vmin) and pd.notna(vmax) else 0
        n_unique = series.nunique()
        if value_range > 0 and n_unique > 100:
            # Target: ~50 bins at level 0, halving each level
            target_bins = max(5, 50 // (2 ** level))

            # Detect skewness — quantile binning for heavily skewed data
            try:
                _skew = abs(float(series.skew()))
            except Exception:
                _skew = 0.0

            if _skew > 3.0:
                _log.info(
                    "[kANON-GEN] Numeric '%s': skew=%.1f → quantile "
                    "binning (%d bins, level %d)",
                    col_name, _skew, target_bins, level)
                try:
                    valid = series.dropna()
                    binned = pd.qcut(valid, q=target_bins, duplicates='drop')
                    intervals = binned.cat.categories
                    labels = []
                    for iv in intervals:
                        lo, hi = iv.left, iv.right
                        if abs(lo) >= 10 and abs(hi) >= 10:
                            labels.append(f"{int(lo)}-{int(hi)}")
                        else:
                            labels.append(f"{lo:.1f}-{hi:.1f}")
                    label_map = dict(zip(intervals, labels))
                    result = series.copy().astype(object)
                    result[valid.index] = binned.map(label_map).astype(str)
                    result[series.isna()] = None
                    return result
                except Exception as _e:
                    _log.warning(
                        "[kANON-GEN] Quantile failed '%s': %s — "
                        "falling back to equal-width", col_name, _e)

            range_bin_size = value_range / target_bins
            # Use the larger of range-based and config-based bin_size
            bin_size = int(max(base_bin_size * (2 ** level), range_bin_size))
            bin_size = max(1, bin_size)  # floor at 1
            _log.info(
                "[kANON-GEN] Numeric '%s': range=%.0f, target_bins=%d, "
                "bin_size=%d (level %d)",
                col_name, value_range, target_bins, bin_size, level)
        else:
            bin_size = base_bin_size * (2 ** level)
        return generalize_numeric(series, bin_size=int(bin_size))

    # String/categorical column
    else:
        string_method = gen_config.get('string_method', 'prefix')
        base_prefix_len = gen_config.get('prefix_length', 3)

        # Get original cardinality
        n_unique_original = series.nunique()

        # ── Pre-binned range detection FIRST ──────────────────────
        # Check for numeric range patterns (e.g., "10-19", "1000-1999")
        # BEFORE prefix truncation, since prefix truncation destroys
        # the ordered structure of pre-binned numeric ranges.
        import re
        # Match lo-hi, lo–hi, (lo, hi], [lo, hi) — all common range formats
        _range_pat = re.compile(
            r'^[\(\[]?(-?[\d.]+)[,\s]*[-–,]\s*(-?[\d.]+)[\)\]]?$'
        )
        _sample_vals = series.dropna().unique()
        _range_matches = [_range_pat.match(str(v)) for v in _sample_vals]
        _is_range_col = (
            len(_sample_vals) >= 4
            and sum(1 for m in _range_matches if m) / len(_sample_vals) > 0.8
        )

        if _is_range_col and n_unique_original >= 5:
            # Parse and sort ranges by their lower bound
            _num_pat = _range_pat  # reuse the same pattern
            range_map = {}
            for v in _sample_vals:
                m = _num_pat.match(str(v))
                if m:
                    try:
                        lo = float(m.group(1))
                        hi = float(m.group(2))
                        range_map[str(v)] = (lo, hi)
                    except ValueError:
                        pass
            sorted_ranges = sorted(range_map.keys(), key=lambda x: range_map[x][0])

            # Dynamic target bins based on distribution uniformity:
            # Uniform bins → can merge more aggressively.
            # Skewed bins → preserve more bins to avoid creating giant groups.
            _bin_counts = series.value_counts()
            _range_counts = [int(_bin_counts.get(r, 0)) for r in sorted_ranges]
            if _range_counts:
                _mean_ct = np.mean(_range_counts) if _range_counts else 1
                _std_ct = np.std(_range_counts) if len(_range_counts) > 1 else 0
                _cv = _std_ct / _mean_ct if _mean_ct > 0 else 0
                # CV < 0.3 = uniform → floor at 5; CV > 1.0 = very skewed → floor at 8
                _dynamic_floor = int(np.clip(5 + _cv * 4, 5, 10))
            else:
                _dynamic_floor = 5

            # Cap floor: never prevent all merging — allow at least halving
            _dynamic_floor = min(_dynamic_floor, max(3, len(sorted_ranges) // 2))

            # If already at or below the floor, stop merging
            if len(sorted_ranges) <= _dynamic_floor:
                return series.copy()

            merge_factor = 2 ** (level + 1)
            target_bins = max(_dynamic_floor, len(sorted_ranges) // merge_factor)
            chunk_size = max(2, len(sorted_ranges) // target_bins)

            # Verify actual output stays above floor
            actual_groups = -(-len(sorted_ranges) // chunk_size)  # ceil div
            if actual_groups < _dynamic_floor:
                return series.copy()  # merging would go below floor

            merge_labels = {}
            for i in range(0, len(sorted_ranges), chunk_size):
                chunk = sorted_ranges[i:i + chunk_size]
                lo = range_map[chunk[0]][0]
                hi = range_map[chunk[-1]][1]
                # Use int formatting when values are whole numbers
                if lo == int(lo) and hi == int(hi):
                    label = f"{int(lo)}-{int(hi)}"
                else:
                    label = f"{lo}-{hi}"
                for r in chunk:
                    merge_labels[r] = label

            result = series.copy().astype(str)
            result = result.map(lambda x: merge_labels.get(x, x))
            n_after = result.nunique()
            if n_after < n_unique_original:
                _log.info(
                    "[kANON-GEN] Range merge '%s': %d → %d bins (level %d)",
                    col_name, n_unique_original, n_after, level)
                return result

        # ── Prefix truncation for non-range text columns ─────────
        # IMPORTANT: Skip prefix truncation for very-low-cardinality columns
        # (e.g., gender M/F) — truncating is nonsensical.  Also skip for
        # range-pattern columns — prefix truncation destroys numeric ranges.
        avg_len = series.dropna().astype(str).str.len().mean()
        skip_truncation = (
            _is_range_col
            or n_unique_original <= 5
            or avg_len <= 2
        )

        if not skip_truncation:
            _TRUNC_FLOOR = 5
            if string_method == 'suppress':
                return pd.Series(['*'] * len(series), index=series.index)

            _target = max(_TRUNC_FLOOR, n_unique_original // (2 ** (level + 1)))
            _max_prefix = int(avg_len) - 1
            _min_prefix = 1
            _best_trunc = None
            _best_dist = float('inf')

            for plen in range(_max_prefix, _min_prefix - 1, -1):
                trunc = generalize_string_prefix(series, prefix_length=plen)
                nu = trunc.nunique()
                if nu < n_unique_original and nu >= _TRUNC_FLOOR:
                    dist = abs(nu - _target)
                    if dist < _best_dist:
                        _best_dist = dist
                        _best_trunc = trunc

            if _best_trunc is not None:
                return _best_trunc

        # Frequency-based grouping: keep top-K categories, merge rest
        # into "Other".  Works for all cardinalities > 5 (including
        # short strings like state codes and moderate categoricals).
        # Skip for range-pattern columns — "Other" breaks ordered structure.
        if n_unique_original > 5 and not _is_range_col:
            keep_k = max(5, n_unique_original // (2 ** (level + 1)))
            top_cats = series.value_counts().head(keep_k).index
            result = series.copy()
            result[~result.isin(top_cats)] = 'Other'
            n_after = result.nunique()
            if n_after < n_unique_original:
                return result

        # Last resort: suppress to single category — only if cardinality ≤ 2
        if level >= 3 and n_unique_original <= 2:
            return pd.Series(['*'] * len(series), index=series.index)
        return series.copy()


def _achieve_kanon_generalization(
    data: pd.DataFrame,
    k: int,
    quasi_identifiers: List[str],
    gen_config: Dict,
    verbose: bool = True,
    sensitive_columns: Optional[List[str]] = None,
    l_target: Optional[int] = None,
    t_target: Optional[float] = None,
) -> pd.DataFrame:
    """
    Achieve k-anonymity (and optionally l-diversity and t-closeness) through
    iterative generalization.

    Uses a greedy approach with info-loss-aware QI scoring:
    1. Check current k-anonymity (and l-diversity/t-closeness if targets set)
    2. If violated, generalize the QI with best violations-per-info-loss ratio
    3. Repeat until all targets achieved or all QIs fully generalized
    4. Detect stalled iterations (no progress) and exit early
    """
    import logging
    log = logging.getLogger(__name__)

    protected_data = data.copy()
    generalization_levels = {qi: 0 for qi in quasi_identifiers}
    max_levels = 5  # Default maximum generalization levels per QI

    # Per-QI max_levels from hierarchy depth (overrides default)
    _hierarchy_objects = gen_config.get('hierarchy_objects', {})
    per_qi_max_levels = {}
    for qi in quasi_identifiers:
        if qi in _hierarchy_objects:
            per_qi_max_levels[qi] = _hierarchy_objects[qi].max_level
        else:
            per_qi_max_levels[qi] = max_levels


    # Log initial state and detect pre-binned range columns
    import re
    _NUMERIC_KW = {'numeric', 'continuous', 'integer', 'float', 'coded'}
    _ct = gen_config.get('column_types', {})
    _range_pat = re.compile(r'^[\(\[]?-?[\d.]+[,\s]*[-–,]\s*-?[\d.]+[\)\]]?$')
    _prebinned_qis = set()  # QIs that are already pre-binned numeric ranges

    for qi in quasi_identifiers:
        nuniq = protected_data[qi].nunique()
        pd_numeric = pd.api.types.is_numeric_dtype(protected_data[qi])
        ct_label = _ct.get(qi, '')
        ct_numeric = bool(ct_label and any(kw in ct_label.lower() for kw in _NUMERIC_KW))

        # Detect pre-binned range columns (from type-aware preprocessing)
        if not pd_numeric:
            sample_vals = protected_data[qi].dropna().unique()
            if len(sample_vals) >= 3:
                range_frac = sum(
                    1 for v in sample_vals if _range_pat.match(str(v))
                ) / len(sample_vals)
                if range_frac > 0.8:
                    _prebinned_qis.add(qi)

        if pd_numeric:
            dtype = 'numeric'
        elif qi in _prebinned_qis:
            dtype = 'pre-binned numeric ranges'
        elif ct_numeric:
            dtype = f'categorical (Configure says numeric: {ct_label})'
        else:
            dtype = 'categorical'
        log.info(f"[kANON-GEN] QI '{qi}': {nuniq} unique values, type={dtype}")

    if _prebinned_qis:
        log.info(
            f"[kANON-GEN] Pre-binned QIs detected: {_prebinned_qis} — "
            f"will use higher cardinality floors for range columns")

    # ── Feasibility-aware per-QI cardinality budgets ──────────────
    # Compute how many categories each QI can have so the combination
    # space (product of all QI cardinalities) fits within n_rows / k.
    n_rows = len(data)
    n_qis = len(quasi_identifiers)
    max_combo = n_rows / k  # max equivalence classes that can satisfy k

    # Each QI gets the nth root of max_combo as its cardinality target
    # This distributes the budget equally across QIs
    target_per_qi = int(max_combo ** (1.0 / n_qis)) if n_qis > 0 else 10
    # Global minimum: never go below 5 categories (was 3)
    _GLOBAL_MIN = 5
    # Pre-binned range columns get a higher floor
    _RANGE_MIN = max(5, target_per_qi)
    target_per_qi = max(_GLOBAL_MIN, target_per_qi)

    # qi_cardinality_targets stores the MINIMUM categories (floor) each QI
    # should retain.  Budget-aware: high-cardinality QIs get a higher floor
    # (up to target_per_qi) so they don't over-generalize, while low-card
    # QIs use _GLOBAL_MIN.
    qi_cardinality_targets = {}
    for qi in quasi_identifiers:
        qi_card = protected_data[qi].nunique()
        # Floor = min(target_per_qi, current cardinality) but never below _GLOBAL_MIN
        qi_cardinality_targets[qi] = max(_GLOBAL_MIN, min(target_per_qi, qi_card))

    combo_product = 1
    for qi in quasi_identifiers:
        combo_product *= protected_data[qi].nunique()
    feasibility = 'feasible' if combo_product <= max_combo * 2 else 'infeasible'

    log.info(
        f"[kANON-GEN] Feasibility: combo_space={combo_product:,.0f} vs "
        f"max_combo={max_combo:,.0f} → {feasibility}. "
        f"Target per QI ≈ {target_per_qi}, "
        f"per-QI targets: {qi_cardinality_targets}")

    iteration = 0
    # Scale max iterations with QI count so each QI gets enough levels
    max_iterations = max(10, len(quasi_identifiers) * max_levels)
    prev_violations = None
    stall_count = 0
    # Fewer stall retries for large datasets (each iteration is expensive)
    max_stalls = 2 if len(protected_data) > 10_000 else 3
    _gen_t0 = _time.monotonic()
    # Time budget: 8s base + 0.5s per 10K rows, max 20s
    _gen_time_budget = min(20, 8 + len(protected_data) / 20_000)

    while iteration < max_iterations:
        # Check current k-anonymity status
        is_kanon, group_sizes, violations = check_kanonymity(
            protected_data, quasi_identifiers, k
        )

        # Check l-diversity if target is set (only when close to k-anonymity)
        l_satisfied = True
        if l_target and sensitive_columns and is_kanon:
            # Only check when k-anonymity is achieved — performance guard
            try:
                from sdc_engine.sdc.post_protection_diagnostics import check_l_diversity
                l_result = check_l_diversity(
                    protected_data, quasi_identifiers, sensitive_columns,
                    l_target=l_target, size_threshold=200)
                l_satisfied = l_result.get('satisfied', True)
                if not l_satisfied:
                    log.info(
                        "[kANON-GEN] k-anonymity achieved but l-diversity NOT met "
                        "(l_achieved=%s, l_target=%d, violations=%d) — continuing",
                        l_result.get('l_achieved'), l_target,
                        l_result.get('violations', 0))
            except Exception as _e:
                log.warning("[kANON-GEN] l-diversity check failed: %s", _e)
                l_satisfied = True  # Don't block on check failure

        # Check t-closeness if target is set (only when k-anonymity achieved)
        t_satisfied = True
        if t_target and sensitive_columns and is_kanon:
            # Only check when k-anonymity is achieved — performance guard
            try:
                from sdc_engine.sdc.post_protection_diagnostics import check_t_closeness
                t_result = check_t_closeness(
                    protected_data, quasi_identifiers, sensitive_columns,
                    t_target=t_target, size_threshold=200)
                t_satisfied = t_result.get('satisfied', True)
                if not t_satisfied:
                    log.info(
                        "[kANON-GEN] k-anonymity achieved but t-closeness NOT met "
                        "(t_achieved=%s, t_target=%.2f, classes_checked=%d) — continuing",
                        t_result.get('t_achieved'), t_target,
                        t_result.get('classes_checked', 0))
            except Exception as _e:
                log.warning("[kANON-GEN] t-closeness check failed: %s", _e)
                t_satisfied = True  # Don't block on check failure

        if is_kanon and l_satisfied and t_satisfied:
            _targets = "k-anonymity"
            if l_target and sensitive_columns:
                _targets += f" AND l-diversity (l={l_target})"
            if t_target and sensitive_columns:
                _targets += f" AND t-closeness (t={t_target:.2f})"
            log.info(f"[kANON-GEN] {_targets} achieved after {iteration} iterations")
            if verbose:
                print(f"{_targets} achieved after {iteration} iterations")
            break

        current_violations = len(violations)
        n_eq_classes = len(group_sizes)
        log.info(
            f"[kANON-GEN] Iter {iteration}: {current_violations} violations "
            f"out of {n_eq_classes} equivalence classes (k={k})")
        if verbose:
            print(f"Iteration {iteration}: {current_violations} equivalence classes with < {k} records")

        # Time guard: bail early if generalization loop is taking too long
        _gen_elapsed = _time.monotonic() - _gen_t0
        if _gen_elapsed > _gen_time_budget:
            log.warning(
                f"[kANON-GEN] Time budget ({_gen_time_budget:.0f}s) exceeded "
                f"after {iteration} iterations — breaking out")
            break

        # Detect stalled iterations (no progress)
        if prev_violations is not None and current_violations >= prev_violations:
            stall_count += 1
            if stall_count >= max_stalls:
                # Before giving up, check if any QIs still have high cardinality
                # and can be forcibly generalized with wider bins
                high_card_qis = [
                    qi for qi in quasi_identifiers
                    if generalization_levels[qi] < per_qi_max_levels.get(qi, max_levels)
                    and protected_data[qi].nunique() > k * 2
                ]
                if high_card_qis:
                    log.info(
                        f"[kANON-GEN] Stalled but {len(high_card_qis)} QIs still "
                        f"have high cardinality — forcing aggressive generalization")
                    stall_count = 0  # Reset and keep trying
                else:
                    log.warning(
                        f"[kANON-GEN] Stalled for {max_stalls} iterations — "
                        f"generalization cannot reduce {current_violations} violations")
                    if verbose:
                        print(f"Warning: No progress for {max_stalls} iterations. Generalization cannot achieve k-anonymity.")
                    break
        else:
            stall_count = 0  # Reset stall counter if progress was made

        prev_violations = current_violations

        # Find which QI to generalize next — violation-aware scoring
        # Score each QI by how many violation rows its generalization would
        # help merge (cardinality within violation groups), with treatment
        # boost as tiebreaker.
        _per_qi = gen_config.get('per_qi_bin_size', {})
        qi_to_generalize = None
        best_score = -1.0

        # Build violation rows once per iteration via merge (no Python loop)
        if len(violations) > 0:
            _viol_keys = violations[quasi_identifiers].copy()
            _viol_df = protected_data.merge(
                _viol_keys, on=quasi_identifiers, how='inner')
        else:
            _viol_df = pd.DataFrame()

        _n_viol_rows = len(_viol_df)

        for qi in quasi_identifiers:
            qi_max = per_qi_max_levels.get(qi, max_levels)
            if generalization_levels[qi] >= qi_max:
                continue  # Already at max generalization

            # Primary score: cardinality of this QI within violation groups
            # Higher cardinality → more fragmentation → generalizing this QI
            # will merge more violation groups
            if _n_viol_rows > 0 and qi in _viol_df.columns:
                viol_card = _viol_df[qi].nunique()
            else:
                viol_card = 0
            # Normalize to [0, 1] range by overall cardinality
            overall_card = max(protected_data[qi].nunique(), 1)
            frag_ratio = viol_card / overall_card

            # Info-loss-aware scoring: "most violations per unit info loss"
            # Prefer cheap generalizations (low info loss) that resolve many violations
            if qi in _hierarchy_objects:
                h = _hierarchy_objects[qi]
                cur_lvl = generalization_levels[qi]
                next_lvl = min(cur_lvl + 1, h.max_level)
                # Marginal info loss for the next generalization step
                marginal_loss = h.info_loss_at(next_lvl) - h.info_loss_at(cur_lvl)
                marginal_loss = max(marginal_loss, 0.01)  # floor to avoid div-by-zero
                # Score = fragmentation reduction per unit info loss
                score = frag_ratio / marginal_loss
            else:
                # No hierarchy: fall back to plain fragmentation ratio
                score = frag_ratio

            # Secondary: prefer less-generalized columns (room to reduce)
            score += 0.1 * (qi_max - generalization_levels[qi]) / max(qi_max, 1)

            # Tiebreaker: Heavy treatment QIs go first
            if _per_qi:
                qi_bs = _per_qi.get(qi, gen_config.get('bin_size', 10))
                global_bs = gen_config.get('bin_size', 10)
                if qi_bs > global_bs:
                    score += 0.05  # Heavy
                elif qi_bs == global_bs:
                    score += 0.02  # Standard

            if score > best_score:
                best_score = score
                qi_to_generalize = qi

        if qi_to_generalize is None:
            log.warning("[kANON-GEN] All QIs at max generalization level — cannot achieve k-anonymity")
            if verbose:
                print("Warning: Cannot achieve k-anonymity with current hierarchies (all QIs at max level)")
            break

        # Per-column cardinality floor: don't collapse below target
        # to preserve analytical value (especially for pre-binned ranges)
        before_nuniq = protected_data[qi_to_generalize].nunique()
        _qi_floor = qi_cardinality_targets.get(qi_to_generalize, _GLOBAL_MIN)
        if before_nuniq <= _qi_floor:
            # Already at floor — skip this QI, mark as max level
            generalization_levels[qi_to_generalize] = max_levels
            log.info(
                f"[kANON-GEN] Skipping '{qi_to_generalize}' — "
                f"already at cardinality floor ({before_nuniq} ≤ {_qi_floor})")
            iteration += 1
            continue

        # Apply generalization to selected QI
        # Use already-generalized data as input so each level builds on the
        # previous reduction.  Generalizing from original data can produce
        # *more* categories than the previous level (different bin boundaries).
        _prev_col = protected_data[qi_to_generalize].copy()  # save for revert
        protected_data[qi_to_generalize] = _generalize_column(
            _prev_col,
            qi_to_generalize,
            gen_config,
            level=generalization_levels[qi_to_generalize]
        )
        after_nuniq = protected_data[qi_to_generalize].nunique()

        # Enforce cardinality floor: revert if too few categories
        if after_nuniq < _qi_floor and before_nuniq >= _qi_floor:
            if generalization_levels[qi_to_generalize] == 0:
                # Even level 0 is too aggressive — keep original unchanged
                protected_data[qi_to_generalize] = data[qi_to_generalize].copy()
            else:
                # Revert to saved previous-level state
                protected_data[qi_to_generalize] = _prev_col
            generalization_levels[qi_to_generalize] = max_levels  # mark as done
            reverted_nuniq = protected_data[qi_to_generalize].nunique()
            log.info(
                f"[kANON-GEN] Reverted '{qi_to_generalize}' — "
                f"would have gone to {after_nuniq} (below floor {_qi_floor}), "
                f"kept at {reverted_nuniq}")
            after_nuniq = reverted_nuniq

        # l-diversity revert check: if this generalization significantly
        # worsens l-diversity, revert it.  Only check when we're close to
        # k-anonymity (< 20% violation rate) to avoid expensive checks early.
        if (l_target and sensitive_columns
                and current_violations < n_eq_classes * 0.20):
            try:
                from sdc_engine.sdc.post_protection_diagnostics import check_l_diversity
                l_after = check_l_diversity(
                    protected_data, quasi_identifiers, sensitive_columns,
                    l_target=l_target, size_threshold=100)
                l_viols_after = l_after.get('violations', 0)

                # Check if l-diversity got significantly worse
                if l_viols_after > 0 and l_after.get('l_achieved', l_target) < l_target:
                    # Check previous state
                    _temp_data = protected_data.copy()
                    _temp_data[qi_to_generalize] = _prev_col
                    l_before = check_l_diversity(
                        _temp_data, quasi_identifiers, sensitive_columns,
                        l_target=l_target, size_threshold=100)
                    l_viols_before = l_before.get('violations', 0)

                    if l_viols_after > l_viols_before * 1.5:
                        # Revert: this generalization merged equiv classes
                        # and reduced sensitive value diversity
                        protected_data[qi_to_generalize] = _prev_col
                        qi_max = per_qi_max_levels.get(qi_to_generalize, max_levels)
                        generalization_levels[qi_to_generalize] = qi_max  # skip this QI
                        log.info(
                            "[kANON-GEN] Reverted '%s': l-diversity degraded "
                            "(%d → %d violations)", qi_to_generalize,
                            l_viols_before, l_viols_after)
                        iteration += 1
                        continue
            except Exception as _e:
                log.warning("[kANON-GEN] l-diversity revert check failed: %s", _e)

        # t-closeness revert check: if this generalization significantly
        # worsens t-closeness, revert it.  Only check when we're close to
        # k-anonymity (< 20% violation rate) to avoid expensive checks early.
        if (t_target and sensitive_columns
                and current_violations < n_eq_classes * 0.20):
            try:
                from sdc_engine.sdc.post_protection_diagnostics import check_t_closeness
                t_after = check_t_closeness(
                    protected_data, quasi_identifiers, sensitive_columns,
                    t_target=t_target, size_threshold=100)
                t_dist_after = t_after.get('t_achieved') or 0.0

                # Check if t-closeness got significantly worse
                if t_dist_after > t_target and not t_after.get('satisfied', True):
                    # Check previous state
                    _temp_data = protected_data.copy()
                    _temp_data[qi_to_generalize] = _prev_col
                    t_before = check_t_closeness(
                        _temp_data, quasi_identifiers, sensitive_columns,
                        t_target=t_target, size_threshold=100)
                    t_dist_before = t_before.get('t_achieved') or 0.0

                    if t_dist_after > t_dist_before * 1.5:
                        # Revert: this generalization merged equiv classes
                        # and worsened distribution closeness
                        protected_data[qi_to_generalize] = _prev_col
                        qi_max = per_qi_max_levels.get(qi_to_generalize, max_levels)
                        generalization_levels[qi_to_generalize] = qi_max  # skip this QI
                        log.info(
                            "[kANON-GEN] Reverted '%s': t-closeness degraded "
                            "(%.3f → %.3f, target=%.2f)", qi_to_generalize,
                            t_dist_before, t_dist_after, t_target)
                        iteration += 1
                        continue
            except Exception as _e:
                log.warning("[kANON-GEN] t-closeness revert check failed: %s", _e)

        generalization_levels[qi_to_generalize] += 1
        log.info(
            f"[kANON-GEN] Generalized '{qi_to_generalize}' "
            f"level {generalization_levels[qi_to_generalize]}: "
            f"{before_nuniq} → {after_nuniq} unique values")
        iteration += 1

    return protected_data


# ---------------------------------------------------------------------------
#  Beam search generalization (ARX-inspired lattice exploration)
# ---------------------------------------------------------------------------

def _beam_search_generalization(
    data: pd.DataFrame,
    k: int,
    quasi_identifiers: List[str],
    gen_config: Dict,
    verbose: bool = True,
    sensitive_columns: Optional[List[str]] = None,
    l_target: Optional[int] = None,
    t_target: Optional[float] = None,
    beam_width: int = 5,
) -> pd.DataFrame:
    """Explore multiple generalization paths simultaneously (beam search).

    Instead of the greedy approach (generalize one QI at a time), this
    maintains a *beam* of the top-B generalization states and expands all
    of them each iteration, keeping only the best successors.  This lets
    the algorithm discover combinations that the greedy path misses.

    A "state" is a tuple of generalization levels, one per QI.
    E.g. ``(0, 0, 0)`` = no generalization; ``(1, 0, 2)`` = QI[0] at
    level 1, QI[2] at level 2.

    Parameters
    ----------
    data : pd.DataFrame
        Input data (will not be mutated).
    k : int
        Minimum equivalence-class size.
    quasi_identifiers : list[str]
        QI column names.
    gen_config : dict
        Generalization config (same as greedy).
    verbose : bool
        Print progress messages.
    sensitive_columns : list[str] or None
        Sensitive columns for l-diversity / t-closeness checks.
    l_target : int or None
        l-diversity target (checked only when k-anonymity is met).
    t_target : float or None
        t-closeness target (reserved for future use).
    beam_width : int
        Number of candidate states kept per iteration (default 5).

    Returns
    -------
    pd.DataFrame
        Best k-anonymous (and optionally l-diverse) DataFrame found,
        or falls back to greedy if no solution is found.
    """
    import logging
    log = logging.getLogger(__name__)

    n_rows = len(data)
    n_qis = len(quasi_identifiers)

    # ── Performance guards ──────────────────────────────────────────
    if beam_width * n_qis > 50:
        beam_width = max(2, 50 // n_qis)
        log.info("[kANON-BEAM] Reduced beam_width to %d (QI fan-out guard)", beam_width)
    if n_rows > 50_000:
        beam_width = min(beam_width, 2)
        log.info("[kANON-BEAM] Reduced beam_width to %d (large dataset guard)", beam_width)

    # Time budget: 8s base + 0.5s per 10K rows, max 20s
    _time_budget = min(20, 8 + n_rows / 20_000)
    _t0 = _time.monotonic()

    # ── Per-QI max levels from hierarchy depth ──────────────────────
    _hierarchy_objects = gen_config.get('hierarchy_objects', {})
    max_levels_per_qi: List[int] = []
    for qi in quasi_identifiers:
        if qi in _hierarchy_objects:
            max_levels_per_qi.append(_hierarchy_objects[qi].max_level)
        else:
            max_levels_per_qi.append(5)

    log.info(
        "[kANON-BEAM] Starting: k=%d, beam_width=%d, QIs=%s, "
        "max_levels=%s, time_budget=%.0fs",
        k, beam_width, quasi_identifiers, max_levels_per_qi, _time_budget)

    # ── State helpers ───────────────────────────────────────────────
    # Cache: state tuple -> (DataFrame, n_violations, total_info_loss)
    _cache: Dict[tuple, Tuple[pd.DataFrame, int, float]] = {}

    def _apply_state(state: tuple) -> pd.DataFrame:
        """Apply a generalization state to the original data."""
        df = data.copy()
        for i, (qi, level) in enumerate(zip(quasi_identifiers, state)):
            if level == 0:
                continue
            # Apply each level incrementally for consistency with
            # _generalize_column's level semantics (level 0 = first gen).
            for lvl in range(level):
                df[qi] = _generalize_column(df[qi], qi, gen_config, level=lvl)
        return df

    def _score_state(state: tuple) -> Tuple[pd.DataFrame, int, float]:
        """Return (DataFrame, n_violations, total_info_loss) for *state*."""
        if state in _cache:
            return _cache[state]

        df = _apply_state(state)
        is_kanon, group_sizes, violations = check_kanonymity(df, quasi_identifiers, k)
        n_violations = 0 if is_kanon else len(violations)

        # Total info loss across all QIs (from hierarchy objects where available)
        total_loss = 0.0
        for i, (qi, level) in enumerate(zip(quasi_identifiers, state)):
            if qi in _hierarchy_objects and level > 0:
                total_loss += _hierarchy_objects[qi].info_loss_at(level)
            elif level > 0:
                # Approximate info loss for QIs without hierarchy:
                # use cardinality reduction ratio
                orig_card = max(data[qi].nunique(), 1)
                gen_card = max(df[qi].nunique(), 1)
                total_loss += 1.0 - (gen_card / orig_card)

        _cache[state] = (df, n_violations, total_loss)
        return df, n_violations, total_loss

    def _privacy_satisfied(df: pd.DataFrame) -> bool:
        """Check k-anonymity + optional l-diversity + t-closeness on *df*."""
        is_kanon, _, _ = check_kanonymity(df, quasi_identifiers, k)
        if not is_kanon:
            return False
        if l_target and sensitive_columns:
            try:
                from sdc_engine.sdc.post_protection_diagnostics import check_l_diversity
                l_res = check_l_diversity(
                    df, quasi_identifiers, sensitive_columns,
                    l_target=l_target, size_threshold=200)
                if not l_res.get('satisfied', True):
                    return False
            except Exception:
                pass  # Don't block on check failure
        if t_target and sensitive_columns:
            try:
                from sdc_engine.sdc.post_protection_diagnostics import check_t_closeness
                t_res = check_t_closeness(
                    df, quasi_identifiers, sensitive_columns,
                    t_target=t_target, size_threshold=200)
                if not t_res.get('satisfied', True):
                    return False
            except Exception:
                pass  # Don't block on check failure
        return True

    # ── Initial beam ────────────────────────────────────────────────
    initial_state = tuple(0 for _ in quasi_identifiers)
    beam: List[tuple] = [initial_state]

    # Check if already k-anonymous
    df0, v0, _ = _score_state(initial_state)
    if v0 == 0 and _privacy_satisfied(df0):
        log.info("[kANON-BEAM] Already k-anonymous — no generalization needed")
        return df0

    best_solution: Optional[Tuple[tuple, pd.DataFrame]] = None  # (state, df)
    best_solution_loss = float('inf')

    iteration = 0
    max_iterations = sum(max_levels_per_qi) + 5  # generous upper bound

    while iteration < max_iterations:
        iteration += 1

        # Time guard
        elapsed = _time.monotonic() - _t0
        if elapsed > _time_budget:
            log.warning(
                "[kANON-BEAM] Time budget (%.0fs) exceeded at iter %d",
                _time_budget, iteration)
            break

        # ── Expand: generate all successors ─────────────────────────
        successors: List[tuple] = []
        seen: set = set()

        for state in beam:
            for i in range(n_qis):
                new_level = state[i] + 1
                if new_level > max_levels_per_qi[i]:
                    continue  # already at max for this QI
                successor = list(state)
                successor[i] = new_level
                successor_t = tuple(successor)
                if successor_t not in seen:
                    successors.append(successor_t)
                    seen.add(successor_t)

        if not successors:
            log.info("[kANON-BEAM] No more successors — all states fully generalized")
            break

        # ── Score all successors ────────────────────────────────────
        scored: List[Tuple[tuple, int, float]] = []
        for s in successors:
            # Time guard inside expansion loop
            if _time.monotonic() - _t0 > _time_budget:
                break
            df_s, n_viol, loss = _score_state(s)

            # Check if this state satisfies all privacy criteria
            if n_viol == 0:
                if _privacy_satisfied(df_s):
                    # Found a valid solution — track the best (lowest loss)
                    if loss < best_solution_loss:
                        best_solution = (s, df_s)
                        best_solution_loss = loss
                        log.info(
                            "[kANON-BEAM] Solution found at state %s "
                            "(info_loss=%.3f)", s, loss)

            scored.append((s, n_viol, loss))

        if not scored:
            break

        # If we found at least one solution, check if we should stop.
        # Continue one more iteration to see if a lower-loss solution exists
        # among states at the same depth.
        if best_solution is not None:
            # Stop if all current successors have equal or higher loss
            all_worse = all(
                loss >= best_solution_loss for _, _, loss in scored
            )
            if all_worse:
                log.info(
                    "[kANON-BEAM] Best solution confirmed at state %s "
                    "(no successor has lower loss)", best_solution[0])
                break

        # ── Select top-B successors for next beam ───────────────────
        # Sort by (n_violations ASC, info_loss ASC)
        scored.sort(key=lambda x: (x[1], x[2]))
        beam = [s for s, _, _ in scored[:beam_width]]

        if verbose and iteration % 3 == 0:
            top_state, top_viol, top_loss = scored[0]
            print(
                f"  Beam iter {iteration}: best state {top_state}, "
                f"{top_viol} violations, info_loss={top_loss:.3f}, "
                f"cache_size={len(_cache)}")

        log.info(
            "[kANON-BEAM] Iter %d: beam=%s, top_violations=%d, "
            "top_loss=%.3f, cache=%d",
            iteration, beam, scored[0][1], scored[0][2], len(_cache))

    # ── Return best result ──────────────────────────────────────────
    if best_solution is not None:
        state, df = best_solution
        log.info(
            "[kANON-BEAM] Returning solution: state=%s, info_loss=%.3f, "
            "iterations=%d, cache_size=%d",
            state, best_solution_loss, iteration, len(_cache))
        if verbose:
            print(
                f"Beam search found k-anonymous solution: "
                f"state={state}, info_loss={best_solution_loss:.3f}")
        return df

    # No full solution — fall back to greedy
    log.warning(
        "[kANON-BEAM] No k-anonymous solution found after %d iterations. "
        "Falling back to greedy.", iteration)
    if verbose:
        print("Beam search did not find full solution — falling back to greedy")

    return _achieve_kanon_generalization(
        data, k, quasi_identifiers, gen_config, verbose=verbose,
        sensitive_columns=sensitive_columns, l_target=l_target,
        t_target=t_target,
    )


def _achieve_kanon_suppression(
    data: pd.DataFrame,
    k: int,
    quasi_identifiers: List[str],
    max_suppression_rate: float,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Achieve k-anonymity through hybrid local suppression.

    Two-phase approach that preserves more QI data than full-row suppression:

    **Phase 1 — Targeted suppression (70% of budget):**
    Iteratively suppress only the highest-cardinality QI for violation groups.
    After each round, re-compute equivalence classes — previous suppressions
    may have merged groups and resolved violations. This touches more rows
    but blanks only 1 QI per row per round, preserving analytical value.

    **Phase 2 — Full-row cleanup (30% of budget):**
    For residual violations, suppress all QIs (standard approach) to maximize
    k-anonymity achievement within the remaining budget.

    Respects ``max_suppression_rate`` overall. Smallest groups (highest risk)
    are suppressed first in both phases.
    """
    import logging
    log = logging.getLogger(__name__)

    protected_data = data.copy()
    n_total = len(data)

    # Find records in groups smaller than k
    group_counts = protected_data.groupby(quasi_identifiers, dropna=False).size()
    small_groups = group_counts[group_counts < k].sort_values()

    # Total records that NEED suppression
    total_need = int(small_groups.sum())
    need_rate = total_need / n_total if n_total > 0 else 0
    max_allowed = int(max_suppression_rate * n_total)

    log.info(
        f"[kANON-SUPPRESS] {total_need:,} records in {len(small_groups)} "
        f"groups < k={k} (need {need_rate:.1%} suppression, "
        f"budget {max_suppression_rate:.0%} = {max_allowed:,} records)")

    if total_need == 0:
        log.info("[kANON-SUPPRESS] No records to suppress — already k-anonymous")
        return protected_data

    # ── Phase 1: Targeted suppression (full budget) ─────────────
    # Suppress only the highest-cardinality QI per round, then re-group.
    # Gets the full row budget since partial suppression (1 QI per row)
    # is far less destructive than full-row suppression (all QIs per row).
    phase1_row_budget = max_allowed
    rows_suppressed_p1 = 0
    max_rounds = 8

    for round_num in range(max_rounds):
        is_kanon, _, violations = check_kanonymity(
            protected_data, quasi_identifiers, k)
        if is_kanon:
            log.info(
                f"[kANON-SUPPRESS] Phase 1 achieved k-anonymity "
                f"after {round_num} targeted rounds")
            break

        group_counts = protected_data.groupby(
            quasi_identifiers, dropna=False).size()
        small_groups = group_counts[group_counts < k].sort_values()
        if len(small_groups) == 0:
            break

        # Re-rank QIs by current cardinality each round
        qi_cards = {qi: protected_data[qi].nunique()
                    for qi in quasi_identifiers}
        qi_ranked = sorted(qi_cards, key=qi_cards.get, reverse=True)

        # Pick the QI with highest cardinality that still has >50% values
        target_qi = None
        for qi in qi_ranked:
            non_null_rate = protected_data[qi].notna().mean()
            if non_null_rate > 0.5:
                target_qi = qi
                break
        if target_qi is None:
            log.info(
                f"[kANON-SUPPRESS] Phase 1: no QI with >50%% non-null "
                f"values — switching to phase 2")
            break

        round_rows = 0
        for combo, count in small_groups.items():
            if rows_suppressed_p1 + count > phase1_row_budget:
                break

            if isinstance(combo, tuple):
                mask = pd.Series(True, index=protected_data.index)
                for qi, val in zip(quasi_identifiers, combo):
                    if pd.isna(val):
                        mask &= protected_data[qi].isna()
                    else:
                        mask &= (protected_data[qi] == val)
            else:
                qi0 = quasi_identifiers[0]
                mask = ((protected_data[qi0] == combo)
                        if not pd.isna(combo)
                        else protected_data[qi0].isna())

            # Suppress ONLY the target QI (not all)
            protected_data.loc[mask, target_qi] = np.nan
            rows_suppressed_p1 += int(count)
            round_rows += int(count)

        log.info(
            f"[kANON-SUPPRESS] Phase 1 round {round_num}: "
            f"suppressed '{target_qi}' for {round_rows:,} records "
            f"({len(violations)} violations remaining)")
        if round_rows == 0:
            break

    # ── Phase 2: Full-row cleanup (remaining budget) ────────────────
    # Use whatever's left from the row budget. Phase 1 may not have used
    # the full budget if k-anonymity was achieved early.
    is_kanon, _, violations = check_kanonymity(
        protected_data, quasi_identifiers, k)
    if not is_kanon and len(violations) > 0:
        rows_already_affected = int(
            protected_data[quasi_identifiers].isna().any(axis=1).sum())
        phase2_allowed = max(0, max_allowed - rows_already_affected)
        if phase2_allowed < 10:
            # No budget left — extend by 5% for full-row cleanup
            phase2_allowed = int(n_total * 0.05)

        group_counts = protected_data.groupby(
            quasi_identifiers, dropna=False).size()
        small_groups = group_counts[group_counts < k].sort_values()
        phase2_need = int(small_groups.sum())

        log.info(
            f"[kANON-SUPPRESS] Phase 2: {phase2_need:,} records in "
            f"{len(small_groups)} groups still < k={k}. "
            f"Full-row budget: {phase2_allowed:,}")

        phase2_suppressed = 0
        for combo, count in small_groups.items():
            if phase2_suppressed + count > phase2_allowed:
                break

            if isinstance(combo, tuple):
                mask = pd.Series(True, index=protected_data.index)
                for qi, val in zip(quasi_identifiers, combo):
                    if pd.isna(val):
                        mask &= protected_data[qi].isna()
                    else:
                        mask &= (protected_data[qi] == val)
            else:
                qi0 = quasi_identifiers[0]
                mask = ((protected_data[qi0] == combo)
                        if not pd.isna(combo)
                        else protected_data[qi0].isna())

            for qi in quasi_identifiers:
                protected_data.loc[mask, qi] = np.nan
            phase2_suppressed += int(count)

        log.info(
            f"[kANON-SUPPRESS] Phase 2: full-row suppressed "
            f"{phase2_suppressed:,} records")

    # ── Final stats ───────────────────────────────────────────────
    nan_rows = protected_data[quasi_identifiers].isna().any(axis=1).sum()
    nan_cells = protected_data[quasi_identifiers].isna().sum().sum()
    actual_row_rate = nan_rows / n_total if n_total > 0 else 0

    is_kanon_final, _, viol_final = check_kanonymity(
        protected_data, quasi_identifiers, k)

    log.info(
        f"[kANON-SUPPRESS] Final: rows_affected={nan_rows:,} "
        f"({actual_row_rate:.1%}), nan_cells={nan_cells:,}, "
        f"k_anonymous={is_kanon_final}, violations={len(viol_final)}")

    if not is_kanon_final:
        log.warning(
            f"[kANON-SUPPRESS] Could not achieve k-anonymity within "
            f"{max_suppression_rate:.0%} budget. "
            f"{len(viol_final)} violations remain. "
            f"Consider a different method (LOCSUPR).")

    if verbose:
        print(f"Suppressed {nan_rows:,} rows ({actual_row_rate:.1%}), "
              f"{nan_cells:,} cells. k-anonymous={is_kanon_final}")

    return protected_data


def _recursive_local_recode(
    original_data: pd.DataFrame,
    protected_data: pd.DataFrame,
    k: int,
    quasi_identifiers: List[str],
    gen_config: Dict,
    max_depth: int = 2,
    verbose: bool = True,
    sensitive_columns: Optional[List[str]] = None,
    l_target: Optional[int] = None,
    t_target: Optional[float] = None,
    max_suppression_rate: float = 0.20,
    _time_budget: Optional[float] = None,
) -> pd.DataFrame:
    """
    ARX-inspired recursive local recoding on suppressed records.

    After standard generalization + suppression, some records are suppressed
    (QI cells set to NaN). Instead of losing them, this function:

    1. Identifies suppressed rows (any QI column is NaN).
    2. Restores their original QI values from the input data.
    3. Re-runs generalization on the suppressed subset with MORE aggressive
       parameters (doubled bin_size, +1 starting generalization level).
    4. Re-runs suppression on the result with a generous budget (20%).
    5. Merges recovered rows back into the main result.
    6. Recurses on any remaining suppressions (depth + 1), up to *max_depth*.

    Parameters
    ----------
    original_data : pd.DataFrame
        The original (pre-anonymization) data — used to restore QI values
        for suppressed rows.
    protected_data : pd.DataFrame
        The already-anonymized data containing NaN in QI columns for
        suppressed records.
    k : int
        Minimum group size for k-anonymity.
    quasi_identifiers : list of str
        QI column names.
    gen_config : dict
        Generalization configuration (bin_size, hierarchy_objects, etc.).
    max_depth : int, default 2
        Maximum recursion depth. Each level uses more aggressive parameters.
    verbose : bool, default True
        Print progress messages.
    sensitive_columns : list of str, optional
        Sensitive columns for l-diversity / t-closeness checks.
    l_target : int, optional
        l-diversity target.
    t_target : float, optional
        t-closeness target.
    max_suppression_rate : float, default 0.20
        Suppression budget for each recursion level (more generous than the
        top-level budget since we are working on already-problematic records).
    _time_budget : float, optional
        Wall-clock deadline (``time.time()`` value). If set, recursion stops
        when time runs out.

    Returns
    -------
    pd.DataFrame
        The protected data with fewer suppressed rows — recovered records
        have re-anonymized QI values instead of NaN.
    """
    import logging
    log = logging.getLogger(__name__)

    result = protected_data.copy()
    n_total = len(result)

    for depth in range(1, max_depth + 1):
        # ── Time guard ──────────────────────────────────────────────
        if _time_budget is not None and _time.time() > _time_budget:
            log.info(
                "[kANON-RECURSIVE] Depth %d: time budget exhausted — stopping",
                depth)
            break

        # ── Identify suppressed rows ────────────────────────────────
        suppressed_mask = result[quasi_identifiers].isna().any(axis=1)
        n_suppressed = int(suppressed_mask.sum())

        if n_suppressed == 0:
            log.info(
                "[kANON-RECURSIVE] Depth %d: no suppressed rows — done",
                depth)
            break

        log.info(
            "[kANON-RECURSIVE] Depth %d: %d suppressed rows (%.1f%%) — "
            "attempting local recoding",
            depth, n_suppressed, 100 * n_suppressed / n_total)

        if verbose:
            print(
                f"Recursive local recode depth {depth}: "
                f"{n_suppressed:,} suppressed rows — re-anonymizing...")

        # ── Restore original QI values for suppressed rows ──────────
        suppressed_idx = result.index[suppressed_mask]
        subset = original_data.loc[suppressed_idx].copy()

        if len(subset) < k:
            log.info(
                "[kANON-RECURSIVE] Depth %d: only %d rows — fewer than k=%d, "
                "cannot form any equivalence class — stopping",
                depth, len(subset), k)
            break

        # ── Build more aggressive gen_config ────────────────────────
        aggressive_config = deepcopy(gen_config)

        # Double the bin_size at each depth (numeric QIs)
        base_bin = gen_config.get('bin_size', 10)
        aggressive_config['bin_size'] = base_bin * (2 ** depth)

        # Halve max_categories at each depth (categorical QIs)
        base_max_cat = gen_config.get('max_categories', 10)
        aggressive_config['max_categories'] = max(
            2, base_max_cat // (2 ** depth))

        # Also scale per-QI bin sizes
        per_qi = gen_config.get('per_qi_bin_size', {})
        if per_qi:
            aggressive_config['per_qi_bin_size'] = {
                qi: sz * (2 ** depth) for qi, sz in per_qi.items()
            }

        # Reduce prefix length for string columns (more aggressive)
        base_prefix = gen_config.get('prefix_length', 3)
        aggressive_config['prefix_length'] = max(1, base_prefix - depth)

        # ── Re-generalize suppressed subset ─────────────────────────
        # Use a higher starting generalization level (+depth) by
        # pre-applying ``depth`` rounds of generalization before
        # handing off to the standard loop.
        recoded_subset = subset.copy()
        for qi in quasi_identifiers:
            for pre_level in range(depth):
                recoded_subset[qi] = _generalize_column(
                    recoded_subset[qi], qi, aggressive_config, level=pre_level)

        recoded_subset = _achieve_kanon_generalization(
            recoded_subset, k, quasi_identifiers, aggressive_config,
            verbose=False,
            sensitive_columns=sensitive_columns,
            l_target=l_target,
            t_target=t_target,
        )

        # ── Suppress remaining violations in the subset ─────────────
        is_kanon, _, violations = check_kanonymity(
            recoded_subset, quasi_identifiers, k)
        if not is_kanon and len(violations) > 0:
            recoded_subset = _achieve_kanon_suppression(
                recoded_subset, k, quasi_identifiers,
                max_suppression_rate, verbose=False)

        # ── Identify recovered rows (not suppressed after recoding) ─
        still_suppressed = recoded_subset[quasi_identifiers].isna().any(axis=1)
        recovered_mask = ~still_suppressed
        n_recovered = int(recovered_mask.sum())
        n_still_bad = int(still_suppressed.sum())

        log.info(
            "[kANON-RECURSIVE] Depth %d: recovered %d / %d suppressed rows "
            "(%.1f%% recovery rate), %d still suppressed",
            depth, n_recovered, n_suppressed,
            100 * n_recovered / n_suppressed if n_suppressed > 0 else 0,
            n_still_bad)

        if verbose:
            print(
                f"  Depth {depth}: recovered {n_recovered:,} of "
                f"{n_suppressed:,} rows "
                f"({100 * n_recovered / n_suppressed:.0f}%)")

        if n_recovered == 0:
            log.info(
                "[kANON-RECURSIVE] Depth %d: zero recovery — stopping",
                depth)
            break

        # ── Merge recovered rows back ───────────────────────────────
        recovered_idx = recoded_subset.index[recovered_mask]
        for qi in quasi_identifiers:
            result.loc[recovered_idx, qi] = recoded_subset.loc[recovered_idx, qi]

    # ── Post-merge k-anonymity check ───────────────────────────────
    # Recovered rows form new QI combinations that may violate k-anonymity
    # in the *combined* dataset. Apply a light suppression pass to fix.
    is_kanon_final, _, viols_final = check_kanonymity(
        result, quasi_identifiers, k)
    if not is_kanon_final and len(viols_final) > 0:
        log.info(
            "[kANON-RECURSIVE] Post-merge: %d violations — applying "
            "cleanup suppression", len(viols_final))
        result = _achieve_kanon_suppression(
            result, k, quasi_identifiers,
            max_suppression_rate, verbose=False)

    # ── Final summary ───────────────────────────────────────────────
    final_suppressed = int(
        result[quasi_identifiers].isna().any(axis=1).sum())
    original_suppressed = int(
        protected_data[quasi_identifiers].isna().any(axis=1).sum())
    total_recovered = original_suppressed - final_suppressed

    log.info(
        "[kANON-RECURSIVE] Done: recovered %d rows total "
        "(suppressed: %d → %d)",
        total_recovered, original_suppressed, final_suppressed)

    if verbose and total_recovered > 0:
        print(
            f"Recursive local recoding recovered {total_recovered:,} rows "
            f"(suppressed: {original_suppressed:,} → {final_suppressed:,})")

    return result


def get_kanon_report(
    original_data: pd.DataFrame,
    protected_data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int
) -> Dict:
    """
    Generate a report on k-anonymity transformation.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original dataset
    protected_data : pd.DataFrame
        k-anonymized dataset
    quasi_identifiers : list of str
        Quasi-identifier columns
    k : int
        Target k value

    Returns:
    --------
    dict
        Report with statistics about the anonymization
    """

    # Check k-anonymity of protected data
    is_kanon, group_sizes, violations = check_kanonymity(
        protected_data, quasi_identifiers, k
    )

    # Count equivalence classes
    n_equivalence_classes = len(group_sizes)
    min_group_size = group_sizes['count'].min()
    max_group_size = group_sizes['count'].max()
    avg_group_size = group_sizes['count'].mean()

    # Count suppressed values
    n_suppressed = protected_data[quasi_identifiers].isna().any(axis=1).sum()
    suppression_rate = n_suppressed / len(protected_data)

    report = {
        'k_value': k,
        'is_k_anonymous': is_kanon,
        'n_records': len(protected_data),
        'n_equivalence_classes': n_equivalence_classes,
        'min_group_size': int(min_group_size),
        'max_group_size': int(max_group_size),
        'avg_group_size': float(avg_group_size),
        'n_suppressed_records': int(n_suppressed),
        'suppression_rate': float(suppression_rate),
        'n_violations': len(violations) if not is_kanon else 0
    }

    return report


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("k-Anonymity (kANON) - Example Usage")
    print("=" * 60)

    # Create sample microdata
    sample_data = pd.DataFrame({
        'id': range(1, 21),
        'age': [25, 26, 27, 28, 29, 45, 46, 47, 48, 49,
                35, 36, 37, 62, 63, 64, 23, 24, 52, 53],
        'gender': ['M', 'M', 'F', 'F', 'M', 'F', 'M', 'F', 'M', 'F',
                   'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'region': ['North', 'North', 'South', 'South', 'East',
                   'West', 'West', 'North', 'North', 'South',
                   'East', 'East', 'West', 'North', 'North',
                   'South', 'East', 'East', 'West', 'West'],
        'income': [35000, 36000, 42000, 41000, 55000,
                   48000, 52000, 33000, 34000, 43000,
                   61000, 58000, 49000, 72000, 71000,
                   44000, 32000, 33000, 51000, 53000]
    })

    print("\nOriginal data (first 10 rows):")
    print(sample_data.head(10))

    # Define quasi-identifiers
    qis = ['age', 'gender', 'region']

    print(f"\n--- Example 1: Default bin_size=10 ---")
    result1 = apply_kanon(sample_data, k=3, quasi_identifiers=qis, bin_size=10)
    print(result1[qis].head(10))

    print(f"\n--- Example 2: Smaller bin_size=5 (more precision) ---")
    result2 = apply_kanon(sample_data, k=3, quasi_identifiers=qis, bin_size=5)
    print(result2[qis].head(10))

    print(f"\n--- Example 3: Larger bin_size=20 (more privacy) ---")
    result3 = apply_kanon(sample_data, k=3, quasi_identifiers=qis, bin_size=20)
    print(result3[qis].head(10))

    print(f"\n--- Example 4: Custom hierarchy for region ---")
    custom_hierarchies = {
        'region': {'North': 'Northern', 'South': 'Southern', 'East': 'Eastern', 'West': 'Western'}
    }
    result4 = apply_kanon(sample_data, k=3, quasi_identifiers=qis, hierarchies=custom_hierarchies)
    print(result4[qis].head(10))
