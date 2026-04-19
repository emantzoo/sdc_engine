"""
SDC Preprocessing Module
========================

Anonymization-specific preprocessing functions for Statistical Disclosure Control.

This module provides pre-anonymization data preparation functions that:
- Remove/hash direct identifiers (mandatory)
- Detect Greek-specific identifiers (ΑΦΜ, ΑΜΚΑ, ΑΔΤ)
- Handle outliers that increase re-identification risk
- Merge rare categories to prevent k-anonymity violations
- Assess dimensionality risk

Integration:
    # Add to sdc_utils.py or import separately
    from sdc_preprocessing import preprocess_for_sdc
    
    clean_data, report = preprocess_for_sdc(raw_data, mode='auto', return_metadata=True)

Author: TA2501 Project
Date: December 2025
"""

import pandas as pd
import numpy as np
import warnings
import hashlib
import re
from typing import Dict, List, Optional, Tuple, Union, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_PREPROCESSING_RULES = {
    # Mandatory steps - always applied unless mode='none'
    'remove_direct_identifiers': {
        'enabled': True,
        'method': 'remove',  # 'remove', 'hash', 'pseudonymize'
        'include_greek': True,
        'hash_salt': None  # Set for reproducible hashing
    },
    
    # Conditional steps - applied based on data characteristics
    'top_bottom_coding': {
        'enabled': 'auto',  # True, False, 'auto'
        'auto_threshold': 0.10,  # Apply if <10% records affected
        'method': 'percentile',  # 'percentile', 'iqr'
        'top_percentile': 99,
        'bottom_percentile': 1
    },
    
    'merge_rare_categories': {
        'enabled': 'auto',
        'auto_threshold': 0.20,  # Apply if >20% categories are rare
        'min_frequency': 3,
        'other_label': 'Other'
    },
    
    # Optional generalization step to reduce QI cardinality
    'generalize': {
        'enabled': 'auto',
        'max_categories': 10,
        'strategy': 'auto',  # 'auto', 'numeric', 'categorical', 'all'
    },
    
    # Warning-only steps - never auto-fix, just report
    'dimensionality_check': {
        'enabled': True,
        'warn_only': True,
        'critical_sparsity': 100,
        'high_sparsity': 10
    }
}


# =============================================================================
# GREEK-SPECIFIC IDENTIFIER DETECTION
# =============================================================================

def detect_greek_identifiers(data: pd.DataFrame) -> Dict[str, str]:
    """
    Detect Greek-specific direct identifiers that must be removed before SDC.
    
    Detects:
    - ΑΦΜ (Tax ID): 9 digits
    - ΑΜΚΑ (Social Security): 11 digits  
    - ΑΔΤ (ID Card): 1-2 Greek letters + 6 digits
    - Greek IBAN: GR + 25 digits
    - Greek VAT: EL + 9 digits
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
        
    Returns
    -------
    Dict[str, str]
        Column name -> identifier type description
        
    Examples
    --------
    >>> detected = detect_greek_identifiers(data)
    >>> print(detected)
    {'tax_id': 'ΑΦΜ (pattern match: 95%)', 'amka_col': 'ΑΜΚΑ (name match)'}
    """
    
    # Pattern definitions: (regex, description, min_match_rate)
    patterns = {
        'afm': (r'^\d{9}$', 'ΑΦΜ - Tax ID', 0.80),
        'amka': (r'^\d{11}$', 'ΑΜΚΑ - Social Security', 0.80),
        'adt': (r'^[Α-ΩA-Z]{1,2}\s?\d{6}$', 'ΑΔΤ - ID Card', 0.70),
        'iban_gr': (r'^GR\d{25}$', 'Greek IBAN', 0.80),
        'vat_gr': (r'^EL\d{9}$', 'Greek VAT Number', 0.80),
    }
    
    # Column name hints (Greek and transliterated)
    name_hints = {
        'afm': ['αφμ', 'afm', 'tin', 'tax_id', 'taxid', 'vat_number'],
        'amka': ['αμκα', 'amka', 'ssn', 'social_security', 'insurance_number'],
        'adt': ['αδτ', 'adt', 'id_card', 'identity', 'ταυτοτητα', 'tautotita'],
        'iban': ['iban', 'bank_account', 'τραπεζικός'],
    }
    
    detected = {}
    
    for col in data.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        col_words = col_lower.split('_')
        
        # Check column name hints first (faster)
        # Use word boundaries to avoid false positives like 'tin' in 'rating'
        for id_type, hints in name_hints.items():
            if any(hint in col_words or hint == col_lower for hint in hints):
                detected[col] = f'{patterns.get(id_type, ("", id_type))[1]} (name match)'
                break
        
        if col in detected:
            continue
            
        # Check value patterns for string columns
        if data[col].dtype == 'object':
            sample = data[col].dropna().astype(str)
            if len(sample) == 0:
                continue
                
            # Take sample for efficiency
            sample = sample.head(200)
            
            for pattern_name, (pattern, description, min_rate) in patterns.items():
                try:
                    match_rate = sample.str.match(pattern, na=False).mean()
                    if match_rate >= min_rate:
                        detected[col] = f'{description} (pattern match: {match_rate:.0%})'
                        break
                except re.error:
                    continue
    
    return detected


# =============================================================================
# DIRECT IDENTIFIER REMOVAL/TRANSFORMATION
# =============================================================================

def remove_direct_identifiers(
    data: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'remove',
    hash_salt: str = None,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Remove or transform direct identifiers before SDC processing.
    
    Per Del1 §5.5.1: Direct identifiers must be removed or transformed
    with non-reversible hashing before anonymization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    columns : List[str], optional
        Columns to process. If None, auto-detects sensitive columns.
    method : str, default='remove'
        - 'remove': Drop columns entirely
        - 'hash': Replace with SHA-256 hash (truncated)
        - 'pseudonymize': Replace with sequential IDs
    hash_salt : str, optional
        Salt for hashing (for reproducibility across runs)
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)
        
    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Processed data, optionally with metadata
        
    Examples
    --------
    >>> clean, meta = remove_direct_identifiers(data, method='hash', return_metadata=True)
    >>> print(meta['columns_processed'])
    ['name', 'email', 'afm']
    """
    
    # Import here to avoid circular dependency if integrated into sdc_utils
    try:
        from .sdc_utils import auto_detect_sensitive_columns
    except ImportError:
        try:
            from sdc_engine.sdc.sdc_utils import auto_detect_sensitive_columns
        except ImportError:
            # Fallback - define minimal version
            def auto_detect_sensitive_columns(df):
                sensitive = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if any(x in col_lower for x in ['name', 'email', 'phone', 'ssn', 'address']):
                        sensitive[col] = 'detected_by_name'
                return sensitive
    
    result = data.copy()
    metadata = {
        'method': method,
        'columns_processed': [],
        'columns_skipped': [],
        'records_affected': len(data)
    }
    
    # Auto-detect if columns not specified
    if columns is None:
        standard_sensitive = auto_detect_sensitive_columns(data)
        greek_sensitive = detect_greek_identifiers(data)
        columns = list(set(list(standard_sensitive.keys()) + list(greek_sensitive.keys())))
        metadata['auto_detected'] = True
        metadata['detection_results'] = {
            'standard': standard_sensitive,
            'greek': greek_sensitive
        }
    else:
        metadata['auto_detected'] = False
    
    # Process each column
    for col in columns:
        if col not in result.columns:
            metadata['columns_skipped'].append(col)
            continue
        
        if method == 'remove':
            result = result.drop(columns=[col])
            
        elif method == 'hash':
            salt = hash_salt or ''
            result[col] = result[col].apply(
                lambda x: hashlib.sha256(f'{salt}{x}'.encode()).hexdigest()[:16]
                if pd.notna(x) else None
            )
            
        elif method == 'pseudonymize':
            unique_vals = result[col].dropna().unique()
            mapping = {v: f'ID_{i:08d}' for i, v in enumerate(unique_vals, 1)}
            result[col] = result[col].map(mapping)
            metadata[f'{col}_mapping_size'] = len(mapping)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'remove', 'hash', or 'pseudonymize'")
        
        metadata['columns_processed'].append(col)
    
    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# OUTLIER DETECTION FOR RE-IDENTIFICATION RISK
# =============================================================================

def detect_reidentification_outliers(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    method: str = 'iqr',
    iqr_multiplier: float = 1.5
) -> Dict:
    """
    Detect outliers in quasi-identifiers that increase re-identification risk.
    
    Outliers in QIs can make records nearly unique, creating disclosure risk
    even after standard anonymization.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns to check
    method : str, default='iqr'
        Detection method: 'iqr' or 'zscore'
    iqr_multiplier : float, default=1.5
        Multiplier for IQR method (1.5 = standard, 3.0 = extreme only)
        
    Returns
    -------
    Dict
        Assessment results with high-risk records and recommendations
        
    Examples
    --------
    >>> risk = detect_reidentification_outliers(data, ['age', 'income'])
    >>> print(f"High risk records: {len(risk['high_risk_records'])}")
    >>> print(risk['recommended_actions'])
    """
    
    results = {
        'high_risk_records': [],
        'problematic_columns': {},
        'recommended_actions': [],
        'summary': {}
    }
    
    numeric_qis = [qi for qi in quasi_identifiers 
                   if qi in data.columns and data[qi].dtype in ['int64', 'float64', 'int32', 'float32']]
    
    if not numeric_qis:
        results['summary']['message'] = 'No numeric quasi-identifiers to check'
        return results
    
    all_outlier_indices = set()
    
    for col in numeric_qis:
        col_data = data[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        if method == 'iqr':
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:  # No spread
                continue
                
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
        elif method == 'zscore':
            mean = col_data.mean()
            std = col_data.std()
            
            if std == 0:
                continue
                
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Find outliers
        outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
        outlier_indices = data[outlier_mask].index.tolist()
        
        if outlier_indices:
            n_outliers = len(outlier_indices)
            outlier_rate = n_outliers / len(data)
            
            all_outlier_indices.update(outlier_indices)
            
            results['problematic_columns'][col] = {
                'count': n_outliers,
                'rate': outlier_rate,
                'indices': outlier_indices[:100],  # Limit stored indices
                'bounds': {'lower': lower_bound, 'upper': upper_bound},
                'actual_range': {'min': data[col].min(), 'max': data[col].max()},
                'recommended_action': 'top_bottom_coding' if outlier_rate < 0.05 else 'binning'
            }
    
    results['high_risk_records'] = list(all_outlier_indices)
    results['summary'] = {
        'total_records': len(data),
        'records_with_outliers': len(all_outlier_indices),
        'outlier_rate': len(all_outlier_indices) / len(data) if len(data) > 0 else 0,
        'columns_affected': list(results['problematic_columns'].keys())
    }
    
    # Generate recommendations
    if results['summary']['outlier_rate'] > 0:
        if results['summary']['outlier_rate'] < 0.10:
            results['recommended_actions'].append(
                'Apply top/bottom coding to cap extreme values'
            )
        else:
            results['recommended_actions'].append(
                'High outlier rate - consider binning or manual review'
            )
    
    return results


# =============================================================================
# TOP/BOTTOM CODING
# =============================================================================

def apply_top_bottom_coding(
    data: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'percentile',
    top_percentile: float = 99,
    bottom_percentile: float = 1,
    top_value: float = None,
    bottom_value: float = None,
    per_qi_percentiles: Optional[Dict[str, tuple]] = None,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Cap extreme values to reduce uniqueness before SDC.
    
    This is a non-perturbative pre-anonymization step from Del1 §2.4
    that reduces outlier-based re-identification risk.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    columns : List[str], optional
        Columns to process. If None, processes all numeric columns.
    method : str, default='percentile'
        - 'percentile': Use percentile-based bounds
        - 'iqr': Use IQR-based bounds
        - 'fixed': Use provided top_value/bottom_value
    top_percentile : float, default=99
        Upper percentile for capping (for 'percentile' method)
    bottom_percentile : float, default=1
        Lower percentile for capping (for 'percentile' method)
    top_value : float, optional
        Fixed upper bound (for 'fixed' method)
    bottom_value : float, optional
        Fixed lower bound (for 'fixed' method)
    per_qi_percentiles : dict, optional
        Per-QI treatment-scaled percentiles ``{col: (bottom_pctile, top_pctile)}``.
        When provided, overrides global ``bottom_percentile``/``top_percentile``
        for columns in the dict.  Built by ``qi_treatment.build_per_qi_percentiles()``.
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)
        
    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Data with capped values, optionally with metadata
        
    Examples
    --------
    >>> coded, meta = apply_top_bottom_coding(data, columns=['income', 'age'], return_metadata=True)
    >>> print(meta['columns_modified'])
    {'income': {'top_coded': 15, 'bottom_coded': 8}}
    """
    
    result = data.copy()
    metadata = {
        'method': method,
        'columns_modified': {},
        'records_affected': 0,
        'total_values_changed': 0
    }
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    affected_records = set()
    
    for col in columns:
        if col not in result.columns:
            continue
            
        if not np.issubdtype(result[col].dtype, np.number):
            continue
        
        col_data = result[col].dropna()
        if len(col_data) == 0:
            continue
        
        # Determine bounds
        if method == 'percentile':
            # Use per-QI treatment-scaled percentiles when available
            _eff_bot = bottom_percentile
            _eff_top = top_percentile
            if per_qi_percentiles and col in per_qi_percentiles:
                _eff_bot, _eff_top = per_qi_percentiles[col]
            lower = col_data.quantile(_eff_bot / 100)
            upper = col_data.quantile(_eff_top / 100)
        elif method == 'iqr':
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        elif method == 'fixed':
            lower = bottom_value if bottom_value is not None else col_data.min()
            upper = top_value if top_value is not None else col_data.max()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Count affected values
        bottom_mask = result[col] < lower
        top_mask = result[col] > upper
        bottom_count = bottom_mask.sum()
        top_count = top_mask.sum()
        
        if bottom_count > 0 or top_count > 0:
            # Track affected records
            affected_records.update(result[bottom_mask | top_mask].index)
            
            # Apply coding
            result[col] = result[col].clip(lower=lower, upper=upper)
            
            col_meta = {
                'lower_bound': float(lower),
                'upper_bound': float(upper),
                'bottom_coded': int(bottom_count),
                'top_coded': int(top_count),
                'total_changed': int(bottom_count + top_count),
            }
            # Log effective percentiles so users can see treatment effect
            if method == 'percentile':
                col_meta['effective_bottom_pctile'] = float(_eff_bot)
                col_meta['effective_top_pctile'] = float(_eff_top)
            metadata['columns_modified'][col] = col_meta
            metadata['total_values_changed'] += bottom_count + top_count
    
    metadata['records_affected'] = len(affected_records)
    metadata['records_affected_rate'] = len(affected_records) / len(data) if len(data) > 0 else 0
    
    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# RARE CATEGORY MERGING
# =============================================================================

def merge_rare_categories(
    data: pd.DataFrame,
    columns: List[str] = None,
    min_frequency: int = 3,
    min_percentage: float = None,
    other_label: str = 'Other',
    per_qi_min_frequency: Optional[Dict[str, int]] = None,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Merge rare categories that would create small equivalence classes.
    
    Pre-anonymization step to reduce quasi-identifier cardinality
    and prevent k-anonymity violations.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    columns : List[str], optional
        Categorical columns to process. If None, auto-detects.
    min_frequency : int, default=3
        Minimum count for a category to remain separate
    min_percentage : float, optional
        Alternative threshold as percentage of total records
    other_label : str, default='Other'
        Label for merged categories
    per_qi_min_frequency : dict, optional
        Per-QI treatment-scaled min_frequency ``{col: min_freq}``.
        When provided, overrides global ``min_frequency`` for columns
        in the dict.  Built by ``qi_treatment.build_per_qi_min_frequency()``.
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Data with merged categories, optionally with metadata

    Examples
    --------
    >>> merged, meta = merge_rare_categories(data, min_frequency=5, return_metadata=True)
    >>> print(meta['columns_modified']['region']['merged_values'])
    ['Region_X', 'Region_Y', 'Region_Z']
    """
    
    result = data.copy()
    metadata = {
        'min_frequency': min_frequency,
        'other_label': other_label,
        'columns_modified': {},
        'total_categories_merged': 0,
        'total_records_affected': 0
    }
    
    # Auto-detect categorical columns if not specified
    if columns is None:
        columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    import re
    _range_pat = re.compile(r'^(\d+)\s*[-–]\s*(\d+)$')

    for col in columns:
        if col not in result.columns:
            continue

        value_counts = result[col].value_counts()

        # Skip range-pattern columns (e.g. "20-24" from age binning).
        # Merging rare ranges into "Other" destroys their ordered structure
        # and makes downstream range-aware generalization impossible.
        _sample = value_counts.index.tolist()
        _n_range = sum(1 for v in _sample if _range_pat.match(str(v)))
        if len(_sample) >= 4 and _n_range / len(_sample) > 0.8:
            continue

        # Determine threshold (per-QI override takes precedence)
        eff_min_freq = min_frequency
        if per_qi_min_frequency and col in per_qi_min_frequency:
            eff_min_freq = per_qi_min_frequency[col]
        threshold = eff_min_freq
        if min_percentage is not None:
            threshold = max(eff_min_freq, int(len(data) * min_percentage / 100))

        # Safe-skip guard: if ALL categories already have count >= threshold,
        # no categories are rare — skip this QI entirely to avoid losing
        # meaningful categories for no risk benefit.
        if (value_counts >= threshold).all():
            continue

        # Find rare categories
        rare_values = value_counts[value_counts < threshold].index.tolist()

        if rare_values:
            records_affected = value_counts[rare_values].sum()
            
            # Merge rare categories
            result[col] = result[col].replace(rare_values, other_label)
            
            metadata['columns_modified'][col] = {
                'merged_values': rare_values,
                'merged_count': len(rare_values),
                'records_affected': int(records_affected),
                'original_categories': len(value_counts),
                'final_categories': len(value_counts) - len(rare_values) + 1,
                'effective_min_frequency': threshold,
            }
            metadata['total_categories_merged'] += len(rare_values)
            metadata['total_records_affected'] += records_affected
    
    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# NUMERIC ROUNDING
# =============================================================================

def apply_numeric_rounding(
    data: pd.DataFrame,
    columns: List[str] = None,
    rounding_base: int = None,
    significant_digits: int = None,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Round numeric values to reduce precision and uniqueness.

    Useful for values like income (54321 → 54000), population counts,
    or any numeric quasi-identifier with high precision.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    columns : List[str], optional
        Columns to round. If None, processes all numeric columns.
    rounding_base : int, optional
        Round to nearest multiple (e.g., 1000 rounds 54321 to 54000).
        If None, auto-detects based on value magnitude.
    significant_digits : int, optional
        Keep only N significant digits (alternative to rounding_base).
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Data with rounded values, optionally with metadata

    Examples
    --------
    >>> rounded, meta = apply_numeric_rounding(data, columns=['income'], rounding_base=1000, return_metadata=True)
    >>> # 54321 → 54000, 67890 → 68000
    """

    result = data.copy()
    metadata = {
        'rounding_base': rounding_base,
        'significant_digits': significant_digits,
        'columns_modified': {},
        'total_values_changed': 0
    }

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in result.columns:
            continue

        if not np.issubdtype(result[col].dtype, np.number):
            continue

        col_data = result[col].dropna()
        if len(col_data) == 0:
            continue

        original_values = result[col].copy()

        # Determine rounding approach
        if rounding_base is not None:
            # Round to nearest multiple of rounding_base
            result[col] = (result[col] / rounding_base).round() * rounding_base
            used_base = rounding_base

        elif significant_digits is not None:
            # Round to N significant digits
            def round_to_n_sig(x, n):
                if pd.isna(x) or x == 0:
                    return x
                return round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))
            result[col] = result[col].apply(lambda x: round_to_n_sig(x, significant_digits))
            used_base = f'{significant_digits} sig. digits'

        else:
            # Auto-detect based on magnitude
            magnitude = col_data.abs().median()
            if magnitude >= 10000:
                auto_base = 1000
            elif magnitude >= 1000:
                auto_base = 100
            elif magnitude >= 100:
                auto_base = 10
            else:
                auto_base = 1  # No rounding for small values

            if auto_base > 1:
                result[col] = (result[col] / auto_base).round() * auto_base
            used_base = f'auto:{auto_base}'

        # Count changes
        changed = (original_values != result[col]) & original_values.notna()
        n_changed = changed.sum()

        if n_changed > 0:
            metadata['columns_modified'][col] = {
                'rounding_base': used_base,
                'values_changed': int(n_changed),
                'original_unique': int(original_values.nunique()),
                'final_unique': int(result[col].nunique()),
                'example_before': float(original_values.dropna().iloc[0]) if len(original_values.dropna()) > 0 else None,
                'example_after': float(result[col].dropna().iloc[0]) if len(result[col].dropna()) > 0 else None
            }
            metadata['total_values_changed'] += n_changed

    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# QUANTILE BINNING
# =============================================================================

def apply_quantile_binning(
    data: pd.DataFrame,
    columns: List[str] = None,
    n_bins: int = 20,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """Bin numeric columns into equal-frequency (quantile) buckets.

    Produces range labels like ``"20000-39999"`` that preserve ordering
    and are compatible with the range-merge logic in kANON.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    columns : list[str], optional
        Columns to bin.  If *None*, processes all numeric columns.
    n_bins : int, default 20
        Target number of bins.  Actual count may be lower when many
        values are identical (``pd.qcut`` drops duplicate edges).
    return_metadata : bool, default False
        If *True*, return ``(DataFrame, metadata_dict)``.
    """
    result = data.copy()
    metadata: Dict[str, Any] = {
        'n_bins_requested': n_bins,
        'columns_modified': {},
        'total_values_changed': 0,
    }

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in result.columns or not np.issubdtype(result[col].dtype, np.number):
            continue

        valid = result[col].dropna()
        if len(valid) < n_bins:
            continue

        before_unique = int(result[col].nunique())

        try:
            binned = pd.qcut(valid, q=n_bins, duplicates='drop')
        except ValueError:
            continue

        intervals = binned.cat.categories
        labels = []
        for iv in intervals:
            lo, hi = iv.left, iv.right
            if abs(lo) >= 10 and abs(hi) >= 10:
                labels.append(f"{int(lo)}-{int(hi)}")
            else:
                labels.append(f"{lo:.1f}-{hi:.1f}")
        label_map = dict(zip(intervals, labels))

        new_col = result[col].copy().astype(object)
        new_col[valid.index] = binned.map(label_map).astype(str)
        new_col[result[col].isna()] = None
        result[col] = new_col

        after_unique = int(result[col].nunique())
        n_changed = int((data[col].astype(str) != result[col].astype(str)).sum())
        metadata['columns_modified'][col] = {
            'n_bins': len(intervals),
            'original_unique': before_unique,
            'final_unique': after_unique,
            'success': True,
            'before_unique': before_unique,
            'after_unique': after_unique,
        }
        metadata['total_values_changed'] += n_changed

    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# DATE TRUNCATION
# =============================================================================

def apply_date_truncation(
    data: pd.DataFrame,
    columns: List[str] = None,
    truncate_to: str = 'month',
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Truncate dates to reduce precision (e.g., 2024-03-15 → 2024-03).

    Common anonymization technique for dates of birth, event dates, etc.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    columns : List[str], optional
        Date columns to truncate. If None, auto-detects date columns.
    truncate_to : str, default='month'
        Truncation level:
        - 'year': Keep only year (2024-03-15 → 2024)
        - 'quarter': Keep year-quarter (2024-03-15 → 2024-Q1)
        - 'month': Keep year-month (2024-03-15 → 2024-03)
        - 'week': Keep year-week (2024-03-15 → 2024-W11)
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Data with truncated dates, optionally with metadata

    Examples
    --------
    >>> truncated, meta = apply_date_truncation(data, columns=['birth_date'], truncate_to='year', return_metadata=True)
    >>> # 1990-05-23 → 1990
    """

    result = data.copy()
    metadata = {
        'truncate_to': truncate_to,
        'columns_modified': {},
        'total_columns_processed': 0
    }

    # Auto-detect date columns if not specified
    if columns is None:
        columns = []
        for col in data.columns:
            # Check if already datetime
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                columns.append(col)
            # Check column name hints
            elif any(hint in col.lower() for hint in ['date', 'time', 'birth', 'dob', 'created', 'updated', 'ημερομηνία']):
                # Try to parse as date
                try:
                    pd.to_datetime(data[col].head(100), errors='raise')
                    columns.append(col)
                except:
                    pass

    for col in columns:
        if col not in result.columns:
            continue

        original_values = result[col].copy()

        # Convert to datetime if not already
        try:
            if not pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = pd.to_datetime(result[col], errors='coerce')
        except:
            continue

        # Skip if conversion failed
        if result[col].isna().all():
            result[col] = original_values
            continue

        original_unique = result[col].nunique()

        # Apply truncation
        if truncate_to == 'year':
            result[col] = result[col].dt.year.astype(str)
        elif truncate_to == 'quarter':
            result[col] = result[col].dt.to_period('Q').astype(str)
        elif truncate_to == 'month':
            result[col] = result[col].dt.to_period('M').astype(str)
        elif truncate_to == 'week':
            result[col] = result[col].dt.strftime('%Y-W%W')
        else:
            raise ValueError(f"Unknown truncate_to value: {truncate_to}. Use 'year', 'quarter', 'month', or 'week'")

        final_unique = result[col].nunique()

        metadata['columns_modified'][col] = {
            'truncate_to': truncate_to,
            'original_unique': int(original_unique),
            'final_unique': int(final_unique),
            'uniqueness_reduction': f'{(1 - final_unique/original_unique)*100:.1f}%' if original_unique > 0 else 'N/A'
        }
        metadata['total_columns_processed'] += 1

    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# AGE BINNING
# =============================================================================

def apply_age_binning(
    data: pd.DataFrame,
    columns: List[str] = None,
    bin_size: int = 5,
    bins: List[int] = None,
    labels: List[str] = None,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Convert exact ages to age ranges/bins (e.g., 34 → 30-34 or 30-39).

    Common anonymization technique for age data.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    columns : List[str], optional
        Age columns to bin. If None, auto-detects columns with 'age' in name.
    bin_size : int, default=5
        Size of each bin (e.g., 5 → 0-4, 5-9, 10-14...)
    bins : List[int], optional
        Custom bin edges (e.g., [0, 18, 35, 50, 65, 100])
    labels : List[str], optional
        Custom labels for bins. If None, generates from bin edges.
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Data with binned ages, optionally with metadata

    Examples
    --------
    >>> binned, meta = apply_age_binning(data, columns=['age'], bin_size=10, return_metadata=True)
    >>> # 34 → '30-39', 45 → '40-49'

    >>> # Custom bins
    >>> binned, meta = apply_age_binning(data, bins=[0, 18, 35, 50, 65, 100],
    ...                                   labels=['0-17', '18-34', '35-49', '50-64', '65+'])
    """

    result = data.copy()
    metadata = {
        'bin_size': bin_size,
        'custom_bins': bins,
        'columns_modified': {},
        'total_columns_processed': 0
    }

    # Auto-detect age columns if not specified
    if columns is None:
        columns = [col for col in data.columns
                   if any(hint in col.lower() for hint in ['age', 'ηλικία', 'ηλικια'])]

    for col in columns:
        if col not in result.columns:
            continue

        # Ensure numeric
        if not np.issubdtype(result[col].dtype, np.number):
            continue

        original_unique = result[col].nunique()

        col_min = result[col].min()
        col_max = result[col].max()

        # Create bins
        if bins is not None:
            use_bins = bins
            if labels is None:
                # Generate labels from bin edges
                use_labels = [f'{bins[i]}-{bins[i+1]-1}' for i in range(len(bins)-2)]
                use_labels.append(f'{bins[-2]}+')
            else:
                use_labels = labels
        else:
            # Generate uniform bins
            bin_start = int(col_min // bin_size * bin_size)
            bin_end = int((col_max // bin_size + 1) * bin_size) + bin_size
            use_bins = list(range(bin_start, bin_end, bin_size))
            use_labels = [f'{use_bins[i]}-{use_bins[i+1]-1}' for i in range(len(use_bins)-1)]

        # Apply binning
        result[col] = pd.cut(result[col], bins=use_bins, labels=use_labels, include_lowest=True, right=False)
        result[col] = result[col].astype(str).replace('nan', np.nan)

        final_unique = result[col].nunique()

        metadata['columns_modified'][col] = {
            'bins_used': use_bins,
            'labels': use_labels,
            'original_unique': int(original_unique),
            'final_unique': int(final_unique),
            'uniqueness_reduction': f'{(1 - final_unique/original_unique)*100:.1f}%' if original_unique > 0 else 'N/A'
        }
        metadata['total_columns_processed'] += 1

    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# GEOGRAPHIC COARSENING
# =============================================================================

def apply_geographic_coarsening(
    data: pd.DataFrame,
    columns: List[str] = None,
    keep_digits: int = 3,
    replacement_char: str = '*',
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Coarsen geographic codes like ZIP/postal codes (e.g., 12345 → 123**).

    Reduces geographic precision while preserving regional information.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    columns : List[str], optional
        Geographic columns to coarsen. If None, auto-detects ZIP/postal columns.
    keep_digits : int, default=3
        Number of leading digits to keep
    replacement_char : str, default='*'
        Character to replace truncated digits with
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Data with coarsened geography, optionally with metadata

    Examples
    --------
    >>> coarsened, meta = apply_geographic_coarsening(data, columns=['zipcode'], keep_digits=3, return_metadata=True)
    >>> # '12345' → '123**', '10001' → '100**'
    """

    result = data.copy()
    metadata = {
        'keep_digits': keep_digits,
        'replacement_char': replacement_char,
        'columns_modified': {},
        'total_columns_processed': 0
    }

    # Auto-detect geographic columns if not specified
    if columns is None:
        geo_hints = ['zip', 'postal', 'postcode', 'zipcode', 'ταχυδρομικός', 'tk', 'τκ']
        columns = [col for col in data.columns
                   if any(hint in col.lower().replace('_', '').replace('-', '') for hint in geo_hints)]

    for col in columns:
        if col not in result.columns:
            continue

        original_unique = result[col].nunique()

        # Convert to string
        result[col] = result[col].astype(str)

        # Apply coarsening
        def coarsen_code(val):
            if pd.isna(val) or val == 'nan' or val == 'None':
                return val
            val_str = str(val).strip()
            if len(val_str) <= keep_digits:
                return val_str
            return val_str[:keep_digits] + replacement_char * (len(val_str) - keep_digits)

        result[col] = result[col].apply(coarsen_code)

        final_unique = result[col].nunique()

        metadata['columns_modified'][col] = {
            'keep_digits': keep_digits,
            'original_unique': int(original_unique),
            'final_unique': int(final_unique),
            'uniqueness_reduction': f'{(1 - final_unique/original_unique)*100:.1f}%' if original_unique > 0 else 'N/A'
        }
        metadata['total_columns_processed'] += 1

    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# STRING TRUNCATION
# =============================================================================

def apply_string_truncation(
    data: pd.DataFrame,
    columns: List[str] = None,
    max_length: int = None,
    keep_chars: int = None,
    truncate_from: str = 'end',
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Truncate string values to reduce uniqueness (e.g., "John Smith" → "John S").

    Useful for names, addresses, and other free-text fields that may need
    partial preservation rather than complete removal.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    columns : List[str], optional
        String columns to truncate. If None, must be specified.
    max_length : int, optional
        Maximum length to allow
    keep_chars : int, optional
        Number of characters to keep (alternative to max_length)
    truncate_from : str, default='end'
        Where to truncate: 'end' (keep start) or 'start' (keep end)
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Data with truncated strings, optionally with metadata

    Examples
    --------
    >>> truncated, meta = apply_string_truncation(data, columns=['address'], max_length=20, return_metadata=True)
    >>> # "123 Main Street, Apt 4B" → "123 Main Street, Apt"
    """

    result = data.copy()
    metadata = {
        'max_length': max_length,
        'keep_chars': keep_chars,
        'truncate_from': truncate_from,
        'columns_modified': {},
        'total_values_changed': 0
    }

    if columns is None:
        metadata['warning'] = 'No columns specified for string truncation'
        if return_metadata:
            return result, metadata
        return result

    length_limit = keep_chars if keep_chars is not None else max_length
    if length_limit is None:
        length_limit = 10  # Default

    for col in columns:
        if col not in result.columns:
            continue

        if result[col].dtype != 'object':
            continue

        original_values = result[col].copy()
        original_unique = result[col].nunique()

        # Apply truncation
        def truncate_string(val):
            if pd.isna(val):
                return val
            val_str = str(val)
            if len(val_str) <= length_limit:
                return val_str
            if truncate_from == 'end':
                return val_str[:length_limit]
            else:
                return val_str[-length_limit:]

        result[col] = result[col].apply(truncate_string)

        # Count changes
        changed = (original_values != result[col]) & original_values.notna()
        n_changed = changed.sum()

        final_unique = result[col].nunique()

        if n_changed > 0:
            metadata['columns_modified'][col] = {
                'length_limit': length_limit,
                'truncate_from': truncate_from,
                'values_changed': int(n_changed),
                'original_unique': int(original_unique),
                'final_unique': int(final_unique)
            }
            metadata['total_values_changed'] += n_changed

    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# RECORD SAMPLING
# =============================================================================

def apply_record_sampling(
    data: pd.DataFrame,
    sample_fraction: float = None,
    sample_size: int = None,
    stratify_by: List[str] = None,
    random_state: int = None,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Sample records to reduce dataset size and re-identification risk.

    Sampling can be a preprocessing step to reduce unique record risk.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    sample_fraction : float, optional
        Fraction of records to keep (e.g., 0.8 for 80%)
    sample_size : int, optional
        Exact number of records to keep (alternative to fraction)
    stratify_by : List[str], optional
        Columns to stratify sampling by (maintains proportions)
    random_state : int, optional
        Random seed for reproducibility
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Sampled data, optionally with metadata

    Examples
    --------
    >>> sampled, meta = apply_record_sampling(data, sample_fraction=0.9, stratify_by=['region'], return_metadata=True)
    >>> print(f"Reduced from {meta['original_size']} to {meta['final_size']} records")
    """

    metadata = {
        'sample_fraction': sample_fraction,
        'sample_size': sample_size,
        'stratify_by': stratify_by,
        'random_state': random_state,
        'original_size': len(data),
        'final_size': None
    }

    if sample_fraction is None and sample_size is None:
        sample_fraction = 1.0  # Keep all

    # Determine target size
    if sample_size is not None:
        target_size = min(sample_size, len(data))
        actual_fraction = target_size / len(data)
    else:
        actual_fraction = sample_fraction
        target_size = int(len(data) * sample_fraction)

    if actual_fraction >= 1.0:
        result = data.copy()
    elif stratify_by and all(col in data.columns for col in stratify_by):
        # Stratified sampling
        try:
            from sklearn.model_selection import train_test_split
            _, result = train_test_split(
                data,
                test_size=actual_fraction,
                stratify=data[stratify_by],
                random_state=random_state
            )
        except (ImportError, ValueError):
            # Fall back to simple sampling if sklearn not available or stratification fails
            result = data.sample(frac=actual_fraction, random_state=random_state)
    else:
        result = data.sample(frac=actual_fraction, random_state=random_state)

    result = result.reset_index(drop=True)

    metadata['final_size'] = len(result)
    metadata['records_removed'] = metadata['original_size'] - metadata['final_size']
    metadata['actual_fraction'] = metadata['final_size'] / metadata['original_size'] if metadata['original_size'] > 0 else 0

    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# DIMENSIONALITY RISK ASSESSMENT
# =============================================================================

def assess_dimensionality_risk(
    data: pd.DataFrame,
    quasi_identifiers: List[str]
) -> Dict:
    """
    Assess if quasi-identifier dimensionality is too high for effective anonymization.
    
    Per Del1 §3 (page 59): Dimensionality reduction may be needed
    as a supplementary measure when the QI space is too sparse.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str]
        Quasi-identifier columns
        
    Returns
    -------
    Dict
        Risk assessment with recommendations
        
    Examples
    --------
    >>> risk = assess_dimensionality_risk(data, ['age', 'gender', 'region', 'occupation'])
    >>> print(risk['risk_level'])  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    >>> print(risk['recommendation'])
    """
    
    n_records = len(data)
    
    # Filter to existing columns
    valid_qis = [qi for qi in quasi_identifiers if qi in data.columns]
    
    if not valid_qis:
        return {
            'error': 'No valid quasi-identifiers found',
            'risk_level': 'UNKNOWN'
        }
    
    n_qi = len(valid_qis)
    
    # Calculate cardinality for each QI
    cardinalities = {}
    for qi in valid_qis:
        cardinalities[qi] = data[qi].nunique()
    
    # Theoretical maximum combinations
    theoretical_combinations = np.prod(list(cardinalities.values()))
    
    # Actual unique combinations
    actual_combinations = data[valid_qis].drop_duplicates().shape[0]
    
    # Sparsity ratio (how many more theoretical combinations than records)
    sparsity = theoretical_combinations / n_records if n_records > 0 else float('inf')
    
    # Uniqueness rate
    uniqueness_rate = actual_combinations / n_records if n_records > 0 else 1.0
    
    # Risk assessment
    if sparsity > 100 or uniqueness_rate > 0.9:
        risk_level = 'CRITICAL'
        recommendation = 'Strongly reduce QI dimensions or apply heavy generalization'
    elif sparsity > 10 or uniqueness_rate > 0.7:
        risk_level = 'HIGH'
        recommendation = 'Consider removing highest-cardinality QI or binning'
    elif sparsity > 2 or uniqueness_rate > 0.5:
        risk_level = 'MEDIUM'
        recommendation = 'Standard generalization should achieve k-anonymity'
    else:
        risk_level = 'LOW'
        recommendation = 'Data suitable for standard SDC methods'
    
    # Sort QIs by cardinality (highest risk first)
    sorted_qis = sorted(cardinalities.items(), key=lambda x: x[1], reverse=True)
    
    # Identify problematic QIs
    suggested_removals = [qi for qi, card in sorted_qis if card > n_records * 0.5]
    suggested_binning = [qi for qi, card in sorted_qis 
                         if card > 20 and qi not in suggested_removals]
    
    return {
        'n_records': n_records,
        'n_quasi_identifiers': n_qi,
        'cardinalities': cardinalities,
        'theoretical_combinations': int(min(theoretical_combinations, 2**63)),  # Cap for display
        'actual_combinations': actual_combinations,
        'sparsity_ratio': float(sparsity),
        'uniqueness_rate': float(uniqueness_rate),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'qi_by_cardinality': sorted_qis,
        'suggested_removals': suggested_removals,
        'suggested_binning': suggested_binning
    }


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def preprocess_for_sdc(
    data: pd.DataFrame,
    quasi_identifiers: List[str] = None,
    mode: str = 'auto',
    config: Dict = None,
    force_apply_recommendations: bool = False,
    return_metadata: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Prepare data for SDC processing with configurable preprocessing pipeline.
    
    This is the main entry point for preprocessing. It integrates with
    the existing workflow:
    
        data = preprocess_for_sdc(raw_data)
        analysis = analyze_data(data)
        result = apply_kanon(data, ...)
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str], optional
        Quasi-identifier columns. Auto-detected if not provided.
    mode : str, default='auto'
        Preprocessing mode:
        - 'auto': Apply mandatory + rule-based optional steps
        - 'mandatory_only': Only remove direct identifiers
        - 'full': Apply all preprocessing steps
        - 'report_only': Don't modify data, just return assessment
        - 'none': Skip preprocessing (return as-is with warnings)
    config : Dict, optional
        Override default preprocessing rules
    return_metadata : bool, default=False
        If True, return (DataFrame, metadata_dict)
        
    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Dict]
        Preprocessed data, optionally with metadata report
        
    Examples
    --------
    >>> # Auto mode (recommended)
    >>> clean, report = preprocess_for_sdc(raw_data, mode='auto', return_metadata=True)
    >>> print(report['steps_applied'])
    ['remove_direct_identifiers', 'top_bottom_coding']
    
    >>> # Mandatory only (conservative)
    >>> clean = preprocess_for_sdc(raw_data, mode='mandatory_only')
    
    >>> # Just get report without modifying
    >>> _, report = preprocess_for_sdc(raw_data, mode='report_only', return_metadata=True)
    """
    
    # Merge config with defaults (deep copy to avoid mutating defaults)
    import copy
    rules = copy.deepcopy(DEFAULT_PREPROCESSING_RULES)
    if config:
        for key, value in config.items():
            if key in rules and isinstance(value, dict):
                rules[key].update(value)
            else:
                rules[key] = value
    
    metadata = {
        'mode': mode,
        'steps_applied': [],
        'steps_skipped': [],
        'warnings': [],
        'changes_summary': {},
        'data_shape_before': data.shape,
        'data_shape_after': None
    }
    
    result = data.copy()
    
    # =========================================================================
    # MODE: NONE - Just warn about issues
    # =========================================================================
    if mode == 'none':
        # Quick scan for issues
        direct_ids = detect_greek_identifiers(result)
        try:
            from .sdc_utils import auto_detect_sensitive_columns
            direct_ids.update(auto_detect_sensitive_columns(result))
        except ImportError:
            pass
        
        if direct_ids:
            metadata['warnings'].append(
                f"Direct identifiers detected but not removed: {list(direct_ids.keys())}"
            )
            warnings.warn(f"Preprocessing skipped. Direct identifiers found: {list(direct_ids.keys())}")
        
        metadata['data_shape_after'] = result.shape
        if return_metadata:
            return result, metadata
        return result
    
    # =========================================================================
    # MODE: REPORT_ONLY - Don't modify, just assess
    # =========================================================================
    if mode == 'report_only':
        metadata['assessment'] = generate_pre_anonymization_report(
            data, quasi_identifiers
        )
        metadata['data_shape_after'] = result.shape
        if return_metadata:
            return result, metadata
        return result
    
    # =========================================================================
    # MANDATORY STEP: Remove direct identifiers
    # =========================================================================
    if rules['remove_direct_identifiers']['enabled']:
        # Detect identifiers
        try:
            from .sdc_utils import auto_detect_sensitive_columns
            standard_ids = auto_detect_sensitive_columns(result)
        except ImportError:
            standard_ids = {}
        
        greek_ids = detect_greek_identifiers(result) if rules['remove_direct_identifiers'].get('include_greek', True) else {}
        
        all_direct_ids = {**standard_ids, **greek_ids}

        # Never remove QIs or sensitive columns — they may have been
        # binned into patterns that look like identifiers (e.g. income
        # "34445-36015" mis-detected as phone number).
        _protected = set(quasi_identifiers or []) | set(
            rules.get('sensitive_columns') or config.get('sensitive_columns') or [])
        all_direct_ids = {c: r for c, r in all_direct_ids.items()
                          if c not in _protected}

        if all_direct_ids:
            result, step_meta = remove_direct_identifiers(
                result,
                columns=list(all_direct_ids.keys()),
                method=rules['remove_direct_identifiers'].get('method', 'remove'),
                hash_salt=rules['remove_direct_identifiers'].get('hash_salt'),
                return_metadata=True
            )
            metadata['steps_applied'].append('remove_direct_identifiers')
            metadata['changes_summary']['direct_identifiers'] = {
                'detected': all_direct_ids,
                'processed': step_meta
            }
            
            warnings.warn(
                f"Removed {len(all_direct_ids)} direct identifier column(s): "
                f"{list(all_direct_ids.keys())}"
            )
    
    # Exit early if mandatory only
    if mode == 'mandatory_only':
        metadata['data_shape_after'] = result.shape
        if return_metadata:
            return result, metadata
        return result
    
    # =========================================================================
    # AUTO-DETECT QUASI-IDENTIFIERS IF NEEDED
    # =========================================================================
    if quasi_identifiers is None:
        try:
            from .sdc_utils import auto_detect_quasi_identifiers
            quasi_identifiers = auto_detect_quasi_identifiers(result)
            metadata['quasi_identifiers_auto_detected'] = True
        except ImportError:
            quasi_identifiers = []
            metadata['warnings'].append(
                "Could not auto-detect quasi-identifiers. "
                "Provide explicitly for full preprocessing."
            )
    
    metadata['quasi_identifiers'] = quasi_identifiers

    # If requested, force-apply recommendations from the pre-anonymization report
    if force_apply_recommendations:
        try:
            report = generate_pre_anonymization_report(data, quasi_identifiers)
            recs = set(report.get('recommended_preprocessing', []) or [])
            # Map report recommendations to rule flags
            if 'remove_direct_identifiers' in recs:
                rules['remove_direct_identifiers']['enabled'] = True
            if 'apply_top_bottom_coding' in recs:
                rules['top_bottom_coding']['enabled'] = True
            if 'merge_rare_categories' in recs:
                rules['merge_rare_categories']['enabled'] = True
            if 'reduce_dimensionality' in recs:
                # Best-effort: enable merge rare and generalize
                rules['merge_rare_categories']['enabled'] = True
                rules['generalize']['enabled'] = True
            if 'apply_generalize' in recs or 'generalize' in recs:
                rules['generalize']['enabled'] = True
        except Exception:
            # Silently continue with defaults if report generation fails
            pass
    
    # =========================================================================
    # Per-QI treatment: build treatment-scaled parameter dicts
    # =========================================================================
    _qi_treatment = rules.get('qi_treatment')
    _per_qi_pctiles = None
    _per_qi_min_freq = None
    if _qi_treatment and quasi_identifiers:
        from .qi_treatment import build_per_qi_percentiles, build_per_qi_min_frequency
        _coding = rules.get('top_bottom_coding', {})
        _per_qi_pctiles = build_per_qi_percentiles(
            quasi_identifiers, _qi_treatment,
            base_bottom=_coding.get('bottom_percentile', 1),
            base_top=_coding.get('top_percentile', 99),
        )
        _merge = rules.get('merge_rare_categories', {})
        _per_qi_min_freq = build_per_qi_min_frequency(
            quasi_identifiers, _qi_treatment,
            base_min_frequency=_merge.get('min_frequency', 3),
        )

    # =========================================================================
    # CONDITIONAL STEP: Top/bottom coding for outliers
    # =========================================================================
    coding_rules = rules['top_bottom_coding']
    
    if coding_rules['enabled'] and quasi_identifiers:
        outlier_assessment = detect_reidentification_outliers(result, quasi_identifiers)
        metadata['changes_summary']['outlier_assessment'] = outlier_assessment['summary']
        
        should_apply = False
        if coding_rules['enabled'] == True:
            should_apply = True
        elif coding_rules['enabled'] == 'auto':
            # Apply if outliers exist and affect < threshold of records
            outlier_rate = outlier_assessment['summary'].get('outlier_rate', 0)
            should_apply = (
                outlier_rate > 0 and 
                outlier_rate < coding_rules.get('auto_threshold', 0.10)
            )
        
        if should_apply and outlier_assessment['problematic_columns']:
            result, step_meta = apply_top_bottom_coding(
                result,
                columns=list(outlier_assessment['problematic_columns'].keys()),
                method=coding_rules.get('method', 'percentile'),
                top_percentile=coding_rules.get('top_percentile', 99),
                bottom_percentile=coding_rules.get('bottom_percentile', 1),
                per_qi_percentiles=_per_qi_pctiles,
                return_metadata=True
            )
            metadata['steps_applied'].append('top_bottom_coding')
            metadata['changes_summary']['top_bottom_coding'] = step_meta
        elif outlier_assessment['summary'].get('outlier_rate', 0) >= coding_rules.get('auto_threshold', 0.10):
            metadata['steps_skipped'].append('top_bottom_coding')
            metadata['warnings'].append(
                f"High outlier rate ({outlier_assessment['summary']['outlier_rate']:.1%}) - "
                "skipped top/bottom coding. Consider manual review or binning."
            )
    
    # =========================================================================
    # CONDITIONAL STEP: Merge rare categories
    # =========================================================================
    merge_rules = rules['merge_rare_categories']

    if merge_rules['enabled'] and quasi_identifiers:
        # Use column_types from Configure to identify truly categorical columns.
        # Columns classified as numeric or date types in Configure are excluded
        # from merge_rare — they'll be handled by GENERALIZE (binning/truncation).
        _col_types = rules.get('column_types', {})
        _numeric_keywords = {'integer', 'float', 'numeric', 'continuous',
                             'income', 'financial', 'age', 'area', 'price',
                             'amount', 'year', 'date', 'temporal', 'time',
                             'datetime', 'birth'}

        def _is_semantic_numeric(col_name):
            # Check Configure's column_types first
            ct = _col_types.get(col_name, '').lower()
            if any(kw in ct for kw in _numeric_keywords):
                return True
            # Fallback: probe actual data for numeric/date strings
            if col_name in result.columns and result[col_name].dtype == 'object':
                sample = result[col_name].dropna().head(300)
                sample = sample[sample.astype(str).str.strip() != '']
                if len(sample) > 0:
                    # Numeric strings?
                    num_ct = pd.to_numeric(sample, errors='coerce').notna().sum()
                    if num_ct / len(sample) > 0.8:
                        return True
                    # European decimal format?
                    if sample.astype(str).str.contains(',').mean() > 0.3:
                        eu = (sample.astype(str)
                              .str.replace('.', '', regex=False)
                              .str.replace(',', '.', regex=False))
                        if pd.to_numeric(eu, errors='coerce').notna().sum() / len(sample) > 0.8:
                            return True
                    # Date strings?
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            date_ct = pd.to_datetime(
                                sample, errors='coerce', dayfirst=True
                            ).notna().sum()
                        if date_ct / len(sample) > 0.8:
                            return True
                    except Exception:
                        pass
            return False

        categorical_qis = [
            qi for qi in quasi_identifiers
            if qi in result.columns
            and result[qi].dtype == 'object'
            and not _is_semantic_numeric(qi)
        ]
        
        if categorical_qis:
            columns_to_merge = []
            
            for col in categorical_qis:
                value_counts = result[col].value_counts()
                rare_count = (value_counts < merge_rules.get('min_frequency', 3)).sum()
                total_categories = len(value_counts)
                
                should_merge = False
                if merge_rules['enabled'] == True:
                    should_merge = rare_count > 0
                elif merge_rules['enabled'] == 'auto':
                    # Merge if >threshold of categories are rare
                    should_merge = (
                        rare_count > 0 and 
                        (rare_count / total_categories) > merge_rules.get('auto_threshold', 0.20)
                    )
                
                if should_merge:
                    columns_to_merge.append(col)
            
            if columns_to_merge:
                result, step_meta = merge_rare_categories(
                    result,
                    columns=columns_to_merge,
                    min_frequency=merge_rules.get('min_frequency', 3),
                    other_label=merge_rules.get('other_label', 'Other'),
                    per_qi_min_frequency=_per_qi_min_freq,
                    return_metadata=True
                )
                metadata['steps_applied'].append('merge_rare_categories')
                metadata['changes_summary']['merge_rare_categories'] = step_meta

    # =========================================================================
    # CONDITIONAL STEP: GENERALIZE (reduce QI cardinality)
    # =========================================================================
    gen_rules = rules.get('generalize', {})
    if gen_rules and gen_rules.get('enabled') and quasi_identifiers:
        try:
            from .GENERALIZE import apply_generalize
            qis_to_gen = [qi for qi in quasi_identifiers if qi in result.columns]
            if qis_to_gen:
                result, gen_meta = apply_generalize(
                    result,
                    quasi_identifiers=qis_to_gen,
                    max_categories=gen_rules.get('max_categories', 10),
                    max_categories_per_qi=gen_rules.get('max_categories_per_qi'),
                    strategy=gen_rules.get('strategy', 'auto'),
                    return_metadata=True,
                    adaptive_binning=gen_rules.get('adaptive_binning', False),
                    verbose=False,
                    var_priority=gen_rules.get('var_priority'),
                    reid_target=gen_rules.get('reid_target'),
                    utility_fn=gen_rules.get('utility_fn'),
                    utility_threshold=gen_rules.get('utility_threshold'),
                    column_types=rules.get('column_types'),
                    qi_treatment=_qi_treatment,
                )
                metadata['steps_applied'].append('generalize')
                metadata['changes_summary']['generalize'] = gen_meta
        except Exception as e:
            metadata['warnings'].append(f"GENERALIZE step failed: {e}")
    
    # =========================================================================
    # WARNING STEP: Dimensionality check
    # =========================================================================
    dim_rules = rules['dimensionality_check']
    
    if dim_rules['enabled'] and quasi_identifiers:
        dim_risk = assess_dimensionality_risk(result, quasi_identifiers)
        metadata['changes_summary']['dimensionality_risk'] = dim_risk
        
        if dim_risk['risk_level'] in ['HIGH', 'CRITICAL']:
            metadata['warnings'].append(
                f"Dimensionality risk: {dim_risk['risk_level']}. "
                f"{dim_risk['recommendation']}. "
                f"Consider removing: {dim_risk['suggested_removals']}"
            )
    
    # =========================================================================
    # FINALIZE
    # =========================================================================
    metadata['data_shape_after'] = result.shape
    
    if return_metadata:
        return result, metadata
    return result


# =============================================================================
# PRE-ANONYMIZATION REPORT GENERATOR
# =============================================================================

def generate_pre_anonymization_report(
    data: pd.DataFrame,
    quasi_identifiers: List[str] = None
) -> Dict:
    """
    Generate comprehensive pre-anonymization assessment report.
    
    This report supports ΤΜΑ Stage 1-2 documentation requirements
    by providing a complete data assessment before SDC processing.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    quasi_identifiers : List[str], optional
        Quasi-identifier columns. Auto-detected if not provided.
        
    Returns
    -------
    Dict
        Comprehensive assessment report
        
    Examples
    --------
    >>> report = generate_pre_anonymization_report(data)
    >>> print(report['ready_for_sdc'])
    False
    >>> print(report['blocking_issues'])
    ['Direct identifiers detected: name, email']
    """
    
    # Auto-detect QIs if needed
    if quasi_identifiers is None:
        try:
            from .sdc_utils import auto_detect_quasi_identifiers
            quasi_identifiers = auto_detect_quasi_identifiers(data)
        except ImportError:
            quasi_identifiers = []
    
    # Detect identifiers
    try:
        from .sdc_utils import auto_detect_sensitive_columns, calculate_disclosure_risk, detect_data_type
        standard_sensitive = auto_detect_sensitive_columns(data)
        disclosure_risk = calculate_disclosure_risk(data, quasi_identifiers) if quasi_identifiers else {}
        data_type = detect_data_type(data)
    except ImportError:
        standard_sensitive = {}
        disclosure_risk = {}
        data_type = 'unknown'
    
    greek_sensitive = detect_greek_identifiers(data)
    
    # Build report
    report = {
        'dataset_info': {
            'records': len(data),
            'columns': len(data.columns),
            'column_names': list(data.columns),
            'data_type': data_type
        },
        
        'direct_identifiers': {
            'standard': standard_sensitive,
            'greek_specific': greek_sensitive,
            'total_detected': len(standard_sensitive) + len(greek_sensitive),
            'action_required': len(standard_sensitive) + len(greek_sensitive) > 0
        },
        
        'quasi_identifiers': {
            'detected': quasi_identifiers,
            'count': len(quasi_identifiers)
        },
        
        'disclosure_risk': disclosure_risk,
        
        'blocking_issues': [],
        'warnings': [],
        'recommended_preprocessing': [],
        'ready_for_sdc': True
    }
    
    # Assess dimensionality if QIs exist
    if quasi_identifiers:
        report['dimensionality'] = assess_dimensionality_risk(data, quasi_identifiers)
        
        # Assess outliers
        report['outlier_risk'] = detect_reidentification_outliers(data, quasi_identifiers)
        
        # Check rare categories
        report['rare_categories'] = {}
        for col in quasi_identifiers:
            if col in data.columns and data[col].dtype == 'object':
                rare = data[col].value_counts()[data[col].value_counts() < 3]
                if len(rare) > 0:
                    report['rare_categories'][col] = {
                        'count': len(rare),
                        'values': rare.index.tolist()[:10]  # Limit display
                    }
    
    # Determine blocking issues and recommendations
    if report['direct_identifiers']['action_required']:
        all_ids = list(standard_sensitive.keys()) + list(greek_sensitive.keys())
        report['blocking_issues'].append(
            f"Direct identifiers detected: {all_ids}"
        )
        report['recommended_preprocessing'].append('remove_direct_identifiers')
        report['ready_for_sdc'] = False
    
    if quasi_identifiers:
        if report.get('outlier_risk', {}).get('high_risk_records'):
            report['recommended_preprocessing'].append('apply_top_bottom_coding')
        
        if report.get('dimensionality', {}).get('risk_level') in ['HIGH', 'CRITICAL']:
            report['warnings'].append(
                f"High dimensionality risk: {report['dimensionality']['recommendation']}"
            )
            report['recommended_preprocessing'].append('reduce_dimensionality')
        
        if report.get('rare_categories'):
            report['recommended_preprocessing'].append('merge_rare_categories')
    
    return report


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def integrate_with_analyze_data(original_analyze_data):
    """
    Decorator to add preprocessing to existing analyze_data function.
    
    Usage:
        from .sdc_utils import analyze_data
        analyze_data = integrate_with_analyze_data(analyze_data)
    """
    def wrapped_analyze_data(
        data: pd.DataFrame,
        quasi_identifiers=None,
        preprocess: bool = True,
        preprocess_mode: str = 'auto',
        verbose: bool = True
    ):
        if preprocess:
            data, prep_meta = preprocess_for_sdc(
                data,
                quasi_identifiers=quasi_identifiers,
                mode=preprocess_mode,
                return_metadata=True
            )
            if verbose and prep_meta['steps_applied']:
                print(f"\n[Preprocessing] Applied: {prep_meta['steps_applied']}")
                for w in prep_meta.get('warnings', []):
                    print(f"  ⚠️  {w}")
        
        result = original_analyze_data(data, quasi_identifiers, verbose)
        result['preprocessing'] = prep_meta if preprocess else None
        return result
    
    return wrapped_analyze_data


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    'DEFAULT_PREPROCESSING_RULES',

    # Main functions
    'preprocess_for_sdc',
    'generate_pre_anonymization_report',

    # Detection functions
    'detect_greek_identifiers',
    'detect_reidentification_outliers',

    # Transformation functions
    'remove_direct_identifiers',
    'apply_top_bottom_coding',
    'merge_rare_categories',
    'apply_numeric_rounding',
    'apply_date_truncation',
    'apply_age_binning',
    'apply_geographic_coarsening',
    'apply_string_truncation',
    'apply_record_sampling',
    'apply_generalize',

    # Assessment functions
    'assess_dimensionality_risk',

    # Integration helpers
    'integrate_with_analyze_data',
]


# =============================================================================
# CLI / STANDALONE USAGE
# =============================================================================

if __name__ == '__main__':
    import sys
    
    print("=" * 70)
    print("  SDC Preprocessing Module - Test Run")
    print("=" * 70)
    
    # Create test data
    np.random.seed(42)
    n = 200
    
    test_data = pd.DataFrame({
        # Direct identifiers (should be removed)
        'name': [f'Person_{i}' for i in range(n)],
        'email': [f'person{i}@test.com' for i in range(n)],
        'afm': [f'{100000000 + i}' for i in range(n)],  # Greek tax ID
        
        # Quasi-identifiers
        'age': np.random.randint(18, 85, n),
        'gender': np.random.choice(['M', 'F'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n),
        'occupation': np.random.choice(
            ['Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Other', 
             'Rare_Job_1', 'Rare_Job_2', 'Rare_Job_3'],  # Some rare categories
            n,
            p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.02, 0.02, 0.01]
        ),
        
        # Sensitive attribute
        'income': np.concatenate([
            np.random.normal(50000, 15000, n-5),
            [200000, 250000, 300000, 5000, 3000]  # Outliers
        ])
    })
    
    print(f"\nTest data: {test_data.shape[0]} records, {test_data.shape[1]} columns")
    print(f"Columns: {list(test_data.columns)}")
    
    # Test 1: Report only
    print("\n" + "-" * 50)
    print("TEST 1: Generate pre-anonymization report")
    print("-" * 50)
    
    _, report = preprocess_for_sdc(
        test_data, 
        mode='report_only',
        return_metadata=True
    )
    
    print(f"\nDirect identifiers found: {report['assessment']['direct_identifiers']['total_detected']}")
    print(f"Ready for SDC: {report['assessment']['ready_for_sdc']}")
    print(f"Blocking issues: {report['assessment']['blocking_issues']}")
    print(f"Recommendations: {report['assessment']['recommended_preprocessing']}")
    
    # Test 2: Auto preprocessing
    print("\n" + "-" * 50)
    print("TEST 2: Auto preprocessing")
    print("-" * 50)
    
    clean_data, meta = preprocess_for_sdc(
        test_data,
        quasi_identifiers=['age', 'gender', 'region', 'occupation'],
        mode='auto',
        return_metadata=True
    )
    
    print(f"\nSteps applied: {meta['steps_applied']}")
    print(f"Shape before: {meta['data_shape_before']}")
    print(f"Shape after: {meta['data_shape_after']}")
    print(f"Warnings: {meta['warnings']}")
    
    if 'direct_identifiers' in meta['changes_summary']:
        print(f"Identifiers removed: {list(meta['changes_summary']['direct_identifiers']['detected'].keys())}")
    
    if 'top_bottom_coding' in meta['changes_summary']:
        print(f"Outliers coded: {meta['changes_summary']['top_bottom_coding']['records_affected']} records")
    
    # Test 3: Greek identifier detection
    print("\n" + "-" * 50)
    print("TEST 3: Greek identifier detection")
    print("-" * 50)
    
    greek_test = pd.DataFrame({
        'tax_id': ['123456789', '234567890', '345678901'],
        'amka_number': ['12345678901', '23456789012', '34567890123'],
        'regular_col': ['A', 'B', 'C']
    })
    
    detected = detect_greek_identifiers(greek_test)
    print(f"Detected: {detected}")
    
    print("\n" + "=" * 70)
    print("  All tests completed!")
    print("=" * 70)
