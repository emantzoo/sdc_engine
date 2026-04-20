"""
Column Type Detection
=====================

Identify and classify column types for SDC method selection.
"""

import logging
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any, Optional

_log = logging.getLogger(__name__)

from ..config import COLUMN_TYPE_KEYWORDS, DIRECT_IDENTIFIER_KEYWORDS, DIRECT_IDENTIFIER_PATTERNS


def identify_column_types(data: pd.DataFrame) -> Dict[str, str]:
    """
    Identify the type of each column in the dataset.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset

    Returns:
    --------
    dict : {column_name: type}
        where type is one of: 'continuous', 'categorical', 'identifier', 'binary'

    Notes:
    ------
    - Identifiers are detected by column name keywords (id, ssn, uuid, etc.)
      OR by being sequential integers starting near 0 or 1
    - Continuous variables are numeric with high cardinality but non-sequential values
    - Categorical variables have low cardinality (< 20 unique values for integers)
    - Binary variables have exactly 2 unique values
    """
    id_keywords = COLUMN_TYPE_KEYWORDS.get('identifier', [
        'id', 'uuid', 'guid', 'ssn', 'nin', 'passport', 'license',
        'serial', 'code', 'key', 'index', 'number', 'num', 'no'
    ])
    value_keywords = [
        'income', 'salary', 'wage', 'price', 'cost', 'amount',
        'score', 'rating', 'value', 'total', 'sum', 'count',
        'weight', 'height', 'age', 'year', 'date', 'time',
        'balance', 'revenue', 'profit', 'loss', 'tax', 'fee'
    ]

    column_types = {}

    for col in data.columns:
        col_lower = col.lower()

        is_id_by_name = any(kw in col_lower for kw in id_keywords)
        is_value_by_name = any(kw in col_lower for kw in value_keywords)

        if pd.api.types.is_numeric_dtype(data[col]):
            n_unique = data[col].nunique()
            uniqueness_ratio = n_unique / len(data) if len(data) > 0 else 0

            # Special-case age-like columns: treat as categorical/ordinal rather than continuous
            if 'age' in col_lower and pd.api.types.is_integer_dtype(data[col]):
                column_types[col] = 'categorical'
                continue

            if n_unique <= 2:
                column_types[col] = 'binary'
            elif is_id_by_name and not is_value_by_name:
                column_types[col] = 'identifier'
            elif is_value_by_name:
                column_types[col] = 'continuous'
            elif _is_sequential_id(data[col]):
                column_types[col] = 'identifier'
            elif n_unique < 20 and (data[col] % 1 == 0).all():
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'continuous'
        else:
            uniqueness_ratio = data[col].nunique() / len(data) if len(data) > 0 else 0

            if is_id_by_name and not is_value_by_name:
                column_types[col] = 'identifier'
            elif uniqueness_ratio > 0.95:
                column_types[col] = 'identifier'
            else:
                column_types[col] = 'categorical'

    return column_types


def detect_data_type(data: pd.DataFrame) -> str:
    """
    Detect whether data is microdata or an aggregated table.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset

    Returns:
    --------
    str : 'microdata' or 'tabular'

    Logic:
    - If all values are numeric and relatively small, likely a table
    - If mixed types or large numeric values, likely microdata
    - Tables with Total rows/columns are still detected as tabular
    - Named string index suggests tabular data (frequency table)
    """
    all_numeric = data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]

    if all_numeric:
        has_named_index = (
            not isinstance(data.index, pd.RangeIndex) and
            data.index.dtype == 'object' and
            len(data.index) > 0
        )

        data_no_totals = data.copy()
        if 'Total' in data_no_totals.index:
            data_no_totals = data_no_totals.drop('Total', axis=0)
        if 'Total' in data_no_totals.columns:
            data_no_totals = data_no_totals.drop('Total', axis=1)

        all_integers = (data_no_totals % 1 == 0).all().all() if len(data_no_totals) > 0 else False

        if all_integers:
            max_val = data_no_totals.max().max() if len(data_no_totals) > 0 else 0
            if max_val < 10000 or (has_named_index and max_val < 50000):
                return 'tabular'

    return 'microdata'


def auto_detect_continuous_variables(
    data: pd.DataFrame,
    exclude_identifiers: bool = True
) -> List[str]:
    """
    Automatically detect continuous variables suitable for perturbative methods.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    exclude_identifiers : bool, default=True
        Whether to exclude identifier-like columns

    Returns:
    --------
    list of str : Column names of continuous variables
    """
    column_types = identify_column_types(data)

    continuous = []
    for col, col_type in column_types.items():
        if col_type == 'continuous':
            if exclude_identifiers or column_types.get(col) != 'identifier':
                continuous.append(col)

    return continuous


def auto_detect_categorical_variables(
    data: pd.DataFrame,
    exclude_identifiers: bool = True,
    max_cardinality: int = 50
) -> List[str]:
    """
    Automatically detect categorical variables.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    exclude_identifiers : bool, default=True
        Whether to exclude identifier-like columns
    max_cardinality : int, default=50
        Maximum number of unique values for a variable to be considered categorical

    Returns:
    --------
    list of str : Column names of categorical variables
    """
    column_types = identify_column_types(data)

    categorical = []
    for col, col_type in column_types.items():
        if col_type in ['categorical', 'binary']:
            if exclude_identifiers or column_types.get(col) != 'identifier':
                if data[col].nunique() <= max_cardinality:
                    categorical.append(col)

    return categorical


def auto_detect_direct_identifiers(
    data: pd.DataFrame,
    check_patterns: bool = True
) -> Dict[str, str]:
    """
    Automatically detect columns containing DIRECT IDENTIFIERS that should be EXCLUDED.

    Direct identifiers are columns that can directly identify individuals without
    linking to external data. These should be REMOVED before applying SDC methods.

    NOTE: This is different from "sensitive attributes" (income, diagnosis, health_score)
    which are the values you want to PROTECT via SDC methods like k-anonymity.

    Column Types in SDC:
    - Direct Identifiers: name, email, SSN → EXCLUDE (detected by this function)
    - Quasi-Identifiers: age, gender, zipcode → PROTECT via SDC methods
    - Sensitive Attributes: income, diagnosis → KEEP, protected by QI anonymization

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    check_patterns : bool, default=True
        If True, also check column values for patterns (email, phone, etc.)

    Returns:
    --------
    dict : {column_name: identifier_type}
        identifier_type is one of: 'name', 'email', 'phone', 'address',
        'national_id', 'financial', 'medical_id', 'other_identifier'

    Note:
    -----
    Keywords and patterns are defined in src/config.py:
    - DIRECT_IDENTIFIER_KEYWORDS: column name patterns
    - DIRECT_IDENTIFIER_PATTERNS: regex patterns for value detection
    """
    # Use config-defined keywords and patterns
    identifier_keywords = DIRECT_IDENTIFIER_KEYWORDS
    value_patterns = DIRECT_IDENTIFIER_PATTERNS

    detected_identifiers = {}

    for col in data.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')

        # Check column name against keywords
        for identifier_type, keywords in identifier_keywords.items():
            # Use word boundaries to avoid false positives like 'tin' in 'rating'
            # Split column name by underscore and check if any keyword matches a full word
            col_words = col_lower.split('_')
            if any(kw in col_words or kw == col_lower for kw in keywords):
                detected_identifiers[col] = identifier_type
                break

        # Check column values against patterns (if not already detected)
        if col not in detected_identifiers and check_patterns:
            if data[col].dtype == 'object':
                sample = data[col].dropna().head(100).astype(str)

                if len(sample) > 0:
                    for pattern_name, pattern in value_patterns.items():
                        try:
                            matches = sample.str.match(pattern, na=False).sum()
                            if matches / len(sample) > 0.5:
                                detected_identifiers[col] = pattern_name
                                break
                        except (ValueError, TypeError):
                            pass  # Pattern matching failure

    return detected_identifiers


# Backward compatibility alias
auto_detect_sensitive_columns = auto_detect_direct_identifiers


def auto_detect_dimensions(
    data: pd.DataFrame,
    max_dimensions: int = 3,
    prefer_categorical: bool = True
) -> List[str]:
    """
    Automatically detect appropriate dimensions for creating frequency tables.

    Parameters:
    -----------
    data : pd.DataFrame
        Input microdata
    max_dimensions : int, default=3
        Maximum number of dimensions to select
    prefer_categorical : bool, default=True
        Whether to prefer categorical over continuous variables

    Returns:
    --------
    list of str : Recommended dimension column names

    Raises:
    -------
    ValueError : If cannot find suitable dimensions
    """
    column_types = identify_column_types(data)

    candidates = {col: ctype for col, ctype in column_types.items()
                  if ctype != 'identifier'}

    if len(candidates) == 0:
        raise ValueError("No suitable dimensions found (only identifier columns)")

    if prefer_categorical:
        categorical = [col for col, ctype in candidates.items()
                       if ctype in ['categorical', 'binary']]

        if len(categorical) >= 2:
            good_cats = []
            for col in categorical:
                n_unique = data[col].nunique()
                if 2 <= n_unique <= 20:
                    good_cats.append(col)

            if len(good_cats) >= 2:
                return good_cats[:max_dimensions]
            elif len(categorical) >= 2:
                return categorical[:max_dimensions]

    available = list(candidates.keys())
    if len(available) >= 2:
        return available[:max_dimensions]

    raise ValueError(f"Need at least 2 dimensions, found only {len(available)}")


def _is_sequential_id(series: pd.Series) -> bool:
    """
    Check if a numeric series looks like a sequential identifier.

    Sequential IDs typically:
    - Are integers
    - Start near 0 or 1
    - Have differences of approximately 1 between sorted values
    """
    try:
        if not (series % 1 == 0).all():
            return False

        min_val = series.min()
        if min_val > 10:
            return False

        sorted_vals = series.sort_values().values
        if len(sorted_vals) < 2:
            return False

        diffs = np.diff(sorted_vals)
        median_diff = np.median(diffs)

        return median_diff <= 1.5 and min_val >= 0
    except (ValueError, TypeError) as exc:
        _log.warning("[column_types] Sequential ID detection failed: %s", exc)
        return False
