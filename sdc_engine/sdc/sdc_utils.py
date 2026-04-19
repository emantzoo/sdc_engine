"""
SDC Utilities (sdc_utils.py)
==============================

Description:
Common utility functions shared across all Statistical Disclosure Control (SDC) methods.
These functions provide data analysis, transformation, validation, and metrics calculation
capabilities that are used by multiple anonymization methods.

This module avoids code duplication and ensures consistency across different methods.

Author: SDC Methods Implementation
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
import warnings


# =============================================================================
# AUTO-DETECTION FUNCTIONS
# =============================================================================

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
    
    # Filter out identifiers
    candidates = {col: ctype for col, ctype in column_types.items() 
                 if ctype != 'identifier'}
    
    if len(candidates) == 0:
        raise ValueError("No suitable dimensions found (only identifier columns)")
    
    # Prioritize categorical and binary
    if prefer_categorical:
        categorical = [col for col, ctype in candidates.items() 
                      if ctype in ['categorical', 'binary']]
        
        if len(categorical) >= 2:
            # Select categorical with reasonable cardinality
            good_cats = []
            for col in categorical:
                n_unique = data[col].nunique()
                if 2 <= n_unique <= 20:  # Reasonable for cross-tabulation
                    good_cats.append(col)
            
            if len(good_cats) >= 2:
                return good_cats[:max_dimensions]
            elif len(categorical) >= 2:
                return categorical[:max_dimensions]
    
    # Fallback: use any available columns
    available = list(candidates.keys())
    if len(available) >= 2:
        return available[:max_dimensions]
    
    raise ValueError(f"Need at least 2 dimensions, found only {len(available)}")


def auto_detect_quasi_identifiers(
    data: pd.DataFrame,
    exclude_identifiers: bool = True,
    max_qis: int = 5
) -> List[str]:
    """
    Automatically detect likely quasi-identifiers in a dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input microdata
    exclude_identifiers : bool, default=True
        Whether to exclude columns that look like identifiers
    max_qis : int, default=5
        Maximum number of QIs to select
    
    Returns:
    --------
    list of str : Recommended quasi-identifier column names
    
    Raises:
    -------
    ValueError : If cannot find suitable QIs
    """
    
    from .config import QI_KEYWORDS

    column_types = identify_column_types(data)

    # Get QI keywords from config (combine definite and probable)
    qi_keywords = list(QI_KEYWORDS.get('definite_qis', {}).keys())
    qi_keywords.extend(QI_KEYWORDS.get('probable_qis', {}).keys())

    potential_qis = []
    
    for col, col_type in column_types.items():
        # Skip identifiers if requested
        if exclude_identifiers and col_type == 'identifier':
            continue
        
        # Include if column name matches QI keywords
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in qi_keywords):
            potential_qis.append(col)
            continue
        
        # Include categorical and binary columns
        if col_type in ['categorical', 'binary']:
            potential_qis.append(col)
            continue
        
        # Include some continuous columns with reasonable cardinality
        if col_type == 'continuous':
            n_unique = data[col].nunique()
            if n_unique < 100:  # Reasonable cardinality for a QI
                potential_qis.append(col)
    
    if len(potential_qis) == 0:
        raise ValueError("No suitable quasi-identifiers found")

    # Limit to max_qis
    return potential_qis[:max_qis]


# =============================================================================
# ENHANCED QI DETECTION (Phase 1 + Phase 2)
# =============================================================================

def detect_quasi_identifiers_enhanced(
    data: pd.DataFrame,
    context: Optional[Dict] = None,
    confidence_threshold: float = 0.6,
    return_scores: bool = False
) -> Union[List[str], Dict]:
    """
    Multi-tiered QI detection with confidence scoring.

    Uses multiple signals to detect quasi-identifiers:
    - Column name patterns (demographic keywords)
    - Data type and cardinality characteristics
    - Sequential ID detection (to exclude)
    - Uniqueness contribution (how much does this column increase re-identification risk)

    Tiers:
    - DEFINITE (confidence >= 0.9): Clear demographic identifiers
    - PROBABLE (confidence 0.6-0.9): Likely identifiers based on characteristics
    - POSSIBLE (confidence 0.3-0.6): Uncertain, may need validation
    - NON_QI (confidence < 0.3): Unlikely to identify individuals

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    context : dict, optional
        Domain context: {'data_category': 'health'|'financial'|'administrative'}
    confidence_threshold : float, default=0.6
        Minimum confidence to include as QI
    return_scores : bool, default=False
        If True, return full scoring dict instead of just column list

    Returns:
    --------
    list or dict : QI columns (or full scores if return_scores=True)

    Examples:
    ---------
    >>> qis = detect_quasi_identifiers_enhanced(data)
    >>> print(qis)
    ['age', 'gender', 'zipcode', 'occupation']

    >>> scores = detect_quasi_identifiers_enhanced(data, return_scores=True)
    >>> for col, score in scores.items():
    ...     print(f"{col}: {score['confidence']:.2f} ({score['tier']})")
    """

    qi_scores = {}

    # Get column types for reference
    column_types = identify_column_types(data)

    for col in data.columns:
        score = _calculate_qi_score(
            column=col,
            data=data[col],
            full_data=data,
            column_type=column_types.get(col, 'unknown'),
            context=context
        )
        qi_scores[col] = score

    if return_scores:
        return qi_scores

    # Return columns above threshold, sorted by score
    qis = [col for col, score in qi_scores.items()
           if score['confidence'] >= confidence_threshold]
    qis.sort(key=lambda x: qi_scores[x]['confidence'], reverse=True)

    return qis


def _calculate_qi_score(
    column: str,
    data: pd.Series,
    full_data: pd.DataFrame,
    column_type: str,
    context: Optional[Dict]
) -> Dict:
    """
    Calculate comprehensive QI score for a column.

    Scoring dimensions:
    1. Name-based signals (keywords in column name)
    2. Type-based signals (data type, cardinality)
    3. Uniqueness contribution (how much does this column contribute to uniqueness)
    """

    score = {
        'column': column,
        'confidence': 0.0,
        'reasons': [],
        'signals': {},
        'tier': None
    }

    # ========================================================================
    # DIMENSION 1: Name-Based Detection (0.0-1.0)
    # ========================================================================

    name_result = _score_column_name(column)
    score['signals']['name'] = name_result

    # Early exit for direct identifiers
    if name_result.get('is_identifier'):
        score['confidence'] = 0.0
        score['tier'] = 'NON_QI'
        score['reasons'] = name_result.get('reasons', [])
        return score

    # ========================================================================
    # DIMENSION 2: Type-Based Detection (0.0-1.0)
    # ========================================================================

    type_result = _score_data_type_for_qi(data, column, column_type)
    score['signals']['type'] = type_result

    # Early exit for sequential IDs
    if type_result.get('is_sequential_id'):
        score['confidence'] = 0.0
        score['tier'] = 'NON_QI'
        score['reasons'] = ['Sequential ID detected']
        return score

    # ========================================================================
    # DIMENSION 3: Uniqueness Contribution (0.0-1.0)
    # ========================================================================

    uniqueness_result = _score_uniqueness_contribution(data, full_data, column)
    score['signals']['uniqueness'] = uniqueness_result

    # ========================================================================
    # Combine Scores with Weights
    # ========================================================================

    weights = {
        'name': 0.40,       # Name is strong signal
        'type': 0.25,       # Type helps but not definitive
        'uniqueness': 0.35  # Uniqueness contribution is key
    }

    total_score = (
        score['signals']['name']['score'] * weights['name'] +
        score['signals']['type']['score'] * weights['type'] +
        score['signals']['uniqueness']['score'] * weights['uniqueness']
    )

    # Apply boosts
    if name_result.get('definite_qi'):
        total_score = min(total_score * 1.2, 1.0)

    score['confidence'] = total_score

    # Determine tier
    if total_score >= 0.9:
        score['tier'] = 'DEFINITE'
    elif total_score >= 0.6:
        score['tier'] = 'PROBABLE'
    elif total_score >= 0.3:
        score['tier'] = 'POSSIBLE'
    else:
        score['tier'] = 'NON_QI'

    # Compile reasons
    for dim, dim_score in score['signals'].items():
        if dim_score.get('reasons'):
            score['reasons'].extend(dim_score['reasons'])

    return score


def _score_column_name(column: str) -> Dict:
    """
    Score based on column name keywords.
    """

    col_lower = column.lower()
    col_normalized = col_lower.replace('_', ' ').replace('-', ' ')

    result = {
        'score': 0.0,
        'reasons': [],
        'is_identifier': False,
        'definite_qi': False
    }

    # -------------------------------------------------------------------------
    # Direct Identifiers (NOT quasi-identifiers) - score 0.0
    # -------------------------------------------------------------------------

    direct_identifiers = [
        'id', 'ssn', 'social_security', 'passport', 'license', 'ein',
        'tax_id', 'national_id', 'patient_id', 'employee_id', 'customer_id',
        'account_number', 'card_number', 'iban', 'swift',
        'name', 'full_name', 'first_name', 'last_name', 'surname',
        'email', 'phone', 'telephone', 'mobile', 'address', 'street'
    ]

    for identifier in direct_identifiers:
        if identifier in col_lower or identifier in col_normalized:
            # Check it's not a partial match (e.g., 'provider' contains 'id')
            if identifier == 'id':
                # More careful check for 'id'
                if col_lower == 'id' or col_lower.endswith('_id') or col_lower.startswith('id_'):
                    result['score'] = 0.0
                    result['is_identifier'] = True
                    result['reasons'].append(f"Direct identifier: {identifier}")
                    return result
            else:
                result['score'] = 0.0
                result['is_identifier'] = True
                result['reasons'].append(f"Direct identifier: {identifier}")
                return result

    # -------------------------------------------------------------------------
    # Definite QI Keywords - score 0.85-1.0
    # -------------------------------------------------------------------------

    definite_qis = {
        # Demographics
        'age': 1.0, 'birth': 0.95, 'dob': 0.95, 'birthday': 0.95,
        'gender': 1.0, 'sex': 1.0,
        'race': 0.95, 'ethnicity': 0.95, 'ethnic': 0.95,

        # Location
        'zip': 1.0, 'zipcode': 1.0, 'postal': 1.0, 'postcode': 1.0,
        'city': 0.95, 'town': 0.95, 'municipality': 0.95,
        'county': 0.90, 'district': 0.90, 'region': 0.85,
        'state': 0.85, 'province': 0.85,
        'country': 0.70,
        'location': 0.85, 'residence': 0.90,

        # Education/Occupation
        'education': 0.85, 'degree': 0.80, 'qualification': 0.80,
        'occupation': 0.90, 'job': 0.85, 'profession': 0.85,
        'employer': 0.85, 'industry': 0.70, 'sector': 0.70,

        # Family/Household
        'marital': 0.85, 'married': 0.85, 'marriage': 0.85,
        'children': 0.75, 'dependents': 0.75,
        'household': 0.80, 'family': 0.75,

        # Other
        'nationality': 0.90, 'citizenship': 0.90,
        'language': 0.75, 'religion': 0.85,
        'income': 0.80, 'salary': 0.80,
        'veteran': 0.85, 'disability': 0.85
    }

    for keyword, keyword_score in definite_qis.items():
        if keyword in col_lower or keyword in col_normalized:
            result['score'] = keyword_score
            result['definite_qi'] = True
            result['reasons'].append(f"QI keyword: {keyword}")
            return result

    # -------------------------------------------------------------------------
    # Probable QI Keywords - score 0.5-0.75
    # -------------------------------------------------------------------------

    probable_qis = {
        'year': 0.65, 'month': 0.60, 'date': 0.70,
        'type': 0.55, 'category': 0.60, 'class': 0.60,
        'status': 0.65, 'group': 0.60,
        'department': 0.70, 'division': 0.65, 'unit': 0.65,
        'position': 0.70, 'title': 0.70, 'role': 0.70,
        'level': 0.65, 'grade': 0.70, 'rank': 0.70,
        'experience': 0.70, 'tenure': 0.70, 'seniority': 0.70
    }

    for keyword, keyword_score in probable_qis.items():
        if keyword in col_lower or keyword in col_normalized:
            result['score'] = keyword_score
            result['reasons'].append(f"Probable QI keyword: {keyword}")
            return result

    # No keyword match
    result['score'] = 0.0
    result['reasons'].append("No QI keywords in name")
    return result


def _score_data_type_for_qi(data: pd.Series, column: str, column_type: str) -> Dict:
    """
    Score based on data type and cardinality characteristics.
    """

    result = {
        'score': 0.0,
        'reasons': [],
        'is_sequential_id': False
    }

    n_total = len(data)
    n_unique = data.nunique()
    cardinality_ratio = n_unique / n_total if n_total > 0 else 0
    col_lower = column.lower()

    # -------------------------------------------------------------------------
    # Sequential ID Detection (only for numeric data)
    # Note: Override column_type='identifier' for known QI patterns
    # -------------------------------------------------------------------------

    # Check if this is a known QI pattern that got misclassified as identifier
    qi_override_patterns = ['zip', 'postal', 'diagnosis', 'icd', 'cpt', 'drg']
    is_qi_override = any(pattern in col_lower for pattern in qi_override_patterns)

    is_seq_id = False
    if column_type == 'identifier' and not is_qi_override:
        # Trust identifier classification unless it's a known QI pattern
        is_seq_id = True
    elif pd.api.types.is_numeric_dtype(data.dtype):
        is_seq_id = _is_sequential_id(data)

    if is_seq_id:
        result['score'] = 0.0
        result['is_sequential_id'] = True
        result['reasons'].append("Sequential ID detected")
        return result

    # -------------------------------------------------------------------------
    # Type-based scoring
    # -------------------------------------------------------------------------

    dtype = data.dtype

    if dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
        # Categorical data
        if 0.02 < cardinality_ratio < 0.50:
            # Moderate cardinality categorical - good QI candidate
            result['score'] = 0.8
            result['reasons'].append(f"Categorical with moderate cardinality ({n_unique} unique)")
        elif cardinality_ratio <= 0.02:
            # Very low cardinality (e.g., gender with 2 values)
            result['score'] = 0.7
            result['reasons'].append(f"Low-cardinality categorical ({n_unique} unique)")
        elif 0.50 <= cardinality_ratio < 0.90:
            # High cardinality but not unique
            result['score'] = 0.5
            result['reasons'].append(f"High-cardinality categorical ({n_unique} unique)")
        else:
            # Near-unique (> 90%) - might be identifier
            result['score'] = 0.2
            result['reasons'].append(f"Near-unique categorical - possibly identifier")

    elif pd.api.types.is_numeric_dtype(dtype):

        if pd.api.types.is_integer_dtype(dtype):
            min_val = data.min()
            max_val = data.max()

            # Age-like range (0-120)
            if 0 <= min_val <= 18 and 50 <= max_val <= 120:
                result['score'] = 0.85
                result['reasons'].append(f"Age-like integer range ({min_val}-{max_val})")

            # Year (1900-2030)
            elif 1900 <= min_val <= 2030 and 1900 <= max_val <= 2030:
                result['score'] = 0.70
                result['reasons'].append(f"Year-like range ({min_val}-{max_val})")

            # ZIP code pattern (5 digits)
            elif 10000 <= min_val <= 99999 and 10000 <= max_val <= 99999:
                result['score'] = 0.90
                result['reasons'].append("5-digit ZIP code pattern")

            # Low-cardinality integer (categorical-like)
            elif n_unique < 20:
                result['score'] = 0.65
                result['reasons'].append(f"Ordinal integer ({n_unique} unique values)")

            # Moderate cardinality integer
            elif 0.05 < cardinality_ratio < 0.30:
                result['score'] = 0.50
                result['reasons'].append(f"Moderate cardinality integer")

            else:
                result['score'] = 0.25
                result['reasons'].append("High-cardinality integer")

        else:
            # Float - usually continuous, less likely QI
            if n_unique < 20:
                result['score'] = 0.45
                result['reasons'].append("Binned continuous (low unique values)")
            else:
                result['score'] = 0.15
                result['reasons'].append("Continuous float (unlikely QI)")

    elif pd.api.types.is_datetime64_any_dtype(dtype):
        result['score'] = 0.70
        result['reasons'].append("Datetime column (often QI)")

    else:
        result['score'] = 0.10
        result['reasons'].append("Unknown data type")

    return result


def _score_uniqueness_contribution(
    data: pd.Series,
    full_data: pd.DataFrame,
    column: str
) -> Dict:
    """
    Score based on how much this column contributes to record uniqueness.

    This is the key insight: a column is a strong QI if adding it to
    the set of QIs significantly increases the uniqueness rate.
    """

    result = {
        'score': 0.0,
        'reasons': [],
        'contribution': 0.0
    }

    # Get potential QI columns (categorical or low-cardinality)
    potential_qis = []
    for col in full_data.columns:
        if col == column:
            continue
        if full_data[col].dtype == 'object':
            potential_qis.append(col)
        elif full_data[col].nunique() / len(full_data) < 0.3:
            potential_qis.append(col)

    # Limit for performance
    potential_qis = potential_qis[:5]

    if len(potential_qis) == 0:
        # This is the only potential QI - assume moderate contribution
        result['score'] = 0.5
        result['contribution'] = 0.0
        result['reasons'].append("Only potential QI - cannot compare")
        return result

    # Calculate uniqueness WITH this column
    try:
        qis_with = potential_qis[:4] + [column]
        groups_with = full_data.groupby(qis_with, observed=True).size()
        uniqueness_with = (groups_with == 1).sum() / len(full_data)
    except Exception:
        result['score'] = 0.4
        result['reasons'].append("Cannot calculate uniqueness with column")
        return result

    # Calculate uniqueness WITHOUT this column
    try:
        qis_without = potential_qis[:4]
        groups_without = full_data.groupby(qis_without, observed=True).size()
        uniqueness_without = (groups_without == 1).sum() / len(full_data)
    except Exception:
        uniqueness_without = 0.0

    # Calculate contribution
    contribution = uniqueness_with - uniqueness_without
    result['contribution'] = contribution

    if contribution > 0.25:
        result['score'] = 1.0
        result['reasons'].append(f"Very high uniqueness contribution (+{contribution:.1%})")
    elif contribution > 0.15:
        result['score'] = 0.85
        result['reasons'].append(f"High uniqueness contribution (+{contribution:.1%})")
    elif contribution > 0.08:
        result['score'] = 0.70
        result['reasons'].append(f"Significant uniqueness contribution (+{contribution:.1%})")
    elif contribution > 0.03:
        result['score'] = 0.50
        result['reasons'].append(f"Moderate uniqueness contribution (+{contribution:.1%})")
    elif contribution > 0.01:
        result['score'] = 0.35
        result['reasons'].append(f"Small uniqueness contribution (+{contribution:.1%})")
    else:
        result['score'] = 0.20
        result['reasons'].append(f"Minimal uniqueness contribution (+{contribution:.1%})")

    return result


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
            if not exclude_identifiers or column_types.get(col) != 'identifier':
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
            if not exclude_identifiers or column_types.get(col) != 'identifier':
                if data[col].nunique() <= max_cardinality:
                    categorical.append(col)
    
    return categorical


# =============================================================================
# DATA TYPE DETECTION AND ANALYSIS
# =============================================================================

def auto_detect_sensitive_columns(
    data: pd.DataFrame,
    check_patterns: bool = True
) -> Dict[str, str]:
    """
    Automatically detect columns containing DIRECT IDENTIFIERS that should be EXCLUDED.

    DEPRECATED: Use auto_detect_direct_identifiers() from src.detection instead.
    This function is kept for backward compatibility.

    Direct identifiers are columns that can directly identify individuals without
    linking to external data. These should be REMOVED before applying SDC methods.

    NOTE: This is different from "sensitive attributes" (income, diagnosis, health_score)
    which are the values you want to PROTECT via SDC methods like k-anonymity.

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
    from .config import DIRECT_IDENTIFIER_KEYWORDS, DIRECT_IDENTIFIER_PATTERNS

    # Use config-defined keywords and patterns
    identifier_keywords = DIRECT_IDENTIFIER_KEYWORDS
    value_patterns = DIRECT_IDENTIFIER_PATTERNS

    detected_identifiers = {}

    for col in data.columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        col_words = col_lower.split('_')

        # Check column name against keywords
        # Use word boundaries to avoid false positives like 'tin' in 'rating'
        for identifier_type, keywords in identifier_keywords.items():
            if any(kw in col_words or kw == col_lower for kw in keywords):
                detected_identifiers[col] = identifier_type
                break

        # If not detected by name and check_patterns is True, check values
        if col not in detected_identifiers and check_patterns:
            # Only check string columns
            if data[col].dtype == 'object':
                # Sample up to 100 non-null values for pattern checking
                sample = data[col].dropna().head(100).astype(str)

                if len(sample) > 0:
                    for pattern_name, pattern in value_patterns.items():
                        try:
                            # Check if majority of values match the pattern
                            matches = sample.str.match(pattern, na=False).sum()
                            if matches / len(sample) > 0.5:
                                detected_identifiers[col] = pattern_name
                                break
                        except Exception:
                            pass

    return detected_identifiers


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

    # Check if all columns are numeric
    all_numeric = data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]

    if all_numeric:
        # Check for named string index (common in frequency tables)
        has_named_index = (
            not isinstance(data.index, pd.RangeIndex) and
            data.index.dtype == 'object' and
            len(data.index) > 0
        )

        # Exclude Total row/column for max calculation (they inflate the values)
        data_no_totals = data.copy()
        if 'Total' in data_no_totals.index:
            data_no_totals = data_no_totals.drop('Total', axis=0)
        if 'Total' in data_no_totals.columns:
            data_no_totals = data_no_totals.drop('Total', axis=1)

        # Check if values are integers (typical for frequency tables)
        all_integers = (data_no_totals % 1 == 0).all().all() if len(data_no_totals) > 0 else False

        if all_integers:
            max_val = data_no_totals.max().max() if len(data_no_totals) > 0 else 0
            # More lenient threshold for tables, especially with named index
            if max_val < 10000 or (has_named_index and max_val < 50000):
                return 'tabular'

    return 'microdata'


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

    # Keywords that suggest identifier columns
    id_keywords = ['id', 'uuid', 'guid', 'ssn', 'nin', 'passport', 'license',
                   'serial', 'code', 'key', 'index', 'number', 'num', 'no']
    # Keywords that suggest the column is NOT an identifier despite high uniqueness
    value_keywords = ['income', 'salary', 'wage', 'price', 'cost', 'amount',
                      'score', 'rating', 'value', 'total', 'sum', 'count',
                      'weight', 'height', 'age', 'year', 'date', 'time',
                      'balance', 'revenue', 'profit', 'loss', 'tax', 'fee']

    column_types = {}

    for col in data.columns:
        col_lower = col.lower()

        # First, check column name for identifier keywords
        is_id_by_name = any(kw in col_lower for kw in id_keywords)
        is_value_by_name = any(kw in col_lower for kw in value_keywords)

        # Check if numeric
        if pd.api.types.is_numeric_dtype(data[col]):
            n_unique = data[col].nunique()
            uniqueness_ratio = n_unique / len(data)

            # Check if binary (only 2 unique values)
            if n_unique <= 2:
                column_types[col] = 'binary'

            # If column name suggests identifier
            elif is_id_by_name and not is_value_by_name:
                column_types[col] = 'identifier'

            # If column name suggests a value/measure, it's continuous
            elif is_value_by_name:
                column_types[col] = 'continuous'

            # Check if looks like sequential ID (starts near 0 or 1, increments by ~1)
            elif _is_sequential_id(data[col]):
                column_types[col] = 'identifier'

            # Check if discrete (integers with few unique values)
            elif n_unique < 20 and (data[col] % 1 == 0).all():
                column_types[col] = 'categorical'

            # High cardinality numeric = continuous (not identifier)
            else:
                column_types[col] = 'continuous'
        else:
            # String/object type
            uniqueness_ratio = data[col].nunique() / len(data)

            # If column name suggests identifier
            if is_id_by_name and not is_value_by_name:
                column_types[col] = 'identifier'

            # Check if looks like ID (unique or nearly unique strings)
            elif uniqueness_ratio > 0.95:
                column_types[col] = 'identifier'

            else:
                column_types[col] = 'categorical'

    return column_types


def _is_sequential_id(series: pd.Series) -> bool:
    """
    Check if a numeric series looks like a sequential identifier.

    Sequential IDs typically:
    - Are integers
    - Start near 0 or 1
    - Have differences of approximately 1 between sorted values

    Parameters:
    -----------
    series : pd.Series
        Numeric series to check

    Returns:
    --------
    bool : True if series looks like a sequential ID
    """
    # Must be integers
    if not (series % 1 == 0).all():
        return False

    # Check if starts near 0 or 1
    min_val = series.min()
    if min_val > 10:  # Sequential IDs usually start low
        return False

    # Check if differences are approximately 1
    sorted_vals = series.sort_values().values
    if len(sorted_vals) < 2:
        return False

    diffs = np.diff(sorted_vals)

    # Most differences should be 1 (allowing for some gaps)
    # A sequential ID has median diff of 1
    median_diff = np.median(diffs)

    return median_diff <= 1.5 and min_val >= 0


def get_data_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive summary of dataset characteristics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    
    Returns:
    --------
    dict : Summary statistics
    """
    
    column_types = identify_column_types(data)
    
    summary = {
        'n_records': len(data),
        'n_columns': len(data.columns),
        'data_type': detect_data_type(data),
        'column_types': column_types,
        'n_continuous': sum(1 for t in column_types.values() if t == 'continuous'),
        'n_categorical': sum(1 for t in column_types.values() if t == 'categorical'),
        'n_identifiers': sum(1 for t in column_types.values() if t == 'identifier'),
        'missing_values': data.isna().sum().to_dict(),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return summary


# =============================================================================
# K-ANONYMITY CHECKING
# =============================================================================

def check_kanonymity(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int
) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
    """
    Check if dataset satisfies k-anonymity.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to check
    quasi_identifiers : list of str
        Column names that are quasi-identifiers
    k : int
        Required minimum group size
    
    Returns:
    --------
    is_kanonymous : bool
        True if dataset satisfies k-anonymity
    group_sizes : pd.DataFrame
        DataFrame showing size of each equivalence class (QI combination)
    violations : pd.DataFrame
        Equivalence classes that violate k-anonymity (size < k)
    """
    
    # Group by quasi-identifiers and count group sizes
    # Always use '_group_size_' to avoid conflicts with any column named 'count'
    count_col = '_group_size_'
    group_sizes = data.groupby(quasi_identifiers, dropna=False).size().reset_index(name=count_col)

    # Check if all groups have at least k members
    is_kanonymous = (group_sizes[count_col] >= k).all()

    # Find violations
    violations = group_sizes[group_sizes[count_col] < k]

    # Rename to 'count' for consistent output (only if no conflict)
    if 'count' not in quasi_identifiers:
        group_sizes = group_sizes.rename(columns={count_col: 'count'})
        violations = violations.rename(columns={count_col: 'count'})

    return is_kanonymous, group_sizes, violations


def calculate_uniqueness_rate(data: pd.DataFrame, columns: List[str]) -> float:
    """
    Calculate the proportion of records with unique combinations of specified columns.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    columns : list of str
        Columns to check for uniqueness
    
    Returns:
    --------
    float : Proportion of unique records (0.0 to 1.0)
    """
    
    if not columns:
        return 0.0
    
    # Count occurrences of each combination
    combo_counts = data.groupby(columns).size()
    
    # Count how many combinations appear exactly once
    unique_combos = (combo_counts == 1).sum()
    
    # Calculate rate
    uniqueness_rate = unique_combos / len(data)

    return uniqueness_rate


def calculate_reid(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    quantiles: List[float] = None
) -> Dict:
    """
    Calculate Re-Identification Risk (ReID) - risk distribution across records.

    ReID provides a more nuanced view of re-identification risk by showing
    what percentage of records fall below various risk thresholds.

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
        - risk_distribution: Full risk series (optional)

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
    valid_data = data.dropna(subset=quasi_identifiers)

    if len(valid_data) == 0:
        # All records have suppressed QIs - return zero risk (fully protected)
        return {
            'reid_50': 0.0, 'reid_90': 0.0, 'reid_95': 0.0, 'reid_99': 0.0,
            'max_risk': 0.0, 'mean_risk': 0.0, 'high_risk_count': 0,
            'high_risk_rate': 0.0, 'suppressed_records': len(data)
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

    return result


def assess_risk_with_reid(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    reid_thresholds: Dict[str, float] = None
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
    combo_counts = data.groupby(columns).size().reset_index(name=count_col)

    # Filter for rare combinations
    rare = combo_counts[combo_counts[count_col] < threshold]

    # Rename back to 'count' for consistent output
    if count_col != 'count':
        rare = rare.rename(columns={count_col: 'count'})

    return rare.sort_values('count')


# =============================================================================
# DATA TRANSFORMATION AND AGGREGATION
# =============================================================================

def aggregate_to_table(
    microdata: pd.DataFrame,
    dimensions: List[str],
    aggfunc: str = 'count',
    value_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert microdata to aggregated frequency table.
    
    Parameters:
    -----------
    microdata : pd.DataFrame
        Input microdata
    dimensions : list of str
        Column names to use as dimensions of the table
    aggfunc : str, default='count'
        Aggregation function: 'count', 'sum', 'mean', 'median', etc.
    value_column : str, optional
        Column to aggregate (required if aggfunc is not 'count')
    
    Returns:
    --------
    pd.DataFrame : Aggregated table
    """
    
    if not dimensions:
        raise ValueError("dimensions cannot be empty")
    
    if aggfunc == 'count':
        # Simple frequency count
        if len(dimensions) == 1:
            table = microdata.groupby(dimensions[0]).size().to_frame('count')
        else:
            table = pd.crosstab(
                index=[microdata[dim] for dim in dimensions[:-1]],
                columns=microdata[dimensions[-1]],
                margins=False
            )
    else:
        # Aggregate with specific function
        if value_column is None:
            raise ValueError(f"value_column must be specified when aggfunc is '{aggfunc}'")
        
        table = microdata.pivot_table(
            values=value_column,
            index=dimensions[:-1] if len(dimensions) > 1 else dimensions[0],
            columns=dimensions[-1] if len(dimensions) > 1 else None,
            aggfunc=aggfunc,
            fill_value=0
        )
    
    return table


def check_small_cells(
    table: pd.DataFrame,
    threshold: int
) -> pd.DataFrame:
    """
    Identify cells in a table that are below a threshold value.
    
    Parameters:
    -----------
    table : pd.DataFrame
        Frequency table to check
    threshold : int
        Minimum acceptable cell value
    
    Returns:
    --------
    pd.DataFrame : Boolean DataFrame with True for cells below threshold
    """
    
    return table < threshold


# =============================================================================
# GENERALIZATION FUNCTIONS
# =============================================================================

def generalize_numeric(
    series: pd.Series,
    bin_size: int = 5,
    return_intervals: bool = True
) -> pd.Series:
    """
    Generalize numeric values into ranges/bins.
    
    Parameters:
    -----------
    series : pd.Series
        Numeric column to generalize
    bin_size : int, default=5
        Size of each bin/range
    return_intervals : bool, default=True
        If True, return as string intervals (e.g., "20-24")
        If False, return as interval objects
    
    Returns:
    --------
    pd.Series : Generalized values
    """
    
    # Calculate bin edges
    min_val = series.min()
    max_val = series.max()
    
    # Round down to nearest bin_size
    min_bin = (min_val // bin_size) * bin_size
    max_bin = ((max_val // bin_size) + 1) * bin_size
    
    bins = range(int(min_bin), int(max_bin) + bin_size, bin_size)
    
    if return_intervals:
        labels = [f"{b}-{b+bin_size-1}" for b in bins[:-1]]
        generalized = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=False)
        return generalized.astype(str)
    else:
        generalized = pd.cut(series, bins=bins, include_lowest=True, right=False)
        return generalized


def generalize_categorical(
    series: pd.Series,
    hierarchy: Dict[str, str]
) -> pd.Series:
    """
    Generalize categorical values using a hierarchy mapping.
    
    Parameters:
    -----------
    series : pd.Series
        Categorical column to generalize
    hierarchy : dict
        Mapping from specific values to general values
        Example: {'Engineer': 'Technical', 'Developer': 'Technical', ...}
    
    Returns:
    --------
    pd.Series : Generalized values
    """
    
    return series.map(hierarchy).fillna(series)


def generalize_string_prefix(
    series: pd.Series,
    prefix_length: int = 3,
    suffix: str = '*'
) -> pd.Series:
    """
    Generalize string values by taking prefix and adding wildcard.
    
    Parameters:
    -----------
    series : pd.Series
        String column to generalize
    prefix_length : int, default=3
        Length of prefix to keep
    suffix : str, default='*'
        Wildcard suffix to add
    
    Returns:
    --------
    pd.Series : Generalized values (e.g., "12345" -> "123*")
    """
    
    return series.astype(str).str[:prefix_length] + suffix


# =============================================================================
# INFORMATION LOSS METRICS
# =============================================================================

def calculate_information_loss(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate various information loss metrics.
    
    Parameters:
    -----------
    original : pd.DataFrame
        Original dataset
    protected : pd.DataFrame
        Anonymized dataset
    columns : list of str, optional
        Columns to analyze (if None, uses all common columns)
    
    Returns:
    --------
    dict : Information loss metrics
    """
    
    if columns is None:
        columns = [col for col in original.columns if col in protected.columns]
    
    metrics = {}
    
    for col in columns:
        if col not in original.columns or col not in protected.columns:
            continue
        
        # For numeric columns
        if pd.api.types.is_numeric_dtype(original[col]):
            # Mean absolute error
            mae = np.abs(original[col] - protected[col]).mean()
            metrics[f'{col}_mae'] = float(mae)
            
            # Relative error
            if original[col].std() > 0:
                rel_error = mae / original[col].std()
                metrics[f'{col}_relative_error'] = float(rel_error)
        
        # For categorical columns
        else:
            # Proportion of changed values
            changed = (original[col] != protected[col]).sum()
            metrics[f'{col}_changed_rate'] = float(changed / len(original))
    
    # Overall metrics
    metrics['total_cells_changed'] = sum(
        (original[col] != protected[col]).sum() 
        for col in columns if col in original.columns and col in protected.columns
    )
    metrics['overall_change_rate'] = metrics['total_cells_changed'] / (len(original) * len(columns))
    
    return metrics


def calculate_disclosure_risk(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int = 3
) -> Dict[str, float]:
    """
    Calculate disclosure risk metrics.

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
    dict : Risk metrics
    """

    # Calculate uniqueness
    uniqueness = calculate_uniqueness_rate(data, quasi_identifiers)

    # Check k-anonymity violations
    is_kanon, group_sizes, violations = check_kanonymity(data, quasi_identifiers, k)

    # Determine which column has the group sizes (may be 'count' or '_group_size_')
    size_col = '_group_size_' if '_group_size_' in group_sizes.columns else 'count'

    # Calculate proportion of records at risk
    if len(violations) > 0:
        records_at_risk = violations[size_col].sum()
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
        'min_group_size': int(group_sizes[size_col].min()),
        'avg_group_size': float(group_sizes[size_col].mean())
    }

    return metrics


# =============================================================================
# DATA ANALYSIS FOR METHOD SELECTION
# =============================================================================

def analyze_data(
    data: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive analysis of data for SDC method selection.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data to analyze
    quasi_identifiers : list, optional
        Quasi-identifier columns. Auto-detected if not provided.
    verbose : bool
        Print analysis results

    Returns:
    --------
    dict : Analysis results including data type, risk metrics, and characteristics
    """
    analysis = {}

    # 1. Basic info
    analysis['n_records'] = len(data)
    analysis['n_columns'] = len(data.columns)
    analysis['columns'] = list(data.columns)

    # 2. Data type detection
    data_type = detect_data_type(data)
    analysis['data_type'] = data_type

    if verbose:
        print("\n" + "=" * 60)
        print("  DATA ANALYSIS REPORT")
        print("=" * 60)
        print(f"\nRecords: {analysis['n_records']}")
        print(f"Columns: {analysis['n_columns']}")
        print(f"Data type: {data_type}")

    # 3. Sensitive columns detection
    sensitive = auto_detect_sensitive_columns(data)
    analysis['sensitive_columns'] = sensitive

    if verbose and sensitive:
        print(f"\n[!] WARNING: Potential sensitive columns detected:")
        for col, reason in sensitive.items():
            print(f"    - {col}: {reason}")

    # 4. Variable type detection
    continuous = auto_detect_continuous_variables(data)
    categorical = auto_detect_categorical_variables(data)
    analysis['continuous_variables'] = continuous
    analysis['categorical_variables'] = categorical

    if verbose:
        print(f"\nContinuous variables: {continuous}")
        print(f"Categorical variables: {categorical}")

    # 5. For microdata: quasi-identifiers and risk analysis
    if data_type == 'microdata':
        # Auto-detect QIs if not provided
        if quasi_identifiers is None:
            quasi_identifiers = auto_detect_quasi_identifiers(data)

        analysis['quasi_identifiers'] = quasi_identifiers

        if verbose:
            print(f"\nQuasi-identifiers: {quasi_identifiers}")

        # Calculate disclosure risk
        if quasi_identifiers:
            risk = calculate_disclosure_risk(data, quasi_identifiers, k=3)
            analysis['disclosure_risk'] = risk

            # Determine risk level
            risk_level = _assess_risk_level(risk)
            analysis['risk_level'] = risk_level

            if verbose:
                print(f"\n--- Disclosure Risk Analysis ---")
                print(f"Uniqueness rate: {risk['uniqueness_rate']:.1%}")
                print(f"Records at risk (k<3): {risk['records_at_risk']} ({risk['risk_rate']:.1%})")
                print(f"Min equivalence class size: {risk['min_group_size']}")
                print(f"Avg equivalence class size: {risk['avg_group_size']:.1f}")
                print(f"\nRisk Level: {risk_level.upper()}")

        # Cardinality analysis
        cardinality = {}
        for col in quasi_identifiers:
            cardinality[col] = data[col].nunique()
        analysis['qi_cardinality'] = cardinality

        high_cardinality = [col for col, n in cardinality.items() if n > 20]
        analysis['high_cardinality_columns'] = high_cardinality

        if verbose and high_cardinality:
            print(f"\nHigh cardinality QIs (>20 values): {high_cardinality}")

    # 6. For tabular data: sparsity and small cell analysis
    elif data_type == 'tabular':
        # Analyze table characteristics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            table_values = data[numeric_cols]

            total_cells = table_values.size
            zero_cells = (table_values == 0).sum().sum()
            small_cells = ((table_values > 0) & (table_values < 3)).sum().sum()

            analysis['total_cells'] = int(total_cells)
            analysis['zero_cells'] = int(zero_cells)
            analysis['small_cells_n3'] = int(small_cells)
            analysis['sparsity'] = float(zero_cells / total_cells) if total_cells > 0 else 0
            analysis['sensitivity_rate'] = float(small_cells / total_cells) if total_cells > 0 else 0

            # Risk level for tabular
            if analysis['sensitivity_rate'] > 0.3:
                analysis['risk_level'] = 'high'
            elif analysis['sensitivity_rate'] > 0.1:
                analysis['risk_level'] = 'medium'
            else:
                analysis['risk_level'] = 'low'

            if verbose:
                print(f"\n--- Table Analysis ---")
                print(f"Total cells: {total_cells}")
                print(f"Zero cells: {zero_cells} ({analysis['sparsity']:.1%})")
                print(f"Small cells (0 < value < 3): {small_cells} ({analysis['sensitivity_rate']:.1%})")
                print(f"\nRisk Level: {analysis['risk_level'].upper()}")

    return analysis


def _assess_risk_level(risk: Dict) -> str:
    """Assess overall risk level from risk metrics."""
    score = 0

    # Uniqueness rate scoring
    if risk['uniqueness_rate'] > 0.5:
        score += 3
    elif risk['uniqueness_rate'] > 0.2:
        score += 2
    elif risk['uniqueness_rate'] > 0.05:
        score += 1

    # Risk rate scoring
    if risk['risk_rate'] > 0.5:
        score += 3
    elif risk['risk_rate'] > 0.2:
        score += 2
    elif risk['risk_rate'] > 0.05:
        score += 1

    # Min group size scoring
    if risk['min_group_size'] == 1:
        score += 2
    elif risk['min_group_size'] < 3:
        score += 1

    # Determine level
    if score >= 5:
        return 'high'
    elif score >= 3:
        return 'medium'
    else:
        return 'low'


# =============================================================================
# METHOD INFORMATION
# =============================================================================

# Method characteristics for active SDC methods (microdata-only)
METHOD_INFO = {
    'kANON': {
        'data_type': 'microdata',
        'approach': 'generalization',
        'preserves': ['record_count'],
        'description': 'Achieves k-anonymity through generalization/suppression',
        'best_for': 'Formal privacy guarantee (k-anonymity)',
        'when_to_use': [
            'Need formal k-anonymity guarantee',
            'High uniqueness rate',
            'Publishing individual records'
        ]
    },
    'PRAM': {
        'data_type': 'microdata',
        'approach': 'perturbation',
        'preserves': ['marginal_distributions'],
        'description': 'Randomly perturbs categorical values',
        'best_for': 'Protecting categorical variables while preserving distributions',
        'when_to_use': [
            'Categorical data',
            'Need to preserve marginal distributions',
            'Acceptable to have some false values'
        ]
    },
    'NOISE': {
        'data_type': 'microdata',
        'approach': 'perturbation',
        'preserves': ['mean', 'correlation'],
        'description': 'Adds random noise to continuous variables',
        'best_for': 'Protecting continuous variables while preserving statistics',
        'when_to_use': [
            'Continuous/numeric data',
            'Need to preserve means and correlations',
            'Differential privacy requirements'
        ]
    },
    'LOCSUPR': {
        'data_type': 'microdata',
        'approach': 'suppression',
        'preserves': ['record_count', 'most_values'],
        'description': 'Suppresses specific cell values to achieve k-anonymity',
        'best_for': 'Achieving k-anonymity with minimal information loss',
        'when_to_use': [
            'Need k-anonymity with minimal changes',
            'Prefer suppression over generalization',
            'Target specific high-risk values'
        ]
    }
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_quasi_identifiers(
    data: pd.DataFrame,
    quasi_identifiers: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that quasi-identifiers exist in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    quasi_identifiers : list of str
        Column names to validate
    
    Returns:
    --------
    is_valid : bool
        True if all QIs exist
    missing : list of str
        List of missing column names
    """
    
    missing = [qi for qi in quasi_identifiers if qi not in data.columns]
    is_valid = len(missing) == 0
    
    return is_valid, missing


def validate_numeric_parameter(
    value: Any,
    param_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    must_be_int: bool = False
) -> None:
    """
    Validate numeric parameters.
    
    Parameters:
    -----------
    value : any
        Value to validate
    param_name : str
        Name of parameter (for error messages)
    min_value : float, optional
        Minimum allowed value
    max_value : float, optional
        Maximum allowed value
    must_be_int : bool, default=False
        Whether value must be an integer
    
    Raises:
    -------
    ValueError : If validation fails
    """
    
    if not isinstance(value, (int, float)):
        raise ValueError(f"{param_name} must be numeric, got {type(value)}")
    
    if must_be_int and not isinstance(value, int):
        raise ValueError(f"{param_name} must be an integer")
    
    if min_value is not None and value < min_value:
        raise ValueError(f"{param_name} must be >= {min_value}, got {value}")
    
    if max_value is not None and value > max_value:
        raise ValueError(f"{param_name} must be <= {max_value}, got {value}")


# =============================================================================
# Column-type coercion (Configure table as source of truth)
# =============================================================================

_NUMERIC_TYPE_KEYWORDS = {'numeric', 'continuous', 'integer', 'float', 'coded'}
# Deduplicate coercion log messages across repeated calls
_coerce_logged: set = set()


def coerce_columns_by_types(
    data: pd.DataFrame,
    column_types: Optional[Dict[str, str]] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert object-dtype columns to numeric based on Configure table types.

    The Configure tab classifies each column (e.g. "Char (numeric) — Continuous").
    When a column's dtype is ``object`` but column_types says it's numeric, this
    function converts it with ``pd.to_numeric(..., errors='coerce')``.

    Parameters
    ----------
    data : DataFrame
        Data to coerce **in-place** (caller should pass a copy if needed).
    column_types : dict, optional
        ``{col_name: type_label}`` from ``SDCConfigure.get_column_types()``.
    columns : list of str, optional
        Subset of columns to consider.  If *None*, checks all columns in *data*.

    Returns
    -------
    DataFrame — same object, potentially with converted columns.
    """
    import logging
    _log = logging.getLogger(__name__)

    if not column_types:
        _log.debug("[coerce] No column_types provided — skipping coercion")
        return data
    cols = columns if columns is not None else list(data.columns)
    for col in cols:
        if col not in data.columns:
            continue
        if data[col].dtype != object:
            continue  # already typed correctly
        ct_label = column_types.get(col, '').lower()
        if not ct_label:
            continue
        if any(kw in ct_label for kw in _NUMERIC_TYPE_KEYWORDS):
            before_dtype = data[col].dtype
            n_total = len(data[col])

            # Count non-null, non-empty original values.
            # Columns with many NaN/empty cells should NOT penalise the
            # conversion rate — only the actually-filled cells matter.
            _orig_str = data[col].astype(str).str.strip()
            _has_value = data[col].notna() & (_orig_str != '') & (_orig_str != 'nan')
            n_non_null = int(_has_value.sum())

            # Try direct conversion first
            coerced = pd.to_numeric(data[col], errors='coerce')
            n_valid = int(coerced.notna().sum())
            pct = n_valid / n_non_null * 100 if n_non_null > 0 else 0

            # If direct fails, try European/Greek format (dot=thousands, comma=decimal)
            if pct < 50:
                try:
                    cleaned = (data[col].astype(str)
                               .str.strip()
                               .str.replace('.', '', regex=False)
                               .str.replace(',', '.', regex=False))
                    coerced_eu = pd.to_numeric(cleaned, errors='coerce')
                    n_valid_eu = int(coerced_eu.notna().sum())
                    pct_eu = n_valid_eu / n_non_null * 100 if n_non_null > 0 else 0
                    if pct_eu > pct:
                        coerced = coerced_eu
                        n_valid = n_valid_eu
                        pct = pct_eu
                        if pct > 50:
                            _log.info(
                                "[coerce] '%s' converted via EU number format "
                                "(dot=thousands, comma=decimal)", col)
                except Exception:
                    pass

            if pct > 50:
                data[col] = coerced
                if col not in _coerce_logged:
                    _log.info(
                        "[coerce] Converted '%s' from %s to numeric "
                        "(%.0f%% valid, %d/%d non-null rows)",
                        col, before_dtype, pct, n_valid, n_non_null)
                    _coerce_logged.add(col)
            else:
                if col not in _coerce_logged:
                    _log.warning(
                        "[coerce] '%s' classified as numeric (%s) but only "
                        "%.0f%% of non-null values convertible — skipping",
                        col, ct_label, pct)
                    _coerce_logged.add(col)
    return data


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SDC Utilities - Example Usage")
    print("=" * 60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'id': range(1, 101),
        'age': np.random.randint(18, 80, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'income': np.random.randint(20000, 100000, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })
    
    print("\n--- Data Summary ---")
    summary = get_data_summary(sample_data)
    for key, value in summary.items():
        if key != 'missing_values' and key != 'column_types':
            print(f"{key}: {value}")
    
    print("\n--- Column Types ---")
    for col, col_type in summary['column_types'].items():
        print(f"  {col}: {col_type}")
    
    print("\n--- K-Anonymity Check (k=3) ---")
    qis = ['age', 'gender', 'region']
    is_kanon, groups, violations = check_kanonymity(sample_data, qis, k=3)
    print(f"Is k-anonymous: {is_kanon}")
    print(f"Number of equivalence classes: {len(groups)}")
    print(f"Number of violations: {len(violations)}")
    
    print("\n--- Uniqueness Analysis ---")
    uniqueness = calculate_uniqueness_rate(sample_data, qis)
    print(f"Uniqueness rate: {uniqueness:.2%}")
    
    print("\n--- Disclosure Risk ---")
    risk = calculate_disclosure_risk(sample_data, qis, k=3)
    for key, value in risk.items():
        print(f"  {key}: {value}")
    
    print("\n--- Generalization Example ---")
    print("Original ages (first 10):")
    print(sample_data['age'].head(10).tolist())
    
    generalized_age = generalize_numeric(sample_data['age'], bin_size=10)
    print("\nGeneralized ages (first 10):")
    print(generalized_age.head(10).tolist())
    
    print("\n--- Aggregation Example ---")
    table = aggregate_to_table(sample_data, ['gender', 'region'])
    print("\nFrequency table (gender × region):")
    print(table)
    
    print("\n--- Small Cell Detection ---")
    small_cells = check_small_cells(table, threshold=5)
    print(f"\nCells below threshold 5:")
    print(small_cells)
