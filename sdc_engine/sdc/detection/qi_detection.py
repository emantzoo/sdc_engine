"""
Quasi-Identifier Detection
===========================

Multi-tiered QI detection with confidence scoring.

Scoring Dimensions:
1. Name-based (40%) - Keyword matching in column names
2. Type-based (25%) - Data type, cardinality, value patterns
3. Uniqueness contribution (35%) - How much column increases re-identification risk
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

_log = logging.getLogger(__name__)

from .column_types import identify_column_types
from ..config import QI_KEYWORDS, QI_KEYWORDS_GR, QI_SCORING_WEIGHTS, QI_CONFIDENCE_TIERS


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
        Maximum number of QIs to return

    Returns:
    --------
    list of str : Detected quasi-identifier column names
    """
    column_types = identify_column_types(data)

    # Filter out identifiers if requested
    candidates = {col: ctype for col, ctype in column_types.items()
                  if not (exclude_identifiers and ctype == 'identifier')}

    # Score candidates
    scored = []
    for col, ctype in candidates.items():
        score = 0

        # Prefer categorical with moderate cardinality
        if ctype in ['categorical', 'binary']:
            n_unique = data[col].nunique()
            if 2 <= n_unique <= 20:
                score += 2
            elif n_unique <= 50:
                score += 1

        # Check for QI-like column names using config keywords
        col_lower = col.lower()
        qi_keyword_list = list(QI_KEYWORDS.get('definite_qis', {}).keys())
        qi_keyword_list.extend(QI_KEYWORDS.get('probable_qis', {}).keys())

        for keyword in qi_keyword_list:
            if keyword in col_lower:
                score += 3
                break

        scored.append((col, score))

    # Sort by score and return top candidates
    scored.sort(key=lambda x: x[1], reverse=True)
    qis = [col for col, score in scored[:max_qis] if score > 0]

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

    # Dimension 1: Name-Based Detection
    name_result = _score_column_name(column)
    score['signals']['name'] = name_result

    # Early exit for direct identifiers
    if name_result.get('is_identifier'):
        score['confidence'] = 0.0
        score['tier'] = 'NON_QI'
        score['reasons'] = name_result.get('reasons', [])
        return score

    # Dimension 2: Type-Based Detection
    type_result = _score_data_type_for_qi(data, column, column_type)
    score['signals']['type'] = type_result

    # Early exit for sequential IDs
    if type_result.get('is_sequential_id'):
        score['confidence'] = 0.0
        score['tier'] = 'NON_QI'
        score['reasons'] = ['Sequential ID detected']
        return score

    # Dimension 3: Uniqueness Contribution
    uniqueness_result = _score_uniqueness_contribution(data, full_data, column)
    score['signals']['uniqueness'] = uniqueness_result

    # Combine Scores with Weights
    weights = QI_SCORING_WEIGHTS

    total_score = (
        score['signals']['name']['score'] * weights['name_based'] +
        score['signals']['type']['score'] * weights['type_based'] +
        score['signals']['uniqueness']['score'] * weights['uniqueness']
    )

    # Apply boosts
    if name_result.get('definite_qi'):
        total_score = min(total_score * 1.2, 1.0)

    # Apply penalty for high-cardinality continuous variables
    # These make poor QIs because they cause massive generalization/suppression
    if column_type == 'continuous':
        n_unique = data.nunique()
        cardinality_ratio = n_unique / len(full_data) if len(full_data) > 0 else 1
        if cardinality_ratio > 0.3:  # High cardinality continuous
            total_score *= 0.4  # Heavy penalty
            score['reasons'].append(f"Penalty: high-cardinality continuous ({n_unique} unique, {cardinality_ratio:.0%})")
        elif cardinality_ratio > 0.1:
            total_score *= 0.6  # Moderate penalty
            score['reasons'].append(f"Penalty: moderate-cardinality continuous ({n_unique} unique)")

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
    """Score based on column name keywords."""
    col_lower = column.lower()
    col_normalized = col_lower.replace('_', ' ').replace('-', ' ')

    result = {
        'score': 0.0,
        'reasons': [],
        'is_identifier': False,
        'definite_qi': False
    }

    # Direct Identifiers (NOT quasi-identifiers) - score 0.0
    direct_identifiers = QI_KEYWORDS['direct_identifiers']

    for identifier in direct_identifiers:
        if identifier in col_lower or identifier in col_normalized:
            if identifier == 'id':
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

    # Definite QI Keywords - score 0.85-1.0
    definite_qis = QI_KEYWORDS['definite_qis']

    for keyword, keyword_score in definite_qis.items():
        if keyword in col_lower or keyword in col_normalized:
            result['score'] = keyword_score
            result['definite_qi'] = True
            result['reasons'].append(f"QI keyword: {keyword}")
            return result

    # Probable QI Keywords - score 0.5-0.75
    probable_qis = QI_KEYWORDS['probable_qis']

    for keyword, keyword_score in probable_qis.items():
        if keyword in col_lower or keyword in col_normalized:
            result['score'] = keyword_score
            result['reasons'].append(f"Probable QI keyword: {keyword}")
            return result

    # ── Greek QI keywords ──
    # Check Greek direct identifiers
    for identifier in QI_KEYWORDS_GR.get('direct_identifiers_gr', []):
        if identifier in col_lower or identifier in col_normalized:
            result['score'] = 0.0
            result['is_identifier'] = True
            result['reasons'].append(f"Greek identifier: {identifier}")
            return result

    # Check Greek definite QIs
    for keyword, keyword_score in QI_KEYWORDS_GR.get('definite_qis_gr', {}).items():
        if keyword in col_lower or keyword in col_normalized:
            result['score'] = keyword_score
            result['definite_qi'] = True
            result['reasons'].append(f"Greek QI keyword: {keyword}")
            return result

    # Check Greek probable QIs
    for keyword, keyword_score in QI_KEYWORDS_GR.get('probable_qis_gr', {}).items():
        if keyword in col_lower or keyword in col_normalized:
            result['score'] = keyword_score
            result['reasons'].append(f"Greek probable QI: {keyword}")
            return result

    # No keyword match
    result['score'] = 0.0
    result['reasons'].append("No QI keywords in name")
    return result


def _score_data_type_for_qi(data: pd.Series, column: str, column_type: str) -> Dict:
    """
    Score based on data type and cardinality characteristics.

    Good QIs are:
    - Low to moderate cardinality categorical (e.g., gender, region, department)
    - Age-like integers with limited range
    - ZIP codes, year values

    Poor QIs are:
    - High-cardinality continuous variables (salary, measurements)
    - Near-unique identifiers
    - Sequential IDs
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

    # Check for QI override patterns
    qi_override_patterns = QI_KEYWORDS.get('qi_override_patterns', [])
    is_qi_override = any(pattern in col_lower for pattern in qi_override_patterns)

    # Sequential ID Detection
    is_seq_id = False
    if column_type == 'identifier' and not is_qi_override:
        is_seq_id = True
    elif pd.api.types.is_numeric_dtype(data.dtype):
        is_seq_id = _is_sequential_id(data)

    if is_seq_id:
        result['score'] = 0.0
        result['is_sequential_id'] = True
        result['reasons'].append("Sequential ID detected")
        return result

    # Type-based scoring - prioritize data characteristics over names
    dtype = data.dtype

    if dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
        # Categorical variables - ideal QIs when low cardinality
        if n_unique <= 10:
            # Very low cardinality - excellent QI candidate
            result['score'] = 0.90
            result['reasons'].append(f"Low-cardinality categorical ({n_unique} unique) - ideal QI")
        elif n_unique <= 30:
            result['score'] = 0.80
            result['reasons'].append(f"Moderate-cardinality categorical ({n_unique} unique)")
        elif cardinality_ratio < 0.10:
            result['score'] = 0.70
            result['reasons'].append(f"Categorical ({n_unique} unique, {cardinality_ratio:.1%} unique ratio)")
        elif cardinality_ratio < 0.50:
            result['score'] = 0.50
            result['reasons'].append(f"High-cardinality categorical ({n_unique} unique)")
        else:
            result['score'] = 0.15
            result['reasons'].append("Near-unique categorical - likely identifier")

    elif pd.api.types.is_numeric_dtype(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            min_val = data.min()
            max_val = data.max()

            # Check for specific patterns first
            if 10000 <= min_val <= 99999 and 10000 <= max_val <= 99999:
                result['score'] = 0.90
                result['reasons'].append("5-digit ZIP code pattern")
            elif 0 <= min_val <= 18 and 50 <= max_val <= 120:
                result['score'] = 0.85
                result['reasons'].append(f"Age-like integer range ({min_val}-{max_val})")
            elif 1900 <= min_val <= 2030 and 1900 <= max_val <= 2030:
                result['score'] = 0.70
                result['reasons'].append(f"Year-like range ({min_val}-{max_val})")
            elif n_unique <= 10:
                # Very low cardinality integer - good QI (like rating 1-5)
                result['score'] = 0.75
                result['reasons'].append(f"Low-cardinality integer ({n_unique} unique)")
            elif n_unique <= 30:
                result['score'] = 0.55
                result['reasons'].append(f"Ordinal integer ({n_unique} unique values)")
            elif cardinality_ratio > 0.50:
                # High cardinality integer - poor QI
                result['score'] = 0.15
                result['reasons'].append(f"High-cardinality integer ({n_unique} unique, {cardinality_ratio:.0%})")
            else:
                result['score'] = 0.35
                result['reasons'].append("Moderate cardinality integer")
        else:
            # Float/continuous - generally poor QIs
            if n_unique <= 10:
                result['score'] = 0.50
                result['reasons'].append("Binned continuous (low unique values)")
            elif n_unique <= 30:
                result['score'] = 0.30
                result['reasons'].append(f"Semi-continuous ({n_unique} unique)")
            else:
                # High cardinality continuous - very poor QI
                result['score'] = 0.10
                result['reasons'].append(f"Continuous ({n_unique} unique) - poor QI candidate")

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
    """Score based on how much this column contributes to record uniqueness."""
    result = {
        'score': 0.0,
        'reasons': [],
        'contribution': 0.0
    }

    # Get potential QI columns
    potential_qis = []
    for col in full_data.columns:
        if col == column:
            continue
        if full_data[col].dtype == 'object':
            potential_qis.append(col)
        elif full_data[col].nunique() / len(full_data) < 0.3:
            potential_qis.append(col)

    potential_qis = potential_qis[:5]

    if len(potential_qis) == 0:
        result['score'] = 0.5
        result['contribution'] = 0.0
        result['reasons'].append("Only potential QI - cannot compare")
        return result

    # Calculate uniqueness WITH this column
    try:
        qis_with = potential_qis[:4] + [column]
        groups_with = full_data.groupby(qis_with, observed=True).size()
        uniqueness_with = (groups_with == 1).sum() / len(full_data)
    except (ValueError, TypeError, KeyError) as exc:
        _log.warning("[qi_detection] Uniqueness-with calculation failed: %s", exc)
        result['score'] = 0.4
        result['reasons'].append("Cannot calculate uniqueness with column")
        return result

    # Calculate uniqueness WITHOUT this column
    try:
        qis_without = potential_qis[:4]
        groups_without = full_data.groupby(qis_without, observed=True).size()
        uniqueness_without = (groups_without == 1).sum() / len(full_data)
    except (ValueError, TypeError, KeyError) as exc:
        _log.warning("[qi_detection] Uniqueness-without calculation failed: %s", exc)
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


def _is_sequential_id(series: pd.Series) -> bool:
    """Check if a numeric series appears to be a sequential ID."""
    if len(series) < 10:
        return False

    try:
        sorted_vals = series.dropna().sort_values().values
        if len(sorted_vals) < 10:
            return False

        diffs = np.diff(sorted_vals)
        if len(diffs) == 0:
            return False

        unique_diffs = np.unique(diffs)
        if len(unique_diffs) <= 3 and np.all(unique_diffs > 0) and np.all(unique_diffs <= 10):
            if series.nunique() / len(series) > 0.95:
                return True
    except (ValueError, TypeError) as exc:
        _log.warning("[qi_detection] Sequential ID check failed: %s", exc)

    return False


def detect_quasi_identifiers_smart(
    data: pd.DataFrame,
    target_k: int = 5,
    max_qis: int = 10,
    confidence_threshold: float = 0.5,
    context: Optional[Dict] = None
) -> Dict:
    """
    Smart QI auto-selection with feasibility-aware filtering.
    
    This function improves on basic detection by:
    1. Running realistic feasibility check on detected QIs
    2. Iteratively removing problematic QIs if target k cannot be achieved
    3. Prioritizing QIs with optimal cardinality (low suppression)
    4. Using tier predictions to guide QI selection
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    target_k : int, default=5
        Target k-anonymity level
    max_qis : int, default=10
        Maximum number of QIs to return
    confidence_threshold : float, default=0.5
        Minimum confidence to consider (lower than default 0.6 to give more options)
    context : dict, optional
        Domain context
    
    Returns:
    --------
    dict : {
        'recommended_qis': List[str],  # Final recommended QI list
        'all_detected': List[str],     # All initially detected QIs
        'removed_qis': List[str],      # QIs removed for feasibility
        'feasibility_status': str,     # Final feasibility status
        'estimated_suppression': float,  # Estimated suppression percentage
        'tier_recommendation': str,    # Recommended preprocessing tier
        'reason': str,                 # Explanation of selection
        'scores': Dict                 # Full scoring details
    }
    
    Example:
    --------
    >>> result = detect_quasi_identifiers_smart(data, target_k=5)
    >>> print(f"Recommended: {result['recommended_qis']}")
    >>> print(f"Feasibility: {result['feasibility_status']}")
    >>> print(f"Use {result['tier_recommendation']} preprocessing")
    """
    from ..preprocessing.diagnose import check_feasibility
    
    # Step 1: Get all candidate QIs with scores
    qi_scores = detect_quasi_identifiers_enhanced(
        data,
        context=context,
        confidence_threshold=confidence_threshold,
        return_scores=True
    )
    
    # Sort by confidence
    sorted_qis = sorted(
        [(col, info) for col, info in qi_scores.items() if info['confidence'] >= confidence_threshold],
        key=lambda x: x[1]['confidence'],
        reverse=True
    )
    
    if not sorted_qis:
        return {
            'recommended_qis': [],
            'all_detected': [],
            'removed_qis': [],
            'feasibility_status': 'UNKNOWN',
            'estimated_suppression': 0,
            'tier_recommendation': 'none',
            'reason': 'No QIs detected with sufficient confidence',
            'scores': qi_scores
        }
    
    # Take top candidates up to max_qis
    initial_qis = [col for col, _ in sorted_qis[:max_qis]]
    all_detected = initial_qis.copy()
    
    # Step 2: Check feasibility with all detected QIs
    try:
        status, msg, details = check_feasibility(
            data,
            initial_qis,
            target_k=target_k,
            realistic=True
        )
        
        realistic_analysis = details.get('realistic_analysis', {})
        
        # If already feasible with minimal suppression, return all QIs
        if status.name == 'FEASIBLE':
            est_supp = realistic_analysis.get('pct_records_below_k', 0) * 100
            if est_supp < 5:
                return {
                    'recommended_qis': initial_qis,
                    'all_detected': all_detected,
                    'removed_qis': [],
                    'feasibility_status': 'FEASIBLE',
                    'estimated_suppression': est_supp,
                    'tier_recommendation': realistic_analysis.get('recommended_tier', 'none'),
                    'reason': f'All {len(initial_qis)} detected QIs are feasible (estimated {est_supp:.1f}% suppression)',
                    'scores': qi_scores
                }
        
        # Step 3: Iterative QI reduction if not immediately feasible
        current_qis = initial_qis.copy()
        min_qis = 3  # Keep at least 3 QIs for meaningful protection
        
        best_result = None
        best_suppression = 100.0
        
        while len(current_qis) >= min_qis:
            status, msg, details = check_feasibility(
                data,
                current_qis,
                target_k=target_k,
                realistic=True
            )
            
            realistic_analysis = details.get('realistic_analysis', {})
            est_supp = realistic_analysis.get('pct_records_below_k', 0) * 100
            recommended_tier = realistic_analysis.get('recommended_tier', 'none')
            
            # Track best result so far
            if est_supp < best_suppression:
                best_suppression = est_supp
                best_result = {
                    'recommended_qis': current_qis.copy(),
                    'all_detected': all_detected,
                    'removed_qis': [qi for qi in all_detected if qi not in current_qis],
                    'feasibility_status': status.name,
                    'estimated_suppression': est_supp,
                    'tier_recommendation': recommended_tier,
                    'reason': f'Optimized to {len(current_qis)} QIs (from {len(all_detected)}) for feasibility',
                    'scores': qi_scores
                }
            
            # Stop if we achieved acceptable suppression
            if est_supp < 10 or status.name == 'FEASIBLE':
                break
            
            # Remove QI with highest cardinality (biggest combination space contributor)
            qi_cardinalities = {qi: data[qi].nunique() for qi in current_qis}
            
            # Penalize continuous high-cardinality columns more
            qi_penalties = {}
            for qi in current_qis:
                cardinality = qi_cardinalities[qi]
                is_continuous = pd.api.types.is_numeric_dtype(data[qi]) and cardinality > 50
                qi_penalties[qi] = cardinality * (5 if is_continuous else 1)
            
            # Remove worst QI
            worst_qi = max(qi_penalties.items(), key=lambda x: x[1])[0]
            current_qis.remove(worst_qi)
        
        # Return best result found
        if best_result:
            return best_result
        
        # Fallback if no good result found
        return {
            'recommended_qis': current_qis,
            'all_detected': all_detected,
            'removed_qis': [qi for qi in all_detected if qi not in current_qis],
            'feasibility_status': 'VERY_HARD',
            'estimated_suppression': 100.0,
            'tier_recommendation': 'very_aggressive',
            'reason': f'Could not achieve good feasibility even with {len(current_qis)} QIs',
            'scores': qi_scores
        }
        
    except Exception as e:
        # If feasibility check fails, return initial detection
        return {
            'recommended_qis': initial_qis,
            'all_detected': all_detected,
            'removed_qis': [],
            'feasibility_status': 'UNKNOWN',
            'estimated_suppression': 0,
            'tier_recommendation': 'none',
            'reason': f'Feasibility check failed: {str(e)}. Using standard detection.',
            'scores': qi_scores
        }
