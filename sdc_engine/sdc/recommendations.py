"""
Shared Recommendations Module
=============================

Single source of truth for preprocessing recommendations.
Used by both diagnosis.py and preprocess.py tabs to ensure consistency.

The recommendations automatically update when:
- QI selection changes in Configure tab
- Protected columns change
- Data characteristics change (after preprocessing)
"""

import pandas as pd
import numpy as np
import hashlib
from typing import Dict, List, Any, Optional


# Sensitive column patterns for differential privacy detection
SENSITIVE_PATTERNS = [
    'income', 'salary', 'wage', 'revenue', 'profit',
    'credit', 'debt', 'balance', 'earning'
]


def get_recommendations_cache_key(
    data: pd.DataFrame,
    direct_ids: Dict[str, str],
    detected_qis: List[str],
    reid_95: float,
    protected_cols: List[str] = None
) -> str:
    """
    Generate a cache key for recommendations based on inputs.
    When inputs change (e.g., QI selection), key changes and recommendations are recalculated.
    """
    key_parts = [
        str(len(data)) if data is not None else "0",
        str(sorted(direct_ids.keys()) if direct_ids else []),
        str(sorted(detected_qis) if detected_qis else []),
        f"{reid_95:.4f}",
        str(sorted(protected_cols) if protected_cols else [])
    ]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()[:16]


def generate_recommendations(
    data: pd.DataFrame,
    direct_ids: Dict[str, str],
    detected_qis: List[str],
    reid_95: float,
    preprocessing_report: Dict[str, Any],
    user_options: Optional[Dict[str, Any]] = None,
    protected_cols: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate all preprocessing recommendations based on data characteristics.

    This is the single source of truth for recommendations used by both
    the Diagnosis and Preprocess tabs.

    Recommendations automatically update when:
    - QI selection changes (detected_qis parameter)
    - Protected columns change (protected_cols parameter)
    - ReID changes (after preprocessing or QI changes)

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze
    direct_ids : dict
        Dict of {column_name: reason} for direct identifiers
    detected_qis : list
        List of detected/selected quasi-identifier column names
    reid_95 : float
        ReID at 95th percentile (0-1)
    preprocessing_report : dict
        Report from auto-scan containing outlier_columns, rare_category_columns, etc.
    user_options : dict, optional
        User selections in manual mode to override automatic recommendations
    protected_cols : list, optional
        Columns marked as protected (should be excluded from transformations)

    Returns
    -------
    dict
        {
            'standard': [list of standard recommendation dicts],
            'advanced': [list of advanced recommendation dicts],
            'all': [combined list sorted by priority],
            'cache_key': str for detecting when recommendations need refresh
        }
    """
    protected_cols = protected_cols or []
    recommendations = {
        'standard': [],
        'advanced': [],
        'all': []
    }

    # Generate cache key for detecting when recommendations need refresh
    cache_key = get_recommendations_cache_key(
        data, direct_ids, detected_qis, reid_95, protected_cols
    )

    # Extract report data safely and filter out protected columns
    outlier_cols = [c for c in _safe_list(preprocessing_report.get('outlier_columns'))
                    if c not in protected_cols]
    rare_cat_cols = [c for c in _safe_list(preprocessing_report.get('rare_category_columns'))
                     if c not in protected_cols]
    high_card_num = [c for c in _safe_list(preprocessing_report.get('high_cardinality_numeric'))
                     if c not in protected_cols]

    # Filter detected QIs to exclude protected columns
    detected_qis = [qi for qi in (detected_qis or []) if qi not in protected_cols]

    # User options for manual mode filtering
    user_options = user_options or {}

    # =========================================================================
    # STANDARD PREPROCESSING RECOMMENDATIONS
    # =========================================================================

    # 1. Direct identifiers (MANDATORY - Priority 1)
    if direct_ids and _should_include('remove_direct_ids', user_options):
        recommendations['standard'].append({
            'id': 'remove_direct_ids',
            'priority': 1,
            'icon': '🔴',
            'method': 'remove_direct_ids',
            'action': 'Remove direct identifiers',
            'columns': list(direct_ids.keys()),
            'where': '📝 Preprocess tab (Auto mode)',
            'why': 'These directly identify individuals and must be removed',
            'mandatory': True
        })

    # 2. Outliers - Top/bottom coding (Priority 2)
    if outlier_cols and _should_include('top_bottom_coding', user_options):
        recommendations['standard'].append({
            'id': 'top_bottom_coding',
            'priority': 2,
            'icon': '🟡',
            'method': 'top_bottom_coding',
            'action': 'Apply top/bottom coding for outliers',
            'columns': outlier_cols,
            'where': '📝 Preprocess tab (Auto mode)',
            'why': 'Outliers increase re-identification risk'
        })

    # 3. Rare categories (Priority 2)
    if rare_cat_cols and _should_include('merge_rare_categories', user_options):
        recommendations['standard'].append({
            'id': 'merge_rare_categories',
            'priority': 2,
            'icon': '🟡',
            'method': 'merge_rare_categories',
            'action': 'Merge rare categories',
            'columns': rare_cat_cols,
            'where': '📝 Preprocess tab (Auto mode)',
            'why': 'Rare categories create small equivalence classes'
        })

    # 4. High-cardinality numeric - Binning (Priority 3)
    if high_card_num and _should_include('bin_numeric', user_options):
        recommendations['standard'].append({
            'id': 'bin_numeric',
            'priority': 3,
            'icon': '🟢',
            'method': 'bin_numeric',
            'action': 'Bin high-cardinality numeric columns',
            'columns': high_card_num,
            'where': '📝 Preprocess tab (Auto mode)',
            'why': 'Binning reduces unique combinations'
        })

    # 5. High-cardinality QIs - GENERALIZE (Priority 2)
    if detected_qis and data is not None and _should_include('generalize_high_card', user_options):
        high_card_qis = [
            qi for qi in detected_qis
            if qi in data.columns and data[qi].nunique() / len(data) > 0.5
        ]
        if high_card_qis:
            recommendations['standard'].append({
                'id': 'generalize_high_card',
                'priority': 2,
                'icon': '🟡',
                'method': 'generalize',
                'action': 'Generalize high-cardinality QIs',
                'columns': high_card_qis,
                'where': '📝 Preprocess tab → GENERALIZE',
                'why': 'High uniqueness increases re-identification risk'
            })

    # 6. High ReID - GENERALIZE (Priority 1)
    if reid_95 >= 0.5 and _should_include('generalize', user_options):
        recommendations['standard'].append({
            'id': 'generalize',
            'priority': 1,
            'icon': '🔴',
            'method': 'generalize',
            'action': 'Apply GENERALIZE before protection',
            'columns': detected_qis[:3] if detected_qis else [],
            'where': '📝 Preprocess tab (Auto mode) or 🛡️ Protect',
            'why': f'ReID₉₅ is {reid_95:.0%} - too high for effective protection without preprocessing'
        })

    # =========================================================================
    # ADVANCED PREPROCESSING RECOMMENDATIONS
    # =========================================================================

    if data is not None:
        # Detect sensitive columns for DP
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        sensitive_cols = [
            col for col in numeric_cols
            if any(p in col.lower() for p in SENSITIVE_PATTERNS)
        ]

        # 7. Differential Privacy (for sensitive numeric) - Priority 3
        enable_dp = 0 < len(sensitive_cols) <= 5
        if enable_dp and _should_include('differential_privacy', user_options):
            recommendations['advanced'].append({
                'id': 'differential_privacy',
                'priority': 3,
                'icon': '🛡️',
                'method': 'differential_privacy',
                'action': 'Apply Differential Privacy to sensitive columns',
                'columns': sensitive_cols,
                'where': '📝 Preprocess tab → Advanced Features',
                'why': 'Adds mathematical privacy guarantees to sensitive numeric data',
                'params': {
                    'epsilon': 1.0,
                    'mechanism': 'laplace'
                }
            })

        # 8. Synthetic Data (for small datasets) - Priority 3
        enable_synthetic = len(data) < 500
        if enable_synthetic and _should_include('synthetic_data', user_options):
            synth_ratio = 0.3 if len(data) < 200 else 0.2
            recommendations['advanced'].append({
                'id': 'synthetic_data',
                'priority': 3,
                'icon': '🤖',
                'method': 'synthetic_data',
                'action': 'Enable synthetic data augmentation',
                'columns': ['(all columns)'],
                'where': '📝 Preprocess tab → Advanced Features',
                'why': f'Small dataset ({len(data)} rows) - synthetic augmentation improves privacy',
                'params': {
                    'ratio': synth_ratio,
                    'method': 'gaussian_copula'
                }
            })

        # 9. Hybrid Perturbation (when DP + traditional needed) - Priority 3
        enable_hybrid = enable_dp and (rare_cat_cols or high_card_num or reid_95 >= 0.5)
        if enable_hybrid and _should_include('hybrid_perturbation', user_options):
            recommendations['advanced'].append({
                'id': 'hybrid_perturbation',
                'priority': 3,
                'icon': '🔀',
                'method': 'hybrid_perturbation',
                'action': 'Enable Hybrid Perturbation',
                'columns': ['(numeric + categorical)'],
                'where': '📝 Preprocess tab → Advanced Features',
                'why': 'Combines DP on numeric with traditional methods on categorical',
                'params': {
                    'dp_epsilon': 0.5,
                    'traditional_method': 'PRAM'
                }
            })

    # Build combined list sorted by priority
    all_recs = recommendations['standard'] + recommendations['advanced']
    recommendations['all'] = sorted(all_recs, key=lambda x: (x['priority'], x['id']))

    # Add cache key for detecting when recommendations need refresh
    recommendations['cache_key'] = cache_key

    # Add metadata about inputs for debugging/display
    recommendations['metadata'] = {
        'n_qis': len(detected_qis),
        'qis': detected_qis,
        'n_protected': len(protected_cols),
        'protected': protected_cols,
        'reid_95': reid_95
    }

    return recommendations


def get_recommendation_by_id(
    recommendations: Dict[str, List[Dict]],
    rec_id: str
) -> Optional[Dict[str, Any]]:
    """Get a specific recommendation by its ID."""
    for rec in recommendations['all']:
        if rec['id'] == rec_id:
            return rec
    return None


def get_enabled_methods(
    recommendations: Dict[str, List[Dict]],
    user_selections: Optional[Dict[str, bool]] = None
) -> List[str]:
    """
    Get list of method IDs that should be applied.

    In auto mode, returns all recommendations.
    In manual mode, filters by user_selections.

    Parameters
    ----------
    recommendations : dict
        Output from generate_recommendations()
    user_selections : dict, optional
        {method_id: bool} for manual mode selections

    Returns
    -------
    list
        List of method IDs to apply
    """
    if user_selections is None:
        # Auto mode - all recommendations enabled
        return [rec['id'] for rec in recommendations['all']]

    # Manual mode - filter by user selections
    return [
        rec['id'] for rec in recommendations['all']
        if user_selections.get(rec['id'], False)
    ]


def get_preview_steps(
    recommendations: Dict[str, List[Dict]],
    enabled_methods: Optional[List[str]] = None
) -> List[str]:
    """
    Get human-readable preview of steps to be applied.

    Parameters
    ----------
    recommendations : dict
        Output from generate_recommendations()
    enabled_methods : list, optional
        List of method IDs to include. If None, includes all.

    Returns
    -------
    list
        List of formatted step descriptions
    """
    steps = []

    for rec in recommendations['all']:
        if enabled_methods is None or rec['id'] in enabled_methods:
            # Format: icon + action + column count
            cols = rec.get('columns', [])
            if cols and cols != ['(all columns)'] and cols != ['(numeric + categorical)']:
                col_info = f" ({len(cols)} columns)"
            else:
                col_info = ""

            steps.append(f"{rec['icon']} {rec['action']}{col_info}")

    return steps


def check_advanced_triggers(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Check which advanced preprocessing features should be triggered.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to analyze

    Returns
    -------
    dict
        {
            'enable_dp': bool,
            'dp_columns': list,
            'enable_synthetic': bool,
            'synthetic_ratio': float,
            'enable_hybrid': bool,
            'triggers': list of explanation strings
        }
    """
    result = {
        'enable_dp': False,
        'dp_columns': [],
        'enable_synthetic': False,
        'synthetic_ratio': 0.0,
        'enable_hybrid': False,
        'triggers': []
    }

    if data is None:
        return result

    # Check for sensitive columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    sensitive_cols = [
        col for col in numeric_cols
        if any(p in col.lower() for p in SENSITIVE_PATTERNS)
    ]

    # DP trigger
    if 0 < len(sensitive_cols) <= 5:
        result['enable_dp'] = True
        result['dp_columns'] = sensitive_cols
        result['triggers'].append(
            f"Differential Privacy: {len(sensitive_cols)} sensitive column(s) detected"
        )

    # Synthetic trigger
    if len(data) < 500:
        result['enable_synthetic'] = True
        result['synthetic_ratio'] = 0.3 if len(data) < 200 else 0.2
        result['triggers'].append(
            f"Synthetic augmentation: Small dataset ({len(data)} rows)"
        )

    return result


def _safe_list(val) -> List:
    """Safely convert value to list."""
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        return [val]
    try:
        return list(val)
    except:
        return []


def _should_include(method_id: str, user_options: Dict[str, Any]) -> bool:
    """
    Check if a method should be included based on user options.

    In auto mode (empty user_options), always returns True.
    In manual mode, checks user_options for explicit inclusion.
    """
    if not user_options:
        # Auto mode - include all
        return True

    # Manual mode - check if method is enabled
    # Map method IDs to user option keys
    option_mapping = {
        'remove_direct_ids': 'remove_ids',
        'top_bottom_coding': 'top_bottom',
        'merge_rare_categories': 'merge_rare',
        'bin_numeric': 'numeric_bin',
        'generalize_high_card': 'generalize_qi',
        'generalize': 'generalize_qi',
        'differential_privacy': 'enable_dp',
        'synthetic_data': 'enable_synthetic',
        'hybrid_perturbation': 'enable_hybrid'
    }

    option_key = option_mapping.get(method_id, method_id)
    return user_options.get(option_key, False)
