"""
QI Diagnosis Module
====================

Analyzes quasi-identifiers for k-anonymity feasibility and provides
recommendations for preprocessing before method selection.

Feasibility Status:
-------------------
- FEASIBLE: Can proceed directly to method selection
- HARD: Fixable with preprocessing (auto or interactive)
- INFEASIBLE: Cannot fix - requires manual QI reduction or alternative approach

This module is OPTIONAL - use it when:
1. You have many QIs or high-cardinality QIs
2. K-anonymity keeps failing with high suppression
3. You want to understand why protection is difficult

Main Functions:
- check_feasibility(): Quick status check (FEASIBLE/HARD/INFEASIBLE)
- diagnose_qis(): Comprehensive QI analysis with recommendations
- recommend_preprocessing(): Get specific preprocessing suggestions
- create_preprocessing_plan(): Create explicit, auditable plan
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class FeasibilityStatus(Enum):
    """K-anonymity feasibility status."""
    FEASIBLE = "feasible"       # Can proceed directly to method selection
    HARD = "hard"               # Fixable with standard preprocessing
    VERY_HARD = "very_hard"     # Requires aggressive preprocessing (was INFEASIBLE)


@dataclass
class DiagnosisResult:
    """Result of QI diagnosis."""
    status: FeasibilityStatus
    expected_eq_size: float
    max_achievable_k: int
    qi_cardinalities: Dict[str, int]
    combination_space: int
    n_records: int
    problematic_qis: List[str]
    recommendations: List[str]
    preprocessing_needed: bool
    blocking_issues: List[str] = field(default_factory=list)  # Issues that make it INFEASIBLE

    @property
    def feasibility(self) -> str:
        """Backward compatibility alias."""
        return self.status.value


@dataclass
class PreprocessingAction:
    """Single preprocessing action for a QI."""
    qi: str
    action: str  # 'bin', 'hierarchy', 'top_k', 'keep', 'exclude', 'reclassify'
    params: Dict
    reason: str
    original_cardinality: int
    estimated_cardinality: int


@dataclass
class PreprocessingPlan:
    """Explicit, auditable preprocessing plan."""
    original_qis: List[str]
    actions: List[PreprocessingAction]
    estimated_feasibility: str
    combination_space_before: int
    combination_space_after: int
    expected_eq_size_before: float
    expected_eq_size_after: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'original_qis': self.original_qis,
            'actions': [
                {
                    'qi': a.qi,
                    'action': a.action,
                    'params': a.params,
                    'reason': a.reason,
                    'original_cardinality': a.original_cardinality,
                    'estimated_cardinality': a.estimated_cardinality,
                }
                for a in self.actions
            ],
            'estimated_feasibility': self.estimated_feasibility,
            'combination_space_before': self.combination_space_before,
            'combination_space_after': self.combination_space_after,
            'expected_eq_size_before': self.expected_eq_size_before,
            'expected_eq_size_after': self.expected_eq_size_after,
            'warnings': self.warnings,
        }


# =============================================================================
# FEASIBILITY CRITERIA
# =============================================================================

# FEASIBLE if ALL true:
FEASIBLE_CRITERIA = {
    'max_qi_count': 6,
    'max_cardinality_per_qi': 100,
    'max_combination_space': 10_000_000,
    'max_uniqueness_per_qi': 0.80,
    'min_records_per_eq': 5,  # At least 5 records per expected EQ (for k=5)
}

# HARD if ANY true (but fixable):
HARD_CRITERIA = {
    'combination_space_threshold': 100_000_000,
    'qi_cardinality_threshold': 500,
    'high_card_qi_count': 3,  # >3 QIs with cardinality > 100
    'min_sample_size': 1000,
}

# VERY_HARD patterns (require aggressive preprocessing):
# Previously called INFEASIBLE - now we try harder instead of giving up
VERY_HARD_PATTERNS = [
    'free_text',      # Free text fields in QI set -> exclude or top_k aggressively
    'identifier',     # IDs, names in QI set -> must exclude
    'sparse',         # Combination space >> data size -> aggressive reduction
]

# =============================================================================
# QI PRIORITY CONFIGURATION (for auto-mode drop decisions)
# =============================================================================
# User can override these by providing required_qis/droppable_qis

QI_PRIORITY_CONFIG = {
    # Patterns that are typically essential for analysis (keep first)
    'high_priority_patterns': ['age', 'sex', 'gender', 'region', 'education'],

    # Patterns with moderate analytical value
    'medium_priority_patterns': ['occupation', 'marital', 'nationality'],

    # Patterns that can often be dropped without major analytical loss
    'low_priority_patterns': ['religion', 'ethnicity', 'postal'],

    # Priority score adjustments
    'base_priority': 100,
    'high_priority_bonus': 50,
    'medium_priority_bonus': 25,
    'high_cardinality_penalty': 30,  # cardinality > 100
    'medium_cardinality_penalty': 15,  # cardinality > 50

    # Cardinality thresholds for penalties
    'high_cardinality_threshold': 100,
    'medium_cardinality_threshold': 50,
}

# Patterns suggesting hierarchy-based aggregation
HIERARCHY_PATTERNS = ['municipality', 'postal', 'occupation', 'region', 'district', 'prefecture']

# =============================================================================
# COLUMN DETECTION CONFIGURATION
# =============================================================================
# User can override these patterns for domain-specific detection

DETECTION_CONFIG = {
    # Direct identifiers - MUST be removed
    'direct_id_patterns': [
        'id', 'ssn', 'social_security', 'tax_id', 'afm', 'amka',
        'passport', 'license', 'phone', 'email', 'address',
        'name', 'surname', 'firstname', 'lastname'
    ],

    # Sensitive attributes - confidential but not identifying alone
    'sensitive_patterns': [
        'salary', 'income', 'wage', 'earnings', 'diagnosis',
        'disease', 'condition', 'treatment', 'score', 'grade',
        'debt', 'loan', 'credit', 'tax_amount', 'benefit'
    ],

    # Quasi-identifier patterns
    'qi_patterns': [
        'age', 'sex', 'gender', 'birth', 'occupation', 'profession',
        'education', 'municipality', 'region', 'postal', 'zip',
        'marital', 'nationality', 'ethnicity', 'religion'
    ],

    # Thresholds for automatic detection
    'identifier_uniqueness_threshold': 0.90,  # >90% unique = likely identifier
    'qi_max_cardinality': 100,  # Categorical with <=100 values = potential QI
    'qi_min_cardinality': 2,    # Need at least 2 values
}

# =============================================================================
# UTILITY CONSTRAINTS DEFAULTS
# =============================================================================
# User can override to preserve analytical precision

UTILITY_DEFAULTS = {
    'default_n_bins': 10,           # Default number of bins for numeric
    'default_top_k': 30,            # Default k for top_k aggregation
    'min_bins_numeric': 5,          # Never bin below this
    'max_bins_numeric': 100,        # Never bin above this
    'preserve_zero': True,          # Keep 0 as separate category when binning
    'preserve_outliers': False,     # Create separate bin for outliers
}


def check_feasibility(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    target_k: int = 5,
    access_tier: str = 'SCIENTIFIC',
    realistic: bool = True
) -> Tuple[FeasibilityStatus, str, Dict]:
    """
    Quick feasibility check for k-anonymity.
    
    Now includes realistic analysis based on actual data distribution patterns
    rather than theoretical combination space calculations.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    quasi_identifiers : list
        QI columns to check
    target_k : int
        Target k-anonymity value
    access_tier : str
        Access tier (for compatibility)
    realistic : bool
        If True, analyze actual data distribution. If False, use theoretical calculations.

    Returns:
    --------
    tuple : (status, message, details)
        - status: FeasibilityStatus (FEASIBLE, HARD, VERY_HARD)
        - message: Human-readable explanation
        - details: Dict with metrics, recommendations, and tier predictions

    Example:
    --------
    >>> status, msg, details = check_feasibility(data, ['age', 'region'], target_k=5)
    >>> if status == FeasibilityStatus.FEASIBLE:
    ...     proceed_to_method_selection()
    >>> elif status == FeasibilityStatus.HARD:
    ...     preprocess_then_select()
    >>> else:
    ...     show_error_and_options()
    """
    n_records = len(data)
    n_qis = len(quasi_identifiers)

    # Calculate metrics
    qi_cardinalities = {}
    combination_space = 1
    high_card_count = 0
    blocking_issues = []

    for qi in quasi_identifiers:
        if qi not in data.columns:
            continue
        card = data[qi].nunique()
        qi_cardinalities[qi] = card
        combination_space *= card

        if card > 100:
            high_card_count += 1

        # Check for INFEASIBLE patterns
        uniqueness = card / n_records if n_records > 0 else 0
        if uniqueness > 0.95:
            # Near-unique - likely an identifier
            blocking_issues.append(f"'{qi}' is near-unique ({uniqueness:.0%}) - likely an identifier")

        # Check for free text (very high cardinality relative to size)
        if card > n_records * 0.5 and not pd.api.types.is_numeric_dtype(data[qi]):
            blocking_issues.append(f"'{qi}' may contain free text ({card} unique values)")

    expected_eq_size = n_records / combination_space if combination_space > 0 else 0
    
    # ==========================================================================
    # REALISTIC ANALYSIS - Analyze actual data distribution
    # ==========================================================================
    realistic_analysis = {}
    recommended_tier = 'none'
    
    if realistic and n_records > 0 and len([qi for qi in quasi_identifiers if qi in data.columns]) > 0:
        try:
            # Get actual equivalence class sizes
            valid_qis = [qi for qi in quasi_identifiers if qi in data.columns]
            actual_eq_sizes = data.groupby(valid_qis, dropna=False).size()
            
            min_eq = actual_eq_sizes.min()
            median_eq = actual_eq_sizes.median()
            mean_eq = actual_eq_sizes.mean()
            pct_below_k = (actual_eq_sizes < target_k).mean()
            
            # Simulate preprocessing tiers (estimate suppression reduction)
            # Based on empirical observation: reducing cardinality merges small EQs
            tier_predictions = {
                'none': {
                    'max_categories': None,
                    'estimated_suppression_pct': pct_below_k * 100,
                    'acceptable': pct_below_k < 0.05
                },
                'light': {
                    'max_categories': 15,
                    'estimated_suppression_pct': pct_below_k * 60,  # 40% reduction
                    'acceptable': pct_below_k * 0.6 < 0.05
                },
                'moderate': {
                    'max_categories': 10,
                    'estimated_suppression_pct': pct_below_k * 30,  # 70% reduction
                    'acceptable': pct_below_k * 0.3 < 0.05
                },
                'aggressive': {
                    'max_categories': 5,
                    'estimated_suppression_pct': pct_below_k * 10,  # 90% reduction
                    'acceptable': pct_below_k * 0.1 < 0.05
                },
                'very_aggressive': {
                    'max_categories': 3,
                    'estimated_suppression_pct': pct_below_k * 5,  # 95% reduction
                    'acceptable': pct_below_k * 0.05 < 0.05
                }
            }
            
            # Find minimum tier needed
            for tier_name, tier_info in tier_predictions.items():
                if tier_info['acceptable']:
                    recommended_tier = tier_name
                    break
            
            # If no tier is acceptable, recommend very_aggressive anyway
            if recommended_tier == 'none' and pct_below_k >= 0.05:
                recommended_tier = 'very_aggressive'
            
            realistic_analysis = {
                'actual_eq_min': float(min_eq),
                'actual_eq_median': float(median_eq),
                'actual_eq_mean': float(mean_eq),
                'pct_records_below_k': float(pct_below_k),
                'num_equivalence_classes': len(actual_eq_sizes),
                'tier_predictions': tier_predictions,
                'recommended_tier': recommended_tier,
                'uses_actual_distribution': True
            }
            
        except Exception as e:
            # Fallback to theoretical if realistic analysis fails
            realistic_analysis = {
                'error': str(e),
                'uses_actual_distribution': False,
                'recommended_tier': 'moderate'  # Safe default
            }

    details = {
        'n_records': n_records,
        'n_qis': n_qis,
        'qi_cardinalities': qi_cardinalities,
        'combination_space': combination_space,
        'expected_eq_size': expected_eq_size,
        'target_k': target_k,
        'high_card_count': high_card_count,
        'blocking_issues': blocking_issues,
        'realistic_analysis': realistic_analysis,
    }

    # ==========================================================================
    # Check VERY_HARD first (requires enhanced preprocessing)
    # Previously INFEASIBLE - now we try harder with stronger binning/grouping
    # ==========================================================================
    if blocking_issues:
        # Add enhanced preprocessing recommendations
        enhanced_recs = []
        for issue in blocking_issues:
            if 'identifier' in issue.lower() or 'near-unique' in issue.lower():
                enhanced_recs.append("EXCLUDE: Remove identifier columns")
            elif 'free text' in issue.lower():
                enhanced_recs.append("EXCLUDE or TOP_K(10): Handle free text fields")
        details['enhanced_recommendations'] = enhanced_recs
        details['try_hard_mode'] = True
        
        # Add realistic tier recommendation if available
        if realistic_analysis.get('recommended_tier'):
            details['recommended_preprocessing_tier'] = realistic_analysis['recommended_tier']
        
        return (
            FeasibilityStatus.VERY_HARD,
            f"VERY_HARD: {'; '.join(blocking_issues)}. Will try enhanced preprocessing.",
            details
        )

    # VERY_HARD if combination space is MUCH larger than records
    # (Expected EQ < 0.1 means >90% would need suppression)
    if combination_space > n_records * 10:
        details['blocking_issues'].append("Combination space vastly exceeds records")
        details['enhanced_recommendations'] = [
            "REDUCE_QIS: Drop lowest-priority QIs until feasible",
            "STRONGER_BINNING: Bin all numeric to 5-10 categories",
            "TOP_K(10): Limit all categorical to top 10 + Other",
        ]
        details['try_hard_mode'] = True
        return (
            FeasibilityStatus.VERY_HARD,
            f"VERY_HARD: Combination space ({combination_space:,}) >> records ({n_records:,}). "
            f"Expected EQ={expected_eq_size:.2f}. Will try enhanced preprocessing.",
            details
        )

    # ==========================================================================
    # Check FEASIBLE (all criteria met)
    # ==========================================================================
    is_feasible = True
    feasibility_issues = []

    if n_qis > FEASIBLE_CRITERIA['max_qi_count']:
        is_feasible = False
        feasibility_issues.append(f"{n_qis} QIs (max {FEASIBLE_CRITERIA['max_qi_count']})")

    for qi, card in qi_cardinalities.items():
        if card > FEASIBLE_CRITERIA['max_cardinality_per_qi']:
            is_feasible = False
            feasibility_issues.append(f"'{qi}' cardinality {card} > {FEASIBLE_CRITERIA['max_cardinality_per_qi']}")

    if combination_space > FEASIBLE_CRITERIA['max_combination_space']:
        is_feasible = False
        feasibility_issues.append(f"Combination space {combination_space:,} > {FEASIBLE_CRITERIA['max_combination_space']:,}")

    if expected_eq_size < FEASIBLE_CRITERIA['min_records_per_eq']:
        is_feasible = False
        feasibility_issues.append(f"Expected EQ size {expected_eq_size:.1f} < {FEASIBLE_CRITERIA['min_records_per_eq']}")

    # Override with realistic analysis if available
    if realistic_analysis.get('uses_actual_distribution'):
        pct_below = realistic_analysis.get('pct_records_below_k', 1.0)
        if pct_below < 0.05:
            # Less than 5% records would be suppressed - FEASIBLE
            is_feasible = True
        elif pct_below > 0.5:
            # More than 50% records would be suppressed - needs aggressive preprocessing
            is_feasible = False
            feasibility_issues.append(f"{pct_below:.0%} of records would be suppressed without preprocessing")

    if is_feasible:
        # Add realistic analysis to message if available
        if realistic_analysis.get('uses_actual_distribution'):
            pct_below = realistic_analysis.get('pct_records_below_k', 0)
            median_eq = realistic_analysis.get('actual_eq_median', expected_eq_size)
            return (
                FeasibilityStatus.FEASIBLE,
                f"FEASIBLE: Only {pct_below:.1%} of records below k={target_k} (Median EQ: {median_eq:.1f}). Proceed to method selection.",
                details
            )
        else:
            return (
                FeasibilityStatus.FEASIBLE,
                f"FEASIBLE: Expected EQ size {expected_eq_size:.1f} >= k={target_k}. Proceed to method selection.",
                details
            )

    # ==========================================================================
    # Otherwise HARD (fixable with preprocessing)
    # ==========================================================================
    recommendations = []

    # Add realistic tier recommendation if available
    if realistic_analysis.get('recommended_tier') and realistic_analysis['recommended_tier'] != 'none':
        tier_name = realistic_analysis['recommended_tier'].replace('_', ' ').title()
        pct_below = realistic_analysis.get('pct_records_below_k', 0)
        recommendations.append(f"Use {tier_name} preprocessing - currently {pct_below:.0%} records below k={target_k}")

    # Identify highest-cardinality QIs
    sorted_qis = sorted(qi_cardinalities.items(), key=lambda x: x[1], reverse=True)
    for qi, card in sorted_qis[:3]:
        if card > FEASIBLE_CRITERIA['max_cardinality_per_qi']:
            if pd.api.types.is_numeric_dtype(data[qi]):
                recommendations.append(f"Bin '{qi}' ({card} values) into ranges")
            else:
                recommendations.append(f"Apply TOP_K or hierarchy to '{qi}' ({card} values)")

    if n_qis > FEASIBLE_CRITERIA['max_qi_count']:
        recommendations.append(f"Consider reducing QI count from {n_qis} to {FEASIBLE_CRITERIA['max_qi_count']}")

    details['recommendations'] = recommendations
    details['feasibility_issues'] = feasibility_issues
    
    # Add tier recommendation to details
    if realistic_analysis.get('recommended_tier'):
        details['recommended_preprocessing_tier'] = realistic_analysis['recommended_tier']

    return (
        FeasibilityStatus.HARD,
        f"HARD: {'; '.join(feasibility_issues[:2])}. Fixable with preprocessing.",
        details
    )


def diagnose_qis(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    access_tier: str = 'SCIENTIFIC',
    verbose: bool = True
) -> DiagnosisResult:
    """
    Comprehensive diagnosis of quasi-identifiers for k-anonymity feasibility.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    quasi_identifiers : list
        List of QI column names
    access_tier : str
        Target access tier: 'PUBLIC', 'SCIENTIFIC', or 'SECURE'
    verbose : bool
        Print diagnosis details

    Returns:
    --------
    DiagnosisResult : Comprehensive diagnosis with recommendations
    """
    status, msg, details = check_feasibility(data, quasi_identifiers, access_tier=access_tier)

    n_records = details['n_records']
    qi_cardinalities = details['qi_cardinalities']
    combination_space = details['combination_space']
    expected_eq_size = details['expected_eq_size']
    blocking_issues = details.get('blocking_issues', [])

    # Determine max achievable k
    if expected_eq_size >= 10:
        max_k = min(int(expected_eq_size), 20)
    elif expected_eq_size >= 5:
        max_k = 5
    elif expected_eq_size >= 3:
        max_k = 3
    else:
        max_k = 0

    # Identify problematic QIs
    problematic_qis = []
    max_card = FEASIBLE_CRITERIA['max_cardinality_per_qi']

    for qi, card in qi_cardinalities.items():
        if card > max_card:
            problematic_qis.append(qi)

    # Generate recommendations
    recommendations = details.get('recommendations', [])
    preprocessing_needed = status != FeasibilityStatus.FEASIBLE

    if status == FeasibilityStatus.VERY_HARD:
        recommendations = [
            "CANNOT FIX AUTOMATICALLY. Options:",
            "1. Remove identifier columns from QI set",
            "2. Use tabular aggregation instead of microdata release",
            "3. Release only summary statistics",
        ]

    if verbose:
        _print_diagnosis(
            status, n_records, quasi_identifiers, qi_cardinalities,
            combination_space, expected_eq_size, max_k,
            problematic_qis, recommendations, blocking_issues
        )

    return DiagnosisResult(
        status=status,
        expected_eq_size=expected_eq_size,
        max_achievable_k=max_k,
        qi_cardinalities=qi_cardinalities,
        combination_space=combination_space,
        n_records=n_records,
        problematic_qis=problematic_qis,
        recommendations=recommendations,
        preprocessing_needed=preprocessing_needed,
        blocking_issues=blocking_issues
    )


def _print_diagnosis(
    status, n_records, quasi_identifiers, qi_cardinalities,
    combination_space, expected_eq_size, max_k,
    problematic_qis, recommendations, blocking_issues
):
    """Print formatted diagnosis report."""
    status_icon = {
        FeasibilityStatus.FEASIBLE: "[OK]",
        FeasibilityStatus.HARD: "[!]",
        FeasibilityStatus.INFEASIBLE: "[X]"
    }

    print("\n" + "=" * 60)
    print(f"  QI DIAGNOSIS REPORT  {status_icon.get(status, '')}")
    print("=" * 60)
    print(f"\n  Dataset: {n_records:,} records")
    print(f"  QIs: {len(quasi_identifiers)}")
    print(f"  Status: {status.value.upper()}")

    max_card = FEASIBLE_CRITERIA['max_cardinality_per_qi']
    print(f"\n  QI Cardinalities (threshold: {max_card}):")
    for qi, card in sorted(qi_cardinalities.items(), key=lambda x: x[1], reverse=True):
        icon = "[!]" if card > max_card else "[OK]"
        print(f"    {icon} {qi}: {card:,} unique values")

    print(f"\n  Combination Space: {combination_space:,}")
    print(f"  Expected EQ Size: {expected_eq_size:.2f}")
    print(f"  Max Achievable k: {max_k}")

    if blocking_issues:
        print(f"\n  BLOCKING ISSUES (cannot fix):")
        for issue in blocking_issues:
            print(f"    [X] {issue}")

    if recommendations:
        print(f"\n  Recommendations:")
        for rec in recommendations:
            print(f"    - {rec}")

    print("=" * 60)


def recommend_preprocessing(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    access_tier: str = 'SCIENTIFIC',
    available_hierarchies: Optional[Dict] = None,
    max_qis: Optional[int] = None,
    required_qis: Optional[List[str]] = None,
    droppable_qis: Optional[List[str]] = None,
    priority_config: Optional[Dict] = None,
    sensitive_vars: Optional[List[str]] = None,
    utility_constraints: Optional[Dict] = None,
    detection_config: Optional[Dict] = None,
    try_hard: bool = False,
    column_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """
    Generate specific preprocessing recommendations for each QI.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    quasi_identifiers : list
        QI columns
    access_tier : str
        Target access tier
    available_hierarchies : dict, optional
        User-provided hierarchies that can be used
    max_qis : int, optional
        Maximum number of QIs to keep. If None, uses FEASIBLE_CRITERIA.
    required_qis : list, optional
        QIs that MUST be kept (never dropped). User specifies these.
    droppable_qis : list, optional
        QIs that CAN be dropped if needed. Drop these first when count exceeds max.
    priority_config : dict, optional
        Custom priority configuration. If None, uses QI_PRIORITY_CONFIG.
    sensitive_vars : list, optional
        Columns containing sensitive data (for l-diversity). These columns
        are flagged but not processed as QIs.
    utility_constraints : dict, optional
        Constraints to preserve analytical utility:
        - 'preserve_precision': list of QIs that need high granularity
        - 'min_bins': dict of {qi: min_bins} for numeric columns
        - 'min_k': dict of {qi: min_k} for top_k aggregation
        Example: {'preserve_precision': ['age'], 'min_bins': {'age': 15}}
    detection_config : dict, optional
        Custom column detection patterns. If None, uses DETECTION_CONFIG.
        - 'direct_id_patterns': list of patterns for identifiers
        - 'sensitive_patterns': list of patterns for sensitive columns
        - 'qi_patterns': list of patterns for quasi-identifiers
        - 'identifier_uniqueness_threshold': float (default 0.90)
    try_hard : bool
        If True, use aggressive preprocessing for VERY_HARD cases:
        - Exclude near-unique columns (identifiers)
        - Aggressively bin numeric to 5-10 categories
        - Limit categorical to top 10 + Other
        - Reduce QI count more aggressively
    column_types : dict, optional
        Semantic column types from Configure's Data Type column.
        Maps column name to type string (e.g. 'Character — Date/Time',
        'Char (numeric) — Continuous'). When provided, used as primary
        signal for numeric/date detection instead of data probing.

    Returns:
    --------
    dict : {qi_name: {action, params, reason, original_cardinality, estimated_cardinality,
                      priority, is_required, is_droppable, is_sensitive, utility_preserved}}

    Example:
    --------
    >>> # Full user customization
    >>> recs = recommend_preprocessing(
    ...     data, qis,
    ...     required_qis=['age', 'sex', 'region'],
    ...     droppable_qis=['religion', 'ethnicity'],
    ...     sensitive_vars=['salary', 'diagnosis'],
    ...     utility_constraints={
    ...         'preserve_precision': ['age', 'income_bracket'],
    ...         'min_bins': {'age': 15}
    ...     },
    ...     detection_config={
    ...         'direct_id_patterns': ['customer_id', 'account_number'],
    ...         'sensitive_patterns': ['health_status'],
    ...     }
    ... )
    >>>
    >>> # Aggressive mode for difficult cases
    >>> recs = recommend_preprocessing(data, qis, try_hard=True)
    """
    max_card = FEASIBLE_CRITERIA['max_cardinality_per_qi']
    max_qi_count = max_qis or FEASIBLE_CRITERIA['max_qi_count']
    n_records = len(data)
    recommendations = {}

    # Aggressive settings for try_hard mode
    if try_hard:
        max_card = min(max_card, 20)  # More aggressive cardinality limit
        aggressive_bins = 5  # Fewer bins for numeric
        aggressive_k = 10  # Fewer categories for categorical

    # Use provided configs or defaults
    priority_cfg = priority_config or QI_PRIORITY_CONFIG
    detect_cfg = detection_config or DETECTION_CONFIG
    utility_cfg = utility_constraints or {}

    # Normalize to sets for fast lookup
    required_set = set(required_qis) if required_qis else set()
    droppable_set = set(droppable_qis) if droppable_qis else set()
    sensitive_set = set(sensitive_vars) if sensitive_vars else set()
    preserve_precision_set = set(utility_cfg.get('preserve_precision', []))
    min_bins_map = utility_cfg.get('min_bins', {})
    min_k_map = utility_cfg.get('min_k', {})

    # Detection thresholds from config
    identifier_threshold = detect_cfg.get('identifier_uniqueness_threshold', 0.90)

    def get_qi_priority(qi_name: str, cardinality: int) -> int:
        """Calculate priority score (higher = more important to keep)."""
        # User-defined required QIs get highest priority
        if qi_name in required_set:
            return 1000  # Always keep

        # User-defined droppable QIs get lowest priority
        if qi_name in droppable_set:
            return 0  # Drop first

        # Otherwise use pattern-based heuristics from config
        qi_lower = qi_name.lower()
        base_priority = priority_cfg['base_priority']

        # Adjust by pattern
        if any(p in qi_lower for p in priority_cfg['high_priority_patterns']):
            base_priority += priority_cfg['high_priority_bonus']
        elif any(p in qi_lower for p in priority_cfg['medium_priority_patterns']):
            base_priority += priority_cfg['medium_priority_bonus']

        # Lower cardinality = higher priority (less risk)
        # Penalize high cardinality
        if cardinality > priority_cfg['high_cardinality_threshold']:
            base_priority -= priority_cfg['high_cardinality_penalty']
        elif cardinality > priority_cfg['medium_cardinality_threshold']:
            base_priority -= priority_cfg['medium_cardinality_penalty']

        return base_priority

    # Semantic type detection — uses Configure's column_types first,
    # then falls back to data probing (same logic as apply_generalize).
    _col_types = column_types or {}
    _numeric_kw = {'continuous', 'numeric', 'integer', 'float',
                   'income', 'financial', 'age', 'area', 'price',
                   'amount'}
    _date_kw = {'date', 'time', 'temporal', 'datetime', 'year', 'birth'}

    def _detect_semantic_type(col_name, col):
        """Return 'numeric', 'date', or 'categorical' for a column."""
        # Priority 1: Configure's Data Type string
        ct = _col_types.get(col_name, '').lower()
        if ct:
            if any(kw in ct for kw in _date_kw):
                return 'date'
            if any(kw in ct for kw in _numeric_kw):
                return 'numeric'
            # Explicit "categorical" in Configure → trust it
            if 'categorical' in ct or 'coded' in ct:
                return 'categorical'

        # Priority 2: pandas dtype
        if pd.api.types.is_numeric_dtype(col):
            return 'numeric'
        if pd.api.types.is_datetime64_any_dtype(col):
            return 'date'

        # Priority 3: data probing (fallback when no column_types)
        if col.dtype == object:
            sample = col.dropna().head(200)
            if len(sample) > 0:
                num_ct = pd.to_numeric(sample, errors='coerce').notna().sum()
                if num_ct / len(sample) > 0.8:
                    return 'numeric'
                # Only attempt date parsing if values look plausibly date-like
                # (contain digits + separators like /, -, or month names)
                _sample_str = sample.astype(str)
                _date_like = _sample_str.str.contains(
                    r'\d{1,4}[/\-\.]\d{1,2}|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',
                    case=False, na=False)
                if _date_like.sum() / len(sample) > 0.5:
                    try:
                        date_ct = pd.to_datetime(
                            sample, errors='coerce', dayfirst=True,
                            infer_datetime_format=True
                        ).notna().sum()
                        if date_ct / len(sample) > 0.8:
                            return 'date'
                    except Exception:
                        pass
        return 'categorical'

    # First pass: analyze all QIs
    qi_analysis = []
    for qi in quasi_identifiers:
        if qi not in data.columns:
            continue
        card = data[qi].nunique()
        uniqueness = card / n_records if n_records > 0 else 0
        priority = get_qi_priority(qi, card)
        qi_analysis.append({
            'qi': qi,
            'cardinality': card,
            'uniqueness': uniqueness,
            'priority': priority,
            'is_identifier': uniqueness > identifier_threshold,
            'is_required': qi in required_set,
            'is_droppable': qi in droppable_set,
            'is_sensitive': qi in sensitive_set,
            'preserve_precision': qi in preserve_precision_set,
            'semantic_type': _detect_semantic_type(qi, data[qi]),
        })

    # Sort by priority (descending) - highest priority first
    qi_analysis.sort(key=lambda x: x['priority'], reverse=True)

    # Count how many we can keep
    kept_count = 0
    for qa in qi_analysis:
        qi = qa['qi']
        card = qa['cardinality']
        uniqueness = qa['uniqueness']
        # Use semantic type detection (handles numeric/date strings)
        sem_type = qa.get('semantic_type', 'categorical')
        is_numeric = sem_type in ('numeric', 'date')

        rec = {
            'original_cardinality': card,
            'action': 'keep',
            'params': {},
            'reason': f"Cardinality {card} is acceptable",
            'estimated_cardinality': card,
            'priority': qa['priority'],
            'is_required': qa['is_required'],
            'is_droppable': qa['is_droppable'],
            'is_sensitive': qa['is_sensitive'],
            'utility_preserved': qa['preserve_precision'],
            'semantic_type': sem_type,
        }

        # Check for identifiers (must exclude) - UNLESS user marked as required
        if qa['is_identifier'] and not qa['is_required']:
            rec['action'] = 'exclude'
            rec['params'] = {}
            rec['reason'] = f"Near-unique ({uniqueness:.0%}) - must exclude (likely identifier)"
            rec['estimated_cardinality'] = 0
            recommendations[qi] = rec
            continue

        # Required QIs are never dropped due to count limits
        if qa['is_required']:
            kept_count += 1
            # But may still need preprocessing for high cardinality
            if card > max_card:
                rec = _get_preprocessing_recommendation(
                    qi, card, is_numeric, max_card, available_hierarchies, rec,
                    min_bins=min_bins_map.get(qi),
                    min_k=min_k_map.get(qi),
                    preserve_precision=qa['preserve_precision'],
                    try_hard=try_hard
                )
            recommendations[qi] = rec
            continue

        # If cardinality is acceptable, keep it
        if card <= max_card:
            kept_count += 1
            recommendations[qi] = rec
            continue

        # Need preprocessing for high cardinality
        kept_count += 1
        rec = _get_preprocessing_recommendation(
            qi, card, is_numeric, max_card, available_hierarchies, rec,
            min_bins=min_bins_map.get(qi),
            min_k=min_k_map.get(qi),
            preserve_precision=qa['preserve_precision'],
            try_hard=try_hard
        )
        recommendations[qi] = rec

    return recommendations


def _get_preprocessing_recommendation(
    qi: str,
    card: int,
    is_numeric: bool,
    max_card: int,
    available_hierarchies: Optional[Dict],
    rec: Dict,
    min_bins: Optional[int] = None,
    min_k: Optional[int] = None,
    preserve_precision: bool = False,
    try_hard: bool = False
) -> Dict:
    """
    Determine preprocessing recommendation for a high-cardinality QI.

    Internal helper function.

    Parameters:
    -----------
    qi : str
        Column name
    card : int
        Current cardinality
    is_numeric : bool
        Whether column is numeric
    max_card : int
        Maximum allowed cardinality
    available_hierarchies : dict, optional
        User-provided hierarchies
    rec : dict
        Recommendation dict to update
    min_bins : int, optional
        Minimum bins for this column (utility constraint)
    min_k : int, optional
        Minimum k for top_k (utility constraint)
    preserve_precision : bool
        If True, use more bins/higher k to preserve analytical precision
    try_hard : bool
        If True, use aggressive settings (fewer bins/categories)
    """
    # Get defaults from config - override with aggressive values if try_hard
    if try_hard:
        default_bins = 5  # Aggressive: fewer bins
        default_k = 10    # Aggressive: fewer categories
        min_bins_limit = 3
    else:
        default_bins = UTILITY_DEFAULTS['default_n_bins']
        default_k = UTILITY_DEFAULTS['default_top_k']
        min_bins_limit = UTILITY_DEFAULTS['min_bins_numeric']
    max_bins_limit = UTILITY_DEFAULTS['max_bins_numeric']

    # User-provided hierarchy
    if available_hierarchies and qi in available_hierarchies:
        rec['action'] = 'hierarchy'
        rec['params'] = {'mapping': available_hierarchies[qi]}
        rec['reason'] = "User-provided hierarchy available"
        rec['estimated_cardinality'] = len(set(available_hierarchies[qi].values()))
    # Numeric - use binning
    elif is_numeric:
        # Calculate bins respecting utility constraints
        if preserve_precision:
            # More bins for precision
            n_bins = min(max_bins_limit, max(default_bins * 2, card // 5))
        else:
            n_bins = min(max_card, max(default_bins, card // 10))

        # Apply user min_bins constraint
        if min_bins is not None:
            n_bins = max(n_bins, min_bins)

        # Enforce global limits
        n_bins = max(min_bins_limit, min(max_bins_limit, n_bins))

        rec['action'] = 'bin'
        rec['params'] = {'bins': n_bins, 'strategy': 'quantile'}
        if preserve_precision:
            rec['reason'] = f"Numeric column - bin into {n_bins} ranges (precision preserved)"
        else:
            rec['reason'] = f"Numeric column - bin into {n_bins} ranges"
        rec['estimated_cardinality'] = n_bins
    # Geographic/occupation patterns - hierarchy preferred
    elif any(p in qi.lower() for p in HIERARCHY_PATTERNS):
        rec['action'] = 'hierarchy'
        rec['params'] = {'level': 'coarse'}
        rec['reason'] = f"Geographic/occupation - use hierarchy if available, else TOP_K"
        rec['estimated_cardinality'] = min(max_card, card // 3)
    # Very high cardinality - top_k
    elif card > max_card * 3:
        # Calculate k respecting utility constraints
        if preserve_precision:
            k = min(max_card * 2, max(default_k, card // 3))
        else:
            k = min(max_card, max(20, card // 5))

        # Apply user min_k constraint
        if min_k is not None:
            k = max(k, min_k)

        rec['action'] = 'top_k'
        rec['params'] = {'k': k}
        if preserve_precision:
            rec['reason'] = f"Very high cardinality ({card}) - keep top {k} + 'Other' (precision preserved)"
        else:
            rec['reason'] = f"Very high cardinality ({card}) - keep top {k} + 'Other'"
        rec['estimated_cardinality'] = k + 1
    # Moderate high cardinality - top_k
    else:
        k = max_card if not preserve_precision else min(card, max_card * 2)

        # Apply user min_k constraint
        if min_k is not None:
            k = max(k, min_k)

        rec['action'] = 'top_k'
        rec['params'] = {'k': k}
        if preserve_precision:
            rec['reason'] = f"High cardinality ({card} > {max_card}) - keep top {k} (precision preserved)"
        else:
            rec['reason'] = f"High cardinality ({card} > {max_card}) - keep top {k}"
        rec['estimated_cardinality'] = k + 1

    return rec


def create_preprocessing_plan(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    recommendations: Optional[Dict] = None,
    access_tier: str = 'SCIENTIFIC'
) -> PreprocessingPlan:
    """
    Create explicit, auditable preprocessing plan.

    This plan can be:
    - Reviewed by user before applying
    - Modified if needed
    - Saved/loaded for reproducibility
    - Audited for compliance

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    quasi_identifiers : list
        QI columns
    recommendations : dict, optional
        Output from recommend_preprocessing(). If None, will generate.
    access_tier : str
        Target access tier

    Returns:
    --------
    PreprocessingPlan : Explicit plan with all actions
    """
    if recommendations is None:
        recommendations = recommend_preprocessing(data, quasi_identifiers, access_tier)

    n_records = len(data)

    # Calculate before metrics
    combination_space_before = 1
    for qi in quasi_identifiers:
        if qi in data.columns:
            combination_space_before *= data[qi].nunique()
    expected_eq_before = n_records / combination_space_before if combination_space_before > 0 else 0

    # Build actions and estimate after metrics
    actions = []
    combination_space_after = 1
    warnings = []

    for qi in quasi_identifiers:
        if qi not in recommendations:
            continue

        rec = recommendations[qi]
        action = PreprocessingAction(
            qi=qi,
            action=rec['action'],
            params=rec['params'],
            reason=rec['reason'],
            original_cardinality=rec['original_cardinality'],
            estimated_cardinality=rec['estimated_cardinality'],
        )
        actions.append(action)

        if rec['action'] != 'exclude':
            combination_space_after *= rec['estimated_cardinality']

    expected_eq_after = n_records / combination_space_after if combination_space_after > 0 else 0

    # Determine estimated feasibility
    if expected_eq_after >= 10:
        estimated_feasibility = 'feasible'
    elif expected_eq_after >= 5:
        estimated_feasibility = 'moderate'
    elif expected_eq_after >= 3:
        estimated_feasibility = 'hard'
    else:
        estimated_feasibility = 'infeasible'
        warnings.append(
            f"Plan may not achieve feasibility. Expected EQ size: {expected_eq_after:.2f}. "
            f"Consider further aggregation or reducing QI count."
        )

    return PreprocessingPlan(
        original_qis=quasi_identifiers,
        actions=actions,
        estimated_feasibility=estimated_feasibility,
        combination_space_before=combination_space_before,
        combination_space_after=combination_space_after,
        expected_eq_size_before=expected_eq_before,
        expected_eq_size_after=expected_eq_after,
        warnings=warnings,
    )


def print_preprocessing_plan(plan: PreprocessingPlan) -> None:
    """Print formatted preprocessing plan."""
    print("\n" + "=" * 70)
    print("  PREPROCESSING PLAN")
    print("=" * 70)

    print(f"\n  Original QIs: {plan.original_qis}")
    print(f"  Estimated Feasibility: {plan.estimated_feasibility.upper()}")

    print(f"\n  Before Preprocessing:")
    print(f"    Combination space: {plan.combination_space_before:,}")
    print(f"    Expected EQ size: {plan.expected_eq_size_before:.2f}")

    print(f"\n  After Preprocessing (estimated):")
    print(f"    Combination space: {plan.combination_space_after:,}")
    print(f"    Expected EQ size: {plan.expected_eq_size_after:.2f}")

    print(f"\n  Actions:")
    for action in plan.actions:
        icon = "[KEEP]" if action.action == 'keep' else f"[{action.action.upper()}]"
        print(f"    {icon} {action.qi}:")
        print(f"        {action.reason}")
        if action.action != 'keep':
            print(f"        {action.original_cardinality} -> {action.estimated_cardinality} values")
            if action.params:
                print(f"        Params: {action.params}")

    if plan.warnings:
        print(f"\n  Warnings:")
        for w in plan.warnings:
            print(f"    [!] {w}")

    print("=" * 70)


def print_diagnosis_summary(diagnosis: DiagnosisResult) -> None:
    """Print a concise diagnosis summary."""
    status_icon = {
        FeasibilityStatus.FEASIBLE: "[OK]",
        FeasibilityStatus.HARD: "[!]",
        FeasibilityStatus.VERY_HARD: "[!!]"
    }

    print(f"\nQI Diagnosis: {diagnosis.status.value.upper()} {status_icon.get(diagnosis.status, '')}")
    print(f"  Expected EQ size: {diagnosis.expected_eq_size:.2f}")
    print(f"  Max achievable k: {diagnosis.max_achievable_k}")

    if diagnosis.problematic_qis:
        print(f"  Problematic QIs: {', '.join(diagnosis.problematic_qis)}")

    if diagnosis.preprocessing_needed:
        print(f"  Status: PREPROCESSING NEEDED")
        if diagnosis.recommendations:
            print(f"  Top recommendation: {diagnosis.recommendations[0]}")
    else:
        print(f"  Status: READY FOR ANONYMIZATION")


# =============================================================================
# ENSURE FEASIBILITY - Automatic fallback system
# =============================================================================

@dataclass
class FeasibilityResult:
    """Result of ensure_feasibility() with fallback tracking."""
    status: FeasibilityStatus
    final_qis: List[str]
    final_k: int
    recommendations: Dict[str, Dict]
    fallbacks_applied: List[str]
    warnings: List[str]
    original_qis: List[str]
    original_k: int


def ensure_feasibility(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    target_k: int = 5,
    access_tier: str = 'SCIENTIFIC',
    required_qis: Optional[List[str]] = None,
    available_hierarchies: Optional[Dict] = None,
    min_k: int = 3,
    max_fallback_iterations: int = 3,
    verbose: bool = True
) -> FeasibilityResult:
    """
    Ensure feasibility by applying preprocessing with automatic fallbacks.

    This function wraps the preprocessing workflow and automatically applies
    fallbacks if initial preprocessing doesn't achieve feasibility:

    1. Try standard preprocessing
    2. If not feasible: try_hard preprocessing
    3. If still not feasible: reduce QI count (respecting required_qis)
    4. If still not feasible: lower target_k (down to min_k)

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    quasi_identifiers : list
        Initial QI columns
    target_k : int
        Initial target k for k-anonymity (default 5)
    access_tier : str
        Target access tier
    required_qis : list, optional
        QIs that MUST be kept (never dropped even in fallbacks)
    available_hierarchies : dict, optional
        User-provided hierarchies
    min_k : int
        Minimum acceptable k (fallback won't go below this, default 3)
    max_fallback_iterations : int
        Maximum number of fallback attempts (default 3)
    verbose : bool
        Print progress

    Returns:
    --------
    FeasibilityResult : Contains final QIs, k, recommendations, and fallback log

    Example:
    --------
    >>> result = ensure_feasibility(
    ...     data, qis,
    ...     target_k=5,
    ...     required_qis=['age', 'sex'],
    ...     min_k=3
    ... )
    >>> if result.status == FeasibilityStatus.FEASIBLE:
    ...     # Use result.final_qis and result.final_k
    ...     pass
    >>> print(f"Fallbacks applied: {result.fallbacks_applied}")
    """
    required_set = set(required_qis) if required_qis else set()
    fallbacks_applied = []
    warnings = []
    current_k = target_k
    current_qis = list(quasi_identifiers)

    if verbose:
        print(f"\n[ensure_feasibility] Starting with {len(current_qis)} QIs, target_k={target_k}")

    # ==========================================================================
    # Step 1: Check initial feasibility
    # ==========================================================================
    status, msg, details = check_feasibility(data, current_qis, target_k=current_k, access_tier=access_tier)

    if status == FeasibilityStatus.FEASIBLE:
        if verbose:
            print(f"  [OK] Already feasible - no preprocessing needed")
        recs = {qi: {'action': 'keep', 'reason': 'Already feasible'} for qi in current_qis}
        return FeasibilityResult(
            status=FeasibilityStatus.FEASIBLE,
            final_qis=current_qis,
            final_k=current_k,
            recommendations=recs,
            fallbacks_applied=[],
            warnings=[],
            original_qis=list(quasi_identifiers),
            original_k=target_k
        )

    # ==========================================================================
    # Step 2: Try standard preprocessing (HARD case)
    # ==========================================================================
    if verbose:
        print(f"  [!] Status: {status.value} - trying standard preprocessing")

    recs = recommend_preprocessing(
        data, current_qis,
        access_tier=access_tier,
        required_qis=list(required_set),
        available_hierarchies=available_hierarchies,
        try_hard=False
    )

    # Check estimated feasibility after preprocessing
    estimated_card = _estimate_combination_space(recs)
    estimated_eq = len(data) / estimated_card if estimated_card > 0 else 0

    if estimated_eq >= current_k:
        if verbose:
            print(f"  [OK] Standard preprocessing achieves feasibility (EQ={estimated_eq:.1f})")
        kept_qis = [qi for qi, rec in recs.items() if rec['action'] != 'exclude']
        return FeasibilityResult(
            status=FeasibilityStatus.FEASIBLE,
            final_qis=kept_qis,
            final_k=current_k,
            recommendations=recs,
            fallbacks_applied=[],
            warnings=[],
            original_qis=list(quasi_identifiers),
            original_k=target_k
        )

    # ==========================================================================
    # Step 3: Try enhanced preprocessing (try_hard=True)
    # Uses stronger binning (5 bins) and fewer categories (10) to reduce cardinality
    # ==========================================================================
    fallbacks_applied.append("enhanced_preprocessing")
    if verbose:
        print(f"  [!] Standard preprocessing insufficient - trying enhanced mode (stronger binning)")

    recs = recommend_preprocessing(
        data, current_qis,
        access_tier=access_tier,
        required_qis=list(required_set),
        available_hierarchies=available_hierarchies,
        try_hard=True
    )

    estimated_card = _estimate_combination_space(recs)
    estimated_eq = len(data) / estimated_card if estimated_card > 0 else 0

    if estimated_eq >= current_k:
        if verbose:
            print(f"  [OK] Enhanced preprocessing achieves feasibility (EQ={estimated_eq:.1f})")
        kept_qis = [qi for qi, rec in recs.items() if rec['action'] != 'exclude']
        return FeasibilityResult(
            status=FeasibilityStatus.FEASIBLE,
            final_qis=kept_qis,
            final_k=current_k,
            recommendations=recs,
            fallbacks_applied=fallbacks_applied,
            warnings=[],
            original_qis=list(quasi_identifiers),
            original_k=target_k
        )

    # ==========================================================================
    # Step 4: Iterative fallbacks - reduce QIs and/or lower k
    # ==========================================================================
    for iteration in range(max_fallback_iterations):
        if verbose:
            print(f"  [!] Fallback iteration {iteration + 1}")

        # Get current kept QIs (non-required, sorted by priority)
        kept_qis = [qi for qi, rec in recs.items() if rec['action'] != 'exclude']
        non_required = [qi for qi in kept_qis if qi not in required_set]

        # Try reducing QI count first (if we have non-required QIs)
        if non_required and len(kept_qis) > len(required_set):
            # Sort by priority (lowest first) and exclude one
            sorted_by_priority = sorted(non_required, key=lambda q: recs[q].get('priority', 100))
            qi_to_drop = sorted_by_priority[0]

            recs[qi_to_drop]['action'] = 'exclude'
            recs[qi_to_drop]['reason'] = f"Excluded in fallback iteration {iteration + 1}"
            recs[qi_to_drop]['estimated_cardinality'] = 0

            fallbacks_applied.append(f"exclude:{qi_to_drop}")
            if verbose:
                print(f"    Excluding low-priority QI: {qi_to_drop}")

            # Re-check feasibility
            estimated_card = _estimate_combination_space(recs)
            estimated_eq = len(data) / estimated_card if estimated_card > 0 else 0

            if estimated_eq >= current_k:
                if verbose:
                    print(f"  [OK] Feasible after excluding {qi_to_drop} (EQ={estimated_eq:.1f})")
                kept_qis = [qi for qi, rec in recs.items() if rec['action'] != 'exclude']
                return FeasibilityResult(
                    status=FeasibilityStatus.FEASIBLE,
                    final_qis=kept_qis,
                    final_k=current_k,
                    recommendations=recs,
                    fallbacks_applied=fallbacks_applied,
                    warnings=warnings,
                    original_qis=list(quasi_identifiers),
                    original_k=target_k
                )

        # If can't reduce QIs further, try lowering k
        elif current_k > min_k:
            new_k = max(min_k, current_k - 1)
            fallbacks_applied.append(f"lower_k:{current_k}->{new_k}")
            warnings.append(f"Target k reduced from {current_k} to {new_k}")
            if verbose:
                print(f"    Lowering target k: {current_k} -> {new_k}")
            current_k = new_k

            # Re-check with lower k
            estimated_card = _estimate_combination_space(recs)
            estimated_eq = len(data) / estimated_card if estimated_card > 0 else 0

            if estimated_eq >= current_k:
                if verbose:
                    print(f"  [OK] Feasible with k={current_k} (EQ={estimated_eq:.1f})")
                kept_qis = [qi for qi, rec in recs.items() if rec['action'] != 'exclude']
                return FeasibilityResult(
                    status=FeasibilityStatus.FEASIBLE,
                    final_qis=kept_qis,
                    final_k=current_k,
                    recommendations=recs,
                    fallbacks_applied=fallbacks_applied,
                    warnings=warnings,
                    original_qis=list(quasi_identifiers),
                    original_k=target_k
                )
        else:
            # Can't reduce further
            break

    # ==========================================================================
    # Exhausted fallbacks - return best effort
    # ==========================================================================
    kept_qis = [qi for qi, rec in recs.items() if rec['action'] != 'exclude']
    warnings.append("Exhausted all fallbacks - feasibility not guaranteed")
    if verbose:
        print(f"  [!!] Exhausted fallbacks - returning best effort ({len(kept_qis)} QIs, k={current_k})")

    return FeasibilityResult(
        status=FeasibilityStatus.VERY_HARD,
        final_qis=kept_qis,
        final_k=current_k,
        recommendations=recs,
        fallbacks_applied=fallbacks_applied,
        warnings=warnings,
        original_qis=list(quasi_identifiers),
        original_k=target_k
    )


def _estimate_combination_space(recommendations: Dict[str, Dict]) -> int:
    """Estimate combination space from recommendations."""
    space = 1
    for qi, rec in recommendations.items():
        if rec['action'] != 'exclude':
            card = rec.get('estimated_cardinality', rec.get('original_cardinality', 1))
            space *= max(1, card)
    return space
