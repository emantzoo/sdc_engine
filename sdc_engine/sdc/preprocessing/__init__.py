"""
Preprocessing Module
====================

Tools for preprocessing data before applying SDC methods.

WORKFLOW:
=========

SIMPLE (Recommended) - Use ensure_feasibility():
```
from src.preprocessing import ensure_feasibility, FeasibilityStatus

# One function handles everything with automatic fallbacks
result = ensure_feasibility(
    data, qis,
    target_k=5,
    required_qis=['age', 'sex'],  # Never drop these
    min_k=3                        # Fallback lower limit
)

# Result contains:
# - result.final_qis: QIs to use (after any exclusions)
# - result.final_k: k to use (may be lowered if needed)
# - result.recommendations: Preprocessing actions per QI
# - result.fallbacks_applied: What fallbacks were used
# - result.warnings: Any warnings to show user

if result.status == FeasibilityStatus.FEASIBLE:
    # Proceed with result.final_qis and result.final_k
    pass
```

DETAILED (Manual control):
```
# 1. Check feasibility
status, msg, details = check_feasibility(data, qis, target_k=5)

# 2. Get recommendations
if status != FeasibilityStatus.FEASIBLE:
    recs = recommend_preprocessing(
        data, qis,
        required_qis=['age', 'sex'],
        try_hard=(status == FeasibilityStatus.VERY_HARD)
    )
    plan = create_preprocessing_plan(data, qis, recs)
    print_preprocessing_plan(plan)

# 3. Apply preprocessing
processed_data, processed_qis, result = preprocess_for_anonymization(
    data, qis, auto_preprocess=True
)

# 4. Method selection
from src.selection import select_method_suite, extract_data_features_with_reid
features = extract_data_features_with_reid(processed_data, analysis, processed_qis)
suite = select_method_suite(features, access_tier='PUBLIC')
```

Fallback Strategy (ensure_feasibility):
---------------------------------------
1. Try standard preprocessing
2. If not feasible: try_hard=True (aggressive binning/top_k)
3. If still not feasible: exclude low-priority QIs (respecting required_qis)
4. If still not feasible: lower target_k (down to min_k)
5. Always returns a result - never fails

Two Sub-modules:
----------------
1. DIAGNOSIS (diagnose.py) - Feasibility checking and recommendations
2. QI HANDLER (qi_handler.py) - Apply preprocessing transformations
"""

# Diagnosis module - feasibility checking
from .diagnose import (
    # Main functions
    check_feasibility,
    diagnose_qis,
    recommend_preprocessing,
    create_preprocessing_plan,
    print_preprocessing_plan,
    print_diagnosis_summary,
    ensure_feasibility,
    # Classes
    FeasibilityStatus,
    DiagnosisResult,
    PreprocessingAction,
    PreprocessingPlan,
    FeasibilityResult,
    # Constants
    FEASIBLE_CRITERIA,
    HARD_CRITERIA,
    VERY_HARD_PATTERNS,
    QI_PRIORITY_CONFIG,
    HIERARCHY_PATTERNS,
    DETECTION_CONFIG,
    UTILITY_DEFAULTS,
)

# QI Handler - apply preprocessing
from .qi_handler import (
    QIHandler,
    PreprocessingStrategy,
    PreprocessingPlan as QIPreprocessingPlan,  # Renamed to avoid conflict
    QIAnalysis,
    AccessTier,
    TIER_CONSTRAINTS,
    COMMON_HIERARCHIES,
    preprocess_for_anonymization,
)

# Backward compatibility aliases
TIER_THRESHOLDS = {
    'PUBLIC': FEASIBLE_CRITERIA.copy(),
    'SCIENTIFIC': FEASIBLE_CRITERIA.copy(),
    'SECURE': FEASIBLE_CRITERIA.copy(),
}

__all__ = [
    # Diagnosis - Main functions
    'check_feasibility',
    'diagnose_qis',
    'recommend_preprocessing',
    'create_preprocessing_plan',
    'print_preprocessing_plan',
    'print_diagnosis_summary',
    'ensure_feasibility',
    # Diagnosis - Classes
    'FeasibilityStatus',
    'DiagnosisResult',
    'PreprocessingAction',
    'PreprocessingPlan',
    'FeasibilityResult',
    # Diagnosis - Constants
    'FEASIBLE_CRITERIA',
    'HARD_CRITERIA',
    'VERY_HARD_PATTERNS',
    'QI_PRIORITY_CONFIG',
    'HIERARCHY_PATTERNS',
    'DETECTION_CONFIG',
    'UTILITY_DEFAULTS',
    # QI Handler
    'QIHandler',
    'PreprocessingStrategy',
    'QIPreprocessingPlan',
    'QIAnalysis',
    'AccessTier',
    'TIER_CONSTRAINTS',
    'COMMON_HIERARCHIES',
    'preprocess_for_anonymization',
    # Backward compatibility
    'TIER_THRESHOLDS',
]
