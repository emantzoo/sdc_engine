"""
SDC Method Selection Module
===========================

Data-driven method selection using ReID metrics and rule-based pipelines.

Main Entry Point:
-----------------
- select_method_suite(): Get prioritized method list with fallbacks

Components:
-----------
- features: Feature extraction with ReID metrics
- rules: Selection rules (data structure, ReID risk, distribution)
- pipelines: Multi-method pipeline selection

Typical Workflow:
-----------------
1. Extract features: features = extract_data_features_with_reid(data, analysis, qis)
2. Get method suite: suite = select_method_suite(features, access_tier='PUBLIC')
3. Try primary: result = apply_method(data, suite['primary'], suite['primary_params'])
4. If fails, try fallbacks: for method, params in suite['fallbacks']: ...

Rule Priority:
--------------
1. Pipeline Rules (dynamic + legacy P4/P5) - Multi-method pipelines
2. Data Structure Rules - Tabular format detection
3. Risk Concentration Rules (RC1-RC3) - Per-QI risk contribution
4. Categorical-Aware Rules (CAT1-CAT2) - PRAM for categorical at moderate risk
5. ReID Risk Rules (QR0-QR4 + MED1) - Risk distribution patterns
6. Low-Risk Rules (LOW1-LOW3) - Type-based at ReID_95 <= 20%
7. Distribution Rules (DP1-DP3) - Outliers, skewness
8. Uniqueness Risk Rules (HR1-HR5) - Heuristic fallback when no ReID
9. Default Rules - Final fallbacks
"""

from .features import extract_data_features_with_reid
from .rules import (
    data_structure_rules,
    categorical_aware_rules,
    low_risk_rules,
    low_risk_structure_rules,
    microdata_structure_rules,  # Alias for backward compatibility
    reid_risk_rules,
    uniqueness_risk_rules,
    distribution_rules,
    default_rules,
    select_method_by_features,
    public_release_rules,
    secure_environment_rules,
    regulatory_compliance_rules,
)
from .pipelines import (
    build_dynamic_pipeline,
    pipeline_rules,
    select_method_suite,
)

# Import ReID pattern classifier from metrics
try:
    from ..metrics.reid import classify_risk_pattern
except ImportError:
    classify_risk_pattern = None

__all__ = [
    # Main entry point
    'select_method_suite',
    # Feature extraction
    'extract_data_features_with_reid',
    'classify_risk_pattern',
    # Selection rules
    'data_structure_rules',
    'categorical_aware_rules',
    'low_risk_rules',
    'low_risk_structure_rules',
    'microdata_structure_rules',
    'reid_risk_rules',
    'uniqueness_risk_rules',
    'distribution_rules',
    'default_rules',
    'select_method_by_features',
    # Context-aware rules
    'public_release_rules',
    'secure_environment_rules',
    'regulatory_compliance_rules',
    # Pipeline rules
    'build_dynamic_pipeline',
    'pipeline_rules',
]
