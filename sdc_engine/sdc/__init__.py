"""
SDC Methods Package
===================

Statistical Disclosure Control methods for data anonymization.

Modules:
- config: Centralized configuration and constants
- detection: QI detection, column types, data analysis
- metrics: ReID, risk, and utility calculations
- selection: Method selection rules and pipelines

Available methods (microdata-only):
- kANON: k-Anonymity for microdata
- PRAM: Post-RAndomization Method for categorical variables
- NOISE: Random noise addition for continuous variables
- LOCSUPR: Local Suppression for k-anonymity

Usage:
    from sdc_engine.sdc import apply_kanon, apply_pram, apply_noise, apply_locsupr
    from sdc_engine.sdc.metrics import calculate_reid, calculate_utility_metrics
    from sdc_engine.sdc.selection import select_method_by_features
"""

# =============================================================================
# R environment setup removed (r_setup.py archived — not needed for Panel app)
# =============================================================================

# =============================================================================
# SDC Methods
# =============================================================================
from .kANON import apply_kanon
from .PRAM import apply_pram
from .NOISE import apply_noise
from .LOCSUPR import apply_locsupr
from .GENERALIZE import apply_generalize, suggest_generalization
from .RANKSWAP import apply_rankswap
from .RECSWAP import apply_recswap

# =============================================================================
# Synthetic Data Release (archived — not exposed in UI)
# =============================================================================
HAS_SYNTHETIC = False
HAS_SYNTHCITY = False

# =============================================================================
# SDC Preprocessing (Greek-specific + standard preprocessing)
# =============================================================================
from .sdc_preprocessing import (
    preprocess_for_sdc,
    detect_greek_identifiers,
    remove_direct_identifiers,
    apply_top_bottom_coding,
    merge_rare_categories,
    detect_reidentification_outliers,
    assess_dimensionality_risk,
    generate_pre_anonymization_report,
    DEFAULT_PREPROCESSING_RULES,
    # New preprocessing functions
    apply_numeric_rounding,
    apply_date_truncation,
    apply_age_binning,
    apply_geographic_coarsening,
    apply_string_truncation,
    apply_record_sampling,
)

# =============================================================================
# Legacy imports from sdc_utils (for backward compatibility)
# =============================================================================
from .sdc_utils import (
    # Detection (also available from src.detection)
    auto_detect_dimensions,
    auto_detect_quasi_identifiers,
    detect_quasi_identifiers_enhanced,
    auto_detect_continuous_variables,
    auto_detect_categorical_variables,
    auto_detect_sensitive_columns,
    detect_data_type,
    identify_column_types,
    get_data_summary,
    # Risk metrics (also available from src.metrics)
    check_kanonymity,
    calculate_uniqueness_rate,
    find_rare_combinations,
    calculate_disclosure_risk,
    calculate_reid,
    assess_risk_with_reid,
    # Utilities
    aggregate_to_table,
    check_small_cells,
    generalize_numeric,
    generalize_categorical,
    generalize_string_prefix,
    calculate_information_loss,
    validate_quasi_identifiers,
    validate_numeric_parameter,
    # Analysis
    analyze_data,
    # Method info
    METHOD_INFO,
)

# =============================================================================
# Method selection (canonical — from selection.rules)
# =============================================================================
from .selection import select_method_by_features

# =============================================================================
# New modular exports
# =============================================================================
# Config module
from .config import (
    METHOD_IMPLEMENTATION_QUALITY,
    MICRODATA_METHODS,
    TABULAR_METHODS,
    DIRECT_IDENTIFIER_KEYWORDS,
    DIRECT_IDENTIFIER_PATTERNS,
    QI_KEYWORDS,
    PROTECTION_THRESHOLDS,
    DEFAULT_METHOD_PARAMETERS,
    PARAMETER_TUNING_SCHEDULES,
    METHOD_FALLBACK_ORDER,
    get_method_defaults,
    get_tuning_schedule,
    get_method_fallbacks,
    get_protection_thresholds,
)

__all__ = [
    # SDC Methods (microdata-only)
    'apply_kanon',
    'apply_pram',
    'apply_noise',
    'apply_locsupr',
    'apply_generalize',
    'suggest_generalization',
    'apply_rankswap',
    'apply_recswap',
    # Detection utilities
    'auto_detect_dimensions',
    'auto_detect_quasi_identifiers',
    'detect_quasi_identifiers_enhanced',
    'auto_detect_continuous_variables',
    'auto_detect_categorical_variables',
    'auto_detect_sensitive_columns',
    'detect_data_type',
    'identify_column_types',
    'get_data_summary',
    'analyze_data',
    # Risk metrics
    'check_kanonymity',
    'calculate_uniqueness_rate',
    'find_rare_combinations',
    'calculate_disclosure_risk',
    'calculate_reid',
    'assess_risk_with_reid',
    # Utilities
    'aggregate_to_table',
    'check_small_cells',
    'generalize_numeric',
    'generalize_categorical',
    'generalize_string_prefix',
    'calculate_information_loss',
    'validate_quasi_identifiers',
    'validate_numeric_parameter',
    # Config
    'METHOD_INFO',
    'METHOD_IMPLEMENTATION_QUALITY',
    'MICRODATA_METHODS',
    'TABULAR_METHODS',
    'DIRECT_IDENTIFIER_KEYWORDS',
    'DIRECT_IDENTIFIER_PATTERNS',
    'QI_KEYWORDS',
    'PROTECTION_THRESHOLDS',
    'DEFAULT_METHOD_PARAMETERS',
    'get_method_defaults',
    'get_tuning_schedule',
    'get_method_fallbacks',
    'get_protection_thresholds',
    # Preprocessing
    'preprocess_for_sdc',
    'detect_greek_identifiers',
    'remove_direct_identifiers',
    'apply_top_bottom_coding',
    'merge_rare_categories',
    'detect_reidentification_outliers',
    'assess_dimensionality_risk',
    'generate_pre_anonymization_report',
    'DEFAULT_PREPROCESSING_RULES',
    # New preprocessing functions
    'apply_numeric_rounding',
    'apply_date_truncation',
    'apply_age_binning',
    'apply_geographic_coarsening',
    'apply_string_truncation',
    'apply_record_sampling',
    # Method selection (canonical)
    'select_method_by_features',
    # Synthetic data release (optional)
    'HAS_SYNTHCITY',
]
