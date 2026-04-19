"""
SDC Metrics Module
==================

Risk assessment and utility calculation for SDC methods.

Components:
- reid: Re-identification risk calculations
- risk: Disclosure risk metrics
- utility: Utility preservation metrics
"""

from .reid import calculate_reid, assess_risk_with_reid, classify_risk_pattern
from .risk import (
    calculate_disclosure_risk,
    check_kanonymity,
    calculate_uniqueness_rate,
    find_rare_combinations,
    assess_risk_level,
)
from .risk_metric import (
    RiskMetricType,
    RiskAssessment,
    normalize_to_risk_score,
    normalize_target,
    compute_risk,
    risk_to_reid_compat,
    get_metric_display_info,
)
from .utility import calculate_utility_metrics, calculate_information_loss
from .ml_utility import compute_ml_utility, compute_ml_utility_multi
from .pareto import pareto_optimal, risk_reduction

__all__ = [
    # ReID risk metrics
    'calculate_reid',
    'assess_risk_with_reid',
    'classify_risk_pattern',
    # Risk metrics
    'calculate_disclosure_risk',
    'check_kanonymity',
    'calculate_uniqueness_rate',
    'find_rare_combinations',
    'assess_risk_level',
    # Pluggable risk metric
    'RiskMetricType',
    'RiskAssessment',
    'normalize_to_risk_score',
    'normalize_target',
    'compute_risk',
    'risk_to_reid_compat',
    'get_metric_display_info',
    # Utility metrics
    'calculate_utility_metrics',
    'calculate_information_loss',
    # ML utility validation
    'compute_ml_utility',
    'compute_ml_utility_multi',
    # Pareto analysis
    'pareto_optimal',
    'risk_reduction',
]
