"""
Pluggable Risk Metric Abstraction
==================================

Normalizes REID95, k-anonymity, uniqueness rate, and l-diversity to a
common 0-1 "risk score" (higher = riskier) so that all decision rules
operate on a single scale without per-metric branching.

Normalization table:
    REID95      → passthrough (already 0-1, higher=riskier)
    k-anonymity → 1/k  (k=1 → 1.0, k=5 → 0.20, k=20 → 0.05)
    Uniqueness  → passthrough (already 0-1, higher=riskier)
    l-diversity → 1/l  (l=1 → 1.0, l=2 → 0.50, l=5 → 0.20)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class RiskMetricType(Enum):
    REID95 = 'reid95'
    K_ANONYMITY = 'k_anonymity'
    UNIQUENESS = 'uniqueness'
    L_DIVERSITY = 'l_diversity'


# Display labels used throughout the UI
_LABELS = {
    RiskMetricType.REID95: 'Re-id risk',
    RiskMetricType.K_ANONYMITY: 'k-Anonymity',
    RiskMetricType.UNIQUENESS: 'Unique records',
    RiskMetricType.L_DIVERSITY: 'l-Diversity',
}

# Default targets per metric (used when no context target supplied)
_DEFAULT_TARGETS = {
    RiskMetricType.REID95: 0.05,
    RiskMetricType.K_ANONYMITY: 5,
    RiskMetricType.UNIQUENESS: 0.05,
    RiskMetricType.L_DIVERSITY: 2,
}


@dataclass
class RiskAssessment:
    """Result of a risk computation — carries both raw and normalized values."""

    metric_type: RiskMetricType
    raw_value: float          # Native metric value (reid95 float, min_k int, uniqueness float)
    normalized_score: float   # 0-1, higher = riskier — THE decision variable
    target_raw: float         # Target in native units
    target_normalized: float  # Target on the normalized 0-1 scale
    meets_target: bool
    details: Dict = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        return _LABELS[self.metric_type]

    @property
    def display_value(self) -> str:
        if self.metric_type == RiskMetricType.K_ANONYMITY:
            return f'k = {int(self.raw_value)}'
        if self.metric_type == RiskMetricType.L_DIVERSITY:
            return f'l = {int(self.raw_value)}'
        return f'{self.raw_value:.4f}'

    @property
    def display_target(self) -> str:
        if self.metric_type == RiskMetricType.K_ANONYMITY:
            return f'k \u2265 {int(self.target_raw)}'
        if self.metric_type == RiskMetricType.L_DIVERSITY:
            return f'l \u2265 {int(self.target_raw)}'
        return f'\u2264 {self.target_raw:.2%}'

    @property
    def display_value_short(self) -> str:
        """Short value for metric cards (e.g. '5.0%' or 'k=5')."""
        if self.metric_type == RiskMetricType.K_ANONYMITY:
            return f'k={int(self.raw_value)}'
        if self.metric_type == RiskMetricType.L_DIVERSITY:
            return f'l={int(self.raw_value)}'
        return f'{self.raw_value:.1%}'


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_to_risk_score(metric_type: RiskMetricType, raw_value: float) -> float:
    """Convert a raw metric value to normalized 0-1 risk score (higher=riskier)."""
    if metric_type == RiskMetricType.REID95:
        return float(np.clip(raw_value, 0, 1))
    elif metric_type == RiskMetricType.K_ANONYMITY:
        k = max(raw_value, 1)
        return float(np.clip(1.0 / k, 0, 1))
    elif metric_type == RiskMetricType.UNIQUENESS:
        return float(np.clip(raw_value, 0, 1))
    elif metric_type == RiskMetricType.L_DIVERSITY:
        l = max(raw_value, 1)
        return float(np.clip(1.0 / l, 0, 1))
    raise ValueError(f"Unknown metric type: {metric_type}")


def normalize_target(metric_type: RiskMetricType, target_raw: float) -> float:
    """Convert a raw target to the normalized risk-score scale."""
    if metric_type == RiskMetricType.REID95:
        return float(target_raw)
    elif metric_type == RiskMetricType.K_ANONYMITY:
        return float(1.0 / max(target_raw, 1))
    elif metric_type == RiskMetricType.UNIQUENESS:
        return float(target_raw)
    elif metric_type == RiskMetricType.L_DIVERSITY:
        return float(1.0 / max(target_raw, 1))
    raise ValueError(f"Unknown metric type: {metric_type}")


# ---------------------------------------------------------------------------
# Unified risk computation
# ---------------------------------------------------------------------------

def compute_risk(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    metric_type: RiskMetricType = RiskMetricType.REID95,
    target_raw: Optional[float] = None,
    *,
    sensitive_columns: Optional[List[str]] = None,
) -> RiskAssessment:
    """Compute risk using the chosen metric and return a normalized assessment.

    This is the single entry point that replaces direct calls to
    ``calculate_reid()`` as a decision variable.

    Parameters
    ----------
    sensitive_columns : list of str, optional
        Required for L_DIVERSITY metric. Ignored by other metrics.
    """
    from .reid import calculate_reid
    from .risk import check_kanonymity, calculate_uniqueness_rate

    if target_raw is None:
        target_raw = _DEFAULT_TARGETS[metric_type]

    if metric_type == RiskMetricType.REID95:
        reid = calculate_reid(data, quasi_identifiers)
        raw = reid.get('reid_95', 0.0)
        details = reid

    elif metric_type == RiskMetricType.K_ANONYMITY:
        _is_k, group_sizes, _violations = check_kanonymity(
            data, quasi_identifiers, k=1,
        )
        col = 'count' if 'count' in group_sizes.columns else '_group_size_'
        if len(group_sizes) > 0:
            min_k = int(group_sizes[col].min())
            mean_gs = float(group_sizes[col].mean())
            median_gs = float(group_sizes[col].median())
        else:
            min_k, mean_gs, median_gs = 0, 0.0, 0.0
        raw = float(min_k)
        details = {
            'min_k': min_k,
            'is_k_anonymous_at_target': min_k >= target_raw,
            'n_groups': len(group_sizes),
            'mean_group_size': mean_gs,
            'median_group_size': median_gs,
        }

    elif metric_type == RiskMetricType.UNIQUENESS:
        raw = float(calculate_uniqueness_rate(data, quasi_identifiers))
        details = {'uniqueness_rate': raw}

    elif metric_type == RiskMetricType.L_DIVERSITY:
        from ..post_protection_diagnostics import check_l_diversity
        l_result = check_l_diversity(
            data, quasi_identifiers,
            sensitive_columns or [],
            l_target=int(target_raw or 2),
            size_threshold=200,  # higher than diagnostic default for gate use
        )
        raw = float(l_result.get('l_achieved') or 1)
        details = l_result

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    normalized = normalize_to_risk_score(metric_type, raw)
    target_norm = normalize_target(metric_type, target_raw)

    return RiskAssessment(
        metric_type=metric_type,
        raw_value=raw,
        normalized_score=normalized,
        target_raw=target_raw,
        target_normalized=target_norm,
        meets_target=normalized <= target_norm,
        details=details,
    )


# ---------------------------------------------------------------------------
# Backward-compatibility bridge
# ---------------------------------------------------------------------------

def get_metric_display_info(risk_metric: str, target_raw: float = None) -> Dict:
    """Get display information for a risk metric.

    Returns dict with 'label', 'target_display', 'card_label', 'metric_type'.
    """
    try:
        _metric_str = risk_metric
        mt = RiskMetricType(_metric_str)
    except ValueError:
        mt = RiskMetricType.REID95

    if target_raw is None:
        target_raw = _DEFAULT_TARGETS[mt]

    label = _LABELS[mt]

    if mt == RiskMetricType.K_ANONYMITY:
        target_display = f'k \u2265 {int(target_raw)}'
        card_label = f'k-Anonymity (target: k \u2265 {int(target_raw)})'
    elif mt == RiskMetricType.UNIQUENESS:
        target_display = f'\u2264 {target_raw:.0%}'
        card_label = f'Uniqueness Rate (target: \u2264 {target_raw:.0%})'
    elif mt == RiskMetricType.L_DIVERSITY:
        target_display = f'l \u2265 {int(target_raw)}'
        card_label = f'l-Diversity (target: l \u2265 {int(target_raw)})'
    else:  # REID95
        target_display = f'\u2264 {target_raw:.0%}'
        card_label = f'ReID95 (target: \u2264 {target_raw:.0%})'

    return {
        'label': label,
        'target_display': target_display,
        'card_label': card_label,
        'metric_type': mt,
    }


def risk_to_reid_compat(assessment: RiskAssessment) -> Dict:
    """Produce a dict shaped like ``calculate_reid()`` output.

    The key insight: decision rules read ``features['reid_95']``.  By mapping
    the normalized risk score INTO that key, all rules work unchanged.

    For REID95 the real dict is returned as-is.  For k-anonymity and
    uniqueness a synthetic dict is constructed.
    """
    if assessment.metric_type == RiskMetricType.REID95:
        return assessment.details  # already a full reid dict

    ns = assessment.normalized_score
    return {
        'reid_50': ns * 0.5,
        'reid_90': ns * 0.8,
        'reid_95': ns,
        'reid_99': min(1.0, ns * 1.2),
        'mean_risk': ns * 0.6,
        'max_risk': min(1.0, ns * 1.5),
        'high_risk_count': 0,
        'high_risk_rate': min(ns * 0.5, 1.0) if ns > 0.20 else 0.0,  # rough estimate for synthetic reid
        'suppressed_records': 0,
        'suppression_rate': 0.0,
        'records_evaluated': len(getattr(assessment, '_data', [])),
        '_synthetic': True,
        '_source_metric': assessment.metric_type.value,
        '_raw_value': assessment.raw_value,
    }

