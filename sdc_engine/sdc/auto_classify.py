"""
Auto-Classification Engine for SDC Column Roles
=================================================

Orchestrates existing detection modules into a unified classification pipeline.
Each column gets: role (Identifier/QI/Sensitive/Unassigned), confidence (High/Medium/Low),
a numeric score, and a human-readable reason.

Key design:
- Reuses detect_quasi_identifiers_enhanced() for keyword+type+uniqueness QI scoring
- Fuses keyword scores with backward-elimination risk contribution (dual-signal)
- DEFINITE keyword floor: obvious QIs stay QI even with low empirical risk
- Genuinely new: suggest_sensitive_columns() — nothing like it existed in the codebase

Usage:
    from sdc_engine.sdc.auto_classify import auto_classify

    result = auto_classify(data, var_priority=var_priority)
    # result['age'] == {'role': 'QI', 'confidence': 'High', ...}
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple

from sdc_engine.sdc.config import (
    SENSITIVE_VALUE_KEYWORDS,
    SENSITIVE_VALUE_KEYWORDS_GR,
    ADMIN_KEYWORDS,
)
from sdc_engine.sdc.detection.qi_detection import detect_quasi_identifiers_enhanced
from sdc_engine.sdc.detection.column_types import (
    auto_detect_direct_identifiers,
    identify_column_types,
)

logger = logging.getLogger(__name__)

# Merge EN + GR keywords once at import time
_ALL_SENSITIVE_KW: Dict[str, float] = {
    **SENSITIVE_VALUE_KEYWORDS,
    **SENSITIVE_VALUE_KEYWORDS_GR,
}

# Domain booster patterns — column-name substrings that increase QI confidence
_DATE_HINTS = [
    'date', 'ημερομηνια', 'ημερομηνία', 'ετος', 'έτος', 'temporal',
    'time', 'period', 'year', 'month', 'quarter',
]
_GEO_HINTS = [
    'zip', 'postal', 'postcode', 'city', 'town', 'county', 'state',
    'province', 'region', 'municipality', 'δημος', 'δήμος',
    'νομαρχια', 'νομαρχία', 'περιφερ', 'κοινοτ', 'geographic',
    'country', 'address', 'location', 'district',
]
_DEMO_HINTS = [
    'age', 'gender', 'sex', 'race', 'ethnic', 'education', 'marital',
    'occupation', 'profession', 'job', 'nationality', 'religion',
    'ηλικια', 'ηλικία', 'φυλο', 'φύλο', 'εκπαιδευση', 'εκπαίδευση',
    'επαγγελμα', 'επάγγελμα',
]

# Ratio/percentage name hints
_RATIO_HINTS = [
    '%', 'pct', 'percent', 'ratio', 'rate', 'proportion', 'share',
    'ποσοστο', 'ποσοστό',
]

# Lowercase admin keywords for fast lookup
_ADMIN_KW_SET = {kw.lower() for kw in ADMIN_KEYWORDS}


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def auto_classify(
    data: pd.DataFrame,
    var_priority: Dict[str, Tuple[str, float]],
    data_type_labels: Optional[Dict[str, str]] = None,
    detected_direct_ids: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """
    Classify every column as Identifier / QI / Sensitive / Unassigned.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to classify.
    var_priority : dict
        Backward-elimination results from risk computation.
        ``{col: (priority_label, contribution_pct)}``
        e.g. ``{'age': ('🔴 HIGH', 22.5), 'sex': ('⚪ LOW', 1.3)}``.
    data_type_labels : dict, optional
        Semantic type labels from Configure table ``{col: 'Integer — Age (demographic)'}``.
        Used for domain boosters.  Falls back to ``identify_column_types()`` if None.
    detected_direct_ids : dict, optional
        Pre-computed direct-identifier detection ``{col: id_type}``.
        If None, ``auto_detect_direct_identifiers()`` is called.

    Returns
    -------
    dict
        ``{column_name: {role, confidence, confidence_score, reason, warnings: list}}``
        Plus a special ``'_diagnostics'`` key with cross-column warnings.
    """
    result: Dict[str, Dict] = {}
    diagnostics: List[str] = []  # cross-column warnings

    # ------------------------------------------------------------------
    # Step 0: Gather baseline signals
    # ------------------------------------------------------------------
    if detected_direct_ids is None:
        detected_direct_ids = auto_detect_direct_identifiers(data, check_patterns=True)

    # Try Greek identifier detection (optional import — may not be present)
    greek_ids: Dict[str, str] = {}
    try:
        from sdc_engine.sdc.sdc_preprocessing import detect_greek_identifiers
        greek_ids = detect_greek_identifiers(data) or {}
    except Exception:
        pass

    all_identifiers = {**detected_direct_ids, **greek_ids}

    # Column types (structural: continuous / categorical / identifier / binary)
    column_types = identify_column_types(data)

    # QI keyword scores — per-column {confidence, tier, reasons, signals}
    qi_scores = detect_quasi_identifiers_enhanced(data, return_scores=True)

    # Build a contribution lookup (col → pct) from var_priority
    contribution_pct: Dict[str, float] = {}
    for col, (_, pct) in (var_priority or {}).items():
        contribution_pct[col] = float(pct)

    max_contribution = max(contribution_pct.values()) if contribution_pct else 1.0
    if max_contribution <= 0:
        max_contribution = 1.0

    # ------------------------------------------------------------------
    # Step 0b: Pre-scan — near-constant & high-missingness detection
    # ------------------------------------------------------------------
    near_constant_cols: Set[str] = set()
    high_missing_cols: Set[str] = set()
    n_rows = len(data)

    for col in data.columns:
        series = data[col]
        miss_pct = series.isna().sum() / n_rows if n_rows > 0 else 0

        if miss_pct > 0.50:
            high_missing_cols.add(col)

        # Near-constant: one value accounts for >= 95% of non-null values
        non_null = series.dropna()
        if len(non_null) > 0:
            top_freq = non_null.value_counts(normalize=True).iloc[0]
            if top_freq >= 0.95:
                near_constant_cols.add(col)
                top_val = non_null.value_counts().index[0]
                result[col] = {
                    'role': 'Unassigned',
                    'confidence': 'High',
                    'confidence_score': 0.0,
                    'reason': (f"Near-constant ({top_freq:.0%} of values are "
                               f"'{top_val}'). Low analytical and re-identification value."),
                    'warnings': [],
                }

    # ------------------------------------------------------------------
    # Step 1: Classify identifiers
    # ------------------------------------------------------------------
    identifier_cols: Set[str] = set()
    for col in data.columns:
        if col in near_constant_cols:
            continue  # already classified
        if col in all_identifiers:
            id_type = all_identifiers[col]
            warnings = []
            if col in high_missing_cols:
                miss_pct = data[col].isna().sum() / n_rows
                warnings.append(f"{miss_pct:.0%} missing values")
            result[col] = {
                'role': 'Identifier',
                'confidence': 'High',
                'confidence_score': 1.0,
                'reason': f"Direct identifier detected ({id_type})",
                'warnings': warnings,
            }
            identifier_cols.add(col)

    # ------------------------------------------------------------------
    # Step 1a: Auto-tag near-unique columns as Identifier
    # ------------------------------------------------------------------
    # Columns with very high uniqueness (>50% unique or >5000 unique values)
    # that were dropped from risk scan are likely identifiers (LEIs, ISINs,
    # free-text IDs, timestamps with microsecond precision).
    _NEAR_UNIQUE_RATIO = 0.50
    _NEAR_UNIQUE_ABS = 5000
    for col in data.columns:
        if col in result or col in identifier_cols or col in near_constant_cols:
            continue
        _nu = data[col].nunique()
        _ratio = _nu / n_rows if n_rows > 0 else 0
        if _ratio > _NEAR_UNIQUE_RATIO or _nu > _NEAR_UNIQUE_ABS:
            warnings = []
            if col in high_missing_cols:
                miss_pct = data[col].isna().sum() / n_rows
                warnings.append(f"{miss_pct:.0%} missing values")
            result[col] = {
                'role': 'Identifier',
                'confidence': 'Medium',
                'confidence_score': 0.80,
                'reason': (f"Near-unique column ({_nu:,} unique in {n_rows:,} rows, "
                           f"{_ratio:.0%}). Too many distinct values for meaningful "
                           f"generalization — likely an identifier or free-text field."),
                'warnings': warnings,
            }
            identifier_cols.add(col)
            logger.info("Auto-classify: %s tagged as Identifier "
                        "(near-unique: %d unique, ratio=%.2f)", col, _nu, _ratio)

    # ------------------------------------------------------------------
    # Step 1b: Handle concentrated risk
    # ------------------------------------------------------------------
    # When backward elimination is dominated by one or two columns (identifiers
    # or high-cardinality numerics absorbing ≥95% of risk), all other columns
    # show ~0% contribution — making risk contribution useless for QI detection.
    # In that case, null out the contribution so fuse_qi_signals falls back to
    # keyword-only scoring.
    _risk_concentrated = False
    if contribution_pct:
        sorted_contribs = sorted(contribution_pct.values(), reverse=True)
        # Check if top-2 columns absorb ≥95% of total risk
        top2_pct = sum(sorted_contribs[:2])
        remaining_max = max(sorted_contribs[2:]) if len(sorted_contribs) > 2 else 0
        if top2_pct >= 95 and remaining_max < 1.0:
            _risk_concentrated = True
            logger.info(
                "Auto-classify: concentrated risk (top-2 columns absorb %.1f%%, "
                "remaining max %.1f%%). Falling back to keyword-only QI scoring.",
                top2_pct, remaining_max,
            )

    # ------------------------------------------------------------------
    # Step 2: Classify QIs (dual-signal fusion)
    # ------------------------------------------------------------------
    qi_cols: Set[str] = set()
    for col in data.columns:
        if col in identifier_cols or col in near_constant_cols:
            continue

        kw_info = qi_scores.get(col)
        if kw_info is None:
            continue

        # When risk is concentrated in 1-2 columns, pass None so fusion
        # uses keyword-only scoring for remaining columns
        rc_pct = None if _risk_concentrated else contribution_pct.get(col)
        dt_label = (data_type_labels or {}).get(col, '')

        fusion = fuse_qi_signals(
            column=col,
            keyword_score_info=kw_info,
            risk_contribution_pct=rc_pct,
            max_contribution=max_contribution,
            data_type_label=dt_label,
        )

        if fusion['is_qi']:
            _kw_tier = kw_info.get('tier', '')

            # ── Admin-keyword guard ──
            # Columns whose names contain administrative keywords (code, type,
            # status, flag) are operational metadata, not quasi-identifiers.
            # High risk contribution alone shouldn't promote them to QI.
            # Exception: DEFINITE QI keywords (e.g. "gender_type" stays QI).
            if _kw_tier != 'DEFINITE':
                _col_lower = col.lower().replace('_', ' ').replace('-', ' ')
                _is_admin = any(kw in _col_lower.split() for kw in ADMIN_KEYWORDS)
                if _is_admin:
                    logger.info(
                        "Auto-classify: %s demoted from QI (admin keyword in name, "
                        "tier=%s, rc=%.1f%%)",
                        col, _kw_tier, contribution_pct.get(col, 0) or 0,
                    )
                    continue  # fall through to Sensitive or Unassigned

            # ── Low-cardinality guard ──
            # Columns with very few unique values relative to dataset size
            # provide minimal re-identification signal and get destroyed by
            # protection methods (e.g. 94% suppression).  Demote unless
            # keyword detection is DEFINITE (e.g. "gender" with 2 values is
            # still a valid QI in small datasets).
            _nunique = data[col].nunique()
            _card_ratio = _nunique / n_rows if n_rows > 0 else 0
            # Proportional low-cardinality guard:
            # Columns with very few unique values relative to dataset size
            # provide negligible re-identification power and get destroyed
            # by structural methods (e.g. 94% suppression in LOCSUPR).
            #
            # Two-tier rule (DEFINITE keywords are never demoted):
            #  Tier 1: ≤3 unique AND ratio < 1%  → demote (binary/ternary
            #          in large data, e.g. status flags)
            #  Tier 2: 4-20 unique AND ratio < 0.05% (1/2000)  → demote
            #          (low-card categoricals in very large data)
            #
            # Examples:
            #   2 unique / 120 rows  = 1.7%   → keep  (meaningful in small data)
            #   2 unique / 5K rows   = 0.04%  → demote (binary flag in medium data)
            #   7 unique / 41K rows  = 0.017% → demote (Tier 2, avg ~6K rows/value)
            #   7 unique / 1K rows   = 0.7%   → keep  (above 0.05%)
            #  50 unique / 10K rows  = 0.5%   → keep  (above 20 unique cap)
            _TIER1_MAX_UNIQUE = 3
            _TIER1_RATIO = 0.01      # 1%
            _TIER2_MAX_UNIQUE = 20
            _TIER2_RATIO = 0.0005    # 0.05%
            _demote = False
            # Never demote DEFINITE keywords or columns with significant
            # risk contribution (>=5%) from backward elimination — the
            # elbow analysis confirms they drive re-identification risk.
            _rc = contribution_pct.get(col, 0) or 0
            _skip_demotion = (_kw_tier == 'DEFINITE' or _rc >= 5.0)
            if not _skip_demotion:
                if _nunique <= _TIER1_MAX_UNIQUE and _card_ratio < _TIER1_RATIO:
                    _demote = True
                elif (_nunique <= _TIER2_MAX_UNIQUE
                      and _card_ratio < _TIER2_RATIO):
                    _demote = True
            if _demote:
                logger.info(
                    "Auto-classify: %s demoted from QI (low cardinality: "
                    "%d unique in %d rows, ratio=%.4f, tier=%s)",
                    col, _nunique, n_rows, _card_ratio, _kw_tier,
                )
                continue  # fall through to Sensitive or Unassigned

            # ── Non-integer numeric guard ──
            # Continuous float columns (prices, areas, amounts) are measurement
            # data — analytical targets, not categorical grouping variables.
            # Demote to let them fall through to Sensitive scoring (Step 3),
            # UNLESS keyword detection is DEFINITE (e.g. column named "age").
            _ct = column_types.get(col, '')
            if _ct == 'continuous' and _kw_tier != 'DEFINITE':
                _series = data[col]
                # Check both native dtype and coerced dtype (object cols
                # storing numeric strings report as object, not float)
                _is_float = pd.api.types.is_float_dtype(_series)
                if not _is_float and _series.dtype == object:
                    _coerced = pd.to_numeric(_series, errors='coerce')
                    _is_float = (
                        _coerced.notna().sum() > 0
                        and pd.api.types.is_float_dtype(_coerced)
                    )
                if _is_float:
                    logger.info(
                        "Auto-classify: %s demoted from QI -> Sensitive candidate "
                        "(continuous float, tier=%s, rc=%.1f%%)",
                        col, _kw_tier, rc_pct or 0.0,
                    )
                    continue  # skip QI classification, will be scored in Step 3

            warnings = []
            if col in high_missing_cols:
                miss_pct = data[col].isna().sum() / n_rows
                warnings.append(f"{miss_pct:.0%} missing values — classification may be unreliable")
            result[col] = {
                'role': 'QI',
                'confidence': fusion['confidence'],
                'confidence_score': fusion['fused_score'],
                'reason': fusion['reason'],
                'warnings': warnings,
            }
            qi_cols.add(col)

    # ------------------------------------------------------------------
    # Step 3: Classify Sensitive columns (the genuine gap)
    # ------------------------------------------------------------------
    exclude = identifier_cols | qi_cols | near_constant_cols
    sensitive_result = suggest_sensitive_columns(
        data,
        exclude_columns=exclude,
        column_types=column_types,
        qi_contribution=contribution_pct,
    )

    for col, info in sensitive_result.items():
        if info['score'] >= 0.20:  # At least Low-confidence Sensitive
            warnings = info.get('warnings', [])
            if col in high_missing_cols:
                miss_pct = data[col].isna().sum() / n_rows
                warnings.append(f"{miss_pct:.0%} missing values — classification may be unreliable")
            result[col] = {
                'role': 'Sensitive',
                'confidence': info['confidence'],
                'confidence_score': info['score'],
                'reason': '; '.join(info['reasons'][:3]),
                'warnings': warnings,
            }

    # ------------------------------------------------------------------
    # Step 4: Everything else → Unassigned
    # ------------------------------------------------------------------
    for col in data.columns:
        if col not in result:
            warnings = []
            if col in high_missing_cols:
                miss_pct = data[col].isna().sum() / n_rows
                warnings.append(f"{miss_pct:.0%} missing values")
            result[col] = {
                'role': 'Unassigned',
                'confidence': 'Medium',
                'confidence_score': 0.0,
                'reason': _unassigned_reason(col, column_types.get(col, ''),
                                              contribution_pct.get(col)),
                'warnings': warnings,
            }

    # ------------------------------------------------------------------
    # Step 5: Cross-column diagnostics
    # ------------------------------------------------------------------
    diagnostics = _cross_column_diagnostics(data, qi_cols, result)

    # ------------------------------------------------------------------
    # Step 6: Feasibility gate — drop excess QIs if cardinality product
    #         is too high for the dataset to satisfy k-anonymity
    # ------------------------------------------------------------------
    qi_cols, result = _feasibility_gate(
        data, qi_cols, result, contribution_pct, diagnostics)

    result['_diagnostics'] = diagnostics

    logger.info(
        "Auto-classify: %d identifiers, %d QIs, %d sensitive, %d unassigned",
        len(identifier_cols),
        len(qi_cols),
        sum(1 for v in result.values() if isinstance(v, dict) and v.get('role') == 'Sensitive'),
        sum(1 for v in result.values() if isinstance(v, dict) and v.get('role') == 'Unassigned'),
    )
    return result


# ---------------------------------------------------------------------------
# DUAL-SIGNAL QI FUSION
# ---------------------------------------------------------------------------

def fuse_qi_signals(
    column: str,
    keyword_score_info: Dict,
    risk_contribution_pct: Optional[float],
    max_contribution: float = 1.0,
    data_type_label: str = '',
) -> Dict:
    """
    Fuse keyword-based QI score with risk contribution.

    Parameters
    ----------
    column : str
        Column name (for logging).
    keyword_score_info : dict
        From ``_calculate_qi_score()``:
        ``{confidence: float, tier: str, reasons: list, signals: dict}``.
    risk_contribution_pct : float or None
        Risk contribution % from backward elimination (e.g. 22.5).
        None if column was not in the risk analysis variables.
    max_contribution : float
        Maximum contribution across all columns, for normalization.
    data_type_label : str
        Semantic type label from Configure table, for domain boosters.

    Returns
    -------
    dict
        ``{is_qi: bool, confidence: str, fused_score: float, reason: str}``
    """
    kw_confidence = keyword_score_info.get('confidence', 0.0)
    kw_tier = keyword_score_info.get('tier', 'NON_QI')
    kw_reasons = keyword_score_info.get('reasons', [])

    # ── DEFINITE floor ──
    # If keyword detection is certain, force QI regardless of empirical risk.
    # Prevents under-classifying obvious QIs (e.g. "age") that happen to have
    # low risk contribution in this particular dataset.
    if kw_tier == 'DEFINITE':
        reason_parts = [f"Keyword: DEFINITE ({kw_confidence:.2f})"]
        if risk_contribution_pct is not None:
            reason_parts.append(f"risk contribution {risk_contribution_pct:.1f}%")
        return {
            'is_qi': True,
            'confidence': 'High',
            'fused_score': max(kw_confidence, 0.90),
            'reason': '; '.join(reason_parts),
        }

    # ── Fusion: 30% keyword + 70% risk contribution ──
    # NOTE: The original spec used 40/60 weighting.  After testing on Greek
    # datasets where column names are domain-specific or coded (keyword score
    # frequently ≈ 0), 30/70 proved more accurate — risk contribution is
    # empirical evidence and carries most of the signal.  The difference is
    # intentional, not a bug.
    normalized_rc = 0.0
    if risk_contribution_pct is not None and max_contribution > 0:
        normalized_rc = min(risk_contribution_pct / max_contribution, 1.0)

    if risk_contribution_pct is None:
        # Column not in risk results (e.g. not in the variable set that was
        # analyzed).  Fall back to keyword score alone.
        fused = kw_confidence
    elif normalized_rc < 0.05:
        # Very low risk contribution — the composite qi_detection score
        # (which includes type-based and uniqueness signals) should carry
        # more weight.  Without this, Greek columns with strong type signals
        # (e.g. 5-unique categorical) but 0% risk get crushed by the 70%
        # weight on near-zero risk.
        fused = 0.60 * kw_confidence + 0.40 * normalized_rc
    else:
        fused = 0.30 * kw_confidence + 0.70 * normalized_rc

    # ── Domain boosters ──
    label_lower = data_type_label.lower() if data_type_label else ''
    col_lower = column.lower().replace('_', ' ').replace('-', ' ')
    boost_reasons: List[str] = []

    if _any_match(col_lower, _DATE_HINTS) or 'date' in label_lower or 'temporal' in label_lower:
        fused = min(fused + 0.15, 1.0)
        boost_reasons.append('+date')

    if _any_match(col_lower, _GEO_HINTS) or 'geographic' in label_lower or 'postal' in label_lower:
        fused = min(fused + 0.15, 1.0)
        boost_reasons.append('+geographic')

    if _any_match(col_lower, _DEMO_HINTS) or 'demographic' in label_lower:
        fused = min(fused + 0.10, 1.0)
        boost_reasons.append('+demographic')

    # ── Thresholds ──
    if fused >= 0.55:
        confidence = 'High'
        is_qi = True
    elif fused >= 0.35:
        confidence = 'Medium'
        is_qi = True
    else:
        confidence = 'Low'
        is_qi = False

    # Build reason string
    reason_parts = [f"Keyword: {kw_tier} ({kw_confidence:.2f})"]
    if risk_contribution_pct is not None:
        reason_parts.append(f"risk {risk_contribution_pct:.1f}%")
    reason_parts.append(f"fused={fused:.2f}")
    if boost_reasons:
        reason_parts.append(f"boosters: {', '.join(boost_reasons)}")

    return {
        'is_qi': is_qi,
        'confidence': confidence,
        'fused_score': fused,
        'reason': '; '.join(reason_parts),
    }


# ---------------------------------------------------------------------------
# SENSITIVE COLUMN DETECTION  (the genuine gap)
# ---------------------------------------------------------------------------

def suggest_sensitive_columns(
    data: pd.DataFrame,
    exclude_columns: Set[str],
    column_types: Optional[Dict[str, str]] = None,
    qi_contribution: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict]:
    """
    Score remaining columns for the Sensitive role.

    Sensitive = "analytical target that the user wants to preserve during SDC,
    but that is not itself a re-identifier."

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset.
    exclude_columns : set
        Columns already assigned as Identifier or QI — skip these.
    column_types : dict, optional
        ``{col: 'continuous'|'categorical'|'identifier'|'binary'}`` from
        ``identify_column_types()``.
    qi_contribution : dict, optional
        ``{col: contribution_pct}`` from backward elimination.

    Returns
    -------
    dict
        ``{col: {score: float, confidence: str, reasons: list}}``
    """
    if column_types is None:
        column_types = identify_column_types(data)
    if qi_contribution is None:
        qi_contribution = {}

    results: Dict[str, Dict] = {}
    for col in data.columns:
        if col in exclude_columns:
            continue
        score_info = _sensitive_score_column(
            data[col], col, column_types.get(col, ''), qi_contribution.get(col),
        )
        results[col] = score_info

    return results


def _sensitive_score_column(
    series: pd.Series,
    col_name: str,
    col_type: str,
    risk_contribution: Optional[float],
) -> Dict:
    """Score a single column for Sensitive role.

    Designed for datasets with unpredictable column names (domain-specific,
    Greek, coded).  Structural signals (data type, cardinality, risk
    contribution, entropy) carry the scoring; keyword matching is a bonus.
    """
    score = 0.0
    reasons: List[str] = []
    col_lower = col_name.lower().replace('_', ' ').replace('-', ' ')

    n_unique = series.nunique()
    n_rows = len(series)
    uniqueness_ratio = n_unique / n_rows if n_rows > 0 else 0

    miss_pct = series.isna().sum() / n_rows if n_rows > 0 else 0
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_continuous = col_type == 'continuous'
    is_binary = col_type == 'binary' or n_unique <= 2
    rc = risk_contribution if risk_contribution is not None else 0.0

    # ── Sparse column penalty (progressive) ──
    # Columns with >50% null are typically conditional/supplementary fields
    # (waiver codes, regulatory notes, reserved fields) — not analytical targets.
    # Scale penalty with sparsity: 50% null → -0.30, 80%+ null → -0.60, 95%+ → -0.80
    if miss_pct > 0.95:
        score -= 0.80
        reasons.append(f"Very sparse column ({miss_pct:.0%} null) — almost certainly not analytical")
    elif miss_pct > 0.80:
        score -= 0.65
        reasons.append(f"Sparse column ({miss_pct:.0%} null) — likely conditional/supplementary")
    elif miss_pct > 0.50:
        score -= 0.30
        reasons.append(f"Sparse column ({miss_pct:.0%} null) — low analytical value")

    # ══════════════════════════════════════════════════════════════════
    # STRUCTURAL signals (work without keyword matches)
    # These are the primary classifiers — column names may be arbitrary
    # ══════════════════════════════════════════════════════════════════

    # 1. Continuous numeric + risk contribution → smooth ramp
    #    A float/int column with many unique values that doesn't drive
    #    re-identification is almost certainly an analysis target.
    #    Uses a smooth ramp instead of hard cutoff to avoid cliff at rc=5%:
    #    rc=0% → +0.40, rc=5% → +0.325, rc=10% → +0.25, rc≥15% → 0
    if is_continuous and rc < 15.0:
        boost = max(0.40 - rc * 0.01, 0.25) if rc < 10.0 else max(0.25 - (rc - 10.0) * 0.05, 0.0)
        score += boost
        reasons.append(f"Continuous numeric, risk {rc:.1f}% (boost +{boost:.2f})")

    # 2. High entropy continuous (rich analytical distribution)
    #    Columns with many distinct values and high entropy are measurement
    #    data (prices, areas, amounts), not classifiers.
    if is_numeric and n_unique > 20:
        try:
            ent = _entropy(series.dropna())
            if ent > 4.0:
                score += 0.25
                reasons.append(f"High entropy ({ent:.1f} bits) — likely measurement data")
            elif ent > 3.0:
                score += 0.15
                reasons.append(f"Moderate entropy ({ent:.1f} bits)")
        except Exception:
            pass

    # 3. Skewness — heavily right-skewed numerics (income, prices, areas)
    #    are almost always analytical targets, not classifiers.
    if is_numeric and n_unique > 20:
        try:
            skew = float(series.dropna().skew())
            if abs(skew) > 2.0:
                score += 0.15
                reasons.append(f"Skewed distribution (skew={skew:.1f}) — typical measurement data")
        except Exception:
            pass

    # 4. Numeric with moderate cardinality (20-500 unique) + low risk
    #    Too many values to be a categorical QI, too few to be an ID.
    if is_numeric and 20 <= n_unique <= 500 and rc < 5.0 and not is_continuous:
        score += 0.20
        reasons.append(f"Numeric, {n_unique} unique values, low risk — likely analytical")

    # 5. Binary / few categories + low risk → outcome/flag variable
    if is_binary and rc < 3.0:
        score += 0.20
        reasons.append("Binary outcome/flag, low re-id risk")
    elif not is_binary and n_unique <= 5 and rc < 3.0:
        score += 0.15
        reasons.append(f"Few categories ({n_unique}), low re-id risk")

    # 6. Ratio / percentage pattern (value range check — no keywords needed)
    if _is_ratio_pattern(series, col_lower):
        score += 0.25
        reasons.append("Ratio/percentage pattern (values in 0-1 or 0-100 range)")

    # ══════════════════════════════════════════════════════════════════
    # KEYWORD signals (bonus when column names happen to match)
    # ══════════════════════════════════════════════════════════════════

    # 6. Value keyword match (income, diagnosis, price, etc.)
    kw_score = _match_sensitive_keyword(col_lower)
    if kw_score > 0:
        score += 0.20
        reasons.append(f"Keyword match ({kw_score:.2f})")

    # ══════════════════════════════════════════════════════════════════
    # NEGATIVE signals (push toward Unassigned)
    # ══════════════════════════════════════════════════════════════════

    # 7. High cardinality non-numeric → likely hidden identifier
    if not is_numeric and uniqueness_ratio > 0.50:
        score -= 0.30
        reasons.append(f"High cardinality non-numeric ({uniqueness_ratio:.0%} unique)")

    # 8. Administrative keyword (code, type, category, status)
    if _any_match(col_lower, _ADMIN_KW_SET):
        score -= 0.15
        reasons.append("Administrative keyword (code/type/status)")

    # 9. High risk contribution on a non-continuous column → more likely a QI
    #    that wasn't caught by the QI detector
    if rc >= 10.0 and not is_continuous:
        score -= 0.20
        reasons.append(f"High risk contribution ({rc:.1f}%) — may be an uncaught QI")

    # 10. Low-cardinality numeric (< 20 unique) → may be a classifier
    #     BUT: sequential integers (1,2,3,4,5,6) are often counts/measurements
    #     (e.g. Πλήθος Προσώπων = 6 values), not arbitrary codes.
    #     Only penalize when values look like arbitrary codes, not sequential counts.
    if is_numeric and n_unique < 20 and not is_binary:
        if _is_sequential_count(series):
            # Sequential integers (1-N) → likely a count/measurement, no penalty
            reasons.append(f"Low-cardinality numeric ({n_unique} values) but sequential — likely a count")
        else:
            score -= 0.15
            reasons.append(f"Low-cardinality numeric ({n_unique} values) — more like a classifier")

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    # ── Confidence mapping ──
    if score >= 0.50:
        confidence = 'High'
    elif score >= 0.35:
        confidence = 'Medium'
    elif score >= 0.20:
        confidence = 'Low'
    else:
        confidence = 'Low'  # Below threshold — caller decides role

    # ── Dual-role warning ──
    # Flag columns that score as Sensitive but also have meaningful risk
    # contribution — these may need manual review (e.g. Επιφάνεια at 15% risk).
    warnings: List[str] = []
    if score >= 0.20 and rc > 5.0:
        warnings.append(
            f"Dual-role: scores as Sensitive but has {rc:.1f}% risk contribution "
            f"— consider whether this column should be QI instead"
        )

    return {'score': score, 'confidence': confidence, 'reasons': reasons, 'warnings': warnings}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _match_sensitive_keyword(col_lower: str) -> float:
    """Return the best keyword match score, or 0."""
    best = 0.0
    for kw, kw_score in _ALL_SENSITIVE_KW.items():
        if kw in col_lower:
            best = max(best, kw_score)
    return best


def _is_ratio_pattern(series: pd.Series, col_lower: str) -> bool:
    """Check if column looks like a ratio/percentage.

    Works on value structure alone (0-1 or 0-100 range with many fractional
    values) OR on name hint.  Column names are often unpredictable, so
    structural detection is primary.

    Limitation: the vmax cap at 105 means per-mille values (0-1000) and
    other non-standard percentage conventions are not detected.  This is
    acceptable because per-mille is rare and false-positive cost is higher
    than false-negative cost for this signal.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    try:
        vals = series.dropna()
        if len(vals) < 5:
            return False
        vmin, vmax = vals.min(), vals.max()

        # Name hint is sufficient (if column name says "rate" or "%")
        if _any_match(col_lower, _RATIO_HINTS):
            return (0 <= vmin and vmax <= 1.05) or (0 <= vmin and vmax <= 105)

        # Structural detection: 0-1 range with fractional values → proportion
        if 0 <= vmin and vmax <= 1.05:
            # Must have fractional values (not just 0/1 binary)
            n_unique = vals.nunique()
            if n_unique > 5:
                return True

        # Structural detection: 0-100 range, float dtype, many values → percentage
        if 0 <= vmin and vmax <= 105 and vals.dtype.kind == 'f':
            n_unique = vals.nunique()
            if n_unique > 10:
                return True

        return False
    except Exception:
        return False


def _is_sequential_count(series: pd.Series) -> bool:
    """Check if values look like sequential integers (1,2,3,...,N).

    Sequential small integers are typically count/measurement variables
    (e.g. number of persons = 1-6), not arbitrary classification codes.
    Returns True if unique non-null values form a contiguous or near-contiguous
    integer sequence starting from 0 or 1.
    """
    try:
        vals = series.dropna()
        if len(vals) == 0:
            return False
        # Must be integer-like (no fractional values)
        if vals.dtype.kind == 'f':
            if not (vals == vals.astype(int)).all():
                return False
        unique_vals = sorted(vals.unique())
        if len(unique_vals) < 2:
            return False
        vmin, vmax = int(unique_vals[0]), int(unique_vals[-1])
        # Must start from 0 or 1
        if vmin not in (0, 1):
            return False
        # Check near-contiguity: at least 80% of the range is covered
        expected_count = vmax - vmin + 1
        coverage = len(unique_vals) / expected_count if expected_count > 0 else 0
        return coverage >= 0.80
    except Exception:
        return False


def _entropy(series: pd.Series) -> float:
    """Compute Shannon entropy in bits."""
    counts = series.value_counts(normalize=True)
    counts = counts[counts > 0]
    return float(-np.sum(counts * np.log2(counts)))


def _any_match(text: str, keywords) -> bool:
    """Return True if any keyword is a substring of text."""
    for kw in keywords:
        if kw in text:
            return True
    return False


def _unassigned_reason(
    col: str, col_type: str, contribution: Optional[float],
) -> str:
    """Build a short reason string for unassigned columns."""
    parts: List[str] = []
    if col_type:
        parts.append(col_type)
    if contribution is not None:
        parts.append(f"{contribution:.1f}% risk")
    else:
        parts.append("no risk data")
    parts.append("not matched by QI or Sensitive heuristics")
    return '; '.join(parts)


# ---------------------------------------------------------------------------
# FEASIBILITY GATE
# ---------------------------------------------------------------------------

def _feasibility_gate(
    data: pd.DataFrame,
    qi_cols: Set[str],
    result: Dict[str, Dict],
    contribution_pct: Dict[str, float],
    diagnostics: List,
    k: int = 5,
    safety_factor: float = 3.0,
) -> Tuple[Set[str], Dict[str, Dict]]:
    """Drop lowest-contribution QIs until the dataset can plausibly satisfy k.

    The cardinality product of all QIs times ``k`` must not exceed
    ``n_rows * safety_factor``.  A safety_factor of 3 means we need
    roughly 3× more records than the minimum for k-anonymity
    (accounting for uneven distributions).

    Demoted QIs are changed to 'Unassigned' with a diagnostic reason.
    """
    if len(qi_cols) <= 1:
        return qi_cols, result

    n_rows = len(data)

    def _est_post_preprocess_card(col):
        """Estimate cardinality after type-aware preprocessing.

        Uses type-aware estimates that mirror what build_type_aware_preprocessing
        and GENERALIZE will actually do, preventing premature QI demotion for
        columns like postal codes (5000 raw → ~5 groups after coarsening).
        """
        nu = max(1, data[col].nunique())
        if nu <= 20:
            return nu  # small cardinality — no preprocessing expected

        is_numeric = pd.api.types.is_numeric_dtype(data[col])
        if is_numeric and nu > 20:
            # Numeric high-card: will be quantile-binned
            return max(5, min(30, n_rows // 50))
        elif not is_numeric and nu > 50:
            # Categorical very-high-card: will be top-K generalized
            return min(10, nu)
        elif not is_numeric and nu > 20:
            # Categorical moderate-card: mild generalization
            return min(20, nu)
        return nu

    def _cardinality_product(cols):
        prod = 1
        for c in cols:
            prod *= _est_post_preprocess_card(c)
        return prod

    # Sort QIs by contribution (lowest first = first to drop)
    sorted_qis = sorted(
        qi_cols,
        key=lambda c: contribution_pct.get(c, 0.0),
    )

    original_card_product = _cardinality_product(qi_cols)
    min_records_needed = original_card_product * k

    if min_records_needed <= n_rows * safety_factor:
        # Feasible — no changes needed
        return qi_cols, result

    logger.info(
        "[Feasibility] Cardinality product %d × k=%d = %d records needed, "
        "but only %d available (%.1f× shortfall)",
        original_card_product, k, min_records_needed, n_rows,
        min_records_needed / (n_rows * safety_factor))

    # Identify hierarchy chains: for multi-level hierarchies like
    # Νομαρχία(54) → Δήμος(886) → Δημοτικό(3329), drop only the finest
    # level(s) and protect the rest.  Build a graph of "finer than" edges,
    # then only the leaves (finest) are candidates for dropping.
    _hier_finer = {}  # col → set of cols it is finer than
    for diag in diagnostics:
        if diag.get('type') == 'hierarchy':
            cols = diag.get('columns', set())
            if len(cols) == 2:
                pair = sorted(cols, key=lambda c: data[c].nunique() if c in data.columns else 0)
                coarse, fine = pair[0], pair[1]
                _hier_finer.setdefault(fine, set()).add(coarse)

    # Columns that are "fine" but not "coarse" of any other pair = true leaves
    _all_coarse = set()
    for fine_col, coarser in _hier_finer.items():
        _all_coarse.update(coarser)
    _hierarchy_fine_cols = set(_hier_finer.keys()) - _all_coarse  # true leaves only
    _hierarchy_coarse_cols = _all_coarse  # everything that has something finer

    dropped = []
    remaining = list(sorted_qis)  # lowest contribution first

    # Phase 1: drop hierarchy fine-grained columns first (geo-guided)
    for fine_col in list(_hierarchy_fine_cols):
        if fine_col in remaining and len(remaining) > 2:
            cur_product = _cardinality_product(remaining)
            if cur_product * k <= n_rows * safety_factor:
                break
            remaining.remove(fine_col)
            dropped.append(fine_col)
            logger.info(
                "[Feasibility] Demoting '%s' (hierarchy fine, %.1f%% contribution, "
                "%d unique) → Unassigned",
                fine_col, contribution_pct.get(fine_col, 0),
                data[fine_col].nunique())

    # Phase 2: only if still infeasible AND hierarchy drops didn't help enough.
    # If hierarchy drops were made, skip further drops — the protection engine
    # handles remaining infeasibility via preprocessing/generalization.
    if not dropped:
        # No hierarchy drops were made — fall back to contribution-based drops
        while len(remaining) > 2:
            cur_product = _cardinality_product(remaining)
            min_needed = cur_product * k
            if min_needed <= n_rows * safety_factor:
                break
            droppable = [c for c in remaining if c not in _hierarchy_coarse_cols]
            if not droppable:
                break
            demoted = droppable[0]
            remaining.remove(demoted)
        dropped.append(demoted)
        logger.info(
            "[Feasibility] Demoting '%s' (%.1f%% contribution, %d unique) "
            "→ Unassigned",
            demoted, contribution_pct.get(demoted, 0), data[demoted].nunique())

    if dropped:
        qi_cols = set(remaining)
        dropped_set = set(dropped)
        for col in dropped:
            if col in _hierarchy_fine_cols:
                reason = (
                    f"Demoted from QI: part of a geographic hierarchy — "
                    f"finer-grained column removed to make protection feasible. "
                    f"Contribution: {contribution_pct.get(col, 0):.1f}%"
                )
            else:
                reason = (
                    f"Demoted from QI: low contribution ({contribution_pct.get(col, 0):.1f}%), "
                    f"removed to make protection feasible."
                )
            result[col] = {
                'role': 'Unassigned',
                'confidence': 'Low',
                'confidence_score': result.get(col, {}).get('confidence_score', 0),
                'reason': reason,
                'warnings': [],
            }

        final_product = _cardinality_product(qi_cols)
        still_tight = final_product * k > n_rows * safety_factor
        status = "still tight" if still_tight else "feasible"
        diagnostics.append({
            'type': 'feasibility',
            'columns': dropped_set,
            'message': (
                f"Dropped {len(dropped)} low-contribution QI(s) "
                f"({', '.join(dropped)}) to make protection feasible. "
                f"Re-add manually if important. "
                f"Cardinality product: {original_card_product:,} → "
                f"{final_product:,}. Status: {status}."
            ),
        })

        # Remove hierarchy diagnostics that are resolved (fine column
        # was dropped, so the hierarchy redundancy is gone).
        diagnostics[:] = [
            d for d in diagnostics
            if not (d.get('type') == 'hierarchy'
                    and d.get('columns', set()) & dropped_set)
        ]


    return qi_cols, result


# ---------------------------------------------------------------------------
# CROSS-COLUMN DIAGNOSTICS
# ---------------------------------------------------------------------------

def _cross_column_diagnostics(
    data: pd.DataFrame,
    qi_cols: Set[str],
    result: Dict[str, Dict],
) -> List[Dict]:
    """Detect cross-column patterns and generate warnings.

    Returns a list of dicts with keys:
    - ``'message'``: human-readable warning text
    - ``'columns'``: set of column names involved
    - ``'type'``: ``'hierarchy'``, ``'functional_dep'``, or ``'correlation'``

    Checks for:
    - Geographic hierarchies (nested categorical columns among QIs)
    - Functional dependencies among QIs (every value of A maps to one value of B)
    - Highly correlated numeric pairs among Sensitive columns

    Hierarchy pairs are merged with functional dependency findings on the same
    pair to avoid duplicate warnings.
    """
    warnings: List[Dict] = []

    # Only run diagnostics on manageable datasets
    if len(data.columns) > 100 or len(data) > 500_000:
        return warnings

    # Track which pairs have already been reported (frozenset of col names)
    reported_pairs: Set[frozenset] = set()

    # --- Geographic / categorical hierarchy detection ---
    qi_cat_cols = []
    for col in qi_cols:
        if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
            n_unique = data[col].nunique()
            if 2 < n_unique < 5000:
                qi_cat_cols.append((col, n_unique))

    qi_cat_cols.sort(key=lambda x: x[1])

    for i, (coarse, coarse_n) in enumerate(qi_cat_cols):
        for fine, fine_n in qi_cat_cols[i + 1:]:
            if fine_n <= coarse_n:
                continue
            if fine_n > coarse_n * 50:
                continue

            try:
                pairs = data[[fine, coarse]].dropna().drop_duplicates()
                fine_to_coarse = pairs.groupby(fine)[coarse].nunique()
                nesting_ratio = (fine_to_coarse == 1).sum() / len(fine_to_coarse) \
                    if len(fine_to_coarse) > 0 else 0
                if nesting_ratio >= 0.90:
                    pair_key = frozenset([coarse, fine])
                    if pair_key not in reported_pairs:
                        reported_pairs.add(pair_key)
                        warnings.append({
                            'type': 'hierarchy',
                            'columns': {coarse, fine},
                            'message': (
                                f"Geographic/categorical hierarchy: {fine} ({fine_n} values) "
                                f"nests within {coarse} ({coarse_n} values). "
                                f"Including both as QIs increases combination space. "
                                f"Consider keeping only the finest or coarsest."
                            ),
                        })
            except Exception:
                pass

    # --- Functional dependencies among QIs ---
    qi_all_cols = [
        (col, data[col].nunique())
        for col in qi_cols
        if col in data.columns and 2 < data[col].nunique() < 5000
    ]
    if len(qi_all_cols) <= 20:
        for i, (col_a, n_a) in enumerate(qi_all_cols):
            for col_b, n_b in qi_all_cols[i + 1:]:
                pair_key = frozenset([col_a, col_b])
                if pair_key in reported_pairs:
                    continue  # already flagged as hierarchy
                try:
                    pairs = data[[col_a, col_b]].dropna().drop_duplicates()
                    if len(pairs) < 3:
                        continue
                    a_to_b = pairs.groupby(col_a)[col_b].nunique()
                    fd_a_to_b = (a_to_b == 1).sum() / len(a_to_b) if len(a_to_b) > 0 else 0
                    b_to_a = pairs.groupby(col_b)[col_a].nunique()
                    fd_b_to_a = (b_to_a == 1).sum() / len(b_to_a) if len(b_to_a) > 0 else 0

                    if fd_a_to_b >= 0.95 and fd_b_to_a >= 0.95:
                        reported_pairs.add(pair_key)
                        warnings.append({
                            'type': 'functional_dep',
                            'columns': {col_a, col_b},
                            'message': (
                                f"Functional dependency: {col_a} ↔ {col_b} determine each other "
                                f"(1:1 mapping). Including both as QIs is redundant — "
                                f"consider dropping one."
                            ),
                        })
                    elif fd_a_to_b >= 0.95:
                        reported_pairs.add(pair_key)
                        warnings.append({
                            'type': 'functional_dep',
                            'columns': {col_a, col_b},
                            'message': (
                                f"Functional dependency: {col_a} → {col_b} "
                                f"({col_a} determines {col_b}). Including both as QIs "
                                f"inflates combination space. Consider keeping only {col_a}."
                            ),
                        })
                    elif fd_b_to_a >= 0.95:
                        reported_pairs.add(pair_key)
                        warnings.append({
                            'type': 'functional_dep',
                            'columns': {col_a, col_b},
                            'message': (
                                f"Functional dependency: {col_b} → {col_a} "
                                f"({col_b} determines {col_a}). Including both as QIs "
                                f"inflates combination space. Consider keeping only {col_b}."
                            ),
                        })
                except Exception:
                    pass

    # --- Highly correlated numeric pairs ---
    sens_num_cols = [
        col for col in data.columns
        if result.get(col, {}).get('role') == 'Sensitive'
        and pd.api.types.is_numeric_dtype(data[col])
        and data[col].nunique() > 5
    ]

    if 2 <= len(sens_num_cols) <= 20:
        try:
            corr_matrix = data[sens_num_cols].corr(method='pearson')
            for i, col_a in enumerate(sens_num_cols):
                for col_b in sens_num_cols[i + 1:]:
                    r = abs(corr_matrix.loc[col_a, col_b])
                    if r > 0.95:
                        warnings.append({
                            'type': 'correlation',
                            'columns': {col_a, col_b},
                            'message': (
                                f"Highly correlated: {col_a} and {col_b} (r={r:.2f}). "
                                f"Near-redundant — consider whether both need the same role."
                            ),
                        })
        except Exception:
            pass

    return warnings
