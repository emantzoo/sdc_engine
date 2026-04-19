"""
Protection Engine
=================

Shared module for rules-engine-based protection across all SDC views
(Protect Auto, Smart Combo, etc.).

Provides:
- build_data_features(): standalone data feature extraction for the rules engine
- run_rules_engine_protection(): shared retry loop (pipeline -> primary + escalation -> fallbacks)
- run_pipeline(): multi-method pipeline runner
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)


def build_data_features(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    reid: Optional[Dict] = None,
    active_cols: Optional[List[str]] = None,
    var_priority: Optional[Dict] = None,
    column_types: Optional[Dict[str, str]] = None,
    risk_metric: Optional[str] = None,
    risk_target_raw: Optional[float] = None,
    sensitive_columns: Optional[List[str]] = None,
    preprocess_metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build the features dict that select_method_suite() expects.

    Standalone version — operates on any DataFrame without needing a view or
    dataset object.  Computes risk metric internally if not provided.

    Parameters
    ----------
    data : DataFrame
        The data to analyze.
    quasi_identifiers : list of str
        Selected quasi-identifier column names.
    reid : dict, optional
        Pre-computed ReID dict (keys: reid_50, reid_95, reid_99).
        If None, will be computed here via the chosen *risk_metric*.
    active_cols : list of str, optional
        Columns to consider. If None, uses all columns in *data*.
    var_priority : dict, optional
        Per-QI risk contribution from backward elimination.
        {col: (priority_label, pct_score)}. Enables risk-concentration rules.
    risk_metric : str, optional
        'reid95', 'k_anonymity', 'uniqueness', or 'l_diversity'. Default 'reid95'.
    risk_target_raw : float, optional
        Raw target for the chosen metric (used by RiskAssessment).
    sensitive_columns : list of str, optional
        Sensitive column names. Required for l_diversity metric.

    Returns
    -------
    dict
        Features dict compatible with ``select_method_suite()``.
    """
    from sdc_engine.sdc.metrics.risk_metric import (
        RiskMetricType, compute_risk, risk_to_reid_compat,
    )

    if active_cols is None:
        active_cols = list(data.columns)

    cols_in_df = [c for c in active_cols if c in data.columns]
    n_rows = len(data)

    # Resolve metric enum
    _mt_map = {
        'reid95': RiskMetricType.REID95,
        'k_anonymity': RiskMetricType.K_ANONYMITY,
        'uniqueness': RiskMetricType.UNIQUENESS,
        'l_diversity': RiskMetricType.L_DIVERSITY,
    }
    mt = _mt_map.get(risk_metric or 'reid95', RiskMetricType.REID95)

    # Compute risk if not provided
    risk_assessment = None
    if reid is None:
        try:
            assessment = compute_risk(
                data, quasi_identifiers, mt, risk_target_raw,
                sensitive_columns=sensitive_columns)
            risk_assessment = assessment
            reid = risk_to_reid_compat(assessment)
        except Exception:
            reid = {}

    # Coerce object-dtype columns that Configure classified as numeric
    from sdc_engine.sdc.sdc_utils import coerce_columns_by_types
    data = data.copy()
    coerce_columns_by_types(data, column_types, cols_in_df)

    # Classify QI columns only (not sensitive or unassigned columns)
    # This prevents PRAM/NOISE from accidentally modifying sensitive columns.
    # Preprocessing metadata overrides dtype-based classification: columns
    # that were originally continuous but got binned by GENERALIZE are still
    # counted as continuous for method selection purposes.
    _discretized = set()
    if preprocess_metadata:
        for qi, meta in preprocess_metadata.items():
            if isinstance(meta, dict) and meta.get('was_continuous'):
                _discretized.add(qi)

    continuous, categorical = [], []
    for col in quasi_identifiers:
        if col not in data.columns:
            continue
        if col in _discretized:
            continuous.append(col)
        elif pd.api.types.is_numeric_dtype(data[col]):
            if data[col].nunique() > 20:
                continuous.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)

    # QI cardinality analysis
    qi_cardinalities = {}
    high_card_qis, low_card_qis = [], []
    for qi in quasi_identifiers:
        if qi in data.columns:
            nu = data[qi].nunique()
            qi_cardinalities[qi] = nu
            rel = nu / n_rows if n_rows > 0 else 0
            if rel > 0.5:
                high_card_qis.append(qi)
            elif rel < 0.1:
                low_card_qis.append(qi)

    # Per-QI max category frequency (for categorical-aware rule selection)
    qi_max_cat_freq = {}
    for qi in quasi_identifiers:
        if qi in data.columns:
            is_cat = not pd.api.types.is_numeric_dtype(data[qi]) or data[qi].nunique() <= 20
            if is_cat:
                vc = data[qi].value_counts(normalize=True)
                qi_max_cat_freq[qi] = float(vc.iloc[0]) if len(vc) > 0 else 0.0

    qi_card_product = 1
    for v in qi_cardinalities.values():
        qi_card_product *= max(v, 1)
    expected_eq = n_rows / qi_card_product if qi_card_product > 0 else 0

    # Fast kANON suppression estimate at multiple k values
    # Single groupby pass, then threshold at k=3,5,7 to support rules
    # selecting different k levels.
    estimated_suppression = {3: 0.0, 5: 0.0, 7: 0.0}
    qi_in_df = [q for q in quasi_identifiers if q in data.columns]
    if qi_in_df and n_rows > 0:
        try:
            eq_sizes = data.groupby(qi_in_df, dropna=False).size()
            for k_val in (3, 5, 7):
                records_below = int(eq_sizes[eq_sizes < k_val].sum())
                estimated_suppression[k_val] = records_below / n_rows
        except Exception as exc:
            log.warning("[Features] Equivalence class estimation failed: %s", exc)
    estimated_suppression_k5 = estimated_suppression[5]  # backward compat

    if expected_eq >= 10:
        feasibility, max_k = 'easy', int(expected_eq)
    elif expected_eq >= 5:
        feasibility, max_k = 'moderate', 5
    elif expected_eq >= 3:
        feasibility, max_k = 'hard', 3
    else:
        feasibility, max_k = 'infeasible', 0

    # Outlier detection (IQR method on numeric cols)
    has_outliers = False
    skewed_cols = []
    for col in continuous:
        if col not in data.columns:
            continue
        s = data[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outlier_rate = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean()
            if outlier_rate > 0.02:
                has_outliers = True
        try:
            if abs(s.skew()) > 1.5:
                skewed_cols.append(col)
        except Exception as exc:
            log.debug("[Features] Skewness calculation failed for %s: %s", col, exc)

    # Uniqueness rate
    if n_rows > 0 and quasi_identifiers:
        qi_in_df = [q for q in quasi_identifiers if q in data.columns]
        if qi_in_df:
            try:
                unique_combos = data[qi_in_df].drop_duplicates().shape[0]
                uniqueness = unique_combos / n_rows
            except Exception:
                uniqueness = 0
        else:
            uniqueness = 0
    else:
        uniqueness = 0

    # Risk pattern classification
    reid_50 = reid.get('reid_50', 0)
    reid_95 = reid.get('reid_95', 0)
    reid_99 = reid.get('reid_99', 0)
    mean_risk = (reid_50 + reid_95) / 2

    if reid_50 >= 0.20 and (reid_99 - reid_50) < 0.10:
        risk_pattern = 'uniform_high'
    elif reid_50 >= 0.20:
        risk_pattern = 'widespread'
    elif reid_50 < 0.05 and reid_95 >= 0.30 and reid_99 >= 0.50:
        risk_pattern = 'severe_tail'
    elif reid_50 < 0.05 and reid_95 >= 0.30:
        risk_pattern = 'tail'
    elif abs(mean_risk - reid_50) > 0.15:
        risk_pattern = 'bimodal'
    elif reid_50 < 0.05 and reid_95 < 0.30:
        risk_pattern = 'uniform_low'
    else:
        risk_pattern = 'moderate'

    # Fraction of records with individual risk > 20%, computed from
    # equivalence class sizes: risk(record) ≈ 1/eq_size.
    high_risk_rate = 0.0
    if qi_in_df and n_rows > 0:
        try:
            # eq_sizes already computed above (line 140)
            high_risk_records = int(eq_sizes[eq_sizes < 5].sum())  # 1/5 = 0.20
            high_risk_rate = high_risk_records / n_rows
        except Exception as exc:
            log.warning("[Features] High-risk rate calculation failed: %s", exc)

    # Lazily compute var_priority for small-to-medium datasets.
    # RC rules become reachable when this is populated.
    if not var_priority and reid_95 > 0:
        var_priority = _compute_var_priority(data, quasi_identifiers, reid_95)

    log.info("[Features] n_records=%d  n_qis=%d  continuous=%d  categorical=%d  "
             "ReID: 50=%.4f 95=%.4f 99=%.4f  pattern=%s  feasibility=%s(k=%d)  "
             "uniqueness=%.2f%%  outliers=%s  high_card_qis=%s",
             n_rows, len(quasi_identifiers), len(continuous), len(categorical),
             reid_50, reid_95, reid_99, risk_pattern,
             feasibility, max_k, uniqueness * 100, has_outliers,
             high_card_qis)
    return {
        'n_records': n_rows,
        'n_columns': len(cols_in_df),
        'data_type': 'microdata',
        'n_continuous': len(continuous),
        'n_categorical': len(categorical),
        'continuous_vars': continuous,
        'categorical_vars': categorical,
        'n_qis': len(quasi_identifiers),
        'quasi_identifiers': quasi_identifiers,
        'high_cardinality_qis': high_card_qis,
        'low_cardinality_qis': low_card_qis,
        'high_cardinality_count': len(high_card_qis),
        'qi_cardinalities': qi_cardinalities,
        'qi_cardinality_product': qi_card_product,
        'expected_eq_size': expected_eq,
        'k_anonymity_feasibility': feasibility,
        'max_achievable_k': max_k,
        'recommended_qi_to_remove': None,
        'uniqueness_rate': uniqueness,
        'has_outliers': has_outliers,
        'skewed_columns': skewed_cols,
        'has_sensitive_attributes': False,
        'sensitive_columns': {},
        'has_reid': True,
        'reid_50': reid_50,
        'reid_95': reid_95,
        'reid_99': reid_99,
        'mean_risk': mean_risk,
        'max_risk': reid_99,
        'risk_pattern': risk_pattern,
        'high_risk_count': int(high_risk_rate * n_rows) if n_rows > 0 else 0,
        'high_risk_rate': high_risk_rate,
        'risk_level': risk_pattern,
        'small_cells_rate': 0.0,
        'qi_max_category_freq': qi_max_cat_freq,
        'estimated_suppression': estimated_suppression,
        'estimated_suppression_k5': estimated_suppression_k5,
        'var_priority': var_priority or {},
        'risk_concentration': _classify_risk_conc(var_priority),
        '_risk_metric_type': mt.value,
        '_risk_assessment': risk_assessment,
    }


def _compute_var_priority(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    reid_95_baseline: float,
) -> Optional[Dict[str, Any]]:
    """Compute per-QI risk contribution via leave-one-out reid_95.

    Returns a dict ``{qi_name: ('HIGH'|'MEDIUM'|'LOW', contribution_pct)}``
    matching the shape expected by ``classify_risk_concentration()``.
    Returns None if the dataset exceeds performance thresholds or fails.
    """
    from sdc_engine.sdc.config import VAR_PRIORITY_COMPUTATION
    cfg = VAR_PRIORITY_COMPUTATION

    if not cfg.get('enabled', True):
        return None

    n = len(data)
    n_qis = len(quasi_identifiers)

    if n > cfg['max_n_records']:
        log.debug("[var_priority] Skipping: n=%d > max_n_records=%d",
                  n, cfg['max_n_records'])
        return None
    if n_qis > cfg['max_n_qis']:
        log.debug("[var_priority] Skipping: n_qis=%d > max_n_qis=%d",
                  n_qis, cfg['max_n_qis'])
        return None

    if reid_95_baseline is None or reid_95_baseline <= 0:
        return None

    import time as _time
    start = _time.monotonic()
    timeout = cfg['timeout_seconds']

    try:
        from sdc_engine.sdc.metrics.reid import calculate_reid

        contributions = {}
        for qi in quasi_identifiers:
            if _time.monotonic() - start > timeout:
                log.warning("[var_priority] Timeout after %.1fs — returning partial",
                            timeout)
                return contributions if contributions else None

            remaining_qis = [q for q in quasi_identifiers if q != qi]
            if not remaining_qis:
                contributions[qi] = ('HIGH', 100.0)
                continue

            reid_without = calculate_reid(data, remaining_qis).get('reid_95', 0)
            drop = max(0, reid_95_baseline - reid_without)
            contrib_pct = round((drop / reid_95_baseline) * 100, 1)

            if contrib_pct >= 30:
                label = 'HIGH'
            elif contrib_pct >= 15:
                label = 'MEDIUM'
            else:
                label = 'LOW'
            contributions[qi] = (label, contrib_pct)

        elapsed = _time.monotonic() - start
        log.info("[var_priority] Computed in %.2fs for n=%d, n_qis=%d: %s",
                 elapsed, n, n_qis, contributions)
        return contributions

    except Exception as exc:
        log.warning("[var_priority] Computation failed: %s", exc)
        return None


def _classify_risk_conc(var_priority):
    """Thin wrapper — delegates to features.classify_risk_concentration."""
    try:
        from sdc_engine.sdc.selection.features import classify_risk_concentration
        return classify_risk_concentration(var_priority)
    except Exception:
        return {'pattern': 'unknown', 'top_qi': None, 'top_pct': 0,
                'top2_pct': 0, 'n_high_risk': 0}


# ---------------------------------------------------------------------------
# Helpers used by the retry loop
# ---------------------------------------------------------------------------

def _get_reid_95(result) -> Optional[float]:
    """Extract ReID95 from a ProtectionResult."""
    if not result or not result.reid_after:
        return None
    return result.reid_after.get('reid_95')


# ---------------------------------------------------------------------------
# Fast-path privacy checks (short-circuit on first violation)
# ---------------------------------------------------------------------------

_PRIVACY_SIZE_THRESHOLD = 50  # only check small equivalence classes


def _quick_l_diversity_ok(
    protected_data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    l_target: int,
) -> bool:
    """Fast l-diversity gate: returns False on first violating class."""
    valid_qis = [q for q in quasi_identifiers if q in protected_data.columns]
    valid_sens = [s for s in sensitive_columns if s in protected_data.columns]
    if not valid_qis or not valid_sens:
        return True

    grouped = protected_data.groupby(valid_qis, observed=True)
    group_sizes = grouped.size()
    small_keys = group_sizes[group_sizes <= _PRIVACY_SIZE_THRESHOLD].index

    for key in small_keys:
        group = grouped.get_group(key if isinstance(key, tuple) else (key,))
        for scol in valid_sens:
            if group[scol].nunique() < l_target:
                return False
    return True


def _quick_t_closeness_ok(
    protected_data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    t_target: float,
) -> bool:
    """Fast t-closeness gate: returns False on first violating class."""
    import numpy as np

    valid_qis = [q for q in quasi_identifiers if q in protected_data.columns]
    valid_sens = [s for s in sensitive_columns if s in protected_data.columns]
    if not valid_qis or not valid_sens:
        return True

    grouped = protected_data.groupby(valid_qis, observed=True)
    group_sizes = grouped.size()
    small_keys = group_sizes[group_sizes <= _PRIVACY_SIZE_THRESHOLD].index

    for scol in valid_sens:
        overall = protected_data[scol].dropna()
        if len(overall) == 0:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(overall)
        if is_numeric:
            overall_sorted = np.sort(overall.values)
        else:
            overall_freq = overall.value_counts(normalize=True)

        for key in small_keys:
            group = grouped.get_group(key if isinstance(key, tuple) else (key,))
            vals = group[scol].dropna()
            if len(vals) == 0:
                continue

            if is_numeric:
                from sdc_engine.sdc.post_protection_diagnostics import _emd_numeric
                dist = _emd_numeric(vals.values, overall_sorted)
            else:
                class_freq = vals.value_counts(normalize=True)
                all_cats = set(class_freq.index) | set(overall_freq.index)
                dist = sum(abs(class_freq.get(c, 0.0) - overall_freq.get(c, 0.0))
                           for c in all_cats) / 2.0

            if dist > t_target:
                return False
    return True


def _meets_targets(
    result, reid_target: float, *,
    l_target: Optional[int] = None,
    t_target: Optional[float] = None,
    sensitive_columns: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
) -> bool:
    """Check if result meets ReID target and optional privacy targets.

    Parameters
    ----------
    l_target : int or None
        Minimum distinct l-diversity. None = disabled.
    t_target : float or None
        Maximum t-closeness distance. None = disabled.
    sensitive_columns, quasi_identifiers : needed for l/t checks.
    """
    if not result or not result.success:
        return False
    reid = result.reid_after.get('reid_95', 1) if result.reid_after else 1
    if reid > reid_target:
        return False

    # Optional privacy gates (only when sensitive columns are assigned)
    if sensitive_columns and quasi_identifiers and result.protected_data is not None:
        if l_target is not None:
            if not _quick_l_diversity_ok(
                    result.protected_data, quasi_identifiers,
                    sensitive_columns, l_target):
                return False
        if t_target is not None:
            if not _quick_t_closeness_ok(
                    result.protected_data, quasi_identifiers,
                    sensitive_columns, t_target):
                return False

    return True


def _utility_ok(result, utility_floor: float) -> bool:
    """Check utility hasn't dropped below floor.

    result.utility_score is scoped to sensitive columns, which may be
    untouched by QI-only methods (kANON, LOCSUPR) → always ~100%.
    When sensitive utility is near-perfect (>= 0.95), also check the
    average per-variable QI utility to catch real degradation.
    """
    if not result:
        return True
    score = result.utility_score
    if score is None:
        return True

    # If sensitive utility itself is low, that's a clear signal
    if score < utility_floor:
        return False

    # When sensitive utility is near-perfect, check QI utility too.
    # This catches cases where kANON/LOCSUPR heavily modify QIs but
    # leave sensitive columns untouched (score ≈ 1.0).
    if score >= 0.95 and result.per_variable_utility:
        qi_scores = []
        for col, metrics in result.per_variable_utility.items():
            if metrics.get('dtype') == 'numeric':
                val = metrics.get('correlation')
            else:
                val = metrics.get('row_preservation')
            if val is not None:
                qi_scores.append(val)
        if qi_scores:
            qi_avg = sum(qi_scores) / len(qi_scores)
            if qi_avg < utility_floor:
                return False

    return True


def _get_utility_score(result) -> Optional[float]:
    """Extract the utility score for logging.

    Note: result.utility_score is already the composite score (set by
    apply_method → compute_composite_utility). Return as-is.
    """
    if not result:
        return None
    return result.utility_score


def _try_perturbative_challenge(
    structural_result: Any,
    structural_method: str,
    input_data: pd.DataFrame,
    quasi_identifiers: List[str],
    data_features: Dict[str, Any],
    apply_method_fn: Callable,
    risk_target_raw: Optional[float],
    log_entries: List[str],
) -> Optional[Any]:
    """Try PRAM as a post-success challenge to a structural method.

    Returns the PRAM result if it beats the structural result on utility
    (by at least min_utility_gain) AND meets the same risk target.
    Returns None otherwise.

    The challenge runs at most once per protection call.
    """
    from sdc_engine.sdc.config import PERTURBATIVE_CHALLENGE

    cfg = PERTURBATIVE_CHALLENGE
    if not cfg.get('enabled', True):
        return None

    # Gate 1: method must be structural
    if structural_method.upper() not in ('KANON', 'LOCSUPR'):
        return None

    # Gate 2: categorical ratio
    n_cat = data_features.get('n_categorical', 0)
    n_cont = data_features.get('n_continuous', 0)
    total = n_cat + n_cont
    if total == 0:
        return None
    cat_ratio = n_cat / total
    if cat_ratio < cfg['min_cat_ratio']:
        return None

    # Gate 3: reid_95 low enough that PRAM could plausibly hit target
    reid_95 = data_features.get('reid_95', 1.0)
    if reid_95 > cfg['max_reid_95']:
        return None

    # Gate 4: structural result showed meaningful suppression
    supp_detail = getattr(structural_result, 'qi_suppression_detail', None) or {}
    if supp_detail:
        max_supp = max(supp_detail.values()) if supp_detail else 0
    else:
        # Fall back to metadata statistics if available
        meta = getattr(structural_result, 'metadata', None) or {}
        stats = meta.get('statistics', {}) if isinstance(meta, dict) else {}
        max_supp = stats.get('suppression_rate', 0)

    if max_supp < cfg['min_structural_suppression']:
        log.debug(
            "[PerturbativeChallenge] Skipped — structural suppression "
            "%.1f%% below threshold %.1f%%",
            max_supp * 100, cfg['min_structural_suppression'] * 100)
        return None

    # Build PRAM params — scale p by reid_95
    p_change = min(
        cfg['max_p'],
        cfg['base_p'] + reid_95 * cfg['scale_p']
    )
    p_change = round(p_change, 2)

    # PRAM variables: top categorical QIs
    from sdc_engine.sdc.selection.features import top_categorical_qis
    pram_vars = top_categorical_qis(data_features) or [
        qi for qi in quasi_identifiers
        if qi in data_features.get('categorical_vars', [])
    ]
    if not pram_vars:
        return None

    pram_params = {'variables': pram_vars, 'p_change': p_change}
    log_entries.append(
        f"  Perturbative challenge: trying PRAM(p={p_change}) "
        f"after {structural_method} success (cat_ratio={cat_ratio:.0%}, "
        f"supp={max_supp:.1%})"
    )
    log.info(
        "[PerturbativeChallenge] Running PRAM p=%.2f vars=%s",
        p_change, pram_vars)

    try:
        pram_result = apply_method_fn(
            'PRAM', quasi_identifiers, pram_params, input_data=input_data,
        )
    except Exception as exc:
        log.warning("[PerturbativeChallenge] PRAM raised: %s", exc)
        return None

    if not pram_result or not pram_result.success:
        log_entries.append("    PRAM challenge failed — keeping structural result")
        return None

    # Compare: PRAM must meet the same risk target AND gain at least
    # min_utility_gain percentage points
    pram_reid = _get_reid_95(pram_result)
    struct_reid = _get_reid_95(structural_result)
    pram_util = _get_utility_score(pram_result) or 0
    struct_util = _get_utility_score(structural_result) or 0

    if risk_target_raw is not None and pram_reid is not None:
        if pram_reid > risk_target_raw:
            log_entries.append(
                f"    PRAM challenge missed target "
                f"(ReID={pram_reid:.4f} > {risk_target_raw:.4f}) — keeping structural"
            )
            return None

    utility_gain = pram_util - struct_util
    if utility_gain < cfg['min_utility_gain']:
        log_entries.append(
            f"    PRAM challenge utility gain insufficient "
            f"({utility_gain:+.2%} < {cfg['min_utility_gain']:.0%}) — keeping structural"
        )
        return None

    log_entries.append(
        f"  PRAM challenge WON: utility {struct_util:.1%} -> {pram_util:.1%} "
        f"({utility_gain:+.2%}), "
        f"ReID {f'{struct_reid:.4f}' if struct_reid is not None else 'N/A'} -> "
        f"{f'{pram_reid:.4f}' if pram_reid is not None else 'N/A'}"
    )
    log.info(
        "[PerturbativeChallenge] Accepted: utility %.1f%% -> %.1f%% (+%.1f%%)",
        struct_util * 100, pram_util * 100, utility_gain * 100)

    # Tag the result so downstream can see it was a challenge win
    if pram_result.metadata is None:
        pram_result.metadata = {}
    pram_result.metadata['perturbative_challenge'] = {
        'replaced_method': structural_method,
        'structural_utility': struct_util,
        'pram_utility': pram_util,
        'utility_gain': utility_gain,
    }
    return pram_result


def _pick_escalation_start(
    schedule_values: List, reid_current: float, reid_target: float,
) -> int:
    """Return index into schedule_values to start escalation.

    Uses the ratio of (current ReID - target) / target to estimate
    how far up the schedule to jump. Small gap → start near beginning.
    Large gap → jump further. Never skips more than 50% of the schedule.
    """
    if not schedule_values or reid_current <= reid_target:
        return 0
    gap_ratio = min(1.0, (reid_current - reid_target) / max(reid_target, 0.01))
    # Map gap_ratio [0,1] → skip fraction [0, 0.5] of the schedule
    skip_frac = gap_ratio * 0.5
    return int(len(schedule_values) * skip_frac)


def _prune_schedule_by_max_k(
    schedule_values: List, max_achievable_k: int, param_name: str,
) -> List:
    """Remove escalation values that exceed the dataset's max achievable k.

    Only applies to k-based parameters (kANON 'k' or LOCSUPR 'k').
    Other parameter types are returned unchanged.
    """
    if param_name != 'k' or max_achievable_k <= 0:
        return schedule_values
    pruned = [v for v in schedule_values if v <= max_achievable_k]
    if len(pruned) < len(schedule_values):
        log.info("[Protection] k-pruning: schedule %s → %s (max_k=%d)",
                 schedule_values, pruned, max_achievable_k)
    return pruned


# Cross-method equivalence for smarter fallback starting points.
# When primary fails at a given strength, start the fallback at an
# equivalent strength rather than from scratch.
_CROSS_METHOD_START = {
    # (primary_method, primary_param_name, threshold): {fallback: {param: val}}
    ('kANON', 'k', 3):  {'LOCSUPR': {'k': 3},  'PRAM': {'p_change': 0.10}, 'NOISE': {'magnitude': 0.05}},
    ('kANON', 'k', 5):  {'LOCSUPR': {'k': 5},  'PRAM': {'p_change': 0.15}, 'NOISE': {'magnitude': 0.10}},
    ('kANON', 'k', 7):  {'LOCSUPR': {'k': 7},  'PRAM': {'p_change': 0.20}, 'NOISE': {'magnitude': 0.15}},
    ('kANON', 'k', 10): {'LOCSUPR': {'k': 7},  'PRAM': {'p_change': 0.20}, 'NOISE': {'magnitude': 0.15}},
    ('kANON', 'k', 15): {'LOCSUPR': {'k': 10}, 'PRAM': {'p_change': 0.25}, 'NOISE': {'magnitude': 0.20}},
    ('LOCSUPR', 'k', 5):  {'kANON': {'k': 5}, 'PRAM': {'p_change': 0.15}, 'NOISE': {'magnitude': 0.10}},
    ('LOCSUPR', 'k', 7):  {'kANON': {'k': 7}, 'PRAM': {'p_change': 0.20}, 'NOISE': {'magnitude': 0.15}},
    ('LOCSUPR', 'k', 10): {'kANON': {'k': 10}, 'PRAM': {'p_change': 0.25}, 'NOISE': {'magnitude': 0.20}},
}




def _step_down_k(
    input_data: pd.DataFrame,
    best_result,
    data_features: Dict,
    quasi_identifiers: List[str],
    reid_target: float,
    target_k: int,
    apply_method_fn,
    log_entries: List[str],
    utility_gain_threshold: float = 0.02,
) -> Optional[Any]:
    """After kANON succeeds, try one step lower k.

    Returns stepped-down result if it meets targets with better utility,
    otherwise returns None (keep primary).

    Gate conditions:
    - Method is kANON
    - Achieved k > target_k + 2 (meaningful overshoot)
    - Suppression > 2% (something to save)
    """
    if best_result is None or not best_result.success:
        return None

    if getattr(best_result, 'method_used', '') != 'kANON':
        return None

    # Extract achieved k from result metadata
    _meta = getattr(best_result, 'metadata', {}) or {}
    achieved_k = _meta.get('k_achieved') or _meta.get('k')
    if achieved_k is None:
        # Try to get from parameters
        _params = _meta.get('parameters', {})
        achieved_k = _params.get('k', 0)
    if not isinstance(achieved_k, (int, float)) or achieved_k <= target_k + 2:
        return None

    primary_suppression = getattr(best_result, 'suppression_rate', 0) or 0
    if primary_suppression <= 0.02:
        return None

    primary_utility = getattr(best_result, 'utility_score', 0) or 0

    # One step down
    _step_map = {30: 20, 25: 15, 20: 15, 15: 10, 10: 7, 7: 5, 5: 3}
    achieved_k = int(achieved_k)
    lower_k = _step_map.get(achieved_k)
    if lower_k is None:
        lower_k = max(target_k, int(achieved_k * 0.7))
    if lower_k < target_k:
        return None

    log.info("[StepDown] kANON succeeded at k=%d (target k=%d). "
             "Trying k=%d (suppression was %.1f%%)",
             achieved_k, target_k, lower_k, primary_suppression * 100)
    log_entries.append(
        f"k step-down: trying k={lower_k} (was k={achieved_k})")

    try:
        # Determine strategy from primary result
        strategy = _meta.get('strategy', 'hybrid')
        stepped_params = {
            'quasi_identifiers': quasi_identifiers,
            'k': lower_k,
            'strategy': strategy,
        }
        stepped_result = apply_method_fn(
            'kANON', quasi_identifiers, stepped_params,
            input_data=input_data,
        )
        if not stepped_result or not stepped_result.success:
            log.info("[StepDown] k=%d failed", lower_k)
            log_entries.append(f"  Step-down: k={lower_k} failed")
            return None

        # Check targets
        stepped_reid = _get_reid_95(stepped_result)
        stepped_meets_reid = (stepped_reid is not None
                              and stepped_reid <= reid_target)

        # Check k-anonymity from result
        _s_reid_after = getattr(stepped_result, 'reid_after', {}) or {}
        stepped_min_k = _s_reid_after.get('min_k')
        if stepped_min_k is None:
            # Compute it
            try:
                from sdc_engine.sdc.metrics.risk import check_kanonymity
                _k_check = check_kanonymity(
                    stepped_result.protected_data, quasi_identifiers, k=target_k)
                stepped_min_k = _k_check[1]['count'].min() if len(_k_check[1]) > 0 else 0
            except Exception:
                stepped_min_k = lower_k  # assume achieved

        stepped_meets_k = (stepped_min_k is not None
                           and stepped_min_k >= target_k)

        if not (stepped_meets_reid and stepped_meets_k):
            log.info("[StepDown] k=%d did not meet targets "
                     "(reid=%s, k=%s). Keeping k=%d",
                     lower_k,
                     f"{stepped_reid:.1%}" if stepped_reid else "N/A",
                     stepped_min_k, achieved_k)
            log_entries.append(f"  Step-down: k={lower_k} did not meet targets")
            return None

        stepped_utility = getattr(stepped_result, 'utility_score', 0) or 0
        utility_gain = stepped_utility - primary_utility
        stepped_suppression = getattr(stepped_result, 'suppression_rate', 0) or 0

        if utility_gain > utility_gain_threshold:
            log.info("[StepDown] k=%d wins: utility %.1f%% vs %.1f%% "
                     "(+%.1f%%), suppression %.1f%% vs %.1f%%",
                     lower_k,
                     stepped_utility * 100, primary_utility * 100,
                     utility_gain * 100,
                     stepped_suppression * 100, primary_suppression * 100)
            log_entries.append(
                f"\u2713 Step-down accepted: k={lower_k} utility "
                f"{stepped_utility:.0%} vs k={achieved_k} {primary_utility:.0%} "
                f"(+{utility_gain:.0%})")
            stepped_result.metadata = stepped_result.metadata or {}
            stepped_result.metadata['stepdown_info'] = (
                f"k step-down: k={lower_k} meets target with "
                f"{utility_gain:.0%} better utility than k={achieved_k} "
                f"({stepped_utility:.0%} vs {primary_utility:.0%}, "
                f"suppression {stepped_suppression:.0%} vs "
                f"{primary_suppression:.0%})")
            return stepped_result
        else:
            log.info("[StepDown] k=%d did not improve enough "
                     "(utility gain %.1f%% < %.1f%% threshold). Keeping k=%d",
                     lower_k, utility_gain * 100,
                     utility_gain_threshold * 100, achieved_k)
            log_entries.append(
                f"  Step-down: k={lower_k} utility gain {utility_gain:.0%} "
                f"< {utility_gain_threshold:.0%} threshold")
            return None

    except Exception as e:
        log.warning("[StepDown] k=%d attempt failed: %s. Keeping k=%d",
                    lower_k, e, achieved_k)
        log_entries.append(f"  Step-down: k={lower_k} failed ({e})")
        return None


def _map_from_current_risk(
    reid_current: float, reid_target: float, fallback_method: str,
) -> Optional[Dict]:
    """When perturbative method partially reduced risk, pick structural starting point."""
    gap = reid_current - reid_target
    if gap <= 0:
        return None
    if fallback_method in ('kANON', 'LOCSUPR'):
        if gap <= 0.05:
            return {'k': 3}
        elif gap <= 0.15:
            return {'k': 5}
        elif gap <= 0.30:
            return {'k': 7}
        else:
            return {'k': 10}
    return None


_PERTURBATIVE = {'PRAM', 'NOISE'}
_STRUCTURAL = {'kANON', 'LOCSUPR'}


def _map_fallback_start(
    primary_method: str,
    primary_params: Dict,
    fallback_method: str,
    reid_current: float = None,
    reid_target: float = None,
) -> Optional[Dict]:
    """Return starting parameters for a fallback method based on where the
    primary failed.

    Supports bidirectional mapping:
    - Structural → perturbative (via _CROSS_METHOD_START table)
    - Perturbative → structural (via _map_from_current_risk using current ReID)

    Returns None if no mapping is available (fallback starts from default).
    """
    # Determine which param to look up
    param_name = None
    for pn in ('k', 'p_change', 'magnitude'):
        if pn in primary_params:
            param_name = pn
            break

    # Perturbative → Structural mapping using current risk gap
    if (primary_method in _PERTURBATIVE
            and fallback_method in _STRUCTURAL
            and reid_current is not None and reid_target is not None):
        result = _map_from_current_risk(reid_current, reid_target, fallback_method)
        if result:
            log.info("[Protection] Cross-method start (perturbative→structural): "
                     "%s → %s start=%s (ReID gap: %.2f%%)",
                     primary_method, fallback_method, result,
                     (reid_current - reid_target) * 100)
            return result

    if not param_name:
        return None

    primary_val = primary_params[param_name]
    if primary_val is None:
        return None

    # Structural → perturbative mapping via equivalence table
    best_entry = None
    best_threshold = -1
    for (pm, pn, threshold), mapping in _CROSS_METHOD_START.items():
        if pm == primary_method and pn == param_name and threshold <= primary_val:
            if threshold > best_threshold and fallback_method in mapping:
                best_threshold = threshold
                best_entry = mapping[fallback_method]

    if best_entry:
        log.info("[Protection] Cross-method start: %s(%s=%s) → %s start=%s",
                 primary_method, param_name, primary_val,
                 fallback_method, best_entry)
    return best_entry


def _inject_per_qi_params(
    method: str,
    params: Dict,
    var_priority: Optional[Dict],
    quasi_identifiers: List[str],
    *,
    smart_config: Optional[Dict] = None,
) -> Dict:
    """Adjust per-QI protection strength based on var_priority risk levels.

    For kANON: inject ``per_qi_bin_size`` — HIGH-risk QIs get smaller bins.
    For PRAM: inject ``per_variable_p_change`` — HIGH-risk QIs get higher
    perturbation probability.
    For NOISE: inject ``per_variable_magnitude`` from smart config's
    IQR-proportional scaling (if available).

    Returns a new params dict (original is not mutated).
    """
    params = dict(params)  # shallow copy

    # ── NOISE: IQR-proportional magnitude from smart config ────
    if method == 'NOISE' and smart_config:
        iqr_mag = smart_config.get('per_variable_magnitude')
        if iqr_mag:
            params['per_variable_magnitude'] = iqr_mag
            log.info("[Protection] Per-QI IQR magnitude: %s", iqr_mag)

    if not var_priority:
        return params

    if method == 'kANON':
        base_bin = params.get('bin_size', 10)
        try:
            from sdc_engine.sdc.GENERALIZE import compute_risk_weighted_limits
            per_qi = compute_risk_weighted_limits(var_priority, base_bin)
            if per_qi:
                params['per_qi_bin_size'] = per_qi
                log.info("[Protection] Per-QI bin_size: %s", per_qi)
        except Exception as exc:
            log.warning("[Protection] Per-QI bin_size injection failed: %s", exc)

    elif method == 'PRAM':
        base_p = params.get('p_change', 0.20)
        per_var_p: Dict[str, float] = {}
        for qi in quasi_identifiers:
            prio = var_priority.get(qi)
            if prio:
                level = prio[0] if isinstance(prio, (tuple, list)) else str(prio)
                level_up = level.upper() if isinstance(level, str) else ''
                if 'HIGH' in level_up:
                    per_var_p[qi] = round(min(base_p * 1.5, 0.50), 3)
                elif 'LOW' in level_up:
                    per_var_p[qi] = round(base_p * 0.6, 3)
                else:
                    per_var_p[qi] = base_p
        if per_var_p:
            params['per_variable_p_change'] = per_var_p
            log.info("[Protection] Per-QI p_change: %s", per_var_p)

    return params


def _is_better(candidate, current_best) -> bool:
    """True if candidate has lower ReID95 than current_best."""
    c_reid = _get_reid_95(candidate)
    b_reid = _get_reid_95(current_best)
    if c_reid is None:
        return False
    if current_best is None or b_reid is None:
        return True
    return c_reid < b_reid


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    pipeline: List[str],
    pipeline_params: Dict[str, Dict],
    input_data: pd.DataFrame,
    quasi_identifiers: List[str],
    apply_method_fn: Callable,
    *,
    risk_metric: Optional[str] = None,
    risk_target_raw: Optional[float] = None,
) -> Any:
    """Run a multi-method pipeline sequentially.

    Parameters
    ----------
    pipeline : list of str
        Ordered method names, e.g. ['NOISE', 'kANON'].
    pipeline_params : dict
        {method_name: {param: value}} for each pipeline step.
    input_data : DataFrame
        Data to protect.
    quasi_identifiers : list of str
        QI column names.
    apply_method_fn : callable
        ``fn(method, quasi_identifiers, params, input_data=df) -> ProtectionResult``
    risk_metric : str, optional
        'reid95', 'k_anonymity', or 'uniqueness'. Default 'reid95'.
    risk_target_raw : float, optional
        Raw target for the chosen metric.

    Returns
    -------
    ProtectionResult with updated reid_before/reid_after on final output.
    """
    from sdc_engine.sdc.metrics.risk_metric import (
        RiskMetricType, compute_risk, risk_to_reid_compat,
    )

    _mt_map = {
        'reid95': RiskMetricType.REID95,
        'k_anonymity': RiskMetricType.K_ANONYMITY,
        'uniqueness': RiskMetricType.UNIQUENESS,
    }
    mt = _mt_map.get(risk_metric or 'reid95', RiskMetricType.REID95)

    current_data = input_data.copy()
    last_result = None

    for i, method in enumerate(pipeline):
        params = pipeline_params.get(method, {})
        result = apply_method_fn(
            method, quasi_identifiers, params, input_data=current_data,
        )
        if not result.success:
            raise RuntimeError(f"Pipeline step {method} failed: {result.error}")
        current_data = result.protected_data
        last_result = result

        # Mid-pipeline risk check: skip remaining steps if target already met.
        # Skip the check after perturbative steps — PRAM/NOISE don't reduce
        # structural ReID, so the check always evaluates against an unchanged
        # value and would never trigger early exit anyway.
        if i < len(pipeline) - 1 and risk_target_raw is not None:
            if method in _PERTURBATIVE:
                continue
            try:
                mid_assess = compute_risk(
                    current_data, quasi_identifiers, mt, risk_target_raw)
                mid_reid = mid_assess.normalized_score
                if method in _STRUCTURAL:
                    multiplier = 1.10
                else:
                    multiplier = 1.20  # GENERALIZE or unknown
                if mid_reid <= (risk_target_raw * multiplier):
                    log.info(
                        "[Pipeline] Mid-check after %s: ReID=%.4f — skipping %s",
                        method, mid_reid, pipeline[i + 1:])
                    break
            except Exception as exc:
                log.warning("[Pipeline] Mid-pipeline risk check failed: %s", exc)

    # Recalculate risk on final output vs original input
    if last_result and last_result.success:
        try:
            a_before = compute_risk(input_data, quasi_identifiers, mt, risk_target_raw)
            a_after = compute_risk(current_data, quasi_identifiers, mt, risk_target_raw)
            last_result.reid_before = risk_to_reid_compat(a_before)
            last_result.reid_after = risk_to_reid_compat(a_after)
            last_result.risk_assessment_before = a_before
            last_result.risk_assessment_after = a_after
        except Exception as exc:
            log.warning("[Pipeline] Post-pipeline risk recalculation failed: %s", exc)
        last_result.method = ' \u2192 '.join(pipeline)
        last_result.protected_data = current_data

    return last_result


# ---------------------------------------------------------------------------
# Main retry engine
# ---------------------------------------------------------------------------

def run_rules_engine_protection(
    input_data: pd.DataFrame,
    quasi_identifiers: List[str],
    data_features: Dict[str, Any],
    access_tier: str,
    reid_target: float,
    utility_floor: float,
    apply_method_fn: Callable,
    max_fallbacks: int = 5,
    *,
    sensitive_columns: Optional[List[str]] = None,
    qi_treatment: Optional[Dict[str, str]] = None,
    preprocess_per_var: Optional[Dict] = None,
    risk_metric: Optional[str] = None,
    risk_target_raw: Optional[float] = None,
    l_target: Optional[int] = None,
    t_target: Optional[float] = None,
    method_constraints: Optional[Dict] = None,
) -> Tuple[Any, List[str]]:
    """Run the rules-engine retry loop: pipeline -> primary + escalation -> fallbacks.

    Parameters
    ----------
    input_data : DataFrame
        Data to protect (may already be preprocessed).
    quasi_identifiers : list of str
        QI column names.
    data_features : dict
        Output from build_data_features() — must reflect *input_data*, not stale cache.
    access_tier : str
        'PUBLIC', 'SCIENTIFIC', or 'SECURE'.
    reid_target : float
        Maximum acceptable ReID95 (e.g. 0.05).
    utility_floor : float
        Minimum acceptable utility (e.g. 0.88). Stops escalation if breached.
    apply_method_fn : callable
        ``fn(method, quasi_identifiers, params, input_data=df) -> ProtectionResult``
    max_fallbacks : int
        Maximum fallback methods to try.
    sensitive_columns : list of str, optional
        Sensitive column names (for l-diversity check).
    qi_treatment : dict, optional
        ``{col: 'Heavy'|'Standard'|'Light'}`` treatment levels.
    preprocess_per_var : dict, optional
        Per-variable utility from preprocessing (for comparison).
    risk_metric : str, optional
        'reid95', 'k_anonymity', or 'uniqueness'. Default 'reid95'.
    l_target : int, optional
        Minimum distinct l-diversity per equivalence class. None = disabled.
    t_target : float, optional
        Maximum t-closeness distance (EMD/TVD). None = disabled.

    When non-REID95, *reid_target* should be the **normalized** target
    (use ``normalize_target()`` from ``risk_metric`` module).

    risk_target_raw : float, optional
        Raw target in native metric units (for RiskAssessment).

    Returns
    -------
    (best_result, log_entries) : tuple
        best_result is a ProtectionResult (or None if everything failed).
        log_entries is a list of human-readable log strings.
        Diagnostics are attached to ``best_result.metadata['auto_diagnostics']``.
    """
    from sdc_engine.sdc.selection.pipelines import select_method_suite
    from sdc_engine.sdc.config import (
        get_tuning_schedule, get_method_fallbacks,
        is_method_allowed_for_metric, filter_methods_for_metric,
    )

    t_total = time.monotonic()
    timing = {}
    escalation_count = 0

    # --- Feasibility diagnosis (max achievable k) ---
    max_achievable_k = data_features.get('max_achievable_k', 0)
    try:
        from sdc_engine.sdc.preprocessing.diagnose import diagnose_qis
        _diag = diagnose_qis(
            input_data, quasi_identifiers,
            access_tier=access_tier, verbose=False)
        max_achievable_k = _diag.max_achievable_k
        log.info("[Protection] Feasibility: status=%s  max_k=%d  "
                 "combo_space=%d  expected_eq=%.1f",
                 _diag.status.value, max_achievable_k,
                 _diag.combination_space, _diag.expected_eq_size)
    except Exception as _diag_err:
        log.debug("[Protection] diagnose_qis failed: %s", _diag_err)

    # --- Privacy target feasibility pre-check ---
    # If sensitive columns don't have enough distinct values, l-diversity
    # is impossible — disable the gate to prevent exhausting escalation.
    _effective_l_target = l_target
    _effective_t_target = t_target
    if l_target is not None and sensitive_columns:
        max_sens_nunique = 0
        for sc in sensitive_columns:
            if sc in input_data.columns:
                max_sens_nunique = max(max_sens_nunique, input_data[sc].nunique())
        if max_sens_nunique < l_target:
            log.warning(
                "[Protection] l-diversity target l=%d impossible — "
                "max sensitive nunique=%d. Disabling l-diversity gate.",
                l_target, max_sens_nunique)
            _effective_l_target = None

    # Build kwargs dict for _meets_targets (avoids repetition at 11 call sites)
    _privacy_kw: Dict[str, Any] = {
        'l_target': _effective_l_target,
        't_target': _effective_t_target,
        'sensitive_columns': sensitive_columns,
        'quasi_identifiers': quasi_identifiers,
    }

    # Inject qi_treatment so rules can apply treatment balance
    data_features['qi_treatment'] = qi_treatment or {}

    # Inject context targets so context-aware rules (PUB1/SEC1/REG1) can read them
    data_features['_reid_target_raw'] = reid_target
    data_features['_utility_floor'] = utility_floor

    suite = select_method_suite(
        features=data_features,
        access_tier=access_tier,
        verbose=False,
    )

    # --- Metric-based fallback filtering ---
    _risk_metric = data_features.get('_risk_metric_type', 'reid95')
    suite['fallbacks'] = [
        (m, p) for m, p in suite.get('fallbacks', [])
        if is_method_allowed_for_metric(m, _risk_metric)
    ]
    for _fb_key in ('reid_fallback', 'utility_fallback'):
        _fb = suite.get(_fb_key)
        if _fb and not is_method_allowed_for_metric(_fb['method'], _risk_metric):
            suite[_fb_key] = None

    best_result = None
    log_entries = []

    # --- Smart method configuration (pre-estimation) ---
    var_priority = data_features.get('var_priority')
    primary = suite.get('primary', 'kANON')

    # Derive target_k for structural methods
    _target_k = suite.get('primary_params', {}).get('k', 5)

    smart_cfg: Dict[str, Any] = {}
    try:
        from sdc_engine.sdc.smart_method_config import get_smart_config
        smart_cfg = get_smart_config(
            primary, input_data, quasi_identifiers,
            target_k=_target_k,
            base_magnitude=suite.get('primary_params', {}).get('magnitude', 0.10),
            base_p_change=suite.get('primary_params', {}).get('p_change', 0.20),
            qi_treatment=qi_treatment,
        )
        # Surface smart config warnings in the log
        for w in smart_cfg.get('warnings', []):
            log_entries.append(f"⚠ {w}")

        # Pre-application method switch: if smart config says the chosen
        # method will be ineffective, swap primary BEFORE trying it.
        suggested_switch = smart_cfg.get('switch_method')
        if suggested_switch and suggested_switch != primary:
            log.info("[Protection] Smart pre-switch: %s → %s (reason: %s)",
                     primary, suggested_switch,
                     smart_cfg.get('warnings', ['(none)'])[-1][:80])
            log_entries.append(
                f"Smart switch: {primary} → {suggested_switch} (pre-estimation)")
            # Move original primary to front of fallbacks so it's still tried
            old_primary = primary
            old_params = suite.get('primary_params', {})
            primary = suggested_switch
            suite['primary'] = primary
            from sdc_engine.sdc.config import get_method_defaults
            suite['primary_params'] = get_method_defaults(primary)
            # Prepend old primary to fallbacks
            fbs = suite.get('fallbacks', [])
            suite['fallbacks'] = [(old_primary, old_params)] + list(fbs)

            # Re-run smart config for the NEW primary
            smart_cfg = get_smart_config(
                primary, input_data, quasi_identifiers,
                target_k=suite['primary_params'].get('k', _target_k),
                base_magnitude=suite['primary_params'].get('magnitude', 0.10),
                base_p_change=suite['primary_params'].get('p_change', 0.20),
                qi_treatment=qi_treatment,
            )
            for w in smart_cfg.get('warnings', []):
                log_entries.append(f"⚠ {w}")
    except Exception as _sc_err:
        log.debug("[Protection] Smart config failed: %s", _sc_err)

    # --- Inject per-QI params based on var_priority + smart config ---
    primary_params = _inject_per_qi_params(
        primary, suite.get('primary_params', {}),
        var_priority, quasi_identifiers,
        smart_config=smart_cfg)
    suite['primary_params'] = primary_params

    # Apply kANON smart starting_k if available
    if primary == 'kANON' and smart_cfg.get('starting_k'):
        smart_k = smart_cfg['starting_k']
        if smart_k != primary_params.get('k'):
            log.info("[Protection] Smart starting k: %d (was %s)",
                     smart_k, primary_params.get('k'))
            primary_params['k'] = smart_k

    # Starting ReID for smart escalation
    reid_current = data_features.get('reid_95', 1.0)

    log.info("[Protection] Rule: %s -> primary=%s | tier=%s | ReID95=%.4f | "
             "target=%.4f | utility_floor=%.0f%%",
             suite.get('rule_applied', 'UNKNOWN'), suite.get('primary', '?'),
             access_tier, reid_current, reid_target, utility_floor * 100)
    if suite.get('pipeline'):
        log.info("[Protection] Pipeline recommended: %s",
                 ' -> '.join(suite['pipeline']))
    if suite.get('fallbacks'):
        log.info("[Protection] Fallback chain: %s",
                 [fb[0] for fb in suite.get('fallbacks', [])])

    log_entries.append(f"Rule: {suite.get('rule_applied', 'UNKNOWN')} \u2192 {suite.get('primary', '?')}")

    # Log risk-concentration hint when available (RC1/RC2/RC3 rules)
    hint = suite.get('preprocessing_hint')
    if hint:
        log_entries.append(
            f"Risk hint: '{hint.get('aggressive_qi', '?')}' dominates risk "
            f"({hint.get('pct', 0):.0f}%)"
        )

    # Snapshot original data before pipeline/GENERALIZE_FIRST may modify it.
    # GENERALIZE_FIRST (Step 1b) must always start from the original state,
    # not from pipeline-modified data that failed to meet targets.
    _pre_generalize_data = input_data.copy()

    # --- Step 1: Try pipeline if recommended ---
    t_phase = time.monotonic()
    if suite.get('use_pipeline') and suite.get('pipeline'):
        pipeline = suite['pipeline']
        pipeline_params = suite.get('parameters', {})
        log_entries.append(f"Trying pipeline: {' \u2192 '.join(pipeline)}")
        try:
            result = run_pipeline(
                pipeline, pipeline_params, input_data,
                quasi_identifiers, apply_method_fn,
                risk_metric=risk_metric, risk_target_raw=risk_target_raw,
            )
            if result.success:
                best_result = result
                reid_val = _get_reid_95(result)
                log_entries.append(
                    f"Pipeline succeeded \u2014 ReID95: {reid_val:.4f}" if reid_val else "Pipeline succeeded")
                if _meets_targets(result, reid_target, **_privacy_kw):
                    log_entries.append("\u2713 Targets met")
                    timing['pipeline'] = round(time.monotonic() - t_phase, 2)
                    timing['total'] = round(time.monotonic() - t_total, 2)
                    timing['escalation_steps'] = 0
                    # Perturbative challenge: try PRAM as potentially better alternative
                    if best_result and best_result.success:
                        _challenger = _try_perturbative_challenge(
                            structural_result=best_result,
                            structural_method=getattr(best_result, 'method', '') or '',
                            input_data=input_data,
                            quasi_identifiers=quasi_identifiers,
                            data_features=data_features,
                            apply_method_fn=apply_method_fn,
                            risk_target_raw=risk_target_raw,
                            log_entries=log_entries,
                        )
                        if _challenger is not None:
                            best_result = _challenger
                    _attach_diagnostics(
                        best_result, log_entries, data_features,
                        quasi_identifiers, reid_target, timing,
                        sensitive_columns, qi_treatment, preprocess_per_var,
                        l_target=_effective_l_target, t_target=_effective_t_target,
                        original_data=_pre_generalize_data)
                    return best_result, log_entries
        except Exception as e:
            log_entries.append(f"Pipeline failed: {e}")
    timing['pipeline'] = round(time.monotonic() - t_phase, 2)

    # --- Step 1b: Handle GENERALIZE_FIRST (preprocess infeasible QI space) ---
    # Restore from snapshot so generalization starts from clean original data,
    # not from a pipeline that partially modified it and failed.
    if primary == 'GENERALIZE_FIRST':
        input_data = _pre_generalize_data
        t_gen = time.monotonic()
        log_entries.append("GENERALIZE_FIRST: preprocessing high-cardinality QIs")
        try:
            from sdc_engine.sdc.GENERALIZE import apply_generalize
            rec_qi = primary_params.get('recommended_qi_to_remove')
            # Build column_types so date columns are properly detected
            import warnings as _w
            _gen_col_types = {}
            for _qi in quasi_identifiers:
                if _qi in input_data.columns:
                    if pd.api.types.is_datetime64_any_dtype(input_data[_qi]):
                        _gen_col_types[_qi] = 'date'
                    elif pd.api.types.is_numeric_dtype(input_data[_qi]):
                        _gen_col_types[_qi] = 'numeric'
                    else:
                        # Probe for date strings (ISO first, then dayfirst)
                        _sample = input_data[_qi].dropna().head(50)
                        if len(_sample) > 0:
                            try:
                                with _w.catch_warnings():
                                    _w.simplefilter("ignore", UserWarning)
                                    _parsed = pd.to_datetime(_sample, errors='coerce')
                                    if _parsed.notna().mean() <= 0.8:
                                        _parsed = pd.to_datetime(_sample, errors='coerce', dayfirst=True)
                                if _parsed.notna().mean() > 0.8:
                                    _gen_col_types[_qi] = 'date'
                            except Exception as exc:
                                log.debug("[GENERALIZE_FIRST] Date detection for %s failed: %s", _qi, exc)
            # Aggressive generalization: max_categories=5 to actually reduce cardinality
            gen_data, gen_meta = apply_generalize(
                input_data.copy(), quasi_identifiers=quasi_identifiers,
                max_categories=5, strategy='auto',
                column_types=_gen_col_types,
                return_metadata=True, verbose=False,
            )
            applied = gen_meta.get('generalization_applied', {})
            log_entries.append(
                f"Generalized: {applied or 'auto bins'} "
                f"({time.monotonic() - t_gen:.1f}s)")
            log.info("[Protection] GENERALIZE_FIRST applied: %s", applied)

            # Rebuild features on generalized data
            input_data = gen_data
            data_features = build_data_features(
                input_data, quasi_identifiers,
                var_priority=var_priority,
                column_types=None,  # re-detect types after generalization
                risk_metric=risk_metric,
                risk_target_raw=risk_target_raw,
                sensitive_columns=sensitive_columns,
            )
            data_features['qi_treatment'] = qi_treatment or {}

            # Re-select method on the generalized data
            suite = select_method_suite(
                features=data_features,
                access_tier=access_tier,
                verbose=False,
            )
            # Re-apply metric-based fallback filtering
            suite['fallbacks'] = [
                (m, p) for m, p in suite.get('fallbacks', [])
                if is_method_allowed_for_metric(m, _risk_metric)
            ]
            for _fb_key in ('reid_fallback', 'utility_fallback'):
                _fb = suite.get(_fb_key)
                if _fb and not is_method_allowed_for_metric(_fb['method'], _risk_metric):
                    suite[_fb_key] = None
            primary = suite.get('primary', 'kANON')
            primary_params = suite.get('primary_params', {})

            # If still GENERALIZE_FIRST (extremely infeasible), fall back to LOCSUPR
            if primary == 'GENERALIZE_FIRST':
                log_entries.append(
                    "Still infeasible after generalization — falling back to LOCSUPR k=3")
                primary = 'LOCSUPR'
                primary_params = {'quasi_identifiers': quasi_identifiers, 'k': 3}
                suite['primary'] = primary
                suite['primary_params'] = primary_params
            else:
                log_entries.append(f"Re-selected after GENERALIZE: {suite.get('rule_applied','?')} -> {primary}")
                # Re-inject per-QI params for new primary
                primary_params = _inject_per_qi_params(
                    primary, primary_params,
                    var_priority, quasi_identifiers,
                    smart_config={})
                suite['primary_params'] = primary_params

            reid_current = data_features.get('reid_95', 1.0)
        except Exception as _gen_err:
            log.warning("[Protection] GENERALIZE_FIRST failed: %s", _gen_err)
            log_entries.append(f"GENERALIZE_FIRST failed: {_gen_err} — falling back to LOCSUPR k=3")
            primary = 'LOCSUPR'
            primary_params = {'quasi_identifiers': quasi_identifiers, 'k': 3}
            suite['primary'] = primary
            suite['primary_params'] = primary_params

    # --- Step 2: Try primary method with escalation ---
    t_phase = time.monotonic()
    # primary and primary_params already set above (with per-QI injection)
    log.info("[Protection] Trying primary: %s  params=%s", primary, primary_params)
    log_entries.append(f"Trying primary: {primary}")

    result = apply_method_fn(
        primary, quasi_identifiers, primary_params, input_data=input_data,
    )
    if result.success:
        reid_val = _get_reid_95(result)
        u_score = _get_utility_score(result)

        # ── QI over-suppression guard ──────────────────────────────
        # If a structural method suppressed >40% of any QI column,
        # don't accept the result — fall through to fallbacks.
        _qi_over_supp = getattr(result, 'qi_over_suppressed', False)
        if _qi_over_supp:
            _supp_detail = getattr(result, 'qi_suppression_detail', {})
            _worst_qi = max(_supp_detail, key=_supp_detail.get) if _supp_detail else '?'
            _worst_pct = _supp_detail.get(_worst_qi, 0)
            log.warning("[Protection] Primary %s over-suppressed QI '%s' (%.0f%%) — "
                        "rejecting, trying fallbacks",
                        primary, _worst_qi, _worst_pct * 100)
            log_entries.append(
                f"Primary {primary} rejected — QI '{_worst_qi}' over-suppressed "
                f"({_worst_pct:.0%}), trying fallbacks")
            # Keep as best_result candidate only if nothing else works
            if _is_better(result, best_result):
                best_result = result
        else:
            log.info("[Protection] Primary %s result: ReID95=%.4f  Utility=%s  meets_target=%s",
                     primary, reid_val or -1,
                     f"{u_score:.2%}" if u_score is not None else "N/A",
                     _meets_targets(result, reid_target, **_privacy_kw))

        msg = f"Primary {primary} \u2014 ReID95: {reid_val:.4f}" if reid_val else f"Primary {primary} applied"
        if u_score is not None:
            msg += f", Utility: {u_score:.2%}"
        log_entries.append(msg)
        if not _qi_over_supp and _meets_targets(result, reid_target, **_privacy_kw):
            best_result = result
            log_entries.append("\u2713 Targets met")
            # Log smart_config details for fast-path exits (debugging aid)
            if smart_cfg:
                sc_items = []
                if smart_cfg.get('starting_k'):
                    sc_items.append(f"starting_k={smart_cfg['starting_k']}")
                if smart_cfg.get('per_variable_magnitude'):
                    sc_items.append("per-QI magnitudes")
                if smart_cfg.get('switch_method'):
                    sc_items.append(f"switch→{smart_cfg['switch_method']}")
                if sc_items:
                    log_entries.append(f"  Smart config: {', '.join(sc_items)}")
            timing['primary_escalation'] = round(time.monotonic() - t_phase, 2)
            timing['total'] = round(time.monotonic() - t_total, 2)
            timing['escalation_steps'] = 0
            # Perturbative challenge: try PRAM as potentially better alternative
            if best_result and best_result.success:
                _challenger = _try_perturbative_challenge(
                    structural_result=best_result,
                    structural_method=getattr(best_result, 'method', '') or '',
                    input_data=input_data,
                    quasi_identifiers=quasi_identifiers,
                    data_features=data_features,
                    apply_method_fn=apply_method_fn,
                    risk_target_raw=risk_target_raw,
                    log_entries=log_entries,
                )
                if _challenger is not None:
                    best_result = _challenger
            _attach_diagnostics(
                best_result, log_entries, data_features,
                quasi_identifiers, reid_target, timing,
                sensitive_columns, qi_treatment, preprocess_per_var,
                l_target=_effective_l_target, t_target=_effective_t_target,
                original_data=_pre_generalize_data)
            return best_result, log_entries
        elif _is_better(result, best_result):
            best_result = result
        # Update ReID current from actual result
        if reid_val is not None:
            reid_current = reid_val

    # Escalate parameters with smart start
    schedule = get_tuning_schedule(primary)
    if schedule and schedule.get('values'):
        param_name = schedule['parameter']
        current_val = primary_params.get(param_name)
        esc_values = schedule['values']

        # Cardinality-aware k pruning — skip impossible k values
        esc_values = _prune_schedule_by_max_k(esc_values, max_achievable_k, param_name)
        if not esc_values:
            log.info("[Protection] All escalation values pruned (max_k=%d)",
                     max_achievable_k)

        # Smart escalation start
        start_idx = _pick_escalation_start(esc_values, reid_current, reid_target)
        log.info("[Protection] Escalation schedule for %s: param=%s  values=%s  start_idx=%d",
                 primary, param_name, esc_values, start_idx)
        if start_idx > 0:
            log_entries.append(
                f"  Smart start: skipping to schedule[{start_idx}] "
                f"(ReID gap: {reid_current:.2%} vs {reid_target:.2%})")

        prev_esc_reid = None
        no_improvement_count = 0
        t_esc_start = time.monotonic()
        ESC_TIME_BUDGET = 30  # seconds — bail to fallbacks if exceeded
        for esc_val in esc_values[start_idx:]:
            if current_val is not None and esc_val <= current_val:
                continue
            escalation_count += 1
            esc_params = {**primary_params, param_name: esc_val}
            log_entries.append(f"  Escalating {primary}: {param_name}={esc_val}")
            result = apply_method_fn(
                primary, quasi_identifiers, esc_params, input_data=input_data,
            )
            if result.success:
                reid_val = _get_reid_95(result)
                u_score = _get_utility_score(result)

                # QI over-suppression guard during escalation
                _esc_over_supp = getattr(result, 'qi_over_suppressed', False)
                if _esc_over_supp:
                    _esc_detail = getattr(result, 'qi_suppression_detail', {})
                    _esc_worst = max(_esc_detail, key=_esc_detail.get) if _esc_detail else '?'
                    _esc_worst_pct = _esc_detail.get(_esc_worst, 0)
                    log_entries.append(
                        f"    QI over-suppressed at {param_name}={esc_val}: "
                        f"'{_esc_worst}' ({_esc_worst_pct:.0%}) — stopping escalation")
                    log.info("[Protection] Escalation %s=%s over-suppressed QI '%s' (%.0f%%) — "
                             "stopping", param_name, esc_val, _esc_worst, _esc_worst_pct * 100)
                    if _is_better(result, best_result):
                        best_result = result
                    break

                if reid_val is not None:
                    msg = f"    ReID95: {reid_val:.4f}"
                    if u_score is not None:
                        msg += f", Utility: {u_score:.2%}"
                    log_entries.append(msg)

                    # Plateau detection: if ReID didn't improve, skip to fallbacks
                    if prev_esc_reid is not None and reid_val >= prev_esc_reid - 0.001:
                        no_improvement_count += 1
                        log_entries.append(
                            f"    ⚠ ReID plateau ({reid_val:.4f}) after "
                            f"{no_improvement_count} attempts — skipping to fallbacks")
                        log.info("[Protection] ReID plateau at %.4f — skipping to fallbacks",
                                 reid_val)
                        break
                    else:
                        no_improvement_count = 0
                    prev_esc_reid = reid_val

                    # Time guard: bail if escalation is taking too long
                    if time.monotonic() - t_esc_start > ESC_TIME_BUDGET:
                        log_entries.append(
                            f"    ⚠ Escalation time budget ({ESC_TIME_BUDGET}s) exceeded "
                            f"— skipping to fallbacks")
                        break

                if not _utility_ok(result, utility_floor):
                    log_entries.append(
                        f"    \u26a0 Utility below floor ({utility_floor:.0%}), stopping escalation")
                    break
                if _meets_targets(result, reid_target, **_privacy_kw):
                    best_result = result
                    log_entries.append("\u2713 Targets met after escalation")
                elif _meets_targets(result, reid_target):
                    # ReID passed but privacy targets not yet satisfied
                    _reasons = []
                    if _effective_l_target and not _quick_l_diversity_ok(
                            result.protected_data, quasi_identifiers,
                            sensitive_columns or [], _effective_l_target):
                        _reasons.append(f"l-diversity < {_effective_l_target}")
                    if _effective_t_target and not _quick_t_closeness_ok(
                            result.protected_data, quasi_identifiers,
                            sensitive_columns or [], _effective_t_target):
                        _reasons.append(f"t-closeness > {_effective_t_target}")
                    if _reasons:
                        log_entries.append(
                            f"    ReID met but {', '.join(_reasons)} — continuing escalation")
                    timing['primary_escalation'] = round(time.monotonic() - t_phase, 2)
                    timing['total'] = round(time.monotonic() - t_total, 2)
                    timing['escalation_steps'] = escalation_count
                    # Perturbative challenge: try PRAM as potentially better alternative
                    if best_result and best_result.success:
                        _challenger = _try_perturbative_challenge(
                            structural_result=best_result,
                            structural_method=getattr(best_result, 'method', '') or '',
                            input_data=input_data,
                            quasi_identifiers=quasi_identifiers,
                            data_features=data_features,
                            apply_method_fn=apply_method_fn,
                            risk_target_raw=risk_target_raw,
                            log_entries=log_entries,
                        )
                        if _challenger is not None:
                            best_result = _challenger
                    _attach_diagnostics(
                        best_result, log_entries, data_features,
                        quasi_identifiers, reid_target, timing,
                        sensitive_columns, qi_treatment, preprocess_per_var,
                        l_target=_effective_l_target, t_target=_effective_t_target,
                        original_data=_pre_generalize_data)
                    return best_result, log_entries
                elif _is_better(result, best_result):
                    best_result = result
    timing['primary_escalation'] = round(time.monotonic() - t_phase, 2)

    # --- Step 3: Try fallback methods with escalation ---
    t_phase = time.monotonic()
    _mc = method_constraints or {}
    _excluded_methods = set(_mc.get('excluded', []))
    fallbacks = [(m, p) for m, p in suite.get('fallbacks', [])
                 if m not in _excluded_methods]
    config_fbs = filter_methods_for_metric(
        get_method_fallbacks(primary), _risk_metric)
    for cfb in config_fbs:
        if cfb in _excluded_methods:
            continue
        if not any(fb[0] == cfb for fb in fallbacks):
            fallbacks.append((cfb, {}))

    # Filter out perturbation-only fallbacks (PRAM/NOISE) when the ReID gap
    # requires structural reduction.  PRAM shuffles categories without changing
    # equivalence-class sizes, so ReID stays flat; NOISE only works on continuous
    # columns.  Trying + escalating them wastes time when ReID still exceeds
    # target — they cannot close a structural gap.
    _best_reid = _get_reid_95(best_result) if best_result else reid_current
    if _best_reid is not None and _best_reid > reid_target:
        _before_len = len(fallbacks)
        fallbacks = [(m, p) for m, p in fallbacks
                     if m.upper() not in ('PRAM', 'NOISE')]
        if len(fallbacks) < _before_len:
            _skipped = _before_len - len(fallbacks)
            log.info("[Protection] Skipped %d perturbation fallback(s) — "
                     "ReID %.2f%% too high for PRAM/NOISE", _skipped, _best_reid * 100)
            log_entries.append(
                f"Skipped PRAM/NOISE fallbacks (ReID {_best_reid:.1%} > "
                f"target {reid_target:.1%} — need structural method)")

    log.info("[Protection] Starting fallbacks: %s", [fb[0] for fb in fallbacks[:max_fallbacks]])
    for fb_method, fb_params in fallbacks[:max_fallbacks]:
        if _meets_targets(best_result, reid_target, **_privacy_kw):
            break

        # Cross-method start: use primary's failure point to set
        # intelligent starting parameters for the fallback (bidirectional)
        xm_start = _map_fallback_start(
            primary, primary_params, fb_method,
            reid_current=reid_current, reid_target=reid_target)
        if xm_start:
            fb_params = {**fb_params, **xm_start}

        # Per-QI injection for fallback method too
        # Run smart config for fallback to get IQR/dominance info
        fb_smart_cfg: Dict[str, Any] = {}
        try:
            from sdc_engine.sdc.smart_method_config import get_smart_config
            fb_smart_cfg = get_smart_config(
                fb_method, input_data, quasi_identifiers,
                target_k=fb_params.get('k', _target_k),
                base_magnitude=fb_params.get('magnitude', 0.10),
                base_p_change=fb_params.get('p_change', 0.20),
                qi_treatment=qi_treatment,
            )
        except Exception as exc:
            log.debug("[Protection] Smart config for fallback %s failed: %s",
                      fb_method, exc)
        fb_params = _inject_per_qi_params(
            fb_method, fb_params, var_priority, quasi_identifiers,
            smart_config=fb_smart_cfg)

        log.info("[Protection] Trying fallback: %s  params=%s", fb_method, fb_params)
        log_entries.append(f"Trying fallback: {fb_method}")
        result = apply_method_fn(
            fb_method, quasi_identifiers, fb_params, input_data=input_data,
        )
        if result.success:
            # QI over-suppression guard for fallbacks
            _fb_over_supp = getattr(result, 'qi_over_suppressed', False)
            if _fb_over_supp:
                _fb_detail = getattr(result, 'qi_suppression_detail', {})
                log.warning("[Protection] Fallback %s over-suppressed QI(s) %s — skipping",
                            fb_method, _fb_detail)
                log_entries.append(f"  {fb_method} rejected — QI over-suppressed")
                if _is_better(result, best_result):
                    best_result = result
                continue

            reid_val = _get_reid_95(result)
            if reid_val is not None:
                log_entries.append(f"  {fb_method} \u2014 ReID95: {reid_val:.4f}")
                reid_current = reid_val  # Track for next fallback's cross-method start
            if _meets_targets(result, reid_target, **_privacy_kw):
                best_result = result
                log_entries.append("\u2713 Targets met with fallback")
                break
            elif _is_better(result, best_result):
                best_result = result

            # Escalate fallback with smart start + k-pruning
            fb_schedule = get_tuning_schedule(fb_method)
            if fb_schedule and fb_schedule.get('values'):
                fb_param_name = fb_schedule['parameter']
                fb_current_val = fb_params.get(fb_param_name)
                fb_values = _prune_schedule_by_max_k(
                    fb_schedule['values'], max_achievable_k, fb_param_name)

                fb_reid = _get_reid_95(result) or reid_current
                fb_start = _pick_escalation_start(
                    fb_values, fb_reid, reid_target)

                fb_prev_reid = None
                fb_no_improvement = 0
                t_fb_esc_start = time.monotonic()
                for esc_val in fb_values[fb_start:]:
                    if fb_current_val is not None and esc_val <= fb_current_val:
                        continue
                    escalation_count += 1
                    esc_params = {**fb_params, fb_param_name: esc_val}
                    result = apply_method_fn(
                        fb_method, quasi_identifiers, esc_params,
                        input_data=input_data,
                    )
                    if result.success:
                        # QI over-suppression guard during fallback escalation
                        _fbe_over_supp = getattr(result, 'qi_over_suppressed', False)
                        if _fbe_over_supp:
                            _fbe_detail = getattr(result, 'qi_suppression_detail', {})
                            _fbe_worst = max(_fbe_detail, key=_fbe_detail.get) if _fbe_detail else '?'
                            log_entries.append(
                                f"    {fb_method} QI over-suppressed at "
                                f"{fb_param_name}={esc_val} — stopping escalation")
                            log.info("[Protection] Fallback %s escalation over-suppressed "
                                     "QI '%s' — stopping", fb_method, _fbe_worst)
                            if _is_better(result, best_result):
                                best_result = result
                            break

                        reid_val = _get_reid_95(result)
                        if reid_val is not None:
                            reid_current = reid_val  # Track for next fallback
                            # Plateau detection (mirrors primary escalation)
                            if fb_prev_reid is not None and reid_val >= fb_prev_reid - 0.001:
                                fb_no_improvement += 1
                                log_entries.append(
                                    f"    ⚠ {fb_method} plateau ({reid_val:.4f}) "
                                    f"— skipping to next fallback")
                                break
                            else:
                                fb_no_improvement = 0
                            fb_prev_reid = reid_val

                            # Time guard for fallback escalation
                            if time.monotonic() - t_fb_esc_start > ESC_TIME_BUDGET:
                                log_entries.append(
                                    f"    ⚠ Fallback escalation time budget exceeded "
                                    f"— skipping to next fallback")
                                break
                        if not _utility_ok(result, utility_floor):
                            log_entries.append(
                                f"    \u26a0 Utility below floor ({utility_floor:.0%})")
                            break
                        if _meets_targets(result, reid_target, **_privacy_kw):
                            best_result = result
                            log_entries.append(
                                f"\u2713 Targets met: {fb_method} {fb_param_name}={esc_val}")
                            break
                        elif _is_better(result, best_result):
                            best_result = result
                if _meets_targets(best_result, reid_target, **_privacy_kw):
                    break
    timing['fallbacks'] = round(time.monotonic() - t_phase, 2)

    # Final log
    if best_result and not _meets_targets(best_result, reid_target, **_privacy_kw):
        reid_val = _get_reid_95(best_result)
        if reid_val is not None:
            log_entries.append(
                f"\u26a0 Best result ReID95={reid_val:.4f}, target={reid_target} \u2014 not fully met")

        # --- Advisory: ensure_feasibility suggestion ---
        # When all methods fail, run ensure_feasibility to suggest QI
        # reduction.  We do NOT auto-reduce QIs — just attach the
        # suggestion for the UI to surface.
        try:
            from sdc_engine.sdc.preprocessing.diagnose import ensure_feasibility
            feasibility_result = ensure_feasibility(
                input_data, quasi_identifiers,
                target_k=5, access_tier=access_tier,
                verbose=False)
            if best_result.metadata is None:
                best_result.metadata = {}
            best_result.metadata['feasibility_suggestion'] = {
                'status': feasibility_result.status.value,
                'final_qis': feasibility_result.final_qis,
                'final_k': feasibility_result.final_k,
                'removed_qis': [
                    qi for qi in quasi_identifiers
                    if qi not in feasibility_result.final_qis],
                'fallbacks_applied': feasibility_result.fallbacks_applied,
                'warnings': feasibility_result.warnings,
            }
            removed = [qi for qi in quasi_identifiers
                       if qi not in feasibility_result.final_qis]
            if removed:
                log.info("[Protection] Feasibility suggestion: remove QIs %s "
                         "to achieve k=%d",
                         removed, feasibility_result.final_k)
                log_entries.append(
                    f"\u26a0 Suggestion: remove QIs {removed} to achieve "
                    f"k={feasibility_result.final_k}")
        except Exception as _feas_err:
            log.debug("[Protection] ensure_feasibility failed: %s", _feas_err)

    # --- Post-success optimisation (step-down then perturbative challenge) ---
    if best_result and _meets_targets(best_result, reid_target, **_privacy_kw):
        # Check 1: Could a lower k also work? (same method, lighter params)
        stepdown_result = _step_down_k(
            input_data=input_data,
            best_result=best_result,
            data_features=data_features,
            quasi_identifiers=quasi_identifiers,
            reid_target=reid_target,
            target_k=_target_k,
            apply_method_fn=apply_method_fn,
            log_entries=log_entries,
        )
        if stepdown_result is not None:
            best_result = stepdown_result
            log.info("[Protection] Accepted k step-down")

        # Check 2: Could PRAM beat the (possibly stepped-down) structural result?
        challenge_result = _try_perturbative_challenge(
            structural_result=best_result,
            structural_method=getattr(best_result, 'method', '') or '',
            input_data=input_data,
            quasi_identifiers=quasi_identifiers,
            data_features=data_features,
            apply_method_fn=apply_method_fn,
            risk_target_raw=risk_target_raw,
            log_entries=log_entries,
        )
        if challenge_result is not None:
            best_result = challenge_result
            log.info("[Protection] Accepted perturbative challenge")

    timing['total'] = round(time.monotonic() - t_total, 2)
    timing['escalation_steps'] = escalation_count
    _best_reid = _get_reid_95(best_result) if best_result else None
    log.info("[Protection] DONE in %.2fs | escalation_steps=%d | best_ReID95=%s | "
             "target_met=%s | timing=%s",
             timing['total'], escalation_count,
             f"{_best_reid:.4f}" if _best_reid else "N/A",
             _meets_targets(best_result, reid_target, **_privacy_kw), timing)

    # Attach diagnostics to result metadata
    _attach_diagnostics(
        best_result, log_entries, data_features,
        quasi_identifiers, reid_target, timing,
        sensitive_columns, qi_treatment, preprocess_per_var,
        l_target=_effective_l_target, t_target=_effective_t_target,
        original_data=_pre_generalize_data)

    return best_result, log_entries


def _attach_diagnostics(
    best_result: Any,
    log_entries: List[str],
    data_features: Dict,
    quasi_identifiers: List[str],
    reid_target: float,
    timing: Dict,
    sensitive_columns: Optional[List[str]],
    qi_treatment: Optional[Dict[str, str]],
    preprocess_per_var: Optional[Dict],
    *,
    l_target: Optional[int] = None,
    t_target: Optional[float] = None,
    original_data: Optional[pd.DataFrame] = None,
) -> None:
    """Build and attach diagnostics dict to best_result.metadata."""
    if not best_result:
        return

    diagnostics: Dict[str, Any] = {'timing': timing}

    try:
        from sdc_engine.sdc.post_protection_diagnostics import (
            compare_qi_utility, check_l_diversity,
            check_entropy_l_diversity, check_t_closeness,
            assess_method_quality, build_failure_guidance,
        )

        if best_result.success:
            diagnostics['qi_utility_comparison'] = compare_qi_utility(
                best_result, preprocess_per_var, quasi_identifiers)

            if sensitive_columns:
                _diag_l = l_target if l_target is not None else 2
                _diag_t = t_target if t_target is not None else 0.30
                diagnostics['l_diversity'] = check_l_diversity(
                    best_result.protected_data,
                    quasi_identifiers,
                    sensitive_columns,
                    l_target=_diag_l,
                )
                diagnostics['entropy_l_diversity'] = check_entropy_l_diversity(
                    best_result.protected_data,
                    quasi_identifiers,
                    sensitive_columns,
                    l_target=_diag_l,
                )
                diagnostics['t_closeness'] = check_t_closeness(
                    best_result.protected_data,
                    quasi_identifiers,
                    sensitive_columns,
                    t_target=_diag_t,
                )

            diagnostics['method_quality'] = assess_method_quality(
                best_result, quasi_identifiers, qi_treatment)

        if not _meets_targets(best_result, reid_target):
            diagnostics['failure_guidance'] = build_failure_guidance(
                log_entries, data_features, quasi_identifiers, qi_treatment)

        # --- ML Utility (optional, non-blocking) ---
        if (best_result.success and original_data is not None
                and best_result.protected_data is not None
                and sensitive_columns):
            try:
                from sdc_engine.sdc.metrics.ml_utility import compute_ml_utility_multi
                ml_results = compute_ml_utility_multi(
                    original_data, best_result.protected_data,
                    sensitive_columns, quasi_identifiers,
                )
                if ml_results:
                    diagnostics['ml_utility'] = ml_results
            except Exception as exc:
                log.debug("[Diagnostics] ML utility failed: %s", exc)

    except Exception as exc:
        log.debug("[Diagnostics] Post-protection diagnostics failed: %s", exc)

    # Extract hierarchy info from kANON metadata (if hierarchies were used)
    if best_result.metadata and isinstance(best_result.metadata, dict):
        _method_params = best_result.metadata.get('parameters', {})
        # The hierarchy_objects are built inside apply_kanon; extract info
        # from the gen_config if available via the method's metadata
        try:
            from sdc_engine.sdc.hierarchies import Hierarchy
            # Try to rebuild hierarchy info from the data and QIs
            if best_result.method and best_result.method.upper() == 'KANON':
                from sdc_engine.sdc.hierarchies import build_hierarchy_for_column
                hier_info = {}
                _col_types = _method_params.get('column_types') or {}
                for qi in quasi_identifiers:
                    if best_result.protected_data is not None and qi in best_result.protected_data.columns:
                        h = build_hierarchy_for_column(qi, best_result.protected_data, _col_types)
                        if h is not None:
                            hier_info[qi] = {
                                'builder_type': h.builder_type,
                                'max_level': h.max_level,
                                'level_used': '?',  # Not tracked at this point
                                'info_loss': h.info_loss_at(h.max_level),
                            }
                if hier_info:
                    diagnostics['hierarchy_info'] = hier_info
        except Exception as exc:
            log.debug("[Diagnostics] Hierarchy info extraction failed: %s", exc)

    if best_result.metadata is None:
        best_result.metadata = {}
    best_result.metadata['auto_diagnostics'] = diagnostics
