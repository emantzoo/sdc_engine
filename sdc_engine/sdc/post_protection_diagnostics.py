"""
Post-Protection Diagnostics
============================

Pure-function module for analysing Auto-Protect results.
No Panel / UI imports — only pandas + stdlib + math.

Provides:
- compare_qi_utility():          per-QI utility before (preprocess) vs after (protection)
- check_l_diversity():           l-diversity check on equivalence classes
- check_entropy_l_diversity():   entropy-based l-diversity (stronger, penalises skew)
- check_t_closeness():           t-closeness check (distribution distance per class)
- assess_method_quality():       method-specific quality checks from existing metadata
- build_failure_guidance():      bottleneck identification when all methods fail
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 3a. Per-QI utility comparison
# ---------------------------------------------------------------------------

def compare_qi_utility(
    result: Any,
    preprocess_per_var: Optional[Dict],
    qi_list: List[str],
) -> List[Dict]:
    """Compare per-QI utility before vs after protection.

    Parameters
    ----------
    result : ProtectionResult
        Must have ``per_variable_utility`` attribute.
    preprocess_per_var : dict or None
        Per-variable utility dict from preprocessing (before protection).
    qi_list : list of str
        QI column names.

    Returns
    -------
    list of dict
        ``[{qi, preprocess_util, protected_util, delta, verdict}]``
        sorted by delta ascending (worst degradation first).
    """
    protected_pv = getattr(result, 'per_variable_utility', None) or {}
    preprocess_pv = preprocess_per_var or {}

    rows: List[Dict] = []
    for qi in qi_list:
        pre = preprocess_pv.get(qi)
        post = protected_pv.get(qi)

        # Extract numeric score — per_variable_utility values may be dicts
        pre_score = _extract_score(pre)
        post_score = _extract_score(post)

        if pre_score is None and post_score is None:
            continue

        delta = None
        verdict = 'n/a'
        if pre_score is not None and post_score is not None:
            delta = post_score - pre_score  # negative = degradation
            abs_d = abs(delta)
            if abs_d < 0.05:
                verdict = 'minimal'
            elif abs_d < 0.15:
                verdict = 'moderate'
            else:
                verdict = 'significant'

        rows.append({
            'qi': qi,
            'preprocess_util': round(pre_score, 4) if pre_score is not None else None,
            'protected_util': round(post_score, 4) if post_score is not None else None,
            'delta': round(delta, 4) if delta is not None else None,
            'verdict': verdict,
        })

    # Sort by delta ascending (worst first), None values last
    rows.sort(key=lambda r: r['delta'] if r['delta'] is not None else 0)
    return rows


def _extract_score(value) -> Optional[float]:
    """Extract numeric utility score from a per-variable entry.

    Handles both ``float`` values and ``dict`` values
    (e.g. ``{'score': 0.95, 'correlation': 0.98, ...}``).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ('score', 'utility', 'correlation', 'preservation'):
            if key in value and isinstance(value[key], (int, float)):
                return float(value[key])
    return None


# ---------------------------------------------------------------------------
# 3b. l-Diversity check
# ---------------------------------------------------------------------------

def check_l_diversity(
    protected_data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    l_target: int = 2,
    size_threshold: int = 50,
) -> Dict:
    """Check l-diversity: do equivalence classes have diverse sensitive values?

    Only checks equivalence classes with ≤ *size_threshold* records for
    performance — large classes almost certainly have diverse values.

    Parameters
    ----------
    protected_data : DataFrame
        Protected dataset.
    quasi_identifiers : list of str
        QI column names.
    sensitive_columns : list of str
        Sensitive column names to check diversity for.
    l_target : int
        Minimum required distinct sensitive values per class.
    size_threshold : int
        Only check classes with ≤ this many records.

    Returns
    -------
    dict with keys: l_achieved, l_target, satisfied, violations,
        total_classes, violation_rate, per_sensitive, classes_checked.
    """
    if not sensitive_columns or not quasi_identifiers:
        return {
            'l_achieved': None,
            'l_target': l_target,
            'satisfied': True,
            'violations': 0,
            'total_classes': 0,
            'violation_rate': 0.0,
            'per_sensitive': {},
            'classes_checked': 0,
        }

    # Only use QIs that exist in the data
    valid_qis = [q for q in quasi_identifiers if q in protected_data.columns]
    valid_sens = [s for s in sensitive_columns if s in protected_data.columns]

    if not valid_qis or not valid_sens:
        return {
            'l_achieved': None,
            'l_target': l_target,
            'satisfied': True,
            'violations': 0,
            'total_classes': 0,
            'violation_rate': 0.0,
            'per_sensitive': {},
            'classes_checked': 0,
        }

    grouped = protected_data.groupby(valid_qis, observed=True)
    total_classes = grouped.ngroups

    # Filter to small classes only (where violations concentrate)
    group_sizes = grouped.size()
    small_group_keys = group_sizes[group_sizes <= size_threshold].index

    per_sensitive: Dict[str, Dict] = {}
    global_min_l = float('inf')
    total_violations = 0
    classes_checked = len(small_group_keys)

    for scol in valid_sens:
        col_min_l = float('inf')
        col_violations = 0

        for key in small_group_keys:
            if isinstance(key, tuple):
                group = grouped.get_group(key)
            else:
                group = grouped.get_group((key,))

            n_distinct = group[scol].nunique()
            col_min_l = min(col_min_l, n_distinct)
            if n_distinct < l_target:
                col_violations += 1

        if col_min_l == float('inf'):
            col_min_l = l_target  # no classes checked

        per_sensitive[scol] = {
            'l_achieved': int(col_min_l),
            'violations': col_violations,
            'total_classes': total_classes,
            'classes_checked': classes_checked,
        }
        global_min_l = min(global_min_l, col_min_l)
        total_violations = max(total_violations, col_violations)

    if global_min_l == float('inf'):
        global_min_l = l_target

    return {
        'l_achieved': int(global_min_l),
        'l_target': l_target,
        'satisfied': global_min_l >= l_target,
        'violations': total_violations,
        'total_classes': total_classes,
        'violation_rate': round(total_violations / max(classes_checked, 1), 4),
        'per_sensitive': per_sensitive,
        'classes_checked': classes_checked,
    }


# ---------------------------------------------------------------------------
# 3b-2. Entropy l-diversity
# ---------------------------------------------------------------------------

def check_entropy_l_diversity(
    protected_data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    l_target: int = 2,
    size_threshold: int = 50,
) -> Dict:
    """Entropy-based l-diversity: penalises skewed sensitive distributions.

    For each equivalence class, Shannon entropy of the sensitive attribute
    must satisfy ``H >= log(l_target)``.  The *effective l* for a class is
    ``exp(H)`` — if one value dominates, effective l can be < 2 even when
    distinct-count >= 2.

    Parameters
    ----------
    protected_data, quasi_identifiers, sensitive_columns, l_target,
    size_threshold : same as :func:`check_l_diversity`.

    Returns
    -------
    dict with keys: entropy_l_achieved (float), l_target, satisfied,
        min_entropy, threshold_entropy, violations, per_sensitive,
        classes_checked.
    """
    _empty = {
        'entropy_l_achieved': None,
        'l_target': l_target,
        'satisfied': True,
        'min_entropy': None,
        'threshold_entropy': math.log(l_target) if l_target > 0 else 0,
        'violations': 0,
        'per_sensitive': {},
        'classes_checked': 0,
    }

    if not sensitive_columns or not quasi_identifiers:
        return _empty

    valid_qis = [q for q in quasi_identifiers if q in protected_data.columns]
    valid_sens = [s for s in sensitive_columns if s in protected_data.columns]
    if not valid_qis or not valid_sens:
        return _empty

    grouped = protected_data.groupby(valid_qis, observed=True)
    total_classes = grouped.ngroups
    group_sizes = grouped.size()
    small_group_keys = group_sizes[group_sizes <= size_threshold].index

    threshold_h = math.log(l_target) if l_target > 0 else 0
    per_sensitive: Dict[str, Dict] = {}
    global_min_h = float('inf')
    total_violations = 0
    classes_checked = len(small_group_keys)

    for scol in valid_sens:
        col_min_h = float('inf')
        col_violations = 0

        for key in small_group_keys:
            group = grouped.get_group(key if isinstance(key, tuple) else (key,))
            counts = group[scol].value_counts()
            n = counts.sum()
            if n == 0:
                continue
            probs = counts / n
            entropy = -float((probs * np.log(probs)).sum())
            col_min_h = min(col_min_h, entropy)
            if entropy < threshold_h:
                col_violations += 1

        if col_min_h == float('inf'):
            col_min_h = threshold_h  # no classes checked

        per_sensitive[scol] = {
            'min_entropy': round(col_min_h, 4),
            'effective_l': round(math.exp(col_min_h), 2),
            'violations': col_violations,
            'classes_checked': classes_checked,
        }
        global_min_h = min(global_min_h, col_min_h)
        total_violations = max(total_violations, col_violations)

    if global_min_h == float('inf'):
        global_min_h = threshold_h

    effective_l = math.exp(global_min_h)

    return {
        'entropy_l_achieved': round(effective_l, 2),
        'l_target': l_target,
        'satisfied': effective_l >= l_target,
        'min_entropy': round(global_min_h, 4),
        'threshold_entropy': round(threshold_h, 4),
        'violations': total_violations,
        'per_sensitive': per_sensitive,
        'classes_checked': classes_checked,
    }


# ---------------------------------------------------------------------------
# 3b-3. t-Closeness
# ---------------------------------------------------------------------------

def check_t_closeness(
    protected_data: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: List[str],
    t_target: float = 0.30,
    size_threshold: int = 50,
) -> Dict:
    """t-Closeness: sensitive distribution in each class ≈ overall distribution.

    Uses Earth Mover's Distance (EMD) for numeric sensitive columns and
    Total Variation Distance (TVD) for categorical ones.

    Parameters
    ----------
    protected_data : DataFrame
    quasi_identifiers : list of str
    sensitive_columns : list of str
    t_target : float
        Maximum allowed distance (0 = identical distributions).
    size_threshold : int
        Only check equivalence classes with ≤ this many records.

    Returns
    -------
    dict with keys: t_achieved (float, max distance), t_target, satisfied,
        per_sensitive, classes_checked.
    """
    _empty = {
        't_achieved': None,
        't_target': t_target,
        'satisfied': True,
        'per_sensitive': {},
        'classes_checked': 0,
    }

    if not sensitive_columns or not quasi_identifiers:
        return _empty

    valid_qis = [q for q in quasi_identifiers if q in protected_data.columns]
    valid_sens = [s for s in sensitive_columns if s in protected_data.columns]
    if not valid_qis or not valid_sens:
        return _empty

    grouped = protected_data.groupby(valid_qis, observed=True)
    group_sizes = grouped.size()
    small_group_keys = group_sizes[group_sizes <= size_threshold].index
    classes_checked = len(small_group_keys)

    per_sensitive: Dict[str, Dict] = {}
    global_max_dist = 0.0

    for scol in valid_sens:
        overall = protected_data[scol].dropna()
        if len(overall) == 0:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(overall)
        col_max_dist = 0.0
        col_violations = 0

        if is_numeric:
            # Pre-compute overall sorted values for EMD
            overall_sorted = np.sort(overall.values)
        else:
            # Pre-compute overall frequency distribution for TVD
            overall_freq = overall.value_counts(normalize=True)

        for key in small_group_keys:
            group = grouped.get_group(key if isinstance(key, tuple) else (key,))
            vals = group[scol].dropna()
            if len(vals) == 0:
                continue

            if is_numeric:
                dist = _emd_numeric(vals.values, overall_sorted)
            else:
                dist = _tvd_categorical(vals, overall_freq)

            col_max_dist = max(col_max_dist, dist)
            if dist > t_target:
                col_violations += 1

        per_sensitive[scol] = {
            'max_distance': round(col_max_dist, 4),
            'metric': 'EMD' if is_numeric else 'TVD',
            'violations': col_violations,
            'classes_checked': classes_checked,
        }
        global_max_dist = max(global_max_dist, col_max_dist)

    return {
        't_achieved': round(global_max_dist, 4),
        't_target': t_target,
        'satisfied': global_max_dist <= t_target,
        'per_sensitive': per_sensitive,
        'classes_checked': classes_checked,
    }


def _emd_numeric(class_values: np.ndarray, overall_sorted: np.ndarray) -> float:
    """Earth Mover's Distance between a class and the overall numeric distribution.

    Uses the CDF-based formula: EMD = (1/n) * sum(|CDF_class - CDF_overall|)
    evaluated at every unique value. Normalised to [0, 1] by dividing by the
    data range.
    """
    if len(class_values) == 0 or len(overall_sorted) == 0:
        return 0.0

    all_vals = np.concatenate([class_values, overall_sorted])
    vmin, vmax = float(all_vals.min()), float(all_vals.max())
    data_range = vmax - vmin
    if data_range == 0:
        return 0.0

    # Build CDFs at sorted unique points
    unique_vals = np.unique(all_vals)

    class_sorted = np.sort(class_values)
    n_class = len(class_sorted)
    n_overall = len(overall_sorted)

    # CDF values at each unique point
    cdf_class = np.searchsorted(class_sorted, unique_vals, side='right') / n_class
    cdf_overall = np.searchsorted(overall_sorted, unique_vals, side='right') / n_overall

    # Approximate EMD via trapezoidal integration of |CDF difference|
    diffs = np.abs(cdf_class - cdf_overall)
    if len(unique_vals) < 2:
        return float(diffs[0]) if len(diffs) > 0 else 0.0

    # Normalise positions to [0, 1]
    positions = (unique_vals - vmin) / data_range
    emd = float(np.trapz(diffs, positions))
    return emd


def _tvd_categorical(class_series: pd.Series, overall_freq: pd.Series) -> float:
    """Total Variation Distance between class and overall categorical distribution."""
    class_freq = class_series.value_counts(normalize=True)
    all_cats = set(class_freq.index) | set(overall_freq.index)
    tvd = sum(abs(class_freq.get(c, 0.0) - overall_freq.get(c, 0.0))
              for c in all_cats) / 2.0
    return float(tvd)


# ---------------------------------------------------------------------------
# 3c. Method-specific quality checks
# ---------------------------------------------------------------------------

def assess_method_quality(
    result: Any,
    qi_list: List[str],
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Dict:
    """Analyse method-specific quality from existing metadata.

    Parameters
    ----------
    result : ProtectionResult
    qi_list : list of str
    qi_treatment : dict or None
        ``{col: 'Heavy'|'Standard'|'Light'}``.

    Returns
    -------
    dict with keys: method, checks, treatment_assessment.
    """
    meta = getattr(result, 'metadata', None) or {}
    method = (getattr(result, 'method', '') or '').upper()

    checks: List[Dict] = []
    treatment_assessment: Optional[Dict[str, str]] = None

    stats = meta.get('statistics', {})

    if method == 'KANON':
        checks = _check_kanon(meta, stats)
    elif method == 'LOCSUPR':
        checks = _check_locsupr(stats, qi_list)
        if qi_treatment:
            treatment_assessment = _assess_treatment_locsupr(
                stats, qi_list, qi_treatment)
    elif method == 'PRAM':
        checks = _check_pram(stats, qi_list)
        if qi_treatment:
            treatment_assessment = _assess_treatment_pram(
                stats, qi_list, qi_treatment)
    elif method == 'NOISE':
        checks = _check_noise(stats)
        if qi_treatment:
            treatment_assessment = _assess_treatment_noise(
                stats, qi_list, qi_treatment)

    return {
        'method': method,
        'checks': checks,
        'treatment_assessment': treatment_assessment,
    }


def _check_kanon(meta: Dict, stats: Dict) -> List[Dict]:
    """Quality checks for kANON."""
    checks = []
    k = meta.get('k', meta.get('parameters', {}).get('k'))

    supp_rate = stats.get('suppression_rate')
    if supp_rate is not None:
        status = 'good' if supp_rate <= 0.10 else ('warn' if supp_rate <= 0.15 else 'bad')
        checks.append({
            'label': 'Suppression rate',
            'value': f'{supp_rate:.1%}',
            'status': status,
        })

    avg_gs = stats.get('avg_group_size')
    if avg_gs is not None and k is not None:
        ratio = avg_gs / max(k, 1)
        status = 'good' if ratio <= 5 else ('warn' if ratio <= 10 else 'bad')
        checks.append({
            'label': 'Avg group / k ratio',
            'value': f'{ratio:.1f}× (avg={avg_gs:.0f}, k={k})',
            'status': status,
        })

    min_gs = stats.get('min_group_size')
    if min_gs is not None and k is not None:
        status = 'good' if min_gs >= k else 'bad'
        checks.append({
            'label': 'Min group size',
            'value': str(min_gs),
            'status': status,
        })

    n_eq = stats.get('n_equivalence_classes')
    n_rec = stats.get('n_records')
    if n_eq is not None and n_rec is not None:
        checks.append({
            'label': 'Equivalence classes',
            'value': f'{n_eq:,} ({n_rec / max(n_eq, 1):.0f} avg records/class)',
            'status': 'good',
        })

    return checks


def _check_locsupr(stats: Dict, qi_list: List[str]) -> List[Dict]:
    """Quality checks for LOCSUPR."""
    checks = []
    supp_per_var = stats.get('suppressions_per_variable', {})
    total = stats.get('total_suppressions', 0)

    if total > 0 and supp_per_var:
        max_qi = max(supp_per_var, key=lambda q: supp_per_var.get(q, 0))
        max_share = supp_per_var[max_qi] / max(total, 1)
        status = 'good' if max_share <= 0.40 else ('warn' if max_share <= 0.60 else 'bad')
        checks.append({
            'label': 'Suppression concentration',
            'value': f'{max_qi}: {max_share:.0%} of {total:,} total',
            'status': status,
        })

    records_supp = stats.get('records_with_suppressions', 0)
    n_rec = stats.get('n_records', stats.get('total_records', 0))
    if n_rec > 0:
        rate = records_supp / n_rec
        status = 'good' if rate <= 0.10 else ('warn' if rate <= 0.20 else 'bad')
        checks.append({
            'label': 'Records affected',
            'value': f'{records_supp:,} / {n_rec:,} ({rate:.1%})',
            'status': status,
        })

    return checks


def _check_pram(stats: Dict, qi_list: List[str]) -> List[Dict]:
    """Quality checks for PRAM."""
    checks = []
    changes = stats.get('changes_per_variable', {})
    total = stats.get('total_changes', 0)
    n_rec = stats.get('total_records', 0)

    if total > 0 and n_rec > 0:
        checks.append({
            'label': 'Total changes',
            'value': f'{total:,} across {n_rec:,} records',
            'status': 'good',
        })

    if len(changes) >= 2:
        vals = [v for v in changes.values() if v > 0]
        if vals:
            ratio = max(vals) / max(min(vals), 1)
            status = 'good' if ratio <= 3 else 'warn'
            checks.append({
                'label': 'Change rate spread',
                'value': f'{ratio:.1f}× (max/min across QIs)',
                'status': status,
            })

    return checks


def _check_noise(stats: Dict) -> List[Dict]:
    """Quality checks for NOISE."""
    checks = []
    noise_pv = stats.get('noise_per_variable', {})
    value_changes = stats.get('value_changes', {})

    low_corr = []
    for var, vstats in {**noise_pv, **value_changes}.items():
        if isinstance(vstats, dict):
            corr = vstats.get('correlation')
            if corr is not None and corr < 0.90:
                low_corr.append((var, corr))

    if low_corr:
        details = ', '.join(f'{v}: {c:.3f}' for v, c in low_corr[:3])
        checks.append({
            'label': 'Low-correlation variables',
            'value': details,
            'status': 'warn' if all(c >= 0.80 for _, c in low_corr) else 'bad',
        })
    elif noise_pv or value_changes:
        checks.append({
            'label': 'Correlation preservation',
            'value': 'All variables ≥ 0.90',
            'status': 'good',
        })

    return checks


# --- Treatment effectiveness assessments ---

def _assess_treatment_locsupr(
    stats: Dict, qi_list: List[str], qi_treatment: Dict[str, str],
) -> Dict[str, str]:
    """Check if Heavy QIs got more suppressions than Light ones."""
    supp = stats.get('suppressions_per_variable', {})
    return _assess_treatment_values(
        {q: supp.get(q, 0) for q in qi_list},
        qi_treatment,
        higher_is_heavier=True,
    )


def _assess_treatment_pram(
    stats: Dict, qi_list: List[str], qi_treatment: Dict[str, str],
) -> Dict[str, str]:
    """Check if Heavy QIs got more PRAM changes than Light ones."""
    changes = stats.get('changes_per_variable', {})
    return _assess_treatment_values(
        {q: changes.get(q, 0) for q in qi_list},
        qi_treatment,
        higher_is_heavier=True,
    )


def _assess_treatment_noise(
    stats: Dict, qi_list: List[str], qi_treatment: Dict[str, str],
) -> Dict[str, str]:
    """Check if Heavy QIs got lower correlation (more noise) than Light."""
    noise_pv = stats.get('noise_per_variable', {})
    value_changes = stats.get('value_changes', {})
    combined = {**noise_pv, **value_changes}

    corr_vals = {}
    for q in qi_list:
        entry = combined.get(q)
        if isinstance(entry, dict):
            c = entry.get('correlation')
            if c is not None:
                corr_vals[q] = c

    return _assess_treatment_values(
        corr_vals,
        qi_treatment,
        higher_is_heavier=False,  # lower correlation = heavier treatment
    )


def _assess_treatment_values(
    per_qi_values: Dict[str, float],
    qi_treatment: Dict[str, str],
    higher_is_heavier: bool,
) -> Dict[str, str]:
    """Generic treatment alignment checker.

    For each QI, checks if the observed value aligns with its treatment:
    - Heavy QIs should have higher (or lower, per *higher_is_heavier*) values
    - Light QIs should have the opposite

    Returns ``{qi: 'aligned' | 'under' | 'over'}``.
    """
    if not per_qi_values or not qi_treatment:
        return {}

    vals = list(per_qi_values.values())
    if not vals:
        return {}
    median_val = sorted(vals)[len(vals) // 2]

    result = {}
    for qi, val in per_qi_values.items():
        level = qi_treatment.get(qi, 'Standard')
        if level == 'Standard':
            result[qi] = 'aligned'
            continue

        if higher_is_heavier:
            if level == 'Heavy':
                result[qi] = 'aligned' if val >= median_val else 'under'
            else:  # Light
                result[qi] = 'aligned' if val <= median_val else 'over'
        else:
            if level == 'Heavy':
                result[qi] = 'aligned' if val <= median_val else 'under'
            else:  # Light
                result[qi] = 'aligned' if val >= median_val else 'over'

    return result


# ---------------------------------------------------------------------------
# 3d. Failure guidance
# ---------------------------------------------------------------------------

def build_failure_guidance(
    log_entries: List[str],
    data_features: Dict,
    qi_list: List[str],
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Dict:
    """Analyse why all methods failed and suggest remediation.

    Parameters
    ----------
    log_entries : list of str
        Log from ``run_rules_engine_protection()``.
    data_features : dict
        Output of ``build_data_features()``.
    qi_list : list of str
    qi_treatment : dict or None

    Returns
    -------
    dict with keys: bottleneck_qi, bottleneck_reason,
        per_method_failure, suggestions.
    """
    # --- Identify bottleneck QI (highest cardinality) ---
    qi_cards = data_features.get('qi_cardinalities', {})
    bottleneck_qi = None
    bottleneck_reason = ''

    if qi_cards:
        bottleneck_qi = max(qi_cards, key=lambda q: qi_cards.get(q, 0))
        card = qi_cards[bottleneck_qi]
        bottleneck_reason = f'{card:,} unique values'

    # --- Parse per-method failure from log ---
    per_method: List[Dict] = []
    current_method = None
    last_param = None

    for entry in log_entries:
        stripped = entry.strip()
        if stripped.startswith('Trying primary:') or stripped.startswith('Trying fallback:'):
            current_method = stripped.split(':')[-1].strip()
            last_param = None
        elif stripped.startswith('Escalating') or stripped.startswith('Escalating'):
            parts = stripped.split(':')
            if len(parts) >= 2:
                last_param = parts[-1].strip()
        elif '⚠' in stripped and 'Utility below' in stripped:
            per_method.append({
                'method': current_method or '?',
                'last_param': last_param or 'initial',
                'reason': 'Utility floor breached',
            })
            current_method = None
        elif stripped.startswith('Pipeline failed:'):
            per_method.append({
                'method': 'Pipeline',
                'last_param': '',
                'reason': stripped,
            })

    # Add methods that didn't explicitly fail (exhausted schedule)
    if current_method and not any(
        m['method'] == current_method for m in per_method
    ):
        per_method.append({
            'method': current_method,
            'last_param': last_param or 'max',
            'reason': 'ReID target not reached at max parameter',
        })

    # --- Generate suggestions ---
    suggestions: List[str] = []

    if bottleneck_qi:
        suggestions.append(
            f'Consider preprocessing "{bottleneck_qi}" more aggressively '
            f'(currently {bottleneck_reason})')

        if qi_treatment:
            level = qi_treatment.get(bottleneck_qi, 'Standard')
            if level != 'Heavy':
                suggestions.append(
                    f'Increase treatment for "{bottleneck_qi}" from '
                    f'{level} to Heavy')

    reid_95 = data_features.get('reid_95')
    if reid_95 is not None and reid_95 > 0.50:
        suggestions.append(
            f'Structural risk is very high ({reid_95:.0%}). '
            f'Consider excluding low-priority QIs to reduce combination space')

    n_qis = len(qi_list)
    if n_qis >= 6:
        suggestions.append(
            f'Using {n_qis} QIs creates a large combination space. '
            f'Consider dropping the least important QI')

    if not suggestions:
        suggestions.append(
            'Try relaxing the ReID target or lowering the utility floor')

    return {
        'bottleneck_qi': bottleneck_qi,
        'bottleneck_reason': bottleneck_reason,
        'per_method_failure': per_method,
        'suggestions': suggestions,
    }
