"""
Shared utility measurement for SDC workflows.

Provides consistent utility metrics across Preprocess, Protect, and Combo tabs.

SDC framing:
  QIs       → where you apply SDC → accept utility loss here
  Sensitive → what you preserve   → measure utility here
  Direct IDs → just remove them

When sensitive_columns is provided, utility is measured on those columns
(the variables analysts care about). QI utility is reported separately
for transparency but does NOT drive pass/fail decisions.

When sensitive_columns is empty, falls back to all non-QI common columns.

Metric layers:
- compute_utility / compute_per_variable_utility — fast, used for threshold checks
- compute_il1s — sdcMicro-style per-record information loss
- compute_benchmark_analysis — means, variances, correlations, frequency tables
- compute_distributional_metrics — KL divergence, Hellinger distance
"""

from typing import Dict, List, Optional, Set
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# COLUMN RESOLUTION — decides which columns to measure utility on
# ============================================================================

def resolve_utility_columns(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    sensitive_columns: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
) -> List[str]:
    """Determine which columns to measure utility on.

    Priority:
    1. If *sensitive_columns* provided → use those (analyst's target variables).
    2. Else → all common columns EXCEPT QIs (non-QI residual).
    3. Else → all common columns (legacy fallback).
    """
    common = [c for c in original.columns if c in processed.columns]

    if sensitive_columns:
        cols = [c for c in sensitive_columns if c in common]
        if cols:
            return cols

    if quasi_identifiers:
        qi_set: Set[str] = set(quasi_identifiers)
        cols = [c for c in common if c not in qi_set]
        if cols:
            return cols

    return common


# ============================================================================
# COLUMN TYPE COERCION — Configure table as source of truth
# ============================================================================

def _coerce_pair(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    column_types: Optional[Dict[str, str]] = None,
) -> tuple:
    """Coerce object-dtype columns to numeric based on Configure column_types.

    Returns (original_copy, processed_copy) with types corrected.
    """
    if not column_types:
        return original, processed
    from sdc_engine.sdc.sdc_utils import coerce_columns_by_types
    original = original.copy()
    processed = processed.copy()
    cols = [c for c in original.columns if c in processed.columns]
    coerce_columns_by_types(original, column_types, cols)
    coerce_columns_by_types(processed, column_types, cols)
    return original, processed


# ============================================================================
# CORE UTILITY (fast — used for step acceptance thresholds)
# ============================================================================

def compute_utility(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    quasi_identifiers: List[str] = None,
    weights: Optional[Dict[str, float]] = None,
    sensitive_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, str]] = None,
) -> float:
    """Overall utility score.

    For numeric cols: Pearson correlation (abs).
    For categorical cols: fraction of rows whose value is unchanged.

    Parameters
    ----------
    original, processed : DataFrames to compare.
    quasi_identifiers : QI columns (used to exclude from measurement when
        *sensitive_columns* is not provided).
    weights : optional per-column weights (e.g. from Util Priority).
        If provided, computes weighted mean instead of simple mean.
    sensitive_columns : the columns to measure utility on. When provided,
        only these columns are used (SDC framing: measure what you preserve).
        When None, falls back to all non-QI columns, then all columns.
    column_types : Configure table types (source of truth for dtype).

    Returns
    -------
    float in [0, 1]. Higher = more data preserved.
    """
    original, processed = _coerce_pair(original, processed, column_types)
    cols = resolve_utility_columns(
        original, processed, sensitive_columns, quasi_identifiers
    )
    if not cols:
        return 0.0

    col_scores: Dict[str, float] = {}
    for col in cols:
        orig = original[col]
        proc = processed[col]
        if (pd.api.types.is_numeric_dtype(orig)
                and pd.api.types.is_numeric_dtype(proc)):
            try:
                corr = orig.corr(proc)
                if pd.notna(corr):
                    col_scores[col] = abs(corr)
            except Exception:
                pass
        else:
            try:
                n = min(len(orig), len(proc))
                if n > 0:
                    same = (orig.iloc[:n].astype(str)
                            == proc.iloc[:n].astype(str)).sum()
                    col_scores[col] = float(same) / n
            except Exception:
                pass

    if not col_scores:
        return 1.0

    if weights:
        w_sum = sum(weights.get(c, 1.0) for c in col_scores)
        if w_sum > 0:
            return sum(col_scores[c] * weights.get(c, 1.0)
                       for c in col_scores) / w_sum
    return sum(col_scores.values()) / len(col_scores)


def compute_per_variable_utility(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    quasi_identifiers: List[str],
    column_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """Per-variable utility metrics for QI columns.

    Returns a dict mapping column name to metrics dict with keys:
    - correlation (numeric) or row_preservation (categorical)
    - unique_before / unique_after
    - category_overlap, categories_added/removed (categorical)
    - range_ratio, mean_shift (numeric)
    - dtype: 'numeric' or 'categorical'
    """
    original, processed = _coerce_pair(original, processed, column_types)
    metrics: Dict[str, Dict] = {}
    for col in quasi_identifiers:
        if col not in original.columns or col not in processed.columns:
            continue

        orig = original[col]
        proc = processed[col]
        m: Dict = {
            'unique_before': int(orig.nunique()),
            'unique_after': int(proc.nunique()),
        }

        # Detect numeric→range-string generalization (e.g., kANON bins
        # 50000 → "14848-30000").  Use midpoints for correlation.
        _orig_numeric = pd.api.types.is_numeric_dtype(orig)
        _proc_numeric = pd.api.types.is_numeric_dtype(proc)
        _range_binned = False
        if _orig_numeric and not _proc_numeric:
            # Try parsing range strings to midpoints
            try:
                import re
                _sample = proc.dropna().head(20).astype(str)
                _range_pat = re.compile(r'^[\[(]?\s*(-?[\d.]+)\s*[-–]\s*(-?[\d.]+)\s*[\])]?$')
                _matches = sum(1 for v in _sample if _range_pat.match(str(v)))
                if _matches >= len(_sample) * 0.5:
                    def _midpoint(val):
                        m_ = _range_pat.match(str(val))
                        if m_:
                            return (float(m_.group(1)) + float(m_.group(2))) / 2
                        return np.nan
                    proc_mid = proc.apply(_midpoint)
                    if proc_mid.notna().sum() > 10:
                        _proc_numeric = True
                        _range_binned = True
                        proc = proc_mid
            except Exception:
                pass

        if _orig_numeric and _proc_numeric:
            # Numeric (or numeric→range with midpoint recovery)
            try:
                corr = orig.corr(proc)
                m['correlation'] = (round(float(corr), 4)
                                    if pd.notna(corr) else None)
            except Exception:
                m['correlation'] = None
            try:
                orig_range = float(orig.max() - orig.min())
                proc_range = float(proc.max() - proc.min())
                m['range_ratio'] = (round(proc_range / orig_range, 4)
                                    if orig_range > 0 else 1.0)
            except Exception:
                m['range_ratio'] = None
            try:
                orig_mean = float(orig.mean())
                proc_mean = float(proc.mean())
                m['mean_shift'] = round(
                    abs(proc_mean - orig_mean) / (abs(orig_mean) or 1), 4)
            except Exception:
                m['mean_shift'] = None
            m['dtype'] = 'numeric'
            if _range_binned:
                m['range_binned'] = True
        else:
            # Categorical
            orig_cats = set(orig.dropna().unique())
            proc_cats = set(proc.dropna().unique())
            overlap = len(orig_cats & proc_cats)
            m['category_overlap'] = (round(overlap / len(orig_cats), 4)
                                     if orig_cats else 1.0)
            m['categories_added'] = len(proc_cats - orig_cats)
            m['categories_removed'] = len(orig_cats - proc_cats)
            try:
                n = min(len(orig), len(proc))
                if n > 0:
                    same = (orig.iloc[:n].astype(str)
                            == proc.iloc[:n].astype(str)).sum()
                    m['row_preservation'] = round(float(same) / n, 4)
                else:
                    m['row_preservation'] = 1.0
            except Exception:
                m['row_preservation'] = None
            m['dtype'] = 'categorical'

        metrics[col] = m

    return metrics


# ============================================================================
# IL1s — sdcMicro-style Information Loss
# ============================================================================

def compute_il1s(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    quasi_identifiers: List[str],
    weights: Optional[Dict[str, float]] = None,
    sensitive_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """sdcMicro-style IL1s information loss.

    For each numeric column: mean of |orig_i - prot_i| / range(orig) per record.
    For each categorical column: proportion of changed values.

    When *sensitive_columns* is given, computes IL1s on those columns
    (the analysis variables you want to preserve). Otherwise uses
    *quasi_identifiers* (legacy behaviour).

    Returns
    -------
    dict with:
      'per_variable': {col: il1_value}  (0 = no loss, 1 = total loss)
      'overall': weighted or simple average across variables
    """
    original, processed = _coerce_pair(original, processed, column_types)
    target_cols = (sensitive_columns if sensitive_columns
                   else quasi_identifiers)
    per_var: Dict[str, float] = {}

    for col in target_cols:
        if col not in original.columns or col not in processed.columns:
            continue

        orig = original[col]
        proc = processed[col]
        n = min(len(orig), len(proc))
        if n == 0:
            continue

        if (pd.api.types.is_numeric_dtype(orig)
                and pd.api.types.is_numeric_dtype(proc)):
            try:
                o = orig.iloc[:n].values.astype(float)
                p = proc.iloc[:n].values.astype(float)
                valid = ~(np.isnan(o) | np.isnan(p))
                if valid.sum() == 0:
                    per_var[col] = 1.0
                    continue
                o_valid = o[valid]
                p_valid = p[valid]
                col_range = float(o_valid.max() - o_valid.min())
                if col_range > 0:
                    per_var[col] = round(
                        float(np.mean(np.abs(o_valid - p_valid) / col_range)),
                        6)
                else:
                    per_var[col] = 0.0
            except Exception:
                per_var[col] = 1.0
        else:
            # Categorical: proportion changed
            try:
                same = (orig.iloc[:n].astype(str)
                        == proc.iloc[:n].astype(str)).sum()
                per_var[col] = round(1.0 - float(same) / n, 6)
            except Exception:
                per_var[col] = 1.0

    if not per_var:
        return {'per_variable': {}, 'overall': 0.0}

    if weights:
        w_sum = sum(weights.get(c, 1.0) for c in per_var)
        overall = (sum(per_var[c] * weights.get(c, 1.0) for c in per_var)
                   / w_sum) if w_sum > 0 else 0.0
    else:
        overall = sum(per_var.values()) / len(per_var)

    return {'per_variable': per_var, 'overall': round(overall, 6)}


# ============================================================================
# BENCHMARK ANALYSIS — means, variances, correlations, frequency tables
# ============================================================================

def compute_benchmark_analysis(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Run standard statistical analyses on both datasets and compare.

    When *sensitive_columns* is given, analyses target those columns.

    Returns
    -------
    dict with:
      'means': {col: {original, protected, abs_diff, rel_diff}}
      'variances': {col: {original, protected, abs_diff, rel_diff}}
      'correlations': {original_matrix, protected_matrix, max_diff, mean_diff}
      'frequency_tables': {col: {tvd, original_top5, protected_top5}}
    """
    original, processed = _coerce_pair(original, processed, column_types)
    target_cols = (sensitive_columns if sensitive_columns
                   else quasi_identifiers)
    result: Dict[str, object] = {}
    n = min(len(original), len(processed))

    # --- Means & Variances ---
    means: Dict[str, Dict] = {}
    variances: Dict[str, Dict] = {}
    numeric_cols = []

    for col in target_cols:
        if col not in original.columns or col not in processed.columns:
            continue
        orig = original[col].iloc[:n]
        proc = processed[col].iloc[:n]

        if (pd.api.types.is_numeric_dtype(orig)
                and pd.api.types.is_numeric_dtype(proc)):
            numeric_cols.append(col)
            o_mean = float(orig.mean()) if not orig.isna().all() else 0.0
            p_mean = float(proc.mean()) if not proc.isna().all() else 0.0
            abs_d = abs(o_mean - p_mean)
            rel_d = abs_d / abs(o_mean) if o_mean != 0 else 0.0
            means[col] = {
                'original': round(o_mean, 4),
                'protected': round(p_mean, 4),
                'abs_diff': round(abs_d, 4),
                'rel_diff': round(rel_d, 4),
            }

            o_var = float(orig.var()) if not orig.isna().all() else 0.0
            p_var = float(proc.var()) if not proc.isna().all() else 0.0
            abs_dv = abs(o_var - p_var)
            rel_dv = abs_dv / abs(o_var) if o_var != 0 else 0.0
            variances[col] = {
                'original': round(o_var, 4),
                'protected': round(p_var, 4),
                'abs_diff': round(abs_dv, 4),
                'rel_diff': round(rel_dv, 4),
            }

    result['means'] = means
    result['variances'] = variances

    # --- Correlation matrix comparison ---
    corr_info: Dict[str, object] = {}
    if len(numeric_cols) >= 2:
        try:
            orig_corr = original[numeric_cols].iloc[:n].corr()
            proc_corr = processed[numeric_cols].iloc[:n].corr()
            diff = (orig_corr - proc_corr).abs()
            corr_info['max_diff'] = round(float(diff.max().max()), 4)
            corr_info['mean_diff'] = round(float(
                diff.values[np.triu_indices(len(numeric_cols), k=1)].mean()
            ), 4)
        except Exception:
            corr_info['max_diff'] = None
            corr_info['mean_diff'] = None
    result['correlations'] = corr_info

    # --- Frequency tables (categorical) ---
    freq_tables: Dict[str, Dict] = {}
    for col in target_cols:
        if col not in original.columns or col not in processed.columns:
            continue
        orig = original[col].iloc[:n]
        proc = processed[col].iloc[:n]
        if pd.api.types.is_numeric_dtype(orig):
            continue

        try:
            o_freq = orig.value_counts(normalize=True)
            p_freq = proc.value_counts(normalize=True)
            # Total Variation Distance
            all_cats = set(o_freq.index) | set(p_freq.index)
            tvd = 0.5 * sum(
                abs(o_freq.get(c, 0) - p_freq.get(c, 0)) for c in all_cats
            )
            freq_tables[col] = {
                'tvd': round(float(tvd), 4),
                'n_categories_orig': len(o_freq),
                'n_categories_prot': len(p_freq),
            }
        except Exception:
            pass

    result['frequency_tables'] = freq_tables

    # --- Cross-tabulation preservation (sensitive × QI relationships) ---
    # Only meaningful when records are suppressed (group composition
    # changes) or sensitive values are perturbed (PRAM/NOISE).
    # For pure binning (QI generalisation only), cross-tab is ~100%
    # by construction and uninformative — skip it.
    # NOTE: For performance, only the first 3 sensitive and 3 QI columns
    # are checked (max 9 pairs). Relationships involving the 4th+ column
    # of either list are not evaluated.
    cross_tab: Dict[str, object] = {}
    if sensitive_columns and quasi_identifiers:
        sens_in_data = [c for c in sensitive_columns
                        if c in original.columns and c in processed.columns]
        qi_in_data = [c for c in quasi_identifiers
                      if c in original.columns and c in processed.columns]
        logger.info("[cross-tab] sens_in_data=%s, qi_in_data=%s",
                    sens_in_data, qi_in_data)

        # Detect whether cross-tab is meaningful
        records_suppressed = len(processed) < len(original)
        # Cell suppression (kANON sets QI values to NaN without dropping
        # rows) also changes group composition — detect new NaNs in QIs.
        cell_suppressed = False
        if not records_suppressed:
            for qc in qi_in_data[:3]:
                try:
                    _orig_na = original[qc].isna().sum()
                    _proc_na = processed[qc].isna().sum()
                    if _proc_na > _orig_na:
                        cell_suppressed = True
                        break
                except Exception:
                    pass
        sensitive_changed = False
        for sc in sens_in_data:
            try:
                n_check = min(len(original), len(processed), 500)
                if not original[sc].iloc[:n_check].equals(
                        processed[sc].iloc[:n_check]):
                    sensitive_changed = True
                    break
            except Exception:
                pass

        # Also meaningful when QIs are generalised (cardinality reduction
        # changes group composition even if sensitive is untouched).
        qi_generalised = False
        if not (records_suppressed or cell_suppressed or sensitive_changed):
            for qc in qi_in_data[:3]:
                try:
                    if processed[qc].nunique() < original[qc].nunique():
                        qi_generalised = True
                        break
                except Exception:
                    pass

        cross_tab_meaningful = (records_suppressed or cell_suppressed
                                or sensitive_changed or qi_generalised)
        logger.info("[cross-tab] meaningful=%s (suppressed=%s, cell_supp=%s, "
                    "sens_changed=%s, qi_gen=%s)",
                    cross_tab_meaningful, records_suppressed, cell_suppressed,
                    sensitive_changed, qi_generalised)

        if not cross_tab_meaningful:
            cross_tab['skip_reason'] = (
                'No cross-tab impact — sensitive values unchanged, '
                'no records suppressed.')
        else:
            # --- Group-mean correlation ---
            # Group original & protected sensitive values by the protected
            # QI bins, compute mean per group, correlate the two vectors.
            # Handles type changes (numeric → categorical) naturally.
            _MIN_GROUP_SIZE = 3  # exclude tiny groups from correlation
            subgroup_stats = []
            skipped_pairs = []
            n_total_pairs = 0
            for sc in sens_in_data[:3]:
                # Determine if sensitive is numeric.
                # Primary: column_types label (handles string-stored numerics).
                # Fallback: pandas dtype.
                _sc_label = (column_types or {}).get(sc, '')
                _sc_numeric = (
                    _is_continuous_type(_sc_label)
                    or 'char (numeric)' in _sc_label.lower()
                    or 'coded' in _sc_label.lower()
                    or pd.api.types.is_numeric_dtype(original[sc])
                )
                logger.info("[cross-tab] sensitive %s: label=%r, "
                            "numeric=%s", sc, _sc_label, _sc_numeric)
                for qc in qi_in_data[:3]:
                    n_total_pairs += 1
                    try:
                        nn = min(len(original), len(processed))
                        p_q_raw = processed[qc].iloc[:nn]
                        o_s = original[sc].iloc[:nn]
                        p_s = processed[sc].iloc[:nn]
                        # Exclude cell-suppressed records (QI set to NaN)
                        # — they form an artificial "nan" group that mixes
                        # unrelated records and skews the correlation.
                        _non_supp = p_q_raw.notna()
                        p_q = p_q_raw[_non_supp].astype(str)
                        o_s = o_s[_non_supp]
                        p_s = p_s[_non_supp]

                        _qc_label = (column_types or {}).get(qc, '')

                        if _sc_numeric:
                            # ── Numeric sensitive: eta² / r² + subgroup
                            #    mean correlation (original path) ──
                            eta_orig = _eta_squared(
                                o_s, original[qc].iloc[:nn], _qc_label)
                            if eta_orig is None or eta_orig < 0.02:
                                skipped_pairs.append({
                                    'sensitive': sc, 'qi': qc,
                                    'reason': 'weak_relationship',
                                    'metric': 'eta_sq',
                                    'strength_original': round(
                                        float(eta_orig), 4)
                                        if eta_orig is not None else 0.0,
                                })
                                continue

                            # Group means by protected QI bins
                            group_counts = p_q.groupby(p_q).count()
                            valid_groups = group_counts[
                                group_counts >= _MIN_GROUP_SIZE].index
                            if len(valid_groups) < 3:
                                skipped_pairs.append({
                                    'sensitive': sc, 'qi': qc,
                                    'reason': 'too_few_groups',
                                    'metric': 'eta_sq',
                                    'strength_original': round(
                                        float(eta_orig), 4),
                                })
                                continue

                            mask = p_q.isin(valid_groups)
                            o_means = o_s[mask].groupby(p_q[mask]).mean()
                            p_means = p_s[mask].groupby(p_q[mask]).mean()

                            common = o_means.index.intersection(
                                p_means.index)
                            if len(common) < 3:
                                skipped_pairs.append({
                                    'sensitive': sc, 'qi': qc,
                                    'reason': 'too_few_common_groups',
                                    'metric': 'eta_sq',
                                    'strength_original': round(
                                        float(eta_orig), 4),
                                })
                                continue
                            corr = o_means[common].corr(p_means[common])
                            if pd.notna(corr):
                                eta_prot = _eta_squared(p_s, p_q, '')
                                # Preservation: blend of group-mean correlation
                                # AND effect-size ratio.  When sensitive is
                                # unchanged, corr≈1.0 but effect-size ratio
                                # reveals how much the binned QI grouping still
                                # explains the sensitive variance.
                                _ratio = 1.0
                                if (eta_prot is not None
                                        and eta_orig > 0.001):
                                    _ratio = min(1.0, float(eta_prot)
                                                 / float(eta_orig))
                                # 50% correlation + 50% effect-size ratio
                                preservation = round(max(0.0,
                                    0.5 * float(corr) + 0.5 * _ratio), 4)
                                subgroup_stats.append({
                                    'sensitive': sc, 'qi': qc,
                                    'n_groups': len(common),
                                    'mean_preservation': preservation,
                                    'metric': 'eta_sq',
                                    'strength_original': round(
                                        float(eta_orig), 4),
                                    'strength_protected': round(
                                        float(eta_prot), 4)
                                        if eta_prot is not None else 0.0,
                                })
                        else:
                            # ── Categorical sensitive: Cramér's V +
                            #    conditional-distribution preservation ──
                            cv_orig = _cramers_v(
                                o_s, original[qc].iloc[:nn][_non_supp])
                            if cv_orig is None or cv_orig < 0.05:
                                skipped_pairs.append({
                                    'sensitive': sc, 'qi': qc,
                                    'reason': 'weak_relationship',
                                    'metric': 'cramers_v',
                                    'strength_original': round(
                                        float(cv_orig), 4)
                                        if cv_orig is not None else 0.0,
                                })
                                continue

                            # Categorical preservation: compare conditional
                            # distributions P(sens|QI_bin) before vs after
                            o_q_aligned = original[qc].iloc[:nn][_non_supp]
                            cat_pres = _categorical_preservation(
                                o_q_aligned, o_s, p_q, p_s)
                            if cat_pres is None:
                                skipped_pairs.append({
                                    'sensitive': sc, 'qi': qc,
                                    'reason': 'too_few_groups',
                                    'metric': 'cramers_v',
                                    'strength_original': round(
                                        float(cv_orig), 4),
                                })
                                continue

                            cv_prot = _cramers_v(p_s, p_q)
                            # Blend conditional-distribution preservation
                            # with effect-size ratio (V_prot / V_orig).
                            _v_ratio = 1.0
                            if (cv_prot is not None
                                    and cv_orig > 0.01):
                                _v_ratio = min(1.0, float(cv_prot)
                                               / float(cv_orig))
                            _blended_pres = round(max(0.0,
                                0.5 * cat_pres + 0.5 * _v_ratio), 4)
                            subgroup_stats.append({
                                'sensitive': sc, 'qi': qc,
                                'n_groups': int(p_q.nunique()),
                                'mean_preservation': _blended_pres,
                                'metric': 'cramers_v',
                                'strength_original': round(
                                    float(cv_orig), 4),
                                'strength_protected': round(
                                    float(cv_prot), 4)
                                    if cv_prot is not None else 0.0,
                            })
                    except Exception:
                        pass

            cross_tab['subgroup_means'] = subgroup_stats
            cross_tab['skipped_pairs'] = skipped_pairs
            cross_tab['n_total_pairs'] = n_total_pairs
            cross_tab['n_qualifying'] = len(subgroup_stats)
            logger.info("[cross-tab] %d qualifying pairs, %d skipped, "
                        "%d total", len(subgroup_stats), len(skipped_pairs),
                        n_total_pairs)
            for s in subgroup_stats:
                logger.info("  pair %s ~ %s: metric=%s, str_orig=%.4f, "
                            "pres=%.4f", s['qi'], s['sensitive'],
                            s['metric'], s['strength_original'],
                            s['mean_preservation'])
            for s in skipped_pairs:
                logger.info("  SKIP %s ~ %s: reason=%s, str=%.4f",
                            s['qi'], s['sensitive'], s['reason'],
                            s.get('strength_original', 0))

            if not subgroup_stats:
                n_weak = sum(1 for s in skipped_pairs
                             if s['reason'] == 'weak_relationship')
                cross_tab['skip_reason'] = (
                    f'No meaningful relationships found between analysis '
                    f'variables and QIs — {n_weak} of {n_total_pairs} pairs '
                    f'below threshold (η² < 0.02 / V < 0.05). '
                    f'Cross-tab diagnostic not applicable.')
            else:
                preservations = [s['mean_preservation']
                                 for s in subgroup_stats]
                cross_tab['mean_subgroup_preservation'] = round(
                    sum(preservations) / len(preservations), 4)

    result['cross_tabulation'] = cross_tab

    return result


def compute_fast_qi_utility(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    qi_name: str,
    sensitive_columns: List[str],
    column_types: Optional[Dict[str, str]] = None,
) -> float:
    """Fast utility proxy for per-QI gate inside GENERALIZE.

    Computes eta-squared ratio for sensitive × qi_name pairs.
    eta²(sens, protected_qi) / eta²(sens, original_qi) measures how
    much explanatory power was lost by generalising this QI.

    Returns 0-1 (1.0 = perfect preservation of group-level relationships).
    """
    n = min(len(original), len(processed))
    if n < 10 or qi_name not in processed.columns:
        return 1.0

    _ct = column_types or {}
    _qi_label = _ct.get(qi_name, '')
    preservations = []
    for sc in sensitive_columns:
        if sc not in original.columns or sc not in processed.columns:
            continue
        try:
            sens_vals = original[sc].iloc[:n]
            o_q = original[qi_name].iloc[:n]
            p_q = processed[qi_name].iloc[:n]

            if pd.api.types.is_numeric_dtype(original[sc]):
                # Numeric sensitive: eta² ratio
                eta_orig = _eta_squared(sens_vals, o_q, _qi_label)
                eta_prot = _eta_squared(sens_vals, p_q, _qi_label)
                if eta_orig is not None and eta_prot is not None:
                    if eta_orig > 0.01:
                        preservations.append(
                            min(1.0, eta_prot / eta_orig))
                    else:
                        preservations.append(1.0)
            else:
                # Categorical sensitive: Cramér's V ratio
                cv_orig = _cramers_v(sens_vals, o_q)
                cv_prot = _cramers_v(sens_vals, p_q)
                if cv_orig is not None and cv_prot is not None:
                    if cv_orig > 0.03:
                        preservations.append(
                            min(1.0, cv_prot / cv_orig))
                    else:
                        preservations.append(1.0)
        except Exception:
            pass

    if not preservations:
        return 1.0
    return sum(preservations) / len(preservations)


def _cramers_v(
    x: 'pd.Series',
    y: 'pd.Series',
) -> Optional[float]:
    """Cramér's V: association strength between two categorical variables.

    Uses the bias-corrected formula (Bergsma 2013) when sample is large
    enough.  Returns 0-1 (higher = stronger).  None if degenerate.
    """
    try:
        # Handle string-stored data: treat empty strings as missing
        _x = x.replace('', np.nan) if x.dtype == 'object' else x
        _y = y.replace('', np.nan) if y.dtype == 'object' else y
        mask = _x.notna() & _y.notna()
        x = _x[mask].astype(str)
        y = _y[mask].astype(str)
        n = len(x)
        if n < 4 or x.nunique() < 2 or y.nunique() < 2:
            return None
        # Cap cardinality to prevent huge contingency tables
        if x.nunique() > 500:
            top = x.value_counts().head(500).index
            mask_top = x.isin(top)
            x, y = x[mask_top], y[mask_top]
            n = len(x)
            if n < 4:
                return None
        if y.nunique() > 500:
            top = y.value_counts().head(500).index
            mask_top = y.isin(top)
            x, y = x[mask_top], y[mask_top]
            n = len(x)
            if n < 4:
                return None
        ct = pd.crosstab(x, y)
        # Vectorised chi-squared (no Python loops)
        observed = ct.values.astype(np.float64)
        row_sums = observed.sum(axis=1, keepdims=True)
        col_sums = observed.sum(axis=0, keepdims=True)
        expected = row_sums * col_sums / n
        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_cells = np.where(expected > 0,
                                  (observed - expected) ** 2 / expected, 0.0)
        chi2 = float(chi2_cells.sum())
        r, k = ct.shape
        # Bias-corrected Cramér's V
        phi2 = chi2 / n
        phi2_corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        k_corr = k - (k - 1) ** 2 / (n - 1)
        r_corr = r - (r - 1) ** 2 / (n - 1)
        denom = min(k_corr - 1, r_corr - 1)
        if denom <= 0:
            return None
        return float(min(1.0, (phi2_corr / denom) ** 0.5))
    except Exception:
        return None


def _categorical_preservation(
    orig_x: 'pd.Series',
    orig_y: 'pd.Series',
    prot_x: 'pd.Series',
    prot_y: 'pd.Series',
) -> Optional[float]:
    """Preservation of categorical × categorical relationship.

    Compares the conditional distribution P(sensitive | QI_bin) between
    original and protected data using mean Total Variation Distance
    across QI groups.  Returns 0-1 (1 = perfect preservation).
    """
    try:
        # Handle string-stored data: treat empty strings as missing
        _px = prot_x.replace('', np.nan) if prot_x.dtype == 'object' else prot_x
        _py = prot_y.replace('', np.nan) if prot_y.dtype == 'object' else prot_y
        _oy = orig_y.replace('', np.nan) if orig_y.dtype == 'object' else orig_y
        mask = _px.notna() & _py.notna() & _oy.notna()
        o_y = _oy[mask].astype(str)
        p_x = _px[mask].astype(str)
        p_y = _py[mask].astype(str)
        if len(p_x) < 10:
            return None

        # Filter to groups with >= 3 records
        gc = p_x.groupby(p_x).count()
        valid = gc[gc >= 3].index
        if len(valid) < 2:
            return None
        mask2 = p_x.isin(valid)
        o_y = o_y[mask2]
        p_x = p_x[mask2]
        p_y = p_y[mask2]

        tvds = []
        for grp in valid:
            g_mask = p_x == grp
            o_dist = o_y[g_mask].value_counts(normalize=True)
            p_dist = p_y[g_mask].value_counts(normalize=True)
            all_cats = set(o_dist.index) | set(p_dist.index)
            tvd = 0.5 * sum(
                abs(o_dist.get(c, 0) - p_dist.get(c, 0)) for c in all_cats)
            tvds.append(tvd)

        mean_tvd = sum(tvds) / len(tvds)
        return round(max(0.0, 1.0 - mean_tvd), 4)
    except Exception:
        return None


def _is_continuous_type(col_type_label: str) -> bool:
    """Delegate to centralised ``sdc.column_types.is_continuous_type``."""
    from sdc_engine.sdc.column_types import is_continuous_type
    return is_continuous_type(col_type_label)


def _eta_squared(
    values: 'pd.Series',
    groups: 'pd.Series',
    col_type_label: str = '',
) -> Optional[float]:
    """Association strength between *values* (numeric) and *groups*.

    Parameters
    ----------
    col_type_label : str
        Centralised column-type label from ``SDCConfigure.get_column_types()``
        (e.g. ``'Date — Temporal'``, ``'Integer — Age (demographic)'``).
        Used as the primary signal for whether the QI is continuous/temporal.
        Falls back to pandas dtype inspection when the label is absent.

    Returns 0-1 (higher = stronger).  None if computation fails.
    """
    try:
        # Handle string-stored data: treat empty strings as NaN
        _vals = values.copy()
        _grps = groups.copy()
        if _vals.dtype == 'object':
            _vals = _vals.replace('', np.nan)
            _vals = pd.to_numeric(_vals, errors='coerce')
        if _grps.dtype == 'object':
            _grps = _grps.replace('', np.nan)
        mask = _vals.notna() & _grps.notna()
        v = _vals[mask].astype(float)
        g = _grps[mask]
        if len(v) < 4 or g.nunique() < 2:
            return None

        _n_unique = g.nunique()

        # Determine whether the QI is continuous/temporal.
        # Primary: centralised column-type label from Configure.
        # Fallback: pandas dtype introspection.
        _is_continuous = _is_continuous_type(col_type_label)
        if not _is_continuous and _n_unique > 20:
            _is_continuous = (pd.api.types.is_numeric_dtype(g)
                              or pd.api.types.is_datetime64_any_dtype(g))

        if _n_unique > 20 and _is_continuous:
            # Convert to float for Pearson r²
            _is_datetime = pd.api.types.is_datetime64_any_dtype(g)
            # Label says temporal but dtype is object → parse dates
            if (not _is_datetime
                    and 'date' in col_type_label.lower()
                    or 'temporal' in col_type_label.lower()):
                try:
                    g = pd.to_datetime(g, errors='coerce')
                    _is_datetime = True
                except Exception:
                    pass
            if _is_datetime:
                try:
                    _epoch = pd.Timestamp('1970-01-01')
                    g_float = (g - _epoch).dt.total_seconds().astype(float)
                except Exception:
                    g_float = g.astype(np.int64).astype(float)
            else:
                try:
                    g_float = g.astype(float)
                except (ValueError, TypeError):
                    g_float = None

            # Pearson r² — correct measure for numeric × numeric
            if g_float is not None:
                try:
                    r = v.corr(g_float)
                    if pd.notna(r):
                        return float(r ** 2)
                except (ValueError, TypeError):
                    pass

            # Fallback: quantile-bin
            try:
                if _is_datetime:
                    g = pd.qcut(g.astype(np.int64), q=min(20, _n_unique),
                                duplicates='drop')
                else:
                    g = pd.qcut(g, q=min(20, _n_unique), duplicates='drop')
                if g.nunique() < 2:
                    return None
            except (ValueError, TypeError):
                return None

        grand_mean = v.mean()
        ss_total = ((v - grand_mean) ** 2).sum()
        if ss_total == 0:
            return 1.0
        group_means = v.groupby(g).transform('mean')
        ss_between = ((group_means - grand_mean) ** 2).sum()
        return float(ss_between / ss_total)
    except Exception:
        return None


# ============================================================================
# DISTRIBUTIONAL METRICS — KL divergence, Hellinger distance
# ============================================================================

def compute_distributional_metrics(
    original: pd.DataFrame,
    processed: pd.DataFrame,
    quasi_identifiers: List[str],
    n_bins: int = 20,
    sensitive_columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """Per-variable distributional comparison.

    When *sensitive_columns* is given, targets those columns.

    Returns
    -------
    dict mapping col -> {kl_divergence, hellinger_distance, dtype}
    """
    original, processed = _coerce_pair(original, processed, column_types)
    target_cols = (sensitive_columns if sensitive_columns
                   else quasi_identifiers)
    result: Dict[str, Dict] = {}

    for col in target_cols:
        if col not in original.columns or col not in processed.columns:
            continue

        orig = original[col]
        proc = processed[col]
        n = min(len(orig), len(proc))
        m: Dict = {}

        if (pd.api.types.is_numeric_dtype(orig)
                and pd.api.types.is_numeric_dtype(proc)):
            m['dtype'] = 'numeric'
            try:
                o_vals = orig.iloc[:n].dropna().values.astype(float)
                p_vals = proc.iloc[:n].dropna().values.astype(float)
                if len(o_vals) < 10 or len(p_vals) < 10:
                    continue
                bins = np.histogram_bin_edges(o_vals, bins=n_bins)
                o_hist, _ = np.histogram(o_vals, bins=bins)
                p_hist, _ = np.histogram(p_vals, bins=bins)
                # Normalize + small epsilon to avoid log(0)
                eps = 1e-10
                o_p = o_hist / o_hist.sum() + eps
                p_p = p_hist / p_hist.sum() + eps
                o_p /= o_p.sum()
                p_p /= p_p.sum()
                # KL divergence: D_KL(orig || prot)
                m['kl_divergence'] = round(
                    float(np.sum(o_p * np.log(o_p / p_p))), 6)
                # Hellinger distance
                m['hellinger_distance'] = round(
                    float(np.sqrt(0.5 * np.sum(
                        (np.sqrt(o_p) - np.sqrt(p_p)) ** 2))), 6)
            except Exception:
                m['kl_divergence'] = None
                m['hellinger_distance'] = None
        else:
            m['dtype'] = 'categorical'
            try:
                o_freq = orig.iloc[:n].value_counts(normalize=True)
                p_freq = proc.iloc[:n].value_counts(normalize=True)
                all_cats = sorted(set(o_freq.index) | set(p_freq.index),
                                  key=str)
                eps = 1e-10
                o_p = np.array([o_freq.get(c, 0) + eps for c in all_cats])
                p_p = np.array([p_freq.get(c, 0) + eps for c in all_cats])
                o_p /= o_p.sum()
                p_p /= p_p.sum()
                m['kl_divergence'] = round(
                    float(np.sum(o_p * np.log(o_p / p_p))), 6)
                m['hellinger_distance'] = round(
                    float(np.sqrt(0.5 * np.sum(
                        (np.sqrt(o_p) - np.sqrt(p_p)) ** 2))), 6)
            except Exception:
                m['kl_divergence'] = None
                m['hellinger_distance'] = None

        result[col] = m

    return result


# ============================================================================
# POST-CHECK GATE — cross-tab preservation validation
# ============================================================================

def check_cross_tab_preservation(
    benchmark: Optional[Dict],
    subgroup_threshold: float = 0.60,
    corr_diff_threshold: float = 0.30,
) -> Optional[Dict]:
    """Post-check gate: detect analytical relationship degradation.

    Runs **once** after all preprocessing/protection steps pass the fast
    ``compute_utility()`` threshold.  Examines cross-tabulation metrics
    from ``compute_benchmark_analysis()`` and flags a warning when
    sensitive × QI relationships are badly distorted (e.g. by record
    suppression or perturbation of sensitive values).

    Only meaningful when records were suppressed or sensitive values were
    perturbed.  For pure binning steps, cross-tab is skipped
    (``skip_reason`` is set in benchmark).

    Parameters
    ----------
    benchmark : dict or None
        Output of ``compute_benchmark_analysis()``.
    subgroup_threshold : float
        Minimum mean subgroup-mean preservation across sensitive × QI
        pairs (0-1, default 0.60).  Below this, group-level analytical
        conclusions may be unreliable.  Lowered from 0.70 because
        group-mean correlation is stricter than eta² ratio.
    corr_diff_threshold : float
        Kept for API compatibility; not used (correlation_pairs removed).

    Returns
    -------
    dict or None
        If degradation detected, returns::

            {
                'degraded': True,
                'mean_subgroup_preservation': float,
                'worst_pair': {
                    'sensitive': str, 'qi': str, 'preservation': float
                } or None,
                'n_pairs_checked': int,
                'message': str,  # human-readable summary
            }

        If no degradation (or no cross-tab data), returns None.
    """
    if not benchmark:
        return None

    cross_tab = benchmark.get('cross_tabulation')
    if not cross_tab:
        return None

    # Skipped (pure binning, no record suppression or sensitive changes)
    if cross_tab.get('skip_reason'):
        return None

    subgroup_stats = cross_tab.get('subgroup_means', [])
    if not subgroup_stats:
        return None

    preservations = [s['mean_preservation'] for s in subgroup_stats]
    mean_subgroup = sum(preservations) / len(preservations)

    worst = min(subgroup_stats, key=lambda s: s['mean_preservation'])
    worst_pair = {
        'sensitive': worst['sensitive'],
        'qi': worst['qi'],
        'preservation': worst['mean_preservation'],
    }

    if mean_subgroup >= subgroup_threshold:
        return None

    message = (
        "Analytical relationship warning: simple utility looks acceptable, "
        "but cross-tabulation analysis shows degradation in sensitive × QI "
        f"relationships. Subgroup mean preservation is {mean_subgroup:.0%} "
        f"(below {subgroup_threshold:.0%} threshold). "
        f"Worst pair: {worst['sensitive']} × {worst['qi']} "
        f"at {worst['mean_preservation']:.0%}. "
        "Consider reviewing the Benchmark section of the Utility Report "
        "for details."
    )

    return {
        'degraded': True,
        'mean_subgroup_preservation': mean_subgroup,
        'worst_pair': worst_pair,
        'n_pairs_checked': len(subgroup_stats),
        'message': message,
    }


# ============================================================================
# COMPOSITE UTILITY — per-variable score on sensitive columns
# ============================================================================

def compute_composite_utility(
    utility_score: float,
    benchmark: Optional[Dict],
    utility_weight: float = 0.30,
    per_variable_utility: Optional[Dict[str, Dict]] = None,
) -> float:
    """Blend sensitive preservation, QI loss, and relationship preservation.

    Three-component blend when cross-tab data is available:

    * **50%** sensitive utility (analysis validity)
    * **20%** QI utility (information loss cost)
    * **30%** relationship preservation (cross-tab: subgroup-mean
      correlation for numeric sensitive, conditional-distribution
      preservation for categorical sensitive)

    When cross-tab data is unavailable (no meaningful relationships,
    skipped, or empty), falls back to two-component blend:

    * **70%** sensitive utility
    * **30%** QI utility

    Parameters
    ----------
    utility_score : float
        Output of ``compute_utility()`` on sensitive columns (0-1).
    benchmark : dict or None
        Benchmark analysis containing ``cross_tabulation`` results.
    utility_weight : float
        Weight for per-variable QI utility (default 0.30, used only in
        two-component fallback).
    per_variable_utility : dict or None
        Per-QI metrics from ``compute_per_variable_utility()``.

    Returns
    -------
    float in [0, 1].
    """
    if not per_variable_utility:
        return utility_score

    # Extract per-QI scores
    qi_scores = []
    for col, metrics in per_variable_utility.items():
        if metrics.get('dtype') == 'numeric':
            val = metrics.get('correlation')
        else:
            rp = metrics.get('row_preservation') or 0
            co = metrics.get('category_overlap') or 0
            val = max(rp, co)
        if val is not None:
            qi_scores.append(max(0.0, float(val)))

    if not qi_scores:
        return utility_score

    qi_avg = sum(qi_scores) / len(qi_scores)

    # --- Cross-tab relationship preservation ---
    relationship_avg = None
    if benchmark:
        cross_tab = benchmark.get('cross_tabulation', {})
        if not cross_tab.get('skip_reason'):
            sg_stats = cross_tab.get('subgroup_means', [])
            if sg_stats:
                preservations = [s['mean_preservation'] for s in sg_stats]
                relationship_avg = sum(preservations) / len(preservations)
                # Floor clamp: prevent composite collapse when cross-tab
                # is very low (the analytical-degradation warning already
                # flags this independently).
                relationship_avg = max(0.40, relationship_avg)

    if relationship_avg is not None:
        # 3-component: 50% sensitive + 20% QI + 30% relationship
        blended = (0.50 * utility_score
                   + 0.20 * qi_avg
                   + 0.30 * relationship_avg)
    else:
        # 2-component fallback: 70% sensitive + 30% QI
        sensitive_weight = 1.0 - utility_weight
        blended = sensitive_weight * utility_score + utility_weight * qi_avg

    return round(max(0.0, min(1.0, blended)), 4)
