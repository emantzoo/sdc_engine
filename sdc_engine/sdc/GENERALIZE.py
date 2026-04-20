"""
Generalization (GENERALIZE)
===========================

Reduces quasi-identifier cardinality by generalizing values into broader categories.
This is often a prerequisite for effective k-anonymity when data has high uniqueness.

Strategies:
- Numeric QIs: Bin into ranges (age 25 -> "20-29")
- Categorical QIs: Group rare categories or use hierarchy
- Automatic: Detect and generalize high-cardinality QIs

When to use:
- Before kANON/LOCSUPR when uniqueness is too high
- When QI cardinality * combinations > records * 0.5
- As first step in a pipeline for high-risk data

Input:
- data: pandas DataFrame
- quasi_identifiers: list of QI columns to generalize
- max_categories: target max categories per QI

Output:
- DataFrame with generalized QI columns (original columns replaced or new columns added)
- Metadata with generalization rules applied

Author: SDC Methods Implementation
Date: December 2025
"""

import logging
import sys
import io
import pandas as pd

_log = logging.getLogger(__name__)

# Fix Windows cp1252 crash on Greek/Unicode column names in print().
# Use a wrapper that checks .closed before every write instead of
# permanently replacing sys.stdout (which breaks inside Streamlit when
# the underlying buffer is recycled).
class _SafeStdout(io.TextIOWrapper):
    """TextIOWrapper that silently degrades if the underlying buffer is closed."""
    def write(self, s):
        try:
            if self.buffer.closed:
                return len(s)
            return super().write(s)
        except (ValueError, OSError):
            return len(s)

    def flush(self):
        try:
            if not self.buffer.closed:
                super().flush()
        except (ValueError, OSError):
            pass

if sys.stdout and hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = _SafeStdout(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, TypeError, ValueError):
        pass  # stdout not wrappable — keep default
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import warnings


def apply_generalize(
    data: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
    max_categories: int = 10,
    max_categories_per_qi: Optional[Dict[str, int]] = None,
    numeric_bin_size: Optional[int] = None,
    hierarchies: Optional[Dict[str, Dict[str, str]]] = None,
    strategy: str = 'auto',
    keep_original: bool = False,
    suffix: str = '_gen',
    return_metadata: bool = False,
    adaptive_binning: bool = False,
    verbose: bool = True,
    var_priority: Optional[Dict[str, tuple]] = None,
    reid_target: Optional[float] = None,
    utility_fn: Optional[object] = None,
    utility_threshold: Optional[float] = None,
    column_types: Optional[Dict[str, str]] = None,
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Generalize quasi-identifiers to reduce cardinality and improve k-anonymity potential.

    Parameters:
    -----------
    data : pd.DataFrame
        Input microdata
    quasi_identifiers : list of str, optional
        Columns to generalize. If None, auto-detects high-cardinality QIs.
    max_categories : int, default=10
        Target maximum categories per QI. QIs with more will be generalized.
    max_categories_per_qi : dict, optional
        Per-QI override for max_categories, e.g. {'age': 5, 'region': 15}.
        QIs not listed fall back to the global ``max_categories``.
    numeric_bin_size : int, optional
        Bin size for numeric QIs. If None, auto-calculated based on max_categories.
    hierarchies : dict, optional
        Custom generalization hierarchies for categorical QIs.
        Format: {'column': {'value1': 'general1', 'value2': 'general1', ...}}
    strategy : str, default='auto'
        - 'auto': Generalize only high-cardinality QIs
        - 'all': Generalize all specified QIs
        - 'numeric': Only generalize numeric QIs
        - 'categorical': Only generalize categorical QIs
    keep_original : bool, default=False
        If True, keep original columns and add generalized as new columns.
        If False, replace original columns with generalized versions.
    suffix : str, default='_gen'
        Suffix for new generalized columns (only used if keep_original=True)
    return_metadata : bool, default=False
        If True, returns (generalized_data, metadata)
    adaptive_binning : bool, default=False
        If True, try multiple bin counts for numeric QIs and pick the one
        with the highest per-column correlation. Searches within a small
        range around effective_max (which may already be risk-weighted).
    verbose : bool, default=True
        Print progress messages
    var_priority : dict, optional
        Risk priority per QI from backward elimination.
        Format: {'col': ('HIGH', 45.0), ...}.
        When provided, QIs are sorted HIGH-first so the most impactful
        columns are generalized first.
    reid_target : float, optional
        ReID95 target (e.g. 0.05 for 5%). When provided, the function
        checks ReID95 after each QI is generalized and stops early if
        the target is already met, preserving cardinality on remaining QIs.
    utility_fn : callable, optional
        Callback ``f(result_df, qi_name) -> float`` returning utility
        score (0-1).  Called after each QI generalization.  If the score
        drops below ``utility_threshold``, that QI's generalization is
        rolled back.  For numeric QIs with ``adaptive_binning=True``,
        gentler bin sizes are tried before skipping entirely.
    utility_threshold : float, optional
        Minimum utility (0-1). QIs whose generalization pushes utility
        below this value are skipped (rolled back).
    column_types : dict, optional
        Configure table's column type classification, e.g.
        ``{'age': 'Char (numeric) — Continuous', ...}``.
        Used as source of truth for type detection — skips inline
        numeric/date probing when a column is already classified.
    qi_treatment : dict, optional
        Per-QI treatment levels ``{col: 'Heavy'|'Standard'|'Light'}``.
        When provided, shifts adaptive binning candidate ranges per QI
        (Heavy → aggressive end, Light → gentle end) and adjusts the
        per-QI utility gate threshold (Heavy → more lenient, Light →
        stricter).

    Returns:
    --------
    If return_metadata=False:
        pd.DataFrame : Data with generalized QIs

    If return_metadata=True:
        generalized_data : pd.DataFrame
        metadata : dict with 'rules', 'cardinality_before', 'cardinality_after'

    Examples:
    ---------
    >>> # Auto-generalize high-cardinality QIs
    >>> gen_data = apply_generalize(data, quasi_identifiers=['age', 'zipcode', 'occupation'])

    >>> # Specific bin size for age
    >>> gen_data = apply_generalize(data, quasi_identifiers=['age'], numeric_bin_size=5)

    >>> # Custom hierarchy for education
    >>> hierarchies = {
    ...     'education': {
    ...         'High School': 'Secondary',
    ...         'Bachelor': 'Higher',
    ...         'Master': 'Higher',
    ...         'PhD': 'Higher'
    ...     }
    ... }
    >>> gen_data = apply_generalize(data, quasi_identifiers=['education'], hierarchies=hierarchies)
    """

    if verbose:
        print("="*60)
        print("  GENERALIZATION")
        print("="*60)

    # Auto-detect QIs if not provided
    if quasi_identifiers is None:
        from .detection import detect_quasi_identifiers_enhanced
        detected = detect_quasi_identifiers_enhanced(data, return_scores=True)
        quasi_identifiers = [col for col, s in detected.items() if s['confidence'] >= 0.5]
        if verbose:
            print(f"Auto-detected QIs: {quasi_identifiers}")

    if not quasi_identifiers:
        if verbose:
            print("No QIs to generalize")
        if return_metadata:
            return data.copy(), {'rules': {}, 'message': 'No QIs provided'}
        return data.copy()

    # Validate QIs exist
    missing = [qi for qi in quasi_identifiers if qi not in data.columns]
    if missing:
        raise ValueError(f"QI columns not found: {missing}")

    # Analyze cardinality
    cardinality_before = {}
    needs_generalization = []

    _per_qi = max_categories_per_qi or {}

    for qi in quasi_identifiers:
        card = data[qi].nunique()
        cardinality_before[qi] = card
        effective_max = _per_qi.get(qi, max_categories)

        if strategy == 'all':
            # Even in 'all' mode, skip columns already at/below target —
            # avoids wasteful re-generalization of type-aware preprocessed columns
            if card <= effective_max:
                logging.info(f"[GENERALIZE] {qi}: card={card} already ≤ "
                             f"max={effective_max}, skipping (strategy=all)")
            else:
                needs_generalization.append(qi)
        elif strategy == 'numeric' and pd.api.types.is_numeric_dtype(data[qi]):
            needs_generalization.append(qi)
        elif strategy == 'categorical' and not pd.api.types.is_numeric_dtype(data[qi]):
            needs_generalization.append(qi)
        elif strategy == 'auto' and card > effective_max:
            needs_generalization.append(qi)

    if verbose:
        print(f"\nCardinality analysis:")
        for qi in quasi_identifiers:
            card = cardinality_before[qi]
            flag = " <- WILL GENERALIZE" if qi in needs_generalization else ""
            print(f"  {qi}: {card} unique{flag}")

    if not needs_generalization:
        if verbose:
            print("\nNo QIs need generalization (all within max_categories)")
        if return_metadata:
            return data.copy(), {
                'rules': {},
                'cardinality_before': cardinality_before,
                'cardinality_after': cardinality_before,
                'message': 'No generalization needed'
            }
        return data.copy()

    # Apply generalization
    result = data.copy()
    rules = {}
    cardinality_after = cardinality_before.copy()

    if hierarchies is None:
        hierarchies = {}

    # --- Sequential early-exit: sort by priority (HIGH first) ---
    if var_priority:
        _pri_order = {'HIGH': 0, 'MED-HIGH': 1, 'MODERATE': 2, 'LOW': 3}
        def _sort_key(qi):
            if qi not in var_priority:
                return (4, 0, qi)
            label, pct = var_priority[qi]
            return (_pri_order.get(label, 3), -pct, qi)
        needs_generalization.sort(key=_sort_key)
        if verbose:
            print(f"\n  Priority order: {needs_generalization}")

    skipped_qis = []
    early_exit = False
    reid_after_each = {}

    # Keywords indicating numeric/date in Configure's column_types
    _numeric_kw = {'numeric', 'continuous', 'integer', 'float', 'coded'}
    _date_kw = {'date', 'datetime', 'temporal', 'time'}
    _col_types = column_types or {}

    for qi in needs_generalization:
        _utility_already_checked = False
        col = data[qi]
        effective_max = _per_qi.get(qi, max_categories)
        is_numeric = pd.api.types.is_numeric_dtype(col)
        is_datetime = pd.api.types.is_datetime64_any_dtype(col)

        # --- Source of truth: Configure table's column_types ---
        ct_label = _col_types.get(qi, '').lower()
        if ct_label and not is_numeric and not is_datetime and col.dtype == object:
            if any(kw in ct_label for kw in _numeric_kw):
                col = pd.to_numeric(col, errors='coerce')
                if col.notna().sum() > 0:
                    is_numeric = True
            elif any(kw in ct_label for kw in _date_kw):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    col = pd.to_datetime(col, errors='coerce', dayfirst=True)
                if col.notna().sum() > 0:
                    is_datetime = True

        # --- Fallback: probe data when no column_types available ---
        if not is_numeric and not is_datetime and col.dtype == object:
            sample = col.dropna().head(300)
            sample = sample[sample.astype(str).str.strip() != '']
            if len(sample) > 0:
                numeric_count = pd.to_numeric(sample, errors='coerce').notna().sum()
                if numeric_count / len(sample) > 0.8:
                    col = pd.to_numeric(col, errors='coerce')
                    is_numeric = True
                elif sample.astype(str).str.contains(',').mean() > 0.3:
                    eu_sample = (sample.astype(str)
                                 .str.replace('.', '', regex=False)
                                 .str.replace(',', '.', regex=False))
                    eu_count = pd.to_numeric(eu_sample, errors='coerce').notna().sum()
                    if eu_count / len(sample) > 0.8:
                        col = pd.to_numeric(
                            col.astype(str)
                            .str.replace('.', '', regex=False)
                            .str.replace(',', '.', regex=False),
                            errors='coerce')
                        is_numeric = True

        # --- Fallback: detect date strings ---
        if not is_numeric and not is_datetime and col.dtype == object:
            sample = col.dropna().head(100)
            if len(sample) > 0:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        parsed = pd.to_datetime(sample, errors='coerce', dayfirst=True)
                    if parsed.notna().sum() / len(sample) > 0.8:
                        col = pd.to_datetime(col, errors='coerce', dayfirst=True)
                        is_datetime = True
                except (ValueError, TypeError, OverflowError) as exc:
                    _log.warning("[GENERALIZE] %s: date string detection failed: %s", qi, exc)

        # Log after type detection so the resolved type is accurate
        _type_str = 'numeric' if is_numeric else ('datetime' if is_datetime else 'categorical')
        logging.info(f"[GENERALIZE] {qi}: card={col.nunique()}, "
                     f"effective_max={effective_max}, type={_type_str}")

        if is_datetime:
            # Date generalization: bin by month (or year if too many months)
            # Floor: never collapse dates below 3 distinct periods
            _DATE_MIN_CATEGORIES = 3
            if effective_max < _DATE_MIN_CATEGORIES:
                logging.info(f"[GENERALIZE] {qi}: date floor activated "
                             f"({effective_max} → {_DATE_MIN_CATEGORIES})")
                effective_max = _DATE_MIN_CATEGORIES

            # Skip if already at or below floor
            if int(col.nunique()) <= _DATE_MIN_CATEGORIES:
                logging.info(f"[GENERALIZE] {qi}: date already at cardinality floor "
                             f"({col.nunique()} ≤ {_DATE_MIN_CATEGORIES}), skipping")
                cardinality_after[qi] = int(col.nunique())
                if verbose:
                    print(f"\n  {qi}: SKIPPED (date already ≤ {_DATE_MIN_CATEGORIES} periods)")
                continue

            generalized = _generalize_date_column(col, effective_max)
            rules[qi] = {
                'type': 'date_binning',
                'was_continuous': True,
                'was_date': True,
                'categories_before': int(data[qi].nunique()),
                'categories_after': int(generalized.nunique())
            }

        elif is_numeric:
            # Numeric generalization: binning
            # Cardinality-proportional floor: high-card numerics need more bins
            card_raw = int(col.nunique())
            if card_raw > 5000:
                _num_floor = 15
            elif card_raw > 1000:
                _num_floor = 10
            else:
                _num_floor = 5
            if effective_max < _num_floor:
                logging.info(f"[GENERALIZE] {qi}: numeric bins floor activated "
                             f"({effective_max} → {_num_floor}, card={card_raw})")
                effective_max = _num_floor
            value_range = col.max() - col.min()
            if value_range == 0:
                # Constant column — nothing to bin
                if return_metadata:
                    rules[qi] = {'type': 'numeric_binning', 'was_continuous': True, 'bin_size': 0,
                                 'original_range': (float(col.min()), float(col.max())),
                                 'bins_created': 1}
                cardinality_after[qi] = 1
                continue

            # Detect skewness — use quantile binning for highly skewed data
            # to prevent equal-width bins from collapsing everything into one bin
            try:
                _skew = abs(float(col.skew()))
            except (ValueError, TypeError) as exc:
                _log.warning("[GENERALIZE] %s: skewness calculation failed: %s", qi, exc)
                _skew = 0.0
            _use_quantile = _skew > 3.0  # skew > 3 → quantile binning

            if numeric_bin_size:
                bin_size = numeric_bin_size
            elif adaptive_binning:
                # Generate candidate bin counts — shifted by treatment level
                card = int(col.nunique())
                _qi_level = (qi_treatment or {}).get(qi, 'Standard')
                if qi_treatment and _qi_level != 'Standard':
                    from sdc_engine.sdc.qi_treatment import get_adaptive_binning_candidates
                    candidates_n = get_adaptive_binning_candidates(
                        effective_max, card, _qi_level)
                else:
                    candidates_n = sorted(set([
                        max(3, effective_max // 2),
                        effective_max,
                        min(card, int(effective_max * 1.5)),
                    ]), reverse=True)  # descending = gentlest first (most bins)
                # Pre-compute all candidate generalizations
                all_gens = []
                for n_bins in candidates_n:
                    if _use_quantile:
                        gen = _generalize_numeric_quantile(col, n_bins)
                        bs = f"q{n_bins}"  # marker for quantile
                    else:
                        bs = value_range / n_bins
                        if bs >= 1:
                            bs = int(np.ceil(bs))
                        if value_range / bs < 2:
                            bs = value_range / 2
                        gen = _generalize_numeric_column(col, bs)
                    try:
                        corr = abs(col.corr(gen.astype(float)))
                    except (ValueError, TypeError) as exc:
                        _log.warning("[GENERALIZE] %s: correlation calculation failed: %s", qi, exc)
                        corr = 0
                    all_gens.append((n_bins, gen, bs, corr))
                if _use_quantile:
                    logging.info(f"[GENERALIZE] {qi}: skew={_skew:.1f} → "
                                 f"using quantile binning (candidates={candidates_n})")

                if utility_fn is not None and utility_threshold is not None:
                    # Utility-aware retry: gentlest first (most bins),
                    # progressively more aggressive. Accept first that passes.
                    import time as _time
                    _utility_already_checked = True
                    accepted = False

                    # Treatment-aware gate threshold
                    _gate_threshold = utility_threshold
                    if qi_treatment and qi in qi_treatment:
                        from sdc_engine.sdc.qi_treatment import TREATMENT_GATE_MULT
                        _gate_mult = TREATMENT_GATE_MULT.get(
                            qi_treatment[qi], 1.0)
                        _gate_threshold = utility_threshold * _gate_mult

                    for attempt_i, (n_bins, gen_cand, bs, corr) in enumerate(all_gens):
                        original_col_snap = result[qi].copy()
                        if keep_original:
                            new_col = f"{qi}{suffix}"
                            result[new_col] = gen_cand
                        else:
                            result[qi] = gen_cand
                        cardinality_after[qi] = int(gen_cand.nunique())

                        _ut0 = _time.perf_counter()
                        try:
                            qi_util = utility_fn(result, qi)
                        except Exception as exc:
                            logging.warning(
                                f"[GENERALIZE] {qi}: utility check failed: {exc}")
                            qi_util = 1.0
                        _ut_ms = (_time.perf_counter() - _ut0) * 1000

                        logging.info(
                            f"[GENERALIZE] {qi}: bins={n_bins} "
                            f"({attempt_i+1}/{len(all_gens)}), "
                            f"corr={corr:.3f}, utility={qi_util:.1%}, "
                            f"threshold={_gate_threshold:.0%} "
                            f"(treatment={_qi_level}), "
                            f"took={_ut_ms:.0f}ms")

                        if qi_util >= _gate_threshold:
                            rules[qi] = {
                                'type': 'numeric_binning', 'was_continuous': True,
                                'bin_size': bs,
                                'original_range': (float(col.min()),
                                                   float(col.max())),
                                'bins_created': int(gen_cand.nunique()),
                                'adaptive_retry': {
                                    'attempts': attempt_i + 1,
                                    'total_candidates': len(all_gens),
                                    'accepted_n_bins': n_bins,
                                    'accepted_correlation': round(corr, 4),
                                    'utility': round(qi_util, 4),
                                },
                            }
                            accepted = True
                            logging.info(
                                f"[GENERALIZE] {qi}: accepted bins={n_bins} "
                                f"(attempt {attempt_i+1}/{len(all_gens)})")
                            break

                        # Roll back for next attempt
                        if keep_original:
                            result.drop(columns=[new_col], inplace=True,
                                        errors='ignore')
                        else:
                            result[qi] = original_col_snap

                    if not accepted:
                        cardinality_after[qi] = cardinality_before[qi]
                        skipped_qis.append(qi)
                        logging.info(
                            f"[GENERALIZE] {qi}: ALL {len(all_gens)} bin "
                            f"candidates failed utility gate, skipping")
                        if verbose:
                            print(f"\n  {qi}: SKIPPED "
                                  f"(all bin levels failed utility gate)")
                        continue
                else:
                    # Standard adaptive: pick best correlation
                    best_gen, best_corr, best_bs = None, -1, effective_max
                    for n_bins, gen, bs, corr in all_gens:
                        if corr > best_corr:
                            best_corr, best_gen, best_bs = corr, gen, bs
                    generalized = best_gen
                    bin_size = best_bs
                    rules[qi] = {
                        'type': 'numeric_binning', 'was_continuous': True,
                        'bin_size': bin_size,
                        'original_range': (float(col.min()),
                                           float(col.max())),
                        'bins_created': int(generalized.nunique()),
                        'adaptive_search': {
                            'candidates_tried': candidates_n,
                            'best_n_bins': (int(value_range / bin_size)
                                           if bin_size else 0),
                            'best_correlation': round(best_corr, 4),
                        },
                    }
            else:
                if _use_quantile:
                    # Skewed: quantile-based binning (deterministic path)
                    bin_size = f"q{effective_max}"
                else:
                    # Deterministic: auto-calculate bin size
                    if value_range < 1e-12:
                        # Near-constant column — skip binning
                        bin_size = 1
                    else:
                        bin_size = value_range / effective_max
                        if bin_size >= 1:
                            bin_size = int(np.ceil(bin_size))
                        # Ensure at least 2 bins
                        if bin_size > 0 and value_range / bin_size < 2:
                            bin_size = value_range / 2

            if qi not in rules:
                if _use_quantile:
                    generalized = _generalize_numeric_quantile(col, effective_max)
                    logging.info(f"[GENERALIZE] {qi}: skew={_skew:.1f} → "
                                 f"quantile binning into {effective_max} bins "
                                 f"(result={generalized.nunique()} categories)")
                else:
                    generalized = _generalize_numeric_column(col, bin_size)
                rules[qi] = {
                    'type': 'numeric_binning', 'was_continuous': True,
                    'bin_size': bin_size,
                    'original_range': (float(col.min()), float(col.max())),
                    'bins_created': generalized.nunique()
                }

        elif qi in hierarchies:
            from sdc_engine.sdc.hierarchies import Hierarchy
            h = hierarchies[qi]
            if isinstance(h, Hierarchy):
                # Smart Hierarchy object: pick level where cardinality fits
                for lvl in range(1, h.max_level + 1):
                    if h.cardinality_at(lvl) <= effective_max:
                        generalized = h.generalize(col, lvl)
                        rules[qi] = {
                            'type': 'smart_hierarchy',
                            'builder_type': h.builder_type,
                            'level_used': lvl,
                            'info_loss': h.info_loss_at(lvl),
                            'categories_before': int(col.nunique()),
                            'categories_after': int(generalized.nunique()),
                        }
                        break
                else:
                    # No level fits — use max level
                    generalized = h.generalize(col, h.max_level)
                    rules[qi] = {
                        'type': 'smart_hierarchy',
                        'builder_type': h.builder_type,
                        'level_used': h.max_level,
                        'info_loss': h.info_loss_at(h.max_level),
                        'categories_before': int(col.nunique()),
                        'categories_after': int(generalized.nunique()),
                    }
            elif isinstance(h, dict):
                # Legacy dict hierarchy
                generalized = col.map(h).fillna(col)
                rules[qi] = {
                    'type': 'hierarchy',
                    'hierarchy': h,
                    'categories_before': int(col.nunique()),
                    'categories_after': int(generalized.nunique())
                }
            else:
                continue

        else:
            # Categorical without hierarchy: keep top N + "Other"

            # Skip range-pattern columns (e.g. "20-24" from age binning).
            # Top-N + "Other" destroys ordered structure; let kANON's
            # range-aware merge handle cardinality reduction instead.
            import re as _re
            _rp = _re.compile(r'^(\d+)\s*[-\u2013]\s*(\d+)$')
            _sv = col.dropna().unique()
            _nr = sum(1 for v in _sv if _rp.match(str(v)))
            if len(_sv) >= 4 and _nr / len(_sv) > 0.8:
                logging.info(f"[GENERALIZE] {qi}: range-pattern column "
                             f"({len(_sv)} bins), skipping top-N grouping")
                cardinality_after[qi] = int(col.nunique())
                skipped_qis.append(qi)
                if verbose:
                    print(f"\n  {qi}: SKIPPED (range-pattern, {len(_sv)} bins)")
                continue

            # Floor: never collapse categoricals below 3 — prevents
            # total information destruction (e.g., all→"Other")
            _CAT_MIN_CATEGORIES = 3
            if effective_max < _CAT_MIN_CATEGORIES:
                logging.info(f"[GENERALIZE] {qi}: categorical floor activated "
                             f"({effective_max} → {_CAT_MIN_CATEGORIES})")
                effective_max = _CAT_MIN_CATEGORIES

            # Skip if already at or below floor
            if int(col.nunique()) <= _CAT_MIN_CATEGORIES:
                logging.info(f"[GENERALIZE] {qi}: already at cardinality floor "
                             f"({col.nunique()} ≤ {_CAT_MIN_CATEGORIES}), skipping")
                cardinality_after[qi] = int(col.nunique())
                if verbose:
                    print(f"\n  {qi}: SKIPPED (already ≤ {_CAT_MIN_CATEGORIES} categories)")
                continue

            if adaptive_binning and utility_fn is not None and utility_threshold is not None:
                # Adaptive retry: gentlest first (most categories kept),
                # progressively more aggressive.
                import time as _time
                _utility_already_checked = True
                card = int(col.nunique())

                # Treatment-shifted candidate range
                _qi_level_cat = (qi_treatment or {}).get(qi, 'Standard')
                if qi_treatment and _qi_level_cat != 'Standard':
                    from sdc_engine.sdc.qi_treatment import get_adaptive_binning_candidates
                    candidates_n = get_adaptive_binning_candidates(
                        effective_max, card - 1, _qi_level_cat)
                else:
                    candidates_n = sorted(set([
                        max(2, effective_max // 2),
                        effective_max,
                        min(card - 1, int(effective_max * 1.5)),
                    ]), reverse=True)  # descending = gentlest first

                # Treatment-aware gate threshold
                _cat_gate_threshold = utility_threshold
                if qi_treatment and qi in qi_treatment:
                    from sdc_engine.sdc.qi_treatment import TREATMENT_GATE_MULT
                    _cat_gate_mult = TREATMENT_GATE_MULT.get(
                        qi_treatment[qi], 1.0)
                    _cat_gate_threshold = utility_threshold * _cat_gate_mult

                accepted = False
                for attempt_i, n_keep in enumerate(candidates_n):
                    gen_cand = _generalize_categorical_topn(col, n_keep)
                    original_col_snap = result[qi].copy()
                    if keep_original:
                        new_col = f"{qi}{suffix}"
                        result[new_col] = gen_cand
                    else:
                        result[qi] = gen_cand
                    cardinality_after[qi] = int(gen_cand.nunique())

                    _ut0 = _time.perf_counter()
                    try:
                        qi_util = utility_fn(result, qi)
                    except Exception as exc:
                        logging.warning(
                            f"[GENERALIZE] {qi}: utility check failed: {exc}")
                        qi_util = 1.0
                    _ut_ms = (_time.perf_counter() - _ut0) * 1000

                    logging.info(
                        f"[GENERALIZE] {qi}: top-{n_keep} "
                        f"({attempt_i+1}/{len(candidates_n)}), "
                        f"card={int(gen_cand.nunique())}, "
                        f"utility={qi_util:.1%}, "
                        f"threshold={_cat_gate_threshold:.0%} "
                        f"(treatment={_qi_level_cat}), "
                        f"took={_ut_ms:.0f}ms")

                    if qi_util >= _cat_gate_threshold:
                        rules[qi] = {
                            'type': 'top_n_grouping',
                            'max_categories': n_keep,
                            'categories_before': int(col.nunique()),
                            'categories_after': int(gen_cand.nunique()),
                            'adaptive_retry': {
                                'attempts': attempt_i + 1,
                                'total_candidates': len(candidates_n),
                                'accepted_n_keep': n_keep,
                                'utility': round(qi_util, 4),
                            },
                        }
                        accepted = True
                        logging.info(
                            f"[GENERALIZE] {qi}: accepted top-{n_keep} "
                            f"(attempt {attempt_i+1}/{len(candidates_n)})")
                        break

                    # Roll back for next attempt
                    if keep_original:
                        result.drop(columns=[new_col], inplace=True,
                                    errors='ignore')
                    else:
                        result[qi] = original_col_snap

                if not accepted:
                    cardinality_after[qi] = cardinality_before[qi]
                    skipped_qis.append(qi)
                    logging.info(
                        f"[GENERALIZE] {qi}: ALL {len(candidates_n)} categorical "
                        f"candidates failed utility gate, skipping")
                    if verbose:
                        print(f"\n  {qi}: SKIPPED "
                              f"(all category levels failed utility gate)")
                    continue
            else:
                generalized = _generalize_categorical_topn(col, effective_max)
                rules[qi] = {
                    'type': 'top_n_grouping',
                    'max_categories': effective_max,
                    'categories_before': int(col.nunique()),
                    'categories_after': int(generalized.nunique())
                }

        if not _utility_already_checked:
            # Store result (tentatively — may roll back if utility gate fails)
            original_col = result[qi].copy()
            if keep_original:
                new_col = f"{qi}{suffix}"
                result[new_col] = generalized
                rules[qi]['new_column'] = new_col
            else:
                result[qi] = generalized

            cardinality_after[qi] = int(generalized.nunique())

            # --- Per-QI utility gate: roll back if utility drops too low ---
            if utility_fn is not None and utility_threshold is not None:
                try:
                    import time as _time
                    _ut0 = _time.perf_counter()
                    qi_util = utility_fn(result, qi)
                    _ut_ms = (_time.perf_counter() - _ut0) * 1000
                    # Treatment-aware gate threshold
                    _det_gate = utility_threshold
                    _det_level = (qi_treatment or {}).get(qi, 'Standard')
                    if _det_level != 'Standard':
                        from sdc_engine.sdc.qi_treatment import TREATMENT_GATE_MULT
                        _det_gate = utility_threshold * TREATMENT_GATE_MULT.get(
                            _det_level, 1.0)
                    logging.info(
                        f"[GENERALIZE] {qi}: utility={qi_util:.1%}, "
                        f"threshold={_det_gate:.0%} "
                        f"(treatment={_det_level}), took={_ut_ms:.0f}ms")
                    if qi_util < _det_gate:
                        # Roll back this QI's generalization
                        if keep_original:
                            result.drop(columns=[new_col], inplace=True)
                        else:
                            result[qi] = original_col
                        cardinality_after[qi] = cardinality_before[qi]
                        del rules[qi]
                        skipped_qis.append(qi)
                        logging.info(
                            f"[GENERALIZE] {qi}: ROLLED BACK "
                            f"(utility {qi_util:.1%} < {_det_gate:.0%})")
                        if verbose:
                            print(f"\n  {qi}: SKIPPED (utility {qi_util:.1%} "
                                  f"< {_det_gate:.0%})")
                        continue  # skip ReID check, move to next QI
                except Exception as e:
                    logging.warning(f"[GENERALIZE] {qi}: utility check failed: {e}")

        if verbose:
            print(f"\n  {qi}: {cardinality_before[qi]} -> {cardinality_after[qi]} categories")
            print(f"    Method: {rules[qi]['type']}")

        # --- Early-exit check: stop if ReID target already met ---
        if reid_target is not None:
            try:
                from .metrics.reid import calculate_reid
                reid_check = calculate_reid(result, quasi_identifiers)
                reid_now = reid_check.get('reid_95', 1.0)
                reid_after_each[qi] = reid_now
                logging.info(f"[GENERALIZE] {qi}: card {cardinality_before[qi]}→"
                             f"{cardinality_after.get(qi, '?')}, "
                             f"ReID95={reid_now:.1%} (target={reid_target:.1%})")
                if verbose:
                    print(f"    ReID95 after {qi}: {reid_now:.1%}"
                          f" (target: {reid_target:.1%})")
                if reid_now <= reid_target:
                    processed = set(rules.keys())
                    skipped_qis = [q for q in needs_generalization
                                   if q not in processed]
                    early_exit = True
                    if verbose:
                        print(f"\n  >> ReID target met after {qi}! "
                              f"Skipping {len(skipped_qis)} remaining QIs: "
                              f"{skipped_qis}")
                    break
            except Exception as e:
                if verbose:
                    print(f"    [ReID check failed: {e}]")

    # Summary
    if verbose:
        total_combos_before = np.prod([cardinality_before[qi] for qi in quasi_identifiers])
        total_combos_after = np.prod([cardinality_after[qi] for qi in quasi_identifiers])
        reduction = 1 - (total_combos_after / total_combos_before) if total_combos_before > 0 else 0

        print(f"\n--- Summary ---")
        print(f"Total QI combinations: {total_combos_before:,} -> {total_combos_after:,}")
        print(f"Reduction: {reduction:.1%}")

    if return_metadata:
        actually_generalized = [qi for qi in needs_generalization
                                if qi not in skipped_qis]
        # Build column_details for UI compatibility
        column_details = {}
        for qi in actually_generalized:
            column_details[qi] = {
                'unique_before': cardinality_before[qi],
                'unique_after': cardinality_after[qi],
                'method': rules[qi]['type'],
                'reduction_pct': ((cardinality_before[qi] - cardinality_after[qi]) / cardinality_before[qi] * 100) if cardinality_before[qi] > 0 else 0
            }

        metadata = {
            'rules': rules,
            'cardinality_before': cardinality_before,
            'cardinality_after': cardinality_after,
            'cardinality_reduction': {qi: (cardinality_before[qi], cardinality_after[qi]) for qi in actually_generalized},
            'quasi_identifiers': quasi_identifiers,
            'generalized_columns': actually_generalized,
            'generalized_qis': actually_generalized,
            'columns_modified': actually_generalized,
            'column_details': column_details,
            'total_qis_generalized': len(actually_generalized),
            'skipped_qis': skipped_qis,
            'early_exit': early_exit,
            'reid_after_each': reid_after_each,
        }
        return result, metadata

    return result


def _generalize_numeric_column(series: pd.Series, bin_size) -> pd.Series:
    """Generalize numeric column into bins.

    Supports both int and float bin_size for small-range data.
    """
    has_nan = series.isna()

    min_val = series.min()
    max_val = series.max()

    # Build bin edges using numpy (handles floats properly)
    use_int = isinstance(bin_size, int) or (isinstance(bin_size, float) and bin_size == int(bin_size) and bin_size >= 1)
    if use_int:
        bin_size = int(bin_size)
        min_bin = (int(min_val) // bin_size) * bin_size
        max_bin = ((int(max_val) // bin_size) + 1) * bin_size
        bins = list(range(min_bin, max_bin + bin_size, bin_size))
        labels = [f"{b}-{b + bin_size - 1}" for b in bins[:-1]]
    else:
        # Float bin sizes for small-range data
        min_bin = np.floor(min_val / bin_size) * bin_size
        max_bin = np.ceil(max_val / bin_size) * bin_size
        bins = list(np.arange(min_bin, max_bin + bin_size * 0.5, bin_size))
        if len(bins) < 2:
            bins = [min_bin, max_bin + bin_size]
        # Format labels with appropriate precision
        decimals = max(1, -int(np.floor(np.log10(abs(bin_size)))) + 1) if bin_size > 0 else 1
        labels = [f"{b:.{decimals}f}-{b + bin_size:.{decimals}f}" for b in bins[:-1]]

    # Ensure at least 2 bins
    if len(bins) < 2:
        bins = [float(min_val), float(max_val) + 1]
        labels = [f"{min_val}-{max_val}"]

    result = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=False)
    result = result.astype(str)
    result[has_nan] = None
    return result


def _generalize_numeric_quantile(series: pd.Series, n_bins: int) -> pd.Series:
    """Quantile-based (equal-frequency) binning for skewed numeric columns.

    Unlike equal-width binning, this ensures each bin contains roughly the
    same number of records, preventing extreme skew from collapsing all
    values into a single bin.
    """
    has_nan = series.isna()
    valid = series.dropna()
    if len(valid) == 0 or valid.nunique() <= 1:
        return series.copy()

    try:
        # Use qcut for equal-frequency bins; duplicates='drop' handles
        # cases where quantile boundaries coincide
        binned = pd.qcut(valid, q=n_bins, duplicates='drop')
    except (ValueError, TypeError):
        try:
            # Fallback 1: rank-based percentile bins
            ranks = valid.rank(method='first')
            binned = pd.qcut(ranks, q=n_bins, duplicates='drop')
        except (ValueError, TypeError):
            # Fallback 2: equal-width bins (always works)
            binned = pd.cut(valid, bins=max(2, n_bins), duplicates='drop')

    # Build readable labels from the interval edges
    intervals = binned.cat.categories
    labels = []
    for iv in intervals:
        lo, hi = iv.left, iv.right
        # Use integer labels when values are large enough
        if abs(lo) >= 10 and abs(hi) >= 10:
            labels.append(f"{int(lo)}-{int(hi)}")
        else:
            labels.append(f"{lo:.1f}-{hi:.1f}")

    label_map = dict(zip(intervals, labels))
    result_valid = binned.map(label_map).astype(str)

    result = pd.Series(None, index=series.index, dtype=object)
    result[valid.index] = result_valid
    result[has_nan] = None
    return result


def _generalize_date_column(series: pd.Series, max_categories: int = 10) -> pd.Series:
    """Bin datetime column by appropriate unit (month, quarter, or year).

    Selects the **finest** granularity that stays within *max_categories*,
    iterating from month → quarter → year.  Falls back to finer granularity
    if the chosen level would produce fewer than 3 distinct periods (which
    destroys temporal information).
    """
    _MIN_PERIODS = 3
    has_nan = series.isna()

    # Compute period counts for each granularity (finest → coarsest)
    valid = series.dropna()
    counts = {
        'M': valid.dt.to_period('M').nunique(),
        'Q': valid.dt.to_period('Q').nunique(),
        'Y': valid.dt.year.nunique(),
    }

    # Pick finest granularity that fits within max_categories
    unit = 'Y'  # fallback
    for u in ('M', 'Q', 'Y'):
        if counts[u] <= max_categories:
            unit = u
            break

    # Guard: if chosen unit produces < 3 periods, fall back to finer
    if counts[unit] < _MIN_PERIODS:
        for finer in ('Q', 'M'):
            if counts.get(finer, 0) >= _MIN_PERIODS:
                unit = finer
                break

    # Apply chosen unit
    if unit == 'Q':
        result = series.dt.to_period('Q').astype(str)
    elif unit == 'Y':
        result = series.dt.strftime('%Y')
    else:  # month
        result = series.dt.to_period('M').astype(str)

    result = result.copy()
    result.loc[has_nan] = None
    return result


def _generalize_categorical_topn(series: pd.Series, n: int) -> pd.Series:
    """Keep top N categories, group rest as 'Other'."""
    value_counts = series.value_counts()

    if len(value_counts) <= n:
        return series

    top_categories = value_counts.nlargest(n - 1).index.tolist()
    return series.apply(lambda x: x if pd.isna(x) else (x if x in top_categories else 'Other'))


def suggest_generalization(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    target_k: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Analyze data and suggest generalization strategy.

    Returns recommendations for how to generalize QIs to achieve target k-anonymity.
    """
    n_records = len(data)

    analysis = {
        'n_records': n_records,
        'target_k': target_k,
        'quasi_identifiers': {},
        'total_combinations': 1,
        'recommendation': None
    }

    for qi in quasi_identifiers:
        if qi not in data.columns:
            continue

        col = data[qi]
        cardinality = col.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(col)

        analysis['quasi_identifiers'][qi] = {
            'cardinality': cardinality,
            'type': 'numeric' if is_numeric else 'categorical',
            'suggested_action': None
        }
        analysis['total_combinations'] *= cardinality

    # Calculate theoretical max groups for k-anonymity
    max_groups = n_records // target_k

    if verbose:
        print("="*60)
        print("  GENERALIZATION RECOMMENDATION")
        print("="*60)
        print(f"\nRecords: {n_records}")
        print(f"Target k: {target_k}")
        print(f"Max possible groups: {max_groups}")
        print(f"Current QI combinations: {analysis['total_combinations']:,}")

    if analysis['total_combinations'] <= max_groups:
        analysis['recommendation'] = 'no_generalization_needed'
        if verbose:
            print(f"\n✓ Current combinations ({analysis['total_combinations']}) <= max groups ({max_groups})")
            print("  No generalization needed for k-anonymity")
    else:
        analysis['recommendation'] = 'generalization_needed'

        # Calculate target cardinality per QI
        n_qis = len(quasi_identifiers)
        target_cardinality = int(np.power(max_groups, 1/n_qis))

        if verbose:
            print(f"\n[!] Current combinations ({analysis['total_combinations']:,}) > max groups ({max_groups})")
            print(f"  Target ~{target_cardinality} categories per QI")
            print("\nSuggested actions:")

        for qi, info in analysis['quasi_identifiers'].items():
            if info['cardinality'] > target_cardinality:
                if info['type'] == 'numeric':
                    value_range = data[qi].max() - data[qi].min()
                    suggested_bin = max(1, int(np.ceil(value_range / target_cardinality)))
                    info['suggested_action'] = f"Bin into {target_cardinality} groups (bin_size={suggested_bin})"
                else:
                    info['suggested_action'] = f"Reduce to top {target_cardinality-1} + Other"

                if verbose:
                    print(f"  {qi}: {info['cardinality']} -> {target_cardinality}")
                    print(f"    {info['suggested_action']}")
            else:
                info['suggested_action'] = 'Keep as-is'
                if verbose:
                    print(f"  {qi}: Keep as-is ({info['cardinality']} categories)")

    return analysis


def compute_risk_weighted_limits(
    var_priority: Dict[str, tuple],
    global_max_categories: int = 10,
) -> Dict[str, int]:
    """Compute per-QI max_categories based on risk contribution.

    Higher-risk QIs get tighter (fewer categories) limits to reduce
    their contribution to combination space.  Lower-risk QIs keep
    more detail to preserve analytical utility.

    The tier thresholds (15%, 8%, 3%) and multipliers are initial
    defaults based on heuristic reasoning.  They should be empirically
    validated once the composite utility metric enables systematic
    comparison of risk-weighted vs global parameters across datasets.

    Parameters
    ----------
    var_priority : dict
        {column: (priority_label, pct_score)} from Configure's
        backward-elimination analysis.
    global_max_categories : int
        Base max_categories value (from tier or user setting).

    Returns
    -------
    dict
        {column: max_categories} for each QI with known priority.
    """
    if not var_priority:
        return {}
    limits: Dict[str, int] = {}
    for col, (label, _pct) in var_priority.items():
        if 'HIGH' in label and 'MED' not in label:
            limits[col] = max(5, global_max_categories // 2)
        elif 'MED' in label:
            limits[col] = max(5, int(global_max_categories * 0.8))
        elif 'MODERATE' in label:
            limits[col] = global_max_categories
        else:
            limits[col] = min(20, int(global_max_categories * 1.5))
    return limits


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("GENERALIZE Examples")
    print("="*60)

    # Create test data
    np.random.seed(42)
    n = 200
    test_data = pd.DataFrame({
        'age': np.random.randint(18, 75, n),
        'gender': np.random.choice(['M', 'F'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'zipcode': np.random.randint(10000, 99999, n),
        'occupation': np.random.choice([
            'Engineer', 'Teacher', 'Doctor', 'Lawyer', 'Artist',
            'Nurse', 'Manager', 'Sales', 'Admin', 'Other'
        ], n),
        'income': np.random.randint(30000, 150000, n)
    })

    print("\nOriginal data:")
    print(f"  age: {test_data['age'].nunique()} unique")
    print(f"  zipcode: {test_data['zipcode'].nunique()} unique")
    print(f"  occupation: {test_data['occupation'].nunique()} unique")

    # Example 1: Auto-generalize
    print("\n" + "="*60)
    print("Example 1: Auto-generalize high-cardinality QIs")
    print("="*60)

    result1, meta1 = apply_generalize(
        test_data,
        quasi_identifiers=['age', 'gender', 'zipcode', 'occupation'],
        max_categories=10,
        return_metadata=True
    )

    # Example 2: Custom bin size
    print("\n" + "="*60)
    print("Example 2: Custom bin size for age")
    print("="*60)

    result2, meta2 = apply_generalize(
        test_data,
        quasi_identifiers=['age'],
        numeric_bin_size=5,
        return_metadata=True
    )

    print(f"\nAge bins: {result2['age'].unique()[:10].tolist()}...")

    # Example 3: Get recommendations
    print("\n" + "="*60)
    print("Example 3: Get generalization recommendations")
    print("="*60)

    rec = suggest_generalization(
        test_data,
        quasi_identifiers=['age', 'gender', 'zipcode', 'occupation'],
        target_k=5
    )
