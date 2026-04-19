"""
Smart Method Configuration
===========================

Pre-application analysis for each protection method.  Runs *before* the
first method call to:

1. **Estimate suppression** (kANON / LOCSUPR) — warns when data loss
   will be excessive and can trigger a pre-application method switch.
2. **Detect category dominance** (PRAM) — flags QIs where one category
   holds >80% of records, making PRAM perturbation ineffective.
3. **Scale noise to IQR** (NOISE) — computes per-variable noise
   magnitude proportional to each column's interquartile range, so
   columns with different scales receive appropriately sized noise.

All functions return a dict with ``warnings`` (list[str]) and optional
parameter overrides that the protection engine injects before calling
the actual method.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── 1. kANON pre-estimation ────────────────────────────────────────────


def suggest_kanon_config(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    target_k: int,
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Estimate suppression rate and select kANON strategy.

    Returns
    -------
    dict with keys:
        starting_k           : int — recommended first k to try
        strategy             : str — 'suppression_dominant' | 'hybrid' | 'generalization_dominant'
        strategy_reason      : str — human-readable justification
        qi_generalization_order : list[str] — QIs sorted by treatment (Heavy first)
        estimated_suppression_rate : float
        warnings             : list[str]
        switch_method        : str | None — if set, engine should switch primary
    """
    qi_treatment = qi_treatment or {}
    warnings: List[str] = []

    qi_in_df = [q for q in quasi_identifiers if q in data.columns]
    if not qi_in_df:
        return {'starting_k': target_k, 'strategy': 'generalization_dominant',
                'strategy_reason': 'No QIs in data', 'warnings': [],
                'estimated_suppression_rate': 0, 'switch_method': None,
                'qi_generalization_order': []}

    # Single groupby to get equivalence class sizes
    try:
        groups = data.groupby(qi_in_df, dropna=False).size()
    except Exception:
        return {'starting_k': target_k, 'strategy': 'generalization_dominant',
                'strategy_reason': 'Groupby failed', 'warnings': [],
                'estimated_suppression_rate': 0, 'switch_method': None,
                'qi_generalization_order': quasi_identifiers}

    n_groups = len(groups)
    n_records = len(data)

    min_group = int(groups.min())
    pct_below_target = (groups < target_k).sum() / n_groups if n_groups else 0
    records_in_small = int(groups[groups < target_k].sum())
    est_suppression = records_in_small / n_records if n_records else 0

    # ── Starting k ──────────────────────────────────────────────
    if pct_below_target < 0.05:
        starting_k = target_k
    elif min_group >= target_k - 2:
        starting_k = target_k
    elif pct_below_target > 0.50:
        starting_k = max(3, min_group)
    else:
        starting_k = max(3, target_k - 2)

    # ── Strategy selection ──────────────────────────────────────
    if pct_below_target < 0.10:
        strategy = 'suppression_dominant'
        reason = (f"Only {pct_below_target:.0%} of equivalence classes violate "
                  f"k={target_k}. Suppressing outlier records is less "
                  f"destructive than generalizing all QIs.")
    elif pct_below_target < 0.40:
        strategy = 'hybrid'
        reason = (f"{pct_below_target:.0%} of classes violate k={target_k}. "
                  f"Hybrid: generalize high-risk QIs, suppress remaining outliers.")
    else:
        strategy = 'generalization_dominant'
        reason = (f"{pct_below_target:.0%} of classes violate k={target_k}. "
                  f"Structural generalization needed across QIs.")

    # ── Suppression warnings ────────────────────────────────────
    switch_method = None
    if est_suppression > 0.25:
        warnings.append(
            f"Estimated suppression: {est_suppression:.0%} of records. "
            f"Consider more aggressive preprocessing or lowering k.")
        switch_method = 'PRAM'  # suggest perturbative instead
    elif est_suppression > 0.15:
        warnings.append(
            f"Estimated suppression: {est_suppression:.0%}. "
            f"Moderate data loss expected.")

    # ── QI order by treatment (Heavy first) ─────────────────────
    qi_order = sorted(quasi_identifiers, key=lambda q: {
        'Heavy': 0, 'Standard': 1, 'Light': 2,
    }.get(qi_treatment.get(q, 'Standard'), 1))

    log.info("[SmartConfig/kANON] target_k=%d  starting_k=%d  strategy=%s  "
             "est_suppression=%.1f%%  pct_below=%.1f%%  switch=%s",
             target_k, starting_k, strategy,
             est_suppression * 100, pct_below_target * 100, switch_method)

    return {
        'starting_k': starting_k,
        'strategy': strategy,
        'strategy_reason': reason,
        'qi_generalization_order': qi_order,
        'estimated_suppression_rate': est_suppression,
        'warnings': warnings,
        'switch_method': switch_method,
    }


# ── 2. LOCSUPR pre-estimation ──────────────────────────────────────────


def suggest_locsupr_config(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    target_k: int,
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Estimate per-QI suppression exposure for LOCSUPR.

    Returns
    -------
    dict with keys:
        importance_weights          : dict[str, int]
        qi_violation_exposure       : dict[str, int]
        estimated_suppression_rate  : float  (cell-level)
        warnings                    : list[str]
        switch_method               : str | None
    """
    from sdc_engine.sdc.qi_treatment import build_locsupr_weights

    qi_treatment = qi_treatment or {}
    warnings: List[str] = []

    # Weights from treatment levels
    importance_weights = build_locsupr_weights(
        quasi_identifiers, qi_treatment) or {q: 3 for q in quasi_identifiers}

    qi_in_df = [q for q in quasi_identifiers if q in data.columns]
    n_records = len(data)
    n_qis = len(qi_in_df)

    # Estimate per-QI violation exposure
    qi_violation_exposure: Dict[str, int] = {}
    total_records_at_risk = 0
    try:
        if qi_in_df:
            groups = data.groupby(qi_in_df, dropna=False).size()
            violating = groups[groups < target_k]
            total_records_at_risk = int(violating.sum())

            # Per-QI: how many unique values of that QI appear in violating rows
            if len(violating) > 0:
                violating_idx = violating.index
                # Build a mask of rows in violating groups
                keys_df = data[qi_in_df].copy()
                keys_df['__group_size__'] = keys_df.groupby(qi_in_df)[qi_in_df[0]].transform('size')
                at_risk_mask = keys_df['__group_size__'] < target_k
                for qi in qi_in_df:
                    qi_violation_exposure[qi] = int(
                        data.loc[at_risk_mask, qi].nunique())
    except Exception:
        pass

    # Cell-level suppression estimate
    est_cell_supp = 0.0
    if n_records > 0 and n_qis > 0:
        est_cells = total_records_at_risk * n_qis
        total_cells = n_records * n_qis
        est_cell_supp = est_cells / total_cells if total_cells > 0 else 0

    # Warnings: concentration + overall
    switch_method = None
    total_exposure = sum(qi_violation_exposure.values()) or 1
    for qi, exposure in qi_violation_exposure.items():
        pct = exposure / total_exposure
        if pct > 0.60:
            warnings.append(
                f"'{qi}' absorbs ~{pct:.0%} of suppressions. "
                f"Consider heavier preprocessing on this QI.")

    if est_cell_supp > 0.10:
        warnings.append(
            f"Estimated cell suppression: {est_cell_supp:.0%}. "
            f"kANON generalization may be less destructive.")
        switch_method = 'kANON'

    log.info("[SmartConfig/LOCSUPR] target_k=%d  est_cell_supp=%.1f%%  "
             "records_at_risk=%d  switch=%s",
             target_k, est_cell_supp * 100, total_records_at_risk, switch_method)

    return {
        'importance_weights': importance_weights,
        'qi_violation_exposure': qi_violation_exposure,
        'estimated_suppression_rate': est_cell_supp,
        'warnings': warnings,
        'switch_method': switch_method,
    }


# ── 3. PRAM dominance detection ───────────────────────────────────────


def suggest_pram_config(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    base_p_change: float = 0.20,
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Detect category dominance that makes PRAM ineffective.

    Returns
    -------
    dict with keys:
        per_variable_effectiveness : dict[str, str]  ('high'|'medium'|'low')
        low_effectiveness_qis      : list[str]
        warnings                   : list[str]
        switch_method              : str | None
    """
    qi_treatment = qi_treatment or {}
    warnings: List[str] = []
    effectiveness: Dict[str, str] = {}

    n_low = 0
    for qi in quasi_identifiers:
        if qi not in data.columns:
            continue
        vc = data[qi].value_counts(normalize=True)
        if len(vc) == 0:
            continue

        dominant_pct = float(vc.iloc[0])
        dominant_cat = vc.index[0]
        cardinality = len(vc)

        if dominant_pct > 0.80:
            effectiveness[qi] = 'low'
            n_low += 1
            warnings.append(
                f"PRAM ineffective on '{qi}' — '{dominant_cat}' dominates "
                f"at {dominant_pct:.0%}. Most records stay in the dominant "
                f"category regardless of perturbation.")
        elif dominant_pct > 0.60 or cardinality < 3:
            effectiveness[qi] = 'medium'
        else:
            effectiveness[qi] = 'high'

    # If more than half of QIs have low effectiveness, suggest switch
    switch_method = None
    if n_low > 0 and n_low >= len(quasi_identifiers) / 2:
        switch_method = 'kANON'
        warnings.append(
            f"{n_low}/{len(quasi_identifiers)} QIs have dominant categories. "
            f"kANON generalization may be more effective.")

    low_eff_qis = [q for q, e in effectiveness.items() if e == 'low']

    log.info("[SmartConfig/PRAM] low_eff=%d/%d  switch=%s  details=%s",
             n_low, len(quasi_identifiers), switch_method, effectiveness)

    return {
        'per_variable_effectiveness': effectiveness,
        'low_effectiveness_qis': low_eff_qis,
        'warnings': warnings,
        'switch_method': switch_method,
    }


# ── 4. NOISE IQR-proportional magnitude ───────────────────────────────


def suggest_noise_config(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    base_magnitude: float = 0.10,
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Compute IQR-proportional per-variable noise magnitude.

    Instead of using a uniform ``magnitude`` for all QIs (which ignores
    column scale), this computes:

        noise_std = base_magnitude × treatment_multiplier × IQR

    where IQR = Q3 - Q1.  Falls back to column std when IQR is zero.

    Also checks whether the planned noise would distort distributions
    or break pairwise correlations.

    Returns
    -------
    dict with keys:
        per_variable_magnitude : dict[str, float]  — {col: effective_magnitude}
        distribution_risk      : dict[str, str]    — {col: 'low'|'moderate'|'high'}
        correlation_warnings   : list[str]
        warnings               : list[str]
    """
    qi_treatment = qi_treatment or {}
    per_var_mag: Dict[str, float] = {}
    dist_risk: Dict[str, str] = {}
    warnings: List[str] = []
    iqr_info: Dict[str, float] = {}    # for correlation check

    treatment_mult = {'Heavy': 1.5, 'Standard': 1.0, 'Light': 0.5}

    for qi in quasi_identifiers:
        if qi not in data.columns:
            continue
        col = data[qi]
        if not pd.api.types.is_numeric_dtype(col):
            continue

        col_clean = col.dropna()
        if len(col_clean) < 5:
            continue

        # IQR (robust to outliers)
        q1, q3 = float(col_clean.quantile(0.25)), float(col_clean.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            iqr = float(col_clean.std())
        if iqr == 0:
            continue  # constant column

        level = qi_treatment.get(qi, 'Standard')
        mult = treatment_mult.get(level, 1.0)

        # Effective relative magnitude (used by NOISE's `relative=True` path)
        # The NOISE method multiplies magnitude × col_std to get noise_scale.
        # We want noise_scale ≈ base_magnitude × mult × IQR.
        # So effective_magnitude = base_magnitude × mult × (IQR / col_std).
        col_std = float(col_clean.std())
        if col_std > 0:
            iqr_ratio = iqr / col_std
            effective_mag = base_magnitude * mult * iqr_ratio
        else:
            effective_mag = base_magnitude * mult

        # Clamp to [0.01, 0.25] — 0.25 = noise std is 25% of column std
        effective_mag = max(0.01, min(0.25, effective_mag))
        per_var_mag[qi] = round(effective_mag, 4)
        iqr_info[qi] = iqr

        # Distribution risk: noise_std / col_std
        noise_std = effective_mag * col_std if col_std > 0 else 0
        ratio = noise_std / col_std if col_std > 0 else 0
        if ratio > 0.50:
            dist_risk[qi] = 'high'
            warnings.append(
                f"Noise on '{qi}' is {ratio:.0%} of its std — "
                f"distribution will be noticeably altered.")
        elif ratio > 0.30:
            dist_risk[qi] = 'moderate'
        else:
            dist_risk[qi] = 'low'

    # ── Pairwise correlation preservation check ─────────────────
    corr_warnings: List[str] = []
    numeric_qis = [q for q in quasi_identifiers
                   if q in per_var_mag and q in data.columns]

    if len(numeric_qis) >= 2:
        try:
            corr_matrix = data[numeric_qis].corr()
            for i, qa in enumerate(numeric_qis):
                for qb in numeric_qis[i + 1:]:
                    orig_r = abs(float(corr_matrix.loc[qa, qb]))
                    if orig_r < 0.30:
                        continue  # weak — not worth preserving

                    var_a = float(data[qa].var())
                    var_b = float(data[qb].var())
                    col_std_a = float(data[qa].std()) if var_a > 0 else 1
                    col_std_b = float(data[qb].std()) if var_b > 0 else 1
                    noise_var_a = (per_var_mag[qa] * col_std_a) ** 2
                    noise_var_b = (per_var_mag[qb] * col_std_b) ** 2

                    if var_a > 0 and var_b > 0:
                        atten = ((var_a / (var_a + noise_var_a)) ** 0.5 *
                                 (var_b / (var_b + noise_var_b)) ** 0.5)
                        est_r = orig_r * atten

                        if est_r < 0.50 and orig_r > 0.70:
                            corr_warnings.append(
                                f"Noise will likely break '{qa}' ↔ '{qb}' "
                                f"correlation ({orig_r:.2f} → {est_r:.2f}). "
                                f"Consider reducing noise on the lighter QI.")
        except Exception:
            pass

    if corr_warnings:
        warnings.extend(corr_warnings)

    log.info("[SmartConfig/NOISE] per_var_mag=%s  dist_risk=%s  corr_warnings=%d",
             per_var_mag, dist_risk, len(corr_warnings))

    return {
        'per_variable_magnitude': per_var_mag,
        'distribution_risk': dist_risk,
        'correlation_warnings': corr_warnings,
        'warnings': warnings,
    }


# ── 5.  Unified entry point for protection engine ──────────────────────


def get_smart_config(
    method: str,
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    target_k: int = 5,
    base_magnitude: float = 0.10,
    base_p_change: float = 0.20,
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Dispatch to the appropriate suggest_* function.

    Returns a unified dict (always has ``warnings`` and ``switch_method``).
    """
    method_upper = method.upper()
    if method_upper == 'KANON':
        return suggest_kanon_config(data, quasi_identifiers, target_k, qi_treatment)
    elif method_upper == 'LOCSUPR':
        return suggest_locsupr_config(data, quasi_identifiers, target_k, qi_treatment)
    elif method_upper == 'PRAM':
        cfg = suggest_pram_config(data, quasi_identifiers, base_p_change, qi_treatment)
        cfg.setdefault('switch_method', None)
        return cfg
    elif method_upper == 'NOISE':
        cfg = suggest_noise_config(data, quasi_identifiers, base_magnitude, qi_treatment)
        cfg.setdefault('switch_method', None)
        return cfg
    else:
        return {'warnings': [], 'switch_method': None}
