"""
Smart Defaults for SDC Workflow

This module provides intelligent default parameter selection based on data characteristics.
Analyzes dataset complexity, number of QIs, cardinality, and initial risk to recommend
appropriate preprocessing and protection method settings.

Key adaptive strategies:
    - More QIs (>7) -> More aggressive preprocessing, prefer PRAM/NOISE over k-ANON
    - High cardinality -> Lower max_categories
    - Large dataset (>10K) -> Can use lower k
    - Very high risk (ReID>90%) -> Very aggressive preprocessing
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sdc_engine.sdc.GENERALIZE import apply_generalize

log = logging.getLogger(__name__)
from sdc_engine.sdc.PRAM import apply_pram
from sdc_engine.sdc.kANON import apply_kanon
from sdc_engine.sdc.sdc_utils import calculate_reid
from sdc_engine.sdc.detection import auto_detect_direct_identifiers
from sdc_engine.sdc.utility import compute_utility
from sdc_engine.sdc.config import GENERALIZE_TIERS, METRIC_ALLOWED_METHODS
from sdc_engine.sdc.metrics.risk_metric import (
    RiskMetricType, compute_risk, normalize_to_risk_score,
)


def _classify_qi_type(
    col: str,
    data: pd.DataFrame,
    type_label: str = '',
) -> Dict[str, bool]:
    """Classify a QI column's semantic type using label + fallback heuristics.

    Shared helper used by both ``build_type_aware_preprocessing()`` and
    ``_detect_data_characteristics()`` to avoid divergent type detection logic.

    Parameters
    ----------
    col : str
        Column name.
    data : pd.DataFrame
        Dataset (needs the column for dtype fallback).
    type_label : str
        Pre-computed semantic type label (e.g. ``'Integer — Age (demographic)'``).

    Returns
    -------
    dict
        Boolean flags: ``is_date``, ``is_geo``, ``is_numeric``, ``is_age``,
        ``is_binary``, ``is_id_like``, ``is_free_text``, ``is_coded``,
        ``is_high_card``, ``is_ordinal``.
    """
    tl = type_label.lower() if type_label else ''
    col_lower = col.lower()

    # Primary: semantic label
    is_date = ('date' in tl or 'temporal' in tl)
    is_geo = ('geographic' in tl or 'zipcode' in tl or 'region' in tl
              or 'city' in tl or 'municipality' in tl)
    is_numeric = ('integer' in tl or 'float' in tl or 'continuous' in tl
                  or 'char (numeric)' in tl or tl == 'numeric')
    is_binary = ('binary' in tl)
    is_id_like = ('identifier' in tl or 'id-like' in tl or 'direct id' in tl)
    is_free_text = ('free text' in tl)
    is_coded = ('coded' in tl)
    is_high_card = ('high-card' in tl or 'high cardinality' in tl)
    is_ordinal = ('ordinal' in tl)

    _age_patterns = ['age', 'ηλικ', 'ηλικία']
    is_age = any(p in col_lower for p in _age_patterns) or 'age' in tl

    # Fallback: data-driven when no semantic label (or label doesn't
    # clarify numeric status — e.g. plain "numeric" or "categorical")
    if not tl or (not is_numeric and col in data.columns
                  and pd.api.types.is_numeric_dtype(data[col])
                  and 'categorical' not in tl):
        is_numeric = pd.api.types.is_numeric_dtype(data[col]) if col in data.columns else is_numeric
    if not tl:
        if col in data.columns:
            is_date = pd.api.types.is_datetime64_any_dtype(data[col])
        _date_pats = ['date', 'ημερομ', 'ημερ', 'time', 'year', 'month',
                      'έτος', 'ετος', 'χρον']
        _geo_pats = ['zip', 'postal', 'δήμος', 'δημος', 'δημοτ', 'κοινοτ',
                     'νομός', 'νομαρχ', 'νομαρχία', 'νομαρχια',
                     'περιφέρ', 'περιφερ',
                     'region', 'city', 'municipality', 'district',
                     'county', 'town', 'province']
        if not is_date:
            is_date = any(p in col_lower for p in _date_pats)
        is_geo = any(p in col_lower for p in _geo_pats)

    return {
        'is_date': is_date, 'is_geo': is_geo, 'is_numeric': is_numeric,
        'is_age': is_age, 'is_binary': is_binary, 'is_id_like': is_id_like,
        'is_free_text': is_free_text, 'is_coded': is_coded,
        'is_high_card': is_high_card, 'is_ordinal': is_ordinal,
    }


def _measure_risk(
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    risk_metric: Optional[str] = None,
    risk_target_raw: Optional[float] = None,
) -> float:
    """Return normalized 0-1 risk score using the chosen metric.

    Falls back to REID95 on error or when no metric specified.
    """
    _mt_map = {
        'reid95': RiskMetricType.REID95,
        'k_anonymity': RiskMetricType.K_ANONYMITY,
        'uniqueness': RiskMetricType.UNIQUENESS,
    }
    mt = _mt_map.get(risk_metric or 'reid95', RiskMetricType.REID95)
    try:
        assessment = compute_risk(data, quasi_identifiers, mt, risk_target_raw)
        return assessment.normalized_score
    except (ValueError, TypeError, KeyError) as exc:
        log.warning("[SmartDefaults] Risk computation failed, falling back to REID95: %s", exc)
        return calculate_reid(data, quasi_identifiers).get('reid_95', 1.0)


def calculate_smart_defaults(
    data: pd.DataFrame,
    detected_qis: List[str],
    initial_reid_95: float,
    structural_risk: float = 0.0,
    risk_metric: Optional[str] = None,
    qi_treatment: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Calculate smart default parameters based on data characteristics.

    Args:
        data: Input DataFrame
        detected_qis: List of detected quasi-identifier columns
        initial_reid_95: Normalized risk score (0-1 scale, higher=riskier).
            For REID95 this is the raw value; for k-anonymity it's 1/k;
            for uniqueness it's the uniqueness rate.
        structural_risk: QI-scoped backward elimination risk (0-1).
            When provided, tightens max_categories for high structural risk.
        qi_treatment: Per-QI treatment levels from Configure table
            (e.g. ``{'age': 'Heavy', 'region': 'Light'}``).
            QIs with Heavy treatment are never recommended for dropping.

    Returns:
        Dictionary with keys:
            - preprocess_params: Dict of preprocessing parameters
            - method: Recommended protection method ('kANON' or 'PRAM')
            - method_params: Dict of method-specific parameters
            - reasoning: List of explanation strings
    """
    n_records = len(data)
    n_qis = len(detected_qis)
    log.info("[SmartDefaults] n_records=%d  n_qis=%d  initial_risk=%.4f  "
             "structural_risk=%.2f  metric=%s",
             n_records, n_qis, initial_reid_95, structural_risk, risk_metric or 'reid95')

    # Calculate cardinalities
    cardinalities = []
    for qi in detected_qis:
        if qi in data.columns:
            cardinalities.append(data[qi].nunique())
    
    avg_cardinality = np.mean(cardinalities) if cardinalities else 0
    max_cardinality = max(cardinalities) if cardinalities else 0
    
    # Complexity score: combines number of QIs with their cardinality
    # Higher score = more complex, needs more aggressive treatment
    complexity = n_qis * np.log10(avg_cardinality + 1)
    
    defaults = {
        'preprocess_params': {},
        'method': '',
        'method_params': {},
        # Target portfolio for acceptable risk (default 5% ReID95)
        'target': {'reid_95': 0.05},
        'reasoning': [],
        'complexity_score': complexity,
        'drop_recommendations': None
    }
    
    # ============ QI REDUCTION RECOMMENDATIONS ============

    # Build set of user-protected QIs (Heavy treatment = user explicitly
    # flagged as important — never recommend dropping these).
    _treatment = qi_treatment or {}
    _protected_qis = {qi for qi, level in _treatment.items()
                      if str(level).lower() == 'heavy'}

    # If too many QIs, recommend which to drop
    if n_qis > 8:
        # Calculate importance score for each QI
        qi_importance = []
        for qi in detected_qis:
            if qi in data.columns:
                cardinality = data[qi].nunique()
                # Lower cardinality = more important (easier to work with)
                # But also consider if it's a standard demographic (age, gender)
                importance_score = 1.0 / (cardinality + 1)  # Inverse of cardinality

                # Boost importance for common demographic QIs
                qi_lower = qi.lower()
                if any(demo in qi_lower for demo in ['age', 'gender', 'sex', 'location', 'region', 'education']):
                    importance_score *= 2.0

                # Never recommend dropping QIs the user set to Heavy treatment
                is_protected = qi in _protected_qis
                if is_protected:
                    importance_score = float('inf')  # Always sort to end (keep)

                qi_importance.append({
                    'qi': qi,
                    'cardinality': cardinality,
                    'importance_score': importance_score,
                    'protected': is_protected,
                })

        # Sort by importance (ascending - least important first)
        qi_importance.sort(key=lambda x: x['importance_score'])

        # Only recommend dropping unprotected QIs
        droppable = [qi for qi in qi_importance if not qi.get('protected')]
        n_to_drop = n_qis - 7
        recommended_drops = [qi['qi'] for qi in droppable[:n_to_drop]]

        defaults['drop_recommendations'] = {
            'n_qis': n_qis,
            'recommended_to_drop': recommended_drops,
            'reason': f"Too many QIs ({n_qis}) detected. Recommend dropping {len(recommended_drops)} least important QI(s) to reach 7 QIs.",
            'qi_importance_scores': qi_importance,
            'protected_qis': list(_protected_qis),
        }
        if _protected_qis:
            defaults['reasoning'].append(
                f"Too many QIs (n={n_qis}) -> Recommend dropping {len(recommended_drops)} QI(s) "
                f"(excluding {len(_protected_qis)} Heavy-treatment QIs)")
        else:
            defaults['reasoning'].append(f"Too many QIs (n={n_qis}) -> Recommend dropping {len(recommended_drops)} QI(s)")
    elif n_qis > 7:
        defaults['reasoning'].append(f"Many QIs (n={n_qis}) -> Consider dropping 1-2 QIs if anonymization fails")
    
    # ============ PREPROCESSING PARAMETERS ============

    # 1. Max categories for categorical generalization
    # Data-aware formula: compute the max categories per QI that allow k=5 anonymity
    # max_groups = n_records / k, then max_cats = max_groups^(1/n_qis)
    # This ensures total combinations (max_cats^n_qis) <= n_records/k
    max_groups = n_records // 5  # For k=5
    if max_groups > 0 and n_qis > 0:
        data_aware_cats = max(2, int(max_groups ** (1 / n_qis)))
        data_aware_cats = min(data_aware_cats, 10)  # Cap at 10
    else:
        data_aware_cats = 3

    # Use the data-aware value, but apply additional constraints for edge cases.
    # Structural Risk (QI-scoped) tightens limits when QI structure is
    # inherently dangerous (backward elimination finds high re-id potential).
    high_struct = structural_risk > 0.50
    if n_qis > 8:
        max_categories = min(data_aware_cats, 3)
        defaults['reasoning'].append(f"Very many QIs (n={n_qis}), data-aware={data_aware_cats} -> max_categories={max_categories}")
    elif complexity > 15:
        max_categories = min(data_aware_cats, 3)
        defaults['reasoning'].append(f"Very complex data (complexity={complexity:.1f}), data-aware={data_aware_cats} -> max_categories={max_categories}")
    elif complexity > 10 or initial_reid_95 > 0.90 or high_struct:
        max_categories = min(data_aware_cats, 4)
        risk_info = f"complexity={complexity:.1f}, ReID={initial_reid_95:.0%}"
        if high_struct:
            risk_info += f", structural={structural_risk:.0%}"
        defaults['reasoning'].append(f"High complexity/risk ({risk_info}), data-aware={data_aware_cats} -> max_categories={max_categories}")
    else:
        max_categories = data_aware_cats
        defaults['reasoning'].append(f"Standard complexity, data-aware -> max_categories={max_categories}")

    defaults['preprocess_params']['max_categories'] = max_categories
    
    # 2. Bin size for numeric generalization (if numeric columns exist)
    numeric_cols = data[detected_qis].select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        max_numeric_card = max([data[col].nunique() for col in numeric_cols])
        if max_numeric_card > 50:
            bin_size = 10
            if max_numeric_card > 100:
                defaults['reasoning'].append(f"High numeric cardinality ({max_numeric_card}) -> bin_size=10")
        else:
            bin_size = 5
            if max_numeric_card > 20:
                defaults['reasoning'].append(f"Medium numeric cardinality ({max_numeric_card}) -> bin_size=5")
        
        defaults['preprocess_params']['bin_size'] = bin_size
    
    # 3. Strategy: 'all' forces generalization on all QIs, 'auto' only high-cardinality
    if n_qis > 7:  # Many QIs - need to reduce all
        strategy = 'all'
        defaults['reasoning'].append(f"Many QIs (n={n_qis}) -> strategy='all' (force all)")
    else:
        strategy = 'auto'
        defaults['reasoning'].append(f"Few QIs (n={n_qis}) -> strategy='auto' (high-card only)")
    
    defaults['preprocess_params']['strategy'] = strategy

    log.info("[SmartDefaults] Preprocess: max_categories=%d  strategy=%s  "
             "complexity=%.1f  avg_card=%.0f  max_card=%d",
             max_categories, strategy, complexity, avg_cardinality, max_cardinality)

    # ============ DATA CHARACTERISTIC WARNINGS ============
    
    # Integrate existing detection functions to provide actionable warnings
    defaults['data_warnings'] = _detect_data_characteristics(data, detected_qis)
    
    # ============ METHOD SELECTION (preprocessing quick-test) ============
    #
    # NOTE: This is a SIMPLIFIED method picker used for the preprocessing tier
    # loop's quick protection test — it checks whether the target ReID is
    # achievable after each preprocessing tier.  It is NOT the final method
    # selection.  The full rules engine in select_method.py (RC1-RC3, CAT1-CAT2,
    # QR0-QR4, etc.) is used for the actual Protection phase and considers risk
    # patterns, data type composition, structural risk, and treatment levels.
    #
    # Keeping this simple is intentional: the tier loop runs 2-4 times and needs
    # a fast, reasonable default — not the optimal method.  The rules engine
    # picks the final method after preprocessing is complete.
    #
    # Decision tree:
    # - Many QIs (>7): PRAM (avoids massive suppression from k-ANON)
    # - Medium QIs (4-7): k-ANON with lower k
    # - Few QIs (<4): Standard k-ANON
    
    if n_qis > 7:
        # Many QIs -> PRAM (works well with many categorical after preprocessing)
        defaults['method'] = 'PRAM'
        defaults['method_params'] = {
            'p_change': 0.15,  # 15% probability to change each value
            'pd_min': 0.60,    # Minimum diagonal probability
            'verbose': False
        }
        defaults['reasoning'].append(f"Many QIs (n={n_qis}) -> PRAM (handles many categoricals)")
    elif n_qis > 4:
        # Medium QIs -> k-ANONYMITY with lower k
        defaults['method'] = 'kANON'
        k_value = 3 if n_records > 10000 else 5
        defaults['method_params'] = {
            'k': k_value,
            'max_suppression_rate': 0.15 if n_records > 10000 else 0.10,
            'verbose': False
        }
        defaults['reasoning'].append(f"Medium QIs (n={n_qis}) -> k-ANON (k={k_value})")
    else:
        # Few QIs -> Standard k-ANONYMITY
        defaults['method'] = 'kANON'
        k_value = 3 if n_records > 10000 else 5
        defaults['method_params'] = {
            'k': k_value,
            'max_suppression_rate': 0.10,
            'verbose': False
        }
        defaults['reasoning'].append(f"Few QIs (n={n_qis}) -> k-ANON (k={k_value})")

    # --- Metric-method gate ---
    # If the selected method is blocked for the active metric, fall back to
    # kANON (universally allowed).  Without this check, the Smart Combo path
    # can produce PRAM output under k_anonymity/uniqueness metrics — a silent
    # violation of the METRIC_ALLOWED_METHODS contract.
    _metric = risk_metric or 'reid95'
    _allowed = METRIC_ALLOWED_METHODS.get(_metric, METRIC_ALLOWED_METHODS['reid95'])
    if defaults['method'] not in _allowed:
        blocked_method = defaults['method']
        defaults['method'] = 'kANON'
        k_value = 3 if n_records > 10000 else 5
        defaults['method_params'] = {
            'k': k_value,
            'max_suppression_rate': 0.15 if n_records > 10000 else 0.10,
            'verbose': False
        }
        defaults['reasoning'].append(
            f"{blocked_method} blocked for metric '{_metric}' -> kANON (k={k_value})")
        log.info("[SmartDefaults] %s blocked for metric '%s', falling back to kANON",
                 blocked_method, _metric)

    log.info("[SmartDefaults] Method: %s  params=%s",
             defaults['method'], defaults['method_params'])
    return defaults


def apply_smart_workflow(
    data: pd.DataFrame,
    detected_qis: List[str],
    initial_reid_95: float,
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply complete SDC workflow with smart defaults.
    
    Args:
        data: Input DataFrame
        detected_qis: List of detected quasi-identifier columns
        initial_reid_95: Initial ReID95 value (0-1 scale)
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (protected_data, results_dict)
        - protected_data: Protected DataFrame
        - results_dict: Dictionary with keys:
            - smart_defaults: The calculated defaults
            - preprocess_result: Preprocessing result
            - protect_result: Protection method result
            - final_reid_95: Final risk level
    """
    # Calculate smart defaults
    smart_defaults = calculate_smart_defaults(data, detected_qis, initial_reid_95)
    
    if verbose:
        print("\n" + "="*70)
        print("SMART DEFAULTS CALCULATED")
        print("="*70)
        print("\nReasoning:")
        for reason in smart_defaults['reasoning']:
            print(f"  X {reason}")
        print(f"\nSelected method: {smart_defaults['method']}")
        print(f"Method params: {smart_defaults['method_params']}")
        print(f"Preprocess params: {smart_defaults['preprocess_params']}")
    
    # Step 1: Preprocessing
    if verbose:
        print("\n" + "="*70)
        print("STEP 1: PREPROCESSING (GENERALIZE)")
        print("="*70)
    
    # Map parameter names to what GENERALIZE expects
    gen_params = {
        'quasi_identifiers': detected_qis,
        'max_categories': smart_defaults['preprocess_params']['max_categories'],
        'strategy': smart_defaults['preprocess_params']['strategy'],
        'return_metadata': True,
        'verbose': False
    }
    if 'bin_size' in smart_defaults['preprocess_params']:
        gen_params['numeric_bin_size'] = smart_defaults['preprocess_params']['bin_size']
    
    preprocessed_data, preprocess_result = apply_generalize(data=data, **gen_params)
    
    if verbose and 'metadata' in preprocess_result:
        meta = preprocess_result['metadata']
        print(f"\nPreprocessing applied: {meta.get('method', 'GENERALIZE')}")
        if 'transformations' in meta:
            print(f"Columns transformed: {len(meta['transformations'])}")
            for col, trans in meta['transformations'].items():
                if isinstance(trans, dict):
                    before = trans.get('unique_before', '?')
                    after = trans.get('unique_after', '?')
                    print(f"  - {col}: {before} -> {after} unique values")
    
    # Calculate ReID after preprocessing
    reid_result = calculate_reid(preprocessed_data, detected_qis)
    reid_after_preprocess = reid_result['reid_95']
    
    if verbose:
        print(f"\nRisk after preprocessing: ReID95 = {reid_after_preprocess*100:.1f}%")
    
    # Step 2: Protection method
    if verbose:
        print("\n" + "="*70)
        print(f"STEP 2: PROTECTION ({smart_defaults['method']})")
        print("="*70)
    
    if smart_defaults['method'] == 'PRAM':
        pram_params = {
            'variables': detected_qis,
            'return_metadata': True,
            **smart_defaults['method_params']
        }
        protected_data, protect_result = apply_pram(data=preprocessed_data, **pram_params)
    else:  # k-ANON
        kanon_params = {
            'quasi_identifiers': detected_qis,
            'return_metadata': True,
            **smart_defaults['method_params']
        }
        protected_data, protect_result = apply_kanon(data=preprocessed_data, **kanon_params)
    
    if verbose and 'metadata' in protect_result:
        meta = protect_result['metadata']
        if 'statistics' in meta:
            stats = meta['statistics']
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                elif isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  - {k}: {v}")
                else:
                    print(f"{key}: {value}")
    
    # Calculate final ReID
    final_reid_result = calculate_reid(protected_data, detected_qis)
    final_reid_95 = final_reid_result['reid_95']
    
    if verbose:
        print(f"\nFinal risk: ReID95 = {final_reid_95*100:.1f}%")
        print(f"Risk reduction: {initial_reid_95*100:.1f}% -> {final_reid_95*100:.1f}%")
    
    results = {
        'smart_defaults': smart_defaults,
        'preprocess_result': preprocess_result,
        'protect_result': protect_result,
        'reid_after_preprocess': reid_after_preprocess,
        'final_reid_95': final_reid_95
    }
    # Backwards compatibility keys expected by older tests/UI
    results['risk_initial'] = initial_reid_95
    results['risk_after_prep'] = reid_after_preprocess
    results['risk_final'] = final_reid_95
    results['total_reduction_pct'] = (initial_reid_95 - final_reid_95) * 100
    results['settings_used'] = {
        'preprocessing': smart_defaults.get('preprocess_params', {}),
        'protection': {**smart_defaults.get('method_params', {}), 'method': smart_defaults.get('method')}
    }
    
    # Determine whether target was met
    target_reid = smart_defaults.get('target', {}).get('reid_95', 0.05)
    results['target_met'] = final_reid_95 <= target_reid

    return protected_data, results


def apply_smart_workflow_with_adaptive_retry(
    data: pd.DataFrame,
    detected_qis: List[str],
    initial_reid_95: float,
    target_reid: float = 0.05,
    max_attempts: int = 4,
    start_tier: str = 'light',
    min_utility: float = 0.50,
    var_priority: Optional[Dict[str, tuple]] = None,
    sensitive_columns: Optional[List[str]] = None,
    structural_risk: float = 0.0,
    qi_treatment: Optional[Dict[str, str]] = None,
    verbose: bool = False,
    risk_metric: Optional[str] = None,
    risk_target_raw: Optional[float] = None,
    column_types: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply SDC workflow with adaptive retry - progressively more aggressive preprocessing
    until target ReID is met.

    This function:
    1. Starts with recommended tier (from feasibility check or user preference)
    2. If target not met, escalates to next more aggressive tier
    3. Always works from original data (never preprocess preprocessed data)
    4. Returns best result with metadata about which tier succeeded

    Design note — no cross-method fallback:
    Each tier uses the same protection method (from calculate_smart_defaults()).
    This is intentional: the preprocessing retry loop varies PREPROCESSING
    intensity (light → aggressive GENERALIZE) while keeping the protection
    method constant.  Cross-method fallback (kANON → LOCSUPR → PRAM) is
    handled by run_rules_engine_protection() in the Protection phase, which
    runs AFTER preprocessing is complete.  The two loops are complementary:
    - This function: "can we preprocess enough to make protection feasible?"
    - Rules engine: "which protection method works best on this preprocessed data?"

    Args:
        data: Input DataFrame (MUST be original, unmodified data)
        detected_qis: List of detected quasi-identifier columns
        initial_reid_95: Initial ReID95 value (0-1 scale)
        target_reid: Target ReID95 to achieve (default 0.05 = 5%)
        max_attempts: Maximum number of tiers to try
        start_tier: Starting tier ('light', 'moderate', 'aggressive', 'very_aggressive')
        min_utility: Minimum utility threshold — stop escalating if utility
            drops below this value (default 0.50). Derived from protection
            context's info_loss_max as 1 - info_loss_max when called from UI.
            Values below 0.50 trigger a warning — data with <50% utility
            is typically unusable for analysis.
        structural_risk: QI-scoped backward elimination risk (0-1).
            When high, overrides start_tier to skip hopeless light tiers.
        verbose: Whether to print progress information

    Returns:
        Tuple of (protected_data, results_dict)
        - protected_data: Best protected DataFrame
        - results_dict: Dictionary with keys:
            - attempts: List of attempt details
            - final_attempt: Which attempt succeeded
            - final_tier: Which tier was used
            - final_reid_95: Final risk level
            - success: Whether target was met
            - all workflow results from successful attempt
    """
    # KEEP ORIGINAL DATA - critical for retry!
    original_data = data.copy()

    # Warn if utility floor is dangerously low (caller should set via context)
    if min_utility < 0.50:
        log.warning("[AdaptiveRetry] min_utility=%.0f%% is very low — results may "
                    "be practically unusable. Consider raising to ≥50%%.",
                    min_utility * 100)

    log.info("[AdaptiveRetry] START: initial_ReID=%.4f  target=%.4f  "
             "min_utility=%.0f%%  start_tier=%s  structural_risk=%.2f  "
             "n_qis=%d  n_records=%d",
             initial_reid_95, target_reid, min_utility * 100, start_tier,
             structural_risk, len(detected_qis), len(data))

    # Override start_tier when Structural Risk indicates hopeless light tiers
    if structural_risk > 0.50 and start_tier in ('light', 'moderate'):
        log.info("[AdaptiveRetry] Structural risk %.0f%% -> overriding start_tier to 'aggressive'",
                 structural_risk * 100)
        start_tier = 'aggressive'
    elif structural_risk > 0.30 and start_tier == 'light':
        log.info("[AdaptiveRetry] Structural risk %.0f%% -> overriding start_tier to 'moderate'",
                 structural_risk * 100)
        start_tier = 'moderate'

    # Start from specified tier (uses shared GENERALIZE_TIERS from config)
    start_idx = next((i for i, t in enumerate(GENERALIZE_TIERS) if t['name'] == start_tier), 1)
    tiers_to_try = GENERALIZE_TIERS[start_idx:start_idx + max_attempts]
    log.info("[AdaptiveRetry] Tiers to try: %s",
             [t['name'] for t in tiers_to_try])

    if verbose:
        print("\n" + "="*70)
        print("ADAPTIVE RETRY WORKFLOW")
        print("="*70)
        print(f"Initial ReID: {initial_reid_95*100:.1f}%")
        print(f"Target ReID: {target_reid*100:.1f}%")
        print(f"Min utility: {min_utility:.0%}")
        print(f"Starting tier: {start_tier.title()}")
        print(f"Will try up to {len(tiers_to_try)} tiers")
        print("="*70)

    # --- Type-aware preprocessing (one-time pass) ---
    # Apply type-specific functions (date truncation, age binning, geo
    # coarsening, etc.) on original_data so each tier retry starts from
    # a type-preprocessed base instead of raw data.
    _data_warnings = _detect_data_characteristics(
        original_data, detected_qis, column_types=column_types)
    _type_plan = build_type_aware_preprocessing(
        original_data, detected_qis, _data_warnings,
        column_types=column_types)
    type_preprocess_meta = {}
    _reid_after_type_preprocess = initial_reid_95  # default: unchanged
    if _type_plan:
        original_data, type_preprocess_meta = apply_type_aware_preprocessing(
            original_data, _type_plan)
        # Re-measure risk on type-preprocessed data so calculate_smart_defaults
        # in the tier loop uses post-type-preprocess risk, not stale original.
        _reid_after_type_preprocess = _measure_risk(
            original_data, detected_qis, risk_metric, risk_target_raw)
        log.info("[AdaptiveRetry] ReID after type-preprocess: %.4f (was %.4f)",
                 _reid_after_type_preprocess, initial_reid_95)
        if verbose:
            applied = [f"{c}: {m['action']}" for c, m in type_preprocess_meta.items()
                       if m.get('success')]
            if applied:
                print(f"\n🔧 Type-aware preprocessing: {', '.join(applied)}")

    attempts = []
    best_result = None
    best_data = None

    for attempt_num, tier in enumerate(tiers_to_try, 1):
        if verbose:
            print(f"\n🔄 Attempt {attempt_num}/{len(tiers_to_try)}: {tier['label']} (max_categories={tier['max_categories']})")

        # ALWAYS START FROM ORIGINAL DATA
        current_data = original_data.copy()

        # Override smart defaults with current tier settings
        # Use post-type-preprocess risk (not stale initial_reid_95) so that
        # complexity scores and method selection reflect actual cardinality.
        smart_defaults = calculate_smart_defaults(
            current_data, detected_qis, _reid_after_type_preprocess,
            qi_treatment=qi_treatment, risk_metric=risk_metric)
        smart_defaults['preprocess_params']['max_categories'] = tier['max_categories']
        smart_defaults['preprocess_params']['strategy'] = 'all'  # Force all columns

        # Apply preprocessing
        gen_params = {
            'quasi_identifiers': detected_qis,
            'max_categories': tier['max_categories'],
            'strategy': 'all',
            'return_metadata': True,
            'adaptive_binning': True,
            'verbose': False
        }
        if var_priority:
            from sdc_engine.sdc.GENERALIZE import compute_risk_weighted_limits
            gen_params['max_categories_per_qi'] = compute_risk_weighted_limits(
                var_priority, tier['max_categories'])
            gen_params['var_priority'] = var_priority
        gen_params['reid_target'] = target_reid
        if qi_treatment:
            gen_params['qi_treatment'] = qi_treatment
        if 'bin_size' in smart_defaults['preprocess_params']:
            gen_params['numeric_bin_size'] = smart_defaults['preprocess_params']['bin_size']
        # Wire per-QI utility gating: GENERALIZE will roll back individual QIs
        # whose binning drops per-variable utility below threshold.
        gen_params['utility_fn'] = lambda orig, proc: compute_utility(
            orig, proc, quasi_identifiers=detected_qis,
            sensitive_columns=sensitive_columns)
        gen_params['utility_threshold'] = max(min_utility, 0.60)
        if column_types:
            gen_params['column_types'] = column_types

        try:
            preprocessed_data, preprocess_result = apply_generalize(data=current_data, **gen_params)

            # --- Utility check: QI-scoped for speed ---
            # NOTE: This measures QI preservation only.  Sensitive columns are
            # not modified by GENERALIZE, so their per-variable utility would
            # always be ~1.0 here (meaningless signal).  The full composite
            # utility (sensitive × QI cross-tab) is computed in the final
            # protection phase by the rules engine.  For the preprocessing
            # tier loop, QI-scoped utility is the right gate: it tells us
            # how much analytical structure GENERALIZE destroyed.
            tier_utility = compute_utility(
                original_data, preprocessed_data,
                quasi_identifiers=detected_qis,
                sensitive_columns=sensitive_columns)

            # Check risk after preprocessing (uses chosen metric)
            reid_after_preprocess = _measure_risk(preprocessed_data, detected_qis, risk_metric, risk_target_raw)

            # Apply protection method
            if smart_defaults['method'] == 'PRAM':
                pram_params = {
                    'variables': detected_qis,
                    'return_metadata': True,
                    **smart_defaults['method_params']
                }
                protected_data, protect_result = apply_pram(data=preprocessed_data, **pram_params)
            else:  # k-ANON
                kanon_params = {
                    'quasi_identifiers': detected_qis,
                    'return_metadata': True,
                    **smart_defaults['method_params']
                }
                protected_data, protect_result = apply_kanon(data=preprocessed_data, **kanon_params)

            # Calculate final risk
            final_reid_95 = _measure_risk(protected_data, detected_qis, risk_metric, risk_target_raw)

            attempt_info = {
                'attempt': attempt_num,
                'tier': tier['name'],
                'tier_label': tier['label'],
                'max_categories': tier['max_categories'],
                'reid_after_preprocess': reid_after_preprocess,
                'final_reid_95': final_reid_95,
                'utility': round(tier_utility, 4),
                'method': smart_defaults['method'],
                'success': final_reid_95 <= target_reid
            }
            attempts.append(attempt_info)
            log.info("[AdaptiveRetry] Attempt %d/%d [%s] max_cat=%d: "
                     "ReID %.1f%% -> %.1f%% -> %.1f%%  utility=%.1f%%  %s",
                     attempt_num, len(tiers_to_try), tier['label'],
                     tier['max_categories'],
                     initial_reid_95 * 100, reid_after_preprocess * 100,
                     final_reid_95 * 100, tier_utility * 100,
                     "SUCCESS" if final_reid_95 <= target_reid else "NOT MET")

            if verbose:
                print(f"   ReID: {initial_reid_95*100:.1f}% → {reid_after_preprocess*100:.1f}% → {final_reid_95*100:.1f}%  |  Utility: {tier_utility:.1%}")
                if final_reid_95 <= target_reid:
                    print(f"   ✅ SUCCESS! Target met ({final_reid_95*100:.1f}% ≤ {target_reid*100:.1f}%)")
                else:
                    print(f"   ❌ Target not met ({final_reid_95*100:.1f}% > {target_reid*100:.1f}%)")

            # Keep track of best result (prefer highest utility among successful,
            # or lowest ReID if none succeeded)
            is_better = False
            if best_result is None:
                is_better = True
            elif final_reid_95 <= target_reid and not best_result.get('_success'):
                is_better = True  # First successful result
            elif final_reid_95 <= target_reid and best_result.get('_success'):
                is_better = tier_utility > best_result.get('_utility', 0)  # Higher utility
            elif not best_result.get('_success'):
                is_better = final_reid_95 < best_result['final_reid_95']  # Lower ReID

            if is_better:
                best_result = {
                    'smart_defaults': smart_defaults,
                    'preprocess_result': preprocess_result,
                    'protect_result': protect_result,
                    'preprocessed_data': preprocessed_data.copy(),
                    'reid_after_preprocess': reid_after_preprocess,
                    'final_reid_95': final_reid_95,
                    'tier_used': tier['name'],
                    'tier_label': tier['label'],
                    '_utility': tier_utility,
                    '_success': final_reid_95 <= target_reid,
                }
                best_data = protected_data.copy()

            # If target met, stop trying
            if final_reid_95 <= target_reid:
                if verbose:
                    print(f"\n✅ Target achieved with {tier['label']} preprocessing!")
                break

            # Stop escalating if utility is already too low — further tiers
            # will only make it worse
            if tier_utility < min_utility:
                if verbose:
                    print(f"\n⚠️ Utility too low ({tier_utility:.1%} < {min_utility:.0%}), stopping escalation")
                attempts.append({
                    'attempt': attempt_num + 1,
                    'tier': 'stopped',
                    'tier_label': 'Stopped (utility floor)',
                    'reason': f'Utility {tier_utility:.1%} below minimum {min_utility:.0%}',
                    'success': False,
                    'utility_floor_hit': True,
                })
                break

        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")
            attempts.append({
                'attempt': attempt_num,
                'tier': tier['name'],
                'tier_label': tier['label'],
                'error': str(e),
                'success': False
            })
    
    # Prepare final results
    if best_result:
        best_result['attempts'] = attempts
        best_result['final_attempt'] = len([a for a in attempts if not a.get('error')])
        best_result['success'] = best_result['final_reid_95'] <= target_reid
        best_result['target_reid'] = target_reid
        best_result['initial_reid_95'] = initial_reid_95
        if type_preprocess_meta:
            best_result['type_preprocess'] = type_preprocess_meta
        
        if verbose:
            print("\n" + "="*70)
            if best_result['success']:
                print(f"✅ SUCCESS: Target achieved with {best_result['tier_label']} preprocessing")
            else:
                print(f"⚠️ WARNING: Target not fully met. Best result: {best_result['final_reid_95']*100:.1f}%")
            print(f"Risk reduction: {initial_reid_95*100:.1f}% → {best_result['final_reid_95']*100:.1f}%")
            print("="*70)
        
        log.info("[AdaptiveRetry] DONE: success=%s  tier=%s  "
                 "ReID %.1f%% -> %.1f%%  attempts=%d",
                 best_result['success'], best_result.get('tier_label', '?'),
                 initial_reid_95 * 100, best_result['final_reid_95'] * 100,
                 len(attempts))
        return best_data, best_result
    else:
        # All attempts failed
        log.warning("[AdaptiveRetry] ALL ATTEMPTS FAILED: %d attempts, initial_ReID=%.1f%%",
                    len(attempts), initial_reid_95 * 100)
        if verbose:
            print("\n❌ All attempts failed")
        return original_data, {
            'attempts': attempts,
            'success': False,
            'error': 'All preprocessing attempts failed',
            'initial_reid_95': initial_reid_95,
            'final_reid_95': initial_reid_95
        }


# =============================================================================
# Type-Aware Preprocessing (Phase 2)
# =============================================================================


def _is_human_age(series: pd.Series) -> bool:
    """Guard: check whether a numeric column looks like human age.

    Requires BOTH a name pattern (checked by caller) AND a plausible
    human-age distribution: median < 100, min >= 0, max <= 135.
    This prevents treating columns like "building_age" (median 150) or
    "account_age_days" (median 500) as demographic age.
    Upper bound of 135 accommodates geriatric/longevity datasets.
    """
    try:
        vals = pd.to_numeric(series.dropna(), errors='coerce').dropna()
        if vals.empty:
            return False
        return vals.median() < 100 and vals.min() >= 0 and vals.max() <= 135
    except (ValueError, TypeError) as exc:
        log.warning("[SmartDefaults] Age-like detection failed: %s", exc)
        return False


def _estimate_other_card_product(
    col: str,
    detected_qis: List[str],
    data: pd.DataFrame,
    column_types: Dict[str, str],
    plan: Dict[str, Dict[str, Any]],
) -> int:
    """Estimate the cardinality product of all QIs except *col* after
    their expected preprocessing, used to compute a cardinality budget
    for context-aware preprocessing decisions."""
    product = 1
    for oq in detected_qis:
        if oq == col or oq not in data.columns:
            continue
        oq_nu = data[oq].nunique()
        ot = _classify_qi_type(oq, data, column_types.get(oq, ''))
        if ot['is_date'] and oq_nu > 30:
            est = max(3, oq_nu // 90)   # ~years
        elif ot['is_age'] and ot['is_numeric']:
            _rng = data[oq].max() - data[oq].min()
            est = max(3, int(_rng / 5) + 1) if _rng > 0 else oq_nu
        elif oq_nu <= 20:
            est = oq_nu  # already small
        elif oq in plan:
            pa = plan[oq].get('params', {})
            est = pa.get('n_bins', pa.get('max_categories',
                                          min(oq_nu, 30)))
        elif ot['is_numeric'] and oq_nu > 20:
            # Will also hit quantile binning rule → estimate ~target bins
            n_rows = len(data)
            est = max(5, min(30, n_rows // 50))
        else:
            # High-card categoricals will be generalized by kANON
            # (prefix truncation / frequency grouping) to ~5-10 groups
            est = min(10, oq_nu)
        product *= max(1, est)
    return product


def build_type_aware_preprocessing(
    data: pd.DataFrame,
    detected_qis: List[str],
    data_warnings: List[Dict[str, Any]],
    column_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Map each QI to its optimal preprocessing function based on data type.

    Uses the semantic ``column_types`` labels from the Configure table
    together with data-driven checks to route each QI to one of the
    specialised preprocessing functions in ``sdc_preprocessing``.

    Parameters
    ----------
    data : pd.DataFrame
        Raw (unmodified) dataset.
    detected_qis : list[str]
        Quasi-identifier column names.
    data_warnings : list[dict]
        Output from ``_detect_data_characteristics()``.  Used to detect
        skewed columns (warning method ``'Bin Numerics + Top/Bottom Coding'``).
    column_types : dict[str, str], optional
        Pre-computed semantic type labels (e.g.
        ``{'age': 'Integer — Age (demographic)', ...}``).

    Returns
    -------
    dict[str, dict]
        ``{col: {action, function, params, reason}}``  where *function*
        is the callable to apply and *params* are the keyword arguments.
        Columns that should stay with the generic GENERALIZE step are
        omitted (they fall through to the tier loop).
    """
    if column_types is None:
        column_types = {}

    # Index warnings by column for fast lookup
    skewed_cols: set = set()
    for w in data_warnings:
        if 'Top/Bottom Coding' in w.get('method', ''):
            skewed_cols.update(w.get('columns', []))

    plan: Dict[str, Dict[str, Any]] = {}

    for col in detected_qis:
        if col not in data.columns:
            continue

        nunique = data[col].nunique()
        type_label = column_types.get(col, '')

        # Shared type classification (avoids duplication with _detect_data_characteristics)
        ct = _classify_qi_type(col, data, type_label)
        is_date = ct['is_date']
        is_geo = ct['is_geo']
        is_numeric = ct['is_numeric']
        is_age_name = ct['is_age']

        # ---- Route to specific function ----

        # 1. Date columns → truncate to year/quarter/month based on range
        # Three tiers: >365 unique → year, >100 → quarter, else → month
        # Guard: if chosen level would produce <3 periods, fall back
        if is_date and nunique > 30:
            if nunique > 365:
                truncate_to = 'year'
            elif nunique > 100:
                truncate_to = 'quarter'
            else:
                truncate_to = 'month'
            # Guard: ensure at least 3 distinct periods.
            # Cascade: year → quarter → month → week → day → keep
            try:
                _dates = pd.to_datetime(data[col], errors='coerce', dayfirst=True)
                _cascade = ['year', 'quarter', 'month', 'week', 'day']
                _start = _cascade.index(truncate_to) if truncate_to in _cascade else 0
                _final = truncate_to
                for _lvl in _cascade[_start:]:
                    if _lvl == 'year':
                        _n_periods = _dates.dt.year.nunique()
                    elif _lvl == 'quarter':
                        _n_periods = _dates.dt.to_period('Q').nunique()
                    elif _lvl == 'month':
                        _n_periods = _dates.dt.to_period('M').nunique()
                    elif _lvl == 'week':
                        _n_periods = _dates.dt.isocalendar().week.nunique()
                    else:  # day
                        _n_periods = _dates.dt.date.nunique()
                    if _n_periods >= 3:
                        _final = _lvl
                        break
                    _final = _lvl  # keep trying finer
                if _final != truncate_to:
                    log.info("[TypePreprocess] %s: fell back from %s to %s "
                             "(%d periods)", col, truncate_to, _final,
                             _n_periods)
                truncate_to = _final
                # If even day-level gives < 3 periods, skip truncation entirely
                if _n_periods < 3:
                    log.info("[TypePreprocess] %s: date truncation skipped "
                             "(only %d day-level periods)", col, _n_periods)
                    plan[col] = {
                        'action': 'keep',
                        'params': {},
                        'reason': f'Date with {nunique} unique values but '
                                  f'only {_n_periods} distinct days — '
                                  f'truncation would destroy information',
                    }
                    continue
            except (ValueError, TypeError, OverflowError) as exc:
                log.warning("[SmartDefaults] Date period check failed for '%s': %s", col, exc)
            plan[col] = {
                'action': 'date_truncation',
                'function': 'apply_date_truncation',
                'params': {'columns': [col], 'truncate_to': truncate_to,
                           'return_metadata': True},
                'reason': f'{nunique} unique dates → truncate to {truncate_to}',
            }
            log.info("[TypePreprocess] %s: date (%d unique) → truncate to %s",
                     col, nunique, truncate_to)
            continue

        # 2. Age-like numeric → bin into 5-year ranges
        if is_age_name and is_numeric and nunique > 10:
            if _is_human_age(data[col]):
                plan[col] = {
                    'action': 'age_binning',
                    'function': 'apply_age_binning',
                    'params': {'columns': [col], 'bin_size': 5,
                               'return_metadata': True},
                    'reason': f'{nunique} unique ages → 5-year bins',
                }
                log.info("[TypePreprocess] %s: age (%d unique) → 5-year bins",
                         col, nunique)
                continue

        # 3. Geographic → route by data type:
        #    Numeric postal codes → digit truncation (context-aware)
        #    Categorical names (ΜΑΡΟΥΣΙ, ΧΑΛΑΝΔΡΙ) → top-K generalization
        if is_geo and nunique > 20:
            col_is_numeric_geo = pd.api.types.is_numeric_dtype(data[col])
            if not col_is_numeric_geo:
                # Check if string values are digit-like (e.g. "12345" stored as str)
                sample = data[col].dropna().head(50).astype(str)
                col_is_numeric_geo = sample.str.match(r'^\d+$').mean() > 0.80

            # --- Cardinality budget: account for ALL other QIs ---
            n_rows = len(data)
            other_card_product = _estimate_other_card_product(
                col, detected_qis, data, column_types, plan)
            k_min = 5
            geo_budget = max(5, n_rows // (other_card_product * k_min))
            simple_target = max(20, n_rows // 5)
            target_groups = min(simple_target, geo_budget)
            log.info("[TypePreprocess] %s: geo budget=%d (other_card=%d, "
                     "rows=%d), simple_target=%d → target=%d",
                     col, geo_budget, other_card_product, n_rows,
                     simple_target, target_groups)

            if col_is_numeric_geo:
                avg_digits = int(data[col].dropna().astype(str).str.len().median())
                keep = min(3, avg_digits - 1)  # start at 3 or shorter
                for try_keep in range(keep, 0, -1):
                    test_col = data[col].dropna().astype(str).str[:try_keep]
                    if test_col.nunique() <= target_groups:
                        keep = try_keep
                        break
                else:
                    keep = 1  # last resort: single digit
                plan[col] = {
                    'action': 'geographic_coarsening',
                    'function': 'apply_geographic_coarsening',
                    'params': {'columns': [col], 'keep_digits': keep,
                               'return_metadata': True},
                    'reason': (f'{nunique} unique postal codes → keep {keep} '
                               f'leading digits (budget={target_groups} '
                               f'given {len(detected_qis)-1} other QIs)'),
                }
                log.info("[TypePreprocess] %s: numeric geo (%d unique) → "
                         "coarsen to %d digits (budget=%d)",
                         col, nunique, keep, target_groups)
            else:
                # Categorical geography (municipality names, city names, etc.)
                max_cat = min(target_groups, max(5, nunique // 3))
                plan[col] = {
                    'action': 'categorical_geo_generalize',
                    'function': 'apply_generalize',
                    'params': {'quasi_identifiers': [col],
                               'max_categories': max_cat,
                               'return_metadata': True},
                    'reason': (f'{nunique} unique place names → top-{max_cat} '
                               f'generalization (budget={target_groups} '
                               f'given {len(detected_qis)-1} other QIs)'),
                }
                log.info("[TypePreprocess] %s: categorical geo (%d unique names) → "
                         "top-%d generalize (budget=%d)",
                         col, nunique, max_cat, target_groups)
            continue

        # 4. Skewed numeric → top/bottom coding for moderate cardinality,
        #    but quantile binning for near-unique skewed columns (top/bottom
        #    coding alone barely reduces cardinality — income 2940→~2800).
        if col in skewed_cols and is_numeric:
            n_rows = len(data)
            if nunique > n_rows // 5:
                # Near-unique AND skewed → quantile binning is better
                # (falls through to step 5 below)
                pass
            else:
                plan[col] = {
                    'action': 'top_bottom_coding',
                    'function': 'apply_top_bottom_coding',
                    'params': {'columns': [col], 'method': 'percentile',
                               'top_percentile': 98, 'bottom_percentile': 2,
                               'return_metadata': True},
                    'reason': f'Skewed distribution (|skew|>2) → cap outliers at p2/p98',
                }
                log.info("[TypePreprocess] %s: skewed numeric → top/bottom coding", col)
                continue

        # 5. High-cardinality numeric without type hint → quantile binning
        #    for very-high-card (>n_rows/5), rounding for moderate-high-card.
        #    Rounding alone barely helps near-unique columns like income
        #    (57353→57000 still leaves ~100 unique for 1K rows).
        if is_numeric and nunique > 20:
            n_rows = len(data)
            # Context-aware target: account for other QIs' cardinalities
            # so the total combination space stays feasible for k=5.
            _other_card = _estimate_other_card_product(
                col, detected_qis, data, column_types, plan)
            _budget = max(5, n_rows // (_other_card * 5))
            target_bins = max(5, min(_budget, 30))
            log.info("[TypePreprocess] %s: numeric budget=%d (other_card=%d) "
                     "→ target_bins=%d", col, _budget, _other_card, target_bins)
            # Quantile binning is the universal best strategy for numeric
            # QIs with >20 unique values.  Rounding only helps integers
            # with very high cardinality where individual values are large,
            # but quantile binning produces more predictable output.
            # For small integers (e.g., monthly_transactions 8-44),
            # rounding doesn't reduce cardinality at all.
            plan[col] = {
                'action': 'quantile_binning',
                'function': 'apply_quantile_binning',
                'params': {'columns': [col], 'n_bins': target_bins,
                           'return_metadata': True},
                'reason': (f'{nunique} unique numeric values '
                           f'→ {target_bins} quantile bins'),
            }
            log.info("[TypePreprocess] %s: numeric (%d unique, "
                     "%d rows) → %d quantile bins",
                     col, nunique, n_rows, target_bins)
            continue

        # 6. High-cardinality categorical → top-K frequency grouping
        #    Keeps the most frequent categories, groups the rest into "Other".
        #    This is more uniform than kANON's prefix truncation (Α*, Κ*, Other)
        #    which produces very uneven groups.
        if not is_numeric and nunique > 20:
            n_rows = len(data)
            _other_card = _estimate_other_card_product(
                col, detected_qis, data, column_types, plan)
            _budget = max(5, n_rows // (_other_card * 5))
            max_cat = max(5, min(_budget, 30))
            plan[col] = {
                'action': 'categorical_generalize',
                'function': 'apply_generalize',
                'params': {'quasi_identifiers': [col],
                           'max_categories': max_cat,
                           'return_metadata': True},
                'reason': (f'{nunique} unique categories '
                           f'→ top-{max_cat} generalization'),
            }
            log.info("[TypePreprocess] %s: high-card categorical (%d unique) "
                     "→ top-%d generalize", col, nunique, max_cat)
            continue

        # 7. Default: no type-specific action → fall through to GENERALIZE
        log.debug("[TypePreprocess] %s: no type-specific action (nunique=%d, "
                  "type=%s) → GENERALIZE fallback",
                  col, nunique, type_label or 'unknown')

    if plan:
        log.info("[TypePreprocess] Plan: %d QIs get type-specific treatment: %s",
                 len(plan),
                 {c: p['action'] for c, p in plan.items()})
    else:
        log.info("[TypePreprocess] No type-specific treatment needed — "
                 "all QIs fall through to GENERALIZE")

    # Build and cache smart hierarchy objects for downstream consumption
    # (built on current data — if type-aware preprocessing modifies data,
    # hierarchies should be rebuilt on the preprocessed result)
    try:
        from sdc_engine.sdc.hierarchies import build_hierarchy_for_column
        hierarchies = {}
        for col in detected_qis:
            if col not in data.columns:
                continue
            h = build_hierarchy_for_column(col, data, column_types)
            if h is not None:
                hierarchies[col] = h
        if hierarchies:
            plan['_hierarchies'] = hierarchies
            log.info("[TypePreprocess] Built %d smart hierarchies: %s",
                     len(hierarchies),
                     {c: h.builder_type for c, h in hierarchies.items()})
    except Exception as _e:
        log.warning("[TypePreprocess] Hierarchy building failed: %s", _e)

    return plan


def apply_type_aware_preprocessing(
    data: pd.DataFrame,
    plan: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply the type-aware preprocessing plan returned by
    :func:`build_type_aware_preprocessing`.

    Applies each QI's specific function in sequence on a copy of *data*.
    Functions are resolved from ``sdc.sdc_preprocessing`` by name.

    Parameters
    ----------
    data : pd.DataFrame
        Raw data (will be copied internally — original not modified).
    plan : dict
        Output from ``build_type_aware_preprocessing()``.

    Returns
    -------
    (pd.DataFrame, dict)
        Preprocessed data and a metadata dict
        ``{col: {action, before_unique, after_unique, success}}``
        documenting what was applied.
    """
    from sdc_engine.sdc import sdc_preprocessing as _pp
    from sdc_engine.sdc import GENERALIZE as _gen

    result = data.copy()
    metadata: Dict[str, Any] = {}

    # Extract smart hierarchies built by build_type_aware_preprocessing().
    # These are Hierarchy objects keyed by column name.
    _hierarchies = plan.get('_hierarchies', {})

    for col, spec in plan.items():
        # Skip internal keys (e.g. _hierarchies) and non-dict entries
        if col.startswith('_') or not isinstance(spec, dict):
            continue
        fn_name = spec['function']
        params = spec['params']
        action = spec['action']

        # Inject smart hierarchies into apply_generalize calls so that
        # categorical geo columns (and others) use the pre-built Hierarchy
        # objects instead of blind top-K grouping.
        if fn_name == 'apply_generalize' and col in _hierarchies:
            params = {**params, 'hierarchies': {col: _hierarchies[col]}}

        # Resolve function from sdc_preprocessing first, then GENERALIZE
        fn = getattr(_pp, fn_name, None) or getattr(_gen, fn_name, None)
        if fn is None:
            log.warning("[TypePreprocess] Function %s not found — skipping %s",
                        fn_name, col)
            metadata[col] = {'action': action, 'success': False,
                             'error': f'{fn_name} not found'}
            continue

        before_unique = result[col].nunique()
        try:
            out = fn(data=result, **params)
            # All sdc_preprocessing functions return (df, meta) when
            # return_metadata=True
            if isinstance(out, tuple):
                result, step_meta = out
            else:
                result = out
                step_meta = {}

            after_unique = result[col].nunique()
            metadata[col] = {
                'action': action,
                'before_unique': before_unique,
                'after_unique': after_unique,
                'success': True,
                'step_meta': step_meta,
            }
            log.info("[TypePreprocess] Applied %s on %s: %d -> %d unique",
                     action, col.encode('ascii', 'replace').decode(), before_unique, after_unique)
        except Exception as exc:
            _safe_col = col.encode('ascii', 'replace').decode()
            _safe_err = str(exc).encode('ascii', 'replace').decode()
            log.warning("[TypePreprocess] %s failed on %s: %s",
                        action, _safe_col, _safe_err)
            metadata[col] = {'action': action, 'success': False,
                             'error': _safe_err}

    return result, metadata


# ---------------------------------------------------------------------------
# Editable plan helpers — roundtrip between plan dict and Tabulator DataFrame
# ---------------------------------------------------------------------------

# Action → (function_name, primary_param_key, param_is_string)
_ACTION_SPEC = {
    'date_truncation': ('apply_date_truncation', 'truncate_to', True),
    'age_binning': ('apply_age_binning', 'bin_size', False),
    'geographic_coarsening': ('apply_geographic_coarsening', 'keep_digits', False),
    'top_bottom_coding': ('apply_top_bottom_coding', 'percentile', False),
    'quantile_binning': ('apply_quantile_binning', 'n_bins', False),
    'categorical_generalize': ('apply_generalize', 'max_categories', False),
    'categorical_geo_generalize': ('apply_generalize', 'max_categories', False),
}

# Column-type → allowed action list (always includes 'keep')
_TYPE_ACTIONS = {
    'numeric': ['keep', 'quantile_binning', 'top_bottom_coding'],
    'age': ['keep', 'age_binning', 'quantile_binning', 'top_bottom_coding'],
    'date': ['keep', 'date_truncation'],
    'geo_numeric': ['keep', 'geographic_coarsening', 'quantile_binning'],
    'geo_categorical': ['keep', 'categorical_generalize', 'categorical_geo_generalize'],
    'categorical': ['keep', 'categorical_generalize'],
    'default': ['keep', 'quantile_binning', 'top_bottom_coding',
                'categorical_generalize'],
}


def _resolve_col_category(
    col: str,
    data: pd.DataFrame,
    column_types: Dict[str, str],
) -> str:
    """Map a column to one of the type-action categories."""
    type_label = column_types.get(col, '')
    ct = _classify_qi_type(col, data, type_label)

    if ct['is_date']:
        return 'date'
    if ct['is_age'] and ct['is_numeric']:
        return 'age'
    if ct['is_geo']:
        is_num_geo = pd.api.types.is_numeric_dtype(data[col])
        if not is_num_geo:
            sample = data[col].dropna().head(50).astype(str)
            is_num_geo = sample.str.match(r'^\d+$').mean() > 0.80
        return 'geo_numeric' if is_num_geo else 'geo_categorical'
    if ct['is_numeric']:
        return 'numeric'
    return 'categorical'


def _extract_primary_param(entry: Dict[str, Any]) -> str:
    """Extract the user-visible tunable param value as a string."""
    action = entry.get('action', 'keep')
    params = entry.get('params', {})

    if action == 'date_truncation':
        return str(params.get('truncate_to', 'year'))
    if action == 'age_binning':
        return str(params.get('bin_size', 5))
    if action == 'geographic_coarsening':
        return str(params.get('keep_digits', 3))
    if action == 'top_bottom_coding':
        return str(params.get('bottom_percentile', 2))
    if action == 'quantile_binning':
        return str(params.get('n_bins', 10))
    if action in ('categorical_generalize', 'categorical_geo_generalize'):
        return str(params.get('max_categories', 20))
    return '\u2014'  # em-dash for keep / unknown


def plan_to_editable_df(
    plan: Dict[str, Dict[str, Any]],
    data: pd.DataFrame,
    quasi_identifiers: List[str],
    column_types: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Convert a type-aware plan dict into an editable DataFrame.

    Returns
    -------
    (plan_df, allowed_actions)
        *plan_df* has columns: Column, Action, Param, Reason,
        Card. Before, Card. After, Tier.
        *allowed_actions* maps each column name to its valid action list.
    """
    if column_types is None:
        column_types = {}

    rows = []
    allowed_actions: Dict[str, List[str]] = {}

    for col in quasi_identifiers:
        if col not in data.columns:
            continue

        nunique = int(data[col].nunique())
        entry = plan.get(col)
        cat = _resolve_col_category(col, data, column_types)
        allowed = _TYPE_ACTIONS.get(cat, _TYPE_ACTIONS['default'])
        allowed_actions[col] = allowed

        if entry:
            action = entry['action']
            param = _extract_primary_param(entry)
            reason = entry.get('reason', '')
            tier = 'type-aware'
        else:
            action = 'keep'
            param = '\u2014'
            reason = f'Cardinality {nunique} — falls through to GENERALIZE'
            tier = 'generalize'

        rows.append({
            'Column': col,
            'Action': action,
            'Param': param,
            'Reason': reason,
            'Card. Before': nunique,
            'Card. After': nunique,  # placeholder until Recalculate
            'Tier': tier,
        })

    plan_df = pd.DataFrame(rows)
    return plan_df, allowed_actions


def edited_df_to_plan(
    df: pd.DataFrame,
    original_plan: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Convert an edited plan DataFrame back to the plan dict format.

    Returns
    -------
    (plan_dict, warning)
        *plan_dict* is the same format as ``build_type_aware_preprocessing()``
        output.  *warning* is a non-empty string if there is something the
        user should see (e.g. all-keep).
    """
    result: Dict[str, Dict[str, Any]] = {}
    warning = ''

    for _, row in df.iterrows():
        col = row['Column']
        action = row['Action']
        param_str = str(row['Param']).strip()

        if action == 'keep':
            continue

        orig = original_plan.get(col)

        # Check if unchanged from original
        if orig and orig['action'] == action:
            # Param might have been tweaked — rebuild with new param
            if param_str == _extract_primary_param(orig):
                result[col] = orig
                continue

        # Build new entry from action + param
        reason_suffix = ''
        if orig and orig['action'] != action:
            reason_suffix = f' (overridden from {orig["action"]})'
        elif orig:
            reason_suffix = ' (param adjusted)'

        spec = _ACTION_SPEC.get(action)
        if not spec:
            log.warning("[EditPlan] Unknown action %s for %s, skipping", action, col)
            continue

        func_name, param_key, param_is_str = spec
        # Parse param value
        if param_is_str:
            param_val = param_str
        else:
            try:
                param_val = int(param_str)
            except (ValueError, TypeError):
                log.warning("[EditPlan] Invalid param '%s' for %s/%s, "
                            "using original", param_str, col, action)
                if orig:
                    result[col] = orig
                continue

        # Build params dict
        if action == 'date_truncation':
            params = {'columns': [col], 'truncate_to': param_val,
                      'return_metadata': True}
            reason = f'Truncate to {param_val}{reason_suffix}'
        elif action == 'age_binning':
            params = {'columns': [col], 'bin_size': param_val,
                      'return_metadata': True}
            reason = f'{param_val}-year age bins{reason_suffix}'
        elif action == 'geographic_coarsening':
            params = {'columns': [col], 'keep_digits': param_val,
                      'return_metadata': True}
            reason = f'Keep {param_val} leading digits{reason_suffix}'
        elif action == 'top_bottom_coding':
            # Symmetric: param "2" → bottom_percentile=2, top_percentile=98
            params = {'columns': [col], 'method': 'percentile',
                      'top_percentile': 100 - param_val,
                      'bottom_percentile': param_val,
                      'return_metadata': True}
            reason = f'Cap outliers at p{param_val}/p{100 - param_val}{reason_suffix}'
        elif action == 'quantile_binning':
            params = {'columns': [col], 'n_bins': param_val,
                      'return_metadata': True}
            reason = f'{param_val} quantile bins{reason_suffix}'
        elif action in ('categorical_generalize', 'categorical_geo_generalize'):
            params = {'quasi_identifiers': [col],
                      'max_categories': param_val,
                      'return_metadata': True}
            reason = f'Top-{param_val} generalization{reason_suffix}'
        else:
            continue

        result[col] = {
            'action': action,
            'function': func_name,
            'params': params,
            'reason': reason,
        }

    if not result and len(df) > 0:
        warning = ('No preprocessing actions selected. All QIs will enter '
                   'generalization with original cardinality.')

    # Carry forward internal metadata (e.g. _hierarchies) from the
    # original plan so that apply_type_aware_preprocessing can use them.
    for key, val in original_plan.items():
        if key.startswith('_'):
            result[key] = val

    return result, warning


def estimate_cardinality(
    data: pd.DataFrame,
    plan_df: pd.DataFrame,
) -> pd.DataFrame:
    """Re-estimate Card. After for each row based on current Action + Param.

    Returns a copy of *plan_df* with the ``Card. After`` column updated.
    """
    df = plan_df.copy()
    estimates = []

    for _, row in df.iterrows():
        col = row['Column']
        action = row['Action']
        param_str = str(row['Param']).strip()
        nunique = int(row['Card. Before'])

        if col not in data.columns or action == 'keep':
            estimates.append(nunique)
            continue

        try:
            if action == 'date_truncation':
                freq_map = {'year': 'Y', 'quarter': 'Q', 'month': 'M'}
                freq = freq_map.get(param_str, 'Y')
                dates = pd.to_datetime(data[col], errors='coerce', dayfirst=True)
                est = int(dates.dt.to_period(freq).nunique())
            elif action == 'age_binning':
                bin_size = int(param_str) if param_str != '\u2014' else 5
                vals = data[col].dropna()
                est = int(np.ceil((vals.max() - vals.min()) / bin_size)) if len(vals) else nunique
            elif action == 'quantile_binning':
                n_bins = int(param_str) if param_str != '\u2014' else 10
                est = min(n_bins, nunique)
            elif action in ('categorical_generalize', 'categorical_geo_generalize'):
                max_cat = int(param_str) if param_str != '\u2014' else 20
                est = min(max_cat + 1, nunique)  # +1 for "Other"
            elif action == 'geographic_coarsening':
                keep_digits = int(param_str) if param_str != '\u2014' else 3
                est = int(data[col].dropna().astype(str).str[:keep_digits].nunique())
            elif action == 'top_bottom_coding':
                est = nunique  # cardinality rarely changes much
            else:
                est = nunique
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("[SmartDefaults] Cardinality estimation failed for '%s': %s", col, exc)
            est = nunique

        estimates.append(est)

    df['Card. After'] = estimates
    return df


def _detect_data_characteristics(
    data: pd.DataFrame,
    detected_qis: List[str],
    column_types: Dict[str, str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect data-specific issues and generate actionable warnings.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset
    detected_qis : List[str]
        Detected quasi-identifiers
    column_types : Dict[str, str], optional
        Pre-computed semantic data type labels from the role assignment table
        (e.g. ``{'age': 'Integer — Age (demographic)', ...}``).
        When provided, these are used instead of re-detecting types from scratch.

    Returns
    -------
    List[Dict]
        List of warnings/recommendations with keys:
        type, icon, method, message, action, columns
    """
    warnings = []
    if column_types is None:
        column_types = {}
    log.debug("[DataChars] Detecting characteristics for %d QIs: %s",
              len(detected_qis), detected_qis)

    # Rule 1: Direct identifiers (CRITICAL)
    try:
        direct_ids = auto_detect_direct_identifiers(data, check_patterns=True)
        if direct_ids:
            warnings.append({
                'type': 'CRITICAL',
                'icon': '🔴',
                'method': 'Remove Direct Identifiers',
                'message': f'Direct identifiers detected: {", ".join(list(direct_ids.keys())[:3])}{"..." if len(direct_ids) > 3 else ""}',
                'action': 'Drop these columns before processing',
                'columns': list(direct_ids.keys()),
                'details': direct_ids
            })
    except (ValueError, TypeError, KeyError) as exc:
        log.warning("[SmartDefaults] Direct identifier check failed: %s", exc)

    # Rule 2: Per-QI warnings — use semantic type label when available
    warned_cols = set()

    for col in detected_qis:
        if col not in data.columns:
            continue
        try:
            nunique = data[col].nunique()
            type_label = column_types.get(col, '')

            # Shared type classification (same helper as build_type_aware_preprocessing)
            ct = _classify_qi_type(col, data, type_label)
            is_date = ct['is_date']
            is_geo = ct['is_geo']
            is_numeric = ct['is_numeric']
            is_binary = ct['is_binary']
            is_id_like = ct['is_id_like']
            is_free_text = ct['is_free_text']
            is_coded = ct['is_coded']
            is_high_card = ct['is_high_card']
            is_ordinal = ct['is_ordinal']

            # Skip binary / low-cardinality ordinal — no action needed
            if is_binary or (is_ordinal and nunique <= 10):
                continue

            # Direct / free-text identifiers should not be QIs
            if is_id_like or is_free_text:
                warnings.append({
                    'type': 'CRITICAL',
                    'icon': '🔴',
                    'method': 'Remove Identifier',
                    'message': f'{col}: {column_types.get(col, "ID-like")} — should not be a QI',
                    'action': 'Remove this column or change role to "—"',
                    'columns': [col]
                })
                warned_cols.add(col)
                continue

            # Date columns with high temporal precision
            if is_date and nunique > 30:
                warnings.append({
                    'type': 'HIGH',
                    'icon': '🟠',
                    'method': 'Date Generalization',
                    'message': f'{col}: {nunique} unique dates — high temporal precision',
                    'action': 'Enable GENERALIZE to bin dates by month/year',
                    'columns': [col]
                })
                warned_cols.add(col)

            # Geographic columns with fine granularity
            elif is_geo and nunique > 20:
                warnings.append({
                    'type': 'HIGH',
                    'icon': '🟠',
                    'method': 'Geographic Generalization',
                    'message': f'{col}: {nunique} unique locations — fine-grained geography',
                    'action': 'Enable GENERALIZE to group into regions',
                    'columns': [col]
                })
                warned_cols.add(col)

            # High-cardinality numeric (continuous or coded numeric strings)
            elif (is_numeric or is_coded) and nunique > 50:
                try:
                    skewness = data[col].skew() if is_numeric else 0
                except (ValueError, TypeError) as exc:
                    log.warning("[SmartDefaults] Skewness failed for '%s': %s", col, exc)
                    skewness = 0
                if abs(skewness) > 2:
                    warnings.append({
                        'type': 'HIGH',
                        'icon': '🟠',
                        'method': 'Bin Numerics + Top/Bottom Coding',
                        'message': (f'{col}: {nunique} unique values, '
                                    f'skewed (skew={skewness:.1f})'),
                        'action': 'Enable "Bin numerics" and "Top/bottom coding"',
                        'columns': [col]
                    })
                else:
                    warnings.append({
                        'type': 'HIGH' if nunique > 200 else 'MEDIUM',
                        'icon': '🟠' if nunique > 200 else '🟡',
                        'method': 'Bin Numerics',
                        'message': f'{col}: {nunique} unique numeric values — high cardinality',
                        'action': 'Enable "Bin numerics" to reduce cardinality',
                        'columns': [col]
                    })
                warned_cols.add(col)

            # High-cardinality categorical (non-numeric, many categories)
            elif is_high_card or (not is_numeric and not is_coded and nunique > 50):
                warnings.append({
                    'type': 'HIGH',
                    'icon': '🟠',
                    'method': 'GENERALIZE (Top-K)',
                    'message': (f'{col}: {nunique} unique categories — '
                                f'too many for k-anonymity'),
                    'action': 'Enable "GENERALIZE" to keep top categories',
                    'columns': [col]
                })
                warned_cols.add(col)

            # Moderate categorical with rare values
            elif not is_numeric and not is_coded and nunique > 10:
                value_counts = data[col].value_counts(normalize=True)
                rare_count = (value_counts < 0.01).sum()
                if rare_count > 0:
                    warnings.append({
                        'type': 'MEDIUM',
                        'icon': '🟡',
                        'method': 'Merge Rare Categories',
                        'message': (f'{col}: {rare_count} rare categories '
                                    f'(<1% frequency) out of {nunique}'),
                        'action': 'Enable "Merge Rare Categories"',
                        'columns': [col]
                    })
                    warned_cols.add(col)
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("[SmartDefaults] QI warning generation failed for '%s': %s", col, exc)

    # Rule 3: Small dataset (MEDIUM)
    if len(data) < 1000:
        warnings.append({
            'type': 'MEDIUM',
            'icon': '🟡',
            'method': 'Synthetic Data Release',
            'message': f'Small dataset ({len(data)} rows) — anonymization may cause high suppression',
            'action': 'Consider full synthetic release via the Synthesize page as an alternative',
            'columns': []
        })

    # Rule 4: Highly sensitive numeric data (HIGH)
    sensitive_patterns = ['income', 'salary', 'wage', 'revenue', 'profit', 'credit', 'debt', 'balance']
    sensitive_numeric = [col for col in data.select_dtypes(include='number').columns
                        if any(p in col.lower() for p in sensitive_patterns)]
    if sensitive_numeric and len(sensitive_numeric) <= 5:
        warnings.append({
            'type': 'HIGH',
            'icon': '🟠',
            'method': 'Differential Privacy',
            'message': f'Sensitive numeric data detected: {", ".join(sensitive_numeric[:3])}{"..." if len(sensitive_numeric) > 3 else ""}',
            'action': 'Consider Differential Privacy for mathematical privacy guarantees',
            'columns': sensitive_numeric
        })

    if warnings:
        log.info("[DataChars] Found %d warnings: %s",
                 len(warnings),
                 [(w['type'], w.get('method', '')) for w in warnings])
    return warnings
