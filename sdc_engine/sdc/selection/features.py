"""
Feature Extraction for Method Selection
========================================

Extracts data characteristics including ReID metrics for rule-based selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Import ReID calculation and pattern classification from metrics
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdc_utils import calculate_reid
from metrics.reid import classify_risk_pattern


def classify_risk_concentration(var_priority: Optional[Dict] = None) -> Dict:
    """Classify how backward-elimination risk is distributed across QIs.

    The thresholds (40% dominated, 60% concentrated, 3+ HIGH) are initial
    heuristics to validate empirically once composite utility enables
    systematic comparison.

    Returns dict with pattern, top_qi, top_pct, top2_pct, n_high_risk.
    """
    if not var_priority:
        return {'pattern': 'unknown', 'top_qi': None, 'top_pct': 0,
                'top2_pct': 0, 'n_high_risk': 0}
    sorted_qis = sorted(var_priority.items(),
                        key=lambda x: x[1][1], reverse=True)
    top_qi, (top_label, top_pct) = sorted_qis[0]
    top2_pct = sum(pct for _, (_, pct) in sorted_qis[:2])
    n_high = sum(1 for _, (label, _) in sorted_qis
                 if 'HIGH' in label and 'MED' not in label)

    if top_pct >= 40:
        pattern = 'dominated'
    elif top2_pct >= 60:
        pattern = 'concentrated'
    elif n_high >= 3:
        pattern = 'spread_high'
    else:
        pattern = 'balanced'

    return {
        'pattern': pattern,
        'top_qi': top_qi,
        'top_pct': top_pct,
        'top2_pct': top2_pct,
        'n_high_risk': n_high,
    }


def extract_data_features_with_reid(
    data: pd.DataFrame,
    analysis: Dict,
    quasi_identifiers: Optional[List[str]]
) -> Dict:
    """
    Extract data features including ReID metrics for method selection.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    analysis : dict
        Output from analyze_data()
    quasi_identifiers : list, optional
        QI columns

    Returns:
    --------
    dict : Feature dictionary with ReID metrics and risk pattern
    """
    if quasi_identifiers is None:
        quasi_identifiers = analysis.get('quasi_identifiers', [])

    features = {
        'n_records': len(data),
        'n_columns': len(data.columns),
        'data_type': analysis.get('data_type', 'microdata'),
        'n_continuous': len(analysis.get('continuous_variables', [])),
        'n_categorical': len(analysis.get('categorical_variables', [])),
        'n_qis': len(quasi_identifiers),
        'continuous_vars': analysis.get('continuous_variables', []),
        'categorical_vars': analysis.get('categorical_variables', []),
        'quasi_identifiers': quasi_identifiers,
        'uniqueness_rate': analysis.get('uniqueness_rate', 0),
        'risk_level': analysis.get('risk_level', 'medium'),
        'high_cardinality_qis': [],
        'low_cardinality_qis': [],
        'high_cardinality_count': 0,
        'has_sensitive_attributes': bool(analysis.get('sensitive_columns')),
        'sensitive_columns': analysis.get('sensitive_columns', {}),
        'small_cells_rate': 0,
        'has_reid': False,
        'has_outliers': False,
        'skewed_columns': [],
        # QI combination space metrics
        'qi_cardinality_product': 1,
        'expected_eq_size': None,
        'k_anonymity_feasibility': None,  # 'easy', 'moderate', 'hard', 'infeasible'
        # New fields for extended rules (SR3, DP4, LDIV1, DATE1)
        'max_qi_uniqueness': 0.0,          # max(nunique/n_records) across QIs
        'integer_coded_qis': [],           # QIs that are numeric with <=15 unique ints
        'sensitive_column_diversity': None, # min n_unique across sensitive columns
        'qi_type_counts': {'date': 0, 'geo': 0, 'numeric': 0, 'categorical': 0},
    }

    # Calculate cardinality for QIs and QI combination space
    if quasi_identifiers:
        qi_cardinality_product = 1
        qi_cardinalities = {}

        for qi in quasi_identifiers:
            if qi in data.columns:
                nunique = data[qi].nunique()
                qi_cardinalities[qi] = nunique
                qi_cardinality_product *= nunique

                # Relative cardinality (as fraction of dataset size)
                rel_card = nunique / len(data)
                if rel_card > 0.5:
                    features['high_cardinality_qis'].append(qi)
                    features['high_cardinality_count'] += 1
                elif rel_card < 0.1:
                    features['low_cardinality_qis'].append(qi)

        features['qi_cardinality_product'] = qi_cardinality_product
        features['qi_cardinalities'] = qi_cardinalities

        # Max QI uniqueness (for SR3: near-unique QI detection)
        n_rec = len(data)
        if n_rec > 0 and qi_cardinalities:
            features['max_qi_uniqueness'] = max(
                nu / n_rec for nu in qi_cardinalities.values())

        # Integer-coded QIs (for DP4: numeric cols with ≤15 unique integer values)
        integer_coded = []
        for qi in quasi_identifiers:
            if qi in data.columns and qi_cardinalities.get(qi, 0) <= 15:
                if pd.api.types.is_numeric_dtype(data[qi]):
                    non_null = data[qi].dropna()
                    if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
                        integer_coded.append(qi)
        features['integer_coded_qis'] = integer_coded

        # QI type counts (for DATE1: date-dominant detection)
        _date_hints = {'date', 'time', 'year', 'month', 'quarter', 'period',
                       'day', 'ημερομηνια', 'ημερομηνία', 'ετος', 'έτος'}
        _geo_hints = {'region', 'city', 'state', 'country', 'zip', 'postal',
                      'county', 'district', 'municipality', 'prefecture',
                      'νομος', 'νομός', 'περιφερεια', 'περιφέρεια', 'δημος', 'δήμος'}
        qi_type_counts = {'date': 0, 'geo': 0, 'numeric': 0, 'categorical': 0}
        for qi in quasi_identifiers:
            if qi not in data.columns:
                continue
            name_lower = qi.lower()
            if any(h in name_lower for h in _date_hints):
                qi_type_counts['date'] += 1
            elif any(h in name_lower for h in _geo_hints):
                qi_type_counts['geo'] += 1
            elif pd.api.types.is_numeric_dtype(data[qi]):
                qi_type_counts['numeric'] += 1
            else:
                qi_type_counts['categorical'] += 1
        features['qi_type_counts'] = qi_type_counts
        features['n_geo_qis'] = qi_type_counts['geo']

        # Geographic QI granularity classification (for GEO1 pipeline)
        geo_qis_by_granularity = {}
        for qi in quasi_identifiers:
            if qi not in data.columns:
                continue
            name_lower = qi.lower()
            if any(h in name_lower for h in _geo_hints):
                card = data[qi].nunique()
                geo_qis_by_granularity[qi] = 'fine' if card > 50 else 'coarse'
        features['geo_qis_by_granularity'] = geo_qis_by_granularity

        # Per-QI max category frequency (for categorical-aware rule selection)
        qi_max_cat_freq = {}
        for qi in quasi_identifiers:
            if qi in data.columns:
                is_cat = not pd.api.types.is_numeric_dtype(data[qi]) or data[qi].nunique() <= 20
                if is_cat:
                    vc = data[qi].value_counts(normalize=True)
                    qi_max_cat_freq[qi] = float(vc.iloc[0]) if len(vc) > 0 else 0.0
        features['qi_max_category_freq'] = qi_max_cat_freq

        # Fast kANON suppression estimate at multiple k values
        # Single groupby pass, then threshold at k=3,5,7 to support rules
        # selecting different k levels.
        estimated_suppression = {3: 0.0, 5: 0.0, 7: 0.0}
        qi_in_df = [q for q in quasi_identifiers if q in data.columns]
        if qi_in_df and len(data) > 0:
            try:
                eq_sizes = data.groupby(qi_in_df, dropna=False).size()
                n = len(data)
                for k_val in (3, 5, 7):
                    records_below = int(eq_sizes[eq_sizes < k_val].sum())
                    estimated_suppression[k_val] = records_below / n
            except Exception:
                pass
        features['estimated_suppression'] = estimated_suppression
        features['estimated_suppression_k5'] = estimated_suppression[5]  # backward compat

        # Calculate expected equivalence class size
        # Expected EQ size = n_records / (product of QI cardinalities)
        if qi_cardinality_product > 0:
            expected_eq_size = len(data) / qi_cardinality_product
            features['expected_eq_size'] = expected_eq_size

            # Determine k-anonymity feasibility
            # - easy: Expected EQ >= 10 (can achieve k=10 easily)
            # - moderate: 5 <= Expected EQ < 10 (can achieve k=5)
            # - hard: 3 <= Expected EQ < 5 (can barely achieve k=3)
            # - infeasible: Expected EQ < 3 (cannot achieve even k=3 without massive suppression)
            if expected_eq_size >= 10:
                features['k_anonymity_feasibility'] = 'easy'
                features['max_achievable_k'] = int(expected_eq_size)
            elif expected_eq_size >= 5:
                features['k_anonymity_feasibility'] = 'moderate'
                features['max_achievable_k'] = 5
            elif expected_eq_size >= 3:
                features['k_anonymity_feasibility'] = 'hard'
                features['max_achievable_k'] = 3
            else:
                features['k_anonymity_feasibility'] = 'infeasible'
                features['max_achievable_k'] = 0
                # Identify the highest-cardinality QI that should be removed/binned
                if qi_cardinalities:
                    highest_card_qi = max(qi_cardinalities, key=qi_cardinalities.get)
                    features['recommended_qi_to_remove'] = highest_card_qi

    # Calculate small cells rate for tabular data
    if features['data_type'] == 'tabular':
        try:
            numeric_data = data.select_dtypes(include=[np.number])
            if numeric_data.size > 0:
                small_cells = (numeric_data < 3).sum().sum()
                features['small_cells_rate'] = small_cells / numeric_data.size
        except:
            pass

    # Detect outliers (1.5 × IQR threshold — intentionally more sensitive than
    # the legacy 3 × IQR in select_method.py, which is dead code that delegates
    # to selection.rules.select_method_by_features())
    continuous_vars = features['continuous_vars']
    if continuous_vars:
        for var in continuous_vars:
            if var in data.columns:
                try:
                    col = data[var].dropna()
                    if len(col) > 10:
                        q1, q3 = col.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outliers = ((col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)).sum()
                        if outliers / len(col) > 0.02:
                            features['has_outliers'] = True
                            break
                except:
                    pass

    # Detect skewed columns
    for var in continuous_vars:
        if var in data.columns:
            try:
                col = data[var].dropna()
                if len(col) > 10:
                    skew = col.skew()
                    if abs(skew) > 1.5:
                        features['skewed_columns'].append(var)
            except:
                pass

    # Calculate risk metric if we have QIs
    if quasi_identifiers and features['data_type'] == 'microdata':
        try:
            reid = calculate_reid(data, quasi_identifiers)
            if reid and 'reid_50' in reid:
                features['has_reid'] = True
                features['reid_50'] = reid['reid_50']
                features['reid_95'] = reid['reid_95']
                features['reid_99'] = reid['reid_99']
                features['mean_risk'] = reid.get('mean_risk', reid['reid_50'])
                features['max_risk'] = reid.get('max_risk', reid['reid_99'])
                features['risk_pattern'] = classify_risk_pattern(reid)

                # Count high-risk records
                risk_scores = reid.get('risk_scores', [])
                if len(risk_scores) > 0:
                    high_risk = sum(1 for r in risk_scores if r > 0.20)
                    features['high_risk_count'] = high_risk
                    features['high_risk_rate'] = high_risk / len(risk_scores)
                else:
                    features['high_risk_count'] = 0
                    features['high_risk_rate'] = 0
                features['_risk_metric_type'] = 'reid95'
        except Exception:
            pass

    # Sensitive column diversity (for LDIV1: l-diversity gap detection)
    sens_cols = analysis.get('sensitive_columns', {})
    if sens_cols:
        min_div = None
        for sc in sens_cols:
            if sc in data.columns:
                nu = data[sc].nunique()
                if min_div is None or nu < min_div:
                    min_div = nu
        features['sensitive_column_diversity'] = min_div

        # Compute actual l-diversity when QIs are available
        if quasi_identifiers and min_div and min_div <= 10:
            try:
                from sdc_engine.sdc.post_protection_diagnostics import check_l_diversity
                sens_col_names = list(sens_cols.keys()) if isinstance(sens_cols, dict) else list(sens_cols)
                l_result = check_l_diversity(
                    data, quasi_identifiers, sens_col_names,
                    l_target=2, size_threshold=100)
                features['min_l'] = l_result.get('l_achieved')
                features['l_diversity'] = l_result
            except Exception:
                pass


    return features


def top_categorical_qis(features: Dict, n: int = 5) -> List[str]:
    """Return up to *n* categorical QIs, ordered by risk contribution.

    When ``var_priority`` is available (from backward elimination),
    the riskiest categorical QIs are returned first.  Otherwise falls
    back to positional order (original behaviour).
    """
    cat_vars = features.get('categorical_vars', [])
    var_priority = features.get('var_priority', {})

    if var_priority:
        def _get_contribution(v):
            entry = var_priority.get(v)
            if entry is None:
                return 0
            # tuple format: (label, pct) from backward elimination
            if isinstance(entry, (tuple, list)):
                return entry[1] if len(entry) > 1 else 0
            # dict format: {'contribution': float}
            if isinstance(entry, dict):
                return entry.get('contribution', 0)
            return 0

        cat_with_risk = [
            (v, _get_contribution(v))
            for v in cat_vars
        ]
        cat_with_risk.sort(key=lambda x: x[1], reverse=True)
        return [v for v, _ in cat_with_risk[:n]]

    return cat_vars[:n]
