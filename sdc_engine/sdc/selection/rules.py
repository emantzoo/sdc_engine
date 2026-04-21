"""
Method Selection Rules
======================

Complete method selection framework with PRIMARY + FALLBACK logic.

VALIDATION CRITERIA (both must be met):
=======================================
  validation_pass = (ReID_target <= threshold) AND (utility >= minimum)

Threshold Examples by Access Type:
  - Public release:     ReID_95 <= 5%,  utility >= 70%
  - Scientific use:     ReID_95 <= 10%, utility >= 80%
  - Secure environment: ReID_95 <= 20%, utility >= 90%

RULE PRIORITY (evaluated in order, first match wins):
=====================================================
0a. REG1 - Regulatory compliance (PUBLIC + reid_target=3%)
    - REG1_high: reid_95 > 15% → kANON k=7 hybrid
    - REG1_moderate: 3% < reid_95 ≤ 15% → kANON k=5 hybrid

1.  Data Structure Rules - Tabular format detection
    - App is microdata-only; tabular data falls through to microdata rules

1b. Structural Risk Rules (SR3) - Near-unique QI with few QIs
    - SR3: n_qis ≤ 2 + max uniqueness > 70% + ReID95 > 20% → LOCSUPR k=3
    - Fires WITHOUT var_priority (catches cases RC1 misses)

2.  Risk Concentration Rules (RC1) - Per-QI risk contribution
    - Only fire when var_priority exists AND ReID_95 > 15%
    - RC1: dominated (top QI ≥40%) → LOCSUPR k=5
    - RC2/RC3/RC4 deleted (preempted-always by RC1, Spec 19 Phase 2)

2b. PUB1 - Public release structural preference (PUBLIC + reid_target=1%)
    - PUB1_high: reid_95 > 20% → kANON k=10 generalization
    - PUB1_moderate: 5% < reid_95 ≤ 20% → kANON k=7 hybrid

2c. SEC1 - Secure environment utility priority (SECURE + utility_floor ≥ 90%)
    - SEC1_cat: 5% < reid_95 ≤ 25%, cat ≥ 60% → PRAM p=0.10-0.225
    - SEC1_cont: 5% < reid_95 ≤ 25%, continuous present → NOISE mag=0.05-0.175

3.  Categorical-Aware Rules (CAT1) - PRAM for categorical data at moderate risk
    - GATED: only fires when risk_metric is l_diversity (PRAM invalidates
      frequency-count metrics like reid_95, k_anonymity, uniqueness)
    - CAT1: ≥70% categorical + ReID95 15-40% + no near-constant QIs → PRAM
    - CAT2 deleted: self-contradictory (NOISE in pipeline, blocked for l_diversity)

3b. L-Diversity Rules (LDIV1) - Sensitive column attribute disclosure
    - LDIV1: sensitive col n_unique ≤ 5 + estimated min_l < 2 → PRAM on sensitive cols
    - Advisory: warns that k-anonymity alone is insufficient

3c. Temporal-Dominant Rules (DATE1) - Date-only QI sets
    - DATE1: ≥80% of QIs are temporal + ReID95 ≤ 40% → PRAM on binned dates
    - Preserves temporal distribution vs kANON range merging

4.  ReID Risk Rules (QR0-QR4 + MED1) - Risk distribution patterns with fallbacks
    Each rule has: PRIMARY method + ReID fallback + utility fallback

    INFEASIBLE:
    - QR0: k-anonymity infeasible (EQ size < 3) → GENERALIZE_FIRST

    HIGH RISK (ReID_95 > 30%):
    - QR1: severe_tail → LOCSUPR(k=5)
    - QR2: heavy tail (>40%) → kANON(k=7), moderate tail (30-40%) → LOCSUPR(k=3)
    - QR3: uniform_high → kANON(k=10)
    - QR4: widespread → kANON(k=7-10)

    MODERATE RISK (ReID_95 > 20% or bimodal or >10% high-risk):
    - MED1: consolidated moderate structural → kANON(k=5) hybrid

5.  Low-Risk Rules (LOW1-LOW3) - Type-based at ReID_95 <= 20%
    - LOW1: categorical-dominant (≥60%) → PRAM
    - LOW2: continuous-dominant (≤40%) → NOISE or kANON
    - LOW3: mixed/high-cardinality → kANON

6.  Distribution Rules (DP1-DP4) - Outliers, skewness, sensitive, integer codes
    - DP4: integer-coded categorical (≤15 unique ints) → PRAM (preserves coding)

7.  Uniqueness Risk Rules (HR1-HR6) - Heuristic fallback when no ReID
    - HR6: < 200 rows → LOCSUPR k=3 + hard small-dataset warning

8.  Default Rules - Final fallbacks

FALLBACK LOGIC FLOW:
====================
1. Apply PRIMARY method
2. Calculate ReID_result and utility_result
3. Check validation:
   - If ReID fails but utility OK → use ReID_fallback (stronger)
   - If utility fails but ReID OK → use utility_fallback (weaker)
   - If both fail → try ReID_fallback first, then utility_fallback
   - If all fail → escalate to manual review

IMPORTANT: Perturbation methods (PRAM, NOISE) do NOT reduce ReID.
Only use perturbation when ReID is already low (< 5%) and utility is priority.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional

from .features import classify_risk_concentration, top_categorical_qis
from sdc_engine.sdc.protection_engine import build_data_features

log = logging.getLogger(__name__)


def _size_adjusted_k(features: Dict, proposed_k: int) -> int:
    """Reduce proposed k for small datasets where structural methods are
    destructive.  Datasets >= 5,000 are unaffected; < 200 are handled by HR6.
    """
    n = features.get('n_records', 10_000)
    if n >= 5_000:
        return proposed_k
    if n < 500:
        return min(proposed_k, 3)
    if n < 1_000:
        return max(3, proposed_k - 2)
    if n < 2_000:
        return max(3, proposed_k - 2)
    # 2,000–5,000: step down one schedule notch
    if proposed_k >= 10:
        return 7
    if proposed_k >= 7:
        return 5
    return proposed_k


def _clamp_k_by_suppression(features: Dict, proposed_k: int,
                             max_suppression: float = 0.30) -> int:
    """Reduce k if estimated suppression at proposed_k exceeds threshold.

    Step 1: Apply dataset-size adjustment (small datasets get lower k).
    Step 2: Walk down k schedule until suppression is acceptable.
    Floor: k=3.
    """
    k = _size_adjusted_k(features, proposed_k)
    est = features.get('estimated_suppression', {})
    for k_try in sorted([kv for kv in (k, 7, 5, 3) if kv <= k],
                        reverse=True):
        if est.get(k_try, 0) <= max_suppression:
            return k_try
    return 3  # floor


def _suppression_gated_kanon(
    features: Dict,
    proposed_k: int,
    rule_name: str,
    reason: str,
    qis: list,
    strategy: str = 'hybrid',
    suppression_threshold: float = 0.25,
) -> Dict:
    """Select kANON at proposed_k, but switch to perturbative if suppression
    is too high.  Embeds the suppression gate INTO the rule rather than
    relying on smart_method_config's post-selection switch.
    """
    est = features.get('estimated_suppression', {})
    est_supp = est.get(proposed_k, features.get('estimated_suppression_k5', 0))

    lower_k = max(3, proposed_k - 2)
    est_supp_lower = est.get(lower_k, est_supp)

    if est_supp > suppression_threshold:
        if est_supp_lower <= suppression_threshold:
            # Stepping down k is sufficient
            clamped_k = _clamp_k_by_suppression(features, proposed_k)
            return {
                'applies': True,
                'rule': f'{rule_name}_Clamped',
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': clamped_k,
                               'strategy': strategy},
                'reason': (f"{reason}. Est. suppression at k={proposed_k}: "
                           f"{est_supp:.0%} → clamped to k={clamped_k}"),
                'confidence': 'HIGH',
                'priority': 'REQUIRED',
                'suppression_estimate': est_supp,
            }
        else:
            # Even lower k would suppress too much — switch method
            n_cat = features.get('n_categorical', 0)
            n_cont = features.get('n_continuous', 0)
            total = n_cat + n_cont

            if total > 0 and n_cat / total >= 0.60:
                return {
                    'applies': True,
                    'rule': f'{rule_name}_Supp_Switch_PRAM',
                    'method': 'PRAM',
                    'parameters': {
                        'variables': top_categorical_qis(features),
                        'p_change': min(0.35, 0.20 + features.get('reid_95', 0.20)),
                    },
                    'reason': (f"{reason}. Est. suppression at k={lower_k}: "
                               f"{est_supp_lower:.0%} — switching to PRAM"),
                    'confidence': 'MEDIUM',
                    'priority': 'REQUIRED',
                    'suppression_estimate': est_supp,
                    'reid_fallback': {
                        'method': 'kANON',
                        'parameters': {'quasi_identifiers': qis, 'k': lower_k,
                                       'strategy': strategy},
                    },
                }
            elif n_cont > 0:
                return {
                    'applies': True,
                    'rule': f'{rule_name}_Supp_Switch_LOCSUPR',
                    'method': 'LOCSUPR',
                    'parameters': {'quasi_identifiers': qis, 'k': 3},
                    'reason': (f"{reason}. Est. suppression at k={lower_k}: "
                               f"{est_supp_lower:.0%} — LOCSUPR k=3 for "
                               f"minimal suppression"),
                    'confidence': 'MEDIUM',
                    'priority': 'REQUIRED',
                    'suppression_estimate': est_supp,
                    'reid_fallback': {
                        'method': 'kANON',
                        'parameters': {'quasi_identifiers': qis, 'k': lower_k,
                                       'strategy': strategy},
                    },
                }
            else:
                clamped_k = _clamp_k_by_suppression(features, proposed_k)
                return {
                    'applies': True,
                    'rule': f'{rule_name}_Clamped',
                    'method': 'kANON',
                    'parameters': {'quasi_identifiers': qis, 'k': clamped_k,
                                   'strategy': strategy},
                    'reason': (f"{reason}. Est. suppression high ({est_supp:.0%}) "
                               f"but no perturbative alternative — "
                               f"clamped to k={clamped_k}"),
                    'confidence': 'MEDIUM',
                    'priority': 'REQUIRED',
                    'suppression_estimate': est_supp,
                }

    # Suppression acceptable — proceed with kANON as planned
    k = _clamp_k_by_suppression(features, proposed_k)
    return {
        'applies': True,
        'rule': rule_name,
        'method': 'kANON',
        'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': strategy},
        'reason': reason,
        'confidence': 'HIGH',
        'priority': 'REQUIRED',
        'suppression_estimate': est.get(k, 0),
    }


def structural_risk_rules(features: Dict) -> Dict:
    """SR3: Near-unique QI with very few QIs — LOCSUPR over kANON.

    When there are only 1-2 QIs and one has very high uniqueness (>70%),
    kANON would collapse the entire column.  LOCSUPR with tight importance
    weighting is more surgical.

    This fires WITHOUT var_priority, catching cases that RC1 would miss
    when backward elimination hasn't run yet.
    """
    if not features.get('has_reid'):
        return {'applies': False}

    n_qis = features.get('n_qis', 0)
    reid_95 = features.get('reid_95', 0)
    max_qi_uniq = features.get('max_qi_uniqueness', 0)

    if n_qis <= 2 and max_qi_uniq > 0.70 and reid_95 > 0.20:
        qis = features['quasi_identifiers']
        # Find the high-uniqueness QI for importance weighting
        qi_cards = features.get('qi_cardinalities', {})
        n_rec = features.get('n_records', 1)
        high_qi = max(qi_cards, key=lambda q: qi_cards.get(q, 0) / n_rec) if qi_cards else None

        return {
            'applies': True,
            'rule': 'SR3_Near_Unique_Few_QIs',
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 3},
            'reason': (
                f"Only {n_qis} QI(s) with max uniqueness {max_qi_uniq:.0%}"
                f"{f' ({high_qi})' if high_qi else ''} — "
                f"targeted cell suppression is more surgical than generalization"
            ),
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'reid_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
            },
            'utility_fallback': None,
        }

    return {'applies': False}


def small_dataset_rules(features: Dict) -> Dict:
    """HR6: Very small dataset — warn and use minimal protection.

    On datasets < 200 rows, any k >= 5 causes catastrophic suppression,
    and GENERALIZE reduces the dataset to a handful of combinations.
    Fires regardless of ReID availability (this is a structural constraint).
    """
    n_records = features.get('n_records', 0)
    n_qis = features.get('n_qis', 0)

    if n_records >= 200 or n_qis < 2:
        return {'applies': False}

    qis = features['quasi_identifiers']
    return {
        'applies': True,
        'rule': 'HR6_Very_Small_Dataset',
        'method': 'LOCSUPR',
        'parameters': {
            'quasi_identifiers': qis,
            'k': 3,
            'max_suppressions_per_record': 1,
        },
        'reason': (
            f"Dataset too small ({n_records} rows) for effective k-anonymity. "
            f"Consider synthetic data release. Using LOCSUPR k=3 with max "
            f"1 suppression/record for partial protection."
        ),
        'confidence': 'MEDIUM',
        'priority': 'REQUIRED',
        'small_dataset_warning': (
            f"WARNING: {n_records}-row dataset is too small for target k>=5. "
            f"k-anonymity will suppress >30% of records. Consider "
            f"synthetic data generation instead."
        ),
        'reid_fallback': {
            'method': 'PRAM',
            'parameters': {'variables': top_categorical_qis(features) or qis[:5], 'p_change': 0.25},
        },
        'utility_fallback': None,
    }


def risk_concentration_rules(features: Dict) -> Dict:
    """
    Risk-concentration rule (RC1) using per-QI backward-elimination data.

    Only fires when ``var_priority`` exists AND ``reid_95 > 0.15`` — i.e. risk
    is elevated AND we have per-QI risk data to make a sharper decision than
    the generic ReID rules.

    RC2/RC3/RC4 were deleted in Spec 19 Phase 2 (preempted-always by RC1).
    The _compute_var_priority contribution metric produces non-normalized
    percentages (each QI's drop / baseline), so top_pct >= 40% always holds
    and the 'dominated' pattern always matches first.
    """
    var_priority = features.get('var_priority', {})
    if not var_priority:
        return {'applies': False}

    reid_95 = features.get('reid_95', 0)
    if reid_95 <= 0.15:
        return {'applies': False}

    risk_conc = features.get('risk_concentration', {})
    pattern = risk_conc.get('pattern', 'unknown')
    if pattern == 'unknown':
        return {'applies': False}

    qis = features.get('quasi_identifiers', [])
    top_qi = risk_conc.get('top_qi')
    top_pct = risk_conc.get('top_pct', 0)

    _audit = features.get('_audit', False)

    rc1_match = (pattern == 'dominated')

    _sub_trace = None
    if _audit:
        _sub_trace = [
            {'rule': 'RC1_Risk_Dominated', 'applies': rc1_match,
             'gate': f'pattern==dominated (actual={pattern}, top_pct={top_pct:.1f}%)'},
        ]

    # RC1: Dominated — one QI accounts for ≥40% of risk.
    # Targeted suppression beats global k-anonymity here.
    if rc1_match:
        result = {
            'applies': True,
            'rule': 'RC1_Risk_Dominated',
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 5},
            'reason': (
                f"Risk dominated by '{top_qi}' ({top_pct:.0f}%) — "
                f"targeted suppression more efficient than global k-anonymity"
            ),
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'preprocessing_hint': {'aggressive_qi': top_qi, 'pct': top_pct},
            'reid_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'},
            },
            'utility_fallback': None,
        }
        if _sub_trace:
            result['_sub_rule_trace'] = _sub_trace
        return result

    # Non-dominated pattern — let generic ReID rules handle it
    result = {'applies': False}
    if _sub_trace:
        result['_sub_rule_trace'] = _sub_trace
    return result


def data_structure_rules(features: Dict) -> Dict:
    """
    Rules based on data structure.

    App is microdata-only — tabular-specific methods (THRES, TABSUPR,
    CROUND, CTA) are no longer available.  If tabular data is detected
    we log a message and fall through to default microdata rules.
    """
    data_type = features['data_type']

    if data_type == 'tabular':
        import logging
        logging.getLogger(__name__).info(
            "Tabular data detected. Tabular-specific methods (THRES, TABSUPR, "
            "CROUND) are not available — microdata methods will be used."
        )

    return {'applies': False}


def _apply_treatment_balance(rule: Dict, qi_treatment: Optional[Dict]) -> Dict:
    """Shift protection aggressiveness based on user-set treatment levels.

    Post-selection adjustment — called after a rule matches, before returning.

    - >=60% Heavy → bump k +2 (or p/mag +0.05)
    - >=60% Light → reduce k -1 (or p/mag -0.05), prefer perturbative at <=30% risk
    - Mixed → no change
    """
    if not qi_treatment:
        return rule

    levels = list(qi_treatment.values())
    total = len(levels)
    if total == 0:
        return rule

    n_heavy = sum(1 for lv in levels if lv == 'Heavy')
    n_light = sum(1 for lv in levels if lv == 'Light')
    heavy_ratio = n_heavy / total
    light_ratio = n_light / total

    params = dict(rule.get('parameters', {}))
    method = rule.get('method', '')

    if heavy_ratio >= 0.60:
        if 'k' in params and method in ('kANON', 'LOCSUPR'):
            params['k'] = min(params['k'] + 2, 20)
        if 'p_change' in params:
            params['p_change'] = min(params['p_change'] + 0.05, 0.50)
        if 'magnitude' in params:
            params['magnitude'] = min(params['magnitude'] + 0.05, 0.50)
        rule = dict(rule)
        rule['parameters'] = params
        rule['treatment_adjustment'] = "Bumped params (>=60% Heavy treatment)"
    elif light_ratio >= 0.60:
        if 'k' in params and method in ('kANON', 'LOCSUPR'):
            params['k'] = max(params['k'] - 1, 3)
        if 'p_change' in params:
            params['p_change'] = max(params['p_change'] - 0.05, 0.05)
        if 'magnitude' in params:
            params['magnitude'] = max(params['magnitude'] - 0.03, 0.01)
        rule = dict(rule)
        rule['parameters'] = params
        rule['treatment_adjustment'] = "Reduced params (>=60% Light treatment)"

    return rule


def _has_dominant_categories(features, threshold=0.80):
    """True if ANY categorical QI has a single category >= threshold frequency.

    Used to EXCLUDE near-constant categoricals from PRAM selection
    (PRAM is ineffective when one category dominates).
    """
    qi_max_freq = features.get('qi_max_category_freq', {})
    return any(freq >= threshold for freq in qi_max_freq.values())


def categorical_aware_rules(features: Dict) -> Dict:
    """CAT1: Select PRAM when categorical data dominates at moderate risk.

    CAT2 deleted in Spec 19 Phase 2 — self-contradictory: gated to l_diversity
    but pipeline contained NOISE (blocked for l_diversity).

    Fills the gap where kANON was always selected at 15-40% risk even for
    predominantly categorical data.  PRAM preserves all records while kANON
    would suppress 15-25%.
    """
    if not features.get('has_reid'):
        return {'applies': False}

    # PRAM invalidates frequency-count-based risk metrics (reid_95, k_anonymity,
    # uniqueness).  Only allow PRAM selection for attribute-disclosure metrics
    # (l_diversity) where it genuinely helps.
    # Reference: sdcMicro docs — "Risk measures based on frequency counts of
    # keys are no longer valid after perturbative methods."
    risk_metric = features.get('_risk_metric_type', 'reid95')
    if risk_metric not in ('l_diversity',):
        return {'applies': False}

    reid_95 = features.get('reid_95', 0)
    qis = features['quasi_identifiers']
    n_cat = features['n_categorical']
    n_cont = features['n_continuous']
    total = n_cat + n_cont
    if total == 0:
        return {'applies': False}
    cat_ratio = n_cat / total

    # CAT1: Moderate risk + predominantly categorical + no near-constant QIs
    # PRAM preserves all records while kANON would suppress 15-25%
    if (0.15 <= reid_95 <= 0.40
            and cat_ratio >= 0.70
            and not _has_dominant_categories(features, 0.80)):
        p = 0.35 if reid_95 > 0.30 else (0.30 if reid_95 > 0.20 else 0.25)
        est_supp = features.get('estimated_suppression', {}).get(5,
                       features.get('estimated_suppression_k5', 0))
        return {
            'applies': True,
            'rule': 'CAT1_Categorical_Dominant',
            'method': 'PRAM',
            'parameters': {'variables': top_categorical_qis(features), 'p_change': p},
            'reason': (f"Categorical-dominant ({cat_ratio:.0%}) at moderate risk "
                       f"(ReID95={reid_95:.1%}). PRAM preserves all records"
                       f"{f' (kANON k=5 would suppress ~{est_supp:.0%})' if est_supp > 0.10 else ''}"),
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED',
            'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'}},
            'utility_fallback': {'method': 'PRAM', 'parameters': {'variables': top_categorical_qis(features), 'p_change': max(0.10, p - 0.10)}},
        }

    # CAT2 deleted in Spec 19 Phase 2 — self-contradictory (NOISE in pipeline,
    # blocked for l_diversity).  The 50-70% categorical window now falls
    # through to QR/MED/LOW rules which select kANON.

    return {'applies': False}


def low_risk_rules(features: Dict) -> Dict:
    """Consolidated low-risk rules (ReID95 <= 20%).

    Replaces QR8, QR9, LR1-LR4 with three type-based rules:
      LOW1: Categorical-dominant (>=60% cat) -> PRAM
      LOW2: Continuous-dominant (<=40% cat) -> NOISE or kANON
      LOW3: Mixed or high-cardinality -> kANON
    """
    if features['data_type'] != 'microdata':
        return {'applies': False}

    reid_95 = features.get('reid_95', 1.0)
    if features.get('has_reid') and reid_95 > 0.20:
        return {'applies': False}
    if not features.get('has_reid'):
        return {'applies': False}  # Defer to uniqueness_risk_rules

    qis = features['quasi_identifiers']
    n_cat = features['n_categorical']
    n_cont = features['n_continuous']
    total = n_cat + n_cont
    if total == 0:
        return {'applies': False}

    cat_ratio = n_cat / total
    is_very_low = reid_95 <= 0.05
    has_outliers = features.get('has_outliers', False)
    high_card = features.get('high_cardinality_count', 0)

    # LOW1: Categorical-dominant (>=60% cat) -> PRAM
    if cat_ratio >= 0.60 and high_card == 0 and reid_95 <= 0.10:
        p = 0.15 if is_very_low else 0.20
        return {
            'applies': True,
            'rule': 'LOW1_Categorical',
            'method': 'PRAM',
            'parameters': {'variables': top_categorical_qis(features), 'p_change': p},
            'reason': f"Low risk (ReID95={reid_95:.1%}), categorical-dominant ({cat_ratio:.0%}) — PRAM preserves all records",
            'confidence': 'HIGH',
            'priority': 'RECOMMENDED',
            'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 3}},
            'utility_fallback': None,
        }

    # LOW2: Continuous-dominant (<=40% cat) -> NOISE or kANON
    # NOISE only at very-low risk (<=5%) or low risk with outliers (<=10%).
    # Above 10%, structural kANON remains the default even with outliers,
    # preserving the structural guarantee from the former QR8/QR9 rules.
    if cat_ratio <= 0.40 and n_cont > 0:
        if is_very_low or (has_outliers and reid_95 <= 0.10):
            mag = 0.20 if has_outliers else 0.15
            return {
                'applies': True,
                'rule': 'LOW2_Continuous_Noise',
                'method': 'NOISE',
                'parameters': {'variables': features['continuous_vars'], 'magnitude': mag},
                'reason': (f"Low risk (ReID95={reid_95:.1%}), continuous-dominant"
                           f"{' — outliers present, ' if has_outliers else ' — '}"
                           f"NOISE preserves distributions"),
                'confidence': 'HIGH',
                'priority': 'RECOMMENDED',
                'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 3}},
                'utility_fallback': None,
            }
        k = 3 if is_very_low else 5
        return {
            'applies': True,
            'rule': 'LOW2_Continuous_kANON',
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': 'generalization'},
            'reason': f"Low risk (ReID95={reid_95:.1%}), continuous-dominant — light kANON sufficient",
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED',
            'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': k + 2}},
            'utility_fallback': None,
        }

    # LOW3: Mixed or high-cardinality -> kANON
    k = 3 if is_very_low else 5
    return {
        'applies': True,
        'rule': 'LOW3_Mixed',
        'method': 'kANON',
        'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': 'generalization'},
        'reason': f"Low risk (ReID95={reid_95:.1%}), mixed types — kANON k={k}",
        'confidence': 'MEDIUM',
        'priority': 'RECOMMENDED',
        'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': k + 2}},
        'utility_fallback': None,
    }


# Backward compatibility aliases
low_risk_structure_rules = low_risk_rules
microdata_structure_rules = low_risk_rules


def reid_risk_rules(features: Dict) -> Dict:
    """
    ReID-based risk rules using risk distribution patterns.

    Now includes k-anonymity feasibility check to warn when QI combination space
    is too large for effective anonymization.
    """
    if not features.get('has_reid'):
        return {'applies': False}  # Defer to uniqueness_risk_rules in rule chain

    reid_50 = features['reid_50']
    reid_95 = features['reid_95']
    reid_99 = features['reid_99']
    risk_pattern = features['risk_pattern']
    high_risk_rate = features.get('high_risk_rate', 0)
    qis = features['quasi_identifiers']

    # Check k-anonymity feasibility first
    feasibility = features.get('k_anonymity_feasibility')
    expected_eq_size = features.get('expected_eq_size')

    # If k-anonymity is infeasible, return special recommendation
    if feasibility == 'infeasible':
        recommended_qi = features.get('recommended_qi_to_remove')
        qi_cardinalities = features.get('qi_cardinalities', {})

        warning_msg = (
            f"K-ANONYMITY INFEASIBLE: QI combination space ({features.get('qi_cardinality_product', 0):,}) "
            f"exceeds dataset size ({features['n_records']:,}). "
            f"Expected EQ size: {expected_eq_size:.2f} < 3."
        )
        if recommended_qi:
            warning_msg += f" Consider removing or binning '{recommended_qi}' ({qi_cardinalities.get(recommended_qi, 0)} unique values)."

        return {
            'applies': True,
            'rule': 'QR0_K_Anonymity_Infeasible',
            'method': 'GENERALIZE_FIRST',  # Special method indicating preprocessing needed
            'parameters': {'quasi_identifiers': qis, 'recommended_qi_to_remove': recommended_qi},
            'reason': warning_msg,
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'feasibility': 'infeasible',
            'expected_eq_size': expected_eq_size,
            'qi_cardinalities': qi_cardinalities,
            # Fallback: if user insists, use PRAM for perturbation (doesn't reduce ReID but adds noise)
            'reid_fallback': {'method': 'PRAM', 'parameters': {'variables': qis[:5], 'p_change': 0.25}},
            'utility_fallback': None
        }

    # QR1: Severe tail risk (ReID_99/ReID_50 > 10)
    # PRIMARY: LOCSUPR - suppress few outliers for better utility than generalizing all
    # FALLBACK: kANON if LOCSUPR fails
    if risk_pattern == 'severe_tail':
        return {
            'applies': True,
            'rule': 'QR1_Severe_Tail_Risk',
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 5},
            'reason': f"Severe tail: ReID_50={reid_50:.1%}, ReID_99={reid_99:.1%} - few records dominate risk",
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'}},
            'utility_fallback': None  # LOCSUPR is already minimal intervention
        }

    # QR2: Tail risk - high ReID_95 needs structural protection
    # Heavy tail (ReID_95 > 40%): kANON for aggressive protection
    # Moderate tail (30-40%): LOCSUPR for targeted suppression
    if risk_pattern == 'tail' or (reid_95 > 0.30 and reid_50 < 0.15):
        if reid_95 > 0.40:
            # Suppression-aware: if kANON k=7 would suppress >25%, LOCSUPR preserves more
            # Use k=7 estimate since that's the k we'd select for heavy tail
            est_supp = features.get('estimated_suppression', {}).get(7,
                           features.get('estimated_suppression_k5', 0))
            if est_supp > 0.25:
                return {
                    'applies': True,
                    'rule': 'QR2_Heavy_Tail_Low_Suppression',
                    'method': 'LOCSUPR',
                    'parameters': {'quasi_identifiers': qis, 'k': 5},
                    'reason': (f"Heavy tail risk (ReID95={reid_95:.1%}) but est. suppression at k=7 "
                               f"{est_supp:.0%} — LOCSUPR preserves more records"),
                    'confidence': 'HIGH',
                    'priority': 'REQUIRED',
                    'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'}},
                    'utility_fallback': None,
                }
            k = _clamp_k_by_suppression(features, 7)
            return {
                'applies': True,
                'rule': 'QR2_Heavy_Tail_Risk',
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': 'hybrid'},
                'reason': f"Heavy tail risk: ReID_95={reid_95:.1%} - k-anonymity k={k} for structural protection",
                'confidence': 'HIGH',
                'priority': 'REQUIRED',
                'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 10, 'strategy': 'generalization'}},
                'utility_fallback': {'method': 'LOCSUPR', 'parameters': {'quasi_identifiers': qis, 'k': 5}}
            }
        return {
            'applies': True,
            'rule': 'QR2_Moderate_Tail_Risk',
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 3},
            'reason': f"Tail risk: ReID_50={reid_50:.1%}, ReID_95={reid_95:.1%} - targeted suppression",
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'}},
            'utility_fallback': None  # LOCSUPR is already minimal intervention
        }

    # QR3: Uniform high risk (ReID_50 > 20% - most records high risk)
    # PRIMARY: kANON(k=10) — suppression-gated
    if risk_pattern == 'uniform_high':
        result = _suppression_gated_kanon(
            features, proposed_k=10,
            rule_name='QR3_Uniform_High_Risk',
            reason=f"Uniform high risk: ReID_50={reid_50:.1%} — widespread protection needed",
            qis=qis, strategy='generalization')
        result.setdefault('reid_fallback', {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 12, 'strategy': 'generalization'}})
        result.setdefault('utility_fallback', {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'}})
        return result

    # QR4: Widespread risk — suppression-gated
    if risk_pattern == 'widespread' and reid_50 > 0.15:
        if reid_95 > 0.50:
            result = _suppression_gated_kanon(
                features, proposed_k=10,
                rule_name='QR4_Widespread_High',
                reason=f"Widespread high risk: ReID_50={reid_50:.1%}, ReID_95={reid_95:.1%} — aggressive k-anonymity",
                qis=qis, strategy='generalization')
            result.setdefault('reid_fallback', {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 12, 'strategy': 'generalization'}})
            result.setdefault('utility_fallback', {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'}})
            return result
        result = _suppression_gated_kanon(
            features, proposed_k=7,
            rule_name='QR4_Widespread_Moderate',
            reason=f"Widespread: ReID_50={reid_50:.1%} — k-anonymity for structural protection",
            qis=qis, strategy='hybrid')
        result.setdefault('reid_fallback', {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 10, 'strategy': 'generalization'}})
        result.setdefault('utility_fallback', {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'}})
        return result

    # MED1: Moderate risk needing structural protection
    # Consolidated from QR5 (moderate spread), QR6 (bimodal), QR7 (many high-risk)
    is_moderate_spread = (reid_95 > 0.20 and reid_50 < 0.10)
    is_bimodal = (risk_pattern == 'bimodal')
    has_significant_tail = (high_risk_rate > 0.10)

    if is_moderate_spread or is_bimodal or has_significant_tail:
        triggers = []
        if is_moderate_spread:
            triggers.append(f"moderate spread (ReID_50={reid_50:.1%})")
        if is_bimodal:
            triggers.append("bimodal pattern")
        if has_significant_tail:
            triggers.append(f"{high_risk_rate:.1%} at risk >20%")
        trigger_str = '; '.join(triggers)

        result = _suppression_gated_kanon(
            features, proposed_k=5,
            rule_name='MED1_Moderate_Structural',
            reason=f"Moderate risk ({trigger_str}) — k-anonymity with hybrid strategy",
            qis=qis, strategy='hybrid')
        result.setdefault('reid_fallback', {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'generalization'}})
        result.setdefault('utility_fallback', {
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 3}})
        # Adjust confidence for non-spread triggers
        if not is_moderate_spread:
            result['confidence'] = 'MEDIUM'
            result['priority'] = 'RECOMMENDED'
        return result

    # No ReID rule applies — defer to low_risk_rules or other downstream rules
    return {'applies': False}


def uniqueness_risk_rules(features: Dict) -> Dict:
    """
    Fallback risk rules when ReID is unavailable.
    """
    uniqueness = features['uniqueness_rate']
    n_qis = features['n_qis']
    n_records = features['n_records']
    qis = features['quasi_identifiers']

    # HR1: Extreme uniqueness
    if uniqueness > 0.20:
        return {
            'applies': True,
            'rule': 'HR1_Extreme_Uniqueness',
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 5},
            'reason': f"Extreme uniqueness ({uniqueness:.1%})",
            'confidence': 'HIGH',
            'priority': 'REQUIRED'
        }

    # HR2: Very high uniqueness
    if uniqueness > 0.10:
        k = _clamp_k_by_suppression(features, 7)
        return {
            'applies': True,
            'rule': 'HR2_Very_High_Uniqueness',
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': 'hybrid'},
            'reason': f"Very high uniqueness ({uniqueness:.1%})",
            'confidence': 'HIGH',
            'priority': 'REQUIRED'
        }

    # HR3: High uniqueness with QIs
    if uniqueness > 0.05 and n_qis >= 2:
        k = _clamp_k_by_suppression(features, 5)
        return {
            'applies': True,
            'rule': 'HR3_High_Uniqueness_QIs',
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': 'generalization'},
            'reason': f"Uniqueness {uniqueness:.1%} with {n_qis} QIs",
            'confidence': 'HIGH',
            'priority': 'REQUIRED'
        }

    # Note: HR6 (very small dataset) is now a standalone early rule
    # (small_dataset_rules) that fires before RC/ReID rules.

    # HR4: Very small dataset
    if n_records < 100:
        return {
            'applies': True,
            'rule': 'HR4_Very_Small_Dataset',
            'method': 'PRAM',
            'parameters': {'variables': top_categorical_qis(features), 'p_change': 0.3},
            'reason': f"Small dataset ({n_records} records)",
            'confidence': 'MEDIUM',
            'priority': 'REQUIRED'
        }

    # HR5: Small dataset
    if 100 <= n_records < 500 and uniqueness > 0.03:
        if features['n_continuous'] > 0:
            return {
                'applies': True,
                'rule': 'HR5_Small_Dataset',
                'method': 'NOISE',
                'parameters': {'variables': features['continuous_vars'], 'magnitude': 0.15},
                'reason': f"Small dataset with {uniqueness:.1%} uniqueness",
                'confidence': 'MEDIUM',
                'priority': 'REQUIRED'
            }
        return {
            'applies': True,
            'rule': 'HR5_Small_Dataset',
            'method': 'PRAM',
            'parameters': {'variables': top_categorical_qis(features), 'p_change': 0.25},
            'reason': f"Small dataset with {uniqueness:.1%} uniqueness",
            'confidence': 'MEDIUM',
            'priority': 'REQUIRED'
        }

    return {'applies': False}


def distribution_rules(features: Dict) -> Dict:
    """
    Rules based on distribution characteristics.
    """
    # DP1: Outliers → NOISE for continuous perturbation
    if features['has_outliers'] and features['n_continuous'] > 0:
        return {
            'applies': True,
            'rule': 'DP1_Outliers',
            'method': 'NOISE',
            'parameters': {'variables': features['continuous_vars'], 'magnitude': 0.20},
            'reason': "Outliers present - noise perturbation for continuous data",
            'confidence': 'HIGH',
            'priority': 'RECOMMENDED',
        }

    # DP2: Skewed columns
    if len(features['skewed_columns']) >= 2:
        return {
            'applies': True,
            'rule': 'DP2_Skewed',
            'method': 'PRAM',
            'parameters': {'variables': features['skewed_columns'][:5], 'p_change': 0.2},
            'reason': f"{len(features['skewed_columns'])} skewed columns",
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED'
        }

    # DP3: Sensitive attributes
    if features['has_sensitive_attributes'] and features['n_qis'] >= 2:
        return {
            'applies': True,
            'rule': 'DP3_Sensitive',
            'method': 'kANON',
            'parameters': {'quasi_identifiers': features['quasi_identifiers'], 'k': 5, 'strategy': 'generalization'},
            'reason': "Sensitive attributes present",
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED'
        }

    # DP4: Integer-coded categorical QIs
    # Numeric columns with ≤15 unique integer values are likely categorical codes
    # (municipality_code, education_level, etc.).  kANON range-binning ("1-5")
    # destroys the coding structure — PRAM preserves it.
    # Reid ceiling tightened from 0.30 to 0.20 (Spec 18 Item 4): at high reid,
    # QR-family rules provide stronger methods; DP4 serves low-risk data only.
    integer_coded = features.get('integer_coded_qis', [])
    reid_95 = features.get('reid_95', 0)
    if integer_coded and reid_95 <= 0.20:
        # Scale p_change: more categories → lower perturbation probability
        n_unique_max = max(
            features.get('qi_cardinalities', {}).get(qi, 10)
            for qi in integer_coded
        )
        if n_unique_max <= 5:
            p = 0.30
        elif n_unique_max <= 10:
            p = 0.25
        else:
            p = 0.20
        return {
            'applies': True,
            'rule': 'DP4_Integer_Coded_Categorical',
            'method': 'PRAM',
            'parameters': {'variables': integer_coded[:5], 'p_change': p},
            'reason': (
                f"Integer-coded categorical QI(s): {', '.join(integer_coded[:3])} "
                f"(≤{n_unique_max} unique) — PRAM preserves coding structure, "
                f"kANON range-binning would destroy category meaning"
            ),
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED',
            'reid_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': features['quasi_identifiers'], 'k': 5, 'strategy': 'hybrid'},
            },
            'utility_fallback': None,
        }

    return {'applies': False}


def l_diversity_rules(features: Dict) -> Dict:
    """LDIV1: Warn when sensitive column has very low diversity.

    If a sensitive column has n_unique ≤ 5, most k-anonymous equivalence
    classes will be homogeneous in the sensitive value (l ≈ 1), making
    k-anonymity useless against attribute disclosure.

    This is advisory — it recommends PRAM on the sensitive column in addition
    to whatever QI protection method is chosen.
    """
    # PRAM is blocked for k-anonymity / uniqueness metrics — skip this rule
    risk_metric = features.get('_risk_metric_type', 'reid95')
    if risk_metric in ('k_anonymity', 'uniqueness'):
        return {'applies': False}

    # Under reid95 metric, LDIV1's action (PRAM on sensitive columns) doesn't
    # reduce QI-based re-identification risk.  When reid95 is elevated (>10%),
    # defer to QR/MED rules that address reid95 directly.  At reid95 ≤ 10%,
    # QI protection is non-urgent and PRAM-on-sensitive is appropriate.
    if risk_metric == 'reid95' and features.get('reid_95', 0) > 0.10:
        return {'applies': False}

    sens_div = features.get('sensitive_column_diversity')
    if sens_div is None or sens_div > 5:
        return {'applies': False}

    # Only fire when we have QIs and some risk context
    if not features.get('quasi_identifiers') or not features.get('has_reid'):
        return {'applies': False}

    # Don't fire when QI space is infeasible — PRAM on sensitive columns
    # won't reduce structural risk; let QR0/QR1+ handle infeasible cases first
    feasibility = features.get('k_anonymity_feasibility')
    est_supp_k3 = features.get('estimated_suppression', {}).get(3, 0)
    if feasibility == 'infeasible' or est_supp_k3 > 0.50:
        return {'applies': False}

    qis = features['quasi_identifiers']
    sens_cols = list(features.get('sensitive_columns', {}).keys())
    if not sens_cols:
        return {'applies': False}

    # Use pre-computed l-diversity from features if available (computed
    # by build_data_features), otherwise heuristic estimate
    estimated_min_l = features.get('min_l')
    if estimated_min_l is None:
        # Fallback heuristic: with only `sens_div` distinct sensitive values,
        # random assignment into EQ classes means min_l ≈ 1 when EQ size ≈ k
        expected_eq = features.get('expected_eq_size', 5)
        estimated_min_l = min(sens_div, expected_eq) if expected_eq else 1

    if estimated_min_l >= 2:
        return {'applies': False}

    # Check if DATE1 would also fire — if so, merge into a pipeline
    # so both sensitive cols and date QIs get PRAM in one pass.
    qi_types = features.get('qi_type_counts', {})
    n_date = qi_types.get('date', 0)
    n_qis_total = features.get('n_qis', 0)
    date_ratio = n_date / n_qis_total if n_qis_total > 0 else 0
    reid_95 = features.get('reid_95', 0)
    date1_would_fire = (n_qis_total >= 2 and n_date >= 2
                        and date_ratio >= 0.50 and reid_95 <= 0.40)

    if date1_would_fire:
        # Merge: PRAM on sensitive cols + PRAM on date QIs
        # Use the higher p_change and deduplicate variable lists
        date_p = 0.25 if reid_95 > 0.20 else 0.20
        merged_p = max(0.15, date_p)
        merged_vars = list(dict.fromkeys(sens_cols[:3] + qis[:5]))  # deduplicated, order preserved
        return {
            'applies': True,
            'rule': 'LDIV1_DATE1_Merged',
            'method': 'PRAM',
            'parameters': {
                'variables': merged_vars,
                'p_change': merged_p,
            },
            'reason': (
                f"Sensitive column(s) have only {sens_div} distinct values (l ≈ {estimated_min_l:.0f}) "
                f"AND date-dominant QIs ({n_date}/{n_qis_total} temporal) — "
                f"merged PRAM on both sensitive + date columns"
            ),
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED',
            'l_diversity_warning': (
                f"Low l-diversity risk: sensitive column has {sens_div} distinct values. "
                f"k-anonymous groups may be homogeneous in the sensitive attribute."
            ),
            'reid_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
            },
            'utility_fallback': {
                'method': 'PRAM',
                'parameters': {'variables': sens_cols[:3], 'p_change': 0.15},
            },
        }

    return {
        'applies': True,
        'rule': 'LDIV1_Low_Sensitive_Diversity',
        'method': 'PRAM',
        'parameters': {
            'variables': sens_cols[:3],
            'p_change': 0.15,
        },
        'reason': (
            f"Sensitive column(s) have only {sens_div} distinct values — "
            f"k-anonymity alone offers no protection against attribute disclosure "
            f"(estimated l ≈ {estimated_min_l:.0f}). Adding light PRAM on sensitive columns."
        ),
        'confidence': 'MEDIUM',
        'priority': 'RECOMMENDED',
        'l_diversity_warning': (
            f"Low l-diversity risk: sensitive column has {sens_div} distinct values. "
            f"k-anonymous groups may be homogeneous in the sensitive attribute. "
            f"Consider l-diversity or PRAM on sensitive columns."
        ),
        'reid_fallback': {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
        },
        'utility_fallback': None,
    }


def temporal_dominant_rules(features: Dict) -> Dict:
    """DATE1: Date-dominant QI sets → PRAM on binned dates.

    When ≥50% of QIs are temporal, kANON generalization produces overlapping
    date ranges that are hard to interpret analytically.  PRAM on the already-
    binned date columns preserves temporal distribution shape.

    Threshold widened from 0.80 to 0.50 (Spec 18 Item 3). The 0.80 bar
    was unreachable on non-time-series data — a dataset needs at least half
    its QIs to be temporal for PRAM-on-dates to be the preferred approach.
    """
    qi_types = features.get('qi_type_counts', {})
    n_date = qi_types.get('date', 0)
    n_qis = features.get('n_qis', 0)
    reid_95 = features.get('reid_95', 0)

    if n_qis < 2 or n_date < 2:
        return {'applies': False}

    date_ratio = n_date / n_qis if n_qis > 0 else 0
    if date_ratio < 0.50 or reid_95 > 0.40:
        return {'applies': False}

    qis = features['quasi_identifiers']
    p = 0.25 if reid_95 > 0.20 else 0.20

    return {
        'applies': True,
        'rule': 'DATE1_Temporal_Dominant',
        'method': 'PRAM',
        'parameters': {'variables': qis[:5], 'p_change': p},
        'reason': (
            f"Date-dominant QI set ({n_date}/{n_qis} temporal) — PRAM on binned "
            f"periods preserves temporal distribution better than kANON range merging"
        ),
        'confidence': 'MEDIUM',
        'priority': 'RECOMMENDED',
        'reid_fallback': {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
        },
        'utility_fallback': None,
    }


def regulatory_compliance_rules(features: Dict) -> Dict:
    """REG1: Regulatory compliance — reid_95 ≤ 3%, utility_floor 88%.

    Distinguished from public_release (reid_95 ≤ 1%) by the target value.
    Fires FIRST in the chain after pipeline rules.
    """
    if features.get('data_type') != 'microdata':
        return {'applies': False}

    access_tier = features.get('_access_tier', 'SCIENTIFIC')
    reid_target = features.get('_reid_target_raw')

    # REG1 fires only when target is 0.03 (regulatory_compliance context)
    if access_tier != 'PUBLIC' or reid_target is None:
        return {'applies': False}
    if abs(reid_target - 0.03) > 1e-6:
        return {'applies': False}

    reid_95 = features.get('reid_95', 0)
    if reid_95 <= 0.03:
        return {'applies': False}  # Already meets target

    qis = features.get('quasi_identifiers', [])

    if reid_95 > 0.15:
        return {
            'applies': True,
            'rule': 'REG1_Regulatory_High_Risk',
            'method': 'kANON',
            'parameters': {
                'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid',
            },
            'reason': (
                f"Regulatory compliance (ReID95={reid_95:.1%}, target=3%) — "
                f"hybrid kANON k=7 with suppression gating"
            ),
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'reid_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 10, 'strategy': 'generalization'},
            },
            'utility_fallback': {
                'method': 'LOCSUPR',
                'parameters': {'quasi_identifiers': qis, 'k': 5},
            },
        }

    return {
        'applies': True,
        'rule': 'REG1_Regulatory_Moderate_Risk',
        'method': 'kANON',
        'parameters': {
            'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid',
        },
        'reason': (
            f"Regulatory compliance (ReID95={reid_95:.1%}, target=3%) — "
            f"hybrid kANON k=5"
        ),
        'confidence': 'HIGH',
        'priority': 'REQUIRED',
        'reid_fallback': {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'},
        },
        'utility_fallback': {
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 5},
        },
    }


def public_release_rules(features: Dict) -> Dict:
    """PUB1: In PUBLIC access tier, prefer structural methods even at moderate ReID.

    PUBLIC tier targets reid_95 ≤ 1%, which PRAM/NOISE cannot achieve reliably.
    Routes to kANON before CAT1/LOW rules can pick a perturbation method.

    Rule priority: fires BEFORE categorical_aware_rules.
    """
    if features.get('data_type') != 'microdata':
        return {'applies': False}

    access_tier = features.get('_access_tier', 'SCIENTIFIC')
    if access_tier != 'PUBLIC':
        return {'applies': False}

    # Distinguish regulatory_compliance (handled by REG1) from public_release
    # regulatory_compliance uses reid_target 0.03, public_release uses 0.01
    reid_target = features.get('_reid_target_raw')
    if reid_target is not None and abs(reid_target - 0.03) < 1e-6:
        return {'applies': False}  # REG1 handles this

    reid_95 = features.get('reid_95', 0)
    qis = features.get('quasi_identifiers', [])

    if reid_95 <= 0.05:
        return {'applies': False}  # Low risk — LOW rules are fine

    if reid_95 > 0.20:
        return {
            'applies': True,
            'rule': 'PUB1_Public_Release_High_Risk',
            'method': 'kANON',
            'parameters': {
                'quasi_identifiers': qis, 'k': 10,
                'strategy': 'generalization',
            },
            'reason': (
                f"PUBLIC release with ReID95={reid_95:.1%} — structural "
                f"protection required (PRAM/NOISE cannot meet reid_95 ≤ 1% target)"
            ),
            'confidence': 'HIGH',
            'priority': 'REQUIRED',
            'reid_fallback': {
                'method': 'LOCSUPR',
                'parameters': {'quasi_identifiers': qis, 'k': 7},
            },
            'utility_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
            },
        }

    # 0.05 < reid_95 <= 0.20
    return {
        'applies': True,
        'rule': 'PUB1_Public_Release_Moderate_Risk',
        'method': 'kANON',
        'parameters': {
            'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid',
        },
        'reason': (
            f"PUBLIC release with ReID95={reid_95:.1%} — hybrid kANON "
            f"balances structural protection and utility"
        ),
        'confidence': 'HIGH',
        'priority': 'REQUIRED',
        'reid_fallback': {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 10, 'strategy': 'generalization'},
        },
        'utility_fallback': {
            'method': 'LOCSUPR',
            'parameters': {'quasi_identifiers': qis, 'k': 5},
        },
    }


def secure_environment_rules(features: Dict) -> Dict:
    """SEC1: In SECURE access tier with moderate ReID, prefer perturbation.

    SECURE tier allows reid_95 ≤ 10% and requires utility ≥ 92%. Structural
    methods are often wasteful here — perturbation can meet target with
    significantly better utility.

    Rule priority: fires BEFORE reid_risk_rules (QR0-QR4).
    """
    if features.get('data_type') != 'microdata':
        return {'applies': False}

    access_tier = features.get('_access_tier', 'SCIENTIFIC')
    if access_tier != 'SECURE':
        return {'applies': False}

    reid_95 = features.get('reid_95', 0)
    if reid_95 > 0.25 or reid_95 < 0.05:
        # Above 0.25: even SECURE needs structural; below 0.05: LOW rules handle it
        return {'applies': False}

    utility_floor = features.get('_utility_floor', 0.80)
    if utility_floor < 0.90:
        return {'applies': False}  # Only fire when utility is really a priority

    qis = features.get('quasi_identifiers', [])
    n_cat = features.get('n_categorical', 0)
    n_cont = features.get('n_continuous', 0)
    total = n_cat + n_cont
    if total == 0:
        return {'applies': False}
    cat_ratio = n_cat / total

    if cat_ratio >= 0.60:
        p_change = round(min(0.225, 0.10 + reid_95 * 0.5), 2)
        pram_vars = top_categorical_qis(features) or qis[:5]
        return {
            'applies': True,
            'rule': 'SEC1_Secure_Categorical',
            'method': 'PRAM',
            'parameters': {'variables': pram_vars, 'p_change': p_change},
            'reason': (
                f"SECURE tier with ReID95={reid_95:.1%}, {cat_ratio:.0%} "
                f"categorical — PRAM preserves utility within risk budget"
            ),
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED',
            'reid_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 3, 'strategy': 'hybrid'},
            },
            'utility_fallback': None,
        }

    if n_cont > 0 and cat_ratio < 0.60:
        magnitude = round(min(0.175, 0.05 + reid_95 * 0.5), 2)
        cont_vars = features.get('continuous_vars', [])
        return {
            'applies': True,
            'rule': 'SEC1_Secure_Continuous',
            'method': 'NOISE',
            'parameters': {'variables': cont_vars, 'magnitude': magnitude},
            'reason': (
                f"SECURE tier with ReID95={reid_95:.1%}, continuous-dominant — "
                f"NOISE preserves distribution within risk budget"
            ),
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED',
            'reid_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': 3, 'strategy': 'hybrid'},
            },
            'utility_fallback': None,
        }

    return {'applies': False}


def default_rules(features: Dict) -> Dict:
    """
    Default fallback rules.
    """
    # Microdata with QIs
    if features['data_type'] == 'microdata' and features['n_qis'] >= 2:
        return {
            'applies': True,
            'rule': 'DEFAULT_Microdata_QIs',
            'method': 'kANON',
            'parameters': {'quasi_identifiers': features['quasi_identifiers'], 'k': 3, 'strategy': 'generalization'},
            'reason': "Default: k-anonymity for microdata with QIs",
            'confidence': 'LOW',
            'priority': 'DEFAULT'
        }

    # Mostly categorical
    if features['n_categorical'] > features['n_continuous']:
        return {
            'applies': True,
            'rule': 'DEFAULT_Categorical',
            'method': 'PRAM',
            'parameters': {'variables': top_categorical_qis(features), 'p_change': 0.2},
            'reason': "Default: mostly categorical",
            'confidence': 'LOW',
            'priority': 'DEFAULT'
        }

    # Mostly continuous
    if features['n_continuous'] > features['n_categorical']:
        return {
            'applies': True,
            'rule': 'DEFAULT_Continuous',
            'method': 'NOISE',
            'parameters': {'variables': features['continuous_vars'], 'magnitude': 0.15},
            'reason': "Default: mostly continuous",
            'confidence': 'LOW',
            'priority': 'DEFAULT'
        }

    # Final fallback
    return {
        'applies': True,
        'rule': 'DEFAULT_Fallback',
        'method': 'PRAM',
        'parameters': {'p_change': 0.2},
        'reason': "Fallback",
        'confidence': 'VERY_LOW',
        'priority': 'FALLBACK'
    }


def select_method_by_features(
    data: pd.DataFrame,
    analysis: Dict,
    quasi_identifiers: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict:
    """
    Select SDC method based on data features and ReID metrics.

    Pure data-driven selection using ReID distribution patterns.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    analysis : dict
        Output from analyze_data()
    quasi_identifiers : list, optional
        QI columns
    verbose : bool
        Print selection details

    Returns:
    --------
    dict : {method, parameters, rule, confidence, reason, features}
    """
    # Extract sensitive_columns from analysis dict (if present) for build_data_features
    _sens = analysis.get('sensitive_columns', {})
    _sens_list = list(_sens.keys()) if isinstance(_sens, dict) else list(_sens) if _sens else None
    features = build_data_features(
        data, quasi_identifiers,
        sensitive_columns=_sens_list or None,
    )

    if verbose and features.get('has_reid'):
        print(f"\n[ReID Metrics]")
        print(f"  ReID_50: {features['reid_50']:.1%}, ReID_95: {features['reid_95']:.1%}, ReID_99: {features['reid_99']:.1%}")
        print(f"  Pattern: {features['risk_pattern']}, High-risk: {features['high_risk_count']} ({features['high_risk_rate']:.1%})")

    # RULE PRIORITY (first match wins):
    # 1.  Data Structure Rules - Tabular format detection
    # 1b. HR6 - Very small dataset (< 200 rows) — structural constraint
    # 1c. SR3 - Near-unique QI with few QIs (no var_priority needed)
    # 2.  Risk Concentration Rules (RC1-RC4) - Per-QI risk data
    # 3.  Categorical-Aware Rules (CAT1) - PRAM at moderate risk
    # 3b. LDIV1 - l-diversity gap for sensitive columns
    # 3c. DATE1 - Date-dominant QI sets
    # 4.  ReID Risk Rules (QR0-QR4 + MED1) - Risk distribution patterns
    # 5.  Low-Risk Rules (LOW1-LOW3) - Type-based at ReID_95 <= 20%
    # 6.  Distribution Rules (DP1-DP4) - Outliers, skewness, integer codes
    # 7.  Uniqueness Risk Rules (HR1-HR5) - Heuristic fallback when no ReID
    # 8.  Default Rules - Final fallbacks
    # Note: Pipeline escalation is handled reactively by smart_protect() if needed
    rule_factories = [
        regulatory_compliance_rules,   # REG1 — first (PUBLIC + target=3%)
        data_structure_rules,
        small_dataset_rules,
        structural_risk_rules,
        risk_concentration_rules,
        public_release_rules,          # PUB1 — before CAT (PUBLIC + target=1%)
        secure_environment_rules,      # SEC1 — before CAT (SECURE tier)
        categorical_aware_rules,
        l_diversity_rules,
        temporal_dominant_rules,
        reid_risk_rules,
        low_risk_rules,
        distribution_rules,
        uniqueness_risk_rules,
        default_rules,
    ]

    for rule_fn in rule_factories:
        rule_check = rule_fn(features)
        if rule_check['applies']:
            # Apply treatment balance post-adjustment
            rule_check = _apply_treatment_balance(
                rule_check, features.get('qi_treatment'))
            log.info("[Rules] Matched: %s -> %s (%s): %s",
                     rule_check['rule'], rule_check.get('method', '?'),
                     rule_check['confidence'], rule_check['reason'][:120])
            if verbose:
                print(f"\n[Rule] {rule_check['rule']} -> {rule_check['method']} ({rule_check['confidence']})")
                print(f"  {rule_check['reason']}")
            # Include features so pipeline rules can be evaluated later
            rule_check['features'] = features
            return rule_check

    return {
        'applies': True,
        'method': 'kANON',
        'parameters': {'k': 5},
        'rule': 'EMERGENCY_FALLBACK',
        'reason': 'No rule matched',
        'confidence': 'VERY_LOW',
        'features': features
    }
