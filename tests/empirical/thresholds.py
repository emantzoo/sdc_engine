"""
Threshold definitions -- what to vary and where.

Each ThresholdTest describes one threshold, a function to patch it
temporarily, and the values to test.
"""
from dataclasses import dataclass
from typing import Callable, List, Any
from contextlib import contextmanager
from unittest.mock import patch


@dataclass
class ThresholdTest:
    id: str
    name: str
    current_value: float
    test_values: List[float]
    description: str
    # Function that takes a value and returns a context manager
    # that monkey-patches the threshold in the rules module
    patcher: Callable[[float], Any]


@contextmanager
def _patch_rc1_dominated(value: float):
    """Patch RC1's 0.40 cutoff in classify_risk_concentration.

    The threshold is ``top_pct >= 40`` (where top_pct is already a
    percentage 0-100).  We replace the function so ``dominated`` triggers
    at ``top_pct >= value * 100``.
    """
    from sdc_engine.sdc.selection import features as feat_mod
    original = feat_mod.classify_risk_concentration

    def patched(var_priority=None):
        result = original(var_priority)
        top_pct = result.get("top_pct", 0)
        # Re-classify with the variable cutoff (value is 0-1 fraction)
        # Only 'dominated' vs 'not_dominated' — RC2/RC3/RC4 deleted.
        if top_pct >= value * 100:
            result["pattern"] = "dominated"
        else:
            result["pattern"] = "not_dominated"
        return result

    with patch.object(feat_mod, "classify_risk_concentration", patched):
        yield


@contextmanager
def _patch_cat1_ratio(value: float):
    """Patch CAT1's 0.70 categorical ratio threshold.

    Replaces ``categorical_aware_rules`` so CAT1 triggers at
    ``cat_ratio >= value`` instead of ``cat_ratio >= 0.70``.

    IMPORTANT: Cannot delegate to original() because original re-checks
    the hardcoded 0.70 gate AND the _has_dominant_categories gate.  We
    build the CAT1 result directly, intentionally removing the
    dominant-categories check so the threshold is the only variable.
    """
    from sdc_engine.sdc.selection import rules as rules_mod
    original = rules_mod.categorical_aware_rules

    def patched(features):
        if not features.get('has_reid'):
            return {'applies': False}
        # PRAM invalidates frequency-count-based risk metrics — gate must
        # match categorical_aware_rules() in rules.py.
        risk_metric = features.get('_risk_metric_type', 'reid95')
        if risk_metric not in ('l_diversity',):
            return {'applies': False}
        n_cat = features.get('n_categorical', 0)
        n_cont = features.get('n_continuous', 0)
        total = n_cat + n_cont
        if total == 0:
            return {'applies': False}
        cat_ratio = n_cat / total
        reid_95 = features.get('reid_95', 1.0)
        qis = features['quasi_identifiers']

        # CAT1 with variable threshold (replaces 0.70)
        # Intentionally omits _has_dominant_categories to isolate threshold
        if cat_ratio >= value and 0.15 <= reid_95 <= 0.40:
            p = 0.35 if reid_95 > 0.30 else (0.30 if reid_95 > 0.20 else 0.25)
            from sdc_engine.sdc.selection.rules import top_categorical_qis
            return {
                'applies': True,
                'rule': 'CAT1_Categorical_Dominant',
                'method': 'PRAM',
                'parameters': {
                    'variables': top_categorical_qis(features),
                    'p_change': p,
                },
                'reason': f"Categorical-dominant ({cat_ratio:.0%}) at moderate risk (ReID95={reid_95:.1%})",
                'confidence': 'MEDIUM',
                'priority': 'RECOMMENDED',
                'reid_fallback': {
                    'method': 'kANON',
                    'parameters': {'quasi_identifiers': qis, 'k': 5, 'strategy': 'hybrid'},
                },
                'utility_fallback': None,
            }

        # CAT2 deleted in Spec 19 Phase 2 — self-contradictory
        # (gated to l_diversity, pipeline used NOISE, blocked for l_diversity).

        return {'applies': False}

    with patch.object(rules_mod, "categorical_aware_rules", patched):
        yield


@contextmanager
def _patch_qr2_suppression_gate(value: float):
    """Patch QR2's 0.25 suppression gate in reid_risk_rules.

    The original condition is ``est_supp > 0.25`` inside the
    ``reid_95 > 0.40`` branch of QR2.  We replace the whole rule
    function so the gate triggers at ``est_supp > value``.
    """
    from sdc_engine.sdc.selection import rules as rules_mod
    original = rules_mod.reid_risk_rules

    def patched(features):
        result = original(features)
        if not result.get('applies'):
            return result
        rule = result.get('rule', '')
        # Only intercept QR2 heavy-tail decisions
        if rule in ('QR2_Heavy_Tail_Risk', 'QR2_Heavy_Tail_Low_Suppression'):
            reid_95 = features.get('reid_95', 0)
            if reid_95 > 0.40:
                est_supp = features.get('estimated_suppression', {}).get(7,
                               features.get('estimated_suppression_k5', 0))
                qis = features['quasi_identifiers']
                if est_supp > value:
                    result['rule'] = 'QR2_Heavy_Tail_Low_Suppression'
                    result['method'] = 'LOCSUPR'
                    result['parameters'] = {'quasi_identifiers': qis, 'k': 5}
                else:
                    result['rule'] = 'QR2_Heavy_Tail_Risk'
                    result['method'] = 'kANON'
                    from sdc_engine.sdc.selection.rules import _clamp_k_by_suppression
                    k = _clamp_k_by_suppression(features, 7)
                    result['parameters'] = {'quasi_identifiers': qis, 'k': k,
                                            'strategy': 'hybrid'}
        return result

    with patch.object(rules_mod, "reid_risk_rules", patched):
        yield


@contextmanager
def _patch_low1_reid_gate(value: float):
    """Patch LOW1's ``reid_95 <= 0.10`` gate in low_risk_rules.

    Replaces the function so LOW1 triggers at ``reid_95 <= value``
    instead of ``reid_95 <= 0.10``.

    IMPORTANT: Cannot delegate to original() for the fallthrough path
    because original re-checks its own LOW1 gate.  We re-implement
    the full LOW1/LOW2/LOW3 logic with only the LOW1 reid gate changed.
    """
    from sdc_engine.sdc.selection import rules as rules_mod
    original = rules_mod.low_risk_rules

    def patched(features):
        if features.get('data_type') != 'microdata':
            return {'applies': False}
        reid_95 = features.get('reid_95', 1.0)
        if features.get('has_reid') and reid_95 > 0.20:
            return {'applies': False}
        if not features.get('has_reid'):
            return {'applies': False}

        qis = features['quasi_identifiers']
        n_cat = features.get('n_categorical', 0)
        n_cont = features.get('n_continuous', 0)
        total = n_cat + n_cont
        if total == 0:
            return {'applies': False}
        cat_ratio = n_cat / total
        is_very_low = reid_95 <= 0.05
        has_outliers = features.get('has_outliers', False)
        high_card = features.get('high_cardinality_count', 0)

        # LOW1 with variable reid gate (replaces hardcoded 0.10)
        if cat_ratio >= 0.60 and high_card == 0 and reid_95 <= value:
            from sdc_engine.sdc.selection.rules import top_categorical_qis
            p = 0.15 if is_very_low else 0.20
            return {
                'applies': True,
                'rule': 'LOW1_Categorical',
                'method': 'PRAM',
                'parameters': {
                    'variables': top_categorical_qis(features),
                    'p_change': p,
                },
                'reason': f"Low risk (ReID95={reid_95:.1%}), categorical-dominant ({cat_ratio:.0%}) — PRAM",
                'confidence': 'HIGH',
                'priority': 'RECOMMENDED',
                'reid_fallback': {
                    'method': 'kANON',
                    'parameters': {'quasi_identifiers': qis, 'k': 3},
                },
                'utility_fallback': None,
            }

        # LOW2/LOW3: delegate to original — LOW1 gate already failed so
        # original's LOW1 check at reid_95 <= 0.10 doesn't matter (if our
        # patched gate at `value` failed, the tighter original gate also fails
        # UNLESS value < 0.10 and reid_95 is between value and 0.10).
        # To be safe, skip LOW1 in original by going directly to LOW2/LOW3:

        # LOW2: Continuous-dominant
        if cat_ratio <= 0.40 and n_cont > 0:
            if is_very_low or (has_outliers and reid_95 <= 0.10):
                mag = 0.20 if has_outliers else 0.15
                return {
                    'applies': True,
                    'rule': 'LOW2_Continuous_Noise',
                    'method': 'NOISE',
                    'parameters': {'variables': features['continuous_vars'], 'magnitude': mag},
                    'reason': f"Low risk (ReID95={reid_95:.1%}), continuous-dominant — NOISE",
                    'confidence': 'HIGH',
                    'priority': 'RECOMMENDED',
                    'reid_fallback': {
                        'method': 'kANON',
                        'parameters': {'quasi_identifiers': qis, 'k': 3},
                    },
                    'utility_fallback': None,
                }
            k = 3 if is_very_low else 5
            return {
                'applies': True,
                'rule': 'LOW2_Continuous_kANON',
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': 'generalization'},
                'reason': f"Low risk (ReID95={reid_95:.1%}), continuous-dominant — kANON",
                'confidence': 'MEDIUM',
                'priority': 'RECOMMENDED',
                'reid_fallback': {
                    'method': 'kANON',
                    'parameters': {'quasi_identifiers': qis, 'k': k + 2},
                },
                'utility_fallback': None,
            }

        # LOW3: Mixed or high-cardinality
        k = 3 if is_very_low else 5
        return {
            'applies': True,
            'rule': 'LOW3_Mixed',
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': k, 'strategy': 'hybrid'},
            'reason': f"Low risk (ReID95={reid_95:.1%}), mixed types — kANON",
            'confidence': 'MEDIUM',
            'priority': 'RECOMMENDED',
            'reid_fallback': {
                'method': 'kANON',
                'parameters': {'quasi_identifiers': qis, 'k': k + 2},
            },
            'utility_fallback': None,
        }

    with patch.object(rules_mod, "low_risk_rules", patched):
        yield


THRESHOLDS: List[ThresholdTest] = [
    ThresholdTest(
        id="T1",
        name="RC1 dominated cutoff",
        current_value=0.40,
        test_values=[0.30, 0.35, 0.40, 0.45, 0.50],
        description="Top-QI contribution required to route to LOCSUPR via RC1.",
        patcher=_patch_rc1_dominated,
    ),
    ThresholdTest(
        id="T2",
        name="CAT1 categorical ratio",
        current_value=0.70,
        test_values=[0.60, 0.65, 0.70, 0.75, 0.80],
        description="Categorical ratio required for CAT1 -> PRAM.",
        patcher=_patch_cat1_ratio,
    ),
    ThresholdTest(
        id="T3",
        name="QR2 suppression gate",
        current_value=0.25,
        test_values=[0.15, 0.20, 0.25, 0.30, 0.35],
        description="Estimated suppression at which QR2 switches from kANON to LOCSUPR.",
        patcher=_patch_qr2_suppression_gate,
    ),
    ThresholdTest(
        id="T4",
        name="LOW1 reid_95 gate",
        current_value=0.10,
        test_values=[0.05, 0.075, 0.10, 0.125, 0.15],
        description="reid_95 ceiling for LOW1 -> PRAM.",
        patcher=_patch_low1_reid_gate,
    ),
]
