"""
Pipeline Selection Rules
========================

Multi-method pipeline selection for complex data scenarios.

Pipelines are triggered when single methods are demonstrably insufficient.
"""

import logging
import os
from typing import Dict, List

from .features import top_categorical_qis

log = logging.getLogger(__name__)

# Spec 16 audit flag — when set, select_method_suite() evaluates all rules
# (no short-circuit) and attaches a _rule_trace to the result.
_RULE_AUDIT = os.environ.get('SDC_RULE_AUDIT', '') == '1'


def _is_infeasible(features: Dict) -> bool:
    """Check if k-anonymity is infeasible for the current QI set.

    Prevents pipelines from selecting kANON/LOCSUPR on data where
    the QI combination space is too large — those rules would suppress
    >50% of records.  Caller should fall through to QR0_Infeasible.
    """
    feasibility = features.get('k_anonymity_feasibility')
    if feasibility == 'infeasible':
        return True
    # Also guard on estimated suppression: if k=5 would suppress >50%,
    # the pipeline shouldn't select kANON/LOCSUPR at any k.
    est = features.get('estimated_suppression', {})
    if est.get(3, 0) > 0.50:
        return True
    return False


def build_dynamic_pipeline(features: Dict) -> Dict:
    """Build pipeline from data features instead of matching hardcoded patterns.

    Only triggers when risk is elevated (ReID95 > 15%) and data has
    mixed types that benefit from multi-method treatment.
    """
    reid_95 = features.get('reid_95', 0)
    if reid_95 <= 0.15 or not features.get('has_reid'):
        return {'applies': False}

    # Guard: don't build kANON/LOCSUPR pipelines when suppression is catastrophic
    if _is_infeasible(features):
        log.info("[Pipeline] Skipping dynamic pipeline — QI space infeasible "
                 "(est. suppression at k=3: %.0f%%)",
                 features.get('estimated_suppression', {}).get(3, 0) * 100)
        return {'applies': False}

    qis = features['quasi_identifiers']
    n_cont = features['n_continuous']
    n_cat = features['n_categorical']
    has_outliers = features.get('has_outliers', False)
    high_risk_rate = features.get('high_risk_rate', 0)

    # Categorical guard: categorical-dominant data is handled by CAT1
    # (when metric is l_diversity) or falls through to reid_risk_rules.
    total_vars = n_cat + n_cont
    if total_vars > 0:
        cat_ratio = n_cat / total_vars
        if cat_ratio >= 0.70:
            log.info("[Pipeline] Skipping dynamic pipeline — cat_ratio=%.0f%%, "
                     "categorical-dominant", cat_ratio * 100)
            return {'applies': False}
        # DYN_CAT deleted in Spec 19 Phase 2 — self-contradictory: gated to
        # l_diversity but pipeline contains NOISE (blocked for l_diversity).

    # GEO1: Multi-level geographic QIs — generalize fine-grained geos first
    geo_gran = features.get('geo_qis_by_granularity', {})
    if len(geo_gran) >= 2:
        has_fine = any(g == 'fine' for g in geo_gran.values())
        has_coarse = any(g == 'coarse' for g in geo_gran.values())
        if has_fine and has_coarse:
            fine_geos = [qi for qi, g in geo_gran.items() if g == 'fine']
            return {
                'applies': True,
                'rule': 'GEO1_Multi_Level_Geographic',
                'use_pipeline': True,
                'pipeline': ['GENERALIZE', 'kANON'],
                'parameters': {
                    'GENERALIZE': {
                        'quasi_identifiers': fine_geos,
                        'max_categories': 10,
                        'strategy': 'all',
                    },
                    'kANON': {
                        'quasi_identifiers': qis,
                        'k': 5,
                        'strategy': 'hybrid',
                    },
                },
                'reason': (f"Multi-level geo QIs ({len(geo_gran)} geo columns, "
                           f"fine: {fine_geos}) — generalize fine-grained first"),
                'confidence': 'MEDIUM',
                'priority': 'RECOMMENDED',
            }

    pipeline = []
    params = {}

    # Step 1: Structural protection first (gentler, preserves more)
    if reid_95 > 0.20:
        k = 7 if reid_95 > 0.40 else 5
        pipeline.append('kANON')
        params['kANON'] = {'quasi_identifiers': qis, 'k': k, 'strategy': 'hybrid'}

    # Step 2: NOISE only if no kANON (they're counterproductive together —
    # NOISE disperses values, forcing kANON to generalize harder)
    if n_cont >= 1 and has_outliers and 'kANON' not in pipeline:
        pipeline.append('NOISE')
        params['NOISE'] = {
            'variables': features['continuous_vars'],
            'magnitude': 0.15 if reid_95 <= 0.30 else 0.20,
        }

    # Step 3: Tail cleanup when significant high-risk subgroup
    # Skip if kANON already in pipeline — kANON k>=5 subsumes LOCSUPR k<=5.
    # Exception: very high risk with manageable suppression → add LOCSUPR at k=7.
    if high_risk_rate > 0.15 and reid_95 > 0.15 and 'LOCSUPR' not in pipeline:
        if 'kANON' not in pipeline:
            pipeline.append('LOCSUPR')
            params['LOCSUPR'] = {'quasi_identifiers': qis, 'k': 3}
        elif high_risk_rate > 0.30:
            est_supp_k7 = features.get('estimated_suppression', {}).get(7, 0)
            if est_supp_k7 < 0.40:
                pipeline.append('LOCSUPR')
                params['LOCSUPR'] = {'quasi_identifiers': qis, 'k': 7}

    if len(pipeline) < 2:
        return {'applies': False}

    parts = []
    if n_cont:
        parts.append(f"{n_cont} continuous")
    if n_cat:
        parts.append(f"{n_cat} categorical")
    if has_outliers:
        parts.append("outliers")

    return {
        'applies': True,
        'rule': 'DYN_Pipeline',
        'use_pipeline': True,
        'pipeline': pipeline,
        'parameters': params,
        'reason': f"Dynamic pipeline for {', '.join(parts)} (ReID95={reid_95:.1%})",
        'confidence': 'HIGH',
        'priority': 'REQUIRED',
        'reid_fallback': {
            'method': 'kANON',
            'parameters': {'quasi_identifiers': qis, 'k': 7, 'strategy': 'hybrid'},
        },
        'utility_fallback': {
            'method': 'PRAM',
            'parameters': {
                'variables': top_categorical_qis(features),
                'p_change': 0.15,
            },
        } if n_cat >= 1 else None,
    }


def _legacy_pipeline_rules(features: Dict) -> Dict:
    """Legacy hardcoded pipelines for edge cases not covered by dynamic builder.

    Kept: P5 (small dataset mixed).
    P4a/P4b deleted in Spec 19 Phase 2 — P4a had a latent KeyError crash,
    P4b's |skew| > 1.5 gate was too narrow for any harness dataset.
    """
    # Guard: don't select kANON pipelines when suppression would be catastrophic
    if _is_infeasible(features):
        log.info("[Pipeline] Skipping legacy pipelines — QI space infeasible")
        return {'applies': False}

    qis = features['quasi_identifiers']
    n_continuous = features['n_continuous']
    n_categorical = features['n_categorical']

    # P5: Sparse dataset with mixed variables — NOISE handles continuous, PRAM handles categorical
    # Uses density (records / QI combination space) instead of flat record count.
    # Guard: let HR6 in single-method chain handle very small datasets (< 200 rows).
    n_records = features.get('n_records', 0)
    uniqueness = features.get('uniqueness_rate', 0)
    if n_records < 200:
        pass  # fall through — HR6 handles very small datasets
    else:
        qi_card_product = features.get('qi_cardinality_product', 1)
        density = n_records / qi_card_product if qi_card_product > 0 else float('inf')
        if (density < 5
                and uniqueness > 0.15
                and n_continuous >= 2
                and n_categorical >= 2):
            mag = min(0.25, round(0.10 + (uniqueness - 0.15) * 0.5, 2))
            return {
                'applies': True,
                'rule': 'P5_Small_Dataset_Mixed_Risks',
                'use_pipeline': True,
                'pipeline': ['NOISE', 'PRAM'],
                'parameters': {
                    'NOISE': {'variables': features['continuous_vars'], 'magnitude': mag},
                    'PRAM': {'variables': top_categorical_qis(features), 'p_change': 0.3}
                },
                'reason': (f"Sparse dataset (density={density:.1f}, {n_records} rows) "
                           f"with {uniqueness:.1%} uniqueness and mixed types"),
                'confidence': 'MEDIUM',
                'priority': 'RECOMMENDED'
            }

    return {'applies': False}


def pipeline_rules(features: Dict) -> Dict:
    """Check if data requires multi-method pipeline.

    Tries dynamic builder first, falls back to legacy P4/P5 patterns.

    Parameters:
    -----------
    features : dict
        Output from build_data_features()

    Returns:
    --------
    dict : Pipeline recommendation or {'applies': False}
    """
    result = build_dynamic_pipeline(features)
    if result.get('applies'):
        return result
    return _legacy_pipeline_rules(features)


def _apply_preference_bias(suite: Dict, preference: str) -> Dict:
    """Nudge method parameters based on user preference (post-rule-selection).

    This is a parameter-only adjustment — it does NOT change which rule fires
    or which method is selected.  It mirrors ``_apply_treatment_balance()``
    but is driven by user preference instead of QI treatment counts.

    Parameters
    ----------
    suite : dict
        Output from the rule chain (has 'primary', 'primary_params', etc.).
    preference : str
        ``'structural'`` or ``'perturbative'``.
    """
    suite = dict(suite)  # shallow copy
    primary = suite.get('primary', '')
    params = dict(suite.get('primary_params', {}))

    if preference == 'structural':
        # Same effect as >=60% Heavy treatment: bump k by +2
        if primary == 'kANON' and 'k' in params:
            params['k'] = params['k'] + 2
            log.info("[Preference] structural bias: k %d → %d",
                     params['k'] - 2, params['k'])
        elif primary == 'LOCSUPR' and 'k' in params:
            params['k'] = params['k'] + 2
    elif preference == 'perturbative':
        # Same effect as >=60% Light treatment: reduce k by -1
        if primary == 'kANON' and 'k' in params:
            params['k'] = max(2, params['k'] - 1)
        elif primary == 'PRAM' and 'p_change' in params:
            params['p_change'] = min(0.50, params['p_change'] + 0.05)
            log.info("[Preference] perturbative bias: p_change → %.2f",
                     params['p_change'])
        elif primary == 'NOISE' and 'magnitude' in params:
            params['magnitude'] = min(0.30, params['magnitude'] + 0.03)
            log.info("[Preference] perturbative bias: magnitude → %.2f",
                     params['magnitude'])

    suite['primary_params'] = params
    return suite


def select_method_suite(
    features: Dict,
    access_tier: str = 'SCIENTIFIC',
    verbose: bool = True
) -> Dict:
    """
    Select a suite of methods to try based on data features.

    This function returns a prioritized list of methods with fallbacks,
    suitable for iterative optimization when initial methods don't meet targets.

    Parameters:
    -----------
    features : dict
        Output from build_data_features()
    access_tier : str
        Target access tier: 'PUBLIC', 'SCIENTIFIC', or 'SECURE'
    verbose : bool
        Print selection details

    Returns:
    --------
    dict : {
        'primary': str,           # Best method to try first
        'primary_params': dict,   # Parameters for primary method
        'fallbacks': list,        # [(method, params), ...] ordered by priority
        'reid_fallback': dict,     # Stronger method if ReID target not met
        'utility_fallback': dict, # Weaker method if utility target not met
        'pipeline': list or None, # Multi-method pipeline if needed
        'rule_applied': str,      # Which rule triggered selection
        'confidence': str,        # HIGH, MEDIUM, LOW
        'reason': str,            # Explanation
    }

    Example:
    --------
    >>> from sdc_engine.sdc.protection_engine import build_data_features
    >>> from sdc_engine.sdc.selection import select_method_suite
    >>> features = build_data_features(data, qis)
    >>> suite = select_method_suite(features, access_tier='PUBLIC')
    >>> print(f"Try {suite['primary']} first, fallbacks: {[m[0] for m in suite['fallbacks']]}")
    """
    from .rules import (
        data_structure_rules,
        small_dataset_rules,
        structural_risk_rules,
        risk_concentration_rules,
        regulatory_compliance_rules,
        public_release_rules,
        categorical_aware_rules,
        l_diversity_rules,
        temporal_dominant_rules,
        secure_environment_rules,
        reid_risk_rules,
        low_risk_rules,
        distribution_rules,
        uniqueness_risk_rules,
        default_rules,
        _apply_treatment_balance,
    )

    from ..config import is_method_allowed_for_metric

    # Propagate context into features so context-aware rules can read it
    features['_access_tier'] = access_tier.upper()

    qis = features.get('quasi_identifiers', [])
    reid_95 = features.get('reid_95', 0.5)
    risk_metric = features.get('_risk_metric_type', 'reid95')

    # --- User method constraints ---
    _constraints = features.get('method_constraints') or {}
    _excluded = set(_constraints.get('excluded', []))
    _preference = _constraints.get('preference', 'auto')

    log.info("[MethodSuite] Selecting: n_qis=%d  ReID95=%.4f  tier=%s  "
             "metric=%s  n_cont=%d  n_cat=%d  has_outliers=%s  uniqueness=%.2f%%"
             "  excluded=%s  preference=%s",
             features.get('n_qis', 0), reid_95, access_tier, risk_metric,
             features.get('n_continuous', 0), features.get('n_categorical', 0),
             features.get('has_outliers', False),
             features.get('uniqueness_rate', 0) * 100,
             _excluded or '{}', _preference)

    # ------------------------------------------------------------------
    # Helper: check whether every method in a list is allowed for the
    # active risk metric AND not user-excluded.
    # ------------------------------------------------------------------
    def _all_allowed(methods):
        return all(
            is_method_allowed_for_metric(m, risk_metric) and m not in _excluded
            for m in methods
        )

    def _is_allowed(method):
        return is_method_allowed_for_metric(method, risk_metric) and method not in _excluded

    # ------------------------------------------------------------------
    # Spec 16 audit trace — when SDC_RULE_AUDIT=1, evaluate all rules
    # (no short-circuit) and attach _rule_trace to the result.
    # When off, behaviour is byte-identical to pre-instrumentation.
    # ------------------------------------------------------------------
    _audit = _RULE_AUDIT
    _trace = [] if _audit else None  # None = audit off, list = collecting
    _winner = None  # first matching result (what we'd return normally)

    # Pass audit flag to rule functions via features so they can return
    # sub-rule-level trace data without importing the flag themselves.
    if _audit:
        features['_audit'] = True

    def _record_trace(rule_name, applies, method_or_pipeline, *,
                      blocked=False, blocked_reason='',
                      sub_rules=None):
        """Append one entry to the audit trace (no-op when audit off)."""
        if _trace is None:
            return
        entry = {
            'rule': rule_name,
            'applies': applies,
            'method': method_or_pipeline,
        }
        if blocked:
            entry['blocked'] = True
            entry['blocked_reason'] = blocked_reason
        if sub_rules:
            entry['sub_rules'] = sub_rules
        _trace.append(entry)

    # Check pipeline rules first
    pipeline_result = pipeline_rules(features)
    _pipeline_applies = (pipeline_result.get('applies')
                         and _all_allowed(pipeline_result.get('pipeline', [])))

    if pipeline_result.get('applies') and not _all_allowed(pipeline_result.get('pipeline', [])):
        _record_trace(pipeline_result.get('rule', 'PIPELINE'),
                      True, pipeline_result.get('pipeline'),
                      blocked=True, blocked_reason=f'metric={risk_metric}')
    elif pipeline_result.get('applies'):
        _record_trace(pipeline_result.get('rule', 'PIPELINE'),
                      True, pipeline_result.get('pipeline'))
    else:
        _record_trace('PIPELINE', False, None)

    if _pipeline_applies:
        log.info("[MethodSuite] Pipeline match: %s -> %s",
                 pipeline_result['rule'], pipeline_result['pipeline'])
        # Use rule-level fallbacks if provided, otherwise default to kANON k=7
        reid_fb = pipeline_result.get('reid_fallback',
                     {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 7}})
        util_fb = pipeline_result.get('utility_fallback')

        primary = pipeline_result['pipeline'][0]
        primary_params = dict(pipeline_result['parameters'].get(primary, {}))
        fb_list = [(m, dict(pipeline_result['parameters'].get(m, {})))
                   for m in pipeline_result['pipeline'][1:]]

        # Apply treatment balance to pipeline params (BUG-7 fix)
        qi_treatment = features.get('qi_treatment')
        if qi_treatment:
            _temp = {'method': primary, 'parameters': primary_params}
            _temp = _apply_treatment_balance(_temp, qi_treatment)
            primary_params = _temp['parameters']
            adjusted_fbs = []
            for fb_m, fb_p in fb_list:
                _t = {'method': fb_m, 'parameters': fb_p}
                _t = _apply_treatment_balance(_t, qi_treatment)
                adjusted_fbs.append((fb_m, _t['parameters']))
            fb_list = adjusted_fbs

        pipeline_suite = {
            'primary': primary,
            'primary_params': primary_params,
            'fallbacks': fb_list,
            'reid_fallback': reid_fb,
            'utility_fallback': util_fb,
            'pipeline': pipeline_result['pipeline'],
            'rule_applied': pipeline_result['rule'],
            'confidence': pipeline_result['confidence'],
            'reason': pipeline_result['reason'],
            'use_pipeline': True,
        }
        # Apply user preference bias (parameter nudge)
        if _preference != 'auto':
            pipeline_suite = _apply_preference_bias(pipeline_suite, _preference)

        if not _audit:
            return pipeline_suite
        _winner = pipeline_suite

    # Apply rules in priority order (first match wins) — lazy evaluation
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
        rule = rule_fn(features)
        if rule.get('applies'):
            # Apply treatment balance post-adjustment
            rule = _apply_treatment_balance(rule, features.get('qi_treatment'))

            # Handle pipeline-format rules from the single-method chain
            if rule.get('use_pipeline'):
                pipeline = rule['pipeline']
                # Skip entire pipeline if any method is blocked for this metric
                if not _all_allowed(pipeline):
                    log.info("[MethodSuite] Skipping pipeline rule %s — "
                             "blocked by metric %s", rule.get('rule', '?'), risk_metric)
                    _record_trace(rule.get('rule', '?'), True, pipeline,
                                  blocked=True, blocked_reason=f'metric={risk_metric}',
                                  sub_rules=rule.get('_sub_rule_trace'))
                    continue
                _record_trace(rule.get('rule', '?'), True, pipeline,
                              sub_rules=rule.get('_sub_rule_trace'))
                if _winner is not None:
                    continue  # audit mode: already have winner, just recording

                pipeline_params = rule.get('parameters', {})
                log.info("[MethodSuite] Inline pipeline from rule: %s -> %s",
                         rule.get('rule', '?'), pipeline)
                reid_fb = rule.get('reid_fallback',
                             {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 7}})
                util_fb = rule.get('utility_fallback')
                # Nullify fallbacks whose method is blocked or excluded
                if reid_fb and not _is_allowed(reid_fb['method']):
                    reid_fb = None
                if util_fb and not _is_allowed(util_fb['method']):
                    util_fb = None
                _result = {
                    'primary': pipeline[0],
                    'primary_params': pipeline_params.get(pipeline[0], {}),
                    'fallbacks': [(m, pipeline_params.get(m, {})) for m in pipeline[1:]],
                    'reid_fallback': reid_fb,
                    'utility_fallback': util_fb,
                    'pipeline': pipeline,
                    'rule_applied': rule.get('rule', 'UNKNOWN'),
                    'confidence': rule.get('confidence', 'MEDIUM'),
                    'reason': rule.get('reason', ''),
                    'use_pipeline': True,
                }
                if not _audit:
                    return _result
                _winner = _result
                continue

            # Skip single-method rule if its primary is blocked or excluded
            if not _is_allowed(rule.get('method', '')):
                log.info("[MethodSuite] Skipping rule %s (method %s) — "
                         "blocked by metric %s or user-excluded",
                         rule.get('rule', '?'), rule.get('method', '?'), risk_metric)
                _record_trace(rule.get('rule', '?'), True, rule.get('method'),
                              blocked=True, blocked_reason=f'metric={risk_metric}',
                              sub_rules=rule.get('_sub_rule_trace'))
                continue

            _record_trace(rule.get('rule', '?'), True, rule.get('method'),
                          sub_rules=rule.get('_sub_rule_trace'))
            if _winner is not None:
                continue  # audit mode: already have winner, just recording

            log.info("[MethodSuite] Rule match: %s -> %s  confidence=%s  reason=%s",
                     rule.get('rule', '?'), rule.get('method', '?'),
                     rule.get('confidence', '?'),
                     rule.get('reason', '')[:100])
            # Build fallbacks list
            fallbacks = []

            # Add ReID fallback if available and allowed
            reid_fb = rule.get('reid_fallback')
            if reid_fb and not _is_allowed(reid_fb['method']):
                reid_fb = None
            if reid_fb:
                fallbacks.append((reid_fb['method'], reid_fb.get('parameters', {})))

            # Add utility fallback if available and allowed
            util_fb = rule.get('utility_fallback')
            if util_fb and not _is_allowed(util_fb['method']):
                util_fb = None
            if util_fb:
                fallbacks.append((util_fb['method'], util_fb.get('parameters', {})))

            # Add alternatives (filtered by metric + user exclusion)
            for alt in rule.get('alternatives', []):
                if _is_allowed(alt):
                    fallbacks.append((alt, {}))

            # Add tier-specific fallbacks (filtered, deduplicated)
            def _fb_key(method, params):
                return (method, params.get('k'), params.get('p_change'),
                        params.get('magnitude'))

            existing_keys = {_fb_key(m, p) for m, p in fallbacks}

            if access_tier.upper() == 'PUBLIC' and reid_95 > 0.05:
                for fb_entry in [
                    ('kANON', {'quasi_identifiers': qis, 'k': 7}),
                    ('kANON', {'quasi_identifiers': qis, 'k': 10}),
                ]:
                    if _fb_key(*fb_entry) not in existing_keys:
                        fallbacks.append(fb_entry)
                        existing_keys.add(_fb_key(*fb_entry))
            elif access_tier.upper() == 'SECURE' and reid_95 < 0.20:
                if _is_allowed('PRAM'):
                    fb = ('PRAM', {'variables': top_categorical_qis(features), 'p_change': 0.15})
                    if _fb_key(*fb) not in existing_keys:
                        fallbacks.append(fb)
                if _is_allowed('NOISE'):
                    fb = ('NOISE', {'variables': features.get('continuous_vars', []), 'magnitude': 0.10})
                    if _fb_key(*fb) not in existing_keys:
                        fallbacks.append(fb)

            if verbose:
                print(f"\n[Method Suite] Rule: {rule.get('rule', 'UNKNOWN')}")
                print(f"  Primary: {rule['method']}")
                print(f"  Fallbacks: {[f[0] for f in fallbacks[:5]]}")
                print(f"  Confidence: {rule.get('confidence', 'MEDIUM')}")
                print(f"  Reason: {rule.get('reason', 'N/A')}")

            _result = {
                'primary': rule['method'],
                'primary_params': rule.get('parameters', {}),
                'fallbacks': fallbacks,
                'reid_fallback': reid_fb,
                'utility_fallback': util_fb,
                'pipeline': None,
                'rule_applied': rule.get('rule', 'UNKNOWN'),
                'confidence': rule.get('confidence', 'MEDIUM'),
                'reason': rule.get('reason', 'Rule-based selection'),
                'use_pipeline': False,
            }
            # Propagate preprocessing hint from risk-concentration rules
            if rule.get('preprocessing_hint'):
                _result['preprocessing_hint'] = rule['preprocessing_hint']
            # Apply user preference bias (parameter nudge, not rule override)
            if _preference != 'auto':
                _result = _apply_preference_bias(_result, _preference)
            if not _audit:
                return _result
            _winner = _result
            continue

        # Rule didn't apply — still capture sub-rule trace if available
        _record_trace(rule_fn.__name__, False, None,
                      sub_rules=rule.get('_sub_rule_trace'))

    # If we reach here with a winner (audit mode), attach trace and return
    if _winner is not None:
        _winner['_rule_trace'] = _trace
        return _winner

    # Emergency fallback
    _fallback = {
        'primary': 'kANON',
        'primary_params': {'quasi_identifiers': qis, 'k': 5},
        'fallbacks': [
            ('kANON', {'quasi_identifiers': qis, 'k': 3}),
            ('kANON', {'quasi_identifiers': qis, 'k': 7}),
            ('LOCSUPR', {'quasi_identifiers': qis, 'k': 3}),
        ],
        'reid_fallback': {'method': 'kANON', 'parameters': {'quasi_identifiers': qis, 'k': 10}},
        'utility_fallback': None,
        'pipeline': None,
        'rule_applied': 'EMERGENCY_FALLBACK',
        'confidence': 'LOW',
        'reason': 'No specific rule matched - using default k-anonymity',
        'use_pipeline': False,
    }
    if _audit:
        _fallback['_rule_trace'] = _trace
    return _fallback
