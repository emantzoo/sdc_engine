"""
Feature Utilities for Method Selection
=======================================

Risk-concentration classification and categorical QI helpers.

The main feature-extraction function is ``build_data_features()`` in
``protection_engine.py``.  ``extract_data_features_with_reid()`` was
deleted in Spec 19 Phase 1.5.
"""

from typing import Dict, List, Optional


def classify_risk_concentration(var_priority: Optional[Dict] = None) -> Dict:
    """Classify how backward-elimination risk is distributed across QIs.

    Only the 'dominated' pattern (top QI >= 40%) is consumed by production
    rules (RC1).  RC2/RC3/RC4 were deleted in Spec 19 Phase 2 — their
    patterns ('concentrated', 'spread_high', 'balanced') are no longer used
    by any rule gate.

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

    pattern = 'dominated' if top_pct >= 40 else 'not_dominated'

    return {
        'pattern': pattern,
        'top_qi': top_qi,
        'top_pct': top_pct,
        'top2_pct': top2_pct,
        'n_high_risk': n_high,
    }


# extract_data_features_with_reid was deleted in Spec 19 Phase 1.5.
# All callers now use build_data_features() from protection_engine.py.


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
