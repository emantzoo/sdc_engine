"""
Per-QI Treatment Levels
=======================

Centralized module for mapping user-defined treatment levels
(Heavy / Standard / Light) to per-variable parameter dicts
consumed by SDC protection and preprocessing methods.

Each QI column can have a different treatment level, allowing
riskier QIs to receive more aggressive protection while safer
QIs are treated gently — improving overall utility.
"""

from typing import Dict, List, Optional, Tuple

# ─── Treatment level multipliers ───────────────────────────────
TREATMENT_MULTIPLIERS = {
    'Heavy': 1.5,
    'Standard': 1.0,
    'Light': 0.5,
}

# ─── Auto-fill: risk priority → treatment level ───────────────
PRIORITY_TO_TREATMENT = {
    '🔴 HIGH': 'Heavy',
    '🟠 MED-HIGH': 'Heavy',
    '🟡 MODERATE': 'Light',
    '⚪ LOW': 'Light',
}

# ─── Per-method parameter info (key name + valid range) ────────
METHOD_PARAM_INFO = {
    'NOISE':    {'param': 'per_variable_magnitude', 'base_key': 'magnitude',
                 'min': 0.01, 'max': 0.50},
    'PRAM':     {'param': 'per_variable_p_change',  'base_key': 'p_change',
                 'min': 0.05, 'max': 0.50},
    'KANON':    {'param': 'per_qi_bin_size',         'base_key': 'bin_size',
                 'min': 2,    'max': 50, 'round': True},
}

# ─── LOCSUPR importance weights by treatment ───────────────────
_LOCSUPR_WEIGHT_MAP = {
    'Heavy': 1,      # low weight → suppressed MORE
    'Standard': 3,
    'Light': 5,      # high weight → preserved MORE
}


def priority_to_treatment(priority_label: str) -> str:
    """Map a risk priority label (e.g. '🔴 HIGH') to a treatment level."""
    return PRIORITY_TO_TREATMENT.get(priority_label, 'Standard')


def build_per_variable_params(
    method: str,
    base_value: float,
    qi_list: List[str],
    qi_treatment: Dict[str, str],
) -> Optional[Dict]:
    """Build ``{col: scaled_value}`` dict for a method's main parameter.

    Returns ``None`` when all QIs have Standard treatment (uniform —
    no per-variable dict needed, keeps current behaviour).
    """
    info = METHOD_PARAM_INFO.get(method.upper())
    if not info:
        return None

    result = {}
    all_standard = True
    for qi in qi_list:
        level = qi_treatment.get(qi, 'Standard')
        mult = TREATMENT_MULTIPLIERS.get(level, 1.0)
        if mult != 1.0:
            all_standard = False
        val = base_value * mult
        # Clamp to valid range
        val = max(info['min'], min(info['max'], val))
        if info.get('round'):
            val = int(round(val))
        result[qi] = val

    return None if all_standard else result


def build_locsupr_weights(
    qi_list: List[str],
    qi_treatment: Dict[str, str],
) -> Optional[Dict[str, int]]:
    """Build ``importance_weights`` for LOCSUPR from treatment levels.

    Heavy → weight 1 (suppressed more),
    Standard → weight 3,
    Light → weight 5 (preserved more).

    Returns ``None`` when all Standard.
    """
    weights = {}
    all_standard = True
    for qi in qi_list:
        level = qi_treatment.get(qi, 'Standard')
        if level != 'Standard':
            all_standard = False
        weights[qi] = _LOCSUPR_WEIGHT_MAP.get(level, 3)
    return None if all_standard else weights


def get_method_base_defaults() -> Dict[str, float]:
    """Return the default base parameter value per method."""
    return {
        'NOISE': 0.10,
        'PRAM': 0.20,
        'KANON': 10,
    }


# ─── Preprocessing parameter clamp ranges ─────────────────────
PREPROCESS_CLAMP = {
    'bottom_percentile': (0.1, 10),
    'top_percentile': (90, 99.9),
    'min_frequency': (1, 50),
}

# ─── GENERALIZE utility gate threshold multipliers ─────────────
TREATMENT_GATE_MULT = {
    'Heavy': 0.75,      # more lenient — accepts more utility loss
    'Standard': 1.0,
    'Light': 1.25,      # stricter — demands better preservation
}


def build_per_qi_percentiles(
    qi_list: List[str],
    qi_treatment: Dict[str, str],
    base_bottom: float = 1,
    base_top: float = 99,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """Build ``{col: (bottom_pctile, top_pctile)}`` scaled by treatment.

    Heavy → tighter percentiles (narrows the kept range),
    Light → wider percentiles (expands the kept range).

    Formula:
        bottom = base_bottom + base_bottom * (mult - 1)
        top    = base_top - (100 - base_top) * (mult - 1)

    Returns ``None`` when all QIs have Standard treatment.
    """
    lo_clamp = PREPROCESS_CLAMP['bottom_percentile']
    hi_clamp = PREPROCESS_CLAMP['top_percentile']

    result = {}
    all_standard = True
    for qi in qi_list:
        level = qi_treatment.get(qi, 'Standard')
        mult = TREATMENT_MULTIPLIERS.get(level, 1.0)
        if mult != 1.0:
            all_standard = False

        bot = base_bottom + base_bottom * (mult - 1.0)
        top = base_top - (100.0 - base_top) * (mult - 1.0)

        # Clamp to valid ranges
        bot = max(lo_clamp[0], min(lo_clamp[1], bot))
        top = max(hi_clamp[0], min(hi_clamp[1], top))

        result[qi] = (round(bot, 2), round(top, 2))

    return None if all_standard else result


def build_per_qi_min_frequency(
    qi_list: List[str],
    qi_treatment: Dict[str, str],
    base_min_frequency: int = 10,
) -> Optional[Dict[str, int]]:
    """Build ``{col: min_freq}`` scaled by treatment, clamped [1, 50].

    Heavy → higher min_frequency (merge more rare categories),
    Light → lower min_frequency (preserve more categories).

    Returns ``None`` when all QIs have Standard treatment.
    """
    lo, hi = PREPROCESS_CLAMP['min_frequency']

    result = {}
    all_standard = True
    for qi in qi_list:
        level = qi_treatment.get(qi, 'Standard')
        mult = TREATMENT_MULTIPLIERS.get(level, 1.0)
        if mult != 1.0:
            all_standard = False

        val = int(round(base_min_frequency * mult))
        val = max(lo, min(hi, val))
        result[qi] = val

    return None if all_standard else result


def get_adaptive_binning_candidates(
    effective_max: int,
    card: int,
    treatment_level: str = 'Standard',
) -> List[int]:
    """Return 3 candidate bin counts shifted by treatment level.

    Each treatment explores a different part of the binning spectrum:

    * Heavy  → aggressive end: ``[max(2, eff//3), eff//2, eff]``
    * Standard → current:      ``[eff//2, eff, min(card, int(eff*1.5))]``
    * Light  → gentle end:     ``[eff, min(card, int(eff*1.5)), min(card, eff*2)]``

    Candidates are sorted descending (gentlest first, most bins).
    """
    if treatment_level == 'Heavy':
        raw = [
            max(2, effective_max // 3),
            max(2, effective_max // 2),
            effective_max,
        ]
    elif treatment_level == 'Light':
        raw = [
            effective_max,
            min(card, int(effective_max * 1.5)),
            min(card, effective_max * 2),
        ]
    else:  # Standard — matches current behaviour
        raw = [
            max(3, effective_max // 2),
            effective_max,
            min(card, int(effective_max * 1.5)),
        ]

    # Deduplicate and sort descending (gentlest = most bins first)
    return sorted(set(c for c in raw if c >= 2), reverse=True)
