"""Parametrized boundary dataset builders for routing validation.

Build synthetic datasets paired with controlled var_priority values that
produce specific classify_risk_concentration() patterns for testing
whether the rules engine routes each pattern to the correct method.

Scope limitation (documented):
    These builders test the ROUTING layer (classify → rule → method → outcome),
    NOT threshold calibration. The four classify_risk_concentration rules
    interact: below the 40% dominated cutoff, whether a dataset classifies as
    'balanced' or 'spread_high' depends on the n_high >= 3 rule. The 40%
    threshold cannot be evaluated in isolation without controlling the others.
    See Family 1 report for the threshold interaction finding.

Approach: organic var_priority computation via leave-one-out reid cannot
produce arbitrary contribution percentages (contributions are non-additive
and constrained by combinatorial geometry). Instead, we build datasets
with realistic reid_95 > 0.15 and construct var_priority dicts directly
with the desired contribution profile. This is the same pattern used in
test_rule_selection_known_cases.py for RC rule testing.

Secondary QI contributions are kept below 15% (HIGH threshold) to avoid
triggering spread_high (n_high >= 3) on datasets intended for 'balanced'.
This is a deliberate constraint, not a calibration test — see the report
for discussion of what organic spread_high at top_pct=30% means.

Usage:
    python -m tests.empirical.sensitivity.builders
"""
from __future__ import annotations

import logging
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset builder — produces a dataset with realistic reid_95
# ---------------------------------------------------------------------------

def _build_base_dataset(
    n_records: int = 5000,
    n_qis: int = 4,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build a base dataset with reid_95 in a moderate range (0.2-0.4).

    Targets combo_space ≈ n_records/6 so that reid_95 falls in a range
    where protection methods can make meaningful progress and the reid
    target (0.05) is achievable, unlike reid_95=1.0 where no method
    can reach the target.
    """
    rng = np.random.default_rng(seed)

    # Target: combo_space ≈ n_records/6 for reid_95 ≈ 0.2-0.4
    # Empirically calibrated: n=5000 with cards=[10,5,4,4] gives reid_95≈0.33
    # First QI gets 2x cardinality (slightly dominant)
    # With n_qis=4: combo = 2b * b^3 = 2*b^4
    # Solve 2*b^4 = n_records/6 → b = (n_records/12)^(1/4)
    base_card = max(3, int((n_records / 12) ** (1.0 / n_qis)) + 1)
    cards = [base_card * 2] + [base_card] * (n_qis - 1)

    cols = {}
    qis = []
    for i, card in enumerate(cards):
        name = f"qi_{i}"
        vals = [f"v{i}_{j}" for j in range(card)]
        cols[name] = rng.choice(vals, n_records)
        qis.append(name)

    return pd.DataFrame(cols), qis


def _verify_reid(df: pd.DataFrame, qis: List[str]) -> float:
    """Compute reid_95 for the dataset."""
    from sdc_engine.sdc.metrics.reid import calculate_reid
    return calculate_reid(df, qis).get('reid_95', 0)


# ---------------------------------------------------------------------------
# var_priority constructors — create controlled contribution profiles
# ---------------------------------------------------------------------------

def _make_var_priority_dominated(
    qis: List[str],
    top_pct: float,
) -> Dict[str, tuple]:
    """Create var_priority where the first QI has top_pct contribution.

    Secondary QIs get LOW contributions (< 15%) to avoid triggering
    the 'spread_high' pattern (n_high >= 3).
    """
    vp = {}
    for i, qi in enumerate(qis):
        if i == 0:
            pct = top_pct
        else:
            # Keep secondaries under HIGH threshold (15%) to avoid
            # accidentally triggering spread_high (n_high >= 3)
            pct = 10.0 - i * 1.0  # e.g., 9.0, 8.0, 7.0, ...
            pct = max(1.0, pct)
        label = _pct_to_label(pct)
        vp[qi] = (label, round(pct, 1))

    return vp


def _make_var_priority_concentrated(
    qis: List[str],
    top2_pct: float,
) -> Dict[str, tuple]:
    """Create var_priority where top-2 QIs sum to top2_pct.

    Split roughly equally between first two QIs, neither exceeding 39%
    (to avoid triggering 'dominated' instead of 'concentrated').
    Secondary QIs get LOW contributions (< 15%) to avoid 'spread_high'.
    """
    # Split top2 contribution: aim for near-equal, cap at 39%
    qi_0_pct = min(39.0, top2_pct / 2 + 1.0)  # slightly asymmetric
    qi_1_pct = top2_pct - qi_0_pct
    # Safety: neither should reach 40%
    if qi_0_pct >= 40:
        qi_0_pct = 39.0
        qi_1_pct = top2_pct - 39.0
    if qi_1_pct >= 40:
        qi_1_pct = 39.0
        qi_0_pct = top2_pct - 39.0

    vp = {}
    for i, qi in enumerate(qis):
        if i == 0:
            pct = qi_0_pct
        elif i == 1:
            pct = qi_1_pct
        else:
            # Keep under HIGH threshold (15%) to avoid spread_high
            pct = 8.0 - (i - 2) * 1.0
            pct = max(1.0, pct)
        label = _pct_to_label(pct)
        vp[qi] = (label, round(pct, 1))

    return vp


def _pct_to_label(pct: float) -> str:
    """Map contribution percentage to var_priority label."""
    if pct >= 15:
        return 'HIGH'
    elif pct >= 8:
        return 'MED-HIGH'
    elif pct >= 3:
        return 'MODERATE'
    else:
        return 'LOW'


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_concentration_boundary(
    target_top_pct: float,
    n_qis: int = 4,
    n_records: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Build a dataset + var_priority where top QI contributes target_top_pct.

    The dataset has realistic reid_95 > 0.15. The var_priority dict is
    constructed directly (not via leave-one-out) with the desired
    contribution profile.

    Returns (DataFrame, qi_names, metadata) where metadata includes:
        var_priority, top_pct, top2_pct, pattern, reid_95, risk_concentration
    """
    from sdc_engine.sdc.selection.features import classify_risk_concentration

    df, qis = _build_base_dataset(n_records, n_qis, seed)
    reid_95 = _verify_reid(df, qis)

    # If reid_95 is too low, increase cardinality
    if reid_95 <= 0.15:
        df, qis = _build_base_dataset(n_records, n_qis + 1, seed)
        reid_95 = _verify_reid(df, qis)

    vp = _make_var_priority_dominated(qis, target_top_pct)
    conc = classify_risk_concentration(vp)

    sorted_vp = sorted(vp.items(), key=lambda x: x[1][1], reverse=True)
    actual_top = sorted_vp[0][1][1]
    actual_top2 = sum(pct for _, (_, pct) in sorted_vp[:2])

    return df, qis, {
        'var_priority': vp,
        'risk_concentration': conc,
        'top_pct': actual_top,
        'top2_pct': actual_top2,
        'pattern': conc['pattern'],
        'reid_95': reid_95,
    }


def build_top2_concentration_boundary(
    target_top2_pct: float,
    n_qis: int = 4,
    n_records: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Build a dataset + var_priority where top-2 QIs sum to target_top2_pct.

    Neither heavy QI exceeds 39% individually (avoids 'dominated' pattern).

    Returns (DataFrame, qi_names, metadata).
    """
    from sdc_engine.sdc.selection.features import classify_risk_concentration

    df, qis = _build_base_dataset(n_records, n_qis, seed)
    reid_95 = _verify_reid(df, qis)

    if reid_95 <= 0.15:
        df, qis = _build_base_dataset(n_records, n_qis + 1, seed)
        reid_95 = _verify_reid(df, qis)

    vp = _make_var_priority_concentrated(qis, target_top2_pct)
    conc = classify_risk_concentration(vp)

    sorted_vp = sorted(vp.items(), key=lambda x: x[1][1], reverse=True)
    actual_top = sorted_vp[0][1][1]
    actual_top2 = sum(pct for _, (_, pct) in sorted_vp[:2])

    return df, qis, {
        'var_priority': vp,
        'risk_concentration': conc,
        'top_pct': actual_top,
        'top2_pct': actual_top2,
        'pattern': conc['pattern'],
        'reid_95': reid_95,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_builders(verbose: bool = True) -> Dict[str, bool]:
    """Acceptance test: verify builders produce correct patterns at boundaries.

    Tests:
    - build_concentration_boundary at [30, 35, 40, 45, 50]
    - build_top2_concentration_boundary at [50, 55, 60, 65]
    - Each with seeds [42, 123, 999]

    Checks:
    - top_pct / top2_pct matches target exactly (constructed, not organic)
    - classify_risk_concentration produces expected pattern at each boundary
    - reid_95 > 0.15 (RC gate condition)

    Returns dict mapping test label -> pass/fail boolean.
    """
    TOP_PCT_TARGETS = [30.0, 35.0, 40.0, 45.0, 50.0]
    TOP2_PCT_TARGETS = [50.0, 55.0, 60.0, 65.0]
    SEEDS = [42, 123, 999]

    results: Dict[str, bool] = {}

    if verbose:
        print(f"{'Test':<35} {'Target':>7} {'Actual':>7} "
              f"{'reid95':>7} {'Pattern':<14} {'Result'}")
        print("-" * 85)

    # Expected patterns at boundary
    # top_pct >= 40 → dominated, < 40 → not_dominated
    # (RC2/RC3/RC4 deleted — only dominated vs not_dominated remains)
    expected_top_pct_pattern = {
        30.0: 'not_dominated', 35.0: 'not_dominated',
        40.0: 'dominated', 45.0: 'dominated', 50.0: 'dominated',
    }

    for target in TOP_PCT_TARGETS:
        for seed in SEEDS:
            label = f"top_pct={target:.0f}_seed={seed}"
            try:
                df, qis, meta = build_concentration_boundary(target, seed=seed)
                actual = meta['top_pct']
                reid_ok = meta['reid_95'] > 0.15
                pattern_expected = expected_top_pct_pattern[target]
                pattern_ok = meta['pattern'] == pattern_expected
                ok = abs(actual - target) < 0.1 and reid_ok and pattern_ok
                results[label] = ok
                if verbose:
                    status = "PASS" if ok else "FAIL"
                    extra = ""
                    if not reid_ok:
                        extra += " LOW_REID"
                    if not pattern_ok:
                        extra += f" WRONG_PATTERN(exp={pattern_expected})"
                    print(f"  {label:<33} {target:>6.1f}% {actual:>6.1f}% "
                          f"{meta['reid_95']:>6.3f} {meta['pattern']:<14} "
                          f"{status}{extra}")
            except Exception as e:
                results[label] = False
                if verbose:
                    print(f"  {label:<33} {target:>6.1f}%    ERROR: {e}")

    # top2_pct patterns — with RC2/RC3/RC4 deleted, all non-dominated
    # patterns map to 'not_dominated' regardless of top2_pct
    expected_top2_pattern = {
        50.0: 'not_dominated', 55.0: 'not_dominated',
        60.0: 'not_dominated', 65.0: 'not_dominated',
    }

    for target in TOP2_PCT_TARGETS:
        for seed in SEEDS:
            label = f"top2_pct={target:.0f}_seed={seed}"
            try:
                df, qis, meta = build_top2_concentration_boundary(target, seed=seed)
                actual = meta['top2_pct']
                reid_ok = meta['reid_95'] > 0.15
                pattern_expected = expected_top2_pattern[target]
                pattern_ok = meta['pattern'] == pattern_expected
                ok = abs(actual - target) < 0.5 and reid_ok and pattern_ok
                results[label] = ok
                if verbose:
                    status = "PASS" if ok else "FAIL"
                    extra = ""
                    if not reid_ok:
                        extra += " LOW_REID"
                    if not pattern_ok:
                        extra += f" WRONG_PATTERN(exp={pattern_expected})"
                    print(f"  {label:<33} {target:>6.1f}% {actual:>6.1f}% "
                          f"{meta['reid_95']:>6.3f} {meta['pattern']:<14} "
                          f"{status}{extra}")
            except Exception as e:
                results[label] = False
                if verbose:
                    print(f"  {label:<33} {target:>6.1f}%    ERROR: {e}")

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    if verbose:
        print(f"\nSummary: {n_pass}/{n_total} passed")
        if n_pass < n_total:
            failed = [k for k, v in results.items() if not v]
            print(f"  Failed: {', '.join(failed)}")

    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    results = verify_builders(verbose=True)
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    sys.exit(0 if n_pass == n_total else 1)
