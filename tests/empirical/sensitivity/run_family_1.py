"""Family 1 — Risk Concentration Routing Validation.

Tests whether classify_risk_concentration() patterns are correctly routed
to the appropriate protection methods by the rules engine.

Scope: This tests the ROUTING layer (classify -> rule -> method -> outcome).
It does NOT test threshold calibration — the four classify_risk_concentration
rules interact (dominated/concentrated/spread_high/balanced), so the 40%
threshold cannot be swept in isolation. See builders.py docstring and the
final report for the threshold interaction finding.

Steps:
    1. Generate 27 boundary datasets (builders.py)
    2. Run 5 counterfactual paths per dataset (135 runs)
    3. Compare outcomes (which path produces best reid/utility?)
    4. Cross-metric regression (only if a consistent crossover is found)

RC4 is NOT tested — it's gated by n_high_risk, not top_pct/top2_pct,
and its GENERALIZE->kANON pipeline is architecturally distinct from the
single-method RC1/RC2/RC3 paths.

Usage:
    python -m tests.empirical.sensitivity.run_family_1
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SensitivityOutcome:
    dataset_id: str           # "top_pct=40_seed=42"
    boundary_type: str        # 'top_pct' or 'top2_pct'
    target_value: float
    actual_value: float
    seed: int
    counterfactual: str       # 'natural', 'dominated', 'concentrated', 'spread_high', 'balanced'
    natural_pattern: str      # what classify_risk_concentration actually returns
    forced_pattern: str       # what we injected (same as natural for 'natural' runs)
    rule_applied: str
    method_selected: str
    reid_before: float
    reid_after: float
    utility_score: float
    suppression_rate: float
    n_iterations: int
    target_met: bool
    elapsed_sec: float
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_PCT_TARGETS = [30.0, 35.0, 40.0, 45.0, 50.0]
TOP2_PCT_TARGETS = [50.0, 55.0, 60.0, 65.0]
SEEDS = [42, 123, 999]
COUNTERFACTUALS = ['natural', 'dominated', 'concentrated', 'spread_high', 'balanced']

RISK_METRIC = 'reid95'
RISK_TARGET = 0.05
UTILITY_FLOOR = 0.80


# ---------------------------------------------------------------------------
# Dataset generation (Step 1)
# ---------------------------------------------------------------------------

def generate_datasets() -> List[Dict]:
    """Generate 27 boundary datasets via builders.

    Returns list of dicts: {id, boundary_type, target, seed, df, qis, meta}
    """
    from tests.empirical.sensitivity.builders import (
        build_concentration_boundary,
        build_top2_concentration_boundary,
    )

    datasets = []

    for target in TOP_PCT_TARGETS:
        for seed in SEEDS:
            ds_id = f"top_pct={target:.0f}_seed={seed}"
            try:
                df, qis, meta = build_concentration_boundary(target, seed=seed)
                datasets.append({
                    'id': ds_id,
                    'boundary_type': 'top_pct',
                    'target': target,
                    'seed': seed,
                    'df': df,
                    'qis': qis,
                    'meta': meta,
                })
            except Exception as e:
                log.error("Builder failed for %s: %s", ds_id, e)
                datasets.append({
                    'id': ds_id,
                    'boundary_type': 'top_pct',
                    'target': target,
                    'seed': seed,
                    'df': None,
                    'qis': None,
                    'meta': None,
                    'error': str(e),
                })

    for target in TOP2_PCT_TARGETS:
        for seed in SEEDS:
            ds_id = f"top2_pct={target:.0f}_seed={seed}"
            try:
                df, qis, meta = build_top2_concentration_boundary(target, seed=seed)
                datasets.append({
                    'id': ds_id,
                    'boundary_type': 'top2_pct',
                    'target': target,
                    'seed': seed,
                    'df': df,
                    'qis': qis,
                    'meta': meta,
                })
            except Exception as e:
                log.error("Builder failed for %s: %s", ds_id, e)
                datasets.append({
                    'id': ds_id,
                    'boundary_type': 'top2_pct',
                    'target': target,
                    'seed': seed,
                    'df': None,
                    'qis': None,
                    'meta': None,
                    'error': str(e),
                })

    return datasets


# ---------------------------------------------------------------------------
# Feature building + injection
# ---------------------------------------------------------------------------

def _build_features_with_injection(
    df: pd.DataFrame,
    qis: List[str],
    meta: Dict,
    pattern_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Build features dict, inject synthetic var_priority, optionally override pattern.

    The key insight: build_data_features() computes organic var_priority via
    leave-one-out, which won't match our synthetic targets. We replace it
    with the synthetic var_priority from the builder, then let
    classify_risk_concentration() derive the pattern.

    For counterfactual runs, we further override the pattern in
    risk_concentration to force a specific routing path.
    """
    from sdc_engine.sdc.protection_engine import build_data_features
    from sdc_engine.sdc.selection.features import classify_risk_concentration

    # Build base features (this computes organic var_priority internally)
    features = build_data_features(df, qis)

    # Replace with synthetic var_priority from builder
    vp = meta['var_priority']
    features['var_priority'] = vp
    features['risk_concentration'] = classify_risk_concentration(vp)

    # Apply counterfactual override
    if pattern_override and pattern_override != 'natural':
        features['risk_concentration']['pattern'] = pattern_override

        if pattern_override == 'spread_high':
            # Also set n_high_risk=3 and relabel top-3 QIs as HIGH
            features['risk_concentration']['n_high_risk'] = 3
            sorted_qis = sorted(vp.items(), key=lambda x: x[1][1], reverse=True)
            modified_vp = dict(vp)
            for qi, (label, pct) in sorted_qis[:3]:
                modified_vp[qi] = ('HIGH', pct)
            features['var_priority'] = modified_vp

    # Ensure metric type is set
    features['_risk_metric_type'] = RISK_METRIC
    features['_reid_target_raw'] = RISK_TARGET

    return features


# ---------------------------------------------------------------------------
# Single counterfactual run (Step 2)
# ---------------------------------------------------------------------------

import re
_RULE_RE = re.compile(r"Rule:\s+(\S+)\s+→\s+(\S+)")


def run_counterfactual(
    ds: Dict,
    counterfactual: str,
) -> SensitivityOutcome:
    """Run one dataset under one counterfactual path.

    Wraps the full protection pipeline with error handling.
    """
    ds_id = ds['id']
    start = time.monotonic()

    # Handle builder failures
    if ds.get('error') or ds['df'] is None:
        return SensitivityOutcome(
            dataset_id=ds_id,
            boundary_type=ds['boundary_type'],
            target_value=ds['target'],
            actual_value=0.0,
            seed=ds['seed'],
            counterfactual=counterfactual,
            natural_pattern='error',
            forced_pattern=counterfactual,
            rule_applied='',
            method_selected='ERROR',
            reid_before=float('nan'),
            reid_after=float('nan'),
            utility_score=0.0,
            suppression_rate=0.0,
            n_iterations=0,
            target_met=False,
            elapsed_sec=0.0,
            error=f"Builder error: {ds.get('error', 'unknown')}",
        )

    try:
        df = ds['df']
        qis = ds['qis']
        meta = ds['meta']

        # Force Python-only fallback
        from sdc_engine.sdc import r_backend as _rb
        _rb._R_CHECK_CACHE["result"] = False

        # Build features with injection
        features = _build_features_with_injection(df, qis, meta,
                                                   pattern_override=counterfactual)

        natural_pattern = meta['pattern']
        forced_pattern = counterfactual if counterfactual != 'natural' else natural_pattern

        # Set up protector
        from sdc_engine.entities.dataset.pandas.dataset import PdDataset
        from sdc_engine.interactors.sdc_protection import SDCProtection
        from sdc_engine.sdc.protection_engine import run_rules_engine_protection

        dataset = PdDataset(data=df.copy(), activeCols=list(df.columns))
        protector = SDCProtection(dataset=dataset)

        result, log_entries = run_rules_engine_protection(
            input_data=df,
            quasi_identifiers=qis,
            data_features=features,
            access_tier='SCIENTIFIC',
            reid_target=RISK_TARGET,
            utility_floor=UTILITY_FLOOR,
            apply_method_fn=protector.apply_method,
            sensitive_columns=[],
            risk_metric=RISK_METRIC,
            risk_target_raw=RISK_TARGET,
        )

        # Extract metrics
        reid_before = (result.reid_before or {}).get('reid_95', float('nan'))
        reid_after = (result.reid_after or {}).get('reid_95', float('nan'))

        supp_detail = getattr(result, 'qi_suppression_detail', {}) or {}
        max_supp = max(supp_detail.values()) if supp_detail else 0.0

        rule_name = ''
        initial_method = ''
        for entry in log_entries:
            m = _RULE_RE.search(entry)
            if m:
                rule_name = m.group(1)
                initial_method = m.group(2)
                break

        target_met = (reid_after or 1.0) <= RISK_TARGET

        actual_value = meta.get('top_pct', 0) if ds['boundary_type'] == 'top_pct' else meta.get('top2_pct', 0)

        return SensitivityOutcome(
            dataset_id=ds_id,
            boundary_type=ds['boundary_type'],
            target_value=ds['target'],
            actual_value=actual_value,
            seed=ds['seed'],
            counterfactual=counterfactual,
            natural_pattern=natural_pattern,
            forced_pattern=forced_pattern,
            rule_applied=rule_name,
            method_selected=result.method or 'UNKNOWN',
            reid_before=reid_before,
            reid_after=reid_after,
            utility_score=result.utility_score or 0.0,
            suppression_rate=max_supp,
            n_iterations=(result.metadata or {}).get('n_iterations', 0)
                         if result.metadata else 0,
            target_met=target_met,
            elapsed_sec=time.monotonic() - start,
        )

    except Exception as exc:
        return SensitivityOutcome(
            dataset_id=ds_id,
            boundary_type=ds['boundary_type'],
            target_value=ds['target'],
            actual_value=0.0,
            seed=ds['seed'],
            counterfactual=counterfactual,
            natural_pattern=meta.get('pattern', 'error') if meta else 'error',
            forced_pattern=counterfactual,
            rule_applied='',
            method_selected='ERROR',
            reid_before=float('nan'),
            reid_after=float('nan'),
            utility_score=0.0,
            suppression_rate=0.0,
            n_iterations=0,
            target_met=False,
            elapsed_sec=time.monotonic() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Full sweep (Steps 1-3)
# ---------------------------------------------------------------------------

def run_sweep(verbose: bool = True) -> pd.DataFrame:
    """Run the full Family 1 sweep: 27 datasets × 5 counterfactuals = 135 runs.

    Returns DataFrame of SensitivityOutcome records.
    """
    print("=" * 70)
    print("Family 1 — Risk Concentration Routing Validation")
    print("=" * 70)

    # Step 1: Generate datasets
    print("\n[Step 1] Generating 27 boundary datasets...")
    t0 = time.monotonic()
    datasets = generate_datasets()
    n_ok = sum(1 for d in datasets if d.get('df') is not None)
    n_err = len(datasets) - n_ok
    print(f"  Generated {n_ok}/{len(datasets)} datasets "
          f"({n_err} errors) in {time.monotonic() - t0:.1f}s")

    if n_err > 0:
        for d in datasets:
            if d.get('error'):
                print(f"    ERROR: {d['id']}: {d['error']}")

    # Step 2: Run counterfactuals
    total = len(datasets) * len(COUNTERFACTUALS)
    print(f"\n[Step 2] Running {total} counterfactual paths "
          f"({len(datasets)} datasets × {len(COUNTERFACTUALS)} paths)...")

    results: List[SensitivityOutcome] = []
    t0 = time.monotonic()
    run_count = 0

    for ds in datasets:
        for cf in COUNTERFACTUALS:
            run_count += 1
            if verbose:
                print(f"  [{run_count:3d}/{total}] {ds['id']} / {cf}...", end='', flush=True)

            outcome = run_counterfactual(ds, cf)
            results.append(outcome)

            if verbose:
                if outcome.error:
                    print(f" ERROR: {outcome.error[:60]}")
                else:
                    print(f" rule={outcome.rule_applied:<25s} "
                          f"method={outcome.method_selected:<8s} "
                          f"reid={outcome.reid_after:.4f} "
                          f"util={outcome.utility_score:.3f} "
                          f"met={outcome.target_met}")

    elapsed = time.monotonic() - t0
    n_errors = sum(1 for r in results if r.error)
    n_met = sum(1 for r in results if r.target_met)
    print(f"\n  Completed {len(results)} runs in {elapsed:.1f}s "
          f"({n_errors} errors, {n_met}/{len(results)} target_met)")

    # Convert to DataFrame
    df_results = pd.DataFrame([r.as_dict() for r in results])

    # Step 3: Analyze
    print("\n[Step 3] Analyzing results...")
    _print_step3_summary(df_results)

    return df_results


# ---------------------------------------------------------------------------
# Step 3 — inline analysis
# ---------------------------------------------------------------------------

def _print_step3_summary(df: pd.DataFrame):
    """Print Step 3 summary: routing validation + crossover detection."""

    # Routing validation: does the natural path match expectations?
    print("\n--- Routing Validation (natural runs only) ---")
    nat = df[df['counterfactual'] == 'natural'].copy()
    if len(nat) == 0:
        print("  No natural runs found!")
        return

    print(f"\n  {'Dataset':<30s} {'Pattern':<14s} {'Rule':<28s} {'Method':<10s} "
          f"{'reid_after':>10s} {'utility':>8s} {'met':>4s}")
    print("  " + "-" * 108)

    for _, row in nat.iterrows():
        print(f"  {row['dataset_id']:<30s} {row['natural_pattern']:<14s} "
              f"{row['rule_applied']:<28s} {row['method_selected']:<10s} "
              f"{row['reid_after']:>10.4f} {row['utility_score']:>8.3f} "
              f"{'YES' if row['target_met'] else 'NO':>4s}")

    # Sanity checks
    print("\n--- Sanity Checks ---")
    _check_routing_sanity(df)

    # Crossover detection
    print("\n--- Crossover Detection ---")
    _detect_crossovers(df)


def _check_routing_sanity(df: pd.DataFrame):
    """Verify expected routing patterns hold."""
    checks = []

    # Check 1: 'dominated' counterfactual always selects RC1/LOCSUPR
    dom = df[df['counterfactual'] == 'dominated']
    dom_rc1 = dom[dom['rule_applied'].str.contains('RC1', na=False)]
    ok = len(dom_rc1) == len(dom) if len(dom) > 0 else True
    checks.append(('dominated -> RC1/LOCSUPR', ok,
                    f"{len(dom_rc1)}/{len(dom)} runs"))

    # Check 2: 'concentrated' counterfactual always selects RC2/kANON
    conc = df[df['counterfactual'] == 'concentrated']
    conc_rc2 = conc[conc['rule_applied'].str.contains('RC2', na=False)]
    ok = len(conc_rc2) == len(conc) if len(conc) > 0 else True
    checks.append(('concentrated -> RC2/kANON', ok,
                    f"{len(conc_rc2)}/{len(conc)} runs"))

    # Check 3: 'spread_high' counterfactual always selects RC3/kANON
    spr = df[df['counterfactual'] == 'spread_high']
    spr_rc3 = spr[spr['rule_applied'].str.contains('RC3', na=False)]
    ok = len(spr_rc3) == len(spr) if len(spr) > 0 else True
    checks.append(('spread_high -> RC3/kANON', ok,
                    f"{len(spr_rc3)}/{len(spr)} runs"))

    # Check 4: 'balanced' counterfactual never selects RC1/RC2/RC3
    bal = df[df['counterfactual'] == 'balanced']
    bal_rc = bal[bal['rule_applied'].str.startswith('RC', na=False)]
    ok = len(bal_rc) == 0 if len(bal) > 0 else True
    checks.append(('balanced -> no RC rules', ok,
                    f"{len(bal_rc)}/{len(bal)} hit RC"))

    # Check 5: natural pattern transitions from balanced->dominated as top_pct crosses 40
    nat_top = df[(df['counterfactual'] == 'natural') & (df['boundary_type'] == 'top_pct')]
    below_40 = nat_top[nat_top['target_value'] < 40]
    at_or_above_40 = nat_top[nat_top['target_value'] >= 40]
    below_ok = all(below_40['natural_pattern'] == 'balanced') if len(below_40) > 0 else True
    above_ok = all(at_or_above_40['natural_pattern'] == 'dominated') if len(at_or_above_40) > 0 else True
    checks.append(('top_pct<40 -> balanced', below_ok,
                    f"{sum(below_40['natural_pattern'] == 'balanced')}/{len(below_40)}"))
    checks.append(('top_pct>=40 -> dominated', above_ok,
                    f"{sum(at_or_above_40['natural_pattern'] == 'dominated')}/{len(at_or_above_40)}"))

    for label, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}: {detail}")


def _detect_crossovers(df: pd.DataFrame):
    """Detect if any counterfactual consistently outperforms the natural path.

    A crossover is: for a given (boundary_type, target_value), a
    non-natural counterfactual produces lower reid_after with utility >= 0.80
    across ALL seeds.
    """
    nat = df[df['counterfactual'] == 'natural'].set_index(['dataset_id'])
    crossovers = []

    for bt in ['top_pct', 'top2_pct']:
        bt_df = df[(df['boundary_type'] == bt) & (df['counterfactual'] != 'natural')]

        for target in bt_df['target_value'].unique():
            target_df = bt_df[bt_df['target_value'] == target]

            for cf in [c for c in COUNTERFACTUALS if c != 'natural']:
                cf_df = target_df[target_df['counterfactual'] == cf]

                # Compare against natural for each seed
                wins = 0
                losses = 0
                draws = 0
                reid_deltas = []
                utility_ok = True

                for _, row in cf_df.iterrows():
                    ds_id = row['dataset_id']
                    if ds_id not in nat.index:
                        continue
                    nat_row = nat.loc[ds_id]
                    if row['error'] or (isinstance(nat_row, pd.Series) and nat_row.get('error')):
                        continue

                    nat_reid = nat_row['reid_after'] if isinstance(nat_row, pd.Series) else float('nan')
                    cf_reid = row['reid_after']

                    if row['utility_score'] < UTILITY_FLOOR:
                        utility_ok = False

                    delta = nat_reid - cf_reid  # positive = cf is better
                    reid_deltas.append(delta)

                    if delta > 0.005:
                        wins += 1
                    elif delta < -0.005:
                        losses += 1
                    else:
                        draws += 1

                # Crossover: wins across ALL seeds, utility maintained
                n_seeds = len(SEEDS)
                if wins >= n_seeds and losses == 0 and utility_ok and reid_deltas:
                    mean_delta = np.mean(reid_deltas)
                    crossovers.append({
                        'boundary_type': bt,
                        'target_value': target,
                        'counterfactual': cf,
                        'wins': wins,
                        'mean_reid_improvement': mean_delta,
                    })

    if crossovers:
        print(f"\n  CROSSOVERS FOUND: {len(crossovers)}")
        for xo in crossovers:
            print(f"    {xo['boundary_type']}={xo['target_value']:.0f}: "
                  f"'{xo['counterfactual']}' beats natural by "
                  f"{xo['mean_reid_improvement']:.4f} reid "
                  f"across {xo['wins']} seeds")
        print("\n  --> Step 4 cross-metric regression check REQUIRED")
    else:
        print("  No consistent crossovers found.")
        print("  --> Step 4 not needed. Current thresholds are stable.")

    return crossovers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.WARNING,
        format='%(levelname)s %(name)s: %(message)s',
    )

    df_results = run_sweep(verbose=True)

    # Save results
    report_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(report_dir, exist_ok=True)

    csv_path = os.path.join(report_dir, 'family_1_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Save crossover summary
    crossovers = _detect_crossovers(df_results)
    if crossovers:
        xo_df = pd.DataFrame(crossovers)
        xo_path = os.path.join(report_dir, 'family_1_crossovers.csv')
        xo_df.to_csv(xo_path, index=False)
        print(f"Crossovers saved to {xo_path}")

    return df_results


if __name__ == '__main__':
    main()
