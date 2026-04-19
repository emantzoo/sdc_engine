"""
Core runner -- orchestrates threshold x dataset matrix.
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd

from .datasets import DATASETS
from .thresholds import THRESHOLDS, ThresholdTest
from .outcomes import Outcome, run_outcome


def run_matrix(
    threshold_ids: Optional[List[str]] = None,
    dataset_names: Optional[List[str]] = None,
    risk_target: float = 0.05,
) -> pd.DataFrame:
    """Run every (threshold x value x dataset) combination.

    Returns a long-format DataFrame of Outcomes.
    """
    thresholds = [t for t in THRESHOLDS
                  if threshold_ids is None or t.id in threshold_ids]
    datasets = [d for d in DATASETS
                if dataset_names is None or d.name in dataset_names]

    if not datasets:
        raise RuntimeError(
            "No datasets registered. Add entries to tests/empirical/datasets.py "
            "and place data files in tests/empirical/data/"
        )

    all_outcomes: List[Outcome] = []
    for threshold in thresholds:
        relevant_datasets = [
            d for d in datasets
            if not d.relevant_thresholds or threshold.id in d.relevant_thresholds
        ]
        if not relevant_datasets:
            print(f"[{threshold.id}] No datasets marked relevant -- skipping")
            continue

        for ds in relevant_datasets:
            print(f"[{threshold.id}] Dataset: {ds.name}")
            df = ds.load()
            # Drop rows with NaN in QI columns to avoid downstream errors
            df = df.dropna(subset=ds.quasi_identifiers).reset_index(drop=True)
            for value in threshold.test_values:
                print(f"  value={value}", end=" ... ", flush=True)
                outcome = run_outcome(
                    df, ds.quasi_identifiers, ds.sensitive_columns,
                    threshold.id, value, threshold.patcher,
                    risk_target=risk_target,
                )
                outcome.dataset = ds.name
                all_outcomes.append(outcome)
                status = "OK" if outcome.target_met else "miss"
                print(f"{status} method={outcome.selected_method}")

    return pd.DataFrame([o.as_dict() for o in all_outcomes])


def find_crossovers(results: pd.DataFrame) -> pd.DataFrame:
    """For each (threshold, dataset), find the value at which the selected
    method changes and compare to the current threshold.

    Returns a DataFrame with columns:
        threshold_id, dataset, observed_crossover, current_value,
        shift_pp, recommendation
    """
    rows = []
    for (tid, ds), group in results.groupby(['threshold_id', 'dataset']):
        if group.empty or group['error'].notna().any():
            continue
        group_sorted = group.sort_values('threshold_value')
        low_method = group_sorted.iloc[0]['selected_method']
        high_method = group_sorted.iloc[-1]['selected_method']
        if low_method == high_method:
            continue

        crossover_row = group_sorted[
            group_sorted['selected_method'] != low_method
        ].iloc[0]
        observed = crossover_row['threshold_value']

        from .thresholds import THRESHOLDS
        current = next((t.current_value for t in THRESHOLDS if t.id == tid), None)
        shift_pp = (observed - current) * 100 if current else 0

        recommendation = ""
        if current is not None and abs(shift_pp) > 5:
            recommendation = f"CONSIDER: move to {observed:.3f}"

        rows.append({
            'threshold_id': tid,
            'dataset': ds,
            'current_value': current,
            'observed_crossover': observed,
            'shift_pp': shift_pp,
            'recommendation': recommendation,
        })

    return pd.DataFrame(rows)


def write_reports(results: pd.DataFrame, out_dir: Path) -> None:
    """Write CSV + markdown report to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)

    crossovers = find_crossovers(results)
    crossovers.to_csv(out_dir / "crossovers.csv", index=False)

    with (out_dir / "report.md").open("w") as f:
        f.write("# Empirical Validation Report\n\n")
        f.write(f"Total runs: {len(results)}\n")
        f.write(f"Failures: {results['error'].notna().sum()}\n\n")

        f.write("## Crossovers detected\n\n")
        if crossovers.empty:
            f.write("_No method crossovers observed in tested ranges._\n\n")
        else:
            flagged = crossovers[crossovers['recommendation'] != ""]
            f.write(f"Total crossovers: {len(crossovers)}  "
                    f"(**{len(flagged)} flagged** -- shift > 5 pp)\n\n")
            f.write(crossovers.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Per-threshold summary\n\n")
        for tid, group in results.groupby('threshold_id'):
            f.write(f"### {tid}\n\n")
            f.write(group[['dataset', 'threshold_value', 'selected_method',
                          'reid_after', 'utility_score', 'target_met']]
                    .to_markdown(index=False))
            f.write("\n\n")
