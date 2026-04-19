"""
CLI entry point for empirical validation.

Usage:
    python -m tests.empirical.run_validation
    python -m tests.empirical.run_validation --thresholds T1 T2
    python -m tests.empirical.run_validation --datasets adult
    python -m tests.empirical.run_validation --out reports/run1
"""
import argparse
from pathlib import Path

from .harness import run_matrix, write_reports


def main():
    ap = argparse.ArgumentParser(
        description="Run empirical validation of rule-engine thresholds"
    )
    ap.add_argument('--thresholds', nargs='+', default=None,
                    help='Threshold IDs to test (default: all)')
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='Dataset names (default: all registered)')
    ap.add_argument('--risk-target', type=float, default=0.05)
    ap.add_argument('--out', type=Path,
                    default=Path('tests/empirical/reports/latest'))
    args = ap.parse_args()

    results = run_matrix(
        threshold_ids=args.thresholds,
        dataset_names=args.datasets,
        risk_target=args.risk_target,
    )
    write_reports(results, args.out)
    print(f"\nReports written to: {args.out}")
    print(f"  results.csv     -- all runs")
    print(f"  crossovers.csv  -- method changes across threshold values")
    print(f"  report.md       -- human-readable summary")


if __name__ == '__main__':
    main()
