"""
CLI entry point for empirical validation.

Usage:
    python -m tests.empirical.run_validation
    python -m tests.empirical.run_validation --thresholds T1 T2
    python -m tests.empirical.run_validation --datasets adult
    python -m tests.empirical.run_validation --metric k_anonymity --target 5
    python -m tests.empirical.run_validation --out reports/run1
"""
import argparse
from pathlib import Path

from .harness import run_matrix, write_reports

_DEFAULT_TARGETS = {
    'reid95': 0.05,
    'k_anonymity': 5.0,
    'uniqueness': 0.05,
}


def main():
    ap = argparse.ArgumentParser(
        description="Run empirical validation of rule-engine thresholds"
    )
    ap.add_argument('--thresholds', nargs='+', default=None,
                    help='Threshold IDs to test (default: all)')
    ap.add_argument('--datasets', nargs='+', default=None,
                    help='Dataset names (default: all registered)')
    ap.add_argument('--risk-target', type=float, default=None,
                    help='(Deprecated) Risk target. Use --target instead.')
    ap.add_argument('--metric', default='reid95',
                    choices=['reid95', 'k_anonymity', 'uniqueness'],
                    help='Risk metric to optimize (default: reid95)')
    ap.add_argument('--target', type=float, default=None,
                    help='Risk target raw value (defaults: reid95=0.05, '
                         'k_anonymity=5, uniqueness=0.05)')
    ap.add_argument('--out', type=Path, default=None,
                    help='Output directory (default: metric-specific)')
    args = ap.parse_args()

    # Resolve risk target: --target > --risk-target > metric default
    risk_target = (args.target if args.target is not None
                   else args.risk_target if args.risk_target is not None
                   else _DEFAULT_TARGETS[args.metric])

    # Default output directory differentiates by metric
    if args.out is None:
        args.out = Path(f'tests/empirical/reports/{args.metric}_latest')

    results = run_matrix(
        threshold_ids=args.thresholds,
        dataset_names=args.datasets,
        risk_metric=args.metric,
        risk_target=risk_target,
    )
    write_reports(results, args.out)
    print(f"\nReports written to: {args.out}")
    print(f"  results.csv     -- all runs")
    print(f"  crossovers.csv  -- method changes across threshold values")
    print(f"  report.md       -- human-readable summary")


if __name__ == '__main__':
    main()
