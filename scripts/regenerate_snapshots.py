"""Export sdcMicro-bundled datasets to CSV snapshots for offline parity tests.

Usage:
    python scripts/regenerate_snapshots.py

Requires R + rpy2 + sdcMicro. Outputs to tests/parity/snapshots/.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

DATASETS = ["testdata", "CASCrefmicrodata", "francdat", "free1"]
OUT_DIR = Path(__file__).parent.parent / "tests" / "parity" / "snapshots"


def main():
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
    except ImportError:
        print("rpy2 not installed — cannot generate snapshots.")
        sys.exit(1)

    ro.r('library(sdcMicro)')
    sdcmicro_version = str(ro.r('packageVersion("sdcMicro")')[0])
    print(f"sdcMicro version: {sdcmicro_version}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in DATASETS:
        try:
            ro.r(f'data({name})')
            with localconverter(ro.default_converter + pandas2ri.converter):
                converted = ro.conversion.rpy2py(ro.r(name))
            # Handle R matrices (e.g. free1) — convert to DataFrame
            if isinstance(converted, np.ndarray):
                colnames = list(ro.r(f'colnames({name})'))
                df = pd.DataFrame(converted, columns=colnames)
            else:
                df = converted
            out_path = OUT_DIR / f"{name}.csv"
            df.to_csv(out_path, index=False)
            print(f"  {name}: {len(df)} rows -> {out_path}")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    # Write metadata file with version info
    meta_path = OUT_DIR / "VERSION.txt"
    with open(meta_path, 'w') as f:
        f.write(f"sdcMicro version: {sdcmicro_version}\n")
        for name in DATASETS:
            csv = OUT_DIR / f"{name}.csv"
            if csv.exists():
                row_count = sum(1 for _ in open(csv)) - 1  # minus header
                f.write(f"{name}: {row_count} rows\n")
    print(f"  Metadata -> {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
