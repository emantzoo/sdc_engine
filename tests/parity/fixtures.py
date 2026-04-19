"""Dataset loaders. Uses sdcMicro's bundled data via rpy2 when R is available,
falls back to checked-in CSV snapshots otherwise."""
from pathlib import Path
import numpy as np
import pandas as pd

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


def load_sdcmicro_dataset(name: str) -> pd.DataFrame:
    """Load a dataset bundled with sdcMicro.

    Supported names: 'testdata', 'CASCrefmicrodata', 'francdat', 'free1'.

    Tries R first, falls back to snapshots/ if R is unavailable.
    Snapshots are checked in so tests run deterministically without R.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        ro.r(f'library(sdcMicro); data({name})')
        with localconverter(ro.default_converter + pandas2ri.converter):
            converted = ro.conversion.rpy2py(ro.r(name))
        # Handle R matrices (e.g. free1) — convert to DataFrame
        if isinstance(converted, np.ndarray):
            colnames = list(ro.r(f'colnames({name})'))
            return pd.DataFrame(converted, columns=colnames)
        return converted
    except Exception:
        snapshot = SNAPSHOTS_DIR / f"{name}.csv"
        if snapshot.exists():
            return pd.read_csv(snapshot)
        raise RuntimeError(
            f"Dataset {name} unavailable: R not loadable and no snapshot at {snapshot}. "
            f"Run scripts/regenerate_snapshots.py when R is available."
        )
