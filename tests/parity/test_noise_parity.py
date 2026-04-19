"""NOISE parity: Python vs R sdcMicro.

R uses correlated noise (addNoise with method='correlated') which preserves
covariance structure. Python uses independent Gaussian noise. These are
genuinely different algorithms -- parity here is about sanity, not equality.

We check:
  1. Mean preserved within 5% on both backends
  2. Pearson correlation between original and noisy > 0.85 for both
"""
import numpy as np
import pandas as pd
import pytest

from sdc_engine.sdc.NOISE import apply_noise
from .conftest import skip_if_no_r
from .fixtures import load_sdcmicro_dataset


@skip_if_no_r
@pytest.mark.parametrize("magnitude", [0.05, 0.10, 0.15])
def test_noise_parity_basic(magnitude):
    df = load_sdcmicro_dataset("CASCrefmicrodata")
    cont_vars = ["AFNLWGT", "AGI", "EMCONTRB", "FEDTAX", "PTOTVAL"]
    cont_vars = [v for v in cont_vars if v in df.columns]
    df = df[cont_vars].dropna().reset_index(drop=True).head(500)

    r_out = apply_noise(
        df.copy(), variables=cont_vars, magnitude=magnitude,
        use_r=True, seed=42, return_metadata=False, verbose=False,
    )
    py_out = apply_noise(
        df.copy(), variables=cont_vars, magnitude=magnitude,
        use_r=False, seed=42, return_metadata=False, verbose=False,
    )

    for var in cont_vars:
        orig = df[var]
        for backend, out in [("R", r_out), ("Python", py_out)]:
            noisy = out[var]
            # Mean preserved within 5%
            if abs(orig.mean()) > 1e-6:
                mean_drift = abs((noisy.mean() - orig.mean()) / orig.mean())
                assert mean_drift < 0.05, \
                    f"{backend} mean drift on {var}: {mean_drift:.1%}"

            # Correlation > 0.85 (noise shouldn't destroy signal)
            corr = np.corrcoef(orig, noisy)[0, 1]
            assert corr > 0.85, \
                f"{backend} low correlation on {var} at mag={magnitude}: {corr:.3f}"
