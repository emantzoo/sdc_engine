"""
Regression test: kANON must handle int32 dtypes (as loaded by R/rpy2).

Isolated from the main test suite to avoid rpy2 stdout capture conflicts
with pytest's capture plugin.
"""
import numpy as np
import pandas as pd
import pytest


class TestKanonInt32:
    """Verify kANON handles R-loaded data (numpy int32) correctly."""

    def test_kanon_handles_int32_dtype(self):
        """R-loaded data uses int32; kANON merge requires int64."""
        # Disable R backend to avoid rpy2 stdout capture conflicts
        from sdc_engine.sdc import r_backend as _rb
        _rb._R_CHECK_CACHE["result"] = False
        _rb._R_CHECK_CACHE["timestamp"] = float('inf')

        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            'a': rng.integers(0, 5, n).astype('int32'),
            'b': rng.integers(0, 5, n).astype('int32'),
            'c': rng.integers(0, 5, n).astype('int32'),
        })
        # Confirm dtypes are int32 before calling
        assert df['a'].dtype == np.int32
        assert df['b'].dtype == np.int32

        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=['a', 'b', 'c'],
            k=3, strategy='generalization',
            max_suppression_rate=0.20,
            return_metadata=True,
        )
        assert isinstance(result, tuple), "apply_kanon should return (data, metadata)"
        protected_data, metadata = result
        assert protected_data is not None, "Protected data should not be None"
        k_check = metadata.get('k_anonymity_check', {})
        assert bool(k_check.get('is_k_anonymous')), \
            f"Should achieve k=3 on low-cardinality int32 data, got {k_check}"
