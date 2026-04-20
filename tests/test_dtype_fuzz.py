"""
Hunt 1 — Dtype fuzz tests.

Verify all protection methods survive non-standard dtypes that arise in practice:
- int32 (R/rpy2 loaded data)
- float32 (some CSV loaders, numpy defaults on some platforms)
- object strings (categorical columns stored as str)

These tests don't assert on protection quality — only that methods don't crash
and return a valid DataFrame.
"""
import numpy as np
import pandas as pd
import pytest

# Disable R backend globally for speed + no rpy2 capture conflicts
@pytest.fixture(autouse=True)
def _no_r():
    from sdc_engine.sdc import r_backend as _rb
    _rb._R_CHECK_CACHE["result"] = False
    _rb._R_CHECK_CACHE["timestamp"] = float('inf')


def _make_base_df(n=200, seed=42):
    """Base dataset with int64 + float64 + object columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'age': rng.integers(18, 80, n),
        'sex': rng.choice(['M', 'F'], n),
        'income': rng.normal(50000, 15000, n).round(2),
        'region': rng.choice(['North', 'South', 'East', 'West'], n),
        'score': rng.integers(1, 10, n),
    })


def _cast_int32(df):
    """Cast integer columns to int32 (R/rpy2 loading)."""
    out = df.copy()
    for c in out.select_dtypes('int64').columns:
        out[c] = out[c].astype('int32')
    return out


def _cast_float32(df):
    """Cast float columns to float32."""
    out = df.copy()
    for c in out.select_dtypes('float64').columns:
        out[c] = out[c].astype('float32')
    return out


def _cast_all_int32(df):
    """Cast ALL integer columns to int32 (worst case)."""
    out = df.copy()
    for c in out.select_dtypes(include=['int', 'int64']).columns:
        out[c] = out[c].astype(np.int32)
    return out


def _cast_object_ints(df):
    """Cast integer columns to object/string (common CSV artifact)."""
    out = df.copy()
    for c in out.select_dtypes(include=['int', 'int64']).columns:
        out[c] = out[c].astype(str)
    return out


DTYPE_CASTERS = {
    'default': lambda df: df,
    'int32': _cast_int32,
    'float32': _cast_float32,
    'all_int32': _cast_all_int32,
    'object_ints': _cast_object_ints,
}

QIS_CAT = ['sex', 'region']
QIS_NUM = ['age', 'income']
QIS_MIXED = ['age', 'sex', 'region']


# ── kANON ──────────────────────────────────────────────────────────────

class TestKanonDtypes:
    @pytest.mark.parametrize("dtype_mode", DTYPE_CASTERS.keys())
    def test_kanon_survives_dtype(self, dtype_mode):
        df = DTYPE_CASTERS[dtype_mode](_make_base_df())
        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=QIS_MIXED, k=3,
            max_suppression_rate=0.20, return_metadata=True, verbose=False,
        )
        assert isinstance(result, tuple)
        protected, meta = result
        assert protected is not None
        assert len(protected) > 0
        assert len(protected) <= len(df)

    @pytest.mark.parametrize("dtype_mode", ['int32', 'all_int32'])
    def test_kanon_generalization_with_int32(self, dtype_mode):
        """Regression: int32 caused merge crash in generalization path."""
        df = DTYPE_CASTERS[dtype_mode](_make_base_df())
        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=['age', 'score'], k=3,
            strategy='generalization', max_suppression_rate=0.20,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None


# ── LOCSUPR ────────────────────────────────────────────────────────────

class TestLocsuprDtypes:
    @pytest.mark.parametrize("dtype_mode", DTYPE_CASTERS.keys())
    def test_locsupr_survives_dtype(self, dtype_mode):
        df = DTYPE_CASTERS[dtype_mode](_make_base_df())
        from sdc_engine.sdc.LOCSUPR import apply_locsupr
        result = apply_locsupr(
            df, quasi_identifiers=QIS_MIXED, k=3,
            return_metadata=True, verbose=False,
        )
        assert isinstance(result, tuple)
        protected, meta = result
        assert protected is not None
        assert len(protected) > 0


# ── PRAM ───────────────────────────────────────────────────────────────

class TestPramDtypes:
    @pytest.mark.parametrize("dtype_mode", DTYPE_CASTERS.keys())
    def test_pram_survives_dtype(self, dtype_mode):
        df = DTYPE_CASTERS[dtype_mode](_make_base_df())
        from sdc_engine.sdc.PRAM import apply_pram
        result = apply_pram(
            df, variables=QIS_CAT, p_change=0.2,
            return_metadata=True, verbose=False, seed=42,
        )
        assert isinstance(result, tuple)
        protected, meta = result
        assert protected is not None
        assert len(protected) == len(df)

    def test_pram_object_ints_as_variables(self):
        """PRAM should handle integer-as-string columns (categorical)."""
        df = _cast_object_ints(_make_base_df())
        from sdc_engine.sdc.PRAM import apply_pram
        result = apply_pram(
            df, variables=['age', 'score'], p_change=0.2,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None


# ── NOISE ──────────────────────────────────────────────────────────────

class TestNoiseDtypes:
    @pytest.mark.parametrize("dtype_mode", ['default', 'int32', 'float32', 'all_int32'])
    def test_noise_survives_dtype(self, dtype_mode):
        df = DTYPE_CASTERS[dtype_mode](_make_base_df())
        from sdc_engine.sdc.NOISE import apply_noise
        result = apply_noise(
            df, variables=QIS_NUM, magnitude=0.1,
            return_metadata=True, verbose=False, seed=42,
        )
        assert isinstance(result, tuple)
        protected, meta = result
        assert protected is not None
        assert len(protected) == len(df)

    def test_noise_float32_precision(self):
        """Verify noise doesn't produce inf/nan on float32 data."""
        df = _cast_float32(_make_base_df())
        from sdc_engine.sdc.NOISE import apply_noise
        protected, meta = apply_noise(
            df, variables=['income'], magnitude=0.1,
            return_metadata=True, verbose=False, seed=42,
        )
        assert not protected['income'].isna().all(), "All values became NaN"
        assert not np.isinf(protected['income']).any(), "Inf values produced"


# ── GENERALIZE ─────────────────────────────────────────────────────────

class TestGeneralizeDtypes:
    @pytest.mark.parametrize("dtype_mode", DTYPE_CASTERS.keys())
    def test_generalize_survives_dtype(self, dtype_mode):
        df = DTYPE_CASTERS[dtype_mode](_make_base_df())
        from sdc_engine.sdc.GENERALIZE import apply_generalize
        result = apply_generalize(
            df, quasi_identifiers=QIS_MIXED, max_categories=5,
            return_metadata=True, verbose=False,
        )
        assert isinstance(result, tuple)
        protected, meta = result
        assert protected is not None
        assert len(protected) == len(df)
