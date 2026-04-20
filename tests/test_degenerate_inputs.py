"""
Hunt 2 — Degenerate input tests.

Test protection methods on edge-case data shapes that commonly break code:
- Single group (all rows identical QI values)
- Single row
- Single QI column
- Single category per QI
- All-NaN QI column
- Duplicate-only dataset
- High cardinality (every row unique)

These tests assert no crash and valid output shape. Some methods may return
the input unchanged or with warnings — that's acceptable.
"""
import numpy as np
import pandas as pd
import pytest

# Disable R backend globally
@pytest.fixture(autouse=True)
def _no_r():
    from sdc_engine.sdc import r_backend as _rb
    _rb._R_CHECK_CACHE["result"] = False
    _rb._R_CHECK_CACHE["timestamp"] = float('inf')


# ── Dataset builders ───────────────────────────────────────────────────

def _single_group(n=50):
    """All rows have identical QI values — one equivalence class."""
    return pd.DataFrame({
        'age': [30] * n,
        'sex': ['M'] * n,
        'income': np.random.default_rng(42).normal(50000, 100, n).round(2),
    })


def _single_row():
    """Minimal dataset — 1 row."""
    return pd.DataFrame({
        'age': [25],
        'sex': ['F'],
        'income': [45000.0],
    })


def _two_identical_rows():
    """Two identical rows and nothing else."""
    return pd.DataFrame({
        'age': [30, 30],
        'sex': ['M', 'M'],
        'income': [50000.0, 50000.0],
    })


def _single_qi():
    """Dataset with only 1 QI column."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'age': rng.integers(18, 80, 100),
        'value': rng.normal(100, 10, 100),
    })


def _single_category_per_qi(n=50):
    """Every QI has exactly 1 unique value."""
    return pd.DataFrame({
        'age': [40] * n,
        'sex': ['F'] * n,
        'region': ['North'] * n,
        'income': np.random.default_rng(42).normal(50000, 100, n).round(2),
    })


def _all_nan_qi(n=50):
    """QI column is entirely NaN."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'age': rng.integers(18, 80, n),
        'sex': [np.nan] * n,
        'income': rng.normal(50000, 100, n).round(2),
    })


def _every_row_unique(n=100):
    """High cardinality — every row has unique QI combination."""
    return pd.DataFrame({
        'age': range(n),
        'sex': [f'cat_{i}' for i in range(n)],
        'income': np.arange(n, dtype=float) * 1000,
    })


def _zero_variance_numeric(n=50):
    """Numeric column with zero variance (constant value)."""
    return pd.DataFrame({
        'age': [30] * n,
        'income': [50000.0] * n,
        'sex': np.random.default_rng(42).choice(['M', 'F'], n),
    })


# ── kANON ──────────────────────────────────────────────────────────────

class TestKanonDegenerate:
    def test_single_group(self):
        """All rows same QI values — already k-anonymous for any k <= n."""
        df = _single_group()
        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=['age', 'sex'], k=3,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None
        assert len(protected) >= 1

    def test_single_row(self):
        df = _single_row()
        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=['age', 'sex'], k=3,
            max_suppression_rate=1.0, return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_two_identical_rows(self):
        df = _two_identical_rows()
        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=['age', 'sex'], k=2,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_single_qi(self):
        df = _single_qi()
        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=['age'], k=3,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_every_row_unique(self):
        """High cardinality — kANON must generalize aggressively or suppress."""
        df = _every_row_unique()
        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=['age', 'sex'], k=3,
            max_suppression_rate=0.50, return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_single_category_per_qi(self):
        df = _single_category_per_qi()
        from sdc_engine.sdc.kANON import apply_kanon
        result = apply_kanon(
            df, quasi_identifiers=['age', 'sex', 'region'], k=3,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None


# ── LOCSUPR ────────────────────────────────────────────────────────────

class TestLocsuprDegenerate:
    def test_single_group(self):
        """Single group — already k-anonymous, nothing to suppress."""
        df = _single_group()
        from sdc_engine.sdc.LOCSUPR import apply_locsupr
        result = apply_locsupr(
            df, quasi_identifiers=['age', 'sex'], k=3,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None
        assert len(protected) == len(df)

    def test_single_row(self):
        """Single row can't satisfy k>=2 — LOCSUPR should suppress all QI cells."""
        df = _single_row()
        from sdc_engine.sdc.LOCSUPR import apply_locsupr
        result = apply_locsupr(
            df, quasi_identifiers=['age', 'sex'], k=2,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_two_identical_rows(self):
        df = _two_identical_rows()
        from sdc_engine.sdc.LOCSUPR import apply_locsupr
        result = apply_locsupr(
            df, quasi_identifiers=['age', 'sex'], k=2,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_single_qi(self):
        df = _single_qi()
        from sdc_engine.sdc.LOCSUPR import apply_locsupr
        result = apply_locsupr(
            df, quasi_identifiers=['age'], k=3,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_every_row_unique(self):
        df = _every_row_unique()
        from sdc_engine.sdc.LOCSUPR import apply_locsupr
        result = apply_locsupr(
            df, quasi_identifiers=['age', 'sex'], k=3,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_single_category_per_qi(self):
        df = _single_category_per_qi()
        from sdc_engine.sdc.LOCSUPR import apply_locsupr
        result = apply_locsupr(
            df, quasi_identifiers=['age', 'sex', 'region'], k=3,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None
        # Already k-anonymous, should not suppress anything
        assert len(protected) == len(df)


# ── PRAM ───────────────────────────────────────────────────────────────

class TestPramDegenerate:
    def test_single_group(self):
        df = _single_group()
        from sdc_engine.sdc.PRAM import apply_pram
        result = apply_pram(
            df, variables=['sex'], p_change=0.2,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None
        assert len(protected) == len(df)

    def test_single_row(self):
        df = _single_row()
        from sdc_engine.sdc.PRAM import apply_pram
        result = apply_pram(
            df, variables=['sex'], p_change=0.2,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None

    def test_single_category(self):
        """PRAM with 1 category should skip the variable (no transitions possible)."""
        df = _single_category_per_qi()
        from sdc_engine.sdc.PRAM import apply_pram
        result = apply_pram(
            df, variables=['sex'], p_change=0.2,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None

    def test_all_nan_variable(self):
        """PRAM on all-NaN variable should not crash."""
        df = _all_nan_qi()
        from sdc_engine.sdc.PRAM import apply_pram
        result = apply_pram(
            df, variables=['sex'], p_change=0.2,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None

    def test_every_row_unique_categories(self):
        df = _every_row_unique()
        from sdc_engine.sdc.PRAM import apply_pram
        result = apply_pram(
            df, variables=['sex'], p_change=0.2,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None


# ── NOISE ──────────────────────────────────────────────────────────────

class TestNoiseDegenerate:
    def test_single_group(self):
        df = _single_group()
        from sdc_engine.sdc.NOISE import apply_noise
        result = apply_noise(
            df, variables=['income'], magnitude=0.1,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None

    def test_single_row(self):
        df = _single_row()
        from sdc_engine.sdc.NOISE import apply_noise
        result = apply_noise(
            df, variables=['income'], magnitude=0.1,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None

    def test_zero_variance(self):
        """Noise on constant column — zero std, should skip or handle gracefully."""
        df = _zero_variance_numeric()
        from sdc_engine.sdc.NOISE import apply_noise
        result = apply_noise(
            df, variables=['income'], magnitude=0.1,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None
        assert not protected['income'].isna().all()

    def test_all_nan_variable(self):
        """Noise on all-NaN column should not crash."""
        df = pd.DataFrame({
            'age': np.random.default_rng(42).integers(18, 80, 50),
            'value': [np.nan] * 50,
        })
        from sdc_engine.sdc.NOISE import apply_noise
        result = apply_noise(
            df, variables=['value'], magnitude=0.1,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None

    def test_two_identical_rows(self):
        df = _two_identical_rows()
        from sdc_engine.sdc.NOISE import apply_noise
        result = apply_noise(
            df, variables=['income'], magnitude=0.1,
            return_metadata=True, verbose=False, seed=42,
        )
        protected, meta = result
        assert protected is not None


# ── GENERALIZE ─────────────────────────────────────────────────────────

class TestGeneralizeDegenerate:
    def test_single_group(self):
        df = _single_group()
        from sdc_engine.sdc.GENERALIZE import apply_generalize
        result = apply_generalize(
            df, quasi_identifiers=['age', 'sex'], max_categories=5,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None
        assert len(protected) == len(df)

    def test_single_row(self):
        df = _single_row()
        from sdc_engine.sdc.GENERALIZE import apply_generalize
        result = apply_generalize(
            df, quasi_identifiers=['age', 'sex'], max_categories=5,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_single_qi(self):
        df = _single_qi()
        from sdc_engine.sdc.GENERALIZE import apply_generalize
        result = apply_generalize(
            df, quasi_identifiers=['age'], max_categories=5,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_single_category_per_qi(self):
        """All QIs have 1 unique value — nothing to generalize."""
        df = _single_category_per_qi()
        from sdc_engine.sdc.GENERALIZE import apply_generalize
        result = apply_generalize(
            df, quasi_identifiers=['age', 'sex', 'region'], max_categories=5,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_every_row_unique(self):
        df = _every_row_unique()
        from sdc_engine.sdc.GENERALIZE import apply_generalize
        result = apply_generalize(
            df, quasi_identifiers=['age', 'sex'], max_categories=5,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None

    def test_all_nan_qi(self):
        """QI column entirely NaN."""
        df = _all_nan_qi()
        from sdc_engine.sdc.GENERALIZE import apply_generalize
        result = apply_generalize(
            df, quasi_identifiers=['age', 'sex'], max_categories=5,
            return_metadata=True, verbose=False,
        )
        protected, meta = result
        assert protected is not None
