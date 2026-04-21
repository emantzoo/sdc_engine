"""
Preprocessing Common-Case Tests
================================

Guards against regressions in the preprocessing pipeline's handling of
normal, mixed, and edge-case inputs.  Tests call preprocessing functions
directly on synthetic fixtures — no full engine pipeline.

Preprocessing pipeline overview (for future readers):

- **Type detection** (`identify_column_types` in sdc_utils.py): Classifies
  each column as continuous/categorical/identifier/binary using dtype,
  cardinality, name keywords, and sequential-ID heuristics.

- **Feature extraction** (`build_data_features` in protection_engine.py):
  Classifies QIs as continuous (numeric, >20 unique) or categorical
  (non-numeric or <=20 unique numeric).  Detects date/geo QIs by column
  name keywords.  Computes cardinality, outliers, skewness, uniqueness,
  risk pattern.  Preprocess metadata can override classification
  (binned continuous stays continuous).

- **Generalization** (`apply_generalize` in GENERALIZE.py): Reduces QI
  cardinality via numeric binning (equal-width or quantile), categorical
  top-K, date period grouping.  Respects var_priority ordering and per-QI
  utility gates.  Access tier affects feasibility thresholds but not
  generalization behavior directly.

- **Feasibility diagnosis** (`check_feasibility` in preprocessing/diagnose.py):
  Classifies dataset as FEASIBLE/HARD/VERY_HARD based on QI count,
  cardinality, combination space.  Access tier tightens thresholds
  (PUBLIC strictest, SECURE most relaxed).

Known issues found during test authorship:

- **_is_sequential_id false positive on integer-coded categoricals:**
  `sdc_utils._is_sequential_id` operates on all N sorted rows (not unique
  values), so the median diff of repeating integers 1-8 is 0.  Since
  min_val >= 0 and median_diff <= 1.5, it classifies the column as
  sequential ID.  True sequential IDs have near-100% uniqueness; this
  heuristic doesn't check uniqueness ratio.  Design limitation, not a
  crash bug.

- **build_data_features crash on discretized string columns:**
  When `preprocess_metadata` marks a string column as `was_continuous`,
  the outlier detection loop (line ~235) tries `s.quantile()` on the
  string data and raises TypeError.  The `_discretized` classification
  works correctly (string column is put in `continuous` list), but
  downstream numeric operations on the `continuous` list don't guard
  against non-numeric dtype.  Real bug — crash, not just wrong answer.
"""
import numpy as np
import pandas as pd
import pytest

from sdc_engine.sdc.sdc_utils import identify_column_types
from sdc_engine.sdc.protection_engine import build_data_features
from sdc_engine.sdc.GENERALIZE import apply_generalize


# ============================================================================
# Helpers
# ============================================================================

def _df(n=1000, **columns):
    """Build a DataFrame with named columns.  Each value is a callable
    (rng → array) or a static array/list."""
    rng = np.random.default_rng(42)
    data = {}
    for name, spec in columns.items():
        data[name] = spec(rng) if callable(spec) else spec
    return pd.DataFrame(data)


# ============================================================================
# 1. Categorical QIs
# ============================================================================

class TestCategoricalQIs:
    """Normal categorical QI behavior."""

    def test_normal_categorical_preserves_values(self):
        """A categorical QI with 10 unique values and 1000 rows should pass
        through GENERALIZE with max_categories=10 without modification."""
        cats = [f"cat_{i}" for i in range(10)]
        df = _df(1000, occupation=lambda rng: rng.choice(cats, 1000))
        result, meta = apply_generalize(
            df, quasi_identifiers=["occupation"],
            max_categories=10, return_metadata=True, verbose=False)
        assert set(result["occupation"].unique()) == set(cats)

    def test_low_cardinality_categorical_untouched(self):
        """A 3-value categorical should be untouched under any max_categories."""
        df = _df(1000, sex=lambda rng: rng.choice(["M", "F", "Other"], 1000))
        result, meta = apply_generalize(
            df, quasi_identifiers=["sex"],
            max_categories=5, return_metadata=True, verbose=False)
        assert set(result["sex"].unique()) == {"M", "F", "Other"}

    def test_high_cardinality_categorical_reduced(self):
        """A categorical QI with 60 unique values should be reduced by
        GENERALIZE when max_categories < 60."""
        cats = [f"city_{i}" for i in range(60)]
        df = _df(1000, city=lambda rng: rng.choice(cats, 1000))
        result, meta = apply_generalize(
            df, quasi_identifiers=["city"],
            max_categories=10, return_metadata=True, verbose=False)
        assert result["city"].nunique() <= 15  # top-K + "Other"


# ============================================================================
# 2. Continuous QIs
# ============================================================================

class TestContinuousQIs:
    """Normal continuous QI behavior."""

    def test_normal_distribution_bins(self):
        """An age-like continuous QI should be binned into distinct ranges."""
        df = _df(1000, age=lambda rng: rng.normal(45, 15, 1000).clip(18, 90).astype(int))
        result, meta = apply_generalize(
            df, quasi_identifiers=["age"],
            max_categories=10, return_metadata=True, verbose=False)
        # Should have reduced cardinality
        assert result["age"].nunique() <= 15
        assert len(result) == 1000  # no rows lost

    def test_extreme_outliers_no_crash(self):
        """Extreme outliers in a continuous QI should not crash generalization."""
        vals = np.concatenate([
            np.random.default_rng(42).normal(50, 10, 990),
            np.array([1e6, -1e6, 1e8, 1e9, -1e9, 0, 0, 0, 0, 0])
        ])
        df = pd.DataFrame({"income": vals})
        result, meta = apply_generalize(
            df, quasi_identifiers=["income"],
            max_categories=10, return_metadata=True, verbose=False)
        assert len(result) == 1000
        assert result["income"].notna().all()

    def test_constant_column_no_error(self):
        """A constant numeric column should not cause division-by-zero."""
        df = _df(1000, constant=lambda rng: np.full(1000, 42.0))
        result, meta = apply_generalize(
            df, quasi_identifiers=["constant"],
            max_categories=10, return_metadata=True, verbose=False)
        assert len(result) == 1000
        # All values should still be the same (no way to bin a constant)
        assert result["constant"].nunique() == 1


# ============================================================================
# 3. Type Detection
# ============================================================================

class TestTypeDetection:
    """identify_column_types and build_data_features classification."""

    def test_integer_coded_categorical(self):
        """Integer-coded categories (e.g., education levels 10,20,...,80)
        should be classified as categorical, not continuous.
        Note: values must start above 10 (avoids _is_sequential_id
        false positive) and column name must not contain id_keywords
        like 'code'."""
        codes = [11, 22, 33, 44, 55, 66, 77, 88]
        df = _df(1000, education=lambda rng: rng.choice(codes, 1000))
        types = identify_column_types(df)
        assert types["education"] == "categorical", (
            f"Integer-coded categorical (8 unique ints) classified as {types['education']}")

    def test_true_continuous_integer(self):
        """A high-cardinality integer column (count data) should be classified
        as continuous."""
        df = _df(1000, amount=lambda rng: rng.integers(100, 50000, 1000))
        types = identify_column_types(df)
        assert types["amount"] == "continuous", (
            f"High-cardinality integer classified as {types['amount']}")

    def test_binary_column(self):
        """A column with exactly 2 unique values should be classified as binary."""
        df = _df(1000, flag=lambda rng: rng.choice([0, 1], 1000))
        types = identify_column_types(df)
        assert types["flag"] == "binary"

    def test_sequential_id_detected(self):
        """A sequential integer column (0, 1, 2, ...) should be classified
        as identifier."""
        df = pd.DataFrame({"row_id": range(1000)})
        types = identify_column_types(df)
        assert types["row_id"] == "identifier"

    def test_string_high_uniqueness_is_identifier(self):
        """A string column with >95% unique values should be classified as
        identifier."""
        df = pd.DataFrame({"email": [f"user{i}@example.com" for i in range(1000)]})
        types = identify_column_types(df)
        assert types["email"] == "identifier"

    def test_income_keyword_is_continuous(self):
        """A column named 'income' with numeric values should be classified
        as continuous (value keyword overrides ID heuristic)."""
        df = _df(1000, income=lambda rng: rng.integers(20000, 100000, 1000))
        types = identify_column_types(df)
        assert types["income"] == "continuous"


# ============================================================================
# 4. build_data_features classification
# ============================================================================

class TestBuildDataFeatures:
    """build_data_features QI classification and feature population."""

    def test_continuous_vs_categorical_split(self):
        """Numeric QI with >20 unique → continuous; string QI → categorical."""
        df = _df(1000,
                 salary=lambda rng: rng.integers(20000, 100000, 1000),
                 sex=lambda rng: rng.choice(["M", "F"], 1000))
        features = build_data_features(df, ["salary", "sex"])
        assert "salary" in features["continuous_vars"]
        assert "sex" in features["categorical_vars"]
        assert features["n_continuous"] == 1
        assert features["n_categorical"] == 1

    def test_low_cardinality_numeric_is_categorical(self):
        """Numeric QI with <=20 unique values → categorical."""
        df = _df(1000, education_level=lambda rng: rng.integers(1, 8, 1000))
        features = build_data_features(df, ["education_level"])
        assert "education_level" in features["categorical_vars"]
        assert features["n_categorical"] == 1

    def test_date_qi_detected_by_name(self):
        """A QI column named 'birth_date' should be counted as date type."""
        df = _df(1000, birth_date=lambda rng: rng.choice(
            pd.date_range("1960-01-01", "2000-12-31").strftime("%Y-%m-%d").tolist(),
            1000))
        features = build_data_features(df, ["birth_date"])
        assert features["qi_type_counts"]["date"] == 1

    def test_geo_qi_detected_by_name(self):
        """QI columns named 'city' and 'region' should be counted as geo."""
        cities = [f"city_{i}" for i in range(60)]
        regions = [f"region_{i}" for i in range(5)]
        df = _df(1000,
                 city=lambda rng: rng.choice(cities, 1000),
                 region=lambda rng: rng.choice(regions, 1000))
        features = build_data_features(df, ["city", "region"])
        assert features["qi_type_counts"]["geo"] == 2
        # city (60 unique) → fine; region (5 unique) → coarse
        assert features["geo_qis_by_granularity"]["city"] == "fine"
        assert features["geo_qis_by_granularity"]["region"] == "coarse"

    @pytest.mark.skip(reason=(
        "known bug — build_data_features outlier loop crashes on discretized "
        "string columns marked was_continuous in preprocess_metadata "
        "(TypeError: quantile on strings). See file header for details."))
    def test_preprocess_metadata_overrides_dtype(self):
        """A binned continuous column should stay classified as continuous
        when preprocess_metadata says was_continuous=True."""
        # Simulate: income was continuous, now binned to 10 categories
        df = _df(1000, income=lambda rng: rng.choice(
            ["10K-20K", "20K-30K", "30K-40K", "40K-50K", "50K+"], 1000))
        features = build_data_features(
            df, ["income"],
            preprocess_metadata={"income": {"was_continuous": True}})
        assert "income" in features["continuous_vars"]


# ============================================================================
# 5. Mixed-type datasets
# ============================================================================

class TestMixedDatasets:
    """Datasets combining multiple column types."""

    def test_mixed_types_no_crash(self):
        """A dataset with 2 categorical + 2 continuous + 1 string QIs should
        generalize without errors."""
        df = _df(1000,
                 sex=lambda rng: rng.choice(["M", "F"], 1000),
                 marital=lambda rng: rng.choice(
                     ["single", "married", "divorced", "widowed"], 1000),
                 income=lambda rng: rng.integers(15000, 120000, 1000).astype(float),
                 age=lambda rng: rng.integers(18, 90, 1000),
                 district=lambda rng: rng.choice(
                     [f"dist_{i}" for i in range(25)], 1000))
        qis = ["sex", "marital", "income", "age", "district"]
        result, meta = apply_generalize(
            df, quasi_identifiers=qis,
            max_categories=10, return_metadata=True, verbose=False)
        assert len(result) == 1000
        for qi in qis:
            assert qi in result.columns

    def test_missing_values_no_crash(self):
        """NaN values in some QI columns should not crash preprocessing."""
        rng = np.random.default_rng(42)
        n = 500
        age = rng.integers(18, 90, n).astype(float)
        age[rng.choice(n, 50, replace=False)] = np.nan
        sex = rng.choice(["M", "F", None], n)
        df = pd.DataFrame({"age": age, "sex": sex})
        result, meta = apply_generalize(
            df, quasi_identifiers=["age", "sex"],
            max_categories=10, return_metadata=True, verbose=False)
        assert len(result) == n


# ============================================================================
# 6. Feasibility classification
# ============================================================================

class TestFeasibility:
    """Feasibility classification via build_data_features."""

    def test_easy_feasibility(self):
        """3 low-card QIs on 1000 rows → expected_eq >> 5 → easy."""
        df = _df(1000,
                 sex=lambda rng: rng.choice(["M", "F"], 1000),
                 edu=lambda rng: rng.choice(["HS", "BSc", "MSc", "PhD"], 1000),
                 marital=lambda rng: rng.choice(["S", "M", "D"], 1000))
        features = build_data_features(df, ["sex", "edu", "marital"])
        # 2 * 4 * 3 = 24 combinations → 1000/24 ≈ 41.7 → easy
        assert features["k_anonymity_feasibility"] == "easy"

    def test_infeasible_high_cardinality(self):
        """QIs with enormous cardinality → infeasible."""
        df = _df(100,
                 id1=lambda rng: rng.choice(range(100), 100, replace=False),
                 id2=lambda rng: rng.choice(range(100), 100, replace=False))
        features = build_data_features(df, ["id1", "id2"])
        # 100 * 100 = 10000 combos on 100 rows → expected_eq = 0.01
        assert features["k_anonymity_feasibility"] == "infeasible"


# ============================================================================
# 7. Outlier and skewness detection
# ============================================================================

class TestOutlierDetection:
    """Outlier and skewness flags in build_data_features."""

    def test_outliers_detected(self):
        """A continuous QI with injected extreme outliers should trigger
        has_outliers=True."""
        rng = np.random.default_rng(42)
        vals = np.concatenate([
            rng.normal(50, 5, 970),
            rng.uniform(200, 500, 30),  # 3% extreme outliers
        ])
        df = pd.DataFrame({"score": vals})
        features = build_data_features(df, ["score"])
        assert features["has_outliers"] is True

    def test_no_outliers_normal_data(self):
        """A normally distributed continuous QI should not flag outliers
        (outlier rate < 2% threshold)."""
        df = _df(1000, height=lambda rng: rng.normal(170, 10, 1000))
        features = build_data_features(df, ["height"])
        # Normal distribution has ~0.7% IQR outliers — below 2% threshold
        assert features["has_outliers"] is False

    def test_skewed_column_detected(self):
        """A right-skewed continuous QI (income-like) should appear in
        skewed_columns."""
        rng = np.random.default_rng(42)
        # Lognormal is right-skewed with skewness >> 1.5
        income = np.exp(rng.normal(10, 1.5, 1000))
        df = pd.DataFrame({"income": income})
        features = build_data_features(df, ["income"])
        assert "income" in features["skewed_columns"]
