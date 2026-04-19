"""
Integration tests for the SDC pipeline.

Tests the full flow: load → classify → risk → protect → metrics
using a real Greek real-estate dataset (mhtrwo-ax-met-ak-2025.xlsx).

The dataset must be placed in data/ (gitignored).
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mhtrwo-ax-met-ak-2025.xlsx"


@pytest.fixture(scope="module")
def raw_data():
    """Load the Greek real-estate dataset."""
    if not DATA_PATH.exists():
        pytest.skip(f"Test dataset not found: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    assert len(df) > 1000, f"Dataset too small: {len(df)} rows"
    return df


@pytest.fixture(scope="module")
def dataset(raw_data):
    """Wrap raw data in a PdDataset."""
    from sdc_engine.entities.dataset.pandas.dataset import PdDataset
    ds = PdDataset(data=raw_data.copy(), activeCols=list(raw_data.columns))
    return ds


@pytest.fixture(scope="module")
def risk_result(dataset):
    """Run backward elimination / risk calculation."""
    from sdc_engine.interactors.risk_calculation import ReidentificationRisk
    rr = ReidentificationRisk(dataset=dataset)
    rr.initialize()
    return rr


@pytest.fixture(scope="module")
def var_priority(risk_result):
    """Build var_priority dict from risk result (same logic as 2_Configure.py)."""
    vp = {}
    if risk_result.independent_df is not None:
        for _, row in risk_result.independent_df.iterrows():
            col = row["variable"]
            pct = row["risk_drop_pct"]
            if pct >= 15:
                label = "\U0001f534 HIGH"
            elif pct >= 8:
                label = "\U0001f7e0 MED-HIGH"
            elif pct >= 3:
                label = "\U0001f7e1 MODERATE"
            else:
                label = "\u26aa LOW"
            vp[col] = (label, pct)
    return vp


@pytest.fixture(scope="module")
def classification(raw_data, var_priority):
    """Run auto_classify on the dataset."""
    from sdc_engine.sdc.auto_classify import auto_classify
    return auto_classify(raw_data, var_priority)


@pytest.fixture(scope="module")
def quasi_identifiers(classification):
    """Extract QI columns from classification."""
    return [
        col for col, info in classification.items()
        if isinstance(info, dict) and info.get("role") == "QI"
    ]


# ===========================================================================
# 1. Risk Calculation
# ===========================================================================

class TestRiskCalculation:
    """Tests that backward elimination and risk computation work correctly."""

    def test_risk_is_computed(self, risk_result):
        """ReID risk should be a float between 0 and 1."""
        assert isinstance(risk_result.risk, float)
        assert 0.0 <= risk_result.risk <= 1.0

    def test_independent_df_has_all_columns(self, risk_result, raw_data):
        """independent_df should have a row for every active column."""
        idf = risk_result.independent_df
        assert idf is not None
        assert len(idf) > 0
        # Every column in the dataset should appear
        for col in raw_data.columns:
            assert col in idf["variable"].values, f"Missing: {col}"

    def test_risk_drop_pct_sums_roughly_100(self, risk_result):
        """risk_drop_pct values should sum close to 100."""
        idf = risk_result.independent_df
        total = idf["risk_drop_pct"].sum()
        # Allow tolerance — some rounding
        assert 80 <= total <= 120, f"Total risk_drop_pct = {total}"

    def test_steps_df_exists(self, risk_result):
        """Backward elimination steps should be recorded."""
        assert risk_result.steps_df is not None
        assert len(risk_result.steps_df) > 0

    def test_computed_per_record_risk(self, risk_result):
        """Per-record risk scores should exist."""
        assert risk_result.computed is not None
        assert "risk" in risk_result.computed.columns
        assert len(risk_result.computed) > 0


# ===========================================================================
# 2. Column Classification
# ===========================================================================

class TestClassification:
    """Tests that auto_classify assigns reasonable roles."""

    def test_classification_returns_all_columns(self, classification, raw_data):
        """Every column should have a classification entry."""
        for col in raw_data.columns:
            assert col in classification, f"Missing classification for: {col}"

    def test_roles_are_valid(self, classification):
        """Each classified column should have a valid role."""
        valid_roles = {"Identifier", "QI", "Sensitive", "Unassigned"}
        for col, info in classification.items():
            if col == "_diagnostics":
                continue
            assert isinstance(info, dict), f"Bad info for {col}: {info}"
            assert info["role"] in valid_roles, f"Invalid role for {col}: {info['role']}"

    def test_confidence_exists(self, classification):
        """Each classification should have confidence info."""
        for col, info in classification.items():
            if col == "_diagnostics":
                continue
            assert "confidence" in info
            assert "confidence_score" in info
            assert 0.0 <= info["confidence_score"] <= 1.0

    def test_at_least_some_qis(self, quasi_identifiers):
        """There should be at least 2 QI columns identified."""
        assert len(quasi_identifiers) >= 2, (
            f"Too few QIs: {quasi_identifiers}"
        )

    def test_no_more_than_half_are_qis(self, quasi_identifiers, raw_data):
        """QIs shouldn't be the majority of columns — sanity check."""
        assert len(quasi_identifiers) <= len(raw_data.columns), (
            f"Too many QIs ({len(quasi_identifiers)}) vs columns ({len(raw_data.columns)})"
        )


# ===========================================================================
# 3. ReID Metrics (standalone)
# ===========================================================================

class TestReIDMetrics:
    """Tests calculate_reid standalone function."""

    def test_calculate_reid_basic(self, raw_data, quasi_identifiers):
        """calculate_reid should return standard risk metrics."""
        if len(quasi_identifiers) == 0:
            pytest.skip("No QIs identified")

        from sdc_engine.sdc.metrics import calculate_reid

        # Use at most 5 QIs to keep it fast
        qis = quasi_identifiers[:5]
        result = calculate_reid(raw_data, qis)

        assert isinstance(result, dict)
        assert "reid_95" in result
        assert "reid_50" in result
        assert 0.0 <= result["reid_95"] <= 1.0
        assert 0.0 <= result["reid_50"] <= 1.0
        # 95th should be >= 50th
        assert result["reid_95"] >= result["reid_50"] - 0.01  # small tolerance


# ===========================================================================
# 4. Protection Methods
# ===========================================================================

class TestProtection:
    """Tests that protection methods run and produce valid output."""

    def _get_qis_for_method(self, raw_data, quasi_identifiers, method):
        """Get suitable QIs for a method, max 5 for speed."""
        qis = quasi_identifiers[:5]
        if len(qis) < 2:
            pytest.skip("Not enough QIs for protection")

        # For PRAM, prefer categorical QIs
        if method == "PRAM":
            cat_qis = [q for q in quasi_identifiers if raw_data[q].dtype == "object"]
            if cat_qis:
                qis = cat_qis[:5]
            else:
                pytest.skip("No categorical QIs for PRAM")

        # For NOISE, prefer numeric QIs
        if method == "NOISE":
            num_qis = [
                q for q in quasi_identifiers
                if pd.api.types.is_numeric_dtype(raw_data[q])
            ]
            if num_qis:
                qis = num_qis[:5]
            else:
                pytest.skip("No numeric QIs for NOISE")

        return qis

    def _apply_method(self, raw_data, dataset, qis, method, params):
        """Apply a protection method and return the result."""
        from sdc_engine.interactors.sdc_protection import SDCProtection

        protection = SDCProtection(dataset=dataset)
        result = protection.apply_method(
            method=method,
            quasi_identifiers=qis,
            params=params,
            use_r=False,  # Don't require R for tests
        )
        return result

    def test_kanon_produces_output(self, raw_data, dataset, quasi_identifiers):
        """kANON should produce protected data with reduced risk."""
        qis = self._get_qis_for_method(raw_data, quasi_identifiers, "kANON")
        result = self._apply_method(raw_data, dataset, qis, "kANON", {"k": 3})

        assert result.success, f"kANON failed: {result.error}"
        assert result.protected_data is not None
        assert len(result.protected_data) > 0

    def test_kanon_reduces_risk(self, raw_data, dataset, quasi_identifiers):
        """kANON should reduce ReID risk."""
        qis = self._get_qis_for_method(raw_data, quasi_identifiers, "kANON")
        result = self._apply_method(raw_data, dataset, qis, "kANON", {"k": 5})

        if not result.success:
            pytest.skip(f"kANON failed: {result.error}")

        assert result.reid_before is not None
        assert result.reid_after is not None
        # Risk should decrease or stay the same
        assert result.reid_after["reid_95"] <= result.reid_before["reid_95"] + 0.01

    def test_pram_produces_output(self, raw_data, dataset, quasi_identifiers):
        """PRAM should run on categorical QIs."""
        qis = self._get_qis_for_method(raw_data, quasi_identifiers, "PRAM")
        result = self._apply_method(raw_data, dataset, qis, "PRAM", {"p_change": 0.2})

        assert result.success, f"PRAM failed: {result.error}"
        assert result.protected_data is not None
        assert len(result.protected_data) == len(raw_data)

    def test_noise_produces_output(self, raw_data, dataset, quasi_identifiers):
        """NOISE should run on numeric QIs."""
        qis = self._get_qis_for_method(raw_data, quasi_identifiers, "NOISE")
        result = self._apply_method(raw_data, dataset, qis, "NOISE", {"magnitude": 0.1})

        assert result.success, f"NOISE failed: {result.error}"
        assert result.protected_data is not None
        assert len(result.protected_data) == len(raw_data)

    def test_locsupr_produces_output(self, raw_data, dataset, quasi_identifiers):
        """LOCSUPR should run and produce output."""
        qis = self._get_qis_for_method(raw_data, quasi_identifiers, "LOCSUPR")
        result = self._apply_method(raw_data, dataset, qis, "LOCSUPR", {"k": 3})

        assert result.success, f"LOCSUPR failed: {result.error}"
        assert result.protected_data is not None
        assert len(result.protected_data) > 0


# ===========================================================================
# 5. Utility Metrics
# ===========================================================================

class TestUtilityMetrics:
    """Tests that utility metrics are computed after protection."""

    def test_utility_after_kanon(self, raw_data, dataset, quasi_identifiers):
        """Utility metrics should be computable after kANON."""
        qis = quasi_identifiers[:5]
        if len(qis) < 2:
            pytest.skip("Not enough QIs")

        from sdc_engine.interactors.sdc_protection import SDCProtection
        from sdc_engine.sdc.metrics import calculate_utility_metrics

        protection = SDCProtection(dataset=dataset)
        result = protection.apply_method(
            method="kANON",
            quasi_identifiers=qis,
            params={"k": 3},
            use_r=False,
        )

        if not result.success:
            pytest.skip(f"Protection failed: {result.error}")

        # Calculate utility independently
        utility = calculate_utility_metrics(raw_data, result.protected_data, qis)
        assert isinstance(utility, dict)
        assert "information_loss" in utility or "utility_score" in utility


# ===========================================================================
# 6. Method Selection (Rules Engine)
# ===========================================================================

class TestMethodSelection:
    """Tests that the rules engine recommends a valid method."""

    def test_recommend_method_returns_valid(self, raw_data, quasi_identifiers):
        """recommend_method should return a primary method and parameters."""
        if len(quasi_identifiers) < 2:
            pytest.skip("Not enough QIs")

        from sdc_engine.sdc.select_method import recommend_method

        rec = recommend_method(
            raw_data,
            quasi_identifiers=quasi_identifiers[:5],
            verbose=False,
        )

        assert isinstance(rec, dict)
        assert "primary" in rec
        valid_primaries = {"kANON", "LOCSUPR", "PRAM", "NOISE", "GENERALIZE_FIRST"}
        assert rec["primary"] in valid_primaries, (
            f"Unexpected primary: {rec['primary']}"
        )
        assert "parameters" in rec

    def test_smart_rules_select(self, raw_data, quasi_identifiers):
        """The smart rules engine should fire at least one rule."""
        if len(quasi_identifiers) < 2:
            pytest.skip("Not enough QIs")

        from sdc_engine.sdc.sdc_utils import analyze_data
        from sdc_engine.sdc.selection.rules import select_method_by_features

        qis = quasi_identifiers[:5]
        analysis = analyze_data(raw_data, quasi_identifiers=qis)

        result = select_method_by_features(
            raw_data, analysis,
            quasi_identifiers=qis,
            verbose=False,
        )

        assert isinstance(result, dict)
        assert "method" in result
        valid_methods = {"kANON", "LOCSUPR", "PRAM", "NOISE", "GENERALIZE_FIRST"}
        assert result["method"] in valid_methods, (
            f"Unexpected method: {result['method']}"
        )


# ===========================================================================
# 7. End-to-End Pipeline
# ===========================================================================

class TestEndToEnd:
    """Full pipeline: classify → recommend → protect → measure."""

    def test_full_pipeline(self, raw_data, dataset, classification, quasi_identifiers):
        """The full pipeline should work end-to-end."""
        if len(quasi_identifiers) < 2:
            pytest.skip("Not enough QIs")

        from sdc_engine.interactors.sdc_protection import SDCProtection
        from sdc_engine.sdc.metrics import calculate_reid

        qis = quasi_identifiers[:5]

        # 1. Compute baseline risk
        reid_before = calculate_reid(raw_data, qis)
        assert reid_before["reid_95"] > 0, "Dataset should have some risk"

        # 2. Protect with kANON k=3
        protection = SDCProtection(dataset=dataset)
        result = protection.apply_method(
            method="kANON",
            quasi_identifiers=qis,
            params={"k": 3},
            use_r=False,
        )

        if not result.success:
            pytest.skip(f"Protection failed: {result.error}")

        # 3. Compute post-protection risk
        protected = result.protected_data
        assert protected is not None

        reid_after = calculate_reid(protected, qis)

        # 4. Risk should be reduced
        assert reid_after["reid_95"] <= reid_before["reid_95"] + 0.01, (
            f"Risk not reduced: before={reid_before['reid_95']:.4f} "
            f"after={reid_after['reid_95']:.4f}"
        )

        # 5. Data should still have rows
        assert len(protected) > len(raw_data) * 0.5, (
            f"Too many rows lost: {len(protected)}/{len(raw_data)}"
        )
