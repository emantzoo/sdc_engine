"""
SDC Detection Interactor
========================

Bridges the dataset model with SDC's QI detection and data analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import logging

from sdc_engine.entities.dataset.base import BaseDataset


@dataclass
class SDCDetectionResult:
    """Result of QI detection and data analysis."""
    quasi_identifiers: List[str] = field(default_factory=list)
    direct_identifiers: List[str] = field(default_factory=list)
    continuous_variables: List[str] = field(default_factory=list)
    categorical_variables: List[str] = field(default_factory=list)
    sensitive_columns: Dict[str, Any] = field(default_factory=dict)
    column_types: Dict[str, str] = field(default_factory=dict)
    data_type: str = 'microdata'  # 'microdata' or 'tabular'
    analysis: Dict[str, Any] = field(default_factory=dict)
    reid_result: Optional[Dict[str, float]] = None
    # Exposure-based QI selection results
    exposure_score: Optional[float] = None
    variable_importance: Optional[List] = None
    exposure_based_qis: Optional[List[str]] = None
    preprocessing_recommendations: Optional[Dict] = None


@dataclass
class SDCDetection:
    """Interactor for running SDC detection on a dataset."""

    dataset: BaseDataset

    def detect_quasi_identifiers(self) -> SDCDetectionResult:
        """Run full QI detection and data analysis on the dataset."""
        df = self.dataset.get()
        active_cols = self.dataset.get_active_columns()
        df_active = df[active_cols] if active_cols else df

        result = SDCDetectionResult()

        # Detect QIs
        try:
            from sdc_engine.sdc.sdc_utils import (
                detect_quasi_identifiers_enhanced,
                analyze_data,
                auto_detect_sensitive_columns,
                identify_column_types,
            )

            # Full data analysis
            result.analysis = analyze_data(df_active)
            result.data_type = result.analysis.get('data_type', 'microdata')
            result.continuous_variables = result.analysis.get('continuous_variables', [])
            result.categorical_variables = result.analysis.get('categorical_variables', [])

            # QI detection
            qis = detect_quasi_identifiers_enhanced(df_active)
            if isinstance(qis, dict):
                # Enhanced version returns dict with tiers
                all_qis = []
                for tier_qis in qis.values():
                    if isinstance(tier_qis, list):
                        all_qis.extend(tier_qis)
                result.quasi_identifiers = list(dict.fromkeys(all_qis))  # dedupe preserving order
            else:
                result.quasi_identifiers = list(qis) if qis else []

            # Column types
            try:
                result.column_types = identify_column_types(df_active)
            except Exception:
                pass

            # Sensitive columns
            try:
                result.sensitive_columns = auto_detect_sensitive_columns(df_active)
            except Exception:
                pass

        except ImportError as e:
            logging.warning(f"SDC detection modules not available: {e}")

        # Direct identifiers
        try:
            from sdc_engine.sdc.sdc_preprocessing import detect_greek_identifiers
            greek_ids = detect_greek_identifiers(df_active)
            if greek_ids:
                result.direct_identifiers.extend(greek_ids)
        except (ImportError, Exception):
            pass

        try:
            from sdc_engine.sdc.detection import auto_detect_direct_identifiers
            direct_ids = auto_detect_direct_identifiers(df_active)
            if direct_ids:
                result.direct_identifiers.extend(
                    [c for c in direct_ids if c not in result.direct_identifiers]
                )
        except (ImportError, Exception):
            pass

        return result

    def calculate_reid(self, quasi_identifiers: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate ReID risk metrics for given QIs."""
        df = self.dataset.get()
        active_cols = self.dataset.get_active_columns()
        df_active = df[active_cols] if active_cols else df

        try:
            from sdc_engine.sdc.sdc_utils import calculate_reid
            reid = calculate_reid(df_active, quasi_identifiers)
            return reid
        except Exception as e:
            logging.error(f"ReID calculation failed: {e}")
            return {}

    # assess_exposure_and_select_qis() removed — exposure.py archived.
    # Variable importance now computed by leave-one-out reid in
    # interactors/risk_calculation.py.
