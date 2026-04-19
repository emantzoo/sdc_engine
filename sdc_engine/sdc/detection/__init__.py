"""
SDC Detection Module
====================

Auto-detection of quasi-identifiers, column types, and data analysis.

Column Types in SDC:
- Direct Identifiers: name, email, SSN → EXCLUDE before processing
- Quasi-Identifiers: age, gender, zipcode → PROTECT via SDC methods
- Sensitive Attributes: income, diagnosis → KEEP, protected by QI anonymization

Components:
- qi_detection: Enhanced quasi-identifier detection with confidence scoring
- column_types: Column type identification and classification
- data_analysis: Comprehensive data analysis for method selection
"""

from .qi_detection import (
    detect_quasi_identifiers_enhanced,
    auto_detect_quasi_identifiers,
    detect_quasi_identifiers_smart,
)
from .column_types import (
    identify_column_types,
    detect_data_type,
    auto_detect_continuous_variables,
    auto_detect_categorical_variables,
    auto_detect_direct_identifiers,
    auto_detect_sensitive_columns,  # Backward compatibility alias
    auto_detect_dimensions,
)
# data_analysis archived — not used by any view or interactor

__all__ = [
    # QI detection
    'detect_quasi_identifiers_enhanced',
    'auto_detect_quasi_identifiers',
    # Column types
    'identify_column_types',
    'detect_data_type',
    'auto_detect_continuous_variables',
    'auto_detect_categorical_variables',
    'auto_detect_direct_identifiers',  # New name (preferred)
    'auto_detect_sensitive_columns',   # Backward compatibility
    'auto_detect_dimensions',
]
