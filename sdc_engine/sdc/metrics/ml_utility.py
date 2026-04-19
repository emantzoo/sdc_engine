"""
ML Utility Validation
=====================

Compare classifier performance on original vs anonymized data.
Uses simple models (logistic regression) with k-fold CV
to measure how well analytical relationships are preserved.

Metric: accuracy_ratio = accuracy_anonymized / accuracy_original
  - 1.0 = perfect utility preservation
  - 0.0 = total information loss
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Minimum requirements for meaningful ML evaluation
_MIN_ROWS = 30
_MIN_CLASSES = 2
_MAX_CLASSES = 10
_MAX_CATEGORICAL_CARDINALITY = 20


def _sklearn_available() -> bool:
    """Check whether scikit-learn is importable."""
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def _prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
) -> pd.DataFrame:
    """One-hot encode categoricals and impute NaN values.

    Returns a purely numeric DataFrame ready for sklearn, or None
    if preparation fails.
    """
    try:
        subset = df[feature_columns].copy()

        # Separate numeric vs categorical
        numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in feature_columns if c not in numeric_cols]

        # Impute numeric with median
        for c in numeric_cols:
            if subset[c].isna().any():
                subset[c] = subset[c].fillna(subset[c].median())

        # Impute categorical with mode, then one-hot encode
        for c in cat_cols:
            if subset[c].isna().any():
                mode_val = subset[c].mode()
                fill = mode_val.iloc[0] if len(mode_val) > 0 else 'MISSING'
                subset[c] = subset[c].fillna(fill)
            subset[c] = subset[c].astype(str)

        if cat_cols:
            subset = pd.get_dummies(subset, columns=cat_cols, drop_first=True)

        # Final safety: drop any remaining NaN columns
        subset = subset.dropna(axis=1, how='any')

        if subset.shape[1] == 0:
            return None

        return subset
    except Exception as exc:
        log.debug("Feature preparation failed: %s", exc)
        return None


def _align_features(
    X_orig: pd.DataFrame,
    X_anon: pd.DataFrame,
) -> tuple:
    """Align one-hot encoded columns between original and anonymized.

    Returns (X_orig_aligned, X_anon_aligned) with identical column sets.
    Missing columns are filled with 0.
    """
    common = sorted(set(X_orig.columns) & set(X_anon.columns))
    if not common:
        # Fall back: use union, fill missing with 0
        all_cols = sorted(set(X_orig.columns) | set(X_anon.columns))
        X_orig = X_orig.reindex(columns=all_cols, fill_value=0)
        X_anon = X_anon.reindex(columns=all_cols, fill_value=0)
    else:
        X_orig = X_orig[common]
        X_anon = X_anon[common]
    return X_orig, X_anon


def compute_ml_utility(
    original: pd.DataFrame,
    anonymized: pd.DataFrame,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    n_folds: int = 3,
) -> Optional[Dict]:
    """Compare classifier accuracy on original vs anonymized data.

    Parameters
    ----------
    original : DataFrame
        Original (pre-anonymization) data.
    anonymized : DataFrame
        Anonymized data with the same schema.
    target_column : str
        Column to predict (must be categorical with 2-10 classes).
    feature_columns : list of str, optional
        Columns to use as features.  When *None*, auto-selects numeric
        columns and low-cardinality categoricals present in both frames.
    n_folds : int
        Number of stratified CV folds (default 3).

    Returns
    -------
    dict or None
        Result dict with accuracy metrics, or None on failure.
    """
    if not _sklearn_available():
        log.warning("sklearn not available — skipping ML utility metric")
        return None

    # --- Validate target column ---
    if target_column not in original.columns or target_column not in anonymized.columns:
        log.debug("Target column '%s' not found in both DataFrames", target_column)
        return None

    # --- Auto-select features if needed ---
    if feature_columns is None:
        feature_columns = _auto_select_features(original, anonymized, target_column)

    if not feature_columns:
        log.debug("No suitable feature columns for ML utility")
        return None

    # --- Validate target ---
    y_orig = original[target_column].dropna()
    y_anon = anonymized[target_column].dropna()
    n_classes_orig = y_orig.nunique()
    n_classes_anon = y_anon.nunique()

    if n_classes_orig < _MIN_CLASSES or n_classes_anon < _MIN_CLASSES:
        log.debug("Too few classes (orig=%d, anon=%d)", n_classes_orig, n_classes_anon)
        return None

    if len(y_orig) < _MIN_ROWS or len(y_anon) < _MIN_ROWS:
        log.debug("Too few rows (orig=%d, anon=%d)", len(y_orig), len(y_anon))
        return None

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder

        # Prepare feature matrices
        X_orig = _prepare_features(original, feature_columns, target_column)
        X_anon = _prepare_features(anonymized, feature_columns, target_column)

        if X_orig is None or X_anon is None:
            log.debug("Feature preparation failed")
            return None

        # Align one-hot columns
        X_orig, X_anon = _align_features(X_orig, X_anon)

        if X_orig.shape[1] == 0:
            log.debug("No features remaining after alignment")
            return None

        # Encode target
        le = LabelEncoder()
        y_orig_enc = le.fit_transform(original[target_column].astype(str))
        y_anon_enc = le.transform(
            anonymized[target_column].astype(str).map(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
        )

        # Ensure enough samples per class for stratified CV
        n_folds_actual = min(n_folds, _min_class_count(y_orig_enc))
        if n_folds_actual < 2:
            n_folds_actual = 2

        n_folds_anon = min(n_folds_actual, _min_class_count(y_anon_enc))
        if n_folds_anon < 2:
            n_folds_anon = 2

        # --- Train on original ---
        model = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
        cv_orig = StratifiedKFold(n_splits=n_folds_actual, shuffle=True, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores_orig = cross_val_score(
                model, X_orig.values, y_orig_enc, cv=cv_orig, scoring='accuracy')

        # --- Train on anonymized ---
        cv_anon = StratifiedKFold(n_splits=n_folds_anon, shuffle=True, random_state=42)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores_anon = cross_val_score(
                model, X_anon.values, y_anon_enc, cv=cv_anon, scoring='accuracy')

        acc_orig = float(np.mean(scores_orig))
        acc_anon = float(np.mean(scores_anon))

        # Compute ratio (cap at 1.0)
        if acc_orig > 0:
            ratio = min(acc_anon / acc_orig, 1.0)
        else:
            ratio = 0.0

        return {
            'accuracy_original': round(acc_orig, 4),
            'accuracy_anonymized': round(acc_anon, 4),
            'accuracy_ratio': round(ratio, 4),
            'features_used': list(feature_columns),
            'target': target_column,
            'model': 'LogisticRegression',
            'n_folds': n_folds_actual,
        }

    except Exception as exc:
        log.debug("ML utility computation failed: %s", exc)
        return None


def _auto_select_features(
    original: pd.DataFrame,
    anonymized: pd.DataFrame,
    target_column: str,
) -> List[str]:
    """Auto-select feature columns present in both DataFrames.

    Picks numeric columns and low-cardinality categoricals (<20 unique).
    Excludes the target column.
    """
    common_cols = sorted(set(original.columns) & set(anonymized.columns))
    features = []

    for col in common_cols:
        if col == target_column:
            continue

        # Numeric columns are always candidates
        if pd.api.types.is_numeric_dtype(original[col]):
            features.append(col)
            continue

        # Low-cardinality categoricals
        n_unique = original[col].nunique()
        if n_unique <= _MAX_CATEGORICAL_CARDINALITY:
            features.append(col)

    return features


def _min_class_count(y: np.ndarray) -> int:
    """Return the size of the smallest class in the target array."""
    _, counts = np.unique(y, return_counts=True)
    return int(counts.min()) if len(counts) > 0 else 0


def compute_ml_utility_multi(
    original: pd.DataFrame,
    anonymized: pd.DataFrame,
    sensitive_columns: List[str],
    quasi_identifiers: List[str],
    n_folds: int = 3,
) -> Dict[str, Optional[Dict]]:
    """Compute ML utility for each eligible sensitive column.

    A sensitive column is eligible if it's categorical with 2-10 unique
    values.  The quasi-identifiers are used as feature columns.

    Parameters
    ----------
    original : DataFrame
        Original data.
    anonymized : DataFrame
        Anonymized data.
    sensitive_columns : list of str
        Candidate target columns.
    quasi_identifiers : list of str
        Feature columns.
    n_folds : int
        Stratified CV folds.

    Returns
    -------
    dict
        ``{column_name: ml_utility_result_or_None}``
    """
    if not _sklearn_available():
        log.warning("sklearn not available — skipping ML utility metric")
        return {}

    results: Dict[str, Optional[Dict]] = {}

    for col in sensitive_columns:
        if col not in original.columns or col not in anonymized.columns:
            continue

        n_unique = original[col].nunique()
        if n_unique < _MIN_CLASSES or n_unique > _MAX_CLASSES:
            log.debug(
                "Skipping ML utility for '%s': %d unique values "
                "(need %d-%d)", col, n_unique, _MIN_CLASSES, _MAX_CLASSES)
            continue

        # Use QIs as features, filtering to those present in both frames
        feature_cols = [
            qi for qi in quasi_identifiers
            if qi in original.columns and qi in anonymized.columns
        ]
        if not feature_cols:
            continue

        result = compute_ml_utility(
            original, anonymized, col,
            feature_columns=feature_cols,
            n_folds=n_folds,
        )
        if result is not None:
            results[col] = result

    return results
