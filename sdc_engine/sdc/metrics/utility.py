"""
Utility Metrics
===============

Calculate utility preservation metrics for protected data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)


def calculate_information_loss(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate various information loss metrics.

    Parameters:
    -----------
    original : pd.DataFrame
        Original dataset
    protected : pd.DataFrame
        Anonymized dataset
    columns : list of str, optional
        Columns to analyze (if None, uses all common columns)

    Returns:
    --------
    dict : Information loss metrics
    """
    if columns is None:
        columns = [col for col in original.columns if col in protected.columns]

    metrics = {}

    for col in columns:
        if col not in original.columns or col not in protected.columns:
            continue

        # For numeric columns
        if pd.api.types.is_numeric_dtype(original[col]):
            # Mean absolute error
            mae = np.abs(original[col] - protected[col]).mean()
            metrics[f'{col}_mae'] = float(mae)

            # Relative error
            if original[col].std() > 0:
                rel_error = mae / original[col].std()
                metrics[f'{col}_relative_error'] = float(rel_error)

        # For categorical columns
        else:
            # Proportion of changed values
            changed = (original[col] != protected[col]).sum()
            metrics[f'{col}_changed_rate'] = float(changed / len(original))

    # Overall metrics
    common_cols = [col for col in columns if col in original.columns and col in protected.columns]
    metrics['total_cells_changed'] = sum(
        (original[col] != protected[col]).sum()
        for col in common_cols
    )
    metrics['overall_change_rate'] = metrics['total_cells_changed'] / (len(original) * len(common_cols)) if len(common_cols) > 0 else 0

    return metrics


def _calculate_tabular_utility_metrics(
    original: pd.DataFrame,
    protected: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate utility metrics specifically for tabular/frequency data.

    Parameters:
    -----------
    original : pd.DataFrame
        Original frequency table
    protected : pd.DataFrame
        Protected frequency table

    Returns:
    --------
    dict : Utility metrics for tabular data
    """
    metrics = {}

    # Align indices
    common_idx = original.index.intersection(protected.index)
    common_cols = [c for c in original.columns if c in protected.columns]

    if len(common_idx) == 0 or len(common_cols) == 0:
        return {
            'information_loss': 1.0,
            'utility_score': 0.0,
            'suppression_rate': 1.0,
        }

    orig_aligned = original.loc[common_idx, common_cols]
    prot_aligned = protected.loc[common_idx, common_cols]

    # Count suppressions (NaN values)
    total_cells = orig_aligned.size
    suppressed = prot_aligned.isna().sum().sum()
    metrics['suppression_rate'] = suppressed / total_cells if total_cells > 0 else 0

    # For non-suppressed cells, calculate deviation
    orig_flat = orig_aligned.values.flatten()
    prot_flat = prot_aligned.values.flatten()

    # Mask for non-NaN values (use pd.to_numeric to handle string ranges gracefully)
    prot_numeric = pd.to_numeric(pd.Series(prot_flat), errors='coerce').values
    valid_mask = ~np.isnan(prot_numeric)
    if valid_mask.sum() > 0:
        orig_valid = pd.to_numeric(pd.Series(orig_flat[valid_mask]), errors='coerce').values
        prot_valid = prot_numeric[valid_mask]

        # Mean absolute deviation
        mad = np.abs(orig_valid - prot_valid).mean()
        orig_mean = np.abs(orig_valid).mean()
        metrics['mean_abs_deviation'] = float(mad)

        # Relative deviation
        if orig_mean > 0:
            metrics['relative_deviation'] = float(mad / orig_mean)
        else:
            metrics['relative_deviation'] = 0.0

        # Total preservation (how much of original sum is preserved)
        orig_sum = orig_valid.sum()
        prot_sum = prot_valid.sum()
        if orig_sum > 0:
            metrics['sum_preserved'] = float(min(prot_sum / orig_sum, 1.0))
        else:
            metrics['sum_preserved'] = 1.0 if prot_sum == 0 else 0.0
    else:
        metrics['mean_abs_deviation'] = 0.0
        metrics['relative_deviation'] = 0.0
        metrics['sum_preserved'] = 0.0

    # Information loss (weighted combination)
    metrics['information_loss'] = (
        0.4 * metrics['suppression_rate'] +
        0.3 * min(metrics['relative_deviation'], 1.0) +
        0.3 * (1 - metrics['sum_preserved'])
    )

    # Utility score (inverse of information loss)
    metrics['utility_score'] = max(0, 1 - metrics['information_loss'])

    # Tabular-specific metrics
    metrics['cells_protected'] = metrics['suppression_rate']
    metrics['mean_preserved'] = metrics['sum_preserved']
    metrics['correlation_preserved'] = 1.0 - min(metrics['relative_deviation'], 1.0)
    metrics['distribution_similarity'] = metrics['sum_preserved']
    metrics['records_suppressed'] = metrics['suppression_rate']

    return metrics


def calculate_utility_metrics(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive utility metrics for protected data.

    Handles generalized, suppressed, and perturbed data properly.
    Supports both microdata and tabular (frequency table) data.

    Parameters:
    -----------
    original : pd.DataFrame
        Original dataset
    protected : pd.DataFrame
        Anonymized dataset (may have generalized/suppressed values)
    columns : list, optional
        Columns to analyze (default: all common columns)

    Returns:
    --------
    dict : Utility metrics including:
        - information_loss: Overall information loss (0 = perfect, 1 = total loss)
        - correlation_preserved: Average correlation preservation
        - mean_preserved: How well means are preserved
        - distribution_similarity: Statistical distribution similarity
        - records_suppressed: Proportion of records with suppressions (NaN)
        - utility_score: Combined score (0-1, higher is better)
    """
    # Detect if this is tabular data (frequency table with named index)
    is_tabular = (
        not isinstance(original.index, pd.RangeIndex) and
        original.index.dtype == 'object' and
        original.select_dtypes(include=[np.number]).shape[1] == original.shape[1]
    )

    # For tabular data, use specialized metrics
    if is_tabular:
        return _calculate_tabular_utility_metrics(original, protected)

    if columns is None:
        columns = [col for col in original.columns if col in protected.columns]

    # Track columns that were dropped entirely (present in original but not protected)
    dropped_cols = [col for col in original.columns if col not in protected.columns]
    n_dropped = len(dropped_cols)
    n_original_cols = len(original.columns)

    # Handle different row counts (after suppression)
    orig_rows = len(original)
    prot_rows = len(protected)
    row_retention = prot_rows / orig_rows if orig_rows > 0 else 0

    if len(columns) == 0:
        return {
            'information_loss': 1.0,
            'correlation_preserved': 0.0,
            'mean_preserved': 0.0,
            'distribution_similarity': 0.0,
            'records_suppressed': 1.0,
            'row_retention': 0.0,
            'utility_score': 0.0
        }

    metrics = {'row_retention': row_retention}

    # Identify column types in ORIGINAL data
    numeric_cols_orig = []
    categorical_cols_orig = []

    for col in columns:
        if col not in original.columns or col not in protected.columns:
            continue
        if pd.api.types.is_numeric_dtype(original[col]):
            numeric_cols_orig.append(col)
        else:
            categorical_cols_orig.append(col)

    # Check which numeric columns are still numeric in protected (not generalized to strings)
    numeric_cols = []
    generalized_cols = []

    for col in numeric_cols_orig:
        if pd.api.types.is_numeric_dtype(protected[col]):
            numeric_cols.append(col)
        else:
            generalized_cols.append(col)

    # 1. Information Loss for numeric columns
    # Suppressed cells (NaN in protected where original had a value) count as
    # full loss for that cell so that suppression is not hidden from MAE.
    numeric_loss = 0
    n_numeric_valid = 0

    for col in numeric_cols:
        try:
            orig_valid = original[col].dropna()
            if len(orig_valid) == 0:
                continue
            orig_std = orig_valid.std()
            if orig_std == 0:
                continue

            if len(protected) == len(original):
                # Same row count — can compare cell by cell
                orig_vals = original.loc[orig_valid.index, col]
                prot_vals = protected.loc[orig_valid.index, col]
                # Cells where protected is NaN (suppressed) get max error
                suppressed_mask = prot_vals.isna()
                n_suppressed = suppressed_mask.sum()
                n_orig = len(orig_vals)
                # Non-suppressed: normal MAE
                non_supp = ~suppressed_mask
                if non_supp.any():
                    mae_part = np.abs(orig_vals[non_supp] - prot_vals[non_supp]).sum()
                else:
                    mae_part = 0
                # Suppressed: treat as full-range error (max - min of original)
                if n_suppressed > 0:
                    supp_penalty = (orig_valid.max() - orig_valid.min()) * n_suppressed
                else:
                    supp_penalty = 0
                mae = (mae_part + supp_penalty) / n_orig
                numeric_loss += mae / orig_std
                n_numeric_valid += 1
            else:
                prot_valid = protected[col].dropna()
                if len(prot_valid) > 0:
                    mean_diff = abs(orig_valid.mean() - prot_valid.mean())
                    numeric_loss += mean_diff / orig_std
                    n_numeric_valid += 1
        except (ValueError, TypeError) as exc:
            _log.warning("[utility_metrics] Numeric loss calculation failed for '%s': %s", col, exc)

    if n_numeric_valid > 0:
        metrics['numeric_info_loss'] = min(numeric_loss / n_numeric_valid, 1.0)
    else:
        metrics['numeric_info_loss'] = 0.0

    # 2. Generalization loss
    generalization_rate = len(generalized_cols) / len(numeric_cols_orig) if numeric_cols_orig else 0
    metrics['generalization_rate'] = generalization_rate

    # 3. Categorical change rate
    cat_change_rate = 0
    n_categorical = 0

    for col in categorical_cols_orig:
        try:
            if len(protected) == len(original):
                orig_str = original[col].astype(str).fillna('__MISSING__')
                prot_str = protected[col].astype(str).fillna('__SUPPRESSED__')
                changed = (orig_str != prot_str).mean()
                cat_change_rate += changed
                n_categorical += 1
        except (ValueError, TypeError) as exc:
            _log.warning("[utility_metrics] Categorical change rate failed for '%s': %s", col, exc)

    if n_categorical > 0:
        metrics['categorical_change_rate'] = cat_change_rate / n_categorical
    else:
        metrics['categorical_change_rate'] = 0.0

    # 4. Combined information loss
    # Generalized columns (numeric→string ranges) count as full loss (1.0 each)
    # Dropped columns (in original but not protected) count as full loss
    metrics['columns_dropped'] = n_dropped
    n_total = n_numeric_valid + n_categorical + len(generalized_cols) + n_dropped
    if n_total > 0:
        metrics['information_loss'] = (
            metrics.get('numeric_info_loss', 0) * (n_numeric_valid / n_total) +
            metrics.get('categorical_change_rate', 0) * (n_categorical / n_total) +
            1.0 * (len(generalized_cols) / n_total) +
            1.0 * (n_dropped / n_total)
        )
    else:
        metrics['information_loss'] = 0.0

    # 5. Correlation Preservation
    # If columns were generalized, they lose correlation — penalize proportionally
    if len(numeric_cols) >= 2 and len(protected) == len(original):
        try:
            valid_mask = protected[numeric_cols].notna().all(axis=1) & original[numeric_cols].notna().all(axis=1)
            if valid_mask.sum() > 10:
                orig_subset = original.loc[valid_mask, numeric_cols]
                prot_subset = protected.loc[valid_mask, numeric_cols]
                orig_corr = orig_subset.corr()
                prot_corr = prot_subset.corr()
                corr_diff = np.abs(orig_corr.values - prot_corr.values)
                corr_score = max(0, 1 - np.nanmean(corr_diff))
            else:
                corr_score = 0.5
        except (ValueError, TypeError) as exc:
            _log.warning("[utility_metrics] Correlation preservation failed: %s", exc)
            corr_score = 0.5
        # Weight by fraction of numeric cols that survived (weren't generalized)
        if len(numeric_cols_orig) > 0:
            survived_frac = len(numeric_cols) / len(numeric_cols_orig)
            metrics['correlation_preserved'] = corr_score * survived_frac
        else:
            metrics['correlation_preserved'] = corr_score
    elif len(generalized_cols) > 0:
        # All numeric cols were generalized — correlation is completely lost
        metrics['correlation_preserved'] = 0.0
    else:
        metrics['correlation_preserved'] = 1.0 if len(numeric_cols_orig) < 2 else 0.5

    # 6. Mean Preservation
    # Generalized columns (numeric→string) have no computable mean — count as 0.0
    mean_errors = []
    for col in numeric_cols:
        try:
            orig_mean = original[col].mean()
            prot_mean = protected[col].mean()
            if not np.isnan(orig_mean) and not np.isnan(prot_mean):
                if orig_mean != 0:
                    rel_error = abs(orig_mean - prot_mean) / abs(orig_mean)
                    mean_errors.append(max(0, 1 - min(rel_error, 1)))
                else:
                    mean_errors.append(1.0 if abs(prot_mean) < 0.001 else 0.5)
        except (ValueError, TypeError) as exc:
            _log.warning("[utility_metrics] Mean preservation failed for '%s': %s", col, exc)
    # Each generalized col has completely lost its mean
    for _ in generalized_cols:
        mean_errors.append(0.0)

    metrics['mean_preserved'] = np.mean(mean_errors) if mean_errors else 1.0

    # 7. Distribution Similarity
    # Generalized columns (numeric→string) have no computable distribution — count as 0.0
    dist_similarities = []
    for col in numeric_cols[:5]:
        try:
            orig_vals = original[col].dropna()
            prot_vals = protected[col].dropna()
            if len(orig_vals) > 10 and len(prot_vals) > 10:
                bins = np.histogram_bin_edges(orig_vals, bins=10)
                orig_hist, _ = np.histogram(orig_vals, bins=bins, density=True)
                prot_hist, _ = np.histogram(prot_vals, bins=bins, density=True)
                if orig_hist.sum() > 0 and prot_hist.sum() > 0:
                    orig_hist = orig_hist / orig_hist.sum()
                    prot_hist = prot_hist / prot_hist.sum()
                    similarity = max(0, 1 - np.mean(np.abs(orig_hist - prot_hist)) / 2)
                    dist_similarities.append(similarity)
        except (ValueError, TypeError) as exc:
            _log.warning("[utility_metrics] Distribution similarity failed for '%s': %s", col, exc)
    # Each generalized col has completely lost its distribution
    for _ in generalized_cols:
        dist_similarities.append(0.0)

    metrics['distribution_similarity'] = np.mean(dist_similarities) if dist_similarities else 1.0

    # 8. Suppression rate
    suppression_count = 0
    total_cells = 0
    for col in columns:
        if col in protected.columns:
            suppression_count += protected[col].isna().sum()
            total_cells += len(protected)

    metrics['records_suppressed'] = suppression_count / total_cells if total_cells > 0 else 0

    # 9. Overall utility score
    metrics['utility_score'] = max(0, min(1, (
        0.25 * (1 - metrics['information_loss']) +
        0.15 * metrics['correlation_preserved'] +
        0.20 * metrics['mean_preserved'] +
        0.15 * metrics['distribution_similarity'] +
        0.15 * (1 - metrics['records_suppressed']) +
        0.10 * metrics['row_retention']
    )))

    return metrics
