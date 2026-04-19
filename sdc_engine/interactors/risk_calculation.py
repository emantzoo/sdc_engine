"""
Re-identification Risk — Leave-One-Out Variable Importance
==========================================================

Uses a simple, fast approach:
- Per-record risk = 1 / equivalence_class_size (groupby)
- Variable importance = leave-one-out reid_95 drop per column
- Backward elimination simulated by sorting contributions

Produces the same output shape (steps_df, independent_df, risk,
computed, top, top_cols) so all downstream consumers work unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import logging

from sdc_engine.entities.dataset.base import BaseDataset

logger = logging.getLogger(__name__)


def _safe_group_sizes(data: pd.DataFrame, columns: list) -> pd.Series:
    """Compute per-record equivalence class sizes using merge (memory-safe)."""
    subset = data[columns].fillna('__NA__')
    counts = subset.groupby(columns, observed=True).size().reset_index(name='_grp_size')
    merged = subset.merge(counts, on=columns, how='left')
    return merged['_grp_size'].fillna(1)


def _reid_95(data: pd.DataFrame, columns: list) -> float:
    """Compute 95th percentile of per-record re-identification risk."""
    if not columns or len(data) == 0:
        return 0.0
    try:
        group_sizes = _safe_group_sizes(data, columns)
        return float((1.0 / group_sizes).quantile(0.95))
    except Exception:
        return 0.0


@dataclass
class ReidentificationRisk:
    """Re-identification risk computation using leave-one-out reid.

    Parameters
    ----------
    dataset : BaseDataset
        The input dataset.
    size_of_tuple : int
        Not used by the new engine (kept for API compatibility).
    alpha : int
        Percentage of top records considered "high risk" (default 5%).
    """

    dataset: BaseDataset
    size_of_tuple: int = 3
    computed: Optional[pd.DataFrame] = None
    distinct: Optional[pd.DataFrame] = None
    alpha: int = 5
    steps_df: Optional[pd.DataFrame] = None
    independent_df: Optional[pd.DataFrame] = None
    steps_res: list = field(default_factory=list)
    top: Optional[pd.DataFrame] = None
    top_cols: Optional[pd.DataFrame] = None
    risk: float = 0.0
    threshold: float = 0.0
    shape: float = 0.0      # Deprecated (GPD) — always 0
    scale: float = 0.0      # Deprecated (GPD) — always 0
    is_small: bool = True    # Prevents GPD rendering paths

    def initialize(self):
        """Compute all risk metrics using leave-one-out reid."""
        from datetime import datetime
        start = datetime.now()

        data = self.dataset.get_data()[self.dataset.get_active_columns()]
        cols = list(self.dataset.get_active_columns())
        n_records = len(data)

        if n_records == 0 or not cols:
            logger.warning('[ReID] Empty dataset or no active columns')
            return

        # Pre-filter: drop high-cardinality columns and cap total.
        # High-cardinality columns make groupby explode. Even moderate-
        # cardinality columns combined guarantee uniqueness on large datasets.
        MAX_RISK_COLS = 15
        col_cards = [(c, data[c].nunique()) for c in cols]
        filtered = []
        dropped = []
        for c, nu in col_cards:
            if nu > n_records * 0.10 or nu > 5000:
                dropped.append(c)
            else:
                filtered.append((c, nu))
        # Sort by cardinality ascending — prefer low-cardinality for risk calc
        filtered.sort(key=lambda x: x[1])
        if len(filtered) > MAX_RISK_COLS:
            extra = [c for c, _ in filtered[MAX_RISK_COLS:]]
            dropped.extend(extra)
            filtered = filtered[:MAX_RISK_COLS]
        filtered = [c for c, _ in filtered]
        if dropped:
            logger.info(f'[ReID] Dropped {len(dropped)} near-unique columns '
                        f'from risk scan: {dropped}')
        if not filtered:
            logger.warning('[ReID] All columns are near-unique — risk = 1.0')
            self.risk = 1.0
            self.threshold = 1.0
            self.computed = pd.DataFrame({'risk': [1.0] * n_records})
            self.top = self.computed.copy()
            self.independent_df = pd.DataFrame([{
                'variable': c, 'ReID_full': 1.0, 'ReID_without': 1.0,
                'risk_drop': 0.0, 'risk_drop_pct': round(100.0 / len(cols), 2),
            } for c in cols])
            self.steps_df = pd.DataFrame([{'id': 0, 'excluded': '', 'ReID': 1.0, 'Mean_entropy': 0.0}])
            return
        cols = filtered

        logger.info(f'[ReID] Computing risk: {n_records:,} records, '
                    f'{len(cols)} columns, alpha={self.alpha}%')

        # --- 1. Per-record risk (baseline) ---
        # Sample for large datasets to avoid memory/time issues
        MAX_ROWS = 50_000
        if n_records > MAX_ROWS:
            logger.info(f'[ReID] Sampling {MAX_ROWS:,} of {n_records:,} rows')
            sample_data = data[cols].sample(n=MAX_ROWS, random_state=42)
        else:
            sample_data = data[cols]

        try:
            group_sizes = _safe_group_sizes(sample_data, cols)
        except Exception as exc:
            logger.warning(f'[ReID] groupby failed: {exc} — assuming risk=1.0')
            self.risk = 1.0
            self.threshold = 1.0
            self.computed = pd.DataFrame({'risk': [1.0] * n_records})
            self.top = self.computed.copy()
            self.independent_df = pd.DataFrame([{
                'variable': c, 'ReID_full': 1.0, 'ReID_without': 1.0,
                'risk_drop': 0.0, 'risk_drop_pct': round(100.0 / len(cols), 2),
            } for c in cols])
            for c in dropped:
                self.independent_df = pd.concat([self.independent_df, pd.DataFrame([{
                    'variable': c, 'ReID_full': 1.0, 'ReID_without': 1.0,
                    'risk_drop': 0.0, 'risk_drop_pct': 0.0,
                }])], ignore_index=True)
            self.steps_df = pd.DataFrame([{'id': 0, 'excluded': '', 'ReID': 1.0, 'Mean_entropy': 0.0}])
            return
        individual_risk = 1.0 / group_sizes
        self.risk = float(individual_risk.quantile(0.95))
        self.threshold = float(individual_risk.quantile(1 - self.alpha / 100))

        # --- 2. Computed DataFrame (per-record risk) ---
        # Use sample-based risk values (not full dataset) to avoid length mismatch
        self.computed = pd.DataFrame({'risk': individual_risk.values})
        self.computed.sort_values('risk', ascending=False, inplace=True,
                                 ignore_index=True)

        # --- 3. Top records (above alpha-percentile threshold) ---
        self.top = self.computed[self.computed['risk'] > self.threshold].copy()
        self.top.reset_index(drop=True, inplace=True)

        # --- 4. Leave-one-out independent analysis ---
        baseline_95 = self.risk
        indep_rows = []
        for col in cols:
            remaining = [c for c in cols if c != col]
            if not remaining:
                indep_rows.append({
                    'variable': col, 'ReID_full': round(baseline_95, 4),
                    'ReID_without': 0.0, 'risk_drop': round(baseline_95, 4),
                    'risk_drop_pct': 100.0,
                })
                continue
            reid_without = _reid_95(sample_data, remaining)
            drop = max(0.0, baseline_95 - reid_without)
            indep_rows.append({
                'variable': col,
                'ReID_full': round(baseline_95, 4),
                'ReID_without': round(reid_without, 4),
                'risk_drop': round(drop, 4),
                'risk_drop_pct': 0.0,  # filled below
            })

        # Add dropped near-unique columns back with 0 contribution
        for c in dropped:
            indep_rows.append({
                'variable': c, 'ReID_full': round(baseline_95, 4),
                'ReID_without': round(baseline_95, 4),
                'risk_drop': 0.0, 'risk_drop_pct': 0.0,
            })

        self.independent_df = pd.DataFrame(indep_rows)
        total_drop = self.independent_df['risk_drop'].sum()
        if total_drop > 0:
            self.independent_df['risk_drop_pct'] = (
                self.independent_df['risk_drop'] / total_drop * 100
            ).round(2)
        else:
            n = len(self.independent_df)
            self.independent_df['risk_drop_pct'] = round(100.0 / max(n, 1), 2)

        # --- 5. Sequential backward elimination (steps_df) ---
        # True greedy elimination: at each step remove the column whose
        # removal causes the largest risk drop, recompute, repeat.
        steps = [{'id': 0, 'excluded': '', 'ReID': baseline_95, 'Mean_entropy': 0.0}]
        remaining_cols = list(cols)
        current_risk = baseline_95
        step_id = 1
        seq_contributions = {}  # {col: risk_drop_at_removal}

        while remaining_cols:
            best_col = None
            best_new_risk = current_risk
            # Try removing each remaining column, pick the one that
            # drops risk the most (or, if tied, has highest cardinality)
            for col in remaining_cols:
                trial = [c for c in remaining_cols if c != col]
                trial_risk = _reid_95(sample_data, trial) if trial else 0.0
                if trial_risk < best_new_risk or best_col is None:
                    best_new_risk = trial_risk
                    best_col = col
            if best_col is None:
                break
            seq_drop = max(0.0, current_risk - best_new_risk)
            seq_contributions[best_col] = round(seq_drop, 6)
            remaining_cols.remove(best_col)
            current_risk = best_new_risk
            steps.append({
                'id': step_id,
                'excluded': best_col,
                'ReID': round(current_risk, 4),
                'Mean_entropy': 0.0,
            })
            step_id += 1

        self.steps_df = pd.DataFrame(steps)

        # --- 5b. Fallback: use sequential contributions when leave-one-out
        #     gives all zeros (saturated uniqueness — every record unique) ---
        if total_drop == 0 and seq_contributions:
            total_seq = sum(seq_contributions.values())
            if total_seq > 0:
                for idx, row in self.independent_df.iterrows():
                    col = row['variable']
                    drop = seq_contributions.get(col, 0.0)
                    self.independent_df.at[idx, 'risk_drop'] = drop
                    self.independent_df.at[idx, 'risk_drop_pct'] = round(
                        drop / total_seq * 100, 2)
                logger.info('[ReID] Used sequential elimination fallback '
                            'for variable importance (leave-one-out saturated)')

        # --- 6. Top columns (per-variable stats in high-risk records) ---
        high_risk_mask = individual_risk > self.threshold
        high_risk_data = sample_data[high_risk_mask.values]
        n_high = len(high_risk_data)

        top_cols_rows = []
        for col in cols:
            series = sample_data[col]
            hr_series = high_risk_data[col] if n_high > 0 else series.iloc[:0]
            n_distinct = int(series.nunique())
            # Entropy
            pis = series.value_counts(dropna=False, normalize=True)
            entropy = float(-np.sum(pis * np.log2(pis.clip(lower=1e-15))))
            # Risk stats in high-risk records
            if n_high > 0:
                hr_risks = individual_risk[high_risk_mask.values]
                min_r = float(hr_risks.min())
                med_r = float(hr_risks.median())
                max_r = float(hr_risks.max())
            else:
                min_r = med_r = max_r = 0.0

            top_cols_rows.append({
                'variable': col,
                'Count': n_high,
                'Distincts': n_distinct,
                'Entropy': round(entropy, 4),
                'Min Risk': round(min_r, 6),
                'Median Risk': round(med_r, 6),
                'Max Risk': round(max_r, 6),
            })
        self.top_cols = pd.DataFrame(top_cols_rows)

        # --- 7. Distinct counts (per-variable, for backward compat) ---
        distinct_rows = []
        for col in cols:
            n_distinct = data[col].nunique()
            distinct_rows.append({
                'risk': n_distinct,
                'vars.1': col,
                'var_ind.1': cols.index(col),
            })
        self.distinct = pd.DataFrame(distinct_rows)

        elapsed = datetime.now() - start
        logger.info(f'[ReID] Done in {elapsed}. risk={self.risk:.4f}, '
                    f'{len(cols)} columns analyzed')

    def updateAlpha(self, val):
        """Recompute threshold and top records for new alpha."""
        self.alpha = val
        if self.computed is not None and len(self.computed) > 0:
            self.threshold = float(
                self.computed['risk'].quantile(1 - self.alpha / 100))
            self.top = self.computed[
                self.computed['risk'] > self.threshold
            ].copy().reset_index(drop=True)

    # --- Backward compat stubs ---
    def compute(self, columns=None):
        return self.risk

    def compute_per_entity(self):
        return self.computed

    def compute_distinct(self):
        return self.distinct

    def compute_distinct_perc(self):
        return self.distinct

    def top_tuples(self, alpha):
        return self.top

    def top_columns(self, top):
        return self.top_cols

    def get_risk(self, alpha, top):
        return [self.risk, 0.0, 0.0, 0.0]
