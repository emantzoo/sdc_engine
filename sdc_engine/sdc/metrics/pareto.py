"""
Pareto Frontier Analysis
========================

Identifies Pareto-optimal scenarios from a set of protection results.
A result is Pareto-optimal when no other result is strictly better on
both risk and utility.

Ported from v2-generalized-pipeline (script4_execute.py).
"""

import pandas as pd
from typing import List, Optional


def pareto_optimal(
    results_df: pd.DataFrame,
    risk_col: str = "reid_95",
    utility_col: str = "utility",
    lower_risk_better: bool = True,
    higher_utility_better: bool = True,
) -> pd.DataFrame:
    """Return the Pareto-optimal subset of scenario results.

    A row is Pareto-optimal if no other row dominates it on both axes.

    Parameters
    ----------
    results_df : DataFrame
        Must contain *risk_col* and *utility_col* columns.
    risk_col : str
        Column measuring risk (default ``reid_95``).
    utility_col : str
        Column measuring utility (default ``utility``).
    lower_risk_better : bool
        If True, lower values of *risk_col* are preferred.
    higher_utility_better : bool
        If True, higher values of *utility_col* are preferred.

    Returns
    -------
    DataFrame
        Subset of *results_df* containing only Pareto-optimal rows,
        sorted by risk ascending.
    """
    if results_df.empty or risk_col not in results_df.columns or utility_col not in results_df.columns:
        return results_df

    df = results_df.dropna(subset=[risk_col, utility_col]).copy()
    if df.empty:
        return df

    # Normalize directions: we want to maximize both adjusted values
    risk_sign = -1.0 if lower_risk_better else 1.0
    util_sign = 1.0 if higher_utility_better else -1.0

    risk_vals = df[risk_col].values * risk_sign
    util_vals = df[utility_col].values * util_sign

    n = len(df)
    is_pareto = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # j dominates i if j is >= on both and > on at least one
            if (risk_vals[j] >= risk_vals[i] and util_vals[j] >= util_vals[i] and
                    (risk_vals[j] > risk_vals[i] or util_vals[j] > util_vals[i])):
                is_pareto[i] = False
                break

    pareto_df = df.iloc[[i for i in range(n) if is_pareto[i]]].copy()
    pareto_df = pareto_df.sort_values(risk_col, ascending=lower_risk_better)
    return pareto_df.reset_index(drop=True)


def risk_reduction(
    results_df: pd.DataFrame,
    baseline_row: str = "S00",
    scenario_col: str = "scenario",
    risk_col: str = "pct_k1",
) -> pd.DataFrame:
    """Compute risk reduction relative to a baseline scenario.

    Parameters
    ----------
    results_df : DataFrame
        Must contain *scenario_col* and *risk_col*.
    baseline_row : str
        Value in *scenario_col* used as the baseline.
    risk_col : str
        Column measuring risk (higher = worse).

    Returns
    -------
    DataFrame
        Copy of *results_df* with an added ``risk_reduction`` column
        (proportion, 0–1).
    """
    df = results_df.copy()
    baseline = df[df[scenario_col] == baseline_row]
    if len(baseline) == 0 or baseline.iloc[0][risk_col] == 0:
        df["risk_reduction"] = 0.0
        return df

    baseline_val = baseline.iloc[0][risk_col]
    df["risk_reduction"] = ((baseline_val - df[risk_col]) / baseline_val).round(4)
    return df
