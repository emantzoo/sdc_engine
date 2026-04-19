"""
Shared UI components for the Streamlit SDC app.
"""
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# ── Numeric recovery ──────────────────────────────────────────────────

def recover_numeric_types(data: pd.DataFrame) -> pd.DataFrame:
    """Convert string-typed columns to proper numeric/datetime types where possible.

    The upload step loads everything as ``dtype=str`` to preserve original
    encoding.  This function applies ``pd.to_numeric`` and ``pd.to_datetime``
    to recover real types for the SDC engine.
    """
    df = data.copy()
    for col in df.columns:
        # Treat empty strings as missing for type detection
        series = df[col].replace("", pd.NA)
        non_empty = series.dropna()
        if len(non_empty) == 0:
            continue
        # Try numeric first
        converted = pd.to_numeric(non_empty, errors="coerce")
        pct_numeric = converted.notna().sum() / len(non_empty)
        if pct_numeric > 0.8:
            df[col] = pd.to_numeric(series, errors="coerce")
            continue
        # Try datetime
        try:
            dt = pd.to_datetime(non_empty, errors="coerce", dayfirst=True)
            if dt.notna().sum() / len(non_empty) > 0.8:
                df[col] = pd.to_datetime(series, errors="coerce", dayfirst=True)
        except Exception:
            pass
    return df


# ── Column stats ──────────────────────────────────────────────────────

def build_column_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with per-column stats for the Configure table."""
    rows = []
    for col in data.columns:
        series = data[col]
        n_unique = series.nunique(dropna=True)
        n_missing = series.isna().sum() + (series == "").sum()
        pct_missing = n_missing / len(series) * 100 if len(series) > 0 else 0
        dtype_label = str(series.dtype)
        # Attempt to detect actual type
        numeric_check = pd.to_numeric(series, errors="coerce")
        if numeric_check.notna().sum() / max(series.notna().sum(), 1) > 0.8:
            dtype_label = "numeric"
        rows.append({
            "Column": col,
            "Type": dtype_label,
            "Unique": n_unique,
            "Missing %": round(pct_missing, 1),
            "Role": "Unassigned",
        })
    return pd.DataFrame(rows)


# ── Risk badge ────────────────────────────────────────────────────────

_RISK_COLORS = [
    (0.05, "#27ae60", "LOW"),
    (0.10, "#f39c12", "MODERATE"),
    (0.20, "#e67e22", "HIGH"),
    (1.01, "#e74c3c", "VERY HIGH"),
]


def risk_badge(reid_95: float) -> None:
    """Render a color-coded risk badge."""
    for threshold, color, label in _RISK_COLORS:
        if reid_95 < threshold:
            break
    else:
        color, label = "#e74c3c", "VERY HIGH"
    st.markdown(
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:4px;font-weight:bold">{label} — {reid_95:.1%}</span>',
        unsafe_allow_html=True,
    )


# ── Metric cards ──────────────────────────────────────────────────────

def metric_cards(metrics: dict) -> None:
    """Render a row of st.metric() cards from a flat dict."""
    cols = st.columns(len(metrics))
    for col_widget, (label, value) in zip(cols, metrics.items()):
        if isinstance(value, float):
            col_widget.metric(label, f"{value:.2%}" if value <= 1 else f"{value:.2f}")
        else:
            col_widget.metric(label, str(value))


def metric_cards_delta(metrics: list[tuple]) -> None:
    """Render metric cards with before/after deltas.

    Each tuple: (label, current_value, delta_value, delta_is_good_when_negative)
    """
    cols = st.columns(len(metrics))
    for col_widget, (label, value, delta, invert) in zip(cols, metrics):
        fmt = f"{value:.2%}" if isinstance(value, float) and value <= 1 else str(value)
        delta_fmt = f"{delta:.2%}" if isinstance(delta, float) and abs(delta) <= 1 else str(delta)
        col_widget.metric(label, fmt, delta=delta_fmt,
                          delta_color="inverse" if invert else "normal")


# ── Before/after risk histogram ───────────────────────────────────────

def risk_histogram(scores_before: list, scores_after: list) -> go.Figure:
    """Plotly overlaid histogram of per-record risk before and after protection."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores_before, nbinsx=50, name="Before",
        marker_color="rgba(231,76,60,0.5)", opacity=0.6,
    ))
    fig.add_trace(go.Histogram(
        x=scores_after, nbinsx=50, name="After",
        marker_color="rgba(39,174,96,0.5)", opacity=0.6,
    ))
    fig.update_layout(
        barmode="overlay",
        title="Per-Record Risk Distribution",
        xaxis_title="Re-identification Risk",
        yaxis_title="Count",
        height=350,
        margin=dict(t=40, b=40),
    )
    return fig
