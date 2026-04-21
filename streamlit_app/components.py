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


# ── Feasibility badge ────────────────────────────────────────────────

def classify_feasibility(expected_eq_size: float,
                         qi_cardinalities: dict | None = None,
                         ) -> tuple[str, str, str, str]:
    """Map expected_eq_size to a (color, label, message, suggestion) tuple.

    Parameters
    ----------
    expected_eq_size : float
        n_records / combination_space (from diagnose_qis).
    qi_cardinalities : dict, optional
        {qi_name: cardinality} — used to identify the highest-cardinality
        QI for the red-band suggestion.

    Returns
    -------
    (color, label, message, suggestion)
    """
    if expected_eq_size >= 10:
        return (
            "#27ae60",
            "Comfortable",
            "Target ReID levels should be achievable.",
            "",
        )
    if expected_eq_size >= 5:
        return (
            "#f39c12",
            "Tight",
            "Target may require aggressive generalization.",
            "Consider starting retry at aggressive tier.",
        )
    # Red band — find highest-cardinality QI for suggestion
    top_qi = ""
    if qi_cardinalities:
        top_qi = max(qi_cardinalities, key=qi_cardinalities.get)
    if expected_eq_size >= 2:
        suggestion = (
            f"Consider dropping or coarsening `{top_qi}` (highest cardinality)."
            if top_qi else
            "Consider dropping or coarsening the highest-cardinality QI."
        )
        return (
            "#e74c3c",
            "Infeasible for low targets",
            "ReID_95 \u2264 5% unlikely without heavy suppression.",
            suggestion,
        )
    suggestion = (
        f"Must reduce QIs \u2014 consider dropping `{top_qi}` (highest cardinality)."
        if top_qi else
        "Must reduce QIs \u2014 combination space exceeds dataset size."
    )
    return (
        "#e74c3c",
        "Infeasible",
        "Combination space exceeds dataset size.",
        suggestion,
    )


def feasibility_badge(expected_eq_size: float,
                      qi_cardinalities: dict | None = None,
                      ) -> None:
    """Render a feasibility status badge with optional suggestion."""
    color, label, message, suggestion = classify_feasibility(
        expected_eq_size, qi_cardinalities
    )
    badge_html = (
        f'<span style="background:{color};color:white;padding:4px 12px;'
        f'border-radius:4px;font-weight:bold">{label}</span>'
        f' &nbsp; <span style="color:#555">{message}</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)
    if suggestion:
        st.caption(f"\U0001f4a1 {suggestion}")


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


# ── QI distribution comparison plots ────────────────────────────────

def qi_distribution_plots(
    original: pd.DataFrame,
    protected: pd.DataFrame,
    qis: list,
    title: str = "QI Distribution Comparison",
    key_prefix: str = "dist",
    max_plots: int = 8,
) -> None:
    """Render per-QI distribution comparison charts (before vs after).

    Numeric QIs get overlaid histograms; categorical/binned QIs get
    grouped bar charts (top 15 categories + Other).
    """
    plot_qis = [q for q in qis if q in original.columns and q in protected.columns]
    if len(plot_qis) > max_plots:
        st.caption(f"Showing top {max_plots} of {len(plot_qis)} QIs.")
        plot_qis = plot_qis[:max_plots]

    if not plot_qis:
        return

    for i, col in enumerate(plot_qis):
        orig_series = original[col].dropna()
        prot_series = protected[col].dropna()
        if len(orig_series) == 0 and len(prot_series) == 0:
            continue

        is_numeric = pd.api.types.is_numeric_dtype(original[col])

        if is_numeric:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=orig_series, nbinsx=30, name="Before",
                marker_color="rgba(231,76,60,0.5)", opacity=0.6,
            ))
            fig.add_trace(go.Histogram(
                x=prot_series, nbinsx=30, name="After",
                marker_color="rgba(39,174,96,0.5)", opacity=0.6,
            ))
            fig.update_layout(
                barmode="overlay",
                title=col,
                xaxis_title=col, yaxis_title="Count",
                height=300, margin=dict(t=40, b=30, l=40, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
        else:
            # Categorical / binned: top 15 categories + Other
            orig_counts = orig_series.value_counts()
            prot_counts = prot_series.value_counts()
            all_cats = orig_counts.index.union(prot_counts.index)

            if len(all_cats) > 15:
                top_cats = orig_counts.head(15).index
                other_orig = orig_counts.loc[~orig_counts.index.isin(top_cats)].sum()
                other_prot = prot_counts.loc[~prot_counts.index.isin(top_cats)].sum()
                cats = list(top_cats) + ["Other"]
                orig_vals = [orig_counts.get(c, 0) for c in top_cats] + [other_orig]
                prot_vals = [prot_counts.get(c, 0) for c in top_cats] + [other_prot]
            else:
                cats = list(all_cats)
                orig_vals = [orig_counts.get(c, 0) for c in cats]
                prot_vals = [prot_counts.get(c, 0) for c in cats]

            cats_str = [str(c) for c in cats]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=cats_str, y=orig_vals, name="Before",
                marker_color="rgba(231,76,60,0.7)",
            ))
            fig.add_trace(go.Bar(
                x=cats_str, y=prot_vals, name="After",
                marker_color="rgba(39,174,96,0.7)",
            ))
            fig.update_layout(
                barmode="group",
                title=col,
                xaxis_title=col, yaxis_title="Count",
                height=300, margin=dict(t=40, b=30, l=40, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )

        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{i}")


# ── Per-QI utility delta bar chart ──────────────────────────────────

def qi_utility_delta_bar(qi_comparison: list, key: str = "qi_delta") -> None:
    """Horizontal bar chart of per-QI utility deltas.

    qi_comparison: list of dicts with keys 'qi', 'delta', 'verdict'.
    Color-coded: green (<5% drop), amber (5-20%), red (>20%).
    """
    valid = [
        q for q in qi_comparison
        if isinstance(q.get("delta"), (int, float))
    ]
    if not valid:
        return

    # Sort by magnitude of delta (worst first)
    valid.sort(key=lambda q: q["delta"])

    qi_names = [q["qi"] for q in valid]
    deltas = [q["delta"] for q in valid]
    colors = []
    for d in deltas:
        abs_d = abs(d)
        if abs_d < 0.05:
            colors.append("#27ae60")  # green
        elif abs_d < 0.20:
            colors.append("#f39c12")  # amber
        else:
            colors.append("#e74c3c")  # red

    fig = go.Figure(go.Bar(
        x=deltas, y=qi_names, orientation="h",
        marker_color=colors,
        text=[f"{d:+.1%}" for d in deltas],
        textposition="auto",
    ))
    fig.update_layout(
        title="Per-QI Utility Change",
        xaxis_title="Utility Delta",
        xaxis=dict(tickformat=".0%"),
        height=max(200, len(qi_names) * 35),
        margin=dict(t=40, b=30, l=10, r=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Retry trajectory plot (P2) ──────────────────────────────────────

def retry_trajectory_plot(
    attempts: list,
    reid_target: float = 0.10,
    utility_floor: float = 0.70,
    key: str = "retry_traj",
) -> None:
    """Two-axis line plot showing risk and utility across retry iterations.

    attempts: list of dicts with keys 'attempt', 'final_reid_95', 'utility',
              'method', 'tier_label', 'success'.
    """
    if not attempts or len(attempts) < 2:
        return

    x = list(range(1, len(attempts) + 1))
    reid_vals = [a.get("final_reid_95", 0) for a in attempts]
    util_vals = [a.get("utility", 0) for a in attempts]
    methods = [a.get("method", "?") for a in attempts]
    labels = [a.get("tier_label", f"Iter {i}") for i, a in enumerate(attempts, 1)]

    fig = go.Figure()

    # ReID line (left y-axis)
    fig.add_trace(go.Scatter(
        x=x, y=reid_vals, mode="lines+markers+text",
        name="ReID 95th", marker=dict(size=8, color="#e74c3c"),
        line=dict(color="#e74c3c", width=2),
        text=[f"{r:.1%}" for r in reid_vals], textposition="top center",
        hovertext=[f"{l}: {m}" for l, m in zip(labels, methods)],
    ))

    # Utility line (right y-axis)
    fig.add_trace(go.Scatter(
        x=x, y=util_vals, mode="lines+markers+text",
        name="Utility", marker=dict(size=8, color="#3498db"),
        line=dict(color="#3498db", width=2, dash="dot"),
        text=[f"{u:.0%}" for u in util_vals], textposition="bottom center",
        yaxis="y2",
    ))

    # Target lines
    fig.add_hline(y=reid_target, line_dash="dash", line_color="rgba(231,76,60,0.4)",
                  annotation_text=f"ReID target ({reid_target:.0%})")
    fig.add_hline(y=utility_floor, line_dash="dash", line_color="rgba(52,152,219,0.4)",
                  annotation_text=f"Utility floor ({utility_floor:.0%})")

    # Method switch annotations
    for i in range(1, len(methods)):
        if methods[i] != methods[i - 1]:
            fig.add_annotation(
                x=i + 1, y=reid_vals[i], text=f"-> {methods[i]}",
                showarrow=True, arrowhead=2, ax=0, ay=-30,
                font=dict(size=10, color="#8e44ad"),
            )

    fig.update_layout(
        title="Retry Trajectory",
        xaxis=dict(
            title="Iteration",
            tickvals=x, ticktext=labels, tickangle=-30,
        ),
        yaxis=dict(title="ReID 95th", tickformat=".0%", range=[0, max(reid_vals) * 1.2]),
        yaxis2=dict(
            title="Utility", tickformat=".0%",
            overlaying="y", side="right",
            range=[0, 1.1],
        ),
        height=400,
        margin=dict(t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Enhanced risk histogram with percentile lines (P4) ──────────────

def risk_histogram_enhanced(
    scores_before: list,
    scores_after: list,
    reid_before: dict = None,
    reid_after: dict = None,
    key: str = "risk_hist_enh",
) -> None:
    """Risk histogram with overlaid percentile lines."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores_before, nbinsx=50, name="Before",
        marker_color="rgba(231,76,60,0.5)", opacity=0.6,
    ))
    fig.add_trace(go.Histogram(
        x=scores_after, nbinsx=50, name="After",
        marker_color="rgba(39,174,96,0.5)", opacity=0.6,
    ))

    # Percentile lines
    if reid_before:
        for pct, label_suffix in [(reid_before.get("reid_50"), "50th"),
                                   (reid_before.get("reid_95"), "95th"),
                                   (reid_before.get("reid_99"), "99th")]:
            if pct:
                fig.add_vline(x=pct, line_dash="dash", line_color="rgba(231,76,60,0.6)",
                              annotation_text=f"B-{label_suffix}: {pct:.1%}",
                              annotation_position="top left",
                              annotation_font_size=9)
    if reid_after:
        for pct, label_suffix in [(reid_after.get("reid_50"), "50th"),
                                   (reid_after.get("reid_95"), "95th"),
                                   (reid_after.get("reid_99"), "99th")]:
            if pct:
                fig.add_vline(x=pct, line_dash="dot", line_color="rgba(39,174,96,0.7)",
                              annotation_text=f"A-{label_suffix}: {pct:.1%}",
                              annotation_position="bottom right",
                              annotation_font_size=9)

    fig.update_layout(
        barmode="overlay",
        title="Per-Record Risk Distribution (with percentiles)",
        xaxis_title="Re-identification Risk",
        yaxis_title="Count",
        height=400,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


# ── Scenario comparison radar chart (P5) ────────────────────────────

def scenario_radar_chart(
    scenarios: list,
    results: list,
    key: str = "scenario_radar",
) -> None:
    """Radar/spider chart comparing up to 4 scenarios on 4 axes.

    scenarios: list of dicts with 'name'.
    results: list of ProtectionResult objects.
    Axes: ReID reduction, Utility, 1 - Suppression rate, min_k (normalized).
    """
    if not results or len(results) < 2:
        return

    categories = ["ReID Reduction", "Utility", "Low Suppression", "min k (norm)"]

    fig = go.Figure()
    colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12"]

    for i, (sc, res) in enumerate(zip(scenarios, results)):
        rb = res.reid_before or {}
        ra = res.reid_after or {}
        reid_before = rb.get("reid_95", 0)
        reid_after = ra.get("reid_95", 0)

        # ReID reduction: how much was risk reduced (0=none, 1=fully eliminated)
        reid_reduction = max(0, (reid_before - reid_after) / reid_before) if reid_before > 0 else 0

        # Utility
        utility = res.utility_score or 0

        # Suppression rate (inverted: 1 = no suppression)
        supp_rate = ra.get("suppression_rate", 0) or 0
        low_suppression = 1.0 - supp_rate

        # min_k normalized: cap at 10 for normalization
        max_risk = ra.get("max_risk", 0)
        min_k = int(1 / max_risk) if max_risk > 0 else 0
        min_k_norm = min(min_k / 10.0, 1.0)

        vals = [reid_reduction, utility, low_suppression, min_k_norm]
        # Close the polygon
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=sc["name"],
            line_color=colors[i % len(colors)],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%")),
        title="Scenario Comparison",
        height=450,
        margin=dict(t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
