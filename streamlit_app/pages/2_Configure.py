"""
Page 2 — Configure Column Roles & Protection Settings
======================================================
Three modes: LLM (Cerebras AI), Auto (rule-based), Manual.
Assign QI / Sensitive / Unassigned roles.
Pick protection context, risk metric, and target.
Preview risk before protection.
"""
import os
import pandas as pd
import streamlit as st

from state import require_step, reset_downstream
from components import build_column_stats, risk_badge, metric_cards, recover_numeric_types

require_step("upload", "Please **upload a dataset** first.")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _run_backward_elimination(data_typed):
    """Run backward elimination → return var_priority + steps_df."""
    from sdc_engine.entities.dataset.pandas.dataset import PdDataset
    from sdc_engine.interactors.risk_calculation import ReidentificationRisk

    ds = PdDataset(data=data_typed.copy(), activeCols=list(data_typed.columns))
    rr = ReidentificationRisk(dataset=ds)
    rr.initialize()

    var_priority = {}
    if rr.independent_df is not None:
        for _, row in rr.independent_df.iterrows():
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
            var_priority[col] = (label, pct)

    return var_priority, rr.steps_df


def _build_suggestions(data_typed, classify_result):
    """Convert classify result to suggestions dict with preprocessing advice."""
    from sdc_engine.sdc.smart_defaults import _classify_qi_type

    qi_types = {}
    for col in data_typed.columns:
        qi_types[col] = _classify_qi_type(col, data_typed)

    roles = {}
    suggestions = {}
    for col in data_typed.columns:
        info = classify_result.get(col, {})
        role = info.get("role", "Unassigned")
        if role == "Identifier":
            role = "Unassigned"
        roles[col] = role

        advice = ""
        if role == "QI":
            t = qi_types[col]
            nunique = data_typed[col].nunique()
            if t.get("is_geo"):
                advice = "Coarsen (keep coarsest geo level)" if nunique > 20 else "Geographic QI"
            elif t.get("is_date"):
                advice = "Truncate (year/quarter)"
            elif t.get("is_age"):
                advice = "Bin into age groups"
            elif t.get("is_high_card") and t.get("is_numeric"):
                advice = "Round or bin"
            elif nunique > 20:
                advice = "Generalize (merge rare categories)"

        # Include warnings from classify result
        warnings = info.get("warnings", [])
        reason = info.get("reason", "")
        if warnings:
            reason = reason + " | " + "; ".join(warnings[:2])

        suggestions[col] = {
            "role": role,
            "confidence": info.get("confidence", "Low"),
            "reason": reason,
            "advice": advice,
            "ai_treatment": info.get("ai_suggested_treatment", ""),
        }

    return roles, suggestions


def _suggest_auto(data_typed):
    """Auto mode: backward elimination + rule-based auto_classify."""
    from sdc_engine.sdc.auto_classify import auto_classify

    var_priority, steps_df = _run_backward_elimination(data_typed)
    result = auto_classify(data_typed, var_priority)
    roles, suggestions = _build_suggestions(data_typed, result)
    return roles, suggestions, var_priority, steps_df


def _suggest_llm(data_typed, dataset_description=""):
    """LLM mode: backward elimination + Cerebras AI + merge with rules."""
    from sdc_engine.sdc.auto_classify import auto_classify
    from sdc_engine.sdc.llm_classify import (
        llm_classify_columns,
        merge_llm_with_rules,
    )

    var_priority, steps_df = _run_backward_elimination(data_typed)

    # Rule-based classification (always runs as baseline)
    rules_result = auto_classify(data_typed, var_priority)

    # LLM classification
    llm_result = llm_classify_columns(
        data_typed, var_priority,
        dataset_description=dataset_description or None,
    )

    if llm_result is not None:
        # Merge: rules primary, LLM adds signals
        merged = merge_llm_with_rules(rules_result, llm_result)
        roles, suggestions = _build_suggestions(data_typed, merged)

        # Extract per-QI treatment hints for the Protect page
        qi_treatments = {}
        for col, info in merged.items():
            if col == "_diagnostics":
                continue
            t = info.get("ai_suggested_treatment")
            if t and info.get("role") == "QI":
                qi_treatments[col] = t
        if qi_treatments:
            st.session_state["_ai_qi_treatments"] = qi_treatments

        return roles, suggestions, var_priority, steps_df, True
    else:
        # LLM failed — fall back to rules only
        roles, suggestions = _build_suggestions(data_typed, rules_result)
        return roles, suggestions, var_priority, steps_df, False


def _check_llm_available():
    """Check if Cerebras API key is configured."""
    return bool(os.environ.get("CEREBRAS_API_KEY"))


# ══════════════════════════════════════════════════════════════════════
# Main UI
# ══════════════════════════════════════════════════════════════════════

st.header("Configure Protection")

data: pd.DataFrame = st.session_state["data"]

# ── 1. Column role assignment ─────────────────────────────────────────

st.subheader("Column Roles")
st.caption(
    "Set each column's role: **QI** (quasi-identifier — will be protected), "
    "**Sensitive** (outcome variable — preserved but monitored), or "
    "**Unassigned** (ignored by the protection engine)."
)

# ── Classification mode selector ─────────────────────────────────────

llm_available = _check_llm_available()

col_mode, col_desc = st.columns([2, 3])
with col_mode:
    mode_options = ["Manual", "Auto (rules)", "AI (Cerebras)"]
    classify_mode = st.radio(
        "Classification mode",
        mode_options,
        horizontal=True,
        help=(
            "**Manual:** assign roles yourself. "
            "**Auto:** backward elimination + keyword rules. "
            "**AI:** Cerebras LLM classifies columns (+ rules merge)."
        ),
    )

with col_desc:
    if classify_mode == "AI (Cerebras)":
        if llm_available:
            desc = st.text_input(
                "Dataset description (optional)",
                value=st.session_state.get("dataset_description", ""),
                placeholder="e.g. real estate transactions, hospital records...",
                help="Helps the AI make better domain-specific decisions.",
            )
            st.session_state["dataset_description"] = desc
        else:
            st.warning(
                "Set `CEREBRAS_API_KEY` environment variable to enable AI mode. "
                "Falling back to Auto."
            )
            classify_mode = "Auto (rules)"

# ── Run classification button ────────────────────────────────────────

if classify_mode != "Manual":
    btn_label = "AI Suggest" if classify_mode == "AI (Cerebras)" else "Suggest Roles"
    suggest_clicked = st.button(btn_label, type="secondary", use_container_width=False)

    if suggest_clicked:
        spinner_msg = (
            "AI analyzing columns (Cerebras + backward elimination)..."
            if classify_mode == "AI (Cerebras)"
            else "Analyzing columns (backward elimination + keyword matching)..."
        )
        with st.spinner(spinner_msg):
            data_typed = recover_numeric_types(data)
            st.session_state["data_typed"] = data_typed

            if classify_mode == "AI (Cerebras)":
                desc = st.session_state.get("dataset_description", "")
                roles, suggestions, var_priority, steps_df, llm_ok = _suggest_llm(
                    data_typed, desc
                )
                if not llm_ok:
                    st.warning("AI classification failed — using rule-based fallback.")
            else:
                roles, suggestions, var_priority, steps_df = _suggest_auto(data_typed)

            st.session_state["column_roles"] = roles
            st.session_state["_suggestions"] = suggestions
            st.session_state["_var_priority"] = var_priority
            st.session_state["_steps_df"] = steps_df

            stats_df = build_column_stats(data_typed)
            stats_df["Role"] = stats_df["Column"].map(lambda c: roles.get(c, "Unassigned"))
            st.session_state["_col_stats"] = stats_df

        n_qi = sum(1 for r in roles.values() if r == "QI")
        n_sens = sum(1 for r in roles.values() if r == "Sensitive")
        st.success(f"Suggested **{n_qi} QIs** and **{n_sens} Sensitive** columns. Review and adjust below.")
        st.rerun()

# ── Show suggestion details ──────────────────────────────────────────

suggestions = st.session_state.get("_suggestions")
if suggestions:
    with st.expander("Suggestion Details", expanded=True):
        detail_rows = []
        has_ai_treatment = any(info.get("ai_treatment") for info in suggestions.values())
        for col, info in suggestions.items():
            row = {
                "Column": col,
                "Suggested Role": info["role"],
                "Advice": info.get("advice", ""),
                "Confidence": info["confidence"],
                "Reason": info["reason"][:120],
            }
            if has_ai_treatment:
                row["AI Treatment"] = info.get("ai_treatment", "")
            detail_rows.append(row)
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    # Variable importance chart
    var_priority = st.session_state.get("_var_priority")
    if var_priority:
        with st.expander("Variable Importance (Risk Contribution)", expanded=False):
            import plotly.graph_objects as go
            vp_sorted = sorted(var_priority.items(), key=lambda x: x[1][1], reverse=True)
            cols_sorted = [c for c, _ in vp_sorted]
            pcts_sorted = [v[1] for _, v in vp_sorted]
            colors = []
            for pct in pcts_sorted:
                if pct >= 15:
                    colors.append("#e74c3c")
                elif pct >= 8:
                    colors.append("#e67e22")
                elif pct >= 3:
                    colors.append("#f39c12")
                else:
                    colors.append("#95a5a6")
            fig = go.Figure(go.Bar(
                x=pcts_sorted, y=cols_sorted, orientation="h",
                marker_color=colors, text=[f"{p:.1f}%" for p in pcts_sorted],
                textposition="auto",
            ))
            fig.update_layout(
                title="Risk Contribution per Variable",
                xaxis_title="Contribution %", yaxis_title="",
                height=max(250, len(cols_sorted) * 30),
                margin=dict(l=10, t=40, b=30),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Backward elimination curve
    steps_df = st.session_state.get("_steps_df")
    if steps_df is not None and len(steps_df) > 1:
        with st.expander("Backward Elimination Curve", expanded=False):
            import plotly.graph_objects as go
            excluded = steps_df["excluded"].tolist()
            risk_vals = steps_df["ReID"].tolist()
            labels = ["All columns"] + [f"- {c}" for c in excluded[1:]]
            fig = go.Figure(go.Scatter(
                x=list(range(len(labels))), y=risk_vals,
                mode="lines+markers+text",
                text=[f"{r:.2%}" for r in risk_vals],
                textposition="top center",
                marker=dict(size=8, color="#e74c3c"),
                line=dict(color="#e74c3c", width=2),
            ))
            fig.update_layout(
                title="Risk After Each Column Removal (greedy order)",
                xaxis=dict(
                    tickvals=list(range(len(labels))),
                    ticktext=labels, tickangle=-45,
                ),
                yaxis_title="ReID 95th percentile",
                height=400,
                margin=dict(t=40, b=120),
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Editable role table ───────────────────────────────────────────────

stats_key = "_col_stats"
# Use typed data for stats so Type column shows numeric/datetime correctly
_stats_data = st.session_state.get("data_typed")
if _stats_data is None:
    _stats_data = recover_numeric_types(data)
    st.session_state["data_typed"] = _stats_data
if stats_key not in st.session_state or set(st.session_state[stats_key]["Column"]) != set(data.columns):
    stats_df = build_column_stats(_stats_data)
    prev_roles = st.session_state.get("column_roles", {})
    if prev_roles:
        stats_df["Role"] = stats_df["Column"].map(
            lambda c: prev_roles.get(c, "Unassigned")
        )
    st.session_state[stats_key] = stats_df

stats_df = st.session_state[stats_key]

edited = st.data_editor(
    stats_df,
    column_config={
        "Column": st.column_config.TextColumn("Column", disabled=True),
        "Type": st.column_config.TextColumn("Type", disabled=True),
        "Unique": st.column_config.NumberColumn("Unique", disabled=True),
        "Missing %": st.column_config.NumberColumn("Missing %", disabled=True, format="%.1f"),
        "Role": st.column_config.SelectboxColumn(
            "Role",
            options=["QI", "Sensitive", "Unassigned"],
            required=True,
        ),
    },
    use_container_width=True,
    hide_index=True,
    num_rows="fixed",
    key="role_editor",
)

# Persist roles
roles = {row["Column"]: row["Role"] for _, row in edited.iterrows()}
st.session_state["column_roles"] = roles

qis = [c for c, r in roles.items() if r == "QI"]
sensitive = [c for c, r in roles.items() if r == "Sensitive"]

# Validation
if not qis:
    st.warning("Select at least one column as **QI** to continue.")
elif len(qis) > 8:
    st.warning(f"You selected {len(qis)} QIs. More than 8 may cause feasibility issues.")

st.divider()

# ── 2. Protection context ────────────────────────────────────────────

st.subheader("Protection Settings")

CONTEXT_MAP = {
    "Public Release": "public_release",
    "Scientific Use": "scientific_use",
    "Secure Environment": "secure_environment",
}

CONTEXT_DEFAULTS = {
    "public_release":     {"reid95": 0.05, "k_anonymity": 10, "uniqueness": 0.05, "l_diversity": 3},
    "scientific_use":     {"reid95": 0.10, "k_anonymity": 5,  "uniqueness": 0.10, "l_diversity": 2},
    "secure_environment": {"reid95": 0.20, "k_anonymity": 3,  "uniqueness": 0.20, "l_diversity": 2},
}

METRIC_MAP = {
    "ReID 95th percentile": "reid95",
    "k-Anonymity": "k_anonymity",
    "Uniqueness Rate": "uniqueness",
    "l-Diversity": "l_diversity",
}

col_a, col_b = st.columns(2)

with col_a:
    context_label = st.selectbox(
        "Protection Context",
        list(CONTEXT_MAP.keys()),
        index=list(CONTEXT_MAP.values()).index(st.session_state["protection_context"]),
        help="Determines default risk thresholds. Public is strictest; Secure is most relaxed.",
    )
    context_key = CONTEXT_MAP[context_label]
    st.session_state["protection_context"] = context_key

with col_b:
    metric_label = st.selectbox(
        "Risk Metric",
        list(METRIC_MAP.keys()),
        index=list(METRIC_MAP.values()).index(st.session_state["risk_metric"]),
        help="Primary metric used to measure disclosure risk.",
    )
    metric_key = METRIC_MAP[metric_label]
    st.session_state["risk_metric"] = metric_key

# Dynamic target input
defaults = CONTEXT_DEFAULTS[context_key]
default_target = defaults.get(metric_key, 0.10)

if metric_key == "k_anonymity":
    target = st.number_input(
        "Minimum k", min_value=2, max_value=50, value=int(default_target),
        help="Every record must appear in a group of at least k identical records.",
    )
    st.session_state["risk_target"] = 1.0 / target
elif metric_key == "l_diversity":
    target = st.number_input(
        "Minimum l", min_value=2, max_value=20, value=int(default_target),
        help="Each equivalence class must have at least l distinct sensitive values.",
    )
    st.session_state["risk_target"] = 1.0 / target
else:
    target = st.number_input(
        f"Max {metric_label}",
        min_value=0.01, max_value=0.50, value=default_target, step=0.01,
        format="%.2f",
        help="Maximum acceptable risk score (0-1 scale, lower is safer).",
    )
    st.session_state["risk_target"] = target

st.divider()

# ── 3. Risk preview ──────────────────────────────────────────────────

st.subheader("Risk Preview")

if not qis:
    st.info("Assign at least one QI above to preview risk.")
    st.stop()

if st.button("Preview Risk", type="secondary"):
    with st.spinner("Computing re-identification risk..."):
        from sdc_engine.sdc.metrics.reid import calculate_reid

        data_typed = recover_numeric_types(data)
        st.session_state["data_typed"] = data_typed
        reid = calculate_reid(data_typed, qis)
        st.session_state["reid_preview"] = reid

reid = st.session_state.get("reid_preview")
if reid:
    risk_badge(reid.get("reid_95", 0))
    max_risk = reid.get("max_risk", 0)
    min_k = int(1 / max_risk) if max_risk > 0 else 0
    metric_cards({
        "ReID 50th": reid.get("reid_50", 0),
        "ReID 95th": reid.get("reid_95", 0),
        "ReID 99th": reid.get("reid_99", 0),
        "High-risk records": f"{reid.get('high_risk_rate', 0):.1%}",
        "min k": min_k,
    })

    qis_str = ", ".join(f"`{q}`" for q in qis)
    st.caption(f"QIs: {qis_str} | Sensitive: {', '.join(f'`{s}`' for s in sensitive) or '(none)'}")

st.divider()

# ── 4. Confirm ────────────────────────────────────────────────────────

if qis and st.button("Confirm Configuration", type="primary"):
    if st.session_state.get("data_typed") is None:
        st.session_state["data_typed"] = recover_numeric_types(data)
    reset_downstream("configure")
    st.session_state["steps_completed"].add("configure")
    st.success("Configuration saved. Go to **Protect** to run protection.")
