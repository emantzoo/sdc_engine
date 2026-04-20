"""
Page 3 — Preprocess & Protect
==============================
1. Editable type-aware preprocessing plan (auto-generated, user can override)
2. Optional AI review of method selection (Cerebras)
3. Run protection engine
4. Show before/after results
"""
import os
import time
import pandas as pd
import streamlit as st

from state import require_step
from components import (
    metric_cards_delta,
    risk_badge,
    risk_histogram_enhanced,
    recover_numeric_types,
    qi_distribution_plots,
    qi_utility_delta_bar,
    retry_trajectory_plot,
    scenario_radar_chart,
)

require_step("configure", "Please **configure column roles** first.")


# ══════════════════════════════════════════════════════════════════════
# Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════

def _generate_preprocess_plan(data_typed, qis, column_types=None):
    """Auto-generate the preprocessing plan from data characteristics."""
    from sdc_engine.sdc.smart_defaults import (
        _detect_data_characteristics,
        build_type_aware_preprocessing,
        plan_to_editable_df,
    )

    data_warnings = _detect_data_characteristics(data_typed, qis)
    plan = build_type_aware_preprocessing(
        data_typed, qis, data_warnings, column_types=column_types
    )
    # Filter out internal keys (like _hierarchies) before building editable DF
    clean_plan = {k: v for k, v in plan.items() if not k.startswith('_')}
    plan_df, allowed_actions = plan_to_editable_df(plan, data_typed, qis)
    return plan, plan_df, allowed_actions


def _apply_preprocess(data_typed, edited_df, original_plan):
    """Apply the (possibly edited) preprocessing plan."""
    from sdc_engine.sdc.smart_defaults import (
        edited_df_to_plan,
        apply_type_aware_preprocessing,
    )

    final_plan, warning = edited_df_to_plan(edited_df, original_plan)
    preprocessed, metadata = apply_type_aware_preprocessing(data_typed, final_plan)
    return preprocessed, metadata, warning


# ══════════════════════════════════════════════════════════════════════
# Protection runner (core — no session-state side-effects)
# ══════════════════════════════════════════════════════════════════════

def _run_single_protection(input_data, qis, sensitive, context,
                           risk_metric, risk_target, mode="auto",
                           manual_method=None, manual_params=None):
    """Run protection on *input_data*.  Returns (ProtectionResult, log_entries).

    This is the reusable core — it never reads/writes session state.
    """
    from sdc_engine.sdc.metrics.reid import calculate_reid
    from sdc_engine.sdc.protection_engine import (
        build_data_features,
        run_rules_engine_protection,
    )
    from sdc_engine.sdc.config import get_context_targets
    from sdc_engine.entities.dataset.pandas.dataset import PdDataset
    from sdc_engine.interactors.sdc_protection import SDCProtection, ProtectionResult

    var_priority = st.session_state.get("_var_priority", {})
    use_r = st.session_state.get("use_sdcmicro_r", True)

    # Baseline risk
    reid_before = calculate_reid(input_data, qis)

    # Protector callback
    dataset = PdDataset(data=input_data.copy(), activeCols=list(input_data.columns))
    protector = SDCProtection(dataset=dataset)

    def apply_fn(method, qi_list, params, **kw):
        return protector.apply_method(
            method, qi_list, params,
            sensitive_columns=sensitive or None,
            use_r=use_r,
            **kw,
        )

    def _features():
        return build_data_features(
            input_data, qis,
            reid=reid_before,
            risk_metric=risk_metric,
            sensitive_columns=sensitive or None,
            var_priority=var_priority or None,
        )

    def _targets():
        targets = get_context_targets(context, risk_metric)
        return (
            targets.get("risk_target_normalized", targets.get("reid_target", risk_target)),
            targets.get("utility_floor", 0.70),
            targets.get("access_tier", "SCIENTIFIC"),
        )

    log_entries = []

    if mode == "manual":
        m = manual_method or "kANON"
        p = manual_params or {}
        result = apply_fn(m, qis, p)
        log_entries = [f"Manual: {m} with {p}"]
        if result and result.protected_data is not None:
            reid_after = calculate_reid(result.protected_data, qis)
            result.reid_after = reid_after
            result.success = reid_after.get("reid_95", 1.0) <= risk_target

    elif mode == "ai":
        from sdc_engine.sdc.llm_method_config import apply_ai_config
        from sdc_engine.sdc.selection.pipelines import select_method_suite

        ai_config = st.session_state.get("_ai_config")
        features = _features()
        reid_tgt, utility_floor, access_tier = _targets()

        if ai_config and ai_config.get("method"):
            suite = select_method_suite(features, access_tier=access_tier, verbose=False)
            base_config = {
                "method": suite.get("primary", "kANON"),
                "method_params": suite.get("primary_params", {}),
            }
            merged_config, ai_warnings = apply_ai_config(base_config, ai_config, qis)

            ai_method = merged_config.get("ai_recommended_method", merged_config.get("method", "kANON"))
            ai_params = merged_config.get("method_params", {})
            log_entries = [
                f"Rules baseline: {base_config['method']} with {base_config['method_params']}",
                f"AI merged: {ai_method} with {ai_params}",
            ]
            for w in ai_warnings:
                log_entries.append(f"  AI: {w}")

            result = apply_fn(ai_method, qis, ai_params)
            if result and result.protected_data is not None:
                reid_after = calculate_reid(result.protected_data, qis)
                result.reid_after = reid_after
                result.success = reid_after.get("reid_95", 1.0) <= risk_target

            # Fallback to full rules engine if target not met
            if not (result and result.success):
                log_entries.append("Falling back to full rules engine pipeline")
                dataset2 = PdDataset(data=input_data.copy(), activeCols=list(input_data.columns))
                protector2 = SDCProtection(dataset=dataset2)

                def apply_fn2(method, qi_list, params, **kw):
                    return protector2.apply_method(
                        method, qi_list, params,
                        sensitive_columns=sensitive or None, use_r=use_r, **kw,
                    )

                result, log2 = run_rules_engine_protection(
                    input_data=input_data, quasi_identifiers=qis,
                    data_features=features, access_tier=access_tier,
                    reid_target=reid_tgt, utility_floor=utility_floor,
                    apply_method_fn=apply_fn2,
                    sensitive_columns=sensitive or None,
                    risk_metric=risk_metric, risk_target_raw=risk_target,
                )
                log_entries.extend(log2)
        else:
            log_entries = ["No AI config, using rules engine"]
            reid_tgt, utility_floor, access_tier = _targets()
            result, log_entries = run_rules_engine_protection(
                input_data=input_data, quasi_identifiers=qis,
                data_features=features, access_tier=access_tier,
                reid_target=reid_tgt, utility_floor=utility_floor,
                apply_method_fn=apply_fn,
                sensitive_columns=sensitive or None,
                risk_metric=risk_metric, risk_target_raw=risk_target,
            )

    elif mode == "combo":
        from sdc_engine.sdc.smart_defaults import (
            apply_smart_workflow_with_adaptive_retry,
        )
        reid_tgt, utility_floor, _access = _targets()

        protected_data, combo_result = apply_smart_workflow_with_adaptive_retry(
            data=input_data, detected_qis=qis,
            initial_reid_95=reid_before.get("reid_95", 0),
            target_reid=reid_tgt, max_attempts=4, start_tier="light",
            min_utility=utility_floor,
            var_priority=var_priority or None,
            sensitive_columns=sensitive or None,
            risk_metric=risk_metric, risk_target_raw=risk_target,
            verbose=False,
        )

        log_entries = ["Smart Combo (adaptive retry)"]
        for attempt in combo_result.get("attempts", []):
            log_entries.append(
                f"  Tier {attempt.get('tier_label', '?')}: "
                f"ReID={attempt.get('final_reid_95', 0):.2%}, "
                f"Utility={attempt.get('utility', 0):.2%}, "
                f"Method={attempt.get('method', '?')}, "
                f"{'OK' if attempt.get('success') else 'insufficient'}"
            )

        reid_after = calculate_reid(protected_data, qis)
        protect_meta = combo_result.get("protect_result", {})
        result = ProtectionResult(
            method=protect_meta.get("method", combo_result.get("tier_used", "combo")),
            protected_data=protected_data,
            reid_before=reid_before, reid_after=reid_after,
            success=combo_result.get("success", False),
            metadata={
                "tier_used": combo_result.get("tier_used"),
                "attempts": combo_result.get("attempts", []),
                "final_attempt": combo_result.get("final_attempt"),
            },
        )
        try:
            from sdc_engine.sdc.utility import (
                compute_utility as _cu,
                compute_benchmark_analysis as _cba,
                compute_composite_utility as _ccu,
                compute_per_variable_utility as _cpv,
            )
            result.utility_score = _cu(
                input_data, protected_data,
                sensitive_columns=sensitive or None, quasi_identifiers=qis,
            )
            result.per_variable_utility = _cpv(input_data, protected_data, qis)
            result.benchmark = _cba(
                input_data, protected_data, qis,
                sensitive_columns=sensitive or None)
            if result.utility_score is not None and result.benchmark:
                result.utility_score = _ccu(
                    result.utility_score, result.benchmark,
                    per_variable_utility=result.per_variable_utility)
        except Exception:
            result.utility_score = combo_result.get("_utility")

    else:
        # Auto: full rules engine pipeline
        features = _features()
        reid_tgt, utility_floor, access_tier = _targets()
        result, log_entries = run_rules_engine_protection(
            input_data=input_data, quasi_identifiers=qis,
            data_features=features, access_tier=access_tier,
            reid_target=reid_tgt, utility_floor=utility_floor,
            apply_method_fn=apply_fn,
            sensitive_columns=sensitive or None,
            risk_metric=risk_metric, risk_target_raw=risk_target,
        )

    # Ensure reid_before is on the result
    if result and not result.reid_before:
        result.reid_before = reid_before

    return result, log_entries


def _run_protection(qis, sensitive, context, risk_metric, risk_target, mode="auto"):
    """Thin wrapper: runs protection and stores results in session state."""
    # Resolve input data
    input_data = st.session_state.get("preprocessed_data")
    if input_data is None:
        input_data = st.session_state.get("data_typed")
    if input_data is None:
        input_data = recover_numeric_types(st.session_state["data"])
        st.session_state["data_typed"] = input_data

    with st.status("Running SDC protection...", expanded=True) as status:
        t0 = time.time()

        result, log_entries = _run_single_protection(
            input_data, qis, sensitive, context,
            risk_metric, risk_target, mode=mode,
            manual_method=st.session_state.get("_manual_method"),
            manual_params=st.session_state.get("_manual_params", {}),
        )

        elapsed = time.time() - t0

        st.session_state["protection_result"] = result
        st.session_state["protection_log"] = log_entries
        st.session_state["steps_completed"].add("protect")

        if result and result.success:
            status.update(label=f"Done in {elapsed:.1f}s", state="complete")
        else:
            status.update(label=f"Finished in {elapsed:.1f}s (target not met)", state="error")


# ══════════════════════════════════════════════════════════════════════
# Result display (reusable for single & scenario comparison)
# ══════════════════════════════════════════════════════════════════════

def _display_result(result, qis, sensitive, orig_data=None, log_entries=None,
                    preprocess_meta=None, key_suffix=""):
    """Render a single ProtectionResult with all metrics and details.

    Parameters
    ----------
    orig_data : pd.DataFrame, optional
        The data that went *into* protection (preprocessed or raw).
        Falls back to session state if not provided.
    log_entries : list[str], optional
        Protection engine log.  Falls back to session state.
    preprocess_meta : dict, optional
        Preprocessing metadata.  Falls back to session state.
    key_suffix : str, optional
        Suffix to make widget keys unique when called multiple times.
    """
    if orig_data is None:
        _pp = st.session_state.get("preprocessed_data")
        orig_data = _pp if _pp is not None else st.session_state.get("data_typed")
    if log_entries is None:
        log_entries = st.session_state.get("protection_log", [])
    if preprocess_meta is None:
        preprocess_meta = st.session_state.get("preprocess_metadata")

    # Success / error banner
    if result.success:
        st.success(f"Protection succeeded using **{result.method}**")
    else:
        st.error(f"Protection did not meet the target. Best method: **{result.method}**")
        if result.error:
            st.warning(result.error)

    # Metric cards
    reid_before = result.reid_before or {}
    reid_after = result.reid_after or {}
    r95_before = reid_before.get("reid_95", 0)
    r95_after = reid_after.get("reid_95", 0)
    utility = result.utility_score or 0
    max_risk_after = reid_after.get("max_risk", 0)
    min_k_after = int(1 / max_risk_after) if max_risk_after > 0 else 0

    metric_cards_delta([
        ("ReID 95th", r95_after, r95_after - r95_before, True),
        ("min k", min_k_after, 0, False),
        ("Utility", utility, 0, False),
        ("Method", result.method, "", False),
    ])
    risk_badge(r95_after)

    # Privacy metrics
    _show_privacy = True
    try:
        from sdc_engine.sdc.metrics.risk import check_kanonymity, calculate_disclosure_risk
        from sdc_engine.sdc.post_protection_diagnostics import (
            check_l_diversity, check_t_closeness,
        )
    except ImportError:
        _show_privacy = False

    if _show_privacy and result.protected_data is not None and orig_data is not None:
        _, gs_before, _ = check_kanonymity(orig_data, qis, k=5)
        _, gs_after, _ = check_kanonymity(result.protected_data, qis, k=5)
        _size_col = "_group_size_" if "_group_size_" in gs_before.columns else "count"
        k_before = int(gs_before[_size_col].min()) if len(gs_before) > 0 else 0
        k_after = int(gs_after[_size_col].min()) if len(gs_after) > 0 else 0

        dr_before = calculate_disclosure_risk(orig_data, qis, k=5)
        dr_after = calculate_disclosure_risk(result.protected_data, qis, k=5)

        with st.expander("Privacy Metrics (Before / After)", expanded=True):
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("k (min group)", f"{k_before} → {k_after}")
            p2.metric("Uniqueness", f"{dr_before.get('uniqueness_rate',0):.1%} → {dr_after.get('uniqueness_rate',0):.1%}")
            p3.metric("Records at Risk", f"{dr_before.get('records_at_risk',0):,} → {dr_after.get('records_at_risk',0):,}")
            p4.metric("Avg Group Size", f"{dr_before.get('avg_group_size',0):.1f} → {dr_after.get('avg_group_size',0):.1f}")

            if sensitive:
                try:
                    ld_before = check_l_diversity(orig_data, qis, sensitive, l_target=2)
                    ld_after = check_l_diversity(result.protected_data, qis, sensitive, l_target=2)
                    tc_before = check_t_closeness(orig_data, qis, sensitive, t_target=0.30)
                    tc_after = check_t_closeness(result.protected_data, qis, sensitive, t_target=0.30)

                    q1, q2, q3 = st.columns(3)
                    q1.metric("l-diversity (min)", f"{ld_before.get('l_achieved', '?')} → {ld_after.get('l_achieved', '?')}")
                    q2.metric("l-diversity violations", f"{ld_before.get('violations', 0)} → {ld_after.get('violations', 0)}")
                    q3.metric("t-closeness (max dist)", f"{tc_before.get('t_achieved', 0):.3f} → {tc_after.get('t_achieved', 0):.3f}")
                except Exception:
                    pass

    st.divider()

    # Risk histogram (enhanced with percentile lines)
    scores_before = reid_before.get("risk_scores", [])
    scores_after = reid_after.get("risk_scores", [])
    if scores_before and scores_after:
        risk_histogram_enhanced(
            scores_before, scores_after,
            reid_before=reid_before, reid_after=reid_after,
            key=f"risk_histogram{key_suffix}",
        )

    # Retry trajectory (Smart Combo mode)
    _attempts = (result.metadata or {}).get("attempts", [])
    if len(_attempts) >= 2:
        with st.expander("Retry Engine Trajectory", expanded=False):
            _reid_tgt = st.session_state.get("risk_target", 0.10)
            retry_trajectory_plot(
                _attempts, reid_target=_reid_tgt, utility_floor=0.70,
                key=f"retry_traj{key_suffix}",
            )

    # QI distribution comparison
    if result.protected_data is not None and orig_data is not None:
        with st.expander("QI Distribution Comparison", expanded=False):
            _data_typed = st.session_state.get("data_typed")
            _preproc = st.session_state.get("preprocessed_data")
            comp_options = ["Original vs Protected"]
            if _preproc is not None and _data_typed is not None:
                comp_options.extend(["Original vs Preprocessed", "Preprocessed vs Protected"])
            comp_mode = st.radio(
                "Compare:", comp_options, horizontal=True,
                key=f"dist_mode{key_suffix}",
            )
            if comp_mode == "Original vs Preprocessed" and _data_typed is not None and _preproc is not None:
                left, right = _data_typed, _preproc
            elif comp_mode == "Preprocessed vs Protected" and _preproc is not None:
                left, right = _preproc, result.protected_data
            else:
                left, right = orig_data, result.protected_data
            qi_distribution_plots(left, right, qis, title="", key_prefix=f"prot_dist{key_suffix}")

    # Sample comparison
    if result.protected_data is not None and orig_data is not None:
        with st.expander("Sample Data Comparison", expanded=False):
            prot = result.protected_data
            n = min(10, len(orig_data))
            sample_idx = orig_data.head(n).index
            st.write("**Original (input to protection):**")
            st.dataframe(orig_data.loc[sample_idx, qis], use_container_width=True)
            st.write("**Protected:**")
            st.dataframe(prot.loc[sample_idx, qis], use_container_width=True)

    # Preprocessing summary
    if preprocess_meta:
        with st.expander("Preprocessing Applied", expanded=False):
            pp_rows = []
            for col, info in preprocess_meta.items():
                if isinstance(info, dict):
                    pp_rows.append({
                        "Column": col,
                        "Action": info.get("action", ""),
                        "Before": info.get("before_unique", ""),
                        "After": info.get("after_unique", ""),
                        "Success": info.get("success", False),
                    })
            if pp_rows:
                st.dataframe(pd.DataFrame(pp_rows), use_container_width=True, hide_index=True)

    # Method details
    with st.expander("Protection Details", expanded=False):
        meta = result.metadata or {}
        if "rule_applied" in meta:
            st.write(f"**Rule:** {meta['rule_applied']}")
        if "reason" in meta:
            st.write(f"**Reason:** {meta['reason']}")
        if "parameters" in meta:
            st.json(meta["parameters"])
        if log_entries:
            st.write("**Engine Log:**")
            for entry in log_entries[-20:]:
                st.text(entry)

    # QI suppression warnings
    if result.qi_over_suppressed and result.qi_suppression_detail:
        st.warning("**QI over-suppression detected** — some QIs lost too many values:")
        for qi_col, supp_pct in result.qi_suppression_detail.items():
            if supp_pct > 0.20:
                st.write(f"  - **{qi_col}**: {supp_pct:.0%} suppressed")

    # Utility breakdown
    with st.expander("Utility Report", expanded=False):
        if result.per_variable_utility:
            st.write("**Per-Variable Utility:**")
            _util_rows = []
            for col, val in result.per_variable_utility.items():
                if isinstance(val, dict):
                    _util_rows.append({
                        "Column": col,
                        "Unique Before": val.get("unique_before", ""),
                        "Unique After": val.get("unique_after", ""),
                        "Category Overlap": f"{val.get('category_overlap', 0):.0%}" if isinstance(val.get("category_overlap"), float) else "",
                        "Row Preservation": f"{val.get('row_preservation', 0):.0%}" if isinstance(val.get("row_preservation"), float) else "",
                    })
                elif isinstance(val, float):
                    _util_rows.append({"Column": col, "Utility": f"{val:.2%}"})
            if _util_rows:
                st.dataframe(pd.DataFrame(_util_rows), use_container_width=True, hide_index=True)

        if result.benchmark:
            st.write("**Cross-Tab Benchmark:**")
            st.json(result.benchmark)

        if result.distributional:
            st.write("**Distributional Comparison:**")
            st.json(result.distributional)

        if result.il1s:
            st.write("**Information Loss (IL1s):**")
            st.json(result.il1s)

        _diag = (result.metadata or {}).get("auto_diagnostics", {})
        if _diag:
            _qi_util = _diag.get("qi_utility_comparison")
            if _qi_util:
                st.write("**Per-QI Utility Comparison:**")
                st.dataframe(pd.DataFrame(_qi_util), use_container_width=True, hide_index=True)
                qi_utility_delta_bar(_qi_util, key=f"qi_delta{key_suffix}")
            _mq = _diag.get("method_quality")
            if _mq:
                st.write("**Method Quality:**")
                st.json(_mq)


# ══════════════════════════════════════════════════════════════════════
# Main UI
# ══════════════════════════════════════════════════════════════════════

st.header("Preprocess & Protect")

roles = st.session_state["column_roles"]
qis = [c for c, r in roles.items() if r == "QI"]
sensitive = [c for c, r in roles.items() if r == "Sensitive"]
context = st.session_state["protection_context"]
risk_metric = st.session_state["risk_metric"]
risk_target = st.session_state["risk_target"]

# ── Sidebar summary ──────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Configuration")
    st.write(f"**Context:** {context.replace('_', ' ').title()}")
    st.write(f"**Metric:** {risk_metric}")
    st.write(f"**Target:** {risk_target:.2%}" if risk_target <= 1 else f"**Target:** {risk_target}")
    st.write(f"**QIs ({len(qis)}):** {', '.join(qis)}")
    if sensitive:
        st.write(f"**Sensitive:** {', '.join(sensitive)}")

# ── 1. Preprocessing Plan ────────────────────────────────────────────

st.subheader("Preprocessing Plan")
st.caption(
    "Review and adjust the preprocessing actions for each QI column. "
    "Change the **Action** or **Param** to override the auto-generated plan."
)

# Ensure data_typed exists
data_typed = st.session_state.get("data_typed")
if data_typed is None:
    data_typed = recover_numeric_types(st.session_state["data"])
    st.session_state["data_typed"] = data_typed

# Generate plan if not cached
if st.session_state.get("preprocess_plan") is None:
    with st.spinner("Generating preprocessing plan..."):
        plan, plan_df, allowed_actions = _generate_preprocess_plan(data_typed, qis)
        st.session_state["preprocess_plan"] = plan
        st.session_state["_plan_df"] = plan_df
        st.session_state["_allowed_actions"] = allowed_actions

plan_df = st.session_state.get("_plan_df")
allowed_actions = st.session_state.get("_allowed_actions", {})

if plan_df is not None and len(plan_df) > 0:
    # Build flat list of all unique allowed actions across all QIs
    all_actions = set()
    for actions_list in allowed_actions.values():
        all_actions.update(actions_list)
    all_actions = sorted(all_actions)

    # Merge AI preprocessing suggestions inline (if available)
    ai_preproc = {}
    _ai_cfg = st.session_state.get("_ai_config")
    if _ai_cfg:
        ai_preproc = _ai_cfg.get("preprocessing_overrides", {})

    display_df = plan_df.copy()
    if ai_preproc:
        display_df["AI Suggestion"] = display_df["Column"].map(
            lambda c: (
                f"{ai_preproc[c].get('action', '')} — {ai_preproc[c].get('reasoning', '')}"
                if c in ai_preproc else ""
            )
        )

    col_config = {
        "Column": st.column_config.TextColumn("Column", disabled=True),
        "Action": st.column_config.SelectboxColumn(
            "Action",
            options=all_actions,
            required=True,
        ),
        "Param": st.column_config.TextColumn("Param"),
        "Reason": st.column_config.TextColumn("Reason", disabled=True),
        "Card. Before": st.column_config.NumberColumn("Card. Before", disabled=True),
        "Card. After": st.column_config.NumberColumn("Card. After", disabled=True),
        "Tier": st.column_config.TextColumn("Tier", disabled=True),
    }
    if ai_preproc:
        col_config["AI Suggestion"] = st.column_config.TextColumn(
            "AI Suggestion", disabled=True
        )

    edited_plan = st.data_editor(
        display_df,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="preprocess_editor",
    )

    # Apply / Skip / AI Plan buttons
    btn_cols = st.columns(3 if ai_preproc else 2)
    with btn_cols[0]:
        apply_clicked = st.button(
            "Apply Preprocessing", type="secondary",
            help="Apply the preprocessing plan to reduce cardinality before protection.",
        )
    with btn_cols[1]:
        skip_clicked = st.button(
            "Skip Preprocessing",
            help="Go directly to protection without preprocessing.",
        )

    ai_plan_clicked = False
    if ai_preproc:
        with btn_cols[2]:
            ai_plan_clicked = st.button(
                "Apply AI Plan",
                help="Replace the current plan with AI-recommended preprocessing actions.",
            )

    if ai_plan_clicked and ai_preproc:
        # Map LLM action names to internal action names
        _AI_ACTION_MAP = {
            "date_truncation": "date_truncation",
            "top_k": "categorical_generalize",
            "numeric_binning": "quantile_binning",
            "top_bottom_coding": "top_bottom_coding",
            "geographic_truncation": "geographic_coarsening",
            "merge_rare": "categorical_generalize",
            "skip": "keep",
        }
        updated_df = edited_plan.copy()
        for idx, row in updated_df.iterrows():
            col = row["Column"]
            if col in ai_preproc:
                ai_action = ai_preproc[col].get("action", "")
                mapped = _AI_ACTION_MAP.get(ai_action, ai_action)
                # Only apply if the mapped action is valid for this column
                col_allowed = allowed_actions.get(col, [])
                if mapped in col_allowed or mapped == "keep":
                    updated_df.at[idx, "Action"] = mapped
                    updated_df.at[idx, "Reason"] = f"AI: {ai_preproc[col].get('reasoning', '')}"
        st.session_state["_plan_df"] = updated_df
        st.success("AI preprocessing plan applied. Review and click **Apply Preprocessing** to execute.")
        st.rerun()

    if apply_clicked:
        with st.spinner("Applying preprocessing..."):
            original_plan = st.session_state.get("preprocess_plan", {})
            preprocessed, metadata, warning = _apply_preprocess(
                data_typed, edited_plan, original_plan
            )
            st.session_state["preprocessed_data"] = preprocessed
            st.session_state["preprocess_metadata"] = metadata

            if warning:
                st.warning(warning)

        # Show before/after cardinality
        st.success("Preprocessing applied.")
        meta_rows = []
        for col, info in metadata.items():
            if isinstance(info, dict) and info.get("success"):
                meta_rows.append({
                    "Column": col,
                    "Action": info.get("action", ""),
                    "Before": info.get("before_unique", ""),
                    "After": info.get("after_unique", ""),
                })
        if meta_rows:
            st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

        with st.expander("QI Distributions: Before vs After Preprocessing", expanded=False):
            qi_distribution_plots(
                data_typed, preprocessed, qis,
                title="", key_prefix="preproc_dist",
            )

    if skip_clicked:
        st.session_state["preprocessed_data"] = None
        st.session_state["preprocess_metadata"] = None
        st.info("Skipping preprocessing — protection will run on raw data.")

    # Show status of preprocessing
    if st.session_state.get("preprocessed_data") is not None:
        st.success("Preprocessing is applied. Protection will use the preprocessed data.")
    elif st.session_state.get("preprocess_metadata") is None and not apply_clicked:
        st.info("Click **Apply Preprocessing** or **Skip Preprocessing** to continue.")
else:
    st.info("No QI columns need preprocessing (all low cardinality).")

st.divider()

# ══════════════════════════════════════════════════════════════════════
# Scenario Comparison (optional)
# ══════════════════════════════════════════════════════════════════════

_MODE_OPTIONS_SC = ["Auto (rules engine)", "Smart Combo (adaptive)", "Manual"]
_MODE_MAP_SC = {
    "Auto (rules engine)": "auto",
    "Smart Combo (adaptive)": "combo",
    "Manual": "manual",
}
_METHOD_OPTIONS_SC = ["kANON", "LOCSUPR", "PRAM", "NOISE", "RANKSWAP", "RECSWAP"]
_MAX_SCENARIOS = 4

if plan_df is not None and len(plan_df) > 0:
    with st.expander("Scenario Comparison", expanded=bool(st.session_state.get("scenarios"))):
        st.caption(
            "Define 2–4 scenarios with different preprocessing plans and/or "
            "protection methods, then run them all to compare results side by side. "
            "Edit the preprocessing plan above, then click **Add Scenario** to snapshot it."
        )

        scenarios = st.session_state.get("scenarios", [])

        # ── Add Scenario form ─────────────────────────────────
        if len(scenarios) < _MAX_SCENARIOS:
            with st.form("add_scenario", clear_on_submit=True):
                fc1, fc2 = st.columns(2)
                with fc1:
                    next_letter = chr(ord("A") + len(scenarios))
                    sc_name = st.text_input("Scenario name", value=f"Scenario {next_letter}")
                with fc2:
                    sc_mode_label = st.selectbox("Protection mode", _MODE_OPTIONS_SC)

                # Manual-mode params
                sc_method = None
                sc_params = {}
                if sc_mode_label == "Manual":
                    mc1, mc2 = st.columns(2)
                    with mc1:
                        sc_method = st.selectbox("Method", _METHOD_OPTIONS_SC, key="_sc_method")
                    with mc2:
                        if sc_method == "kANON":
                            sc_params = {"k": st.number_input("k", 2, 50, 5, key="_sc_k")}
                        elif sc_method == "LOCSUPR":
                            sc_params = {"k": st.number_input("Threshold", 1, 50, 3, key="_sc_t")}
                        elif sc_method == "PRAM":
                            sc_params = {"p_change": st.slider("p_change", 0.01, 0.50, 0.10, 0.01, key="_sc_p")}
                        elif sc_method == "NOISE":
                            sc_params = {"magnitude": st.slider("Noise", 0.01, 0.50, 0.10, 0.01, key="_sc_n")}
                        elif sc_method == "RANKSWAP":
                            sc_params = {"p": st.number_input("Rank distance (p)", 1, 100, 10, key="_sc_rsp"),
                                         "R0": st.slider("R0", 0.50, 1.00, 0.95, 0.01, key="_sc_rsr0")}
                        elif sc_method == "RECSWAP":
                            sc_params = {"swap_rate": st.slider("Swap rate", 0.01, 0.50, 0.05, 0.01, key="_sc_recsw")}

                st.info("The current preprocessing plan (as shown above) will be captured for this scenario.")

                if st.form_submit_button("Add Scenario"):
                    # Snapshot the current edited plan.
                    # edited_plan (from st.data_editor) is defined in outer scope.
                    try:
                        snap_df = edited_plan.copy()
                    except Exception:
                        snap_df = plan_df.copy()
                    scenarios.append({
                        "name": sc_name,
                        "plan_df": snap_df,
                        "original_plan": st.session_state.get("preprocess_plan", {}),
                        "mode": _MODE_MAP_SC.get(sc_mode_label, "auto"),
                        "manual_method": sc_method,
                        "manual_params": sc_params,
                    })
                    st.session_state["scenarios"] = scenarios
                    st.session_state["scenario_results"] = None  # invalidate old results
                    st.rerun()

        # ── Scenario list ─────────────────────────────────────
        if scenarios:
            summary_rows = []
            for s in scenarios:
                non_keep = sum(
                    1 for _, row in s["plan_df"].iterrows()
                    if row.get("Action", "keep") != "keep"
                )
                summary_rows.append({
                    "Scenario": s["name"],
                    "Mode": s["mode"],
                    "Method": s.get("manual_method") or "auto-select",
                    "Preproc Steps": non_keep,
                })
            st.dataframe(
                pd.DataFrame(summary_rows),
                use_container_width=True, hide_index=True,
            )

            # Remove buttons
            rm_cols = st.columns(len(scenarios))
            for i, (rm_col, s) in enumerate(zip(rm_cols, scenarios)):
                with rm_col:
                    if st.button(f"Remove {s['name']}", key=f"_sc_rm_{i}"):
                        scenarios.pop(i)
                        st.session_state["scenarios"] = scenarios
                        st.session_state["scenario_results"] = None
                        st.rerun()

            # ── Run All Scenarios button ──────────────────────
            if len(scenarios) >= 2:
                if st.button("Run All Scenarios", type="primary", use_container_width=True):
                    from sdc_engine.sdc.smart_defaults import (
                        edited_df_to_plan,
                        apply_type_aware_preprocessing,
                    )
                    from sdc_engine.interactors.sdc_protection import ComparisonResult

                    comparison = ComparisonResult()
                    n_sc = len(scenarios)
                    progress = st.progress(0, text="Starting scenarios...")
                    base_data = st.session_state.get("data_typed")

                    for i, sc in enumerate(scenarios):
                        progress.progress(i / n_sc, text=f"Running {sc['name']}...")

                        # 1. Preprocess
                        has_preproc = any(
                            row.get("Action", "keep") != "keep"
                            for _, row in sc["plan_df"].iterrows()
                        )
                        if has_preproc:
                            final_plan, _w = edited_df_to_plan(
                                sc["plan_df"], sc["original_plan"]
                            )
                            input_data, _meta = apply_type_aware_preprocessing(
                                base_data, final_plan
                            )
                        else:
                            input_data = base_data
                            _meta = None

                        # 2. Protect
                        result, log_entries = _run_single_protection(
                            input_data, qis, sensitive, context,
                            risk_metric, risk_target,
                            mode=sc["mode"],
                            manual_method=sc.get("manual_method"),
                            manual_params=sc.get("manual_params", {}),
                        )
                        result.metadata = result.metadata or {}
                        result.metadata["scenario_name"] = sc["name"]
                        result.metadata["_log"] = log_entries
                        result.metadata["_preprocess_meta"] = _meta
                        result.metadata["_input_data_id"] = id(input_data)
                        comparison.results.append(result)

                    progress.progress(1.0, text="All scenarios complete.")

                    # Build summary
                    summary_rows = []
                    for sc, res in zip(scenarios, comparison.results):
                        rb = res.reid_before or {}
                        ra = res.reid_after or {}
                        summary_rows.append({
                            "Scenario": sc["name"],
                            "Mode": sc["mode"],
                            "Method": res.method,
                            "ReID 95 Before": f"{rb.get('reid_95', 0):.2%}",
                            "ReID 95 After": f"{ra.get('reid_95', 0):.2%}",
                            "Utility": f"{(res.utility_score or 0):.2%}",
                            "Target Met": "Yes" if res.success else "No",
                        })
                    comparison.summary = pd.DataFrame(summary_rows)

                    # Best scenario
                    successes = [
                        (i, r) for i, r in enumerate(comparison.results) if r.success
                    ]
                    if successes:
                        best_i, best_r = min(
                            successes,
                            key=lambda t: t[1].reid_after.get("reid_95", 1) if t[1].reid_after else 1,
                        )
                        comparison.best_method = scenarios[best_i]["name"]

                    st.session_state["scenario_results"] = comparison
                    st.rerun()
            else:
                st.info("Add at least 2 scenarios to enable comparison.")

        # ── Display comparison results ────────────────────────
        comp = st.session_state.get("scenario_results")
        if comp and comp.results:
            st.subheader("Comparison Results")

            # Summary table
            if comp.summary is not None:
                st.dataframe(comp.summary, use_container_width=True, hide_index=True)

            if comp.best_method:
                st.success(f"Best scenario: **{comp.best_method}**")

            # Radar chart comparison
            scenario_radar_chart(scenarios, comp.results, key="scenario_radar")

            # Side-by-side metric cards
            card_cols = st.columns(len(comp.results))
            for col, (sc, res) in zip(card_cols, zip(scenarios, comp.results)):
                with col:
                    st.markdown(f"**{sc['name']}**")
                    ra = res.reid_after or {}
                    risk_badge(ra.get("reid_95", 0))
                    st.metric("Utility", f"{(res.utility_score or 0):.2%}")
                    st.metric("Method", res.method)
                    if res.success:
                        st.success("Target met")
                    else:
                        st.error("Target not met")

            # Drill-down tabs
            tab_names = [sc["name"] for sc in scenarios[:len(comp.results)]]
            tabs = st.tabs(tab_names)
            for i, (tab, (sc, res)) in enumerate(zip(tabs, zip(scenarios, comp.results))):
                with tab:
                    _display_result(
                        res, qis, sensitive,
                        log_entries=res.metadata.get("_log", []),
                        preprocess_meta=res.metadata.get("_preprocess_meta"),
                        key_suffix=f"_sc{i}",
                    )
                    # "Use This Scenario" button
                    if st.button(
                        f"Use {sc['name']} as Final Result",
                        key=f"_sc_use_{sc['name']}",
                    ):
                        st.session_state["protection_result"] = res
                        st.session_state["protection_log"] = res.metadata.get("_log", [])
                        st.session_state["steps_completed"].add("protect")
                        st.success(f"Promoted **{sc['name']}** to final result.")

st.divider()

# ── 2. Protection Mode ───────────────────────────────────────────────

_llm_backend = "Cerebras" if os.environ.get("CEREBRAS_API_KEY") else (
    "Gemini" if os.environ.get("GEMINI_API_KEY") else None
)
llm_available = _llm_backend is not None

st.subheader("Protection Mode")

mode_options = ["Auto (rules engine)", "Smart Combo (adaptive)"]
if llm_available:
    mode_options.append(f"AI ({_llm_backend})")
mode_options.append("Manual")

protect_mode = st.radio(
    "How should protection be configured?",
    mode_options,
    horizontal=True,
    help=(
        "**Auto:** rules engine selects the best method automatically. "
        "**Smart Combo:** adaptive retry — escalates preprocessing tiers until target is met. "
        "**AI:** LLM reviews and may override the rules engine. "
        "**Manual:** you choose the method and parameters."
    ),
)

# ── AI Method Review (when AI mode selected) ─────────────────────────

ai_config = st.session_state.get("_ai_config")

_ai_label = f"AI ({_llm_backend})" if _llm_backend else "AI"
if protect_mode == _ai_label:
    st.caption(
        "The AI will analyze your data profile and recommend the best "
        "protection method, potentially overriding the rules engine."
    )
    if st.button("Get AI Recommendation", type="secondary"):
        with st.spinner(f"Consulting AI ({_llm_backend})..."):
            from sdc_engine.sdc.metrics.reid import calculate_reid
            from sdc_engine.sdc.llm_method_config import (
                llm_select_method,
                cerebras_response_to_ai_config,
            )

            _pp_ai = st.session_state.get("preprocessed_data")
            use_data = _pp_ai if _pp_ai is not None else data_typed
            reid = calculate_reid(use_data, qis)

            risk_metrics = {
                "reid_95": reid.get("reid_95", 0),
                "reid_50": reid.get("reid_50", 0),
                "reid_99": reid.get("reid_99", 0),
            }

            llm_result = llm_select_method(
                data=use_data,
                quasi_identifiers=qis,
                sensitive_columns=sensitive,
                risk_metrics=risk_metrics,
                qi_treatment=st.session_state.get("_ai_qi_treatments", {}),
                protection_context=context,
                dataset_description=st.session_state.get("dataset_description", ""),
            )

            if llm_result:
                ai_config = cerebras_response_to_ai_config(llm_result)
                st.session_state["_ai_config"] = ai_config
            else:
                st.error("AI recommendation unavailable. Falling back to rules engine.")
                ai_config = None

        st.rerun()

    # Display stored AI recommendation
    if ai_config:
        st.success(f"**AI recommends: {ai_config.get('method', 'N/A')}**")
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.write(f"**Confidence:** {ai_config.get('method_confidence', 'N/A')}")
        with col_rec2:
            review = ai_config.get("rules_review", {})
            if review:
                if review.get("agrees"):
                    st.write(f"Agrees with rules engine ({review.get('rules_method', '')})")
                else:
                    st.write(f"Overrides rules ({review.get('rules_method', '')})")

        with st.expander("AI Reasoning", expanded=True):
            st.write(ai_config.get("method_reason", "No reasoning provided."))

            # Rules review detail
            if review and not review.get("agrees"):
                st.warning(f"Override reason: {review.get('override_reason', '')}")

            # Warnings
            for w in ai_config.get("warnings", []):
                st.warning(w)

            # Preprocessing suggestions from AI
            preproc = ai_config.get("preprocessing_overrides", {})
            if preproc:
                st.write("**AI preprocessing suggestions:**")
                for col, spec in preproc.items():
                    st.write(f"- {col}: {spec.get('action', '?')} — {spec.get('reasoning', '')}")

            # Expected outcome
            utility_g = ai_config.get("utility_guidance", {})
            if utility_g:
                est_util = utility_g.get("estimated_utility")
                est_risk = utility_g.get("estimated_reid_after")
                if est_util:
                    st.write(f"**Estimated utility:** {est_util}")
                if est_risk:
                    st.write(f"**Estimated risk after:** {est_risk}")

elif protect_mode == "Smart Combo (adaptive)":
    st.caption(
        "Adaptive retry: starts with light preprocessing, escalates through "
        "moderate → aggressive → very aggressive tiers until the risk target "
        "is met or utility drops below the floor."
    )

elif protect_mode == "Manual":
    st.caption("Choose the protection method and parameters manually.")

    # ── Get the rules engine recommendation as default ──────────────
    auto_rec = st.session_state.get("_auto_recommendation")
    if auto_rec is None:
        with st.spinner("Getting rules engine recommendation..."):
            from sdc_engine.sdc.metrics.reid import calculate_reid
            from sdc_engine.sdc.protection_engine import build_data_features
            from sdc_engine.sdc.selection.pipelines import select_method_suite
            from sdc_engine.sdc.config import get_context_targets

            _pp_manual = st.session_state.get("preprocessed_data")
            use_data = _pp_manual if _pp_manual is not None else data_typed
            reid = calculate_reid(use_data, qis)
            features = build_data_features(
                use_data, qis,
                reid=reid,
                risk_metric=risk_metric,
                sensitive_columns=sensitive or None,
                var_priority=st.session_state.get("_var_priority") or None,
            )
            targets = get_context_targets(context, risk_metric)
            access_tier = targets.get("access_tier", "SCIENTIFIC")

            suite = select_method_suite(
                features=features,
                access_tier=access_tier,
                verbose=False,
            )
            auto_rec = {
                "method": suite.get("primary", "kANON"),
                "params": suite.get("primary_params", {}),
                "rule": suite.get("rule_applied", ""),
                "confidence": suite.get("confidence", ""),
                "reason": suite.get("reason", ""),
                "fallbacks": [m for m, _ in suite.get("fallbacks", [])],
            }
            st.session_state["_auto_recommendation"] = auto_rec

    # Show what Auto would pick
    st.info(
        f"Rules engine recommends **{auto_rec['method']}** "
        f"(rule: {auto_rec['rule']}, confidence: {auto_rec['confidence']}). "
        f"Adjust below to override."
    )
    if auto_rec.get("reason"):
        st.caption(f"Reason: {auto_rec['reason']}")

    METHOD_OPTIONS = ["kANON", "LOCSUPR", "PRAM", "NOISE", "RANKSWAP", "RECSWAP"]
    default_idx = METHOD_OPTIONS.index(auto_rec["method"]) if auto_rec["method"] in METHOD_OPTIONS else 0
    manual_method = st.selectbox("Protection method", METHOD_OPTIONS, index=default_idx)
    st.session_state["_manual_method"] = manual_method

    # Pre-populate params from auto recommendation if method matches
    auto_params = auto_rec.get("params", {}) if manual_method == auto_rec["method"] else {}

    # Method-specific params
    if manual_method == "kANON":
        default_k = auto_params.get("k", 5)
        k_val = st.number_input("k value", min_value=2, max_value=50, value=int(default_k))
        st.session_state["_manual_params"] = {"k": k_val}
    elif manual_method == "LOCSUPR":
        default_t = auto_params.get("k", 3)
        threshold = st.number_input(
            "Suppression threshold", min_value=1, max_value=50, value=int(default_t)
        )
        st.session_state["_manual_params"] = {"k": threshold}
    elif manual_method == "PRAM":
        default_p = auto_params.get("p_change", 0.10)
        p_change = st.slider("p_change", 0.01, 0.50, float(default_p), 0.01)
        st.session_state["_manual_params"] = {"p_change": p_change}
    elif manual_method == "NOISE":
        default_n = auto_params.get("magnitude", 0.10)
        noise_level = st.slider("Noise level", 0.01, 0.50, float(default_n), 0.01)
        st.session_state["_manual_params"] = {"magnitude": noise_level}
    elif manual_method == "RANKSWAP":
        p_val = st.number_input("Rank distance (p)", min_value=1, max_value=100, value=10)
        r0_val = st.slider("Correlation preservation (R0)", 0.50, 1.00, 0.95, 0.01)
        st.session_state["_manual_params"] = {"p": p_val, "R0": r0_val}
    elif manual_method == "RECSWAP":
        swap_rate = st.slider("Swap rate", 0.01, 0.50, 0.05, 0.01)
        st.session_state["_manual_params"] = {"swap_rate": swap_rate}

    # Show fallbacks
    if auto_rec.get("fallbacks"):
        st.caption(f"Fallback order: {' → '.join(auto_rec['fallbacks'])}")

else:
    st.caption("The rules engine will automatically select the best protection method.")

st.divider()

# ── 3. Run Protection ────────────────────────────────────────────────

result = st.session_state.get("protection_result")

if result is None:
    btn_label = {
        "Auto (rules engine)": "Run Protection (Auto)",
        "Smart Combo (adaptive)": "Run Protection (Smart Combo)",
        _ai_label: "Run Protection (AI)",
        "Manual": "Run Protection (Manual)",
    }.get(protect_mode, "Run Protection")

    # Map radio label to mode string
    _mode_map = {
        "Auto (rules engine)": "auto",
        "Smart Combo (adaptive)": "combo",
        _ai_label: "ai",
        "Manual": "manual",
    }
    run_mode = _mode_map.get(protect_mode, "auto")

    if st.button(btn_label, type="primary", use_container_width=True):
        _run_protection(qis, sensitive, context, risk_metric, risk_target, mode=run_mode)
        st.rerun()
    else:
        st.info("Click the button above to start protection.")
        st.stop()

# ── 4. Display results ───────────────────────────────────────────────

if result is None:
    st.stop()

_display_result(result, qis, sensitive)

# ── Re-run option ─────────────────────────────────────────────────────
st.divider()
if st.button("Re-run Protection"):
    st.session_state["protection_result"] = None
    st.session_state["protection_log"] = []
    st.rerun()

if result.success:
    st.success("Go to **Download** to export the protected dataset.")
