"""
Page 4 — Download Protected Data
==================================
Export the protected dataset as CSV and an HTML summary report.
"""
import datetime
import html as html_mod
import streamlit as st

from state import require_step
from components import risk_badge, metric_cards


# ══════════════════════════════════════════════════════════════════════
# HTML report helpers
# ══════════════════════════════════════════════════════════════════════

def _esc(text) -> str:
    """HTML-escape arbitrary text."""
    return html_mod.escape(str(text)) if text else ""


def _section_protection_log(state: dict) -> str:
    """Collapsible section: Protection engine log."""
    log = state.get("protection_log", [])
    if not log:
        return ""
    entries_html = "".join(
        f"<div style='font-family:monospace;font-size:13px;padding:2px 0;'>"
        f"{_esc(entry)}</div>"
        for entry in log[-50:]          # cap at last 50 entries
    )
    truncated = (
        f"<p style='color:#888;font-size:12px;'>"
        f"Showing last 50 of {len(log)} entries.</p>"
        if len(log) > 50 else ""
    )
    return f"""
<details>
<summary><strong>Protection Engine Log</strong> ({len(log)} entries)</summary>
{truncated}
<div style="max-height:400px;overflow-y:auto;border:1px solid #eee;padding:8px;margin-top:8px;">
{entries_html}
</div>
</details>
"""


def _section_ai_recommendation(state: dict) -> str:
    """Collapsible section: AI recommendation details (if AI mode was used)."""
    ai_config = state.get("_ai_config")
    if not ai_config:
        return ""

    method = _esc(ai_config.get("method", "N/A"))
    confidence = _esc(ai_config.get("method_confidence", "N/A"))
    reason = _esc(ai_config.get("method_reason", "No reasoning provided."))

    # Rules review
    review = ai_config.get("rules_review", {})
    review_html = ""
    if review:
        agrees = review.get("agrees", True)
        rules_method = _esc(review.get("rules_method", ""))
        if agrees:
            review_html = (
                f"<p><strong>Rules engine agreement:</strong> "
                f"Agrees with rules engine ({rules_method})</p>"
            )
        else:
            override_reason = _esc(review.get("override_reason", ""))
            review_html = (
                f"<p style='color:#e67e22;'><strong>Rules engine override:</strong> "
                f"AI overrides rules engine ({rules_method})"
                f"{' &mdash; ' + override_reason if override_reason else ''}</p>"
            )

    # Warnings
    warnings_html = ""
    warnings_list = ai_config.get("warnings", [])
    if warnings_list:
        items = "".join(f"<li>{_esc(w)}</li>" for w in warnings_list)
        warnings_html = f"<p><strong>Warnings:</strong></p><ul>{items}</ul>"

    # Preprocessing overrides
    preproc_html = ""
    preproc = ai_config.get("preprocessing_overrides", {})
    if preproc:
        rows = "".join(
            f"<tr><td>{_esc(col)}</td>"
            f"<td>{_esc(spec.get('action', ''))}</td>"
            f"<td>{_esc(spec.get('reasoning', ''))}</td></tr>"
            for col, spec in preproc.items()
        )
        preproc_html = (
            "<p><strong>AI Preprocessing Suggestions:</strong></p>"
            "<table><tr><th>Column</th><th>Action</th><th>Reasoning</th></tr>"
            f"{rows}</table>"
        )

    # Utility guidance
    util_html = ""
    utility_g = ai_config.get("utility_guidance", {})
    if utility_g:
        parts = []
        if utility_g.get("estimated_utility"):
            parts.append(
                f"<strong>Estimated utility:</strong> {_esc(utility_g['estimated_utility'])}"
            )
        if utility_g.get("estimated_reid_after"):
            parts.append(
                f"<strong>Estimated risk after:</strong> {_esc(utility_g['estimated_reid_after'])}"
            )
        if parts:
            util_html = "<p>" + "<br>".join(parts) + "</p>"

    return f"""
<details>
<summary><strong>AI Recommendation (Cerebras)</strong></summary>
<table>
<tr><th>Method</th><td>{method}</td></tr>
<tr><th>Confidence</th><td>{confidence}</td></tr>
<tr><th>Reasoning</th><td>{reason}</td></tr>
</table>
{review_html}
{warnings_html}
{preproc_html}
{util_html}
</details>
"""


def _section_preprocess_metadata(state: dict) -> str:
    """Collapsible section: Preprocessing metadata (columns, actions, cardinality)."""
    meta = state.get("preprocess_metadata")
    if not meta:
        return ""

    rows_html = ""
    for col, info in meta.items():
        if not isinstance(info, dict):
            continue
        action = _esc(info.get("action", ""))
        before = _esc(info.get("before_unique", ""))
        after = _esc(info.get("after_unique", ""))
        success = info.get("success", False)
        status_icon = "&#10003;" if success else "&#10007;"
        status_color = "#27ae60" if success else "#e74c3c"
        rows_html += (
            f"<tr>"
            f"<td>{_esc(col)}</td>"
            f"<td>{action}</td>"
            f"<td style='text-align:center;'>{before}</td>"
            f"<td style='text-align:center;'>{after}</td>"
            f"<td style='text-align:center;color:{status_color};'>{status_icon}</td>"
            f"</tr>"
        )

    if not rows_html:
        return ""

    return f"""
<details>
<summary><strong>Preprocessing Metadata</strong></summary>
<table>
<tr>
  <th>Column</th><th>Action</th>
  <th style="text-align:center;">Before (unique)</th>
  <th style="text-align:center;">After (unique)</th>
  <th style="text-align:center;">OK</th>
</tr>
{rows_html}
</table>
</details>
"""


def _section_variable_importance(state: dict) -> str:
    """Collapsible section: Variable importance / backward elimination."""
    var_priority = state.get("_var_priority")
    if not var_priority:
        return ""

    # Sort descending by contribution %
    sorted_vars = sorted(var_priority.items(), key=lambda x: x[1][1], reverse=True)

    rows_html = ""
    for col, (label, pct) in sorted_vars:
        # Color-code by importance level
        if pct >= 15:
            color = "#e74c3c"   # HIGH — red
        elif pct >= 8:
            color = "#e67e22"   # MED-HIGH — orange
        elif pct >= 3:
            color = "#f39c12"   # MODERATE — amber
        else:
            color = "#95a5a6"   # LOW — grey
        rows_html += (
            f"<tr>"
            f"<td>{_esc(col)}</td>"
            f"<td>{_esc(label)}</td>"
            f"<td style='text-align:right;'>"
            f"<span style='color:{color};font-weight:bold;'>{pct:.1f}%</span></td>"
            f"</tr>"
        )

    return f"""
<details>
<summary><strong>Variable Importance (Backward Elimination)</strong></summary>
<table>
<tr><th>Column</th><th>Importance</th><th style="text-align:right;">Contribution</th></tr>
{rows_html}
</table>
</details>
"""


# ══════════════════════════════════════════════════════════════════════
# HTML report builder (must be defined before main UI logic)
# ══════════════════════════════════════════════════════════════════════

def _section_privacy_metrics(result, qis, sensitive) -> str:
    """Privacy metrics before/after: k-anonymity, l-diversity, t-closeness."""
    try:
        from sdc_engine.sdc.metrics.risk import check_kanonymity, calculate_disclosure_risk
        from sdc_engine.sdc.post_protection_diagnostics import (
            check_l_diversity, check_t_closeness,
        )
    except ImportError:
        return ""

    if result.protected_data is None:
        return ""

    # We need the original data — use preprocessed if available, else raw
    import streamlit as st
    orig = st.session_state.get("preprocessed_data") or st.session_state.get("data_typed")
    if orig is None:
        return ""

    rows_html = ""
    try:
        _, gs_b, _ = check_kanonymity(orig, qis, k=5)
        _, gs_a, _ = check_kanonymity(result.protected_data, qis, k=5)
        sc = "_group_size_" if "_group_size_" in gs_b.columns else "count"
        k_b = int(gs_b[sc].min()) if len(gs_b) > 0 else 0
        k_a = int(gs_a[sc].min()) if len(gs_a) > 0 else 0
        dr_b = calculate_disclosure_risk(orig, qis, k=5)
        dr_a = calculate_disclosure_risk(result.protected_data, qis, k=5)
        rows_html += (
            f"<tr><td>k-anonymity (min group)</td>"
            f"<td>{k_b}</td><td>{k_a}</td></tr>"
            f"<tr><td>Uniqueness rate</td>"
            f"<td>{dr_b.get('uniqueness_rate',0):.1%}</td>"
            f"<td>{dr_a.get('uniqueness_rate',0):.1%}</td></tr>"
            f"<tr><td>Records at risk (k&lt;5)</td>"
            f"<td>{dr_b.get('records_at_risk',0):,}</td>"
            f"<td>{dr_a.get('records_at_risk',0):,}</td></tr>"
            f"<tr><td>Avg group size</td>"
            f"<td>{dr_b.get('avg_group_size',0):.1f}</td>"
            f"<td>{dr_a.get('avg_group_size',0):.1f}</td></tr>"
        )
    except Exception:
        pass

    if sensitive:
        try:
            ld_b = check_l_diversity(orig, qis, sensitive, l_target=2)
            ld_a = check_l_diversity(result.protected_data, qis, sensitive, l_target=2)
            tc_b = check_t_closeness(orig, qis, sensitive, t_target=0.30)
            tc_a = check_t_closeness(result.protected_data, qis, sensitive, t_target=0.30)
            rows_html += (
                f"<tr><td>l-diversity (min)</td>"
                f"<td>{ld_b.get('l_achieved', '?')}</td>"
                f"<td>{ld_a.get('l_achieved', '?')}</td></tr>"
                f"<tr><td>l-diversity violations</td>"
                f"<td>{ld_b.get('violations', 0)}</td>"
                f"<td>{ld_a.get('violations', 0)}</td></tr>"
                f"<tr><td>t-closeness (max dist)</td>"
                f"<td>{tc_b.get('t_achieved', 0):.3f}</td>"
                f"<td>{tc_a.get('t_achieved', 0):.3f}</td></tr>"
            )
        except Exception:
            pass

    if not rows_html:
        return ""

    return f"""
<h2>Privacy Metrics</h2>
<table>
<tr><th>Metric</th><th>Before</th><th>After</th></tr>
{rows_html}
</table>
"""


def _build_html_report(result, state: dict) -> str:
    """Build a standalone HTML summary report."""
    reid_before = result.reid_before or {}
    reid_after = result.reid_after or {}
    roles = state.get("column_roles", {})
    qis = [c for c, r in roles.items() if r == "QI"]
    sensitive = [c for c, r in roles.items() if r == "Sensitive"]
    context = state.get("protection_context", "")
    metric = state.get("risk_metric", "")
    target = state.get("risk_target", 0)
    filename = state.get("filename", "")

    per_var_rows = ""
    if result.per_variable_utility:
        for col, val in result.per_variable_utility.items():
            v = f"{val:.2%}" if isinstance(val, float) else str(val)
            per_var_rows += f"<tr><td>{col}</td><td>{v}</td></tr>"

    # Privacy metrics section
    privacy_section = _section_privacy_metrics(result, qis, sensitive)

    # Build the four new collapsible appendix sections
    appendix_sections = "".join([
        _section_preprocess_metadata(state),
        _section_variable_importance(state),
        _section_ai_recommendation(state),
        _section_protection_log(state),
    ])

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>SDC Protection Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; margin-top: 28px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; color: white; font-weight: bold; }}
  .success {{ background: #27ae60; }}
  .fail {{ background: #e74c3c; }}
  details {{ margin: 16px 0; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px 16px; }}
  details summary {{ cursor: pointer; font-size: 15px; padding: 4px 0; }}
  details[open] summary {{ margin-bottom: 8px; }}
</style>
</head><body>
<h1>SDC Protection Report</h1>
<p><strong>File:</strong> {filename}<br>
<strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}<br>
<strong>Context:</strong> {context.replace('_',' ').title()}<br>
<strong>Metric:</strong> {metric} | <strong>Target:</strong> {target:.2%}</p>

<h2>Result</h2>
<p>
  <span class="badge {'success' if result.success else 'fail'}">
    {'TARGET MET' if result.success else 'TARGET NOT MET'}
  </span>
  &mdash; Method: <strong>{result.method}</strong>
</p>

<table>
<tr><th>Metric</th><th>Before</th><th>After</th></tr>
<tr><td>ReID 95th</td><td>{reid_before.get('reid_95',0):.2%}</td><td>{reid_after.get('reid_95',0):.2%}</td></tr>
<tr><td>ReID 99th</td><td>{reid_before.get('reid_99',0):.2%}</td><td>{reid_after.get('reid_99',0):.2%}</td></tr>
<tr><td>High-risk rate</td><td>{reid_before.get('high_risk_rate',0):.2%}</td><td>{reid_after.get('high_risk_rate',0):.2%}</td></tr>
<tr><td>Utility</td><td colspan="2">{(result.utility_score or 0):.2%}</td></tr>
</table>

<h2>Configuration</h2>
<p><strong>Quasi-identifiers:</strong> {', '.join(qis)}<br>
<strong>Sensitive columns:</strong> {', '.join(sensitive) or '(none)'}</p>

{privacy_section}

{'<h2>Per-Variable Utility</h2><table><tr><th>Column</th><th>Utility</th></tr>' + per_var_rows + '</table>' if per_var_rows else ''}

{appendix_sections}

<hr><p style="color:#888;font-size:12px">Generated by SDC Protect (Streamlit)</p>
</body></html>"""


# ══════════════════════════════════════════════════════════════════════
# Main UI
# ══════════════════════════════════════════════════════════════════════

require_step("protect", "Please **run protection** first.")

result = st.session_state.get("protection_result")
if result is None or result.protected_data is None:
    st.warning("No protection result available. Go back to **Protect**.")
    st.stop()

st.header("Download Protected Data")

# ── Summary ───────────────────────────────────────────────────────────
reid_after = result.reid_after or {}
r95 = reid_after.get("reid_95", 0)
risk_badge(r95)

metric_cards({
    "Method": result.method,
    "Utility": result.utility_score or 0,
    "ReID 95th (after)": r95,
    "Rows": len(result.protected_data),
})

st.divider()

# ── Download CSV ──────────────────────────────────────────────────────
csv_buf = result.protected_data.to_csv(index=False).encode("utf-8")
filename = st.session_state.get("filename", "data.csv")
base = filename.rsplit(".", 1)[0]
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

st.download_button(
    label="Download Protected CSV",
    data=csv_buf,
    file_name=f"{base}_protected_{ts}.csv",
    mime="text/csv",
    type="primary",
    use_container_width=True,
)

# ── Download HTML report ──────────────────────────────────────────────
report_html = _build_html_report(result, st.session_state)

st.download_button(
    label="Download Summary Report (HTML)",
    data=report_html.encode("utf-8"),
    file_name=f"{base}_report_{ts}.html",
    mime="text/html",
    use_container_width=True,
)

st.divider()

# ── Start over ────────────────────────────────────────────────────────
if st.button("Start Over"):
    preserve = {"cerebras_api_key", "use_sdcmicro_r"}
    for key in list(st.session_state.keys()):
        if key not in preserve:
            del st.session_state[key]
    st.rerun()
