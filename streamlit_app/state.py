"""
Session state schema and helpers.
"""
import streamlit as st


def init_state():
    """Initialise all session-state keys with safe defaults."""
    defaults = {
        "data": None,                  # pd.DataFrame — raw uploaded data
        "data_typed": None,            # pd.DataFrame — after numeric recovery
        "filename": None,              # str
        "column_roles": {},            # {col: "QI"|"Sensitive"|"Unassigned"}
        "protection_context": "scientific_use",
        "risk_metric": "reid95",
        "risk_target": 0.10,
        "reid_preview": None,          # dict from calculate_reid()
        "protection_result": None,     # ProtectionResult
        "protection_log": [],          # list[str]
        "steps_completed": set(),      # {"upload", "configure", "protect"}
        "dataset_description": "",     # Optional description for LLM
        "preprocess_plan": None,       # dict from build_type_aware_preprocessing()
        "preprocessed_data": None,     # pd.DataFrame after preprocessing
        "preprocess_metadata": None,   # dict — per-column before/after stats
        "scenarios": [],               # list[dict] — scenario comparison specs
        "scenario_results": None,      # ComparisonResult or None
        "use_sdcmicro_r": True,        # bool — use R/sdcMicro backend when available
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def require_step(step: str, redirect_msg: str | None = None):
    """Show warning and stop if a prerequisite step hasn't been completed."""
    if step not in st.session_state.get("steps_completed", set()):
        msg = redirect_msg or f"Please complete the **{step.title()}** step first."
        st.warning(msg)
        st.stop()


def reset_downstream(from_step: str):
    """Clear state for steps that depend on an earlier step being re-done."""
    order = ["upload", "configure", "protect"]
    idx = order.index(from_step) if from_step in order else -1
    for step in order[idx + 1:]:
        st.session_state["steps_completed"].discard(step)
    # Clear results when reconfiguring
    if from_step in ("upload", "configure"):
        st.session_state["protection_result"] = None
        st.session_state["protection_log"] = []
        st.session_state["reid_preview"] = None
        st.session_state["preprocess_plan"] = None
        st.session_state["preprocessed_data"] = None
        st.session_state["preprocess_metadata"] = None
        # Clear cached recommendations
        st.session_state["scenarios"] = []
        st.session_state["scenario_results"] = None
        for k in ("_auto_recommendation", "_ai_config", "_plan_df", "_allowed_actions"):
            st.session_state.pop(k, None)
    if from_step == "upload":
        st.session_state["column_roles"] = {}
        st.session_state["data_typed"] = None
