"""
Streamlit SDC App — Main Entry Point
=====================================
Statistical Disclosure Control for microdata.
Upload → Configure → Protect → Download.

Run:  streamlit run streamlit_app/app.py
"""
import os
import sys
from pathlib import Path

# Ensure the SDC package is importable
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import streamlit as st

st.set_page_config(
    page_title="SDC Protect",
    page_icon="\U0001f512",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise session state ──────────────────────────────────────────
from state import init_state  # noqa: E402

init_state()

# ── Sidebar settings ─────────────────────────────────────────────────
with st.sidebar:
    # Cerebras API key (required for AI-assisted classification/method selection)
    api_key = st.text_input(
        "Cerebras API Key",
        value=st.session_state.get("cerebras_api_key", os.environ.get("CEREBRAS_API_KEY", "")),
        type="password",
        help="Enter your Cerebras API key to enable AI-assisted classification and method selection.",
    )
    if api_key:
        st.session_state["cerebras_api_key"] = api_key
        os.environ["CEREBRAS_API_KEY"] = api_key

    # sdcMicro (R) toggle
    try:
        from sdc_engine.sdc.LOCSUPR import _check_r_available
        r_available = _check_r_available()
    except Exception:
        r_available = False

    r_label = "Use sdcMicro (R)  —  " + (":green[available]" if r_available else ":red[not found]")
    st.toggle(
        r_label,
        value=st.session_state.get("use_sdcmicro_r", True),
        key="use_sdcmicro_r",
        disabled=not r_available,
        help=(
            "When enabled, LOCSUPR and NOISE use R/sdcMicro for higher-quality results "
            "(fewer suppressions, less distortion). Falls back to Python if R fails."
        ),
    )

# ── Navigation ────────────────────────────────────────────────────────
upload_page = st.Page("pages/1_Upload.py", title="Upload", icon="\U0001f4c2")
configure_page = st.Page("pages/2_Configure.py", title="Configure", icon="\u2699\ufe0f")
protect_page = st.Page("pages/3_Protect.py", title="Protect", icon="\U0001f6e1\ufe0f")
download_page = st.Page("pages/4_Download.py", title="Download", icon="\u2b07\ufe0f")

nav = st.navigation([upload_page, configure_page, protect_page, download_page])
nav.run()
