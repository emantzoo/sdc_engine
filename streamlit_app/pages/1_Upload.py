"""
Page 1 — Upload Dataset
========================
Load a CSV or Excel file, preview the data, see basic stats.
"""
import pandas as pd
import streamlit as st

from state import reset_downstream

st.header("Upload Dataset")

file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Data is loaded as text to preserve original encoding. "
         "Numeric recovery happens automatically before protection.",
)

if file is not None:
    # Detect if this is a *new* file (different from what's already loaded)
    prev = st.session_state.get("filename")
    if prev != file.name:
        reset_downstream("upload")

    try:
        if file.name.endswith(".csv"):
            # Auto-detect delimiter (comma vs semicolon vs tab)
            sample = file.read(8192).decode("utf-8", errors="replace")
            file.seek(0)
            counts = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t")}
            sep = max(counts, key=counts.get)
            data = pd.read_csv(file, sep=sep, dtype=str, na_values=[], keep_default_na=False)
        else:
            data = pd.read_excel(file, dtype=str, na_values=[], keep_default_na=False)
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        st.stop()

    st.session_state["data"] = data
    st.session_state["filename"] = file.name
    st.session_state["steps_completed"].add("upload")

    # ── Stats ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", f"{len(data.columns):,}")
    c3.metric("Size", f"{data.memory_usage(deep=True).sum() / 1024:.0f} KB")

    # ── Preview ───────────────────────────────────────────────────
    st.subheader("Data Preview")
    st.dataframe(data.head(100), use_container_width=True, height=400)

    st.success("File loaded. Go to **Configure** to assign column roles.")
else:
    if st.session_state.get("data") is not None:
        data = st.session_state["data"]
        st.info(f"Using previously loaded file: **{st.session_state['filename']}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(data):,}")
        c2.metric("Columns", f"{len(data.columns):,}")
        c3.metric("Size", f"{data.memory_usage(deep=True).sum() / 1024:.0f} KB")
        st.dataframe(data.head(50), use_container_width=True, height=300)
