"""
Page 1 — Upload Dataset
========================
Load a CSV or Excel file, preview the data, see basic stats.
Built-in sdcMicro benchmark datasets are also available.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from state import reset_downstream

_REPO_ROOT = Path(__file__).resolve().parents[2]

BUILTIN_DATASETS = {
    "-- Select a built-in dataset --": None,
    "testdata (4,580 rows) — Household income & expenditures": "tests/empirical/data/testdata.csv",
    "CASCrefmicrodata (1,080 rows) — Census microdata": "tests/empirical/data/CASCrefmicrodata.csv",
    "free1 (4,000 rows) — mu-Argus demo data": "tests/empirical/data/free1.csv",
    "francdat (8 rows) — Tiny CASC demo": "tests/parity/snapshots/francdat.csv",
    "adult (30,162 rows) — UCI Adult dataset": "tests/empirical/data/adult.csv",
}

st.header("Upload Dataset")

# ── Built-in datasets ──────────────────────────────────────────
choice = st.selectbox("Load a built-in dataset:", list(BUILTIN_DATASETS.keys()))

data = None

if BUILTIN_DATASETS.get(choice) is not None:
    csv_path = _REPO_ROOT / BUILTIN_DATASETS[choice]
    fname = csv_path.name

    prev = st.session_state.get("filename")
    if prev != fname:
        reset_downstream("upload")

    data = pd.read_csv(csv_path, dtype=str, na_values=[], keep_default_na=False)
    st.session_state["data"] = data
    st.session_state["filename"] = fname
    st.session_state["steps_completed"].add("upload")

# ── File uploader (custom data) ───────────────────────────────
st.markdown("**Or** upload your own file:")
file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Data is loaded as text to preserve original encoding. "
         "Numeric recovery happens automatically before protection.",
)

if file is not None:
    prev = st.session_state.get("filename")
    if prev != file.name:
        reset_downstream("upload")

    try:
        if file.name.endswith(".csv"):
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

# ── Show loaded data (from either source) ─────────────────────
if data is not None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", f"{len(data.columns):,}")
    c3.metric("Size", f"{data.memory_usage(deep=True).sum() / 1024:.0f} KB")

    st.subheader("Data Preview")
    st.dataframe(data.head(100), use_container_width=True, height=400)

    st.success("File loaded. Go to **Configure** to assign column roles.")
elif st.session_state.get("data") is not None:
    data = st.session_state["data"]
    st.info(f"Using previously loaded file: **{st.session_state['filename']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", f"{len(data.columns):,}")
    c3.metric("Size", f"{data.memory_usage(deep=True).sum() / 1024:.0f} KB")
    st.dataframe(data.head(50), use_container_width=True, height=300)
