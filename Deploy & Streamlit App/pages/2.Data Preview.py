import sqlite3
import pandas as pd
import streamlit as st





st.title(" Data Preview")

# ---------------------------
# CONFIG
# ---------------------------
DB_PATH = "preprocessor_output.db"
TABLE_NAME = "process_data"

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    try:
        
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
        return df
    except Exception:
        return None

df = load_data()

# ---------------------------
# DISPLAY
# ---------------------------
if df is not None:

    # Save to session if needed
    st.session_state["df"] = df

    # --- About the Dataset ---
    st.subheader(" About the Dataset")
    st.write("""
This dataset contains information on porous materials and their CO₂ uptake performance.
It includes **527+ entries** with **54 features**, covering material properties, experimental conditions, 
and chemical structure descriptors. The primary target variable is **CO₂ uptake (mmol/g)**. 
Each material is identified by its name (e.g., HKUST-1, MOF-5, ZIF-8).
    """)

    st.markdown("** Key Features:**")
    st.markdown("""
- **Material Properties:** Surface Area (m²/g), Total Pore Volume (cm³/g), Specific Pore Volume (cm³/g)
- **Experimental Conditions:** Pressure (bar), Temperature (°C)
- **CO₂ Uptake:** CO₂ uptake (mmol/g) per gram of material
- **Chemical Structure Descriptors:**
  - Bond types: C–C, C=C, C≡C, C–O, C=O, C–N, C≡N, etc.
  - Ring structures: (ring) C–C (ring), (ring) N–S (ring)
  - Metal nodes: Al, Cd, Co, Cu, Mg, Mn, Ni, Zn, Zr (binary indicators)
    """)

    st.divider()

    # --- Raw Data --
    st.subheader(" Raw Data")
    st.dataframe(df.head(), use_container_width=True)

    st.divider()

    # --- Data Description ---
    st.subheader(" Data Description")
    st.write(df.describe())

else:
    st.error(" Could not load data. Check DB path and table name.")
    st.warning(" Make sure your SQLite database exists and contains the correct table.")