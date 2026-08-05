import streamlit as st


st.title(" Experimental, Methodology & Data Preprocessing")

st.image("Picture2.png",
    caption="Experimental Test",
    use_container_width=True)

st.image("Picture.png",
    caption="Methodology & Data Preprocessing",
    use_container_width=True)

# ✅ FIXED: correct number of tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Test Preparation",
    "Methodology",
    "Data Preparation",
    "Missing Values",
    "Outliers & Correlation"])

# ---------------------------
# TAB 1: TEST PREPARATION
# ---------------------------
with tab1:
    st.subheader("Test Preparation")

    st.markdown("""
##  CO₂ Uptake Experiment – Preparation & Measurement

###  Sample Preparation
- The porous material is **vacuum-dried at 200 °C for 2 hours**  
- Removes moisture and impurities that may affect adsorption performance  

---

###  System Preparation
- The sample (1–2 g) is placed in a **fixed-bed reactor or autoclave**  
- The system is purged with **N₂ gas (100 mL/min)**  
- Heated to **400 °C for 2 hours** to remove pre-adsorbed gases and stabilize the material  

---

###  Adsorption Measurement
- The system is cooled to **25–30 °C (room temperature)**  
- A gas mixture of **0.5% CO₂ in N₂** is introduced at **100 mL/min**  
- CO₂ adsorption is monitored in real time using:
  - Thermogravimetric Analysis (TGA)  
  - Infrared CO₂ sensor (e.g., MH-Z19B)  

---

###  Data Analysis
- **CO₂ Uptake (mg/g):** Calculated from weight change (TGA) or breakthrough curves  
- **Breakthrough Analysis:** Determines adsorption capacity and kinetics using CO₂ concentration vs time  

---

###  Outcome
This method provides **accurate and reproducible CO₂ adsorption data**, enabling:
- Performance comparison between materials  
- Process optimization  
- Model development for AI-based prediction systems  
""")

# ---------------------------
# TAB 2: METHODOLOGY
# ---------------------------
with tab2:
    st.subheader("Methodology")
    st.write("""
- Experimental data for various carbon porous adsorbents were collected to develop ML models predicting CO₂ adsorption capacity.
- Key features: pore volume, mean pore diameter, BET surface area, adsorption temperature & pressure.
- Target variable: CO₂ uptake (mmol/g).
- Models assess the influence of each parameter and predict adsorption performance.
- Considerations: adsorption potential, selectivity, stability, cost-effectiveness, reusability, fast adsorption–desorption kinetics.
- 3D plots from best ML model used to analyze combined effects and compare with previous studies.
""")

# ---------------------------
# TAB 3: DATA PREPARATION
# ---------------------------
with tab3:
    st.subheader("Data Preparation")
    st.write("""
- **Data Collection:** 527 entries from 52 research articles.
- **Feature Groups:**
    - Morphological: BET surface area, mean pore diameter, pore volume
    - Operational: Adsorption temperature & pressure
- **Target:** CO₂ adsorption capacity under corresponding conditions.
- **Feature Scaling:** Standard Scaler applied for normalization; improves convergence for some ML models.
""")

# ---------------------------
# TAB 4: MISSING VALUES
# ---------------------------
with tab4:
    st.subheader("Handling Missing Values")
    st.write("""
- Total Pore Volume (TPV) and Micropore Volume (MV) had missing values.
- Newton Interpolation initially used; GBR-based imputation improved predictive performance.
""")

# ---------------------------
# TAB 5: OUTLIERS & CORRELATION
# ---------------------------
with tab5:
    st.subheader("Outlier Detection")
    st.write("""
- Z-score normalization applied.
- Euclidean distance from dataset center computed for each point.
- Points beyond 99th percentile flagged as outliers.
- 4 outliers removed → 454 data points used for ML.
""")

    st.subheader("Feature Correlation")
    st.write("""
- Pearson Correlation Coefficient (PCC) calculated between features and target.
- PCC ranges from -1 (perfect negative) to 1 (perfect positive).
- Shows strength and direction of linear relationships; useful for feature selection.
""")