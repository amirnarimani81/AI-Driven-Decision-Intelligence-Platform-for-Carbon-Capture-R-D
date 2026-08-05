import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import math

from bayes_opt import BayesianOptimization
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Simulation, Optimization & Scale-Up",
    layout="wide"
)

st.title("⚙️ Process Simulation, Optimization & Scale-Up")


# =========================
# PATHS
# =========================
DB_PATH = r"D:\Chemical Engineering\Chemical Engineering\Softeware\ML\AI-Driven CO₂ Sequestration Decision Intelligence System\1. AI-Driven CO₂ Sequestration Decision Intelligence System\preprocessor_output.db"
TABLE_NAME = "process_data"

MODEL_PATH = r"D:\Chemical Engineering\Chemical Engineering\Softeware\ML\AI-Driven CO₂ Sequestration Decision Intelligence System\2. ETL and ML Workflow for CO₂ Uptake Data Cleaning, Feature Scaling, and Predictive Modeling\uptake_model.pkl"
PREPROCESSOR_PATH = r"D:\Chemical Engineering\Chemical Engineering\Softeware\ML\AI-Driven CO₂ Sequestration Decision Intelligence System\2. ETL and ML Workflow for CO₂ Uptake Data Cleaning, Feature Scaling, and Predictive Modeling\preprocessor.pkl"

MODEL_COLUMNS = [
    'surface area (m2/g)',
    'total pore volume(cm3/g)',
    'Flow Rate (L/min)',
    'micropore volume (cm3/g)',
    'Mixing Time (min)',
    'temp (°c)',
    'pressure (bar)'
]

TARGET = "co2 uptake (mmol/g)"


# =========================
# LOAD FUNCTIONS
# =========================
@st.cache_data
def load_database():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


df = load_database()
model, preprocessor = load_model()


# =========================
# PREDICTION FUNCTIONS
# =========================
def predict_uptake(feature_dict):
    df_input = pd.DataFrame([feature_dict])
    df_input = df_input.reindex(columns=MODEL_COLUMNS, fill_value=0)
    X = preprocessor.transform(df_input)
    return float(model.predict(X)[0])


def what_if_analysis(feature_dict):
    results = []
    base_pred = predict_uptake(feature_dict)
    for param in ["temp (°c)", "pressure (bar)"]:
        for factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
            new = feature_dict.copy()
            new[param] *= factor
            new_pred = predict_uptake(new)
            results.append({
                "Parameter": param,
                "Change": f"{int((factor-1)*100)}%",
                "New Value": round(new[param], 2),
                "Uptake": round(new_pred, 3),
                "Delta": round(new_pred - base_pred, 3)
            })
    return pd.DataFrame(results)


def sensitivity_analysis(feature_dict):
    base = predict_uptake(feature_dict)
    sens = {}
    for key in MODEL_COLUMNS:
        perturbed = feature_dict.copy()
        perturbed[key] *= 1.01
        new_pred = predict_uptake(perturbed)
        sens[key] = (new_pred - base) / base 
    return sens


def predict_with_uncertainty(feature_dict, n_samples=30):
    preds = []
    keys = list(feature_dict.keys())
    base_values = np.array(list(feature_dict.values()))
    
    for _ in range(n_samples):
        noise = np.random.normal(0, 0.02, len(keys))
        noisy_values = base_values + noise
        noisy_dict = dict(zip(keys, noisy_values))
        preds.append(predict_uptake(noisy_dict))

    preds = np.array(preds)
    mean = np.mean(preds)
    std = np.std(preds)
    cv = (std / mean) * 100 
    ci_lower = np.percentile(preds, 5)
    ci_upper = np.percentile(preds, 95)
    confidence = max(0, 1 - (std / mean))

    return {
        "mean": float(mean),
        "std": float(std),
        "cv": float(cv),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence": float(confidence)
    }


def optimization(df):
    def objective(surface_area, pore_volume, micropore, temperature, pressure, flow, time):
        features = {
            "surface area (m2/g)": surface_area,
            "total pore volume(cm3/g)": pore_volume,
            "micropore volume (cm3/g)": micropore,
            "temp (°c)": temperature,
            "pressure (bar)": pressure,
            "Flow Rate (L/min)": flow,
            "Mixing Time (min)": time
        }
        return predict_uptake(features)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={
            "surface_area": (100, 2000),
            "pore_volume": (0.1, 1.5),
            "micropore": (0.05, 0.8),
            "temperature": (20, 500),
            "pressure": (1, 100),
            "flow": (1, 50),
            "time": (10, 120)
        },
        random_state=42,
        verbose=0
    )

    optimizer.maximize(init_points=10, n_iter=30)
    best = optimizer.max["params"]

    best_conditions = {
        "surface area (m2/g)": best["surface_area"],
        "total pore volume(cm3/g)": best["pore_volume"],
        "micropore volume (cm3/g)": best["micropore"],
        "temp (°c)": best["temperature"],
        "pressure (bar)": best["pressure"],
        "Flow Rate (L/min)": best["flow"],
        "Mixing Time (min)": best["time"]
    }

    return {
        "best_conditions": best_conditions,
        "best_uptake": optimizer.max["target"]
    }


# =========================
# SCALE-UP CLASS
# =========================
class IndustrialScaleUp:
    @staticmethod
    def scale_to_industrial(lab, target_tons_year=20):
        V_lab = math.pi * (lab["bed_diameter_cm"]/2)**2 * lab["bed_height_cm"] / 1000
        mass_kg = V_lab * lab.get("bulk_density_kg_L", 0.6)
        co2_cycle_kg = mass_kg * lab["uptake_mmol_g"] / 1000 * 0.044
        cycle_time = lab["breakthrough_time_min"] + 15
        cycles_per_year = (330 * 24 * 60) / cycle_time
        required_cycles = (target_tons_year * 1000) / co2_cycle_kg
        scale_factor = max(1, required_cycles / cycles_per_year)
        diameter_m = (lab["bed_diameter_cm"]/100) * math.sqrt(scale_factor)
        height_m = (lab["bed_height_cm"]/100) * (scale_factor ** 0.33)
        flow_m3h = (lab["flow_rate_Lmin"] * scale_factor) / 1000 * 60
        compressor_power = flow_m3h * lab["pressure_bar"] / 36

        return {
            "scale_factor": round(scale_factor, 1),
            "bed_diameter_m": round(diameter_m, 2),
            "bed_height_m": round(height_m, 2),
            "flow_m3h": round(flow_m3h, 1),
            "compressor_power_kW": round(compressor_power, 1)
        }

    @staticmethod
    def generate_scale_up_report(lab, industrial):
        return f"""
SCALE-UP REPORT
Lab: diam={lab['bed_diameter_cm']}cm, height={lab['bed_height_cm']}cm, flow={lab['flow_rate_Lmin']}L/min, BT={lab['breakthrough_time_min']}min
Industrial (target {lab.get('target_tons_year',20)} tons/year):
- Scale factor: {industrial['scale_factor']}
- Bed: {industrial['bed_diameter_m']}m x {industrial['bed_height_m']}m
- Flow: {industrial['flow_m3h']} m³/h
- Compressor power: {industrial['compressor_power_kW']} kW
"""


# =========================
# STREAMLIT UI - TABS
# =========================
tab1, tab2 = st.tabs(["⚙️ Process Simulation & Optimization", "🏭 Industrial Scale-Up"])


# =====================================================
# TAB 1: PROCESS SIMULATION & OPTIMIZATION
# =====================================================
with tab1:
    st.markdown("### Process Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        surf_sim = st.number_input("Surface Area (m2/g)", 800.0, key="sim_s")
        pore_sim = st.number_input("Pore Volume (cm3/g)", 0.6, key="sim_pv")
        micro_sim = st.number_input("Micropore (cm3/g)", 0.3, key="sim_m")
    
    with col2:
        temp_sim = st.number_input("temp (°c)", 25.0, key="sim_t")
        press_sim = st.number_input("pressure (bar)", 5.0, key="sim_pr")
        flow_sim = st.number_input("Flow Rate (L/min)", 10.0, key="sim_f")
    
    if st.button("Run Simulation"):
        feat = {
            "surface area (m2/g)": surf_sim,
            "total pore volume(cm3/g)": pore_sim,
            "micropore volume (cm3/g)": micro_sim,
            "Flow Rate (L/min)": flow_sim,
            "Mixing Time (min)": 45,
            "temp (°c)": temp_sim,
            "pressure (bar)": press_sim
        }
        
        base = predict_uptake(feat)
        whatif_df = what_if_analysis(feat)
        sens = sensitivity_analysis(feat)
        risk = predict_with_uncertainty(feat)
    
        st.subheader("Base Prediction")
        st.metric("Predicted Uptake", f"{base:.2f} mmol/g")
        
        st.subheader("What-If Analysis")
        st.dataframe(whatif_df, use_container_width=True)
        
        st.subheader("Sensitivity (1% change)")
        st.json(sens)
        
        st.subheader("Risk & Uncertainty")
        st.write(f"CV: {risk['cv']:.1f}% | 90% CI: [{risk['ci_lower']:.2f}, {risk['ci_upper']:.2f}]")
    
    st.markdown("---")
    st.subheader("Bayesian Optimization (based on database)")
    
    if st.button("Run Optimization"):
        if len(df) == 0:
            st.warning("No data.")
        else:
            opt_res = optimization(df)
            st.success(f"Optimal Temp: {opt_res['best_conditions']['temp (°c)']:.1f}°C, Pressure: {opt_res['best_conditions']['pressure (bar)']:.1f} bar → {opt_res['best_uptake']:.2f} mmol/g")


# =====================================================
# TAB 2: INDUSTRIAL SCALE-UP
# =====================================================
with tab2:
    st.markdown("### Convert lab-scale reactor data to industrial dimensions using engineering similarity laws.")
    
    with st.form("scaleup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            diam = st.number_input(
                "Lab Bed Diameter (cm)",
                min_value=0.5,
                value=5.0,
                help="Internal diameter of the lab-scale adsorption column"
            )
            height = st.number_input(
                "Lab Bed Height (cm)",
                min_value=1.0,
                value=15.0,
                help="Height of the packed bed"
            )
            flow = st.number_input(
                "Lab Flow Rate (L/min)",
                min_value=0.1,
                value=0.5,
                help="Gas flow rate through the column"
            )
            bt = st.number_input(
                "Breakthrough Time (min)",
                min_value=1.0,
                value=30.0,
                help="Time until CO₂ concentration reaches 5% of inlet at outlet"
            )
        
        with col2:
            uptake_lab = st.number_input(
                "Lab Uptake (mmol/g)",
                min_value=0.1,
                value=4.2,
                help="CO₂ adsorption capacity at breakthrough conditions"
            )
            press_lab = st.number_input(
                "Lab Pressure (bar)",
                min_value=1.0,
                value=5.0,
                help="Operating pressure"
            )
            bulk = st.number_input(
                "Bulk Density (kg/L)",
                min_value=0.1,
                value=0.6,
                help="Density of packed bed including void space"
            )
            target = st.number_input(
                "Target Capacity (tons/year)",
                min_value=1,
                value=1000,
                help="Annual CO₂ capture target in metric tons"
            )
        
        submitted = st.form_submit_button(
            " Calculate Scale-Up",
            use_container_width=True,
            type="primary"
        )
    
    if submitted:
        lab_data = {
            "bed_diameter_cm": diam,
            "bed_height_cm": height,
            "flow_rate_Lmin": flow,
            "breakthrough_time_min": bt,
            "uptake_mmol_g": uptake_lab,
            "pressure_bar": press_lab,
            "bulk_density_kg_L": bulk,
            "target_tons_year": target
        }
    
        industrial = IndustrialScaleUp.scale_to_industrial(lab_data, target)
        report = IndustrialScaleUp.generate_scale_up_report(lab_data, industrial)
    
        st.markdown("### Scale-Up Results")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.metric("Scale Factor", f"{industrial['scale_factor']:,}x")
            st.metric("Bed Diameter", f"{industrial['bed_diameter_m']} m")
        
        with col_m2:
            ld_ratio = industrial['bed_height_m'] / industrial['bed_diameter_m']
            st.metric("Bed Height", f"{industrial['bed_height_m']} m")
            st.metric("L/D Ratio", f"{ld_ratio:.2f}")
        
        with col_m3:
            st.metric("Flow Rate", f"{industrial['flow_m3h']} m³/h")
            st.metric("Compressor Power", f"{industrial['compressor_power_kW']} kW")
        
        with st.expander(" View Raw Scale-Up Report"):
            st.code(report, language='text')


