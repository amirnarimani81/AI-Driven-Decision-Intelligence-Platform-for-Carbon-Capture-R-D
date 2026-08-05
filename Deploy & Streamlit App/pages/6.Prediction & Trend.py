import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------
# TITLE
# -----------------------------
st.title(" Predict CO₂ Uptake & Trend Analysis")

# -----------------------------
# DEFINE COLUMNS (MUST BE FIRST)
# -----------------------------
MODEL_COLUMNS = [
    'surface area (m2/g)',
    'total pore volume(cm3/g)',
    'micropore volume (cm3/g)',
    'temp (°c)',
    'pressure (bar)']

TARGET_COLUMN = 'co2 uptake (mmol/g)'


# -----------------------------
# LOAD MODEL & PREPROCESSOR
# -----------------------------
@st.cache_resource

def load_model_preprocessor():
    model = joblib.load("uptake_model.pkl")
    preprocessor = joblib.load( "preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model_preprocessor()




# -----------------------------
# CHECK REQUIRED OBJECTS
# -----------------------------
if "df" not in st.session_state:
    st.warning(" Please upload dataset first.")
    st.stop()

df = st.session_state.df

if "MODEL_COLUMNS" not in globals() or "TARGET_COLUMN" not in globals():
    st.error(" MODEL_COLUMNS or TARGET_COLUMN not defined.")
    st.stop()

if "model" not in globals() or "preprocessor" not in globals():
    st.error(" Model or Preprocessor not loaded.")
    st.stop()


# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader(" Custom Input Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    surface = st.number_input("Surface Area (m²/g)", value=1200.0, step=10.0)

with col2:
    total_pore = st.number_input("Total Pore Volume (cm³/g)", value=0.5, step=0.01)

with col3:
    micropore = st.number_input("Micropore Volume (cm³/g)", value=0.2, step=0.01)


col4, col5 = st.columns(2)

with col4:
    temp = st.number_input("Temperature (°C)", value=25.0, min_value=0.0, max_value=100.0)

with col5:
    pressure = st.number_input("Pressure (bar)", value=1.0, min_value=0.1, max_value=50.0)


# -----------------------------
# PREDICTION
# -----------------------------
input_df = pd.DataFrame([[
    surface,
    total_pore,
    micropore,
    temp,
    pressure]], columns=MODEL_COLUMNS)

input_X = preprocessor.transform(input_df)
predicted_uptake = model.predict(input_X)[0]

st.success(f" Predicted CO₂ Uptake: **{predicted_uptake:.4f} mmol/g**")


# -----------------------------
# TREND ANALYSIS FUNCTION
# -----------------------------
st.subheader(" Trend Analysis")


def plot_trend(var_name, var_range):
    uptakes = []

    base_values = [surface, total_pore, micropore, temp, pressure]

    idx = MODEL_COLUMNS.index(var_name)

    for val in var_range:
        temp_values = base_values.copy()
        temp_values[idx] = val

        df_input = pd.DataFrame([temp_values], columns=MODEL_COLUMNS)
        pred = model.predict(preprocessor.transform(df_input))[0]
        uptakes.append(pred)

    fig = px.line(
        x=var_range,
        y=uptakes,
        labels={'x': var_name, 'y': 'CO₂ Uptake (mmol/g)'},
        title=f" Uptake vs {var_name}")

    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# TREND PLOTS
# -----------------------------
plot_trend('surface area (m2/g)', np.linspace(500, 2000, 50))
plot_trend('total pore volume(cm3/g)', np.linspace(0.1, 1.0, 50))
plot_trend('micropore volume (cm3/g)', np.linspace(0.05, 1.0, 50))
plot_trend('temp (°c)', np.linspace(0, 100, 50))
plot_trend('pressure (bar)', np.linspace(1, 50, 50))


# -----------------------------
# MODEL PERFORMANCE (DATASET)
# -----------------------------
if "df" in st.session_state:

    st.subheader(" Model Performance on Dataset")

    X_all = preprocessor.transform(df[MODEL_COLUMNS])
    y_all = df[TARGET_COLUMN]
    y_pred = model.predict(X_all)

    r2 = r2_score(y_all, y_pred)
    rmse = np.sqrt(mean_squared_error(y_all, y_pred))

    col1, col2 = st.columns(2)

    with col1:
        st.metric("R² Score", f"{r2:.4f}")

    with col2:
        st.metric("RMSE", f"{rmse:.4f} mmol/g")