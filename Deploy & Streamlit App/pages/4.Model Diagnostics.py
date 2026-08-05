import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from scipy.optimize import minimize


# ----------------------------
# Load Model & Preprocessor
# -----------------------------
@st.cache_resource

def load_model_preprocessor():
    model = joblib.load("uptake_model.pkl")
    preprocessor = joblib.load( "preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model_preprocessor()
# -----------------------------
# Standard Columns
# -----------------------------
MODEL_COLUMNS = [
    'surface area (m2/g)',
    'total pore volume(cm3/g)',
    'micropore volume (cm3/g)',
    'temp (°c)',
    'pressure (bar)']

TARGET_COLUMN = 'co2 uptake (mmol/g)'



st.title(" Model Diagnostics")


# -----------------------------
# CHECK DATA
# -----------------------------
if "df" not in st.session_state:
    st.warning(" Please load dataset first.")
    st.stop()

df = st.session_state.df


# -----------------------------
# PREPROCESS & PREDICT
# -----------------------------
X = df[MODEL_COLUMNS]
y = df[TARGET_COLUMN]

X_proc = preprocessor.transform(X)
y_pred = model.predict(X_proc)


# -----------------------------
# METRICS
# -----------------------------
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["R²", "RMSE", "MSE", "MAE"],
    "Value": [r2, rmse, mse, mae]})

st.subheader(" Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("R²", f"{r2:.4f}")
col2.metric("RMSE", f"{rmse:.4f}")
col3.metric("MSE", f"{mse:.4f}")
col4.metric("MAE", f"{mae:.4f}")


with st.expander(" Show Detailed Metrics Table"):
    st.dataframe(metrics_df, use_container_width=True)


# -----------------------------
# RESIDUAL PLOT
# -----------------------------
with st.expander(" Residuals Plot"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_pred, y - y_pred, alpha=0.6)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Predicted")
    st.pyplot(fig)


# -----------------------------
# ACTUAL VS PREDICTED (MATPLOTLIB)
# -----------------------------
with st.expander(" Predicted vs Actual (Matplotlib)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    x_ax = range(len(y))

    ax.scatter(x_ax, y, s=5, color="blue", label="Actual")
    ax.plot(x_ax, y_pred, lw=0.8, color="red", label="Predicted")

    ax.legend()
    ax.set_xlabel("Sample")
    ax.set_ylabel("CO₂ Uptake (mmol/g)")
    ax.set_title("Gradient Boosting: Actual vs Predicted")

    st.pyplot(fig)


# -----------------------------
# PLOTLY INTERACTIVE
# -----------------------------
with st.expander(" Predicted vs Actual (Interactive Plotly)"):
    residuals = y - y_pred

    fig_pred = px.scatter(
        x=range(len(y)),
        y=y_pred,
        labels={'x': 'Sample Index', 'y': 'Predicted CO₂ Uptake (mmol/g)'},
        title="Predicted vs Actual with Residuals",
        opacity=0.7,
        hover_data={
            "Actual CO₂ Uptake": y,
            "Predicted CO₂ Uptake": y_pred,
            "Residual": residuals })

    # Perfect fit line
    fig_pred.add_trace(
        px.line(x=range(len(y)), y=y).data[0])

    fig_pred.data[1].name = "Perfect Fit"
    fig_pred.data[1].line.color = "red"
    fig_pred.data[1].line.dash = "dash"
    fig_pred.data[1].line.width = 2

    fig_pred.update_layout(
        width=700,
        height=500,
        margin=dict(l=40, r=40, t=60, b=40))

    st.plotly_chart(fig_pred, use_container_width=True)


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader(" Feature Importance (Permutation Importance)")

perm_res = permutation_importance(model, X_proc, y)

importance_df = pd.DataFrame({
    "Feature": MODEL_COLUMNS,
    "Importance": perm_res.importances_mean}).sort_values(by="Importance", ascending=False)


with st.expander(" Feature Importance Table"):
    st.dataframe(importance_df, use_container_width=True)

fig_imp = px.bar(
    importance_df,
    x='Feature',
    y='Importance',
    title="Permutation Feature Importance")

st.plotly_chart(fig_imp, use_container_width=True)


