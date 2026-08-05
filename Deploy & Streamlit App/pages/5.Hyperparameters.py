import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# -----------------------------
# TITLE
# -----------------------------
st.title(" Gradient Boosting Hyperparameter Tuning")

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
    preprocessor = joblib.load( "preprocessor.pkl" )
    return model, preprocessor


model, preprocessor = load_model_preprocessor()


# -----------------------------
# CHECK DATA
# -----------------------------
if "df" not in st.session_state:
    st.warning(" Please upload your dataset first to perform hyperparameter tuning.")
    st.stop()

df = st.session_state.df


# -----------------------------
# VALIDATE COLUMNS EXIST IN DATA
# -----------------------------
missing_cols = [col for col in MODEL_COLUMNS if col not in df.columns]

if missing_cols:
    st.error(f" Missing columns in dataset: {missing_cols}")
    st.stop()

if TARGET_COLUMN not in df.columns:
    st.error(f" Target column '{TARGET_COLUMN}' not found.")
    st.stop()


# -----------------------------
# TRANSFORM DATA
# -----------------------------
X = preprocessor.transform(df[MODEL_COLUMNS])
y = df[TARGET_COLUMN]


# -----------------------------
# MODEL + GRID SEARCH
# -----------------------------
gbr = GradientBoostingRegressor(random_state=42)

param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [100, 250, 500],
    'max_depth': [2, 4, 6]}

grid = GridSearchCV(
    gbr,
    param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1)

grid.fit(X, y)


# -----------------------------
# RESULTS
# -----------------------------
st.subheader(" Best Hyperparameters")
st.write(grid.best_params_)

st.subheader(" Best Cross-Validation R² Score")
st.metric("R² Score", f"{grid.best_score_:.4f}")


results = pd.DataFrame(grid.cv_results_)

with st.expander(" Grid Search Full Results"):
    st.dataframe(results, use_container_width=True)


# -----------------------------
# VISUALIZATION 1
# -----------------------------
with st.expander(" R² vs Learning Rate"):
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x='param_learning_rate',
        y='mean_test_score',
        hue='param_max_depth',
        data=results,
        marker='o',
        ax=ax1)
    ax1.set_title('R² vs Learning Rate (Gradient Boosting)')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('R² Score')
    ax1.grid(True)
    st.pyplot(fig1)


# -----------------------------
# VISUALIZATION 2
# -----------------------------
with st.expander(" R² vs Number of Estimators"):
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x='param_n_estimators',
        y='mean_test_score',
        hue='param_max_depth',
        data=results,
        marker='o',
        ax=ax2)
    ax2.set_title('R² vs Number of Estimators')
    ax2.set_xlabel('Number of Estimators')
    ax2.set_ylabel('R² Score')
    ax2.grid(True)
    st.pyplot(fig2)


# -----------------------------
# VISUALIZATION 3
# -----------------------------
with st.expander(" R² vs Max Depth"):
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x='param_max_depth',
        y='mean_test_score',
        hue='param_learning_rate',
        data=results,
        marker='o',
        ax=ax3)
    ax3.set_title('R² vs Max Depth (Gradient Boosting)')
    ax3.set_xlabel('Max Depth')
    ax3.set_ylabel('R² Score')
    ax3.grid(True)
    st.pyplot(fig3)


