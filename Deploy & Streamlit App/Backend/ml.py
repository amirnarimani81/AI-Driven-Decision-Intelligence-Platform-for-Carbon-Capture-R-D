import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance


# =============================
# LOAD MODEL + PREPROCESSOR
# =============================
def load_model_preprocessor():
    model = joblib.load("uptake_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl" )
    return model, preprocessor


model, preprocessor = load_model_preprocessor()



# =============================
# FEATURES
# =============================
MODEL_COLUMNS = [
    'surface area (m2/g)',
    'total pore volume(cm3/g)',
    'micropore volume (cm3/g)',
    'temp (°c)',
    'pressure (bar)']

TARGET_COLUMN = "co2 uptake (mmol/g)"


def predict_uptake(df):

    X = df[MODEL_COLUMNS]
    y = df[TARGET_COLUMN]

    X_proc = preprocessor.transform(X)
    y_pred = model.predict(X_proc)

    # -------------------------
    # METRICS
    # -------------------------
    metrics = {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mse": float(mean_squared_error(y, y_pred)),
        "mae": float(mean_absolute_error(y, y_pred))}

    # -------------------------
    # RESIDUALS
    # -------------------------
    residuals = (y - y_pred).tolist()

    # -------------------------
    # FEATURE IMPORTANCE
    # -------------------------
    perm = permutation_importance(model, X_proc, y)

    importance_df = pd.DataFrame({
        "feature": MODEL_COLUMNS,
        "importance": perm.importances_mean}).sort_values("importance", ascending=False)

    # -------------------------
    # OUTPUT PLOTS (matplotlib)
    # -------------------------
    fig_residual, ax1 = plt.subplots()
    ax1.scatter(y_pred, y - y_pred)
    ax1.axhline(0, color="red")
    ax1.set_title("Residuals vs Predicted")

    fig_actual, ax2 = plt.subplots()
    ax2.plot(y.values, label="Actual")
    ax2.plot(y_pred, label="Predicted")
    ax2.legend()
    ax2.set_title("Actual vs Predicted")

    return {
        "metrics": metrics,
        "residuals": residuals,
        "feature_importance": importance_df,
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
        "figures": {
            "residual_plot": fig_residual,
            "prediction_plot": fig_actual}}