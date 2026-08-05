# =========================
# IMPORTS
# =========================
import os
import json
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

from dotenv import load_dotenv
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# ENV
# =========================
load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE")


# =========================
# CONFIG
# =========================
DB_PATH = "preprocessor_output.db"
TABLE = "process_data"


MODEL_COLUMNS = [
    "surface area (m2/g)",
    "total pore volume(cm3/g)",
    "micropore volume (cm3/g)",
    "temp (°c)",
    "pressure (bar)"]

TARGET = "co2 uptake (mmol/g)"


# =========================
# DATA LOADER
# =========================
def load_data():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f"SELECT * FROM {TABLE}", conn)
    return df.dropna()

df = load_data()


# =========================
# ML MODEL (XGBOOST)
# =========================
def train_model(df):
    X = df[MODEL_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42)

    model.fit(X_train, y_train)

    # evaluation
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R2 Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")

    return model, (r2, rmse, mae)

model, metrics = train_model(df)

# =========================
# SAFE PREDICTION
# =========================
def predict(model, x):
    x_df = pd.DataFrame([x], columns=MODEL_COLUMNS)
    return float(model.predict(x_df)[0])


# =========================
# EDA ENGINE
# =========================
def run_eda(df):
    return {
        "mean": float(df[TARGET].mean()),
        "std": float(df[TARGET].std()),
        "corr_temp": float(df["temp (°c)"].corr(df[TARGET])),
        "corr_pressure": float(df["pressure (bar)"].corr(df[TARGET])) }


# =========================
# UNCERTAINTY ENGINE
# =========================
def uncertainty(model, x, n=30):
    preds = []

    for _ in range(n):
        noise = np.random.normal(0, 0.01, len(x))
        x_noisy = np.array(x) + noise
        preds.append(predict(model, x_noisy))

    return {
        "mean": float(np.mean(preds)),
        "std": float(np.std(preds)),
        "risk": float(np.std(preds) / (np.mean(preds) + 1e-8)) }


# =========================
# WHAT-IF ENGINE
# =========================
def what_if(model, x):
    s, pt, pm, t, p = x

    scenarios = {
        "temp +10%": (s, pt, pm, t*1.1, p),
        "pressure +20%": (s, pt, pm, t, p*1.2),
        "surface +15%": (s*1.15, pt, pm, t, p)}

    return [
        {"scenario": k, "prediction": predict(model, v)}
        for k, v in scenarios.items()]


# =========================
# BAYESIAN OPTIMIZATION
# =========================
def optimize(model):

    def objective(temp, pressure):
        x = (500, 0.5, 0.2, temp, pressure)
        return predict(model, x)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"temp": (300, 450), "pressure": (2, 8)},
        random_state=42,
        verbose=0)

    optimizer.maximize(init_points=5, n_iter=15)

    return optimizer.max


# =========================
# MULTI LLM ENGINE
# =========================

def call_llm(provider, prompt):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json"}

    try:

        # ================= GPT (OpenRouter)
        if provider == "gpt":

            r = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers=headers,
                json={"model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a senior CO2 capture engineer."}, {"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.2})

            data = r.json()

            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            else:
                return f"GPT Error: {data}"


        # ================= DEEPSEEK (OpenRouter)
        elif provider == "deepseek":

            r = requests.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers=headers,
                json={
                    "model": "deepseek/deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a CO2 capture process engineer."},
                        {"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.2} )

            data = r.json()

            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            else:
                return f"DeepSeek Error: {data}"


        # ================= OLLAMA (LOCAL)
        elif provider == "ollama":

            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2:latest",
                    "messages": [
                        {"role": "system", "content": "You are a senior CO2 capture engineer."},
                        {"role": "user", "content": prompt}], "options": {"temperature": 0.2, "num_predict": 300}})

            data = r.json()

            if "message" in data:
                return data["message"]["content"]
            else:
                return f"Ollama Error: {data}"


    except Exception as e:
        return f"LLM Failure: {str(e)}"
    

# =========================
# PROMPT ENGINE
# =========================
def build_prompt(data):
    return  f"""
You are a senior CO2 capture process engineer working in industrial scale-up.

You are analyzing an AI-driven Sequestration and process optimization system.

Your task is to produce a **decision-focused industrial report**.

==================================================
SYSTEM DATA
==================================================
{json.dumps(data, indent=2)}

==================================================
OUTPUT STRUCTURE (STRICT)
==================================================

--------------------------------------------------
1. EXECUTIVE SUMMARY
--------------------------------------------------
- 4–6 lines
- Include:
  • system behavior (temp & pressure sensitivity from EDA)
  • ML prediction reliability
  • optimization outcome
  • uncertainty level
  • scale-up readiness

--------------------------------------------------
2. PROCESS PERFORMANCE (EDA + ML + EVALUATION)
--------------------------------------------------
Include:

EDA:
- Average CO2 uptake
- Standard deviation (process stability)
- Correlation:
  • temperature vs uptake
  • pressure vs uptake

ML:
- Predicted CO2 uptake

MODEL EVALUATION:
- R² (model fit quality)
- RMSE (real prediction error)
- MAE (robustness)

Interpret:
→ Is the model reliable for industrial decision-making?

End with:
→ One-line industrial feasibility statement

--------------------------------------------------
3. SENSITIVITY ANALYSIS (WHAT-IF)
--------------------------------------------------
Scenario | Impact

- Temperature +10%
- Pressure +20%
- Surface area +15%

Then:
- 2–3 bullets explaining physical behavior

--------------------------------------------------
4. RISK ANALYSIS
--------------------------------------------------
Include:
- Prediction uncertainty (std)
- Risk index (std/mean)
- Model error (RMSE)

Classify:
→ LOW / MEDIUM / HIGH

Interpret:
- Is model trustworthy?
- Is experimental validation required?

--------------------------------------------------
5. OPTIMIZATION RESULT
--------------------------------------------------
Provide:
- Optimal temperature
- Optimal pressure
- Maximum CO2 uptake

End:
→ "Recommended industrial operating window"

--------------------------------------------------
6. ENGINEERING INTERPRETATION
--------------------------------------------------
Explain:

- sequestration mechanism and materials
- pore structure effect
- temperature effect (thermodynamics vs kinetics)
- pressure effect
- performance vs stability trade-off

End:
→ "Process design guideline"

--------------------------------------------------
7. FINAL DECISION
--------------------------------------------------


Justify with:
- model reliability (R², RMSE)
- uncertainty level
- optimization quality
- scale-up risk
- materials optimization


"""

# =========================
# PDF GENERATOR (ROBUST)
# =========================
def generate_pdf(text, filename="report.pdf"):

    if isinstance(text, dict):
        text = json.dumps(text, indent=2)

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(line, styles["BodyText"])
        for line in text.split("\n")]

    doc.build(content)
    return filename


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
st.title("AI Analysis (Multi-LLM)")


# =========================
# INPUTS
# =========================
st.header("Process Inputs")

surface = st.number_input("Surface area")
pore_total = st.number_input("Total pore volume")
pore_micro = st.number_input("Micropore volume")
temp = st.number_input("Temperature")
pressure = st.number_input("Pressure")


x = (surface, pore_total, pore_micro, temp, pressure)


# =========================
# EDA VISUALIZATION
# =========================
st.header("EDA Analysis")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.scatter(df["temp (°c)"], df["pressure (bar)"])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Pressure")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.scatter(df["temp (°c)"], df[TARGET], color="orange")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("CO2 Uptake")
    st.pyplot(fig)


# =========================
# RUN SYSTEM
# =========================
if st.button("Run AI System"):

    eda = run_eda(df)
    pred = predict(model, x)
    opt = optimize(model)
    whatif = what_if(model, x)
    unc = uncertainty(model, x)

    payload = {
        "EDA": eda,
        "Prediction": pred,
        "Optimization": opt,
        "WhatIf": whatif,
        "Uncertainty": unc,
        "ModelMetrics": {
            "R2": float(metrics[0]),
            "RMSE": float(metrics[1]),
            "MAE": float(metrics[2])}}


    st.subheader("Structured Output")
    st.json(payload)

    provider = st.selectbox("LLM", ["gpt", "deepseek", "ollama"])

    prompt = build_prompt(payload)

    result = call_llm(provider, prompt)

    st.subheader("AI Insight")
    st.markdown(result)

    pdf = generate_pdf(result)

    with open(pdf, "rb") as f:
        st.download_button(
            "Download Report",
            f,file_name="CO2_report.pdf")



































