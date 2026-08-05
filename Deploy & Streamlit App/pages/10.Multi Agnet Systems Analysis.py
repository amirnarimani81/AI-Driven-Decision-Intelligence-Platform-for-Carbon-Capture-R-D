import streamlit as st
import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import random
import tempfile
import PyPDF2
from scipy.stats import pearsonr
from dotenv import load_dotenv
from openai import OpenAI
from bayes_opt import BayesianOptimization

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from longchain_embeddings import HuggingFaceEmbeddings
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# LOAD ENV
# =========================
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")

DEEPSEEK_MODEL = "deepseek/deepseek-chat"
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"))



# =========================
# DATABASE
# =========================
DB_PATH = "preprocessor_output.db"
TABLE_NAME = "process_data"

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def add_experiment(materials, method, surface, pore_volume, micropore,
                   mixing_time, flow_rate, pressure, uptake, efficiency):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (
            materials, method,
            "surface area (m2/g)", "total pore volume(cm3/g)", "micropore volume (cm3/g)",
            "Mixing Time (min)", "Flow Rate (L/min)", "Pressure (bar)",
            "co2 uptake (mmol/g)", "Efficiency (%)")
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (materials, method, surface, pore_volume, micropore,
          mixing_time, flow_rate, pressure, uptake, efficiency))
    conn.commit()
    conn.close()

# =========================
# BASIC STATISTICS
# =========================
def analyze_dataset(df):
    summary = df.describe().to_dict()
    corr = df.corr(numeric_only=True)

    insights = []

    # =========================
    # CORRELATION INSIGHTS
    # =========================
    if "temp (°c)" in df.columns and "co2 uptake (mmol/g)" in df.columns:
        r, _ = pearsonr(df["temp (°c)"], df["co2 uptake (mmol/g)"])
        insights.append(f"Temperature vs Uptake correlation: {round(r, 3)}")

    if "Pressure (kPa)" in df.columns and "co2 uptake (mmol/g)" in df.columns:
        r, _ = pearsonr(df["Pressure (bar)"], df["co2 uptake (mmol/g)"])
        insights.append(f"Pressure vs Uptake correlation: {round(r, 3)}")
                        
    # =========================
    # PEAK OPERATING REGION
    # =========================
    if "temp (°c)" in df.columns and "co2 uptake (mmol/g)" in df.columns:
        peak_temp_range = df.groupby(
            pd.cut(df["temp (°c)"], 10)
        )["co2 uptake (mmol/g)"].mean().idxmax()
    else:
        peak_temp_range = "Not available"

    # =========================
    # ANOMALY DETECTION (Z-SCORE)
    # =========================
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df) > 0:
        z = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        anomalies = int((z > 3).sum().sum())
    else:
        anomalies = 0

    # =========================
    # INDUSTRIAL KPIs
    # =========================
    result = {
        "summary": summary,
        "correlation_matrix": corr.to_dict(),
        "insights": insights,
        "peak_temperature_range": str(peak_temp_range),
        "anomalies": anomalies}

    # =========================
    # OPTIONAL DOMAIN-SPECIFIC KPIs
    # =========================
    if "co2 uptake (mmol/g)" in df.columns:
        result.update({
            "avg_uptake": float(df["co2 uptake (mmol/g)"].mean()),
            "max_uptake": float(df["co2 uptake (mmol/g)"].max())})

    return result

# ML MODEL CONFIGURATION
# =========================
MODEL_COLUMNS = [
    'surface area (m2/g)',
    'total pore volume(cm3/g)',"Flow Rate (L/min)",
    'micropore volume (cm3/g)',"Mixing Time (min)",
    'temp (°c)',
    'Pressure (bar)']

TARGET_COLUMN = 'co2 uptake (mmol/g)'


# Load Model & Preprocessor
# -----------------------------

def load_model_preprocessor():
    model = joblib.load("uptake_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor
model, preprocessor = load_model_preprocessor()

# PREDICTION ENGINE
# =========================
def predict_uptake(feature_dict):
    df = pd.DataFrame([feature_dict])
    df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)
    X = preprocessor.transform(df)
    return float(model.predict(X)[0])


# MODEL EVALUATION
# =========================
def evaluate_model(df):
    X = df[MODEL_COLUMNS]
    y = df[TARGET_COLUMN]

    X_proc = preprocessor.transform(X)
    y_pred = model.predict(X_proc)

    metrics = {
        "r2": float(r2_score(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred))}

    perm = permutation_importance(model, X_proc, y)
    importance = dict(zip(MODEL_COLUMNS, perm.importances_mean))

    return {
        "metrics": metrics,
        "feature_importance": importance}

# UNCERTAINTY-AWARE PREDICTION
# =========================
def predict_with_uncertainty(feature_dict, n_samples=20):

    preds = []

    base = np.array([feature_dict[c] for c in MODEL_COLUMNS], dtype=float)

    for _ in range(n_samples):
        noise = np.random.normal(0, 0.02, len(base))
        noisy = base + noise

        noisy_dict = dict(zip(MODEL_COLUMNS, noisy))
        preds.append(predict_uptake(noisy_dict))

    preds = np.array(preds)

    mean = preds.mean()
    std = preds.std()

    return {
        "mean": float(mean),
        "std": float(std),
        "risk": float(std / mean) if mean != 0 else 0,
        "confidence": float(1 - (std / mean)) if mean != 0 else 0}

# SENSITIVITY ANALYSIS
# =========================
def sensitivity_analysis(feature_dict):

    base = predict_uptake(feature_dict)
    sensitivities = {}

    for i, key in enumerate(MODEL_COLUMNS):
        perturbed = feature_dict.copy()
        perturbed[key] *= 1.01

        new_pred = predict_uptake(perturbed)

        sensitivities[key] = (new_pred - base) / base

    return sensitivities

# WHAT-IF ANALYSIS 
# =========================
def advanced_what_if(x):
    base_pred = predict_uptake(*x)
    
    scenarios = []
    
    # Temperature scenarios
    for temp_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        pred = predict_uptake(x[0], x[1], x[2], x[3] * temp_factor, x[4])
        scenarios.append({
            "parameter": "Temperature",
            "change": f"{int((temp_factor-1)*100)}%",
            "prediction": pred,
            "change_from_base": pred - base_pred })
    
    # Pressure scenarios
    for press_factor in [0.8, 0.9, 1.0, 1.1, 1.2]:
        pred = predict_uptake(x[0], x[1], x[2], x[3], x[4] * press_factor)
        scenarios.append({
            "parameter": "Pressure",
            "change": f"{int((press_factor-1)*100)}%",
            "prediction": pred,
            "change_from_base": pred - base_pred})
    
    return pd.DataFrame(scenarios)

# BUILD FEATURE VECTOR
# =========================
def build_feature_vector(opt):
    if opt and opt.get("best_conditions"):
        best = opt["best_conditions"]
        return (
            best.get("surface area (m2/g)", 500),
            best.get("total pore volume(cm3/g)", 0.5),
            best.get("micropore volume (cm3/g)", 0.2),
            best.get("Flow Rate (L/min)", 10),
            best.get("Mixing Time (min)", 45),
            best.get("temp (°c)", 300),
            best.get("Pressure (bar)", 1))
    else:
        return (500, 0.5, 0.2, 10, 45, 300, 1)


# Engineering - REACTOR AGENT
# =========================
def reactor_agent(feature_dict):
    temp = feature_dict.get("temp (°c)", 25)
    pressure_bar = feature_dict.get("Pressure (bar)", 5)
    efficiency = (0.6*temp + 0.8*pressure_bar) / 100
    return {"efficiency": efficiency, "constraints": {"safe_pressure": pressure_bar < 50, "safe_temp": temp < 200}}

# Engineering - MATERIAL AGENT
# =========================
def material_agent(feature_dict):
    surface = feature_dict.get("surface area (m2/g)", 500)
    pore = feature_dict.get("total pore volume(cm3/g)", 0.5)
    micro = feature_dict.get("micropore volume (cm3/g)", 0.2)
    score = 0.5*surface + 2*pore + 3*micro
    return {"material_score": score, "adsorption_potential": score / 100}


# Engineering - COST MODEL
# =========================
def cost_agent(x):
    feature_dict = x

    surface = feature_dict["surface area (m2/g)"]
    temp = feature_dict["temp (°c)"]
    pressure = feature_dict["Pressure (bar)"]

    return 0.002 * temp + 0.05 * pressure + 0.01 * (surface / 100)

# Engineering - BAYESIAN OPTIMIZATION
# =========================
BOUNDS = {
    "surface_area": (100, 2000),
    "pore_volume": (0.1, 1.5),
    "micropore": (0.05, 0.8),
    "temperature": (20, 500),
    "pressure": (1, 100)}

def optimization(df):
    # Use actual data bounds
    if len(df) > 0:
        bounds = {
            "surface_area": (float(df["surface area (m2/g)"].min()), float(df["surface area (m2/g)"].max())),
            "pore_volume": (float(df["total pore volume(cm3/g)"].min()), float(df["total pore volume(cm3/g)"].max())),
            "micropore": (float(df["micropore volume (cm3/g)"].min()), float(df["micropore volume (cm3/g)"].max())),
            "temperature": (float(df["temp (°c)"].min()), float(df["temp (°c)"].max())),
            "pressure": (float(df["Pressure (bar)"].min()), float(df["pressure (bar)"].max()))}
    else:
        bounds = BOUNDS
    
    def objective(surface_area, pore_volume, micropore, temperature, pressure):
        return predict_uptake(surface_area, pore_volume, micropore, temperature, pressure)

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=bounds,
        random_state=42,
        verbose=0)

    optimizer.maximize(
        init_points=10,
        n_iter=25)

    best_params = optimizer.max["params"]
    
    # Rename back to original column names
    best_conditions = {
        "surface area (m2/g)": best_params["surface_area"],
        "total pore volume(cm3/g)": best_params["pore_volume"],
        "micropore volume (cm3/g)": best_params["micropore"],
        "temp (°c)": best_params["temperature"],
        "Pressure (bar)": best_params["pressure"]}

    return {
        "best_conditions": best_conditions,
        "best_uptake": optimizer.max["target"]}


# RAG SYSTEM
# =========================

VECTOR_DB_PATH = "vector_db"
PAPERS_PATH = "papers"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vector_db():
    docs = []
    
    if os.path.exists(PAPERS_PATH):
        for file in os.listdir(PAPERS_PATH):
            if file.endswith(".pdf"):
                file_path = os.path.join(PAPERS_PATH, file)
                try:
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                docs.append(Document(page_content=text))
                except FileNotFoundError:
                    print(f"File not found (skipping): {file}")
                    continue  # skip this file and continue
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
        
    # fallback if no PDFs found
    if len(docs) == 0:
        docs.append(Document(page_content="CO2 capture using MOFs shows promise for industrial applications."))
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100)
    
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    db.save_local(VECTOR_DB_PATH)
    
    return db

# LOAD VECTOR DB
# =========================
def load_vector_db():
    if not os.path.exists(VECTOR_DB_PATH):
        return build_vector_db()
    
    try:
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True)
    except:
        return build_vector_db()

# INIT DB (agent-ready)
# =========================
DB = load_vector_db()

# RAG PIPELINE (CORE AGENT MEMORY)
# =========================
def rag_pipeline(query):
    try:
        docs = DB.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
    except:
        context = "No relevant documents found."
    
    prompt = f"""
You are a senior Chemical Engineering researcher.

Analyze CO2 capture scientific documents.

====================
CONTEXT
====================
{context}

QUESTION:
{query}

====================
TASK
====================

Provide:

INTRODUCTION:
METHODS:
RESULTS:
CONCLUSION:
LIMITATIONS:
ENGINEERING_INSIGHT:
RECOMMENDED_NEXT_EXPERIMENT:
"""

    res = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500)
    
    return res.choices[0].message.content



# LLM ANALYSIS
# =========================
def llm_analysis(df, pred, sens, bottleneck):
    prompt = f"""
You are a senior CO2 capture engineer.

Data Summary:
{df.describe().to_string() if len(df) > 0 else "No data available"}

Prediction: {pred}
Sensitivity: {sens}
Bottleneck: {bottleneck}

Explain:
- root cause
- physical reasoning
- risks
- insights
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}])

    return res.choices[0].message.content

# =========================
# DECISION ENGINE
# =========================
def decision_engine(query, eda, ml, opt, rag):
    prompt = f"""
Return ONLY valid JSON.

{{
 "insight": "...",
 "root_cause": "...",
 "risk": "LOW|MEDIUM|HIGH",
 "next_experiment": "..."}}

DATA:
EDA Summary: {eda.get('insights', [])}
ML Performance: R²={ml.get('metrics', {}).get('r2', 'N/A')}
Optimization Best: {opt.get('best_uptake', 'N/A')} mmol/g
RAG Insight: {rag[:200] if rag else 'N/A'}

Query: {query}
"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0)
    
    try:
        return json.loads(res.choices[0].message.content)
    except:
        return {
            "insight": "Analysis complete",
            "root_cause": "Check process conditions",
            "risk": "MEDIUM",
            "next_experiment": "Validate with experimental data"}

# DOE GENERATOR
# =========================
def next_experiment_suggestion(decision):
    prompt = f"""
Based on this decision:

{decision}

Design 3 next experiments (T, P, material)
Include:
- Temp
- Pressure
- Material type
- Expected outcome
"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500)

    return res.choices[0].message.content

# REPORT + PDF
# =========================
def generate_report(query, eda, ml, opt, rag, decision):

    # =========================
    # BUILD CLEAN DATA OBJECT (CRITICAL FIX)
    # =========================
    data = {
        "query": query,
        "eda": eda,
        "ml": ml,
        "optimization": opt,
        "rag": rag,
        "decision": decision}

 
    structured_report = f"""
==============================
CO2 CAPTURE INDUSTRIAL REPORT
==============================

EXECUTIVE DATA SNAPSHOT
- ML R²: {ml.get('metrics', {}).get('r2', 'N/A')}
- RMSE: {ml.get('metrics', {}).get('rmse', 'N/A')}
- Best Uptake: {opt.get('best_uptake', 'N/A')}
- Decision: {decision.get('insight', 'N/A')}

EDA INSIGHTS
- Anomalies: {eda.get('anomalies', 'N/A')}
- Key Trends: {eda.get('insights', [])}

OPTIMIZATION
- Best Conditions: {opt.get('best_conditions', {})}


==============================
"""
    # =========================
    # INDUSTRIAL PROMPT (FIXED)
    # =========================
    prompt = f"""
You are a senior Chemical Process Engineer in industrial CO₂ capture systems.

Your task is to convert structured system outputs into a **decision-grade industrial report**.

==================================================
SYSTEM DATA (JSON)
==================================================
{json.dumps(data, indent=2)}


INSTRUCTIONS
==================================================

Generate a structured **INDUSTRIAL REPORT** with the following sections:

--------------------------------------------------
1. EXECUTIVE SUMMARY
--------------------------------------------------
- Objective of analysis
- Key CO₂ uptake performance
- Model reliability (R², RMSE)
- Final performance rating (Excellent / Good / Moderate / Poor)
- One-line engineering verdict

--------------------------------------------------
2. AIM OF STUDY
--------------------------------------------------
- Define process goal (adsorption / optimization / scale-up)
- Industrial relevance (capture efficiency, cost reduction)

--------------------------------------------------
3. TEST CONDITIONS
--------------------------------------------------
Extract and present:
- Material properties (surface, pore, micropore)
- Operating conditions (temperature, pressure, flow rate if implied)

--------------------------------------------------
4. RESULTS & PERFORMANCE
--------------------------------------------------
- Predicted uptake vs expected
- Performance classification
- Efficiency, material effectiveness, cost indication

--------------------------------------------------
5. PROCESS INTERPRETATION
--------------------------------------------------
- Adsorption mechanism (physical reasoning)
- Temperature and pressure effects
- Material structure impact

--------------------------------------------------
6. OPTIMIZATION ANALYSIS
--------------------------------------------------
- Compare current vs optimal conditions
- Quantify improvement potential (% or mmol/g)

--------------------------------------------------
8. ECONOMIC EVALUATION
--------------------------------------------------
- Cost drivers (pressure, temperature, material)
- Efficiency vs cost trade-off

--------------------------------------------------
9. REACTOR & SCALE-UP ASSESSMENT
--------------------------------------------------
- Operational feasibility
- Safety (temperature / pressure)
- Industrial scalability

--------------------------------------------------
10. RISK & UNCERTAINTY
--------------------------------------------------
- Model uncertainty
- Data limitations
- Sensitivity risks

--------------------------------------------------
11. FINAL VERDICT
--------------------------------------------------
- Rating: ★ to ★★★★★
- Decision: GO / CONDITIONAL / NO-GO
- Justification (2–3 lines)

--------------------------------------------------
12. NEXT EXPERIMENTS (DOE)
--------------------------------------------------
Suggest 3 experiments:
- Temperature
- Pressure
- Material change
- Expected improvement

--------------------------------------------------
ENGINEERING RULES
--------------------------------------------------
- Be concise and technical
- No academic explanations
- Use numbers wherever possible
- Focus on decisions, not theory
- Avoid generic statements

END
"""
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a senior chemical engineer."},
            {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1500)

    return res.choices[0].message.content


# MULTI-AGENT ORCHESTRATOR
# =========================
def run_agents(query):
  
    df = load_data()
    eda = analyze_dataset(df)
    ml = evaluate_model(df)
    opt = optimization(df)
    rag = rag_pipeline(query)
    x = build_feature_vector(opt)
    prediction = predict_uptake(x)
    uncertainty = predict_with_uncertainty(x)
    reactor = reactor_agent(x)
    material = material_agent(x)
    cost = cost_agent(x)
    sens = sensitivity_analysis(x)
    bottleneck = detect_bottleneck(sens, uncertainty)
    whatif = advanced_what_if(x)
    decision = decision_engine(query, eda, ml, opt, rag)
    doe = next_experiment_suggestion(decision)
    report = generate_report(query, eda, ml, opt, rag, decision)
    pdf = export_pdf(report)
    powerbi_file = export_powerbi({
        "prediction": prediction,
        "uncertainty": uncertainty.get("mean", prediction),
        "decision": decision.get("insight", "N/A")})

    return {
        "eda": eda,
        "ml": ml,
        "prediction": prediction,
        "uncertainty": uncertainty,
        "optimization": opt,
        "rag": rag,
        "reactor": reactor,
        "material": material,
        "cost": cost,
        "sensitivity": sens,
        "bottleneck": bottleneck,
        "what_if": whatif,
        "decision": decision,
        "doe": doe,
        "report": report,
        "pdf": pdf,
        "powerbi": powerbi_file}


# SIMPLE CHATBOT USING run_agents (JUST Q&A)
# =========================

def get_chatbot_response(question):
    """Simple Q&A using the multi-agent system"""
    
    # Run all agents with the user's question
    res = run_agents(question)
    
    # Just return the decision insight as answer
    return res['decision'].get('insight', 'Analysis complete')


st.set_page_config(page_title="CO2 Capture AI Agent", layout="wide")

st.subheader("Multi-Agent Decision Intelligence Pipeline")

st.write("""
An integrated AI workflow that combines data analytics, machine learning,
Bayesian optimization, RAG-based scientific knowledge retrieval,
engineering-specific AI agents, uncertainty analysis, and LLM reasoning
to deliver automated process insights, optimization recommendations,
next-experiment planning, and industrial-grade decision reports.
""")


st.title("CO2 Capture Multi-Agent (Industrial System)")
st.markdown(f"*Connected to existing database | Table: `{TABLE_NAME}`*")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

mode = st.radio("Select Mode", [" AI Assistant (Add & Analyze)", " AI Chatbot (Q&A)", " Process Simulation"])

material_map = {"MOF-5": 1, "ZIF-8": 2, "Activated Carbon": 3, "Graphene": 4, "Custom": 5}
MATERIALS_LIST = ["MOF-5", "ZIF-8", "Activated Carbon", "Graphene", "MOF-508", "MIL-101", "UiO-66", "HKUST-1", "ZIF-67", "MCM-41", "SBA-15", "MOF-74", "COF-1", "PPN-6", "NU-1000"]
METHODS_LIST = ["Solvothermal", "Hydrothermal", "Chemical Activation", "CVD", "Electrochemical", "Template", "Solvent-assisted", "Microwave-assisted", "Sonochemical", "Mechanical"]

# CHAT ASSISTANT MODE
# =========================
if mode == " AI Assistant (Add & Analyze)":
    st.markdown("### Add New Experimental Result")
    col1, col2 = st.columns(2)
    with col1:
        material = st.selectbox("Material Type", MATERIALS_LIST)
        method = st.selectbox("Synthesis Method", METHODS_LIST)
        surface = st.number_input("Surface Area (m²/g)", value=800.0, step=50.0)
        pore_volume = st.number_input("Total Pore Volume (cm³/g)", value=0.6, step=0.05)
        micropore = st.number_input("Micropore Volume (cm³/g)", value=0.3, step=0.02)
    with col2:
        mixing_time = st.number_input("Mixing Time (min)", value=45.0, step=5.0)
        flow_rate = st.number_input("Flow Rate (L/min)", value=8.5, step=0.5)
        pressure = st.number_input("Pressure (bar)", value=5.0, step=0.5)
        uptake = st.number_input("CO₂ Uptake (mmol/g)", value=4.2, step=0.1)
        efficiency = st.number_input("Efficiency (%)", value=85.0, step=5.0)
    query = st.text_area("Research Question (optional)", height=100)
    temp = st.number_input("Temp (°C)", value=25.0, step=5.0)
    
    feature_dict = {"surface area (m2/g)": surface, "total pore volume(cm3/g)": pore_volume,
                    "micropore volume (cm3/g)": micropore, "Flow Rate (L/min)": flow_rate,
                    "Mixing Time (min)": mixing_time, "temp (°c)": temp, "Pressure (bar)": pressure}
    predicted = predict_uptake(feature_dict)
    st.info(f"Model Prediction: {predicted:.2f} mmol/g | Your Measurement: {uptake:.2f} mmol/g")
    
    if st.button(" RUN AGENT ANALYSIS", type="primary"):
        add_experiment(material, method, surface, pore_volume, micropore, mixing_time, flow_rate, pressure, uptake, efficiency)
        with st.spinner("Agent analyzing..."):
            res = run_agents(query if query else "Analyze this experiment")
        st.success("Analysis complete!")
        st.balloons()
        st.subheader("AGENT DECISION")
        st.info(res["decision"].get("insight", "Analysis complete"))
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CO₂ Uptake", f"{res['prediction']:.3f} mmol/g")
            st.metric("Confidence", f"{res['uncertainty']['confidence']:.1%}")
        with col2:
            st.metric("Reactor Efficiency", f"{res['reactor']['efficiency']:.1%}")
            st.metric("Estimated Cost", f"${res['cost']:.2f}/ton")
        if res["bottleneck"]:
            st.warning("Bottlenecks detected:")
            for issue in res["bottleneck"]:
                st.write(f"- {issue}")
        st.subheader("What-If Analysis")
   

# CHATBOT MODE
# =========================
elif mode == " AI Chatbot (Q&A)":
    st.markdown("### AI Chatbot - Ask Questions")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chatbot_response(prompt)
                st.write(response)
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    if st.button("Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()
