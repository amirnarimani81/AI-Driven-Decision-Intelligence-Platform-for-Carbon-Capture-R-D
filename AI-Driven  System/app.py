import streamlit as st

st.set_page_config(page_title="CO2 Capture AI Agent", layout="wide")



# =========================
# Multi-Agent LLM System for Industrial Carbon Capture Optimization
# =========================

import streamlit as st
import os
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
import tempfile
import PyPDF2
import math
from datetime import datetime
from scipy.stats import pearsonr
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

from bayes_opt import BayesianOptimization
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import warnings
warnings.filterwarnings("ignore")


# =========================LOAD ENV=========================
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")

DEEPSEEK_MODEL = "deepseek/deepseek-chat"
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")



client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"))



# --------------------------- DATABASE ---------------------------

DB_PATH = "preprocessor_output.db"
TABLE_NAME = "process_data"



def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    return df


def add_experiment(materials, method, surface, pore_volume, micropore,
                   mixing_time, flow_rate, pressure, uptake, efficiency, temp):
    """Add a new experiment to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (
            materials, 
            method,
            "surface area (m2/g)", 
            "total pore volume(cm3/g)", 
            "micropore volume (cm3/g)",
            "Mixing Time (min)", 
            "Flow Rate (L/min)", 
            "pressure (bar)",
            "co2 uptake (mmol/g)", 
            "Efficiency (%)", 
            "temp (°c)"
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (materials, method, surface, pore_volume, micropore,
          mixing_time, flow_rate, pressure, uptake, efficiency, temp))
    
    conn.commit()
    conn.close()

def get_stats():
    df = load_data()
    if len(df) == 0:
        return {"count": 0, "avg_uptake": 0, "max_uptake": 0}
    return {"count": len(df), "avg_uptake": df["co2 uptake (mmol/g)"].mean(), "max_uptake": df["co2 uptake (mmol/g)"].max()}



# Get the most recent experiment (for storytelling and decision context)
def get_last_experiment():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} ORDER BY rowid DESC LIMIT 1", conn)
    conn.close()
    
    return df.iloc[0].to_dict()  
   

# --------------------------- EDA & INSIGHTS ---------------------------

def analyze_dataset(df):
    summary = df.describe().to_dict()
    corr = df.corr(numeric_only=True)

    insights = []

    # CORRELATION INSIGHTS
    if "temp (°c)" in df.columns and "co2 uptake (mmol/g)" in df.columns:
        r, _ = pearsonr(df["temp (°c)"], df["co2 uptake (mmol/g)"])
        insights.append(f"Temperature vs Uptake correlation: {round(r, 3)}")

    if "pressure (bar)" in df.columns and "co2 uptake (mmol/g)" in df.columns:
        r, _ = pearsonr(df["pressure (bar)"], df["co2 uptake (mmol/g)"])
        insights.append(f"Pressure vs Uptake correlation: {round(r, 3)}")
                        
    # PEAK OPERATING REGION
    if "temp (°c)" in df.columns and "co2 uptake (mmol/g)" in df.columns:
        peak_temp_range = df.groupby(
            pd.cut(df["temp (°c)"], 10))["co2 uptake (mmol/g)"].mean().idxmax()
    else:
        peak_temp_range = "Not available"

    # ANOMALY DETECTION (Z-SCORE)
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df) > 0:
        z = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        anomalies = int((z > 3).sum().sum())
    else:
        anomalies = 0

    # INDUSTRIAL KPIs
    result = {
        "summary": summary,
        "correlation_matrix": corr.to_dict(),
        "insights": insights,
        "peak_temperature_range": str(peak_temp_range),
        "anomalies": anomalies}

    if "co2 uptake (mmol/g)" in df.columns:
        result.update({
            "avg_uptake": float(df["co2 uptake (mmol/g)"].mean()),
            "max_uptake": float(df["co2 uptake (mmol/g)"].max()) })

    return result

# --------------------------- ML MODEL ---------------------------

MODEL_COLUMNS = [
    'surface area (m2/g)',
    'total pore volume(cm3/g)',"Flow Rate (L/min)",
    'micropore volume (cm3/g)',"Mixing Time (min)",
    'temp (°c)',
    'pressure (bar)']

TARGET_COLUMN = 'co2 uptake (mmol/g)'




def load_model_preprocessor():
    model = joblib.load("uptake_model.pkl")
    preprocessor = joblib.load( "preprocessor.pkl")
    return model, preprocessor


model, preprocessor = load_model_preprocessor()



def predict_uptake(feature_dict):
    df = pd.DataFrame([feature_dict])
    df = df.reindex(columns=MODEL_COLUMNS, fill_value=0)
    X = preprocessor.transform(df)
    return float(model.predict(X)[0])


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

# --------------------------- PROCESS INTELLIGENCE MODULE (Sensitivity + Uncertainty + Optimization) ---------------------------


def sensitivity_analysis(feature_dict):
    base = predict_uptake(feature_dict)
    sens = {}
    for key in MODEL_COLUMNS:
        perturbed = feature_dict.copy()
        perturbed[key] *= 1.01  # 1% increase
        new_pred = predict_uptake(perturbed)
        sens[key] = (new_pred - base)/base 
    return sens


def what_if_analysis(feature_dict):
    results = []
    base_pred = predict_uptake(feature_dict)
    for param in ["temp (°c)", "pressure (bar)"]:
        for factor in [0.8,0.9,1.0,1.1,1.2]:
            new = feature_dict.copy()
            new[param]  *= factor
            new_pred = predict_uptake(new)
            results.append({"Parameter":param, "Change":f"{int((factor-1)*100)}%",
                            "New Value":round(new[param],2),"Uptake":round(new_pred,3),
                            "Delta":round(new_pred-base_pred,3)})
    return pd.DataFrame(results)



def predict_with_uncertainty(feature_dict, n_samples=30):

    preds = []

    keys = list(feature_dict.keys())
    base_values = np.array(list(feature_dict.values()))

    
    for _ in range(n_samples):
        noise = np.random.normal(0,0.02,len(keys))  # 2% noise
        noisy_values = base_values + noise
        noisy_dict = dict(zip(keys, noisy_values))
        preds.append(predict_uptake(noisy_dict))

    preds = np.array(preds)
    mean = np.mean(preds)
    std = np.std(preds)

   
    cv = (std / mean) * 100 
    ci_lower = np.percentile(preds, 5)
    ci_upper = np.percentile(preds, 95)
    confidence = max( 0, 1 - (std / mean))

    return {
        "mean": float(mean),
        "std": float(std),
        "cv": float(cv),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence": float(confidence)}


def optimization(df):

    def objective(
        surface_area,
        pore_volume,
        micropore,
        temperature,
        pressure,
        flow,
        time):

        features = {
            "surface area (m2/g)": surface_area,
            "total pore volume(cm3/g)": pore_volume,
            "micropore volume (cm3/g)": micropore,
            "temp (°c)": temperature,
            "pressure (bar)": pressure,
            "Flow Rate (L/min)": flow,
            "Mixing Time (min)": time}

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
            "time": (10, 120)},

        random_state=42,
        verbose=0 )

    optimizer.maximize(
        init_points=10,
        n_iter=30 )

    best = optimizer.max["params"]

    best_conditions = {
        "surface area (m2/g)": best["surface_area"],
        "total pore volume(cm3/g)": best["pore_volume"],
        "micropore volume (cm3/g)": best["micropore"],
        "temp (°c)": best["temperature"],
        "pressure (bar)": best["pressure"],
        "Flow Rate (L/min)": best["flow"],
        "Mixing Time (min)": best["time"]}

    return {
        "best_conditions": best_conditions,
        "best_uptake": optimizer.max["target"]}


# ---------------------------Engineering Agent ---------------------------

def reactor_agent(feature_dict):
    temp = feature_dict.get("temp (°c)", 25)
    pressure = feature_dict.get("pressure (bar)", 5)
    flow = feature_dict.get("Flow Rate (L/min)", 10)
    time = feature_dict.get("Mixing Time (min)", 45)

    
    efficiency = (
        0.4 * np.tanh(temp / 150) +   # scaler for normalizatyion 150
        0.3 * np.tanh(pressure / 40) +   # scaler for normalizatyion 40
        0.2 * np.tanh(flow / 20) +
        0.1 * np.tanh(time / 60))

    constraints = {
        "safe_pressure": pressure < 50,
        "safe_temp": temp < 200}
 

    return {
        "efficiency": float(np.clip(efficiency, 0, 1)),
        "safe": constraints["safe_pressure"] and constraints["safe_temp"]}



def material_agent(feature_dict):
    s = feature_dict.get("surface area (m2/g)", 500)
    p = feature_dict.get("total pore volume(cm3/g)", 0.5)
    m = feature_dict.get("micropore volume (cm3/g)", 0.2)

    score = (
        0.5 * (s / 2000) +
        0.3 * (p / 1.5) +
        0.2 * (m / 0.8))
    return {
        "material_score": round(score * 100, 2),
        "adsorption_potential": round(score, 3)}

def cost_agent(feature_dict):  # cost in $/ton CO2 captured, lower is better
    surface = feature_dict.get("surface area (m2/g)", 500)
    temp = feature_dict.get("temp (°c)", 25)
    pressure = feature_dict.get("pressure (bar)", 5)

    capex = 0.01 * (surface / 1000)
    opex = 0.003 * temp + 0.08 * pressure

    total_cost = capex + opex

    return round(total_cost, 3)


# --------------------------- RAG (SIMPLIFIED) ---------------------------

PAPERS_PATH = "General Articles"
VECTOR_DB_PATH = "vector_db"

# =========================
# EMBEDDINGS 
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# =========================
# BUILD VECTOR DB
# =========================
def build_vector_db():
    docs = []
    
    if os.path.exists(PAPERS_PATH):
        for file in os.listdir(PAPERS_PATH):
            if file.endswith(".pdf"):
                file_path = os.path.join(PAPERS_PATH, file)
                try:
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        for page in reader.pages[:3]:  # limit to first 3 pages
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
    db = FAISS.from_documents(chunks[:200], embeddings)
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    db.save_local(VECTOR_DB_PATH)
    
    return db

# =========================
# LOAD VECTOR DB
# =========================
def load_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        try:
            return FAISS.load_local(
                VECTOR_DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True)
        except:
            pass

    return build_vector_db()

# =========================
# INIT DB (agent-ready)
# =========================
@st.cache_resource
def get_vector_db():
    return load_vector_db()

DB = get_vector_db()

# =========================
# RAG PIPELINE (CORE AGENT MEMORY)
# =========================
def rag_pipeline(query):
    try:
        docs = DB.similarity_search(query, k=2)  # retrieve top 2 most relevant chunks
        context = "\n\n".join([d.page_content[:500] for d in docs])
    except:
        context = "No relevant documents found."
    
    prompt = f"""
You are a senior Chemical Engineering researcher.

Analyze CO2 capture scientific documents.

Provide:

INTRODUCTION:
METHODS:
MATERIALS:
OPERATIONAL PARAMETERS:
RESULTS:
CONCLUSION:
ENGINEERING_INSIGHT:
RECOMMENDED_NEXT_EXPERIMENT:
"""

    res = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt[:2000]}],
        temperature=0.2,
        max_tokens=120)
    
    return res.choices[0].message.content


# --------------------------- DECISION ENGINE ---------------------------
def decision_engine(query, eda, ml, opt, rag):

    prompt = f"""
You are an expert Chemical Process Decision Agent for CO2 capture systems with deep knowledge in:
- Adsorption thermodynamics (ΔH_ads, ΔS_ads, isotherms)
- Mass transfer kinetics (diffusion coefficients, rate constants)
- Porous materials (MOFs, zeolites, carbon)
- Process optimization (Pareto frontier, trade-offs)

Your role is NOT just to summarize.
You must perform:

1. Evidence evaluation (EDA + ML + Optimization + Literature) with engineering reasoning
2. Risk assessment (process + model + uncertainty) with quantitative limits
3. Root cause reasoning (causal reasoning, not description)
4. Next experiment design (active learning logic

========================
INPUT DATA
========================

QUERY:
{query}

EDA INSIGHTS:
{eda.get('insights', [])}

ML PERFORMANCE:
- R2: {ml.get('metrics', {}).get('r2', 'N/A')}
- RMSE: {ml.get('metrics', {}).get('rmse', 'N/A')}

OPTIMIZATION RESULT:
- Best Uptake: {opt.get('best_uptake', 'N/A')} mmol/g
- Optimal Conditions: {opt.get('best_conditions', {})}

RAG KNOWLEDGE:
{rag[:500] if rag else "No literature found"}

========================
DECISION FRAMEWORK
========================

Step 1: Analyze consistency between:
- ML prediction vs Optimization
- EDA trends vs RAG knowledge

Step 2: Identify:
- performance bottlenecks
- physical limitations
- data uncertainty issues
- Pressure drop - excessive ΔP, energy loss


Step 3: Assign risk score:
- LOW: R²≥0.90, all sources agree, CV<8%, mechanism understood
- MEDIUM: R² 0.70-0.89, partial conflict, CV 8-15%
- HIGH: R²<0.70, major conflict, CV>15%, outside stability limit

Step 4: Design next experiment using:
- high uncertainty regions
- near-optimal region
- physically meaningful perturbations

========================
OUTPUT FORMAT (STRICT JSON ONLY)
========================

{{
  "insight": "2-3 sentence deep engineering interpretation",
  "root_cause": "single most likely physical/chemical reason",
  "risk": "LOW | MEDIUM | HIGH",
  "next_experiment": {{
      "objective": "exploration | optimization | validation",
      "temperature": "...",
      "pressure": "...",
      "material": "...",
      "reason": "why this experiment is selected"}}}}

IMPORTANT:
- No extra text
- No markdown
- Only valid JSON
"""
    
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior chemical engineering AI decision system "
                    "specialized in CO2 capture process optimization, Bayesian reasoning, and experimental design.")},
            {"role": "user", "content": prompt}],
        temperature=0.1)

    try:
        return json.loads(res.choices[0].message.content)
    except:
        return {
            "insight": "Decision completed with fallback reasoning",
            "root_cause": "System uncertainty or data inconsistency",
            "risk": "MEDIUM",
            "confidence_score": 0.5,
            "decision_logic": "Fallback triggered due to parsing failure",
            "next_experiment": {
                "objective": "validation",
                "temperature": 120,
                "pressure": 10,
                "material": "MOF-5",
                "reason": "safe default condition for validation"}}


# ---------------------------  Experimental design (DOE)  ---------------------------
def next_experiment_suggestion(decision_dic, best_conditions=None):
    prompt = f"""
You are an expert in chemical engineering experimental design (DOE) for CO2 capture systems.

Your task is to design 3 high-value next experiments based on:
- model predictions
- optimization results
- uncertainty and risk
- industrial feasibility

Decision Summary:
{decision_dic}

Best Known Operating Region (if available):
{best_conditions}

========================
REQUIREMENTS
========================
Design 3 experiments that:

1. Explore OPTIMUM region (refinement around best conditions)
2. Explore EDGE conditions (stress testing)
3. Explore NEW MATERIAL performance

Each experiment MUST include:
- Temperature (°C)
- Pressure (bar)
- Material type (MOF-5 / ZIF-8 / Activated Carbon / Graphene)
- Flow rate
- Mixing time
- Expected CO2 uptake (mmol/g)
- Engineering justification (1-2 lines)
- Purpose: (Optimization / Robustness / Exploration)
- Material treatment method (chemical modification or activation method)
...
"""
    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600)
    return res.choices[0].message.content


# --------------------------- PDF-REPORT ---------------------------

def generate_pdf_report(text, filename="decision_report.pdf"):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    style = styles["Normal"]

    story = []

    for line in text.split("\n"):
        story.append(Paragraph(line, style))
        story.append(Spacer(1, 6))

    doc.build(story)

    return filename

# --------------------------- LANGCHAIN TOOLS (Multi-Agent) ---------------------------
@tool
def add_experiment_tool(materials: str, surface: float, pore_volume: float, micropore: float,
                        temperature: float, pressure: float, uptake: float, efficiency: float,
                        flow_rate: float = 8.5, mixing_time: float = 45) -> str:
    """Add a new experiment to the database."""
    add_experiment(
        materials=materials,
        method="User",
        surface=surface,
        pore_volume=pore_volume,
        micropore=micropore,
        mixing_time=mixing_time,
        flow_rate=flow_rate,
        pressure=pressure,
        temperature=temperature,  # ← ADD THIS LINE (missing)
        uptake=uptake,
        efficiency=efficiency)
    return f"Experiment added. Now {get_stats()['count']} records."



@tool
def analyze_data_tool(empty_input: str = "") -> str:
    """Get statistics and correlations from existing data."""
    df = load_data()
    if len(df)==0: return "No data."
    stats = get_stats()
    insights = analyze_dataset(df)["insights"]
    return f"Experiments: {stats['count']}, Avg uptake: {stats['avg_uptake']:.2f}, Max: {stats['max_uptake']:.2f}\nInsights: {insights}"



@tool
def predict_uptake_tool(surface: float = 800, pore_volume: float = 0.6, micropore: float = 0.3,
                        temperature: float = 25, pressure: float = 5, flow_rate: float = 10,
                        mixing_time: float = 45) -> str:
    """Predict CO2 uptake for given inputs."""
    feat = {"surface area (m2/g)":surface, "total pore volume(cm3/g)":pore_volume,
            "micropore volume (cm3/g)":micropore, "Flow Rate (L/min)":flow_rate,
            "Mixing Time (min)":mixing_time, "temp (°c)":temperature, "pressure (bar)":pressure}
    pred = predict_uptake(feat)
    return f"Predicted uptake: {pred:.2f} mmol/g."



@tool
def optimize_conditions_tool(empty_input: str = "") -> str:
    """Bayesian optimization to find best T and P."""
    df = load_data()
    if len(df)==0: return "Not enough data."
    opt = optimization(df)
    return f"Optimal: T={opt['best_conditions']['temp (°c)']:.1f}°C, P={opt['best_conditions']['pressure (bar)']:.1f} bar → {opt['best_uptake']:.2f} mmol/g."


@tool
def simulate_process_tool(surface: float = 800, pore_volume: float = 0.6, micropore: float = 0.3,
                          temperature: float = 25, pressure: float = 5, flow_rate: float = 10,
                          mixing_time: float = 45) -> str:
    """What-if, sensitivity, risk analysis."""
    feat = {"surface area (m2/g)":surface, "total pore volume(cm3/g)":pore_volume,
            "micropore volume (cm3/g)":micropore, "Flow Rate (L/min)":flow_rate,
            "Mixing Time (min)":mixing_time, "temp (°c)":temperature, "pressure (bar)":pressure}
    base = predict_uptake(feat)
    whatif = what_if_analysis(feat).head(4).to_string()
    sens = sensitivity_analysis(feat)
    risk = predict_with_uncertainty(feat)
    return f"Base: {base:.2f} mmol/g\n\nWhat-if:\n{whatif}\n\nSensitivity:\n{sens}\n\nRisk:\nCV={risk['cv']:.1f}%, Confidence={risk['confidence']*100:.1f}%"


@tool
def reactor_performance_tool(temperature: float = 25, pressure: float = 5) -> str:
    """Reactor efficiency and safety."""
    feat = {"temp (°c)":temperature, "pressure (bar)":pressure, "Flow Rate (L/min)":10, "Mixing Time (min)":45}
    r = reactor_agent(feat)
    return f"Efficiency: {r['efficiency']*100:.1f}%, Safety: {'OK' if r['safe'] else 'LIMIT EXCEEDED'}."



# --------------------------- MULTI-AGENT SUPERVISOR ---------------------------
tools = [add_experiment_tool, analyze_data_tool, predict_uptake_tool, optimize_conditions_tool,ml_evaluation_tool,
         simulate_process_tool, scientific_context_tool, reactor_performance_tool]

llm = ChatOpenAI(
    model=MODEL,
    temperature=0.2,
    openai_api_key=api_key,
    base_url=base_url)

CHEMICAL_ENGINEERING_PROMPT = """You are a Senior Chemical Engineering AI Agent specialized in CO₂ capture and sequestration.

YOUR EXPERTISE:
- Thermodynamics: Adsorption isotherms, enthalpy/entropy (ΔH, ΔS), equilibrium constants
- Kinetics: Rate constants, diffusion mechanisms (Knudsen, molecular, surface), mass transfer
- Materials: MOFs, ZIFs, Activated Carbon, pore structure, surface chemistry
- Risk Assessment: Safety limits, degradation, thermal runaway

YOUR CAPABILITIES:
- Use available tools to analyze data, predict uptake, optimize conditions
- Provide chemical engineering reasoning with units (kJ/mol, mmol/g, bar, °C)
- Give quantitative, actionable recommendations

RULES:
- Always use engineering units
- If data missing, suggest what experiments are needed
- Be precise and practical

You have access to the following tools:
{tools}

Use the following format:
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Observation: the result
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer (include engineering reasoning and units)

Begin!
Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(CHEMICAL_ENGINEERING_PROMPT)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)


def run_multi_agent(question):
    """
    Multi-agent reasoning system (ReAct + tool selection)
    """

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]

    except Exception as e:
        return f"Agent error: {str(e)}"


# --------------------------- AI AGENT ANALYSIS-ORCHESTRA PIPELINE ---------------------------
def full_agent_analysis(query="Analyze CO2 capture"):
    
    #  DATA LAYER (Deterministic)
    df = load_data()
    eda = analyze_dataset(df)
    opt = optimization(df)
    ml = evaluate_model(df)

    default_feat = {
        "surface area (m2/g)": 800,
        "total pore volume(cm3/g)": 0.6,
        "micropore volume (cm3/g)": 0.3,
        "Flow Rate (L/min)": 10,
        "Mixing Time (min)": 45,
        "temp (°c)": 25,
        "pressure (bar)": 5}

    #  ENGINEERING AGENTS (Deterministic micro-agents)
    risk = predict_with_uncertainty(default_feat)
    reactor = reactor_agent(default_feat)
    material = material_agent(default_feat)
    cost = cost_agent(default_feat)


    
    #  RAG (Knowledge layer)
    rag = rag_pipeline(query)

    # DECISION ENGINE (LLM only here)
    decision_dict = decision_engine(query, eda, ml, opt, rag)

    # DOE GENERATION (LLM tool)
    doe_suggestion = next_experiment_suggestion(decision_dict)
    decision_dict["next_experiments"] = doe_suggestion
    
    #  REPORT GENERATION (LLM synthesis layer)
    report = generate_complete_industrial_report(
        query=query,
        eda=eda,
        ml=ml,
        opt=opt,
        rag=rag,
        decision_dict=decision_dict,
        reactor=reactor,
        material=material,
        cost=cost,
        risk=risk)

    pdf_file = generate_pdf_report(report)

    # ORCHESTRATION OUTPUT 
    results = {
        "eda": eda,
        "optimization": opt,
        "risk": risk,
        "reactor": reactor,
        "material": material,
        "cost": cost,
        "rag": rag,
        "decision": decision_dict,
        "ml": ml,
        "get_last_experiment": get_last_experiment,
        "doe": doe_suggestion}

    return results, report, pdf_file



# --------------------------- SCALE-UP CLASS ---------------------------
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
        
  
       
        return {
            "scale_factor": round(scale_factor, 1),           
            "bed_diameter_m": round(diameter_m, 2),           
            "bed_height_m": round(height_m, 2),               
            "flow_m3h": round(flow_m3h, 1),                   
            "compressor_power_kW": round(compressor_power, 1)}


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


def llm_explain_scale_up_code(lab_data, results):
    """
    Explain the industrial scale-up results from a chemical engineering perspective.
    """
    
    prompt = f"""
You are a Senior Chemical Process Engineer with 20 years of experience in:
- Fixed bed adsorption reactors or pressurized reactors
- CO₂ capture systems
- Industrial scale-up (Buckingham π theorem)
- Process economics

========================================
INPUT DATA (Lab Scale):
========================================
- Bed diameter: {lab_data.get('bed_diameter_cm', 5)} cm
- Bed height: {lab_data.get('bed_height_cm', 15)} cm
- Flow rate: {lab_data.get('flow_rate_Lmin', 0.5)} L/min
- Breakthrough time: {lab_data.get('breakthrough_time_min', 30)} min
- Uptake capacity: {lab_data.get('uptake_mmol_g', 4.2)} mmol/g
- Pressure: {lab_data.get('pressure_bar', 5)} bar

========================================
CALCULATION RESULTS:
========================================
- Scale Factor: {results.get('scale_factor', 0)}
- Bed Diameter: {results.get('bed_diameter_m', 0)} m
- Bed Height: {results.get('bed_height_m', 0)} m
- Flow Rate: {results.get('flow_m3h', 0)} m³/h
- Compressor Power: {results.get('compressor_power_kW', 0)} kW
- L/D Ratio: {results.get('bed_height_m', 0) / results.get('bed_diameter_m', 1):.2f}

========================================
YOUR TASK:
========================================
Explain these scale-up results to a chemical engineer.

Provide:
1. **Executive Summary** - What do these numbers mean?
2. **Scale Factor Analysis** - Is this scale factor realistic?
3. **Geometric Interpretation** - What does the L/D ratio tell us?
4. **Engineering Warnings** - Potential issues at this scale
5. **Recommendation** - Next steps for scale-up

"""

    response = client.chat.completions.create(
        model=MODEL,  
        messages=[
            {"role": "system", "content": "You are a Senior Chemical Process Engineer. Give practical, actionable advice."},
            {"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600)
    
    return response.choices[0].message.content

