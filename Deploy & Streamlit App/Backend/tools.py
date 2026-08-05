import json
from .ml import predict_uptake
from .analysis import run_analysis
from .optimization import optimize_process
from .rag import rag_pipeline


# =========================
# SAFE STRING WRAPPERS
# =========================

def analysis_tool(_input: str = ""):
    result = run_analysis()
    return json.dumps(result)


def prediction_tool(x: str):
    # format: "t,p,s,po"
    t, p, s, po = map(float, x.split(","))
    pred = predict_uptake(t, p, s, po)
    return str(pred)


def optimization_tool(_input: str = ""):
    result = optimize_process()
    return json.dumps(result)


def rag_tool(query: str):
    return str(rag_pipeline(query))


def explain_tool(_input: str = ""):
    return "SHAP explanation not implemented"


def recommend_tool(_input: str = ""):
    return "Recommendation not implemented"


# =========================
# EXPORT TOOL LISTS
# =========================

tools_data = [
    {
        "type": "function",
        "function": {
            "name": "Analysis",
            "description": "Run dataset analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": []}}},
    {
        "type": "function",
        "function": {
            "name": "Prediction",
            "description": "Predict CO2 uptake. input format: t,p,s,po",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"] } } }]

tools_science = [
    {
        "type": "function",
        "function": {
            "name": "RAG",
            "description": "Scientific retrieval",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"] } }}]