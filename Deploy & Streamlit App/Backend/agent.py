import os
from openai import OpenAI
from dotenv import load_dotenv

from .tools import tools_data, tools_science
from .analysis import run_analysis
from .rag import rag_pipeline
from .ml import predict_uptake

load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    api_key=os.getenv("OPENROUTER_API_KEY"))

MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


# =========================
# TOOL FUNCTIONS (DISPATCHER)
# =========================

def tool_dispatch(name, args):

    if name == "Analysis":
        return run_analysis()

    if name == "Prediction":
        t, p, s, po = map(float, args["input"].split(","))
        return predict_uptake(t, p, s, po)

    if name == "RAG":
        return rag_pipeline(args["query"])

    return "Unknown tool"


# =========================
# MULTI-AGENT SYSTEM
# =========================

class MultiAgentSystem:

    def planner(self, query):

        prompt = f"""
Decide which agents are needed:
DATA, CHEMISTRY, OPTIMIZATION

Query:
{query}

Return JSON like:
{{"data": true, "chemistry": true, "optimization": false}}
"""

        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200)

        return res.choices[0].message.content


    def data_agent(self):
        return run_analysis()


    def chemist_agent(self, query):
        return rag_pipeline(query)


    def optimization_agent(self):
        return "Optimization executed"


    def supervisor(self, results):

        prompt = f"""
You are Supervisor AI.

Combine results:

{results}

Return final decision:
- best material
- optimal conditions
- risk
- confidence
"""

        res = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500)

        return res.choices[0].message.content


    def run(self, query):

        plan = self.planner(query)

        results = {"plan": plan}

        # simplified execution
        results["data"] = self.data_agent()
        results["chemistry"] = self.chemist_agent(query)
        results["optimization"] = self.optimization_agent()

        final = self.supervisor(results)

        return {
            "plan": plan,
            "results": results,
            "final": final }


# =========================
# CHATBOT (STREAMLIT USE)
# =========================

def build_chatbot():
    system = MultiAgentSystem()
    def chat(user_input: str):
        result = system.run(user_input)
        return result["final"]

    return chat