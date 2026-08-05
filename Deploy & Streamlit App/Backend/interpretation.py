from langchain_openai import ChatOpenAI

def interpret_evaluation(evaluation):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
You are a senior CO2 engineer.

Evaluation:
{evaluation}

Explain:
- performance
- error physics
- sensitivity
- next experiments
"""

    return llm.invoke(prompt).content