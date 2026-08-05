

import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt


st.set_page_config(layout="wide", page_title=" AI-Driven Decision Intelligence Platform for Carbon Capture R&D", page_icon=":bar_chart:")

st.title(" AI-Driven CO₂ Capture, Process & Materials Optimization Decision Intelligence Platform")
st.image(
        "Picture1.png",
        caption="CO₂ Adsorption Prediction & Analysis",
        use_container_width=True)
st.markdown("""
## AI-Driven CO₂ Capture Decision Intelligence Platform

This platform acts as an **AI-powered engineering assistant** that transforms experimental data into 
actionable insights for CO₂ capture research and process optimization.

### Why This Platform?
Traditional R&D requires manual analysis of experiments, literature review, and trial-and-error optimization.
This system accelerates the decision process by combining:

- **Engineering Data Intelligence**  
  Organizes experimental data, material properties, and process parameters into a structured database.

- **Machine Learning Prediction**  
  Predicts CO₂ uptake performance and identifies important process parameters.

- **Scientific AI Knowledge Engine (RAG)**  
  Connects experimental results with scientific literature to provide context-aware engineering insights.

- **Multi-Agent Decision Intelligence**  
  Combines AI reasoning, optimization algorithms, and engineering constraints to recommend improvements and next experiments.

- **Process Optimization & Scale-Up Support**  
  Uses Bayesian optimization, sensitivity analysis, and uncertainty evaluation to identify better operating conditions.

### Key Outcomes
✓ Faster experimental interpretation  
✓ Data-driven material and process optimization  
✓ AI-assisted R&D decision making  
✓ Reduced trial-and-error experimentation  
✓ Bridge between laboratory research and industrial application
""")
