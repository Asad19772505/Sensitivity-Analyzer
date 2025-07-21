import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit Page Config
st.set_page_config(page_title="Sensitivity & Monte Carlo Simulator", page_icon="📊", layout="wide")
st.title("🎯 Sensitivity Analysis with Monte Carlo Simulation")

# ------------------------------
# Dummy Input Data
# ------------------------------
st.sidebar.header("🔢 Assumptions")
revenue_growth = st.sidebar.slider("Annual Revenue Growth (%)", 0.0, 30.0, 10.0)
cost_growth = st.sidebar.slider("Annual Cost Growth (%)", 0.0, 20.0, 5.0)
discount_rate = st.sidebar.slider("Discount Rate (WACC) %", 5.0, 15.0, 10.0)

initial_revenue = 1_000_000
initial_cost = 600_000
years = 5
simulations = 1000

# Monte Carlo Simulation
np.random.seed(42)
rev_growth_dist = np.random.normal(revenue_growth / 100, 0.02, simulations)
cost_growth_dist = np.random.normal(cost_growth / 100, 0.02, simulations)

npvs = []
for i in range(simulations):
    revenue = initial_revenue
    cost = initial_cost
    cash_flows = []
    for y in range(1, years + 1):
        revenue *= 1 + rev_growth_dist[i]
        cost *= 1 + cost_growth_dist[i]
        cash_flows.append(revenue - cost)
    npv = sum(cf / ((1 + discount_rate / 100) ** (y + 1)) for y, cf in enumerate(cash_flows))
    npvs.append(npv)

# Show Results
st.subheader("📈 NPV Distribution from Monte Carlo Simulation")
fig, ax = plt.subplots()
sns.histplot(npvs, bins=30, kde=True, color="skyblue")
ax.axvline(np.mean(npvs), color='red', linestyle='--', label=f"Mean NPV: ${np.mean(npvs):,.0f}")
ax.set_title("NPV Distribution")
ax.set_xlabel("NPV")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# Display Statistics
st.markdown("### 📊 Summary Statistics")
st.write(pd.DataFrame({
    "Mean NPV": [np.mean(npvs)],
    "Median NPV": [np.median(npvs)],
    "Min NPV": [np.min(npvs)],
    "Max NPV": [np.max(npvs)],
    "Std Dev": [np.std(npvs)],
}).T.rename(columns={0: "Value"}))

# Send data to GROQ for commentary
st.subheader("🤖 AI-Generated NPV Commentary")
data_for_ai = {
    "initial_revenue": initial_revenue,
    "initial_cost": initial_cost,
    "years": years,
    "revenue_growth": revenue_growth,
    "cost_growth": cost_growth,
    "discount_rate": discount_rate,
    "simulations": simulations,
    "mean_npv": np.mean(npvs),
    "npv_distribution": npvs[:50]  # show first 50 values only for brevity
}

client = Groq(api_key=GROQ_API_KEY)
prompt = f"""
You are the Head of FP&A at a SaaS company. Your task is to analyze a financial Monte Carlo simulation.
Here is the dataset in JSON format:

{data_for_ai}

Provide:
- Key insights from the NPV simulation.
- Areas of concern and key drivers.
- A CFO-ready summary using the Pyramid Principle.
- Actionable recommendations to improve financial performance.
"""

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a financial planning and analysis (FP&A) expert, specializing in SaaS companies."},
        {"role": "user", "content": prompt}
    ],
    model="llama3-8b-8192"
)

ai_commentary = response.choices[0].message.content
st.markdown("### 📖 AI-Generated Commentary")
st.write(ai_commentary)
