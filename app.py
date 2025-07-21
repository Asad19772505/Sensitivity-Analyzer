# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from groq import Groq
from dotenv import load_dotenv

# 🎯 Config
st.set_page_config(page_title="Sensitivity Analysis with Monte Carlo", page_icon="📈", layout="wide")
st.title("📊 AI-Powered Sensitivity Analysis with Monte Carlo Simulation")

# 🔐 Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in .env file.")
    st.stop()

# 📤 Upload Excel file
uploaded_file = st.file_uploader("Upload Excel file with Revenue%, Cost%, Growth Rate%, and WACC%", type=["xlsx"])
if not uploaded_file:
    st.warning("Please upload an Excel file to continue.")
    st.stop()

# 📊 Load and clean data
df = pd.read_excel(uploaded_file)
required_columns = {"Revenue%", "Cost%", "Growth Rate%", "WACC%"}
if not required_columns.issubset(set(df.columns)):
    st.error(f"Excel must contain columns: {required_columns}")
    st.stop()

st.subheader("📈 Input Data Preview")
st.dataframe(df.head())

# 🎲 Monte Carlo Simulation
st.subheader("🎲 Monte Carlo NPV Simulation")

# Assume base revenue is $1M and 5-year model
years = 5
simulations = 1000
base_revenue = 1_000_000

npvs = []

for i in range(simulations):
    row = df.sample(n=1).iloc[0]
    revenue_growth = 1 + row["Revenue%"] / 100
    cost_growth = 1 + row["Cost%"] / 100
    growth_rate = row["Growth Rate%"] / 100
    wacc = row["WACC%"] / 100

    revenue = base_revenue
    cost = base_revenue * 0.6  # assume 60% initial cost
    cash_flows = []

    for _ in range(years):
        revenue *= revenue_growth
        cost *= cost_growth
        cash_flows.append(revenue - cost)

    npv = sum([cf / ((1 + wacc) ** year) for year, cf in enumerate(cash_flows, 1)])
    npvs.append(npv)

# 📊 Plot NPV distribution
fig, ax = plt.subplots()
sns.histplot(npvs, bins=50, kde=True, ax=ax)
ax.set_title("NPV Distribution (Monte Carlo Simulation)")
ax.set_xlabel("Net Present Value")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# 📈 Simulation Statistics
st.subheader("📌 Key Stats")
st.write(f"Mean NPV: ${np.mean(npvs):,.2f}")
st.write(f"Standard Deviation: ${np.std(npvs):,.2f}")
st.write(f"Min NPV: ${np.min(npvs):,.2f}")
st.write(f"Max NPV: ${np.max(npvs):,.2f}")

# 🤖 AI Commentary from GROQ
st.subheader("🤖 AI-Generated Financial Commentary")

# Format data for AI
data_for_ai = {
    "Input Sample": df.sample(n=5, random_state=42).to_dict(orient="records"),
    "Simulation Summary": {
        "Mean NPV": round(np.mean(npvs), 2),
        "Std Dev NPV": round(np.std(npvs), 2),
        "Min NPV": round(np.min(npvs), 2),
        "Max NPV": round(np.max(npvs), 2)
    }
}

client = Groq(api_key=GROQ_API_KEY)
prompt = f"""
You are the Head of FP&A at a SaaS company. Your task is to build a Financial Model and provide:
- Key insights from the data.
- Areas of concern and key drivers for variance.
- A CFO-ready summary using the Pyramid Principle.
- Actionable recommendations to improve financial performance.

Here is the full dataset in JSON format:
{data_for_ai}
"""

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a financial planning and analysis (FP&A) expert, specializing in SaaS companies."},
        {"role": "user", "content": prompt}
    ],
    model="llama3-8b-8192",
)

ai_commentary = response.choices[0].message.content
st.markdown(ai_commentary)
