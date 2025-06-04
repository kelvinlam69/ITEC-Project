# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import requests

st.set_page_config(layout="centered", page_title="Starbucks Revenue Forecast")

st.title("Starbucks Revenue Forecasting App")

# Load data
df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# Forecast revenue using ARIMA
model = ARIMA(df["revenue"], order=(1,1,1))
results = model.fit()
forecast = results.get_forecast(steps=4)
predicted = forecast.predicted_mean
ci = forecast.conf_int()
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.offsets.QuarterEnd(), periods=4, freq='Q')

# --- User Input ---
st.sidebar.header("Audit Committee Tools")
user_expected_revenue = st.sidebar.number_input("Enter your expected revenue (next quarter):", min_value=0.0, value=round(predicted.iloc[0], 2))

# --- AI-generated Summary ---
ai_summary = f"""
Our ARIMA-based forecast estimates Starbucks' next quarter revenue to be approximately ${predicted.iloc[0]:,.2f}. The audit team may consider this against the expected revenue input (${user_expected_revenue:,.2f}) and recent SG&A trends. The revenue trend appears {'aggressive' if user_expected_revenue > predicted.iloc[0] * 1.1 else 'reasonable'}, given the SG&A trajectory.
"""

# --- Plot Forecast ---
st.subheader("Revenue Forecast (Next 4 Quarters)")
fig, ax = plt.subplots(figsize=(10, 5))
df["revenue"].plot(ax=ax, label="Actual")
ax.plot(future_dates, predicted, label="Forecast", linestyle="--", marker="o")
ax.fill_between(future_dates, ci.iloc[:, 0], ci.iloc[:, 1], color="lightgray", alpha=0.5, label="95% Confidence Interval")
ax.axhline(user_expected_revenue, color="red", linestyle=":", label="Expected Revenue")
ax.legend()
ax.set_title("Starbucks Revenue Forecast")
ax.set_ylabel("Revenue")
ax.grid(True)
st.pyplot(fig)

# --- SG&A vs Revenue ---
st.subheader("New Insight: SG&A vs Revenue")
insight_fig = px.line(df.reset_index(), x="date", y=["revenue", "marketing_spend"], 
                      labels={"value": "USD", "variable": "Metric"},
                      title="Revenue vs SG&A (Marketing Spend)")
st.plotly_chart(insight_fig)

# --- CPI Integration ---
st.subheader("Live Economic Indicator: CPI (Consumer Price Index)")
api_key = "INSERT_YOUR_FRED_API_KEY"
cpi_url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key={api_key}&file_type=json"
cpi_data = requests.get(cpi_url).json()

try:
    latest_obs = cpi_data['observations'][-1]
    latest_cpi = float(latest_obs['value'])
    st.markdown(f"**Latest CPI (from FRED):** {latest_cpi:.2f} — {latest_obs['date']}")
except:
    st.warning("CPI data unavailable at the moment.")

# --- Revenue Risk Flag ---
if predicted.iloc[0] > df["revenue"].iloc[-1] * 1.1 and df["marketing_spend"].iloc[-1] < df["marketing_spend"].mean():
    st.error("⚠️ Revenue growth appears aggressive while SG&A is below average. Potential overstatement risk.")

# --- Summary ---
st.subheader("Audit Summary")
st.write(ai_summary)

st.markdown("---")
st.caption("App created by Kelvin Lam & Nick Mitchoff | ITEC 3155 / ACTG 4155 - Final Project")
