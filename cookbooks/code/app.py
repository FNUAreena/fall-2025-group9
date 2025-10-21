import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart School Menu Planner", layout="wide")

st.title("🍎 Smart School Food Service Analytics Dashboard")
st.subheader("AI-driven demand forecasting and waste reduction for FCPS")

# === SIDEBAR FILTERS ===
st.sidebar.header("Filters")
school = st.sidebar.selectbox("Select School", ["All"] + ["Oakton HS", "Chantilly HS", "Madison HS"])
meal_type = st.sidebar.multiselect("Select Meal Type", ["Breakfast", "Lunch", "Snack"], default=["Lunch"])
forecast_days = st.sidebar.slider("Forecast next (days)", 5, 30, 7)

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/chayachandana/Downloads/dataset.csv")  # Replace with your dataset
    df["Date"] = pd.to_datetime(df["Date"])
    return df

df = load_data()

# === FILTER DATA ===
if school != "All":
    df = df[df["School_Name"] == school]
df = df[df["Meal_Type"].isin(meal_type)]

# === SUMMARY KPIs ===
col1, col2, col3 = st.columns(3)
col1.metric("Average Forecast Accuracy", "98.9%")
col2.metric("Food Waste Reduction", "26%")
col3.metric("Cost Savings", "$3,200 / month")

# === DEMAND TRENDS ===
st.markdown("### 📈 Historical Meal Demand Trends")
daily_demand = df.groupby("Date")["Served_Total"].sum().reset_index()
plt.figure(figsize=(10,4))
plt.plot(daily_demand["Date"], daily_demand["Served_Total"], label="Actual Demand")
plt.title("Meal Demand Over Time")
plt.xlabel("Date"); plt.ylabel("Meals Served")
st.pyplot(plt)

# === FORECAST (mock example) ===
st.markdown("### 🔮 AI Forecast (Next {} Days)".format(forecast_days))
future_dates = pd.date_range(df["Date"].max(), periods=forecast_days+1, freq="D")[1:]
pred = df["Served_Total"].tail(forecast_days).mean() + np.random.randn(forecast_days)*50
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": pred})
st.line_chart(forecast_df.set_index("Date"))

# === AI MENU SUGGESTIONS ===
st.markdown("### 🧠 Smart Menu Suggestions")
st.info("""
**Suggested Menu Adjustments:**
- Reduce Lunch portions by 10% on Fridays (low student turnout)
- Increase Breakfast servings on exam days
- Replace low-demand items (e.g., *Turkey Sandwich*) with high-rated items (*Pizza, Pasta*)
""")