# app_dashboard_nav.py
import streamlit as st
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from model import ForecastingModel
from utils import preprocessing

# Page config
st.set_page_config(page_title="School Meal Analytics Navigator",
                   page_icon="📊",
                   layout="wide")

@st.cache_data
def load_data(path="Data/Output/meals_combined.csv"):
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        df = df.sort_values(['school_name','meal_type','date'])
    df = preprocessing(df)
    if "production_cost_total" in df.columns:
        df["production_cost_total"] = df["production_cost_total"]\
            .replace({"\\$":"",",":""}, regex=True).astype(float)
    return df

df = load_data()

# Navigation menu
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select view:", 
                        ("Overview", "Waste Insights", "Forecast Scenario", "Benchmarking"))

# Filters common to views
school_sel = st.sidebar.selectbox("Select School", sorted(df["school_name"].unique()))
meal_sel   = st.sidebar.selectbox("Select Meal Type", sorted(df["meal_type"].unique()))
date_range = st.sidebar.date_input("Select Date Range",
                                   [df['date'].min(), df['date'].max()])

filtered = df[(df["school_name"] == school_sel) &
              (df["meal_type"] == meal_sel) &
              (df["date"] >= pd.to_datetime(date_range[0])) &
              (df["date"] <= pd.to_datetime(date_range[1]))]

if page == "Overview":
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Served", f"{int(filtered['served_total'].sum())}")
    col2.metric("Total Cost", f"${filtered['production_cost_total'].sum():,.2f}")
    col3.metric("Avg Cost per Meal", f"${(filtered['production_cost_total'].sum()/filtered['served_total'].sum()):,.2f}")
    waste_pct = ((filtered['discarded_total'] + filtered['left_over_total']).sum()/filtered['planned_total'].sum())*100
    col4.metric("Waste %", f"{waste_pct:.1f}%")
    st.markdown("---")
    st.header(f"Cost Trend for {school_sel} / {meal_sel}")
    st.line_chart(filtered.set_index('date')['production_cost_total'], use_container_width=True)

elif page == "Waste Insights":
    st.header("Waste Insights")
    filtered2 = filtered.copy()
    filtered2['waste_ratio'] = (filtered2['discarded_total'] + filtered2['left_over_total']) / filtered2['planned_total']
    st.subheader("Top Items by Waste Ratio")
    top_waste = filtered2.sort_values('waste_ratio', ascending=False).head(10)[['name','planned_total','left_over_total','discarded_total','waste_ratio']]
    st.table(top_waste)
    st.markdown("---")
    st.subheader("Waste Ratio Distribution")
    st.bar_chart(top_waste.set_index('name')['waste_ratio'])

elif page == "Forecast Scenario":
    st.header("Forecast Scenario")
    st.sidebar.subheader("What‐if Inputs")
    served_adj    = st.sidebar.slider("Served Total", int(filtered["served_total"].min()), int(filtered["served_total"].max()), int(filtered["served_total"].iloc[-1]))
    planned_adj   = st.sidebar.slider("Planned Total", int(filtered["planned_total"].min()), int(filtered["planned_total"].max()), int(filtered["planned_total"].iloc[-1]))
    discarded_adj = st.sidebar.slider("Discarded Total", int(filtered["discarded_total"].min()), int(filtered["discarded_total"].max()), int(filtered["discarded_total"].iloc[-1]))
    leftover_adj  = st.sidebar.slider("Left Over Total", int(filtered["left_over_total"].min()), int(filtered["left_over_total"].max()), int(filtered["left_over_total"].iloc[-1]))

    features  = ["served_total","planned_total","discarded_total","left_over_total"]
    MODEL_TYPE = "GRU"
    INPUT_DIM  = len(features)
    HIDDEN_DIM = 256
    NUM_LAYERS = 4
    OUTPUT_DIM = 1
    DROPOUT    = 0.25
    model = ForecastingModel(MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT)
    model.load_state_dict(torch.load(f"results/{MODEL_TYPE}.pth", map_location=torch.device("cpu")))
    model.eval()

    input_vec = torch.tensor([[served_adj, planned_adj, discarded_adj, leftover_adj]], dtype=torch.float32)
    lengths   = torch.tensor([1], dtype=torch.long)

    if st.button("Run Prediction"):
        with torch.no_grad():
            pred = model(input_vec, lengths).item()
        st.success(f"Predicted Production Cost: **${pred:,.2f}**")

elif page == "Benchmarking":
    st.header("Benchmarking Across Schools")
    bench = df.groupby('school_name')['production_cost_total'].sum() / df.groupby('school_name')['served_total'].sum()
    bench_df = bench.reset_index().rename(columns={0:'avg_cost_per_meal'})
    st.subheader("Average Cost per Meal by School")
    st.bar_chart(bench_df.set_index('school_name')['avg_cost_per_meal'])

st.markdown("---")
st.caption("© 2025 Your Organization — Meal Analytics Dashboard")