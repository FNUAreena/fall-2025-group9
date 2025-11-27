# app.py
import os
import math
import time
from datetime import datetime
from typing import Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Path setup ---
import sys

# Project root: C:\Github\fall-2025-group9
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Univariate module directory: C:\Github\fall-2025-group9\src\component\univariate
UNIVARIATE_DIR = os.path.join(BASE_DIR, "src", "component", "univariate")

# Add univariate folder to sys.path so `forecasting` and `utils` are top-level modules
sys.path.insert(0, UNIVARIATE_DIR)

# Import your project modules exactly like in univariate_main.py
from forecasting import forecast_future_dates, next_days
from utils import load_and_aggregate_district, safe_time_split

# --- Config / paths ---
CSV_PATH = os.path.join(BASE_DIR, "src", "Data", "Output", "meals_combined.csv")
DATE_COL = "date"
TARGET_COL = "production_cost_total"

# Model directory and stub path (only dirname is used inside forecasting.py)
MODEL_DIR = os.path.join(UNIVARIATE_DIR, "LSTM_models")
MODEL_STUB = os.path.join(MODEL_DIR, "LSTM.pth")  # just a placeholder; forecasting swaps filename

# Example train/test/forecast image
TRAIN_TEST_IMAGE = os.path.join(
    BASE_DIR, "demo", "images", "univariate_plots", "LSTM_train_test_forecast_example.png"
)

st.set_page_config(page_title="School Production Cost Forecasting", layout="wide")

# --- Helpers ---
@st.cache_data
def load_series(csv_path=CSV_PATH):
    dates, values, _, _ = load_and_aggregate_district(
        CSV_PATH=csv_path, DATE_COL=DATE_COL, TARGET_COL=TARGET_COL, dayfirst="auto", debug=False
    )
    records = [(school, meal, pd.to_datetime(dt), float(v)) for (school, meal, dt), v in zip(dates, values.reshape(-1))]
    df = pd.DataFrame(records, columns=["school_name", "meal_type", DATE_COL, TARGET_COL])
    df = df.sort_values(["school_name", "meal_type", DATE_COL]).reset_index(drop=True)
    return df

def get_unique_school_meal(df: pd.DataFrame) -> List[Tuple[str, str]]:
    uniq = df[['school_name', 'meal_type']].drop_duplicates()
    return [(r.school_name, r.meal_type) for r in uniq.itertuples(index=False)]

def agg_total_by_date(df: pd.DataFrame) -> pd.DataFrame:
    # produce a daily district total (sum over all schools & meals)
    daily = df.groupby(DATE_COL, sort=True)[TARGET_COL].sum().reset_index()
    return daily

def business_days_between(start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
    """Return number of business days from (start_date, exclusive) to end_date (inclusive).
    If end_date <= start_date -> 0
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if end <= start:
        return 0
    # pandas.bdate_range includes start if exact match; we want days AFTER start
    rng = pd.bdate_range(start + pd.offsets.BDay(1), end)
    return len(rng)

@st.cache_data(show_spinner=False)
def compute_all_school_forecasts_for_horizon(df: pd.DataFrame, k_steps: int, model_type="LSTM"):
    """Return a DataFrame with concatenated forecasts for each (school,meal) for next k_steps.
       This tries to call forecast_future_dates for each pair and aggregates results.
    """
    combos = df[['school_name', 'meal_type']].drop_duplicates()
    all_outs = []
    total_failed = 0
    for _, row in combos.iterrows():
        school = row['school_name']
        meal = row['meal_type']
        try:
            out, bt_true, bt_pred = forecast_future_dates(
                csv_path=CSV_PATH,
                date_col=DATE_COL,
                target_col=TARGET_COL,
                k_steps=k_steps,
                model_type=model_type,
                model_path=MODEL_STUB,   # use our model directory
                school_name=school,
                meal_type=meal,
            )
            if out is not None and not out.empty:
                all_outs.append(out)
        except Exception as e:
            total_failed += 1
            # skip missing models / too-short series
            continue
    if len(all_outs) == 0:
        return pd.DataFrame(), total_failed
    df_all = pd.concat(all_outs, ignore_index=True)
    return df_all, total_failed

def aggregate_forecasts_for_date(df_forecasts: pd.DataFrame, target_date: pd.Timestamp) -> float:
    """Sum target_col across all forecasts where forecast_date == target_date"""
    mask = pd.to_datetime(df_forecasts['forecast_date']).dt.date == pd.to_datetime(target_date).date()
    if mask.sum() == 0:
        return float('nan')
    total = df_forecasts.loc[mask, TARGET_COL].sum()
    return float(total)

# --- UI Layout ---
st.title("School Production Cost — Forecasting Dashboard")

df_series = load_series()

left_col, right_col = st.columns([2, 1])

with left_col:
    st.header("1) Forecast a single School + Meal")
    school_names = sorted(df_series['school_name'].unique())
    selected_school = st.selectbox("Choose School", school_names)
    meal_types = sorted(df_series[df_series['school_name'] == selected_school]['meal_type'].unique())
    selected_meal = st.selectbox("Choose Meal Type", meal_types)
    k_steps = st.number_input("Days to forecast (business days)", min_value=1, max_value=60, value=10, step=1)
    run_forecast_btn = st.button("Run Forecast for this School")

    if run_forecast_btn:
        with st.spinner(f"Forecasting {selected_school} / {selected_meal} for {k_steps} days..."):
            try:
                out, bt_true, bt_pred = forecast_future_dates(
                csv_path=CSV_PATH,
                date_col=DATE_COL,
                target_col=TARGET_COL,
                k_steps=k_steps,
                model_type="LSTM",
                model_path=MODEL_STUB,
                school_name=selected_school,
                meal_type=selected_meal,
                )
                st.success(f"Forecast produced: {len(out)} rows")
                st.subheader("Forecast table")
                st.dataframe(out)

                # Historical series for selected school+meal
                df_sm = df_series[(df_series['school_name'] == selected_school) & (df_series['meal_type'] == selected_meal)].copy()
                # combine historic & forecast for plotting
                hist_dates = pd.to_datetime(df_sm[DATE_COL])
                hist_vals = df_sm[TARGET_COL].values

                fdates = pd.to_datetime(out['forecast_date'])
                fvals = out[TARGET_COL].values

                fig_df = pd.DataFrame({
                    'date': np.concatenate([hist_dates.values, fdates.values]),
                    'value': np.concatenate([hist_vals, fvals]),
                    'type': ['hist'] * len(hist_dates) + ['forecast'] * len(fdates),
                })
                fig = px.line(fig_df, x='date', y='value', color='type', markers=True,
                              title=f"{selected_school} — {selected_meal}: Historical + Forecast")
                st.plotly_chart(fig, use_container_width=True)

                # show backtest (if available)
                if bt_true is not None and bt_pred is not None:
                    st.subheader("Backtest (last k-step true vs predicted)")
                    bt_df = pd.DataFrame({'true': bt_true, 'pred': bt_pred})
                    st.write(bt_df)
                    fig2 = px.scatter(bt_df, x='true', y='pred', title="Backtest: True vs Pred")
                    st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Forecast failed: {e}")

    st.markdown("---")
    st.header("2) Per-school Train/Test/Forecast plot")
    st.info("If you have a train/test/forecast image per school, upload it or use example.")
    uploaded_file = st.file_uploader("Upload train/test plot image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    use_uploaded = False
    if uploaded_file is not None:
        # display uploaded image immediately
        st.image(uploaded_file, caption="Uploaded Train/Test/Forecast", use_column_width=True)
        use_uploaded = True
    else:
        if os.path.exists(TRAIN_TEST_IMAGE):
            show_example = st.checkbox("Show example train/test/forecast image", value=True)
            if show_example:
                st.image(TRAIN_TEST_IMAGE, caption="Example Train/Test/Forecast", use_column_width=True)

with right_col:
    st.header("3) Top Production Cost Schools")
    top_n = st.slider("Top N schools to show (by mean production cost)", min_value=3, max_value=30, value=10)
    # compute per-school aggregated metric
    school_agg = df_series.groupby('school_name')[TARGET_COL].mean().sort_values(ascending=False).reset_index()
    top_schools = school_agg.head(top_n)
    fig = px.bar(top_schools, x='school_name', y=TARGET_COL, title=f"Top {top_n} Schools by Average Production Cost")
    st.plotly_chart(fig, use_container_width=True)

    st.write("Click a school to drill down (choose from dropdown):")
    drill_school = st.selectbox("Select school to drill down", top_schools['school_name'].tolist())
    drill_meal_types = sorted(df_series[df_series['school_name'] == drill_school]['meal_type'].unique())
    drill_meal = st.selectbox("Choose meal type for drill-down", drill_meal_types)
    if st.button("Show drill-down series"):
        df_drill = df_series[(df_series['school_name'] == drill_school) & (df_series['meal_type'] == drill_meal)]
        fig2 = px.line(df_drill, x=DATE_COL, y=TARGET_COL, title=f"{drill_school} ({drill_meal}) historical")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.header("4) Forecast total production cost across ALL schools for a selected future date")
    # choose a date to forecast total
    max_hist_date = df_series[DATE_COL].max()
    st.write(f"Latest historical date available: **{pd.to_datetime(max_hist_date).date()}**")
    pick_date = st.date_input("Select future date to estimate total production cost", value=(pd.to_datetime(max_hist_date) + pd.offsets.BDay(7)).date())
    k_days_ahead = business_days_between(max_hist_date, pick_date)

    if st.button("Compute Total Forecast for Selected Date"):
        if k_days_ahead == 0:
            st.warning("Selected date is not after last historical date — choose a later business date.")
        else:
            st.info(f"Computing aggregated forecasts for next {k_days_ahead} business day(s). This may take a while.")
            with st.spinner("Running per-school forecasts (aggregating)..."):
                df_all_forecasts, failed = compute_all_school_forecasts_for_horizon(df_series, k_steps=k_days_ahead, model_type="LSTM")
            if df_all_forecasts.empty:
                st.error("No per-school forecasts could be generated (maybe model files missing).")
            else:
                total_for_date = aggregate_forecasts_for_date(df_all_forecasts, pick_date)
                if math.isnan(total_for_date):
                    st.warning("No forecasts matched the selected date (maybe models skipped some schools).")
                else:
                    st.success(f"Estimated total production cost across all schools for {pick_date}: {total_for_date:,.2f}")
                st.write(f"[Info] Per-school forecast models failed / skipped: {failed}")

st.markdown("---")
st.header("5) Historical total production cost & aggregated forecasts")
daily_totals = agg_total_by_date(df_series)
st.line_chart(daily_totals.set_index(DATE_COL)[TARGET_COL])

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Produce aggregated K-day forecast (all schools) and plot")
    k_horizon = st.number_input("Horizon (business days) to forecast total across all schools", min_value=1, max_value=60, value=10, step=1)
    if st.button("Compute aggregated horizon forecast"):
        with st.spinner("Generating per-school forecasts and summing..."):
            df_all_forecasts, failed = compute_all_school_forecasts_for_horizon(df_series, k_steps=k_horizon, model_type="LSTM")
        if df_all_forecasts.empty:
            st.error("No forecasts produced (likely missing per-school model files).")
        else:
            # aggregate per unique forecast date
            agg = df_all_forecasts.groupby('forecast_date')[TARGET_COL].sum().reset_index().sort_values('forecast_date')
            st.subheader("Aggregated forecast (total across all schools)")
            st.dataframe(agg)
            fig3 = px.line(agg, x='forecast_date', y=TARGET_COL, title=f"Aggregated forecast for next {k_horizon} business days")
            st.plotly_chart(fig3, use_container_width=True)
            st.write(f"[Info] Some per-school forecasts failed/skipped: {failed}")

with col_b:
    st.subheader("Useful downloads & diagnostics")
    if st.button("Download historical totals (CSV)"):
        tmp = daily_totals.copy()
        tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL]).dt.date
        csv = tmp.to_csv(index=False)
        st.download_button("Download CSV", csv, file_name="historical_totals.csv", mime="text/csv")

st.sidebar.header("Suggestions & Next features you can add")
st.sidebar.markdown("""
- Add a **model selection** dropdown (LSTM/GRU/baseline) so users can compare models.  
- Show **error metrics** per-school (MAE, RMSE, R²) after backtesting.  
- Enable **batch model training** from the UI (with safety checks).  
- Add **seasonal decomposition** (STL) and show components per school.  
- Allow **calendar-style** selection and bulk forecast exports.  
- Provide **confidence intervals** using bootstrap or MC-dropout.  
""")

st.sidebar.markdown("---")
st.sidebar.write("App note: per-school forecasting requires saved per-school model files created by `training.py`. If many models are missing the aggregation steps will skip those schools (you'll see counts).")

st.markdown("### End of Dashboard")