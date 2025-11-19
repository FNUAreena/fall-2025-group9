# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import glob
import torch
import re
from forecasting import forecast_future_dates
from utils import _coerce_currency_column
import pickle
from model import ForecastingModel


CSV_PATH = "Data/Output/meals_combined.csv"
LSTM_MODELS_FOLDER = "univariate/LSTM_models"   # folder containing files like LSTM_Aldrin_Elementary_lunch.pth
GRU_MODELS_FOLDER  = "univariate/GRU_models"    # folder containing files like GRU_Aldrin_Elementary_lunch.pth
TEMP_SCHOOL_CSV = "temp_school.csv"


st.set_page_config(page_title="School Meals Forecasting Dashboard", layout="wide")

def clean_currency_series(col: pd.Series) -> pd.Series:
    def parse_cell(x):
        try:
            if pd.isna(x):
                return np.nan
            s = str(x).replace("$", "").replace(",", "").strip()
            return float(s)
        except:
            return np.nan
    return col.apply(parse_cell)

@st.cache_data(ttl=600)
def load_raw_data(path=CSV_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df_local = pd.read_csv(path)
    if "production_cost_total" in df_local.columns:
        df_local["production_cost_total"] = clean_currency_series(df_local["production_cost_total"])
    return df_local

def list_pth_files(folder: str):
    if not os.path.exists(folder):
        return []
    return sorted(glob.glob(os.path.join(folder, "*.pth")))

def parse_model_filename(fname: str):
    """
    Parse basename like 'GRU_Aldrin_Elementary_lunch.pth' -> (model_type, school_name_underscored, meal_type)
    Returns (None, None, None) if parsing fails.
    """
    base = os.path.basename(fname)
    if base.lower().endswith(".pth"):
        base = base[:-4]
    parts = base.split("_")
    if len(parts) < 3:
        return None, None, None
    model_type = parts[0].upper()
    meal_type = parts[-1]
    school = "_".join(parts[1:-1])
    return model_type, school, meal_type

def load_models_from_folder(folder_path: str):
    """Return list of dicts with path, model_type, school, meal_type (underscored school)."""
    out = []
    for p in list_pth_files(folder_path):
        mt, school, meal = parse_model_filename(os.path.basename(p))
        if mt is None:
            continue
        out.append({"path": p, "model_type": mt, "school": school, "meal_type": meal})
    return out

# Updated safe_forecast_single_model to accept school_name and meal_type and forward them to forecasting.forecast_future_dates
def safe_forecast_single_model(csv_path: str, model_type: str, model_path: str, k_steps: int = 10, school_name: str = None, meal_type: str = None) -> pd.DataFrame:
    """
    Calls forecasting.forecast_future_dates and returns a DataFrame.
    Passes through school_name and meal_type as keywords (forecasting may expect them).
    Returns empty DataFrame on error.
    """
    try:
        # forecasting.forecast_future_dates signature in your code supports these args (user indicated it expects school_name, meal_type)
        return forecast_future_dates(
            csv_path=csv_path,
            date_col="date",
            target_col="production_cost_total",
            window=7,
            asplit=0.7,
            k_steps=k_steps,
            model_type=model_type,
            model_path=model_path,
            hidden_dim=256,
            num_layers=4,
            dropout=0.25,
            school_name=school_name,
            meal_type=meal_type,
        )
    except TypeError:
        # If the user's forecast_future_dates doesn't accept school_name/meal_type kwargs, call without them:
        try:
            return forecast_future_dates(
                csv_path=csv_path,
                date_col="date",
                target_col="production_cost_total",
                window=7,
                asplit=0.7,
                k_steps=k_steps,
                model_type=model_type,
                model_path=model_path,
                hidden_dim=256,
                num_layers=4,
                dropout=0.25,
            )
        except Exception:
         return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# This is the updated function containing your provided updated loop:
def forecast_all_models_in_folder(df_all: pd.DataFrame, folder_path: str, k_steps: int = 10, progress_label: bool = False):
    """
    For each model file in folder_path:
      - parse school & meal from filename
      - convert school underscores -> spaces to match df_all['school_name']
      - subset df_all for that school
      - write temp csv and call safe_forecast_single_model(...) with school_name & meal_type
    Returns (merged_df, total_cost_sum, errors_list)
    """
    models = load_models_from_folder(folder_path)
    forecasts = []
    total_cost = 0.0
    errors = []

    if progress_label:
        prog = st.progress(0)

    for i, m in enumerate(models):


        filename = os.path.basename(m["path"])
        parts = filename.replace(".pth", "").split("_")

        model_type = parts[0]                      
        school_name = "_".join(parts[1:-1])       
        meal_type = parts[-1]                     

        
        school_name_csv = school_name.replace("_", " ").strip()

        if "school_name" in df_all.columns:
            subset = df_all[df_all["school_name"].astype(str).str.strip().str.lower() == school_name_csv.lower()].copy()
        else:
            subset = pd.DataFrame()

        if subset.empty:
            errors.append(
                f"No rows in CSV for school '{school_name_csv}' (filename {filename})"
            )
            if progress_label:
                prog.progress(int((i+1)/len(models) * 100))
            continue


        subset.to_csv(TEMP_SCHOOL_CSV, index=False)

        fc = safe_forecast_single_model(
            TEMP_SCHOOL_CSV,
            model_type,
            m["path"],
            k_steps=k_steps,
            school_name=school_name_csv,
            meal_type=meal_type
        )

        if fc.empty:
            errors.append(f"Forecast failed for {filename}")
            if progress_label:
                prog.progress(int((i+1)/len(models) * 100))
            continue


        fc["school_name"] = school_name_csv
        fc["meal_type"] = meal_type

        # Ensure consistent cost column
        if "production_cost_total" not in fc.columns and len(fc.columns) >= 3:
            fc = fc.rename(columns={fc.columns[-1]: "production_cost_total"})


        try:
            total_cost += fc["production_cost_total"].sum()
        except Exception:
            pass

        forecasts.append(fc)

        if progress_label:
            prog.progress(int((i+1)/len(models) * 100))

    if progress_label:
        try:
            prog.empty()
        except Exception:
            pass

    if forecasts:
        merged = pd.concat(forecasts, ignore_index=True)
    else:
        merged = pd.DataFrame()

    return merged, total_cost, errors

def school_loss_analysis(df_local):
    dfc = df_local.copy()
    dfc["loss_reason"] = ""
    dfc["discarded_total"] = dfc.get("discarded_total", 0).fillna(0)
    dfc["left_over_total"] = dfc.get("left_over_total", 0).fillna(0)
    dfc["planned_total"] = dfc.get("planned_total", 0).fillna(0)
    dfc["served_total"] = dfc.get("served_total", 0).fillna(0)

    dfc["loss_amount"] = dfc["discarded_total"] + dfc["left_over_total"]
    dfc["loss_pct"] = (dfc["loss_amount"] / dfc["planned_total"].replace(0, np.nan)) * 100

    dfc.loc[dfc["loss_pct"] > 20, "loss_reason"] = "High food wastage"
    dfc.loc[dfc["served_total"] < dfc["planned_total"] * 0.6, "loss_reason"] = "Low turnout"
    dfc.loc[dfc["left_over_total"] > dfc["served_total"], "loss_reason"] = "Over-production"

    losing = dfc[dfc["loss_reason"] != ""]
    return losing

try:
    df = load_raw_data(CSV_PATH)
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

loss_df = school_loss_analysis(df)

st.sidebar.title("⚙ Controls")
model_choice = st.sidebar.selectbox(
    "Select Forecast Model Folder",
    ["LSTM_models (all schools)", "GRU_models (all schools)"]
)
forecast_days = st.sidebar.slider("Forecast Next Business Days", 3, 30, 10)
run_forecast_btn = st.sidebar.button("Run Forecast")
compare_btn = st.sidebar.button("Compare LSTM vs GRU")
show_ai_btn = st.sidebar.button("AI Recommendations")

st.title("📊 School Meals Forecasting Dashboard")
st.markdown("Predict production cost, identify loss-making schools, and compare total cost before & after forecasting.")

forecast_df = pd.DataFrame()
forecast_agg_by_date = pd.DataFrame()
forecast_summary_total = 0.0

if run_forecast_btn:
    folder = LSTM_MODELS_FOLDER if "LSTM" in model_choice else GRU_MODELS_FOLDER
    if not os.path.exists(folder):
        st.error(f"Models folder not found: {folder}")
    else:
        with st.spinner("Running folder forecasts..."):
            merged, total_cost, errors = forecast_all_models_in_folder(df, folder, k_steps=forecast_days, progress_label=True)
            forecast_df = merged
            forecast_summary_total = total_cost

        if not merged.empty:
            st.success(f"Forecast completed across {len(load_models_from_folder(folder))} models (folder={folder}).")
            # show merged head
            st.dataframe(merged.head(200))

            # AGGREGATE: sum production_cost_total across all schools & meal types for each forecast_date
            try:
                merged["_parsed_date"] = pd.to_datetime(merged["forecast_date"])
                agg_by_date = merged.groupby("_parsed_date", observed=True)["production_cost_total"].sum().reset_index().rename(columns={"_parsed_date": "forecast_date", "production_cost_total": "total_production_cost"})
                agg_by_date = agg_by_date.sort_values("forecast_date")
                forecast_agg_by_date = agg_by_date
                st.subheader("📅 Total Forecasted Production Cost (summed across all schools & meals)")
                st.dataframe(agg_by_date)
                fig_total = px.line(agg_by_date, x="forecast_date", y="total_production_cost", title="Total Forecasted Production Cost by Date")
                st.plotly_chart(fig_total, use_container_width=True)
                st.metric("📦 TOTAL Forecasted Production Cost (sum over horizon)", f"${forecast_summary_total:,.2f}")
            except Exception as e:
                st.warning(f"Failed to aggregate forecasts by date: {e}")

            # Per-school totals
            school_totals = merged.groupby("school_name", observed=True)["production_cost_total"].sum().reset_index().sort_values(by="production_cost_total", ascending=False)
            st.subheader("🏫 Forecasted Cost by School (sum over forecast horizon)")
            st.dataframe(school_totals)
            fig_school = px.bar(school_totals, x="school_name", y="production_cost_total", title="Forecasted Cost by School")
            st.plotly_chart(fig_school, use_container_width=True)

            # Per-school & per-meal viewer
            st.subheader("🔎 View Forecast Per School / Meal Type")
            school_list = ["ALL"] + sorted(merged["school_name"].unique().tolist())
            selected_school = st.selectbox("Select School", school_list, index=0)
            if selected_school == "ALL":
                view_df = merged
            else:
                view_df = merged[merged["school_name"] == selected_school]

            meal_list = ["ALL"] + sorted(view_df["meal_type"].unique().tolist())
            selected_meal = st.selectbox("Select Meal Type", meal_list, index=0)
            if selected_meal != "ALL":
                view_df = view_df[view_df["meal_type"] == selected_meal]

        else:
            st.error("No forecasts generated from folder models. See errors below.")
            for e in errors:
                st.write(f"- {e}")
    

if compare_btn:
    st.header("⚔ LSTM_models vs GRU_models Comparison")
    if not os.path.exists(LSTM_MODELS_FOLDER):
        st.warning(f"LSTM folder not found: {LSTM_MODELS_FOLDER}")
    if not os.path.exists(GRU_MODELS_FOLDER):
        st.warning(f"GRU folder not found: {GRU_MODELS_FOLDER}")

    with st.spinner("Running LSTM folder forecasts..."):
        lstm_df, lstm_total, lstm_errors = forecast_all_models_in_folder(df, LSTM_MODELS_FOLDER, k_steps=forecast_days, progress_label=False)
    with st.spinner("Running GRU folder forecasts..."):
        gru_df, gru_total, gru_errors = forecast_all_models_in_folder(df, GRU_MODELS_FOLDER, k_steps=forecast_days, progress_label=False)

    col1, col2 = st.columns(2)
    col1.metric("LSTM_models Total Forecast", f"${lstm_total:,.2f}")
    col2.metric("GRU_models Total Forecast", f"${gru_total:,.2f}", delta=f"${gru_total - lstm_total:,.2f}")

    if not lstm_df.empty and not gru_df.empty:
        lstm_df["ModelKind"] = "LSTM"
        gru_df["ModelKind"] = "GRU"
        combined_cmp = pd.concat([lstm_df, gru_df], ignore_index=True)

   
        try:
            lstm_df["_pdate"] = pd.to_datetime(lstm_df["forecast_date"])
            gru_df["_pdate"] = pd.to_datetime(gru_df["forecast_date"])
            lstm_agg = lstm_df.groupby("_pdate", observed=True)["production_cost_total"].sum().reset_index().rename(columns={"_pdate":"forecast_date","production_cost_total":"total_cost"})
            gru_agg  = gru_df.groupby("_pdate", observed=True)["production_cost_total"].sum().reset_index().rename(columns={"_pdate":"forecast_date","production_cost_total":"total_cost"})
            lstm_agg["Model"] = "LSTM"
            gru_agg["Model"]  = "GRU"
            cmp_df = pd.concat([lstm_agg, gru_agg], ignore_index=True)
            fig_cmp = px.line(cmp_df, x="forecast_date", y="total_cost", color="Model", title="LSTM vs GRU: Total Forecasted Cost by Date")
            st.plotly_chart(fig_cmp, use_container_width=True)
        except Exception:
            
            fig_cmp = px.line(combined_cmp, x="forecast_date", y="production_cost_total", color="ModelKind", title="LSTM vs GRU (raw)")
            st.plotly_chart(fig_cmp, use_container_width=True)

        
        lstm_school_tot = lstm_df.groupby("school_name", observed=True)["production_cost_total"].sum().reset_index().rename(columns={"production_cost_total":"LSTM_total"})
        gru_school_tot  = gru_df.groupby("school_name", observed=True)["production_cost_total"].sum().reset_index().rename(columns={"production_cost_total":"GRU_total"})
        merged_tot = pd.merge(lstm_school_tot, gru_school_tot, on="school_name", how="outer").fillna(0)
        merged_tot["Difference (GRU - LSTM)"] = merged_tot["GRU_total"] - merged_tot["LSTM_total"]
        st.subheader("Per-School Totals & Differences (GRU - LSTM)")
        st.dataframe(merged_tot.sort_values("Difference (GRU - LSTM)", ascending=False))
        fig_diff = px.bar(merged_tot, x="school_name", y="Difference (GRU - LSTM)", title="Per-School Forecast Difference (GRU - LSTM)")
        st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.info("Not enough data to compare both sets. Check CSV school_name values and folder files.")

st.header("🏫 School Loss Analysis")
if loss_df.empty:
    st.success("No schools are losing money ❗")
else:
    st.warning("Some schools are losing money")
    st.dataframe(loss_df[["school_name", "meal_type", "loss_amount", "loss_pct", "loss_reason"]])
    fig_loss = px.bar(loss_df, x="school_name", y="loss_amount", color="loss_reason", title="Schools Losing Money & Reasons")
    st.plotly_chart(fig_loss, use_container_width=True)

st.header("🔥 Wastage Heatmap (School × Weekday)")
if "date" in df.columns:
    df["_parsed_date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["weekday"] = df["_parsed_date"].dt.day_name()
else:
    df["weekday"] = np.nan

heat_df = df.dropna(subset=["weekday"])
if heat_df.empty:
    st.info("No valid dates available for heatmap.")
else:
    heatmap_df = heat_df.pivot_table(index="school_name", columns="weekday", values="discarded_total", aggfunc="sum").fillna(0)
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    cols_present = [d for d in weekday_order if d in heatmap_df.columns]
    heatmap_df = heatmap_df[cols_present]
    fig_heat = px.imshow(heatmap_df, title="Heatmap: Food Wastage Across Schools by Weekday")
    st.plotly_chart(fig_heat, use_container_width=True)

st.header("🥧 Production Cost Breakdown")
if len(df) > 0 and pd.notna(df.iloc[-1].get("production_cost_total", np.nan)):
    latest = df.iloc[-1]["production_cost_total"]
    breakdown = {"Ingredients": latest * 0.55, "Labor": latest * 0.30, "Overhead": latest * 0.15}
    fig_break = px.pie(names=list(breakdown.keys()), values=list(breakdown.values()), title="Cost Breakdown of Latest Meal Production")
    st.plotly_chart(fig_break, use_container_width=True)
else:
    st.info("No production cost value found for latest row.")

st.header("🤖 AI Recommendations")
if "show_ai" not in st.session_state:
    st.session_state.show_ai = False

if show_ai_btn:
    st.session_state.show_ai = not st.session_state.show_ai

if st.session_state.show_ai:
    st.success("Generating AI Recommendations...")
    high_waste = loss_df[loss_df["loss_pct"] > 15]
    if high_waste.empty:
        st.info("No high-waste schools identified.")
    else:
        for _, row in high_waste.iterrows():
            st.markdown(f"### {row['school_name']}")
            st.write(f"**Reason:** {row['loss_reason']}")
            st.write("- Adjust planned meals downward by 10–25%")
            st.write(f"- Improve demand estimation for **{row.get('meal_type', 'N/A')}**")
            st.write("- Monitor leftover trend weekly")
else:
    st.info("Click 'AI Recommendations' in the sidebar to toggle suggestions.")

st.markdown("---")
st.markdown("#### Built School Meal Forecasting Systems")