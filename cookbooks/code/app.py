# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import BDay
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart School Food Service Forecasting",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------
# Sidebar controls
# -----------------------------------------------
st.sidebar.title("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload FCPS data (CSV)", type=['csv'])
n_forecast = st.sidebar.slider("Forecast Horizon (Business Days)", 5, 30, 10)
show_arima = st.sidebar.checkbox("Run ARIMA Model", value=True)
show_sarimax = st.sidebar.checkbox("Run SARIMAX Model", value=True)

# -----------------------------------------------
# Step 1: Load data
# -----------------------------------------------
st.title("📊 Smart School Food Service Analytics Dashboard")
st.markdown("AI-Driven Demand Forecasting and Waste Reduction using FCPS Data")

if uploaded_file is not None:
    df = pd.read_csv("/Users/chayachandana/Downloads/dataset.csv")
    st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # -----------------------------------------------
    # Step 2: Preprocess and create time series
    # -----------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Served_Total'])
    df['Served_Total'] = pd.to_numeric(df['Served_Total'], errors='coerce')

    time_series_data = (
        df.groupby('Date')['Served_Total']
        .sum()
        .sort_index()
        .astype(float)
        .asfreq('B')
        .interpolate(limit_direction='both')
    )

    st.subheader("📈 Time Series Overview")
    st.write(f"Data Range: **{time_series_data.index.min().date()} → {time_series_data.index.max().date()}**")
    st.line_chart(time_series_data, use_container_width=True)

    # -----------------------------------------------
    # Step 3: Train-Test Split
    # -----------------------------------------------
    n_test = max(5, int(len(time_series_data) * 0.25))
    y_train = time_series_data[:-n_test]
    y_test = time_series_data[-n_test:]

    # -----------------------------------------------
    # Step 4: ARIMA Forecast
    # -----------------------------------------------
    if show_arima:
        st.subheader("🔮 ARIMA Forecasting")

        candidates = [(p, d, q) for d in [0, 1, 2] for p in [0, 1] for q in [0, 1]]
        best_aic = np.inf
        best_model, best_order = None, None

        for order in candidates:
            try:
                model = ARIMA(y_train, order=order)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_model = model_fit
            except:
                continue

        if best_model:
            y_pred = best_model.forecast(steps=len(y_test))
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = (np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))).mean() * 100

            st.write(f"**Best ARIMA Order:** {best_order}  |  AIC={best_aic:.2f}")
            st.write(f"**MAE:** {mae:.2f}  |  **RMSE:** {rmse:.2f}  |  **MAPE:** {mape:.2f}%")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y_train.index, y_train, label='Train')
            ax.plot(y_test.index, y_test, label='Test (actual)')
            ax.plot(y_test.index, y_pred, '--', label=f'ARIMA{best_order} Forecast')
            ax.legend(); ax.grid(alpha=0.3)
            st.pyplot(fig)

            # Future forecast
            final_model = ARIMA(time_series_data, order=best_order).fit()
            future_forecast = final_model.forecast(steps=n_forecast)
            future_df = pd.DataFrame({
                'Date': future_forecast.index,
                'Forecast_Served': future_forecast.values
            })
            st.write("📅 **Next Forecasted Days (ARIMA):**")
            st.dataframe(future_df.round(1))

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(time_series_data.index, time_series_data, label='History')
            ax2.plot(future_forecast.index, future_forecast, 'o--', color='red', label='Forecast')
            ax2.legend(); ax2.grid(alpha=0.3)
            st.pyplot(fig2)
        else:
            st.warning("No ARIMA model could be fit. Try checking data range or structure.")

    # -----------------------------------------------
    # Step 5: SARIMAX Forecast
    # -----------------------------------------------
    if show_sarimax:
        st.subheader("🔮 SARIMAX Forecasting (with trend & lag)")

        exog = pd.DataFrame(index=time_series_data.index)
        exog['lag5'] = time_series_data.shift(5)
        exog['trend'] = np.arange(len(exog))
        exog = exog.fillna(0.0)

        X_train, X_test = exog.loc[y_train.index], exog.loc[y_test.index]

        pdq_combinations = [(1, 1, 0), (0, 1, 1), (1, 1, 1)]
        seasonal_combinations = [(0, 0, 0, 5), (1, 0, 1, 5)]

        best_aic = np.inf
        best_pdq, best_seasonal, best_model = None, None, None

        for pdq in pdq_combinations:
            for seasonal in seasonal_combinations:
                try:
                    model = SARIMAX(y_train, exog=X_train, order=pdq, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
                    fit = model.fit(disp=False)
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_pdq, best_seasonal = pdq, seasonal
                        best_model = fit
                except:
                    continue

        if best_model:
            y_pred = best_model.predict(start=y_test.index[0], end=y_test.index[-1], exog=X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = (np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))).mean() * 100

            st.write(f"**Best SARIMAX Order:** {best_pdq} x {best_seasonal}  |  AIC={best_aic:.2f}")
            st.write(f"**MAE:** {mae:.2f}  |  **RMSE:** {rmse:.2f}  |  **MAPE:** {mape:.2f}%")

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.plot(y_train.index, y_train, label='Train')
            ax3.plot(y_test.index, y_test, label='Test (actual)')
            ax3.plot(y_test.index, y_pred, '--', label=f'SARIMAX{best_pdq}x{best_seasonal}')
            ax3.legend(); ax3.grid(alpha=0.3)
            st.pyplot(fig3)

            # Future forecast
            all_index = time_series_data.index.append(pd.bdate_range(time_series_data.index[-1] + BDay(1), periods=n_forecast))
            tmp = pd.Series(time_series_data, index=all_index)
            exog_future = pd.DataFrame(index=all_index)
            exog_future['lag5'] = tmp.shift(5)
            exog_future['trend'] = np.arange(len(exog_future))
            exog_future = exog_future.fillna(0.0)
            X_future = exog_future.loc[all_index[-n_forecast:]]

            future_res = best_model.get_forecast(steps=n_forecast, exog=X_future)
            y_fcst = future_res.predicted_mean
            ci = future_res.conf_int(alpha=0.2)

            st.write("📅 **Next Forecasted Days (SARIMAX):**")
            st.dataframe(pd.DataFrame({
                'Date': y_fcst.index,
                'Forecast': y_fcst,
                'Lower80': ci.iloc[:, 0],
                'Upper80': ci.iloc[:, 1]
            }).round(1))

            fig4, ax4 = plt.subplots(figsize=(10, 4))
            ax4.plot(time_series_data.index, time_series_data, label='Actual')
            ax4.plot(y_fcst.index, y_fcst, '--', color='red', label='Forecast')
            ax4.fill_between(y_fcst.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2)
            ax4.legend(); ax4.grid(alpha=0.3)
            st.pyplot(fig4)
        else:
            st.warning("No SARIMAX model could be fit. Try different parameter combinations.")

else:
    st.info("👆 Upload your FCPS dataset to begin forecasting.")

st.markdown("---")
st.markdown("**Developed by Chandana Gowda — Smart School Food Service Analytics** 🍎")