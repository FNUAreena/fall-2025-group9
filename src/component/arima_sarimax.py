#%% 
# arima_sarimax.py
# Classical TS analysis for production_cost_total: ACF/PACF, ARIMA, SARIMAX
import warnings; warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import importlib, utils
importlib.reload(utils)
from utils import load_and_aggregate_district, safe_time_split

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ===================== CONFIG =====================
CSV_PATH   = r"C:\Users\B Varshith Reddy\OneDrive\Documents\GitHub\fall-2025-group9\src\component\Data\Output\meals_combined.csv"
DATE_COL   = "date"                   # your loader can also auto-detect if needed
TARGET_COL = "production_cost_total"

ASPLIT        = 0.8        # 80% train, 20% test
USE_LOG1P     = True       # stabilize variance: model log(1 + y)
FILL_MISSING  = "zero"     # 'zero' (school off days -> 0) or 'ffill'
SEASON_LENGTH = 7          # weekly seasonality is typical for school meals
MAX_PQ        = 3          # AR/MA orders search cap for ARIMA
MAX_PQ_SEAS   = 2          # seasonal AR/MA cap for SARIMAX
MAX_D         = 2          # maximum non-seasonal differencing tried
# ==================================================

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)

def enforce_daily_index(dates, values, fill=FILL_MISSING):
    """Ensure a contiguous daily index; fill gaps with zeros (or ffill)."""
    s = pd.Series(values.reshape(-1), index=pd.to_datetime(dates)).sort_index()
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    if fill == "zero":
        s = s.fillna(0.0)
    else:
        s = s.ffill().fillna(0.0)
    return s

def best_adfuller_d(series, max_d=MAX_D):
    """Pick smallest d in [0..max_d] such that ADF p-value < 0.05."""
    s = series.copy()
    for d in range(max_d + 1):
        pval = adfuller(s, autolag="AIC")[1]
        if pval < 0.05:
            return d
        s = s.diff().dropna()
    return max_d

def grid_search_arima(train, d, max_p=MAX_PQ, max_q=MAX_PQ):
    """AIC-based selection for ARIMA(p,d,q) (non-seasonal)."""
    best = None
    best_aic = np.inf
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = SARIMAX(train, order=(p, d, q), seasonal_order=(0,0,0,0), trend="n", enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best = (p, d, q, res)
                    best_aic = res.aic
            except Exception:
                continue
    return best  # (p,d,q,results)

def grid_search_sarimax(train, d, s=SEASON_LENGTH, max_p=MAX_PQ, max_q=MAX_PQ, max_P=MAX_PQ_SEAS, max_Q=MAX_PQ_SEAS):
    """AIC-based selection for SARIMAX(p,d,q)(P,D,Q,s) with seasonal D in {0,1}."""
    best = None
    best_aic = np.inf
    for P in range(0, max_P + 1):
        for Q in range(0, max_Q + 1):
            for D in (0, 1):
                for p in range(0, max_p + 1):
                    for q in range(0, max_q + 1):
                        try:
                            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s), trend="n",
                                            enforce_stationarity=False, enforce_invertibility=False)
                            res = model.fit(disp=False)
                            if res.aic < best_aic:
                                best = (p, d, q, P, D, Q, s, res)
                                best_aic = res.aic
                        except Exception:
                            continue
    return best  # (p,d,q,P,D,Q,s,results)

def main():
    print("Loading:", os.path.abspath(CSV_PATH))
    dates, values, DATE, TARGET = load_and_aggregate_district(
        CSV_PATH=CSV_PATH, DATE_COL=DATE_COL, TARGET_COL=TARGET_COL, dayfirst="auto", debug=True
    )
    print(f"DATE_COL={DATE} | TARGET_COL={TARGET} | points={len(values)}")

    # Make daily, fill gaps
    y = enforce_daily_index(dates, values, fill=FILL_MISSING)
    print("Daily points after reindex:", len(y))

    # Optional transform (log1p)
    y_work = np.log1p(y.clip(min=0.0)) if USE_LOG1P else y.astype(float)

    # Train/Test split (index-aware, no leakage)
    split_idx = int(len(y_work) * ASPLIT)
    train = y_work.iloc[:split_idx]
    test  = y_work.iloc[split_idx:]

    print(f"Train={len(train)} | Test={len(test)} | From {y.index.min().date()} to {y.index.max().date()}")

    # ---------------- ADF + ACF/PACF ----------------
    # Pick differencing d via ADF
    d = best_adfuller_d(train, MAX_D)
    print(f"Selected differencing order d={d} (ADF-based)")

    # Plot ACF/PACF of differenced train to inspect AR/MA cues
    diff_train = train.diff(d).dropna() if d > 0 else train
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    plot_acf(diff_train, ax=ax[0], lags=40, title="ACF (after differencing)")
    plot_pacf(diff_train, ax=ax[1], lags=40, method="ywm", title="PACF (after differencing)")
    plt.tight_layout(); plt.show()

    # ---------------- ARIMA (non-seasonal) ----------------
    print("\n[ARIMA] AIC grid search ...")
    best_arima = grid_search_arima(train, d=d, max_p=MAX_PQ, max_q=MAX_PQ)
    if best_arima is None:
        raise RuntimeError("ARIMA search failed to fit any model.")
    p,q = best_arima[0], best_arima[2]
    res_arima = best_arima[3]
    print(f"[ARIMA] Best order: (p,d,q)=({p},{d},{q}), AIC={res_arima.aic:.2f}")

    # Forecast on test horizon
    h = len(test)
    fc_arima = res_arima.get_forecast(steps=h).predicted_mean
    # Back-transform if log was used
    if USE_LOG1P:
        y_pred_arima = np.expm1(fc_arima.values)
        y_true_test  = y.iloc[split_idx:].values
    else:
        y_pred_arima = fc_arima.values
        y_true_test  = y.iloc[split_idx:].values

    # ---------------- SARIMAX (seasonal) ----------------
    print("\n[SARIMAX] AIC grid search ...")
    best_sarimax = grid_search_sarimax(train, d=d, s=SEASON_LENGTH, max_p=MAX_PQ, max_q=MAX_PQ,
                                       max_P=MAX_PQ_SEAS, max_Q=MAX_PQ_SEAS)
    if best_sarimax is None:
        raise RuntimeError("SARIMAX search failed to fit any model.")
    p2,_,q2,P,D,Q,s, res_sarimax = best_sarimax
    print(f"[SARIMAX] Best order: (p,d,q)=({p2},{d},{q2}), seasonal=({P},{D},{Q},{s}), AIC={res_sarimax.aic:.2f}")

    fc_sarimax = res_sarimax.get_forecast(steps=h).predicted_mean
    if USE_LOG1P:
        y_pred_sarimax = np.expm1(fc_sarimax.values)
    else:
        y_pred_sarimax = fc_sarimax.values

    # ---------------- Metrics ----------------
    def metrics(y_true, y_pred, name):
        mse  = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = mean_absolute_error(y_true, y_pred)
        mp   = mape(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        print(f"{name:8s} | MSE={mse:.2f} | RMSE={rmse:.2f} | MAE={mae:.2f} | MAPE%={mp:.2f} | R²={r2:.3f}")
        return mse, rmse, mae, mp, r2

    print("\n=== Test Metrics (original units) ===")
    _ = metrics(y_true_test, y_pred_arima,  "ARIMA")
    _ = metrics(y_true_test, y_pred_sarimax,"SARIMAX")

    # ---------------- Plots ----------------
    # 1) Forecast vs Actual over time (test window highlighted)
    plt.figure(figsize=(10,4))
    plt.plot(y.index, y.values, label="Actual")
    t_idx = y.index[split_idx:]
    plt.plot(t_idx, y_pred_arima,  label="ARIMA forecast")
    plt.plot(t_idx, y_pred_sarimax,label="SARIMAX forecast")
    plt.axvline(y.index[split_idx], color="k", ls="--", alpha=0.5)
    ttl = "Production Cost Total — Forecasts (test window)"
    if USE_LOG1P: ttl += " [log1p modeled, back-transformed]"
    plt.title(ttl)
    plt.xlabel("Date"); plt.ylabel(TARGET_COL)
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    # 2) Parity plots (test only)
    def parity(y_true, y_pred, title):
        yt = y_true.reshape(-1); yp = y_pred.reshape(-1)
        lo = 0.0
        hi = float(max(yt.max(), yp.max()))
        plt.figure(figsize=(6,6))
        plt.scatter(yt, yp, alpha=0.6, s=16)
        plt.plot([lo, hi], [lo, hi], "--", lw=1.5, label="Ideal (y=x)")
        plt.xlim(lo, hi); plt.ylim(lo, hi); plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title)
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

    parity(y_true_test, y_pred_arima,  "ARIMA — Predicted vs Actual (Test)")
    parity(y_true_test, y_pred_sarimax,"SARIMAX — Predicted vs Actual (Test)")

    # 3) Residual diagnostics (SARIMAX best)
    resid = y_true_test - y_pred_sarimax
    fig, ax = plt.subplots(1,3, figsize=(12,3))
    ax[0].plot(resid); ax[0].axhline(0, color="k", lw=1); ax[0].set_title("Residuals (SARIMAX)")
    plot_acf(resid, ax=ax[1], lags=40, title="Residual ACF")
    plot_pacf(resid, ax=ax[2], lags=40, method="ywm", title="Residual PACF")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()

# %%
