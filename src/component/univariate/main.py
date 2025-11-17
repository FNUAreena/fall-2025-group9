#%%
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import importlib, utils
importlib.reload(utils)
from utils import (
    seed_everything,
    TimeSeriesDataset,
    safe_time_split,
    load_and_aggregate_district,
)
from model import ForecastingModel
from forecasting import forecast_future_dates

# Paths and column names
CSV_PATH   = "Data/Output/meals_combined.csv"
DATE_COL   = "date"
TARGET_COL = "production_cost_total"

# Model and training hyperparameters
WINDOW     = 3
ASPLIT     = 0.6
MODEL_TYPE = "LSTM"
HIDDEN_DIM = 256
INPUT_DIM  = 1
OUTPUT_DIM = 1
NUM_LAYERS = 4
DROPOUT    = 0.25

EPOCHS     = 100
BATCH_SIZE = 64
LR         = 0.001
SEED       = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(SEED)
torch.manual_seed(SEED)

print("Using utils from:", getattr(utils, "__file__", "<unknown>"))
print("Using file:", os.path.abspath(CSV_PATH))

MODEL_DIR   = os.path.join("univariate", f"{MODEL_TYPE}_models")
RESULTS_DIR = os.path.join("univariate", "results")          # keep common results folder
PLOTS_DIR   = os.path.join("univariate", "plots")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load aggregated data: (school_name, meal_type, date) -> cost
# ----------------------------------------------------------------------

dates, values, DATE_COL, TARGET_COL = load_and_aggregate_district(
    CSV_PATH=CSV_PATH,
    DATE_COL=DATE_COL,
    TARGET_COL=TARGET_COL,
    dayfirst="auto",
    debug=True,
)
print(f"DATE_COL={DATE_COL} | TARGET_COL={TARGET_COL}")
print(f"Total rows after aggregation (school, meal, date): {len(values)}")

# Rebuild a tidy DataFrame from the multi-index style arrays
records = []
for (school, meal, dt), v in zip(dates, values.reshape(-1)):
    records.append((school, meal, pd.to_datetime(dt), float(v)))

df_series = pd.DataFrame(records, columns=["school_name", "meal_type", DATE_COL, TARGET_COL])
df_series = df_series.sort_values(["school_name", "meal_type", DATE_COL]).reset_index(drop=True)

print(f"Unique schools: {df_series['school_name'].nunique()} | meal_types: {df_series['meal_type'].unique()}")
grouped = df_series.groupby(["school_name", "meal_type"], sort=True)

# Minimum points needed so both train and test have at least one window
min_needed = 2 * (WINDOW + 1) + 1
# ----------------------------------------------------------------------
# 2. Train a separate univariate LSTM for each (school, meal) series
# ----------------------------------------------------------------------

all_true = []
all_pred = []

for (school, meal), g in grouped:
    series_values = g[TARGET_COL].values.astype("float32").reshape(-1, 1)
    n_points = len(series_values)
    print("\n" + "=" * 80)
    print(f"Training model for school={school!r}, meal_type={meal!r} | points={n_points}")

    if n_points < min_needed:
        print(f"[skip] Too few points ({n_points}) for WINDOW={WINDOW}. Need at least {min_needed}.")
        continue

    # Time-based split with safety for window size
    try:
        split_idx = safe_time_split(series_values, ASPLIT, WINDOW)
    except ValueError as e:
        print(f"[skip] {school!r}/{meal!r} due to split error: {e}")
        continue

    train_raw = series_values[:split_idx]
    test_raw  = series_values[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled  = scaler.transform(test_raw)

    print(
        f"Total={n_points} | split_idx={split_idx} | train_len={len(train_raw)} | "
        f"test_len={len(test_raw)} | WINDOW={WINDOW}"
    )

    train_ds = TimeSeriesDataset(train_scaled, WINDOW)
    test_ds  = TimeSeriesDataset(test_scaled,  WINDOW)
    print(f"train_windows={len(train_ds)} | test_windows={len(test_ds)}")

    if len(train_ds) == 0:
        print(f"[skip] No train windows for WINDOW={WINDOW} in {school!r}/{meal!r}.")
        continue

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = ForecastingModel(
        MODEL_TYPE,
        INPUT_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        OUTPUT_DIM,
        DROPOUT,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # -------------------- training loop --------------------
    for epoch in tqdm(
        range(1, EPOCHS + 1),
        desc=f"Training {MODEL_TYPE} for {school} / {meal}",
        leave=False,
    ):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.unsqueeze(-1).to(device)   # (batch, window) -> (batch, window, 1)
            yb = yb.unsqueeze(-1).to(device)   # (batch,)       -> (batch, 1)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train = train_loss_sum / max(1, len(train_loader))

        # simple test loss each epoch (optional)
        model.eval()
        test_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.unsqueeze(-1).to(device)
                yb = yb.unsqueeze(-1).to(device)
                p = model(xb)
                test_loss_sum += criterion(p, yb).item()

        avg_test = test_loss_sum / max(1, len(test_loader)) if len(test_ds) > 0 else float("nan")
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | train_mse={avg_train:.6f} | "
                f"test_mse={avg_test:.6f}"
            )

    # -------------------- evaluation on final model --------------------
    eval_loader = test_loader if len(test_ds) > 0 else train_loader
    eval_scope  = "TEST" if len(test_ds) > 0 else "TRAIN"

    model.eval()
    preds_scaled, ys_scaled = [], []
    with torch.no_grad():
        for xb, yb in eval_loader:
            xb = xb.unsqueeze(-1).to(device)
            yb = yb.unsqueeze(-1).to(device)
            p = model(xb)
            preds_scaled.append(p.detach().cpu().numpy())
            ys_scaled.append(yb.detach().cpu().numpy())

    if len(preds_scaled) == 0:
        print(f"[warn] No windows available for evaluation in {school!r}/{meal!r}.")
        continue

    preds_scaled = np.vstack(preds_scaled)
    ys_scaled    = np.vstack(ys_scaled)

    y_pred = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(ys_scaled)

    all_true.append(y_true.reshape(-1, 1))
    all_pred.append(y_pred.reshape(-1, 1))

    # -------------------- save model for this school+meal --------------------
    school_safe = str(school).replace(" ", "_").replace("/", "_")
    meal_safe   = str(meal).replace(" ", "_").replace("/", "_")
    model_path  = os.path.join(
            MODEL_DIR,
            f"{MODEL_TYPE}_{school_safe}_{meal_safe}.pth",
    )
    torch.save(model.state_dict(), model_path)
    print(f"[saved] {model_path}")


if len(all_true) > 0:
    all_true_arr = np.vstack(all_true)
    all_pred_arr = np.vstack(all_pred)

    # ---- overall metrics across all schools and meals ----
    overall_mse  = mean_squared_error(all_true_arr.ravel(), all_pred_arr.ravel())
    overall_rmse = float(np.sqrt(overall_mse))
    overall_r2   = r2_score(all_true_arr.ravel(), all_pred_arr.ravel())

    print("\n=== OVERALL RESULTS for LSTM (all schools & meals) ===")
    print(f"MSE : {overall_mse:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"R^2 : {overall_r2:.4f}")

    plt.figure(figsize=(6,6))
    plt.scatter(all_true_arr, all_pred_arr, alpha=0.6)

    lo = float(min(np.min(all_true_arr), np.min(all_pred_arr)))
    hi = float(max(np.max(all_true_arr), np.max(all_pred_arr)))
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5, label='Ideal (y = x)')
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Actual Production Cost")
    plt.ylabel("Predicted Production Cost")
    plt.title(f"{MODEL_TYPE} Predicted vs Actual")
    plt.grid(True)
    plt.savefig(f"univariate/plots/{MODEL_TYPE}.png")
    plt.close()

print("\nAll school / meal models trained.\n")

# Forecast future dates for each school+meal")

os.makedirs("univariate/results", exist_ok=True)

all_forecasts = []

unique_combinations = df_series[['school_name', 'meal_type']].drop_duplicates()
print(f"Total unique school+meal combinations to forecast: {len(unique_combinations)}")

for _, row in unique_combinations.iterrows():
    school = row['school_name']
    meal = row['meal_type']

    print(f"Forecasting for school={school!r}, meal_type={meal!r}...")
    try:
        df_forecast = forecast_future_dates(
            csv_path=CSV_PATH,
            date_col=DATE_COL,
            target_col=TARGET_COL,
            window=WINDOW,
            asplit=ASPLIT,
            k_steps=10,  # Forecast 10 days ahead
            model_type=MODEL_TYPE,
            model_path=f"univariate/LSTM_models/{MODEL_TYPE}.pth",  # Base path
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            school_name=school,
            meal_type=meal,
        )
        all_forecasts.append(df_forecast)
    except Exception as e:
        print(f"[error] Forecasting failed for school={school!r}, meal_type={meal!r}: {e}")
        continue


# combine and save all forecasts
if len(all_forecasts) > 0:
    df_all_forecast = pd.concat(all_forecasts, ignore_index=True)

    print("\n=== COMBINED 10-DAY FORECAST FOR ALL SCHOOLS & MEALS (head) ===")
    print(df_all_forecast.head(20))

    output_path = "univariate/results/all_school_meal_forecasts.csv"
    df_all_forecast.to_csv(output_path, index=False)
    print(f"\n[saved] Combined 10-day forecasts for all schools/meals -> {output_path}")
else:
    print("\n[warn] No forecasts were generated. Check training/forecast logs.")