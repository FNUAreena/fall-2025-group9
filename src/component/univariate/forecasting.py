import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from utils import seed_everything, TimeSeriesDataset, load_and_aggregate_district, safe_time_split
from model import ForecastingModel


# Hyperparameters
CSV_PATH = "Data/Output/meals_combined.csv"
DATE_COL = "date"
TARGET_COL= "production_cost_total"

MODEL_TYPE = "LSTM"
HIDDEN_DIM = 256   
NUM_LAYERS = 4  
DROPOUT= 0.25 
WINDOW = 7 
ASPLIT = 0.7
K_STEPS = 10

MODEL_PATH = "univariate/results/LSTM.pth"

def next_days(last_date: pd.Timestamp, k: int) -> pd.DatetimeIndex:
    """Return the next k business days after last_date."""
    start = pd.to_datetime(last_date) + pd.offsets.BDay(1)
    return pd.bdate_range(start, periods=k)

def last_date_from_dates_array(dates_array) -> pd.Timestamp:
    """Get the last date from a numpy array of dates."""
    last_item = dates_array[-1]
    if isinstance(last_item, tuple) and len(last_item) >= 3:
        return pd.to_datetime(last_item[2])
    return pd.to_datetime(last_item)

def forecast_future_dates(
        csv_path: str = CSV_PATH,
        date_col: str = DATE_COL,
        target_col: str = TARGET_COL,
        window: int = WINDOW,
        asplit: float = ASPLIT,
        k_steps: int = K_STEPS,
        model_type: str = MODEL_TYPE,
        model_path: str = MODEL_PATH,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
) -> pd.DataFrame:
    
    """Forecast the next k_steps future dates using a trained model."""
    # Load and aggregate data
    dates, values, _, _ = load_and_aggregate_district(
        CSV_PATH=csv_path,
        DATE_COL=date_col,
        TARGET_COL=target_col,
        dayfirst="auto",
        debug=False,
    )

    if len(values) < window + 1:
        raise ValueError(f"Series too short (n={len(values)}) for WINDOW={window}.")

    # Recompute split and fit scaler on train only (exactly like main.py)
    split_idx = safe_time_split(values, asplit, window)
    train_raw = values[:split_idx]  # shape (T_train, 1)

    scaler = MinMaxScaler()
    scaler.fit(train_raw)
    full_scaled = scaler.transform(values).ravel()

    # Load trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastingModel(
        model_type=model_type,
        input_dim=1,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Forecast next K steps in scaled space, then inverse-transform
    last = full_scaled[-window:].astype(np.float32).copy()
    preds_scaled = []
    with torch.no_grad():
        for _ in range(k_steps):
            xb = torch.from_numpy(last).view(1, window, 1).to(device)
            yhat = model(xb).detach().cpu().numpy().reshape(-1)[0]
            preds_scaled.append(yhat)
            last = np.roll(last, -1)
            last[-1] = yhat

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()

    # Use the calendar date of the last observed tuple to generate future business dates
    last_date = last_date_from_dates_array(dates)
    fdates = next_days(last_date, k_steps)

    out = pd.DataFrame(
        {
            "forecast_date": fdates,
            "step_ahead": np.arange(1, k_steps + 1),
            target_col: preds,
        }
    )
    print(f"forecasting: {len(out)} rows")
    return out


if __name__ == "__main__":
    df = forecast_future_dates()
    print(df.head())
