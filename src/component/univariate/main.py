#%%
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
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
    seed_everything, TimeSeriesDataset, safe_time_split, load_and_aggregate_district
)
from model import ForecastingModel

# ===================== CONFIG =====================
CSV_PATH   = r"C:\Users\B Varshith Reddy\OneDrive\Documents\GitHub\fall-2025-group9\src\component\Data\Output\meals_combined.csv"

# If these names are wrong/missing in the CSV, the loader will auto-detect when allow_autodetect=True
DATE_COL   = "date"
TARGET_COL = "production_cost_total"

WINDOW     = 7           # lookback steps
ASPLIT     = 0.7         # first 70% train, last 30% test
MODEL_TYPE = "LSTM"      # "GRU" or "LSTM"
HIDDEN_DIM = 256
INPUT_DIM  = 1
NUM_LAYERS = 4
OUTPUT_DIM = 1
DROPOUT    = 0.25

EPOCHS     = 100
BATCH_SIZE = 64
LR         = 0.001
SEED       = 42
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(SEED)
torch.manual_seed(SEED)

print("Using utils from:", getattr(utils, "__file__", "<unknown>"))
print("Using file:", os.path.abspath(CSV_PATH))

#  Load & aggregate data
dates, values, DATE_COL, TARGET_COL = load_and_aggregate_district(
    CSV_PATH=CSV_PATH,
    DATE_COL=DATE_COL,
    TARGET_COL=TARGET_COL,
    dayfirst="auto",
    debug=True,
)
print(f"DATE_COL={DATE_COL} | TARGET_COL={TARGET_COL}")
print(f"District series points (days): {len(values)}")

min_needed = 2 * (WINDOW + 1) + 1
if len(values) < min_needed:
    new_window = max(1, (len(values) - 3) // 2)
    print(f"[info] Not enough points for WINDOW={WINDOW}. Setting WINDOW -> {new_window}")
    globals()['WINDOW'] = new_window

# ---- Split BEFORE scaling; fit scaler on TRAIN ONLY (avoid leakage)

split = safe_time_split(values, ASPLIT, WINDOW)
train_raw = values[:split]
test_raw  = values[split:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_raw)      # (T_train, 1)
test_scaled  = scaler.transform(test_raw)           # (T_test, 1)

print(f"Total={len(values)} | split_idx={split} | train_len={len(train_raw)} | test_len={len(test_raw)} | WINDOW={WINDOW}")

# ---- Datasets / Loaders
train_ds = TimeSeriesDataset(train_scaled, WINDOW)
test_ds  = TimeSeriesDataset(test_scaled,  WINDOW)
print(f"train_windows={len(train_ds)} | test_windows={len(test_ds)}")

if len(train_ds) == 0:
    raise ValueError(f"No train windows for WINDOW={WINDOW}. Increase data or reduce WINDOW.")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# ---- Model
model = ForecastingModel(MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT)
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ---- Train loop
for epoch in tqdm(range(1, EPOCHS + 1), desc=f"Training {MODEL_TYPE}"):
    model.train()
    train_loss_sum = 0.0
    for xb, yb in train_loader:
        xb = xb.unsqueeze(-1).to(device)  # (batch, window, 1)
        yb = yb.unsqueeze(-1).to(device)  # (batch, 1)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()

    print(f"Epoch {epoch:03d} | Train loss: {train_loss_sum/len(train_loader):.5f}")
    avg_train = train_loss_sum / max(1, len(train_loader))


    model.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.unsqueeze(-1).to(device)
            yb = yb.unsqueeze(-1).to(device)
            p = model(xb)
            test_loss_sum += criterion(p, yb).item()
    avg_test = test_loss_sum / max(1, len(test_loader)) if len(test_ds) > 0 else float("nan")

    print(f"Epoch {epoch:03d}/{EPOCHS} | train_mse={avg_train:.6f} | test_mse={avg_test:.6f}")

# ---- Evaluate (use test windows; if none, fall back to train)
eval_loader = test_loader if len(test_ds) > 0 else train_loader
eval_scope  = "TEST" if len(test_ds) > 0 else "TRAIN"
scope_dates = dates[split:] if eval_scope == "TEST" else dates[:split]

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
    raise ValueError("No windows available for evaluation. Adjust WINDOW or split length.")

preds_scaled = np.vstack(preds_scaled)   # (N,1)
ys_scaled    = np.vstack(ys_scaled)      # (N,1)

# invert scaling with the TRAIN-fitted scaler
y_pred = scaler.inverse_transform(preds_scaled)
y_true = scaler.inverse_transform(ys_scaled)

mse  = mean_squared_error(y_true, y_pred)
rmse = float(np.sqrt(mse))
r2   = r2_score(y_true, y_pred)

print(f"\n=== RESULTS for {MODEL_TYPE} ===")
print(f"Eval: {eval_scope}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

# ---- Plot Actual vs Predicted on eval split
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6)

lo = float(min(np.min(y_true), np.min(y_pred)))
hi = float(max(np.max(y_true), np.max(y_pred)))
plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1.5, label='Ideal (y = x)')
plt.xlim(lo, hi); plt.ylim(lo, hi)
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost")
plt.title(f"{MODEL_TYPE} Predicted vs Actual")
plt.grid(True)
plt.show()

#%%

