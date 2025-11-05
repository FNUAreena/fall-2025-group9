#%%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from utils1 import seed_everything, TimeSeriesDataset, safe_time_split, load_and_aggregate_district
from model1 import ForecastingModel

CSV_PATH   = "C:/Users/B Varshith Reddy/OneDrive/Documents/GitHub/fall-2025-group9/src/component/Data/Output/meals_combined.csv"  # <-- change if needed
DATE_COL   = "date"
TARGET_COL = "production_cost_total"

WINDOW     = 7
ASPLIT     = 0.7         # first 70% train, last 30% test
MODEL_TYPE = "LSTM"
HIDDEN_DIM = 256
INPUT_DIM  = 1
NUM_LAYERS  = 4
OUTPUT_DIM = 1
DROPOUT    = 0.25
EPOCHS     = 100
BATCH_SIZE = 64
LR         = 0.001
SEED       = 42


seed_everything(SEED)
torch.manual_seed(SEED)

# ---- Load & aggregate district-by-date series
dates, values = load_and_aggregate_district(
    CSV_PATH=CSV_PATH,
    DATE_COL=DATE_COL,
    TARGET_COL=TARGET_COL,
    dayfirst=True
)
print(f"District series points: {len(values)}")

if len(values) <= WINDOW + 1:
    raise ValueError(f"Series too short ({len(values)}) for WINDOW={WINDOW}. Reduce WINDOW or use more data.")

# ---- Scale & split
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)            # shape (T,1)

split = safe_time_split(scaled, ASPLIT, WINDOW)  # ensures both sides have ≥ WINDOW+1
train = scaled[:split]
test  = scaled[split:]
print(f"Total={len(scaled)} | split={split} | train_len={len(train)} | test_len={len(test)} | WINDOW={WINDOW}")

# ---- Datasets / Loaders (professor style)
train_ds = TimeSeriesDataset(train, WINDOW)
test_ds  = TimeSeriesDataset(test,  WINDOW)
print(f"train_windows={len(train_ds)} | test_windows={len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ---- Model
model = ForecastingModel(MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT)
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# ---- Train loop (print epoch train/test MSE)
for epoch in tqdm(range(1, EPOCHS + 1), desc=f"Training {MODEL_TYPE} (univariate, pad+lengths)"):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.unsqueeze(-1)  # (batch, window, 1)
        yb = yb.unsqueeze(-1)  # (batch, 1)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item()

    print(f"Epoch {epoch:03d} | Train loss: {running/len(train_loader):.5f}")
    avg_train = running / max(1, len(train_loader))

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.unsqueeze(-1)
            yb = yb.unsqueeze(-1)
            p = model(xb)
            test_loss += criterion(p, yb).item()
    avg_test = test_loss / max(1, len(test_loader)) if len(test_loader) > 0 else float("nan")

    print(f"Epoch {epoch:03d}/{EPOCHS} | train_mse={avg_train:.6f} | test_mse={avg_test:.6f}")

# ---- Evaluate (use test windows; if none, fall back to train)
eval_loader = test_loader if len(test_ds) > 0 else train_loader
eval_scope  = "TEST" if len(test_ds) > 0 else "TRAIN"
scope_dates = dates[split:] if eval_scope == "TEST" else dates[:split]

model.eval()
preds_scaled, ys_scaled = [], []
with torch.no_grad():
    for xb, yb in eval_loader:
        xb = xb.unsqueeze(-1)
        yb = yb.unsqueeze(-1)
        p = model(xb)
        preds_scaled.append(p.numpy())
        ys_scaled.append(yb.numpy())

if len(preds_scaled) == 0:
    raise ValueError("No windows available for evaluation. Adjust WINDOW or split length.")

preds_scaled = np.vstack(preds_scaled)   # (N,1)
ys_scaled    = np.vstack(ys_scaled)      # (N,1)

y_pred = scaler.inverse_transform(preds_scaled)
y_true = scaler.inverse_transform(ys_scaled)

mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_true, y_pred)

print("\n=== RESULTS (District) ===")
print(f"Eval: {eval_scope}")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²  : {r2:.4f}")

# ---- Plot Pred vs Actual
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.5)
lims = [float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))]
plt.plot(lims, lims)
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost")
plt.title(f"Univariate LSTM (W={WINDOW}) • Pred vs Actual")
plt.grid(True)
plt.show()

# %%
