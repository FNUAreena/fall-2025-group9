#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from utils import preprocessing
from model import ForecastingModel
import warnings

warnings.filterwarnings("ignore")
torch.manual_seed(42)

df = pd.read_csv("Data/Output/meals_combined.csv")

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.sort_values('date')

preprocessed_data = preprocessing(df)

if "production_cost_total" in preprocessed_data.columns:
    preprocessed_data["production_cost_total"] = preprocessed_data["production_cost_total"].replace(
        {"\$": "", ",": ""}, regex=True
    ).astype(float)

features = [
    "meal_type",
    "served_total",
    "planned_total",
    "discarded_total",
    "left_over_total"
]
target = "production_cost_total"

# Scale Data
scaler = MinMaxScaler()
scaled_df = preprocessed_data[features + [target]].copy()
scaled_df[features + [target]] = scaler.fit_transform(scaled_df[features + [target]])

SEQ_LENGTH = 14  # look back 14 days

def create_sequences(data, target_col, seq_len=SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data.iloc[i:i+seq_len][features].values)
        y.append(data.iloc[i+seq_len][target_col])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_df, target, seq_len=SEQ_LENGTH)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                  torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)),
    batch_size=64, shuffle=False
)

MODEL_TYPE = "GRU"
INPUT_DIM = len(features)
HIDDEN_DIM = 128
NUM_LAYERS = 2
OUTPUT_DIM = 1
DROPOUT = 0.3
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10  # for early stopping

model = ForecastingModel(
    model_type=MODEL_TYPE,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    output_dim=OUTPUT_DIM,
    dropout=DROPOUT
)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

train_losses, test_losses = [], []
best_loss = float('inf')
early_stop_counter = 0

for epoch in tqdm(range(EPOCHS), desc=f"Training {MODEL_TYPE}"):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))


    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_losses[-1]:.6f}")

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
y_preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        preds = model(X_batch)
        y_preds.append(preds.detach().numpy())
y_preds = np.concatenate(y_preds)

# Inverse transform
inv_y = scaler.inverse_transform(
    np.hstack((np.zeros((len(y_test), len(features))), y_test.reshape(-1, 1)))
)[:, -1]
inv_preds = scaler.inverse_transform(
    np.hstack((np.zeros((len(y_preds), len(features))), y_preds))
)[:, -1]

mse = mean_squared_error(inv_y, inv_preds)
r2 = r2_score(inv_y, inv_preds)
print(f"\n✅ Final Results for {MODEL_TYPE}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(inv_y, inv_preds, alpha=0.6)
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title(f"{MODEL_TYPE} Predicted vs Actual Cost")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title(f"{MODEL_TYPE} Training vs Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.show()