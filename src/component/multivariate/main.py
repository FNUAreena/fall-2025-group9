#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from utils import preprocessing
from model import ForecastingModel, FeedForwardNN
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import warnings
import pickle

warnings.filterwarnings("ignore")
torch.manual_seed(42)

df = pd.read_csv("Data/Output/meals_combined.csv")

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df = df.sort_values(['school_name', 'meal_type', 'date'])

preprocessed_data = preprocessing(df)

if "production_cost_total" in preprocessed_data.columns:
    preprocessed_data["production_cost_total"] = preprocessed_data["production_cost_total"].replace(
        {"\$": "", ",": ""}, regex=True
    ).astype(float)

features = ["served_total", "planned_total", "discarded_total", "left_over_total"]
target = "production_cost_total"

X = preprocessed_data[features].values
y = preprocessed_data[target].values

X_train_tab, X_test_tab, y_train_tab, y_test_tab = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train_tab, y_train_tab)
y_pred_lr = lr_model.predict(X_test_tab)
mse_lr = mean_squared_error(y_test_tab, y_pred_lr)
r2_lr  = r2_score(y_test_tab, y_pred_lr)
mae_lr = mean_absolute_error(y_test_tab, y_pred_lr)
print(f"Linear Regression → MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}, MAE: {mae_lr:.4f}")
lr_filepath = f"results/linear_regression.pkl"
with open(lr_filepath, "wb") as f:
    pickle.dump(lr_model, f)
print(f"Saved LR model to {lr_filepath}")

plt.figure(figsize=(6,6))
plt.scatter(y_test_tab, y_pred_lr, alpha=0.6)
min_val = min(min(y_test_tab), min(y_pred_lr))
max_val = max(max(y_test_tab), max(y_pred_lr))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction line")
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost (LR)")
plt.title("Linear Regression: Predicted vs Actual")
plt.grid(True)
plt.legend()
plot_lr_path = f"plots/linear_regression.png"
plt.savefig(plot_lr_path, bbox_inches='tight', dpi=300)
plt.close()

#%%
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1)
xgb_model.fit(X_train_tab, y_train_tab)
y_pred_xgb = xgb_model.predict(X_test_tab)
mse_xgb = mean_squared_error(y_test_tab, y_pred_xgb)
r2_xgb  = r2_score(y_test_tab, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test_tab, y_pred_xgb)
print(f"XGBoost → MSE: {mse_xgb:.4f}, R²: {r2_xgb:.4f}, MAE: {mae_xgb:.4f}")
xgb_filepath = f"results/xgboost_model.json"
xgb_model.save_model(xgb_filepath)

plt.figure(figsize=(6,6))
plt.scatter(y_test_tab, y_pred_xgb, alpha=0.6)
min_val = min(min(y_test_tab), min(y_pred_xgb))
max_val = max(max(y_test_tab), max(y_pred_xgb))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction line")
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost (XGBoost)")
plt.title("XGBoost: Predicted vs Actual")
plt.grid(True)
plt.legend()
plot_xgb_path = f"plots/xgboost_model.png"
plt.savefig(plot_xgb_path, bbox_inches='tight', dpi=300)
plt.close()

X_tensor_train = torch.tensor(X_train_tab, dtype=torch.float32)
y_tensor_train = torch.tensor(y_train_tab, dtype=torch.float32).unsqueeze(1)
X_tensor_test  = torch.tensor(X_test_tab,  dtype=torch.float32)
y_tensor_test  = torch.tensor(y_test_tab,  dtype=torch.float32).unsqueeze(1)

fnn_model = FeedForwardNN(input_dim=len(features), hidden_dim=256, output_dim=1)
criterion_fnn = nn.MSELoss()
optimizer_fnn = torch.optim.Adam(fnn_model.parameters(), lr=0.0005)

epochs_fnn = 60
for epoch in range(epochs_fnn):
    fnn_model.train()
    optimizer_fnn.zero_grad()
    preds_fnn = fnn_model(X_tensor_train)
    loss_fnn  = criterion_fnn(preds_fnn, y_tensor_train)
    loss_fnn.backward()
    optimizer_fnn.step()
    if epoch % 5 == 0:
        print(f"FNN Epoch {epoch}/{epochs_fnn} — Loss: {loss_fnn.item():.6f}")

fnn_model.eval()
with torch.no_grad():
    y_pred_fnn = fnn_model(X_tensor_test).numpy().flatten()
mse_fnn = mean_squared_error(y_test_tab, y_pred_fnn)
r2_fnn  = r2_score(y_test_tab, y_pred_fnn)
mae_fnn = mean_absolute_error(y_test_tab, y_pred_fnn)
print(f"FNN → MSE: {mse_fnn:.4f}, R²: {r2_fnn:.4f}, MAE: {mae_fnn:.4f}")
fnn_filepath = f"results/fnn_model.pth"
torch.save(fnn_model.state_dict(), fnn_filepath)

plt.figure(figsize=(6,6))
plt.scatter(y_test_tab, y_pred_fnn, alpha=0.6)
min_val = min(min(y_test_tab), min(y_pred_fnn))
max_val = max(max(y_test_tab), max(y_pred_fnn))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction line")
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost (FNN)")
plt.title("FNN: Predicted vs Actual")
plt.grid(True)
plt.legend()
plot_fnn_path = f"plots/fnn_model.png"
plt.savefig(plot_fnn_path, bbox_inches='tight', dpi=300)
plt.close()

class SchoolMealDataset(Dataset):
    def __init__(self, df, features, target):
        self.data = []
        groups = df.groupby(["school_name", "meal_type"])
        for _, group in groups:
            group = group.sort_values("date")
            X = torch.tensor(group[features].values, dtype=torch.float32)
            y = torch.tensor(group[target].values, dtype=torch.float32)
            self.data.append((X, y))
        self.max_len = max(len(x[0]) for x in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X, y = self.data[idx]
        return X, y, len(X)

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)  
    Xs, ys, lengths = zip(*batch)
    X_padded = pad_sequence(Xs, batch_first=True, padding_value=0.0)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=0.0)
    return X_padded, y_padded, torch.tensor(lengths)

dataset = SchoolMealDataset(preprocessed_data, features, target)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

MODEL_TYPE = "GRU" 
INPUT_DIM = len(features)
HIDDEN_DIM = 256
NUM_LAYERS = 4
OUTPUT_DIM = 1
DROPOUT = 0.25
LEARNING_RATE = 0.001
EPOCHS = 60

model = ForecastingModel(MODEL_TYPE, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, DROPOUT)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []

for epoch in tqdm(range(EPOCHS), desc=f"Training {MODEL_TYPE}"):
    model.train()
    running_loss = 0
    for X_batch, y_batch, lengths in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch, lengths)
        y_true = torch.stack([y_batch[i, l - 1] for i, l in enumerate(lengths)]).unsqueeze(1)
        loss = criterion(preds, y_true)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_losses[-1]:.6f}")

model.eval()
y_true_all, y_pred_all = [], []

with torch.no_grad():
    for X_batch, y_batch, lengths in test_loader:
        preds = model(X_batch, lengths)
        y_true = torch.stack([y_batch[i, l - 1] for i, l in enumerate(lengths)]).unsqueeze(1)
        y_true_all.extend(y_true.numpy())
        y_pred_all.extend(preds.numpy())

mse = mean_squared_error(y_true_all, y_pred_all)
r2 = r2_score(y_true_all, y_pred_all)
mae = mean_absolute_error(y_true_all, y_pred_all)

print(f"\n Final Results for {MODEL_TYPE}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")

torch.save(model.state_dict(), f"results/{MODEL_TYPE}.pth")

plt.figure(figsize=(6,6))
plt.scatter(y_true_all, y_pred_all, alpha=0.6, label="Predicted vs Actual")
min_val = min(min(y_true_all), min(y_pred_all))
max_val = max(max(y_true_all), max(y_pred_all))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect prediction line")
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost")
plt.title(f"{MODEL_TYPE} Predicted vs Actual")
plt.grid(True)
plt.legend()
plt.savefig(f"plots/{MODEL_TYPE}.png", bbox_inches='tight', dpi=300)
plt.show()