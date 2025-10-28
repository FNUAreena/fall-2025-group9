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
from sklearn.metrics import mean_squared_error, r2_score
from utils import preprocessing
from model import ForecastingModel
import warnings

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

MODEL_TYPE = "LSTM"
INPUT_DIM = len(features)
HIDDEN_DIM = 128
NUM_LAYERS = 3
OUTPUT_DIM = 1
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 40

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

print(f"\n Final Results for {MODEL_TYPE}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(y_true_all, y_pred_all, alpha=0.6)
plt.xlabel("Actual Production Cost")
plt.ylabel("Predicted Production Cost")
plt.title(f"{MODEL_TYPE} Predicted vs Actual")
plt.grid(True)
plt.show()