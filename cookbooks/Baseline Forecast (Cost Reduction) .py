#%% 
import pandas as pd
import numpy as np
import math, time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

#%%
IN_FILE = "/Users/chayachandana/Downloads/combined_breakfast_lunch.csv"
OUT_FILE = "/Users/chayachandana/Downloads/cost_reduction_with_stockouts_fixed.csv"
MAX_SAMPLE = 40000            # sample for training speed 
RANDOM_STATE = 42
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 12
SAFETY_BUFFER = 1.05          # 1.05 => 5% safety margin

start = time.time()
df = pd.read_csv(IN_FILE, low_memory=False)
print("Loaded rows:", len(df))

print("\nColumns available (first 40):")
print(df.columns.tolist()[:40])

#%%
if 'production_cost_total' in df.columns:
    df['production_cost_total'] = (
        df['production_cost_total'].astype(str)
        .str.replace(r'[\$,]', '', regex=True)
        .str.strip()
    )
# coerce to numeric (invalid -> NaN)
df['production_cost_total'] = pd.to_numeric(df['production_cost_total'], errors='coerce')

# planned_total to numeric
df['planned_total'] = pd.to_numeric(df['planned_total'], errors='coerce')

# served_total to numeric (if not already)
df['served_total'] = pd.to_numeric(df['served_total'], errors='coerce')

#%%
print("\nDiagnostics before computing unit_cost:")
print("production_cost_total: non-null / total =", df['production_cost_total'].notna().sum(), "/", len(df))
print("planned_total: non-null / total =", df['planned_total'].notna().sum(), "/", len(df))
print("served_total: non-null / total =", df['served_total'].notna().sum(), "/", len(df))
print("Some sample values (production_cost_total, planned_total, served_total):")
print(df[['production_cost_total','planned_total','served_total']].head(10).to_string(index=False))

# Avoid division by zero: compute unit_cost where planned_total>0; otherwise NaN
df['unit_cost'] = np.where(
    (df['planned_total'] > 0) & (df['production_cost_total'].notna()),
    df['production_cost_total'] / df['planned_total'],
    np.nan
)

#%%
# Fill missing unit_cost with median unit_cost (robust fallback)
median_unit_cost = df['unit_cost'].median(skipna=True)
if pd.isna(median_unit_cost):
    # last resort fallback to 1.0
    median_unit_cost = 1.0
df['unit_cost'] = df['unit_cost'].fillna(median_unit_cost)

print(f"\nMedian unit cost used for fallback: {median_unit_cost:.4f}")

# ---------- DEFINE ORIGINAL COST ----------
# If production_cost_total exists and is >0, prefer that as original_cost (more reliable).
# Otherwise estimate original_cost = planned_total * unit_cost
df['original_cost'] = np.where(
    df['production_cost_total'].notna() & (df['production_cost_total'] > 0),
    df['production_cost_total'],
    df['planned_total'].fillna(0) * df['unit_cost']
)

#%%
print("\nBaseline sums (check):")
print("Sum production_cost_total (non-NaN contributions):", df['production_cost_total'].sum(skipna=True))
print("Sum original_cost (computed):", df['original_cost'].sum())

# ---------- FEATURE PREP FOR FORECAST ----------
# add simple features: planned_total and offered_total
features = ['planned_total']
if 'offered_total' in df.columns:
    df['offered_total'] = pd.to_numeric(df['offered_total'], errors='coerce').fillna(0)
    features.append('offered_total')

df['planned_total'] = df['planned_total'].fillna(0)

# Prepare X and y
X_full = df[features].fillna(0).to_numpy()
y_full = df['served_total'].fillna(0).to_numpy()

# ---------- TRAIN RANDOM FOREST (on a sample for speed) ----------
if MAX_SAMPLE and len(df) > MAX_SAMPLE:
    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(df), MAX_SAMPLE, replace=False)
    X_train_sample = X_full[sample_idx]
    y_train_sample = y_full[sample_idx]
    df_sample = df.reset_index(drop=True).loc[sample_idx].reset_index(drop=True)
else:
    X_train_sample = X_full
    y_train_sample = y_full
    df_sample = df.copy()

print(f"\nTraining RandomForest on {len(X_train_sample)} rows (this may take a moment)...")
rf = RandomForestRegressor(n_estimators=RF_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_sample, y_train_sample)

# quick validation on a holdout from the training sample
X_tr, X_te, y_tr, y_te = train_test_split(X_train_sample, y_train_sample, test_size=0.2, random_state=RANDOM_STATE)
yhat_te = rf.predict(X_te)
mae = mean_absolute_error(y_te, yhat_te)
rmse = math.sqrt(mean_squared_error(y_te, yhat_te))
print(f"Model diagnostics on sample holdout: MAE={mae:.4f}, RMSE={rmse:.4f}")

#%%
# ---------- PREDICT ON FULL DATA ----------
print("Predicting on full dataset...")
df['forecast_served'] = rf.predict(X_full)

# If any negative predictions, clip to zero
df['forecast_served'] = df['forecast_served'].clip(lower=0)

# ---------- COMPUTE OPTIMIZED PLAN & COST ----------
df['optimized_planned'] = np.ceil(df['forecast_served'] * SAFETY_BUFFER).astype(int)
df['optimized_cost'] = df['optimized_planned'] * df['unit_cost']

# ---------- STOCKOUT METRICS ----------
df['stockout_flag'] = (df['optimized_planned'] < df['served_total']).astype(int)
df['stockout_qty'] = (df['served_total'] - df['optimized_planned']).clip(lower=0).fillna(0).astype(int)

total_original_cost = df['original_cost'].sum()
total_optimized_cost = df['optimized_cost'].sum()
total_savings = total_original_cost - total_optimized_cost
stockout_cases = int(df['stockout_flag'].sum())
stockout_rate = stockout_cases / len(df) * 100
stockout_qty_total = int(df['stockout_qty'].sum())

print("\n===== RESULTS =====")
print(f"Rows processed: {len(df)}")
print(f"Original total cost:   {total_original_cost:,.2f}")
print(f"Optimized total cost:  {total_optimized_cost:,.2f}")
print(f"Estimated savings:     {total_savings:,.2f}")
print(f"Stockout cases:        {stockout_cases}  ({stockout_rate:.2f}%)")
print(f"Total stockout qty:    {stockout_qty_total}")

#%%
df.to_csv(OUT_FILE, index=False)
print("\nSaved results to:", OUT_FILE)
print("Elapsed time: {:.1f}s".format(time.time() - start))
# %%
