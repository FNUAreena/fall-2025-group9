#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/chayachandana/Desktop/combined_breakfast_lunch.csv")
df['date'] = pd.to_datetime(df['date'])
df['weekday'] = df['date'].dt.day_name()

for col in ['left_over_total', 'served_total', 'offered_total']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

baseline = (
    df.groupby(['school_id', 'session', 'name', 'weekday'])['left_over_total']
      .median()
      .reset_index()
      .rename(columns={'left_over_total': 'median_leftover'})
)

df = df.merge(baseline, on=['school_id', 'session', 'name', 'weekday'], how='left')

df['baseline_error'] = df['left_over_total'] - df['median_leftover']
df['abs_baseline_error'] = df['baseline_error'].abs()

mae = df['abs_baseline_error'].mean()
print(f"Baseline MAE (mean absolute error): {mae:.2f}")

print(df[['date', 'school_id', 'session', 'name', 'left_over_total', 'median_leftover', 'baseline_error']].head(10))

# %%
df_sorted = df.sort_values('date')

plt.figure(figsize=(12,6))
plt.plot(df_sorted['date'], df_sorted['left_over_total'], label='Actual Leftover', marker='o')
plt.plot(df_sorted['date'], df_sorted['median_leftover'], label='Baseline Prediction', marker='x')
plt.xlabel('Date')
plt.ylabel('Leftover Portions')
plt.title('Actual vs Baseline Predicted Leftover Food Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(7,7))
plt.scatter(df['median_leftover'], df['left_over_total'], alpha=0.5)
plt.plot([df['median_leftover'].min(), df['median_leftover'].max()],
         [df['median_leftover'].min(), df['median_leftover'].max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Baseline Predicted Leftover')
plt.ylabel('Actual Leftover')
plt.title('Actual vs Baseline Predicted Leftover')
plt.legend()
plt.tight_layout()
plt.show()

# %%
