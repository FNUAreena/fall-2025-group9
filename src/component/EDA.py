#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import svd

df = pd.read_csv("/Users/chayachandana/Downloads/Production_data.csv")

#%%

cols = [
    "production_cost_total", "level", "fcps_region", "session",
    "served_total", "planned_total", "discarded_total", "left_over_total"
]
df = df[cols].copy()

df.dropna(inplace=True)

if df['discarded_total'].dtype == 'object':
    df['discarded_total'] = df['discarded_total'].str.replace('%', '', regex=False)
    df['discarded_total'] = pd.to_numeric(df['discarded_total'], errors='coerce') / 100.0

label_encoders = {}
for col in ["level", "fcps_region", "session"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

def clean_currency(x):
    if isinstance(x, str):
        return float(x.replace("$", "").replace(",", "").strip())
    return x
df["production_cost_total"] = df["production_cost_total"].apply(clean_currency)

corr = df.corr(numeric_only=True)

#%%
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#%%

X = df[["level", "fcps_region", "session",
        "served_total", "planned_total", "discarded_total", "left_over_total"]].astype(float)

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\n=== Variance Inflation Factor (VIF) Scores ===")
print(vif_data)

# %%

X_svd = df[["level", "fcps_region", "session", 
            "served_total", "planned_total", 
            "discarded_total", "left_over_total"]]

X_svd = (X_svd - X_svd.mean()) / X_svd.std(ddof=0)

U, S, VT = svd(X_svd, full_matrices=False)

print("\n=== SVD Collinearity Analysis ===")
print("Singular Values:\n", S)

threshold = 1e-5
if any(S < threshold):
    print("Collinearity detected — one or more singular values are near zero.")
else:
    print("No strong collinearity detected (all singular values are large).")

plt.figure(figsize=(6, 4))
plt.plot(range(1, len(S)+1), S, 'bo-', linewidth=2)
plt.title("Singular Values (SVD)")
plt.xlabel("Component Number")
plt.ylabel("Singular Value Magnitude")
plt.tight_layout()
plt.show()
# %%
