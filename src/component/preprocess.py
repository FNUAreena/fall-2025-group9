# %%

"""This script mirrors your Colab logic but uses paths that match your repo:
    src/component/Data/data_breakfast_with_coordinates .csv
    src/component/Data/data_lunch_with_coordinates.csv
It writes:
    src/component/Data/combined_breakfast_lunch.csv
"""

import pandas as pd
from pathlib import Path

# --- Paths ---
breakfast_csv = Path("src/component/Data/data_breakfast_with_coordinates .csv")
lunch_csv     = Path("src/component/Data/data_lunch_with_coordinates.csv")
out_path      = Path("src/component/Data/combined_breakfast_lunch.csv")

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, snake_case, alnum + underscore."""
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"\s+", "_", regex=True)
          .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
          .str.lower()
    )
    return df

def ensure_session(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Ensure a 'session' column and coerce to {'Breakfast','Lunch'}."""
    df = df.copy()
    if "session" not in df.columns:
        df["session"] = label
    else:
        df["session"] = (
            df["session"].astype(str).str.title().str.strip()
              .replace({"Bfast": "Breakfast", "Brkfst": "Breakfast", "Lunch ": "Lunch"})
        )
        bad = ~df["session"].isin(["Breakfast","Lunch"])
        df.loc[bad, "session"] = label
    return df

# --- Load ---
b = pd.read_csv(breakfast_csv, low_memory=False)
l = pd.read_csv(lunch_csv, low_memory=False)

# --- Normalize headers ---
b = normalize_cols(b)
l = normalize_cols(l)

# --- Ensure session labels ---
b = ensure_session(b, "Breakfast")
l = ensure_session(l, "Lunch")

# --- Combine ---
combined = pd.concat([b, l], ignore_index=True, sort=False)

# --- Type casts (only if present) ---
if "date" in combined.columns:
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
if "served_total" in combined.columns:
    combined["served_total"] = pd.to_numeric(combined["served_total"], errors="coerce")

# --- Drop rows with missing date or non-positive served_total (when columns exist) ---
drop_idx = pd.Series(False, index=combined.index)
if "date" in combined.columns:
    drop_idx |= combined["date"].isna()
if "served_total" in combined.columns:
    drop_idx |= combined["served_total"].fillna(0).le(0)

if "date" in combined.columns and "served_total" in combined.columns:
    combined = combined[~(combined["date"].isna() & combined["served_total"].fillna(0).le(0))]

# --- Stats ---
rows_total = len(combined)
cols_total = combined.shape[1]
by_sess = combined["session"].value_counts(dropna=False).to_dict() if "session" in combined.columns else {}

print(f"Rows: {rows_total} | Cols: {cols_total}")
print("By session:", by_sess)
out_path.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(out_path, index=False)


rows_total = len(combined)
cols_total = combined.shape[1]
by_sess = combined["session"].value_counts(dropna=False).to_dict() if "session" in combined.columns else {}

print(f"Saved: {out_path}")
print(f"Rows: {rows_total} | Cols: {cols_total}")
print("By session:", by_sess)



# %%
