# %%
import pandas as pd

def clean_data(input_file, output_file):
    # Load
    df = pd.read_csv(input_file)

    # 1. Convert Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # 2. Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 3. Handle missing values
    # Numeric cols: fill NaN with 0 (you can change to df[num_cols].fillna(df[num_cols].mean()) if averaging makes more sense)
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Object cols: replace NaN with "Unknown"
    obj_cols = df.select_dtypes(include=["object"]).columns
    df[obj_cols] = df[obj_cols].fillna("Unknown")

    # 4. Drop duplicates
    df = df.drop_duplicates()

    # 5. Sort by date (if exists)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

        # 6. Add time features
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week
        df["day"] = df["date"].dt.day

    # Save cleaned CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned file saved to {output_file}")

# Run cleaning on both files
clean_data("//Users/chayachandana/Downloads/fact_meal_day.csv", "/Users/chayachandana/Downloads/fact_meal_day_clean.csv")
clean_data("/Users/chayachandana/Downloads/fact_production.csv", "/Users/chayachandana/Downloads/fact_production_clean.csv")

# %%
