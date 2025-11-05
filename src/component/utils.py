import re, pandas as pd
from sympy import series
import os, random, numpy as np, torch
from torch.utils.data import Dataset

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TimeSeriesDataset(Dataset):
    """Professor-style dataset: last WINDOW values -> next value."""
    def __init__(self, data_2d, window: int):
        self.data = np.asarray(data_2d, dtype="float32")
        self.window = int(window)

    def __len__(self):
        return max(0, len(self.data) - self.window)

    def __getitem__(self, idx: int):
        X = self.data[idx : idx + self.window, 0]  # (window,)
        y = self.data[idx + self.window, 0]        # scalar
        return torch.tensor(X).float(), torch.tensor(y).float()

def safe_time_split(scaled: np.ndarray, asplit: float, window: int) -> int:
    """
    Choose a split so BOTH train and test have at least (window+1) points,
    allowing at least one (window->next) sample on each side.
    """
    n = len(scaled)
    low  = window + 1
    high = n - (window + 1)
    if high <= low:
        raise ValueError(
            f"Series too short ({n}) for WINDOW={window}. Need > {2*window+2} total points."
        )
    split = int(n * asplit)
    return max(low, min(split, high))


def _coerce_currency_column(col: pd.Series) -> pd.Series:
    """
    Convert messy currency strings to float.
    If a cell has multiple numbers like '$13.32$9.35', extract all and sum them.
    Handles commas and negatives in parentheses like ($123.45).
    """
    def parse_cell(x):
        if pd.isna(x): return np.nan
        s = str(x)
        is_neg = "(" in s and ")" in s
        nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
        if not nums: return np.nan
        val = sum(float(n) for n in nums)
        return -val if is_neg and val > 0 else val
    return col.apply(parse_cell).astype(float)

def load_and_aggregate_district(CSV_PATH, DATE_COL, TARGET_COL, dayfirst=True):
    import re, numpy as np, pandas as pd

    df = pd.read_csv(CSV_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()

    def _coerce_currency_column(col: pd.Series) -> pd.Series:
        def parse_cell(x):
            if pd.isna(x): return np.nan
            s = str(x); is_neg = "(" in s and ")" in s
            nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
            if not nums: return np.nan
            val = sum(float(n) for n in nums)
            return -val if is_neg and val > 0 else val
        return col.apply(parse_cell).astype(float)

    df[TARGET_COL] = _coerce_currency_column(df[TARGET_COL])

 
    series = (
        df.groupby(['school_name', 'meal_type', DATE_COL], as_index=True)[TARGET_COL]
          .sum()
          .sort_index()
    )

 
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series[series.index.notna()]


    dates  = series.index.to_pydatetime()
    values = series.astype("float32").to_numpy().reshape(-1, 1)
    return dates, values
